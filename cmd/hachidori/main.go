// Command hachidori is the single user-facing entry point of the Hachidori
// semantic inference runtime.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const usage = `usage: hachidori <command> [flags]

runtime (inference host):
  setup      materialize the pinned runtime and model under HACHIDORI_HOME
  serve      run the HTTP runtime with a resident inference worker
  dashboard  serve, plus a host-local Web dashboard (status, start/stop/restart,
             doctor, SSH reverse-tunnel launcher) on 127.0.0.1:7844
  doctor     verify the installation, including a real smoke inference

client (caller side, uses HACHIDORI_ENDPOINT):
  status     print /v1/status
  decide     send a v1 decide request (JSON file or - for stdin)
  eval       evaluate a local JSONL dataset through the endpoint
  benchmark  eval with warmup and repeated passes for latency
  replay     re-send recorded decisions of an eval/benchmark report

Run 'hachidori <command> -h' for flags.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cmds := map[string]func([]string) error{
		"setup": cmdSetup, "serve": cmdServe, "doctor": cmdDoctor, "status": cmdStatus,
		"dashboard": func(a []string) error { return runHost("dashboard", a) },
		"decide":    cmdDecide, "eval": func(a []string) error { return cmdEval("eval", a) },
		"benchmark": func(a []string) error { return cmdEval("benchmark", a) },
		"replay":    cmdReplay,
	}
	run, ok := cmds[os.Args[1]]
	if !ok {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	if err := run(os.Args[2:]); err != nil {
		fmt.Fprintln(os.Stderr, "hachidori:", err)
		os.Exit(1)
	}
}

func cmdSetup(args []string) error {
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	device := fs.String("device", "cuda", "inference device: cuda or cpu")
	fs.Parse(args)
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	return setup.Run(h, *device, os.Stderr)
}

func cmdServe(args []string) error { return runHost("serve", args) }

// runHost runs the inference runtime (HTTP API + resident worker) and, for
// the dashboard command, the host-local dashboard beside it.
func runHost(name string, args []string) error {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	listen := fs.String("listen", server.DefaultListen, "loopback address to bind")
	var dashAddr, sshExe *string
	if name == "dashboard" {
		dashAddr = fs.String("addr", dashboard.DefaultListen, "loopback address of the dashboard")
		sshExe = fs.String("ssh", "ssh", "host ssh client used by the tunnel launcher")
	}
	fs.Parse(args)
	if err := server.CheckLoopback(*listen); err != nil {
		return err
	}
	if dashAddr != nil {
		if err := server.CheckLoopback(*dashAddr); err != nil {
			return err
		}
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	logf, err := os.OpenFile(h.Path("logs", "worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer logf.Close()
	cfg, rt, err := server.WorkerConfig(h, logf)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	sup := worker.NewSupervisor(cfg, worker.DefaultPolicy)
	lc := worker.NewLifecycle(ctx, sup)
	lc.Start()
	defer lc.Stop()

	started := time.Now()
	srv := &http.Server{Addr: *listen, Handler: server.HandlerSince(sup, rt, started), ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 2)
	go func() { errc <- srv.ListenAndServe() }()
	fmt.Fprintf(os.Stderr, "hachidori: serving %s (runtime %s, model %s, device %s); worker log %s\n",
		*listen, rt.Runtime, rt.Model, rt.Device, logf.Name())
	go logTransitions(ctx, sup)

	var dash *http.Server
	if dashAddr != nil {
		d := dashboard.New(dashboard.Config{
			APIAddr:   *listen,
			Status:    func() server.Status { return server.StatusBody(sup, rt, started) },
			Lifecycle: lc,
			Doctor:    func(out io.Writer) bool { return doctor.Run(h.Root, out) },
			Tunnel:    tunnel.NewManager(*sshExe),
			PrefsPath: h.Path("state", "dashboard.json"),
		})
		// Terminate the managed ssh child on every exit path of this process
		// that runs deferred code; see docs/runtime.md for hard kills.
		defer d.Close()
		dash = &http.Server{Addr: *dashAddr, ReadHeaderTimeout: 10 * time.Second, Handler: d}
		go func() { errc <- dash.ListenAndServe() }()
		fmt.Fprintf(os.Stderr, "hachidori: dashboard http://%s/\n", *dashAddr)
	}

	select {
	case err := <-errc:
		stop()
		srv.Close()
		if dash != nil {
			dash.Close()
		}
		return err
	case <-ctx.Done():
	}
	shut, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if dash != nil {
		_ = dash.Shutdown(shut)
	}
	_ = srv.Shutdown(shut)
	return nil
}

func logTransitions(ctx context.Context, sup *worker.Supervisor) {
	last := ""
	for ctx.Err() == nil {
		if s := sup.State(); s != last {
			last = s
			msg := "worker " + s
			if f := sup.LastFailure(); f != nil && s != worker.StateReady {
				msg += fmt.Sprintf(" (%s: %s)", f.Class, f.Message)
			}
			fmt.Fprintln(os.Stderr, "hachidori:", msg)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func cmdDoctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	fs.Parse(args)
	if !doctor.Run(*homeFlag, os.Stdout) {
		return errors.New("doctor found problems")
	}
	return nil
}

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	endpoint := fs.String("endpoint", "", "endpoint (default: $HACHIDORI_ENDPOINT or "+client.DefaultEndpoint+")")
	fs.Parse(args)
	raw, err := client.New(*endpoint).Status()
	if err != nil {
		return err
	}
	return printJSON(raw)
}

func cmdDecide(args []string) error {
	fs := flag.NewFlagSet("decide", flag.ExitOnError)
	endpoint := fs.String("endpoint", "", "endpoint (default: $HACHIDORI_ENDPOINT or "+client.DefaultEndpoint+")")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori decide [-endpoint URL] <request.json|->")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	var data []byte
	var err error
	if fs.Arg(0) == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(fs.Arg(0))
	}
	if err != nil {
		return err
	}
	var req api.DecideRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return err
	}
	if req.Schema == "" {
		req.Schema = api.SchemaV1
	}
	resp, err := client.New(*endpoint).Decide(req)
	if err != nil {
		return err
	}
	return printJSON(resp)
}

func cmdEval(name string, args []string) error {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	endpoint := fs.String("endpoint", "", "endpoint (default: $HACHIDORI_ENDPOINT or "+client.DefaultEndpoint+")")
	out := fs.String("out", "", "write the full JSON report (with per-observation results) to this local file")
	opt := eval.Options{Passes: 1}
	if name == "benchmark" {
		fs.IntVar(&opt.Warmup, "warmup", 5, "warmup requests excluded from latency")
		fs.IntVar(&opt.Passes, "passes", 3, "passes over the dataset for latency (accuracy uses pass 1)")
	}
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: hachidori %s [flags] <dataset.jsonl>\n", name)
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	cases, sum, err := eval.Load(fs.Arg(0))
	if err != nil {
		return err
	}
	c := client.New(*endpoint)
	if h, err := c.Health(); err != nil || !h.Ready {
		return fmt.Errorf("endpoint %s not ready (state %q): %v", c.Endpoint, h.State, err)
	}
	r, err := eval.Run(c, cases, opt)
	if err != nil {
		return err
	}
	r.Endpoint, r.Dataset, r.DatasetSHA256 = c.Endpoint, fs.Arg(0), sum
	eval.Summary(os.Stdout, r)
	if *out != "" {
		if err := writeJSON(*out, r); err != nil {
			return err
		}
		fmt.Printf("report written to %s\n", *out)
	}
	if len(r.Errors) > 0 {
		return fmt.Errorf("%d request errors", len(r.Errors))
	}
	if !r.ServedConsistent {
		return errors.New("served runtime identity changed during the run; evidence is marked served_consistent=false")
	}
	return nil
}

type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(v string) error { *l = append(*l, v); return nil }

// cmdReplay reconstructs the /v1/decide requests behind recorded evidence
// from the local dataset, refusing a dataset whose digest differs, and
// either prints them or re-sends them and compares the results.
func cmdReplay(args []string) error {
	fs := flag.NewFlagSet("replay", flag.ExitOnError)
	endpoint := fs.String("endpoint", "", "endpoint (default: $HACHIDORI_ENDPOINT or "+client.DefaultEndpoint+")")
	dataset := fs.String("dataset", "", "dataset JSONL (default: the dataset path recorded in the report)")
	out := fs.String("out", "", "write the replay comparison JSON to this local file instead of stdout")
	printOnly := fs.Bool("print", false, "print the reconstructed requests without contacting the endpoint")
	var sel eval.Selection
	fs.Var((*listFlag)(&sel.Cases), "case", "replay only this case id (repeatable)")
	fs.Var((*listFlag)(&sel.Questions), "question", "replay only this question id (repeatable)")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori replay [flags] <report.json>")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	r, err := eval.LoadReport(fs.Arg(0))
	if err != nil {
		return err
	}
	path := *dataset
	if path == "" {
		path = r.Dataset
	}
	cases, sum, err := eval.Load(path)
	if err != nil {
		return fmt.Errorf("replay source %q: %w", path, err)
	}
	items, err := eval.ReplayRequests(r, cases, sum, sel)
	if err != nil {
		return err
	}
	if *printOnly {
		reqs := make([]api.DecideRequest, len(items))
		for i, it := range items {
			reqs[i] = it.Request
		}
		return printJSON(reqs)
	}
	c := client.New(*endpoint)
	rep, err := eval.Replay(c, r, items)
	if err != nil {
		return err
	}
	rep.Endpoint = c.Endpoint
	if *out != "" {
		if err := writeJSON(*out, rep); err != nil {
			return err
		}
		fmt.Printf("replay written to %s\n", *out)
	} else if err := printJSON(rep); err != nil {
		return err
	}
	if !rep.ServedMatches {
		fmt.Fprintln(os.Stderr, "hachidori: note: endpoint serves a different identity than the recorded evidence")
	}
	for _, res := range rep.Results {
		if res.Error != nil {
			return fmt.Errorf("replay request errors (see output)")
		}
	}
	return nil
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func printJSON(v any) error {
	var b []byte
	var err error
	if raw, ok := v.(json.RawMessage); ok {
		var x any
		if err = json.Unmarshal(raw, &x); err != nil {
			return err
		}
		v = x
	}
	if b, err = json.MarshalIndent(v, "", "  "); err != nil {
		return err
	}
	fmt.Println(strings.TrimSpace(string(b)))
	return nil
}
