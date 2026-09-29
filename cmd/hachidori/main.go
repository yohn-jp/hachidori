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
  setup      materialize the pinned runtime and a catalog model (--model) under
             HACHIDORI_HOME and activate them
  serve      run the HTTP runtime with a resident inference worker
  dashboard  serve, plus a host-local Web dashboard (status, start/stop/restart,
             doctor, SSH reverse-tunnel launcher) on 127.0.0.1:7844
  doctor     verify the installation, including a real smoke inference

client (caller side, uses HACHIDORI_ENDPOINT):
  status     print /v1/status
  decide     send a v1 decide request (JSON file or - for stdin)
  eval       evaluate a local JSONL dataset through the endpoint
  benchmark  eval with warmup and repeated passes for latency

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

// setupFlags are the flags of `hachidori setup`. A model is selected only by
// its catalog ID; there is deliberately no repository or revision input.
type setupFlags struct{ home, device, model string }

func newSetupFlags() (*flag.FlagSet, *setupFlags) {
	var f setupFlags
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	fs.StringVar(&f.home, "home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	fs.StringVar(&f.device, "device", "cuda", "inference device: cuda or cpu")
	fs.StringVar(&f.model, "model", setup.DefaultModel, "catalog model ID to materialize and activate ("+strings.Join(modelIDs(), ", ")+")")
	return fs, &f
}

func modelIDs() []string {
	var ids []string
	for _, m := range setup.Models {
		ids = append(ids, m.ID)
	}
	return ids
}

func cmdSetup(args []string) error {
	fs, f := newSetupFlags()
	fs.Parse(args)
	h, err := home.Resolve(f.home)
	if err != nil {
		return err
	}
	return setup.Run(h, f.device, f.model, os.Stderr)
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
	fmt.Fprintf(os.Stderr, "hachidori: serving %s (runtime %s, model %s (%s), device %s); worker log %s\n",
		*listen, rt.Runtime, rt.ModelID, rt.Model, rt.Device, logf.Name())
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
	r := eval.Run(c, cases, opt)
	r.Endpoint, r.Dataset, r.DatasetSHA256 = c.Endpoint, fs.Arg(0), sum
	eval.Summary(os.Stdout, r)
	if *out != "" {
		b, _ := json.MarshalIndent(r, "", "  ")
		if err := os.WriteFile(*out, append(b, '\n'), 0o644); err != nil {
			return err
		}
		fmt.Printf("report written to %s\n", *out)
	}
	if len(r.Errors) > 0 {
		return fmt.Errorf("%d request errors", len(r.Errors))
	}
	return nil
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
