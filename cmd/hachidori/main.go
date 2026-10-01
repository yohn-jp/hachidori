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
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
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
  desktop    (Windows) the desktop application in a native WebView2 window with
             a tray icon: first-run setup or normal startup; closing the window
             hides it to the tray, Quit stops the runtime (--home /
             HACHIDORI_HOME; --background is used by start at sign-in)
  (no command) on Windows, hachidori.exe with no arguments is the same desktop
             application
  doctor     verify the installation, including a real smoke inference

client (caller side, uses HACHIDORI_ENDPOINT):
  status     print /v1/status
  decide     send a v1 decide request (JSON file or - for stdin)
  eval       evaluate a local JSONL dataset through the endpoint
  benchmark  eval with warmup and repeated passes for latency
  question   validate local Question Definitions and print their identity
             and compiled v1 question (no endpoint)
  replay     reconstruct and re-send decisions from a Decision Evidence report

Run 'hachidori <command> -h' for flags.
`

// commands is the command dispatch table; every command in usage must be here.
func commands() map[string]func([]string) error {
	return map[string]func([]string) error{
		"setup": cmdSetup, "serve": cmdServe, "doctor": cmdDoctor, "status": cmdStatus,
		"dashboard": func(a []string) error { return runHost("dashboard", a) },
		"desktop":   func(a []string) error { return cmdDesktop(desktop.Native(), a) },
		"decide":    cmdDecide, "eval": func(a []string) error { return cmdEval("eval", a) },
		"benchmark": func(a []string) error { return cmdEval("benchmark", a) },
		"question":  cmdQuestion,
		"replay":    cmdReplay,
	}
}

// entry is the startup measurement epoch: taken first thing in main, it is
// the one origin every desktop startup mark is measured from.
var entry desktop.Epoch

func main() {
	entry = desktop.Epoch(time.Now())
	os.Exit(run(os.Args[1:], noArgLaunch))
}

// run dispatches the command line. noArg, when non-nil, is the no-argument
// Windows desktop entry point. Explicit CLI commands remain unchanged.
func run(args []string, noArg func() error) int {
	if len(args) == 0 {
		if noArg != nil {
			if err := noArg(); err != nil {
				fmt.Fprintln(os.Stderr, "hachidori:", err)
				return 1
			}
			return 0
		}
		fmt.Fprint(os.Stderr, usage)
		return 2
	}
	cmd, ok := commands()[args[0]]
	if !ok {
		fmt.Fprint(os.Stderr, usage)
		return 2
	}
	if err := cmd(args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "hachidori:", err)
		return 1
	}
	return 0
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
// the dashboard command, the host-local dashboard beside it. The Windows
// desktop is a separate composition (desktopApp).
func runHost(name string, args []string) error {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	listen := fs.String("listen", server.DefaultListen, "loopback address to bind")
	var residents pathList
	fs.Var(&residents, "resident", "catalog model ID kept resident beside the active model, each in its own worker process ("+strings.Join(modelIDs(), ", ")+"; repeatable). The active model stays the default route")
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
	// Resolve the active runtime before opening the log: a home that was never
	// set up has no logs directory, and the operator needs to be told to run
	// setup, not shown a missing-path error from the log file.
	cfg, info, err := server.WorkerConfig(h, nil)
	if err != nil {
		return err
	}
	logf, err := os.OpenFile(h.Path("logs", "worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer logf.Close()
	cfg.Log = logf
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var (
		dec    server.Decider
		lc     dashboard.Lifecycle
		status func() server.Status
		rt     = info
		now    = time.Now()
	)
	if len(residents) == 0 {
		sup := worker.NewSupervisor(cfg, worker.DefaultPolicy)
		wl := worker.NewLifecycle(ctx, sup)
		wl.Start()
		defer wl.Stop()
		dec, lc = sup, wl
		status = func() server.Status { return server.StatusBody(sup, rt, now) }
		go logTransitions(ctx, "", sup)
	} else {
		set, err := app.OpenResidents(ctx, h, logf, worker.DefaultPolicy, residents)
		if err != nil {
			return err
		}
		set.Start()
		defer set.Stop()
		dec, lc, rt, now, status = set, set, set.Runtime(), set.Started(), set.Status
		for _, st := range set.ResidentStatuses() {
			r, _ := set.Resident(st.Model)
			go logTransitions(ctx, st.Model, r.Supervisor)
		}
	}

	srv := &http.Server{Addr: *listen, Handler: server.HandlerSince(dec, rt, now), ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 2)
	go func() { errc <- srv.ListenAndServe() }()
	fmt.Fprintf(os.Stderr, "hachidori: serving %s (runtime %s, model %s (%s), device %s); worker log %s\n",
		*listen, rt.Runtime, rt.ModelID, rt.Model, rt.Device, logf.Name())
	if len(residents) > 0 {
		fmt.Fprintf(os.Stderr, "hachidori: resident set: default %s, extra %s (one worker process each)\n", rt.ModelID, strings.Join(residents, ", "))
	}

	var dash *http.Server
	if dashAddr != nil {
		d := dashboard.New(dashboard.Config{
			APIAddr:   *listen,
			Status:    status,
			Lifecycle: lc,
			Doctor:    func(out io.Writer) bool { return doctor.Run(h.Root, out) },
			Tunnel:    tunnel.NewManager(*sshExe),
			PrefsPath: h.Path("state", "dashboard.json"),
		})
		// Terminate the managed ssh child on every exit path of this process
		// that runs deferred code; see docs/runtime.md for hard kills.
		defer d.Close()
		dash = &http.Server{Addr: *dashAddr, ReadHeaderTimeout: 10 * time.Second, Handler: d}
		ln, err := net.Listen("tcp", *dashAddr)
		if err != nil {
			stop()
			srv.Close()
			return err
		}
		go func() { errc <- dash.Serve(ln) }()
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

// logTransitions logs the supervisor's state changes; model names the
// resident for a resident set ("" for the single worker).
func logTransitions(ctx context.Context, model string, sup *worker.Supervisor) {
	who := "worker"
	if model != "" {
		who += " " + model
	}
	last := ""
	for ctx.Err() == nil {
		if s := sup.State(); s != last {
			last = s
			msg := who + " " + s
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
	var defPaths pathList
	fs.Var(&defPaths, "questions", "Question Definition file or directory of *.json resolving question_refs (repeatable)")
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
	var defs *question.Set
	if len(defPaths) > 0 {
		var err error
		if defs, err = question.Load(defPaths...); err != nil {
			return err
		}
	}
	cases, sum, err := eval.Load(fs.Arg(0), defs)
	if err != nil {
		return err
	}
	c := client.New(*endpoint)
	if h, err := c.Health(); err != nil || !h.Ready {
		return fmt.Errorf("endpoint %s not ready (state %q): %v", c.Endpoint, h.State, err)
	}
	r, err := eval.RunEvidence(c, cases, opt)
	if err != nil {
		return err
	}
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
	if !r.ServedConsistent {
		return errors.New("served runtime identity changed during the run; evidence is marked served_consistent=false")
	}
	return nil
}

// cmdReplay reconstructs requests from Decision Evidence and the original local
// dataset. Question Definitions are resolved caller-side exactly as for eval.
func cmdReplay(args []string) error {
	fs := flag.NewFlagSet("replay", flag.ExitOnError)
	endpoint := fs.String("endpoint", "", "endpoint (default: $HACHIDORI_ENDPOINT or "+client.DefaultEndpoint+")")
	dataset := fs.String("dataset", "", "dataset JSONL (default: path recorded in report)")
	out := fs.String("out", "", "write replay comparison JSON to this local file")
	printOnly := fs.Bool("print", false, "print reconstructed requests without contacting the endpoint")
	var defPaths, caseIDs, questionIDs pathList
	fs.Var(&defPaths, "questions", "Question Definition file or directory resolving question_refs (repeatable)")
	fs.Var(&caseIDs, "case", "replay only this case id (repeatable)")
	fs.Var(&questionIDs, "question", "replay only this question id (repeatable)")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori replay [flags] <report.json>")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("replay requires one report path")
	}
	r, err := eval.LoadReport(fs.Arg(0))
	if err != nil {
		return err
	}
	path := *dataset
	if path == "" {
		path = r.Dataset
	}
	var defs *question.Set
	if len(defPaths) > 0 {
		defs, err = question.Load(defPaths...)
		if err != nil {
			return err
		}
	}
	cases, sum, err := eval.Load(path, defs)
	if err != nil {
		return fmt.Errorf("replay source %q: %w", path, err)
	}
	items, err := eval.ReplayRequests(r, cases, sum, eval.Selection{Cases: []string(caseIDs), Questions: []string(questionIDs)})
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
		b, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(*out, append(b, '\n'), 0o644); err != nil {
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
			return errors.New("replay request errors (see output)")
		}
	}
	return nil
}

// cmdQuestion validates definitions locally and prints, per definition, its
// identity and the exact api.Question it compiles to.
func cmdQuestion(args []string) error {
	fs := flag.NewFlagSet("question", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori question <definition.json|dir>...")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() == 0 {
		fs.Usage()
		os.Exit(2)
	}
	set, err := question.Load(fs.Args()...)
	if err != nil {
		return err
	}
	if set.Len() == 0 {
		return errors.New("no question definitions found")
	}
	type compiled struct {
		question.Identity
		Question api.Question `json:"question"`
	}
	var out []compiled
	for _, d := range set.Definitions() {
		out = append(out, compiled{Identity: d.Identity(), Question: d.Compile()})
	}
	return printJSON(out)
}

type pathList []string

func (p *pathList) String() string     { return strings.Join(*p, ",") }
func (p *pathList) Set(v string) error { *p = append(*p, v); return nil }

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
