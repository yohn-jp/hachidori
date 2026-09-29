package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// desktopApp is the no-argument Windows entry point: home discovery, then the
// first-run/recovery flow or normal startup, in one native window. It is a
// composition of existing authorities: home.Discover/Remember (bootstrap
// locator), app.Controller (setup and the one resident worker), the same
// server/dashboard handlers as `hachidori dashboard`, and the desktop shell.
type desktopApp struct {
	Platform desktop.Platform
	Picker   desktop.FolderPicker
	Discover func() (home.Discovery, error)
	Remember func(root string) (home.Home, error)
	Env      firstrun.Env
	APIAddr  string // loopback address of the inference API
	DashAddr string // loopback address of the window's origin
	Setup    app.SetupFunc
	Stderr   io.Writer
}

func newDesktopApp(p desktop.Platform, pk desktop.FolderPicker) *desktopApp {
	return &desktopApp{
		Platform: p, Picker: pk,
		Discover: func() (home.Discovery, error) { return home.Discover("") },
		Remember: home.Remember,
		Env:      firstrun.Env{FreeSpace: firstrun.DefaultFreeSpace},
		APIAddr:  server.DefaultListen, DashAddr: dashboard.DefaultListen,
		Stderr: os.Stderr,
	}
}

// runDesktopApp is the production no-argument entry point.
func runDesktopApp(p desktop.Platform, pk desktop.FolderPicker) error {
	return newDesktopApp(p, pk).run()
}

// swap holds the handlers bound to the current runtime; the listeners are
// created once so the window's origin never changes across setup.
type swap struct {
	mu   sync.Mutex
	api  http.Handler
	dash http.Handler
}

func (s *swap) set(api, dash http.Handler) {
	s.mu.Lock()
	s.api, s.dash = api, dash
	s.mu.Unlock()
}

func (s *swap) dashboard() http.Handler {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dash
}

func (s *swap) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	h := s.api
	s.mu.Unlock()
	if h == nil {
		http.Error(w, "runtime is not installed or started yet", http.StatusServiceUnavailable)
		return
	}
	h.ServeHTTP(w, r)
}

func (a *desktopApp) run() error {
	if err := server.CheckLoopback(a.APIAddr); err != nil {
		return err
	}
	if err := server.CheckLoopback(a.DashAddr); err != nil {
		return err
	}
	version, release, err := desktop.Preflight(a.Platform)
	if err != nil {
		return err
	}
	defer release()
	fmt.Fprintf(a.Stderr, "hachidori: WebView2 Runtime %s\n", version)

	d, derr := a.Discover()
	plan := firstrun.Decide(d, derr, a.Env)
	fmt.Fprintf(a.Stderr, "hachidori: desktop start: %s\n", plan.Mode)

	// Workers live until the whole application is done, independent of the
	// window: the window's end triggers an orderly Close below.
	rctx, rcancel := context.WithCancel(context.Background())
	defer rcancel()
	sw := &swap{}
	tun := tunnel.NewManager("ssh")
	defer tun.Disconnect()
	var logMu sync.Mutex
	var logs []*os.File
	defer func() {
		logMu.Lock()
		defer logMu.Unlock()
		for _, f := range logs {
			f.Close()
		}
	}()

	open := func(root string) (app.Runtime, error) {
		if err := os.MkdirAll(filepath.Join(root, "logs"), 0o755); err != nil {
			return nil, err
		}
		lf, err := os.OpenFile(filepath.Join(root, "logs", "worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return nil, err
		}
		rt, err := app.WorkerRuntime(rctx, lf, worker.DefaultPolicy, func(b *app.WorkerBinding) {
			h := home.Home{Root: root}
			dash := dashboard.New(dashboard.Config{
				APIAddr:   a.APIAddr,
				Status:    b.Status,
				Lifecycle: b.Lifecycle,
				Doctor:    func(out io.Writer) bool { return doctor.Run(root, out) },
				Tunnel:    tun,
				PrefsPath: h.Path("state", "dashboard.json"),
			})
			sw.set(server.HandlerSince(b.Supervisor, b.Info, b.Started), dash)
		})(root)
		if err != nil {
			lf.Close()
			return nil, err
		}
		logMu.Lock()
		logs = append(logs, lf)
		logMu.Unlock()
		return rt, nil
	}

	selected := ""
	if plan.Mode == firstrun.ModeLaunch || plan.Mode == firstrun.ModeResume {
		selected = plan.Home
	}
	ctl := app.New(app.Config{Home: selected, Open: open, Setup: a.Setup})
	flow := firstrun.New(firstrun.Config{Ctl: ctl, Plan: plan, Picker: a.Picker, Env: a.Env, Remember: a.Remember})
	if plan.Mode == firstrun.ModeLaunch {
		// A configured, installed home: normal startup. A failure is kept by
		// the controller and shown with Retry, never as an opaque exit.
		if err := ctl.Start(); err != nil {
			fmt.Fprintln(a.Stderr, "hachidori: start:", err)
		}
	}

	apiLn, err := net.Listen("tcp", a.APIAddr)
	if err != nil {
		return err
	}
	dashLn, err := net.Listen("tcp", a.DashAddr)
	if err != nil {
		apiLn.Close()
		return err
	}
	apiSrv := &http.Server{Handler: sw, ReadHeaderTimeout: 10 * time.Second}
	dashSrv := &http.Server{Handler: firstrun.NewHandler(flow, sw.dashboard), ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 2)
	go func() { errc <- apiSrv.Serve(apiLn) }()
	go func() { errc <- dashSrv.Serve(dashLn) }()

	dashURL, err := desktop.DashboardURL(dashLn.Addr().String())
	if err == nil {
		var pol desktop.Policy
		if pol, err = desktop.NewPolicy(dashURL); err == nil {
			dataDir, cleanup := a.webviewDataDir(plan)
			defer cleanup()
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			go func() {
				select {
				case e := <-errc:
					if e != nil && !errors.Is(e, http.ErrServerClosed) {
						fmt.Fprintln(a.Stderr, "hachidori: server:", e)
					}
					stop()
				case <-ctx.Done():
				}
			}()
			err = a.Platform.Open(ctx, desktop.Window{Title: "Hachidori", URL: dashURL, DataDir: dataDir, Policy: pol})
			stop()
		}
	}

	shut, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if cerr := ctl.Close(shut); cerr != nil {
		fmt.Fprintln(a.Stderr, "hachidori:", cerr)
	}
	rcancel()
	_ = dashSrv.Shutdown(shut)
	_ = apiSrv.Shutdown(shut)
	return err
}

// webviewDataDir is the WebView2 profile folder. With a selected home it is
// the usual HACHIDORI_HOME/cache/webview2. Before any home exists (first run
// or recovery) there is no home to put it in and Hachidori must not create
// state elsewhere, so the profile is a throwaway temporary folder removed on
// exit.
func (a *desktopApp) webviewDataDir(plan firstrun.Plan) (string, func()) {
	if (plan.Mode == firstrun.ModeLaunch || plan.Mode == firstrun.ModeResume) && plan.Home != "" {
		return home.Home{Root: plan.Home}.Path("cache", "webview2"), func() {}
	}
	dir, err := os.MkdirTemp("", "hachidori-webview2-")
	if err != nil {
		return filepath.Join(os.TempDir(), "hachidori-webview2"), func() {}
	}
	return dir, func() { _ = os.RemoveAll(dir) }
}
