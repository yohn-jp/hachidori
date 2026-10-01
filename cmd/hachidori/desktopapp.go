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
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// errActivated is returned by run when another desktop instance of this user
// was already running and was asked to show its window instead.
var errActivated = errors.New("activated the running desktop instance")

// desktopApp is the one Windows desktop composition, entered by both the
// no-argument launch and `hachidori desktop`: single-instance guard, home
// discovery, then the first-run/recovery flow or normal startup, and the
// resident tray lifecycle, in one native window. It is a composition of
// existing authorities: home.Discover/Remember (bootstrap locator),
// app.Controller (setup, start, restart and the one resident worker), the same
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
	// Open, when set, replaces the production worker binding (tests only).
	Open   app.OpenFunc
	Stderr io.Writer

	// Startup and PrefsPath back the tray and dashboard desktop preferences.
	Startup   desktop.Startup
	PrefsPath string
	// SSH is the host ssh client used by the tunnel launcher.
	SSH string
	// Background marks a start-at-sign-in launch: it honors Start minimized,
	// but only while the application is READY; setup, recovery and a failed
	// start are always shown.
	Background bool
	// Epoch is the executable entry time; startup marks are measured from it.
	Epoch desktop.Epoch
	// Exe is the running executable, the destination of an update (empty:
	// os.Executable). UpdateTransport carries the update subsystem's requests
	// (nil: the default transport) and UpdateStart starts its replacement
	// helper (nil: a detached process); tests substitute them.
	Exe             string
	UpdateTransport http.RoundTripper
	UpdateStart     func(exe string, args []string) error
}

// newDesktopApp builds the production composition. homeFlag is an explicit
// home (`desktop --home`); empty means HACHIDORI_HOME, then the locator.
func newDesktopApp(p desktop.Platform, pk desktop.FolderPicker, st desktop.Startup, prefsPath, homeFlag string) *desktopApp {
	return &desktopApp{
		Platform: p, Picker: pk,
		Discover: func() (home.Discovery, error) { return home.Discover(homeFlag) },
		Remember: home.Remember,
		Env:      firstrun.Env{FreeSpace: firstrun.DefaultFreeSpace},
		APIAddr:  server.DefaultListen, DashAddr: dashboard.DefaultListen,
		Stderr:  os.Stderr,
		Startup: st, PrefsPath: prefsPath, SSH: "ssh",
		Epoch: entry,
	}
}

// runDesktopApp is the production no-argument entry point.
func runDesktopApp(p desktop.Platform, pk desktop.FolderPicker) error {
	prefsPath, err := desktop.PrefsPath()
	if err != nil {
		return err
	}
	return newDesktopApp(p, pk, desktop.NativeStartup(), prefsPath, "").launch()
}

// launch runs the application and settles the two outcomes that are not
// failures of the caller: a duplicate launch that activated the running
// instance, and a sign-in launch that failed with no console to say why.
func (a *desktopApp) launch() error {
	err := a.run()
	switch {
	case errors.Is(err, errActivated):
		fmt.Fprintln(a.Stderr, "hachidori: already running for this user; activated the existing window")
		return nil
	case err != nil && a.Background:
		a.Platform.ReportError("Hachidori could not start", err.Error())
	}
	return err
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
	if errors.Is(err, desktop.ErrAlreadyRunning) {
		// Duplicate launch: activate the running instance instead of starting
		// a second controller or worker. Nothing has been started yet.
		if aerr := a.Platform.Activate(); aerr != nil {
			return fmt.Errorf("%w (and it could not be activated: %v)", err, aerr)
		}
		return errActivated
	}
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
	tun := tunnel.NewManager(a.sshExe())
	defer tun.Disconnect()
	var logMu sync.Mutex
	var logs []*os.File
	var dashNow *dashboard.Dashboard // guarded by logMu; the dashboard bound to the current runtime
	defer func() {
		logMu.Lock()
		defer logMu.Unlock()
		for _, f := range logs {
			f.Close()
		}
	}()

	// The startup entry runs this executable in background mode. The home is
	// embedded only when the desktop cannot rediscover it by itself; a home
	// chosen in first run is remembered in the bootstrap locator.
	homeArg := ""
	if d.Source == home.SourceExplicit || d.Source == home.SourceEnv {
		homeArg = d.Home.Root
	}
	mgr := &desktop.Manager{Path: a.PrefsPath, Startup: a.Startup}
	if exe, err := os.Executable(); err == nil {
		// An unusable command only disables enabling start at sign-in.
		mgr.Command, _ = desktop.StartupCommand(exe, homeArg)
	}

	// The typed settings authority composes the desktop preference manager
	// (start at sign-in stays the OS startup entry) and owns only the saved
	// runtime defaults, stored beside desktop.json.
	prefs := settingsStore(a.PrefsPath, mgr)

	// The Models & Runtimes manager acts only through the application
	// controller (which owns setup, activation state and restart) and the
	// setup/home inventory behind it; the dashboard gets no filesystem
	// authority. ctl is assigned below, before anything can open a runtime.
	var ctl *app.Controller
	models := modelManager{ctl: func() *app.Controller { return ctl }}

	// The update subsystem does nothing until the operator posts a Check,
	// Download or Restart & update action on Settings > Updates: no timer, no
	// startup check, no request while a page is rendered. quit ends the
	// application for the replacement helper once the window exists.
	var quitMu sync.Mutex
	var quitFn func()
	updates := a.newUpdateManager(prefs, func() *app.Controller { return ctl }, homeArg, func() {
		quitMu.Lock()
		f := quitFn
		quitMu.Unlock()
		if f != nil {
			f()
		}
	})

	open := func(root string) (app.Runtime, error) {
		if a.Open != nil {
			return a.Open(root)
		}
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
				// Saved experiment history lives under HACHIDORI_HOME only.
				HistoryDir: h.Path("state", "history"),
				Desktop:    prefs,
				Settings:   prefs,
				Models:     models,
				// Development Connection profiles persist in the same
				// settings authority; the live tunnel is the one tun
				// manager, shared by every runtime's dashboard.
				Connections: prefs,
				Updates:     updates,
				WebView2:    version,
				PathPicker:  dashboardPathPicker(a.Picker),
			})
			// An experiment of a replaced runtime's dashboard must not keep
			// running against the next runtime.
			logMu.Lock()
			prev := dashNow
			dashNow = dash
			logMu.Unlock()
			if prev != nil {
				prev.StopExperiment()
			}
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

	// Bind both loopback addresses before anything can start a worker: a
	// port that is already taken must fail here, not after the runtime was
	// started and with nothing left to stop it.
	apiLn, err := net.Listen("tcp", a.APIAddr)
	if err != nil {
		return err
	}
	dashLn, err := net.Listen("tcp", a.DashAddr)
	if err != nil {
		apiLn.Close()
		return err
	}

	selected := ""
	if plan.Mode == firstrun.ModeLaunch || plan.Mode == firstrun.ModeResume {
		selected = plan.Home
	}
	ctl = app.New(app.Config{Home: selected, Open: open, Setup: a.Setup, Installed: a.Env.IsInstalled})
	flow := firstrun.New(firstrun.Config{Ctl: ctl, Plan: plan, Picker: a.Picker, Env: a.Env, Remember: a.Remember})
	startFailed := false
	if plan.Mode == firstrun.ModeLaunch {
		// A configured, installed home: normal startup. A failure is kept by
		// the controller and shown with Retry, never as an opaque exit.
		if err := ctl.Start(); err != nil {
			startFailed = true
			fmt.Fprintln(a.Stderr, "hachidori: start:", err)
		}
	}

	apiSrv := &http.Server{Handler: sw, ReadHeaderTimeout: 10 * time.Second}
	// First run uses the same settings authority for its locale: it lives
	// beside desktop.json, so it exists before any home is selected.
	fr := firstrun.NewHandler(flow, sw.dashboard)
	fr.Locale = prefs.ResolvedLocale
	dashSrv := &http.Server{Handler: fr, ReadHeaderTimeout: 10 * time.Second}
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
			quitMu.Lock()
			quitFn = stop
			quitMu.Unlock()
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
			res := &desktop.Resident{App: ctl, Prefs: mgr}
			// Only a healthy configured launch may begin hidden in the tray:
			// first run, recovery and a failed start are always visible.
			hidden := plan.Mode == firstrun.ModeLaunch && !startFailed &&
				desktop.StartHidden(a.Background, res.Preferences())
			openURL := dashURL
			if plan.Mode == firstrun.ModeLaunch {
				openURL = desktop.OpenURL(dashURL, res.Summary())
			}
			a.Epoch.Mark(a.Stderr, "shell composed")
			err = a.Platform.Open(ctx, desktop.Window{Title: "Hachidori", URL: openURL, DataDir: dataDir, Policy: pol, Epoch: a.Epoch,
				Resident: res, StartHidden: hidden})
			res.Wait()
			stop()
		}
	}

	logMu.Lock()
	last := dashNow
	logMu.Unlock()
	if last != nil {
		last.StopExperiment()
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

// dashboardPathPicker exposes the native picker to the dashboard when it can
// choose files, translating a dismissed dialog to the dashboard's
// cancellation. A folder-only picker leaves typed paths.
func dashboardPathPicker(p desktop.FolderPicker) dashboard.PathPicker {
	if picker, ok := p.(desktop.PathPicker); ok {
		return pathPicker{picker}
	}
	return nil
}

type pathPicker struct{ p desktop.PathPicker }

func pickResult(path string, err error) (string, error) {
	if errors.Is(err, desktop.ErrPickCancelled) {
		return "", dashboard.ErrPickCancelled
	}
	return path, err
}

func (a pathPicker) PickOpen(ctx context.Context, title string) (string, error) {
	return pickResult(a.p.PickOpen(ctx, title))
}

func (a pathPicker) PickSave(ctx context.Context, title string) (string, error) {
	return pickResult(a.p.PickSave(ctx, title))
}

func (a pathPicker) PickFolder(ctx context.Context, title string) (string, error) {
	return pickResult(a.p.PickFolder(ctx, title))
}

// modelManager adapts the application controller to the dashboard's Models
// & Runtimes manager. It adds no behavior: every action is the controller's,
// which delegates to the setup authority and never restarts implicitly.
type modelManager struct{ ctl func() *app.Controller }

func (m modelManager) State() dashboard.ModelsState {
	c := m.ctl()
	snap := c.Snapshot()
	st := dashboard.ModelsState{RestartRequired: snap.RestartRequired, Busy: modelOp(snap.Operation, snap.Home), Last: modelOp(snap.Maintenance, snap.Home)}
	if len(snap.Checks) > 0 {
		st.Checks = make(map[string]dashboard.ModelCheck, len(snap.Checks))
		for k, c := range snap.Checks {
			st.Checks[k] = dashboard.ModelCheck{OK: c.OK, Msg: c.Message, Time: c.Time}
		}
	}
	inv, err := c.Inventory(false)
	if err != nil {
		st.Err = err.Error()
	}
	st.Inventory = inv
	return st
}

// modelOp is the dashboard's view of one application action. It restates the
// controller's operation (plan, phase, step and progress); nothing is added.
func modelOp(o *app.Operation, root string) *dashboard.ModelOp {
	if o == nil {
		return nil
	}
	op := &dashboard.ModelOp{Kind: o.Kind, Device: o.Device, Model: o.Model, Target: o.Target, Phase: o.Phase,
		Plan: o.Plan, Phases: o.Phases, Started: o.Started, Finished: o.Finished}
	if p := o.Progress; p != nil {
		op.Step, op.Detail, op.Done, op.Total, op.Item, op.Items = string(p.Step), p.Detail, p.Done, p.Total, p.Item, p.Items
	}
	if o.Failure != nil {
		op.Failure, op.FailurePhase, op.FailureStep = o.Failure.Message, o.Failure.Phase, o.Failure.Step
	}
	if root != "" {
		op.Log = app.SetupLogPath(root)
	}
	return op
}

func (m modelManager) Verify(kind, id string) error { return m.ctl().Verify(kind, id) }
func (m modelManager) Materialize(device, model string) error {
	return m.ctl().Materialize(app.SetupParams{Device: device, Model: model})
}
func (m modelManager) Repair(device, model string) error {
	return m.ctl().Repair(app.SetupParams{Device: device, Model: model})
}
func (m modelManager) Activate(device, model string) error {
	return m.ctl().Activate(app.SetupParams{Device: device, Model: model})
}
func (m modelManager) Remove(kind, id string) error { return m.ctl().Remove(kind, id) }
func (m modelManager) Start() error                 { return m.ctl().Start() }
func (m modelManager) Stop() error                  { return m.ctl().Stop() }
func (m modelManager) Restart() error               { return m.ctl().Restart() }

// settingsStore is the desktop's settings authority: the desktop preference
// manager plus the saved runtime defaults in settings.json beside desktop.json.
func settingsStore(prefsPath string, d settings.Desktop) *settings.Store {
	return &settings.Store{Path: desktop.SettingsPath(prefsPath), Desktop: d}
}

func (a *desktopApp) sshExe() string {
	if a.SSH == "" {
		return "ssh"
	}
	return a.SSH
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
