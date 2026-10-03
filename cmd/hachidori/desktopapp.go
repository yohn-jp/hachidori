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
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
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
	Open app.OpenFunc
	// OpenRuntime, when set, replaces app.ConfiguredRuntime beneath the real
	// dashboard binding (tests only): it receives the desired-residents
	// reader and the bind callback the production opener would use.
	OpenRuntime func(requested func() ([]string, error), bind bindFunc) app.OpenFunc
	Stderr      io.Writer

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

	// One form token for the whole process: every dashboard a runtime
	// rebind creates accepts the pages the window already shows.
	formToken := dashboard.NewToken()
	apiAddr := a.APIAddr
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
		// bind attaches the dashboard and the API handler to a newly opened
		// runtime, whether it is the one worker or a resident set.
		bind := func(status func() server.Status, lc dashboard.Lifecycle, dec server.Decider, info server.Runtime, started time.Time) {
			h := home.Home{Root: root}
			dash := dashboard.New(dashboard.Config{
				APIAddr:   apiAddr,
				Status:    status,
				Lifecycle: lc,
				Doctor:    func(out io.Writer) bool { return doctor.Run(root, out) },
				Tunnel:    tun,
				PrefsPath: h.Path("state", "dashboard.json"),
				// Saved experiment history lives under HACHIDORI_HOME only.
				HistoryDir: h.Path("state", "history"),
				Desktop:    prefs,
				Settings:   prefs,
				Models:     models,
				// System One variant actions go through the same controller.
				Variants: models,
				// The desired resident models are saved by the same settings
				// authority and honored by the next open of the runtime.
				Residency: prefs,
				// Development Connection profiles persist in the same
				// settings authority; the live tunnel is the one tun
				// manager, shared by every runtime's dashboard.
				Connections: prefs,
				Updates:     updates,
				WebView2:    version,
				PathPicker:  dashboardPathPicker(a.Picker),
				Token:       formToken,
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
			sw.set(server.HandlerSince(dec, info, started), dash)
		}
		rt, err := a.openRuntime(rctx, lf, prefs.Residents, bind)(root)
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
	apiAddr = apiLn.Addr().String()
	localEndpoint, err := tunnel.LocalEndpointFromAddr(apiAddr)
	if err != nil {
		apiLn.Close()
		dashLn.Close()
		return err
	}
	if err := prefs.SetLocalEndpoint(localEndpoint); err != nil {
		apiLn.Close()
		dashLn.Close()
		return err
	}

	selected := ""
	if plan.Mode == firstrun.ModeLaunch || plan.Mode == firstrun.ModeResume {
		selected = plan.Home
	}
	ctl = app.New(app.Config{Home: selected, Open: open, Setup: a.Setup, Installed: a.Env.IsInstalled,
		// The controller compares the desired residents with the bound set
		// to tell the operator a restart is required; it keeps no copy.
		Residents: func() []string { r, _ := prefs.Residents(); return r }})
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

// bindFunc attaches the dashboard and API handler to a newly opened runtime.
type bindFunc func(status func() server.Status, lc dashboard.Lifecycle, dec server.Decider, info server.Runtime, started time.Time)

// openRuntime is the production runtime binding of the desktop: the active
// model's one worker, or a resident set when the saved selection adds
// residents beside it (app.ConfiguredRuntime). requested is the settings
// authority's desired residents, read at every open.
func (a *desktopApp) openRuntime(parent context.Context, log io.Writer, requested func() ([]string, error), bind bindFunc) app.OpenFunc {
	if a.OpenRuntime != nil {
		return a.OpenRuntime(requested, bind)
	}
	return app.ConfiguredRuntime(parent, log, worker.DefaultPolicy, requested,
		func(b *app.WorkerBinding) { bind(b.Status, b.Lifecycle, b.Supervisor, b.Info, b.Started) },
		func(s *app.ResidentSet) { bind(s.Status, s, s, s.Runtime(), s.Started()) })
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
	st := dashboard.ModelsState{RestartRequired: snap.RestartRequired, ResidencyChanged: snap.ResidencyChanged, Busy: modelOp(snap.Operation, snap.Home), Last: modelOp(snap.Maintenance, snap.Home)}
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
	st.Forge = forgeState(c.Forge())
	if snap.Operation != nil && snap.Operation.Kind == app.OpForgeBuildEvaluate {
		st.Forge.Resolution = forgeResolution(snap.Operation.ForgeResolution)
	} else if snap.Maintenance != nil && snap.Maintenance.Kind == app.OpForgeBuildEvaluate {
		st.Forge.Resolution = forgeResolution(snap.Maintenance.ForgeResolution)
	}
	return st
}

func forgeResolution(r *app.ForgeBuildEvaluateResolution) *dashboard.ForgeResolution {
	if r == nil {
		return nil
	}
	return &dashboard.ForgeResolution{
		Source: r.Source, Recipe: r.Recipe, Variant: r.Variant,
		CandidateDevice: forgeResolvedValue(r.CandidateDevice),
		ReferenceDevice: forgeResolvedValue(r.ReferenceDevice),
		ReferenceDType:  forgeResolvedValue(r.ReferenceDType),
		CandidateDType:  r.CandidateDType,
	}
}

func forgeResolvedValue(r app.ForgeResolvedValue) dashboard.ForgeResolvedValue {
	mode := "Override"
	if r.Mode == app.ForgeSelectionAuto {
		mode = "Auto"
	}
	return dashboard.ForgeResolvedValue{Mode: mode, Value: r.Value}
}

// forgeState restates the controller's Forge records (preflights, probes,
// diagnostics) for the dashboard. Nothing is derived beyond the evidence
// state the controller judged: a preflight that is not current evidence is
// shown with that state as its outcome (never READY) and the reason first.
func forgeState(f app.ForgeState) dashboard.ForgeState {
	var out dashboard.ForgeState
	for _, r := range f.Preflights {
		row := dashboard.PreflightRow{Kind: r.Kind, Model: r.Model, Variant: r.Variant, Recipe: r.Recipe, Device: r.Device, At: r.CreatedAt, Outcome: r.Outcome,
			Pass: r.Counts.Pass, Warning: r.Counts.Warning, Blocker: r.Counts.Blocker, Unknown: r.Counts.Unknown, NotMeasured: r.NotMeasured}
		if !r.Current() {
			row.Outcome = r.Evidence
			if row.Outcome == "" {
				row.Outcome = setup.EvidenceLegacy
			}
			why := "this report predates identity binding and proves nothing about the current runtime, source, variant or recipe"
			if len(r.Stale) > 0 {
				why = "recorded for identities that changed since (" + strings.Join(r.Stale, ", ") + ")"
			}
			row.Findings = append(row.Findings, dashboard.FindingRow{ID: "preflight.evidence", Status: row.Outcome,
				Summary: "not current: " + why + "; its recorded outcome was " + r.Outcome + ". Run the preflight again."})
		}
		for _, fd := range setup.SortedFindings(r.Findings) {
			if fd.Status != setup.FindingPass {
				row.Findings = append(row.Findings, dashboard.FindingRow{ID: fd.ID, Status: string(fd.Status), Summary: fd.Summary})
			}
		}
		out.Preflights = append(out.Preflights, row)
	}
	for _, p := range f.Probes {
		row := dashboard.ProbeRow{Variant: p.Variant, Device: p.Device, Result: p.Result, Phase: p.Phase, Error: p.Error, StartedAt: p.StartedAt,
			ManifestSHA256: p.VariantManifestSHA256, Provider: p.Provider, DType: p.Execution.DType, DeviceName: p.Execution.DeviceName,
			StartupMS: p.Timing.StartupMS, LoadMS: p.Timing.LoadMS, WarmupMS: p.Timing.WarmupMS, RequestMS: p.Timing.RequestMS}
		if d := p.Decision; d != nil {
			row.Choice, row.Confidence = d.Choice, d.Confidence
		}
		out.Probes = append(out.Probes, row)
	}
	for _, d := range f.Diagnostics {
		out.Diagnostics = append(out.Diagnostics, dashboard.DiagnosticRow{ID: d.ID, Kind: d.Kind, Phase: d.Phase, Model: d.Model, Variant: d.Variant, Created: d.Created, Error: d.Error})
	}
	return out
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
		op.Step, op.Detail, op.Done, op.Total, op.Item, op.Items, op.Resumed = string(p.Step), p.Detail, p.Done, p.Total, p.Item, p.Items, p.Resumed
	}
	if o.Failure != nil {
		op.Failure, op.FailurePhase, op.FailureStep, op.Diagnostic = o.Failure.Message, o.Failure.Phase, o.Failure.Step, o.Failure.Diagnostic
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

// The System One variant actions are likewise the controller's.
func (m modelManager) ActivateVariant(device, model, variant string, experimental bool) error {
	return m.ctl().ActivateVariant(app.SetupParams{Device: device, Model: model}, variant, experimental)
}
func (m modelManager) Optimize(model, recipe string) error { return m.ctl().Optimize(model, recipe) }

func (m modelManager) BuildAndEvaluate(r dashboard.ForgeBuildEvaluateRequest) error {
	return m.ctl().BuildAndEvaluate(app.ForgeBuildEvaluateParams{
		Source: r.Source, Optimization: optimize.Request{Model: r.Source, Recipe: r.Profile}, Device: r.Device,
		ReferenceDevice: r.ReferenceDevice, ReferenceDType: r.ReferenceDType,
		Dataset: r.Dataset, Questions: r.Questions, Policy: r.Policy, Materialize: r.Materialize,
	})
}

// CertifyVariant is the self-contained Forge certification: the controller
// produces and binds both runs itself from the semantic inputs.
func (m modelManager) CertifyVariant(r dashboard.CertifyRequest) error {
	return m.ctl().CertifyVariant(app.ForgeCertifyParams{Variant: r.Variant, Device: r.Device, ReferenceDevice: r.ReferenceDevice,
		ReferenceDType: r.ReferenceDType, Dataset: r.Dataset, Questions: r.Questions, Policy: r.Policy, Materialize: r.Materialize})
}

// Apply is the controller's certified-variant apply transaction
// (ApplyCertifiedVariant) on this persistent desktop controller, accepted
// as one background action: the dashboard calls it once and never chains
// activation, restart or verification itself.
func (m modelManager) Apply(device, model, variant string, materialize bool) error {
	return m.ctl().StartApply(app.ApplyParams{Variant: variant, Device: device, Model: model, Materialize: materialize})
}
func (m modelManager) Preflight(kind, model, recipe, variant, device string) error {
	return m.ctl().Preflight(app.PreflightParams{Kind: kind, Model: model, Recipe: recipe, Variant: variant, Device: device})
}
func (m modelManager) Probe(variant, device string) error {
	return m.ctl().Probe(app.ProbeParams{Variant: variant, Device: device})
}

// Diagnostic is one stored Forge failure diagnostic, as the document that was
// written (bounded and redacted when it was recorded).
func (m modelManager) Diagnostic(id string) ([]byte, error) {
	snap := m.ctl().Snapshot()
	if snap.Home == "" {
		return nil, errors.New("no home is selected")
	}
	d, err := diagnostics.LoadForge(snap.Home, id)
	if err != nil {
		return nil, err
	}
	return diagnostics.FormatForge(d)
}
func (m modelManager) Start() error   { return m.ctl().Start() }
func (m modelManager) Stop() error    { return m.ctl().Stop() }
func (m modelManager) Restart() error { return m.ctl().Restart() }

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
