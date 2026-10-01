// Package firstrun is the desktop first-run / recovery flow: it lets a user
// pick one storage root with the native folder picker, validates it, records
// it in the bootstrap locator, and drives setup, start and warmup through the
// application controller (internal/app).
//
// It owns no runtime state. Setup progress, failure and READY are read from
// app.Controller snapshots; the home is remembered through home.Remember; the
// folder picker returns only a path, which Validate checks before anything is
// written. The flow never falls back from CUDA to CPU: the device is an
// explicit user choice passed to setup unchanged.
package firstrun

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Mode says how the desktop was entered.
type Mode string

const (
	ModeFirstRun Mode = "first_run"        // no home has ever been selected
	ModeResume   Mode = "resume"           // a home is selected but has no valid runtime
	ModeMissing  Mode = "recovery_missing" // the stored home no longer exists / is unavailable
	ModeInvalid  Mode = "recovery_invalid" // the bootstrap locator cannot be used
	ModeLaunch   Mode = "launch"           // a configured, installed home: normal startup
)

// Plan is the startup decision made from home discovery.
type Plan struct {
	Mode   Mode
	Home   string // the selected home (launch/resume) or the stored, unavailable one (missing)
	Detail string // diagnostic text for the invalid-locator case
}

// Decide turns the outcome of home.Discover into a startup Plan. A stored
// home that is missing or a malformed locator is never treated as a first
// run: the user is asked to locate or choose a home explicitly.
func Decide(d home.Discovery, err error, env Env) Plan {
	if err != nil {
		p := Plan{Mode: ModeInvalid, Detail: err.Error()}
		if errors.Is(err, home.ErrStoredHomeMissing) {
			p.Mode = ModeMissing
		}
		var be *home.BootstrapError
		if errors.As(err, &be) {
			p.Home = be.Home
		}
		return p
	}
	if d.Source == home.SourceUnconfigured || d.Home.Root == "" {
		return Plan{Mode: ModeFirstRun}
	}
	if env.IsInstalled(d.Home.Root) {
		return Plan{Mode: ModeLaunch, Home: d.Home.Root}
	}
	return Plan{Mode: ModeResume, Home: d.Home.Root}
}

// Errors returned by flow actions.
var (
	ErrNoSelection = errors.New("choose a storage folder first")
	ErrBusy        = errors.New("a setup action is already in progress")
	ErrExisting    = errors.New("an existing Hachidori installation was found there; use it instead of installing over it")
	ErrNoExisting  = errors.New("the selected folder has no existing Hachidori installation")
	ErrNotRetry    = errors.New("there is nothing to retry")
	ErrActive      = errors.New("the installation is already activated; the storage location cannot be changed here")
	ErrDevice      = errors.New(`device must be "cuda" or "cpu"`)
)

// Controller is the part of app.Controller the flow uses.
type Controller interface {
	Snapshot() app.Snapshot
	SetHome(root string) error
	Setup(app.SetupParams) error
	Start() error
	Restart() error
	Subscribe() (<-chan struct{}, func())
}

var _ Controller = (*app.Controller)(nil)

// Config wires the flow.
type Config struct {
	Ctl      Controller
	Plan     Plan
	Picker   desktop.FolderPicker
	Env      Env
	Remember func(root string) (home.Home, error) // home.Remember
	// Model is the catalog model to install; empty selects setup.DefaultModel.
	Model string
}

// Flow is the first-run/recovery state machine. Create it with New.
type Flow struct {
	cfg Config

	mu     sync.Mutex
	sel    *Validation
	device string
	busy   bool // an install (setup + commit + start hand-off) is in flight
	// unremembered is a home whose setup succeeded but whose bootstrap record
	// could not be written (rememberErr says why); Retry repeats only that.
	unremembered string
	rememberErr  error
	done         bool // READY reached; the desktop home takes over
	picking      bool
	logPath      string
}

// New creates the flow. A resumable home is pre-selected (and validated) so
// the user can simply install into it.
func New(cfg Config) *Flow {
	if cfg.Model == "" {
		cfg.Model = setup.DefaultModel
	}
	f := &Flow{cfg: cfg, device: "cuda"}
	if cfg.Plan.Mode == ModeResume && cfg.Plan.Home != "" {
		v := Validate(cfg.Plan.Home, cfg.Env)
		f.sel = &v
	}
	return f
}

// Select shows the native folder picker, validates the chosen root and keeps
// it as the current selection. A cancelled picker changes nothing.
func (f *Flow) Select(ctx context.Context) (Validation, error) {
	f.mu.Lock()
	if f.picking {
		f.mu.Unlock()
		return Validation{}, ErrBusy
	}
	if err := f.canChangeLocked(); err != nil {
		f.mu.Unlock()
		return Validation{}, err
	}
	f.picking = true
	f.mu.Unlock()
	defer func() { f.mu.Lock(); f.picking = false; f.mu.Unlock() }()

	path, err := f.cfg.Picker.PickFolder(ctx, "Choose where Hachidori keeps its models and runtime")
	if err != nil {
		return Validation{}, err
	}
	v := Validate(path, f.cfg.Env)
	f.mu.Lock()
	f.sel = &v
	f.mu.Unlock()
	return v, nil
}

// canChangeLocked reports whether the selection may change now: not while an
// action is in flight and never after a successful activation.
func (f *Flow) canChangeLocked() error {
	if f.busy {
		return ErrBusy
	}
	snap := f.cfg.Ctl.Snapshot()
	switch snap.State {
	case app.Unconfigured, app.NotInstalled:
		return nil
	case app.Failed:
		if snap.Home != "" && f.cfg.Env.IsInstalled(snap.Home) {
			return ErrActive
		}
		return nil
	case app.Installing:
		return app.ErrSetupRunning
	}
	return ErrActive
}

// Install validates the selection again and starts setup through the
// controller. The selection is written to the bootstrap locator only when
// setup has succeeded (the commit point): a failed or interrupted setup never
// replaces what the locator said before, so a relaunch returns to exactly the
// screen it started from and never claims an installation that was not
// completed. Starting the runtime after a successful setup is chained
// automatically. Install returns as soon as setup is accepted.
func (f *Flow) Install(device string) error {
	if device != "cuda" && device != "cpu" {
		return ErrDevice
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.busy {
		return ErrBusy
	}
	if f.sel == nil {
		return ErrNoSelection
	}
	if f.sel.Existing != nil {
		return ErrExisting
	}
	if err := f.canChangeLocked(); err != nil {
		return err
	}
	v := Validate(f.sel.Picked, f.cfg.Env)
	f.sel = &v
	if !v.OK() {
		return errors.New(v.Problem)
	}
	if v.Existing != nil {
		return ErrExisting
	}
	f.device = device
	return f.beginSetupLocked(v.Home)
}

// beginSetupLocked hands home to the controller and starts setup.
func (f *Flow) beginSetupLocked(root string) error {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return fmt.Errorf("creating the storage folder: %w", err)
	}
	if err := f.cfg.Ctl.SetHome(root); err != nil {
		return err
	}
	logw, closeLog := f.openLog(root)
	ch, cancel := f.cfg.Ctl.Subscribe()
	t0 := time.Now()
	err := f.cfg.Ctl.Setup(app.SetupParams{Device: f.device, Model: f.cfg.Model, Log: logw})
	if err != nil {
		cancel()
		closeLog()
		return err
	}
	f.busy = true
	go f.watch(ch, cancel, closeLog, root, t0)
	return nil
}

func (f *Flow) openLog(root string) (io.Writer, func()) {
	dir := filepath.Join(root, "logs")
	if os.MkdirAll(dir, 0o755) != nil {
		return io.Discard, func() {}
	}
	p := filepath.Join(dir, "setup.log")
	fh, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return io.Discard, func() {}
	}
	f.logPath = p
	return fh, func() { fh.Close() }
}

// watch waits for the setup started at t0 to finish. On success it commits
// the home to the bootstrap locator and starts the runtime; on failure it
// changes nothing (the controller keeps the structured failure).
func (f *Flow) watch(ch <-chan struct{}, cancel func(), closeLog func(), root string, t0 time.Time) {
	defer cancel()
	for {
		snap := f.cfg.Ctl.Snapshot()
		if snap.Operation == nil && snap.Last != nil && snap.Last.Kind == app.OpSetup && !snap.Last.Started.Before(t0) {
			closeLog()
			if snap.Last.Failure == nil {
				f.mu.Lock()
				f.commitLocked(root)
				f.mu.Unlock()
			}
			f.mu.Lock()
			f.busy = false
			f.mu.Unlock()
			return
		}
		<-ch
	}
}

// commitLocked remembers root and starts the runtime. If the bootstrap
// record cannot be written the runtime is not started and the error is shown
// with Retry; nothing claims the desktop is configured.
func (f *Flow) commitLocked(root string) {
	if _, err := f.cfg.Remember(root); err != nil {
		f.unremembered, f.rememberErr = root, fmt.Errorf("remembering the storage folder: %w", err)
		return
	}
	f.unremembered, f.rememberErr = "", nil
	_ = f.cfg.Ctl.Start() // a failure is recorded by the controller and shown as failed
}

// UseExisting activates a valid installation found at the selected root
// without running setup: the root is remembered and the runtime started.
func (f *Flow) UseExisting() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.busy {
		return ErrBusy
	}
	if f.sel == nil {
		return ErrNoSelection
	}
	if err := f.canChangeLocked(); err != nil {
		return err
	}
	v := Validate(f.sel.Picked, f.cfg.Env)
	f.sel = &v
	if !v.OK() {
		return errors.New(v.Problem)
	}
	if v.Existing == nil {
		return ErrNoExisting
	}
	if err := f.cfg.Ctl.SetHome(v.Home); err != nil {
		return err
	}
	// The installation is already complete: remember it, then start.
	f.commitLocked(v.Home)
	return f.rememberErr
}

// Retry repeats the failed action: setup again (the same reconciliation, one
// operation at a time) or the runtime start when the activation exists.
func (f *Flow) Retry() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.busy {
		return ErrBusy
	}
	if f.unremembered != "" {
		f.commitLocked(f.unremembered)
		return f.rememberErr
	}
	snap := f.cfg.Ctl.Snapshot()
	if snap.State != app.Failed || snap.Home == "" {
		return ErrNotRetry
	}
	if f.cfg.Env.IsInstalled(snap.Home) {
		if snap.Status != nil {
			return f.cfg.Ctl.Restart()
		}
		return f.cfg.Ctl.Start()
	}
	return f.beginSetupLocked(snap.Home)
}

// Change returns to storage selection. It is allowed only before a
// successful activation and never touches the bootstrap record or any files:
// nothing changes until a new location is installed.
func (f *Flow) Change() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.canChangeLocked(); err != nil {
		return err
	}
	if err := f.cfg.Ctl.SetHome(""); err != nil {
		return err
	}
	f.sel = nil
	f.unremembered, f.rememberErr = "", nil
	return nil
}

// Done reports whether the desktop home (dashboard) should be shown instead
// of this flow: after READY, or, for a normal launch, once the runtime is
// bound (its own status then carries starting/warming/failed).
func (f *Flow) Done() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.done {
		return true
	}
	snap := f.cfg.Ctl.Snapshot()
	if snap.State == app.Ready || (f.cfg.Plan.Mode == ModeLaunch && snap.Status != nil) {
		f.done = true
	}
	return f.done
}

// Stage names of the view.
const (
	StageSelect     = "select"
	StageInstalling = "installing"
	StageStarting   = "starting"
	StageReady      = "ready"
	StageFailed     = "failed"
)

// SetupPhases are the real setup boundaries, in order (internal/setup).
var SetupPhases = []setup.Phase{setup.PhasePreparing, setup.PhaseRuntime, setup.PhaseModel, setup.PhaseActivation}

// PhaseView is one setup phase. Status is done, current, failed or pending;
// a phase is current or done only if setup actually entered it.
type PhaseView struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

// StepView is the work inside the current setup phase, as the setup authority
// reported it. Total is zero for a step whose amount of work is unknown; the
// page then shows it as in progress without a percentage.
type StepView struct {
	Step   string `json:"step"`
	Detail string `json:"detail,omitempty"`
	Done   int64  `json:"done,omitempty"`
	Total  int64  `json:"total,omitempty"`
	Item   int    `json:"item,omitempty"`
	Items  int    `json:"items,omitempty"`
}

// Identity is the runtime identity shown at READY, from the canonical
// /v1/status document.
type Identity struct {
	Home    string `json:"home"`
	Runtime string `json:"runtime"`
	ModelID string `json:"model_id"`
	Model   string `json:"model"`
	Device  string `json:"device"`
}

// View is everything the first-run screen renders.
type View struct {
	Mode       Mode        `json:"mode"`
	Notice     string      `json:"notice,omitempty"`
	StoredHome string      `json:"stored_home,omitempty"`
	Stage      string      `json:"stage"`
	State      app.State   `json:"state"`
	Selection  *Validation `json:"selection,omitempty"`
	Device     string      `json:"device"`
	Model      string      `json:"model"`
	Phases     []PhaseView `json:"phases,omitempty"`
	Step       *StepView   `json:"step,omitempty"`
	Worker     string      `json:"worker,omitempty"`
	// WorkerPhase is the worker's own phase (spawning, importing, loading,
	// warming) while the stage is starting.
	WorkerPhase string       `json:"worker_phase,omitempty"`
	Failure     *app.Failure `json:"failure,omitempty"`
	CanRetry    bool         `json:"can_retry"`
	CanChange   bool         `json:"can_change"`
	SetupLog    string       `json:"setup_log,omitempty"`
	Identity    *Identity    `json:"identity,omitempty"`
}

// View projects the current screen from the controller snapshot.
func (f *Flow) View() View {
	f.mu.Lock()
	busy, dev, logPath, remErr := f.busy, f.device, f.logPath, f.rememberErr
	var sel *Validation
	if f.sel != nil {
		c := *f.sel
		sel = &c
	}
	f.mu.Unlock()
	snap := f.cfg.Ctl.Snapshot()

	v := View{Mode: f.cfg.Plan.Mode, StoredHome: f.cfg.Plan.Home, State: snap.State, Selection: sel,
		Device: dev, Model: f.cfg.Model, Failure: snap.Failure, SetupLog: logPath}
	switch f.cfg.Plan.Mode {
	case ModeMissing:
		v.Notice = "The Hachidori storage folder " + f.cfg.Plan.Home + " is missing or unavailable. " +
			"Locate the folder that holds your existing Hachidori installation, or choose a new location to install again. " +
			"Nothing has been installed or deleted."
	case ModeInvalid:
		v.Notice = "Hachidori could not read its saved storage location (" + f.cfg.Plan.Detail + "). " +
			"Locate your Hachidori folder or choose a new location. Nothing has been installed or deleted."
	}
	switch snap.State {
	case app.Installing:
		v.Stage = StageInstalling
	case app.Installed, app.Starting, app.Warming, app.Stopping:
		v.Stage = StageStarting
	case app.Ready:
		v.Stage = StageReady
	case app.Failed:
		v.Stage = StageFailed
	default:
		v.Stage = StageSelect
	}
	if remErr != nil && snap.State != app.Ready {
		v.Stage = StageFailed
		v.Failure = &app.Failure{Source: "bootstrap", Message: remErr.Error()}
	}
	v.Phases = phaseViews(snap)
	if op := snap.Operation; op != nil && op.Kind == app.OpSetup && op.Progress != nil {
		p := op.Progress
		v.Step = &StepView{Step: string(p.Step), Detail: p.Detail, Done: p.Done, Total: p.Total, Item: p.Item, Items: p.Items}
	}
	if snap.Status != nil {
		w := snap.Status.Worker
		v.WorkerPhase = w.Phase
		v.Worker = w.State
		if w.Phase != "" {
			v.Worker += " (" + w.Phase + ")"
		}
		if snap.State == app.Ready {
			r := snap.Status.Runtime
			v.Identity = &Identity{Home: r.Home, Runtime: r.Runtime, ModelID: r.ModelID, Model: r.Model, Device: r.Device}
		}
	}
	if (snap.State == app.Failed || remErr != nil) && !busy {
		v.CanRetry = true
		if snap.Home != "" && !f.cfg.Env.IsInstalled(snap.Home) {
			v.CanChange = true
		}
	}
	return v
}

func phaseViews(snap app.Snapshot) []PhaseView {
	op := snap.Operation
	if op == nil || op.Kind != app.OpSetup {
		op = snap.Last
	}
	if op == nil || op.Kind != app.OpSetup {
		return nil
	}
	entered := map[string]int{}
	for i, p := range op.Phases {
		entered[p] = i
	}
	inFlight := snap.Operation != nil
	out := make([]PhaseView, 0, len(SetupPhases))
	for _, p := range SetupPhases {
		i, ok := entered[string(p)]
		pv := PhaseView{Name: string(p), Status: "pending"}
		switch {
		case !ok:
		case i < len(op.Phases)-1:
			pv.Status = "done"
		case inFlight:
			pv.Status = "current"
		case op.Failure != nil:
			pv.Status = "failed"
		default:
			pv.Status = "done"
		}
		out = append(out, pv)
	}
	return out
}
