package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Explicit model/runtime maintenance actions (Operation kinds).
const (
	OpMaterialize = "materialize"
	OpRepair      = "repair"
	OpActivate    = "activate"
	OpVerify      = "verify"
	OpRemove      = "remove"
)

// Rejections of the maintenance actions.
var (
	ErrOperationRunning = errors.New("this action is already running")
	ErrRestartRequired  = errors.New("the active runtime/model changed; restart the runtime first")
)

// SetupFunc materializes and activates the runtime under root. It must call
// onPhase at each real phase boundary it enters (setup.RunObserved does).
type SetupFunc func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error

// OpenFunc binds a serving Runtime to the active runtime under root. It is
// called on the first Start/Restart and again after a successful setup.
type OpenFunc func(root string) (Runtime, error)

// InstalledFunc reports whether root has a valid activation record.
type InstalledFunc func(root string) bool

// Config wires the controller to the existing authorities.
type Config struct {
	// Home is the selected Hachidori home; "" means unresolved. The
	// controller never discovers a home on its own.
	Home string
	// Open binds the runtime for serving (production: WorkerRuntime).
	Open OpenFunc
	// Setup defaults to setup.RunObserved.
	Setup SetupFunc
	// Installed defaults to home.Home.LoadActive succeeding.
	Installed InstalledFunc
	// Maintenance is the setup/home authority for the explicit model and
	// runtime maintenance actions. Its defaults are the internal/setup
	// operations on home.Home{Root: root}; tests substitute fakes.
	Maintenance Maintenance
}

// Maintenance are the explicit model/runtime operations of the setup/home
// authority, addressed by home root.
type Maintenance struct {
	Inspect     func(root string, verify bool) setup.Inventory
	Materialize func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error
	Repair      func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error
	Activate    func(root, device, model string, log io.Writer) (changed bool, err error)
	Verify      func(root, kind, id string) error
	Remove      func(root, kind, id string) error
}

func (m Maintenance) withDefaults() Maintenance {
	if m.Inspect == nil {
		m.Inspect = func(root string, verify bool) setup.Inventory { return setup.Inspect(home.Home{Root: root}, verify) }
	}
	if m.Materialize == nil {
		m.Materialize = func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error {
			return setup.Materialize(home.Home{Root: root}, device, model, log, onPhase)
		}
	}
	if m.Repair == nil {
		m.Repair = func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error {
			return setup.Repair(home.Home{Root: root}, device, model, log, onPhase)
		}
	}
	if m.Activate == nil {
		m.Activate = func(root, device, model string, log io.Writer) (bool, error) {
			return setup.Activate(home.Home{Root: root}, device, model, log)
		}
	}
	if m.Verify == nil {
		m.Verify = func(root, kind, id string) error { return setup.Verify(home.Home{Root: root}, kind, id) }
	}
	if m.Remove == nil {
		m.Remove = func(root, kind, id string) error { return setup.Remove(home.Home{Root: root}, kind, id) }
	}
	return m
}

// SetupParams are the explicit parameters of a setup action.
type SetupParams struct {
	Device string    // inference device, as for `hachidori setup --device`
	Model  string    // catalog model ID; empty selects setup.DefaultModel
	Log    io.Writer // setup's human-readable log; nil discards it
}

// Operation is one application action. Setup reports the setup phases it
// has actually entered; there is never a percentage.
type Operation struct {
	Kind     string    `json:"kind"`
	Device   string    `json:"device,omitempty"`
	Model    string    `json:"model,omitempty"`
	Target   string    `json:"target,omitempty"` // verify/remove: "<kind> <id>"
	Phase    string    `json:"phase,omitempty"`  // setup: the phase currently/last entered
	Phases   []string  `json:"phases,omitempty"` // setup: every phase entered, in order
	Started  time.Time `json:"started"`
	Finished time.Time `json:"finished,omitzero"`
	Failure  *Failure  `json:"failure,omitempty"`
	// Cancellable is always false: setup materialization is not safely
	// interruptible, and Stop/Restart are bounded by the worker's own
	// shutdown timeout.
	Cancellable bool `json:"cancellable"`
}

func (o *Operation) clone() *Operation {
	if o == nil {
		return nil
	}
	c := *o
	c.Phases = append([]string(nil), o.Phases...)
	return &c
}

// Recovery states. They are additive to State: a Recovery is only present
// while the application is recovering from, or has given up after, an
// unexpected exit of the resident worker.
const (
	// RecoveryRecovering: the supervisor is restarting a worker that exited
	// unexpectedly. Attempts are bounded by the supervisor's restart policy.
	RecoveryRecovering = "recovering"
	// RecoveryGaveUp: the bounded automatic recovery is exhausted (or the
	// restart attempt itself failed). Nothing restarts until the operator
	// acts; the state is Failed / Needs attention.
	RecoveryGaveUp = "gave_up"
)

// Recovery describes recovery from an unexpected worker exit. It is derived
// from the supervisor's status, never from a second restart counter, and it
// is never present for an operator Stop/Restart/Quit or a startup failure.
type Recovery struct {
	State string `json:"state"`
	// Restarts is the number of automatic restarts the supervisor has made in
	// its current window. The window is the only automatic reset: it expires
	// after a stable interval without another exit. An operator Start or
	// Restart after RecoveryGaveUp binds a fresh supervisor, which is the
	// explicit-operator reset.
	Restarts int      `json:"restarts"`
	Cause    *Failure `json:"cause,omitempty"`
	Message  string   `json:"message"`
}

// Snapshot is the application view at one instant.
type Snapshot struct {
	State     State      `json:"state"`
	Home      string     `json:"home,omitempty"`
	Operation *Operation `json:"operation,omitempty"`      // in flight
	Last      *Operation `json:"last_operation,omitempty"` // most recently finished
	Failure   *Failure   `json:"failure,omitempty"`        // set iff State is Failed
	// Maintenance is the most recently finished model/runtime maintenance
	// action. Its failure never changes State: the active runtime is
	// untouched by a failed maintenance action.
	Maintenance *Operation `json:"last_maintenance,omitempty"`
	// RestartRequired is set while the running worker was started from a
	// runtime/model that is no longer the active one: an explicit Activate
	// changed the activation record and nothing has restarted the worker.
	RestartRequired bool           `json:"restart_required,omitempty"`
	Status          *server.Status `json:"status,omitempty"` // the /v1/status document, when a runtime is bound
	// Recovery is set while an unexpected worker exit is being recovered or
	// recovery gave up. OperatorStopped is set while the worker is down
	// because the operator chose Stop (or Quit), so the two are never
	// confused.
	Recovery        *Recovery `json:"recovery,omitempty"`
	OperatorStopped bool      `json:"operator_stopped,omitempty"`
}

// Controller orchestrates the application actions over the existing
// authorities. Its concurrency rules are deterministic:
//
//   - at most one action (setup, start, stop, restart) is in flight;
//   - Start or Restart while a start/restart is in flight, and Start while
//     the worker is running, are no-ops returning nil;
//   - Stop while a stop is in flight or with no running worker is a no-op;
//   - Setup while setup is in flight returns ErrSetupRunning; Setup while the
//     runtime is running returns ErrRuntimeBusy (setup never changes the
//     activation under a live worker);
//   - any other action while a different action is in flight returns ErrBusy
//     (so Start/Restart never race setup's activation);
//   - rejected actions change nothing.
//
// Setup is not cancellable. Close stops accepting actions, waits for the
// in-flight action, then stops the runtime, all bounded by its context.
type Controller struct {
	cfg Config

	mu     sync.Mutex
	home   string
	rt     Runtime
	op     *Operation
	opDone chan struct{}
	last   *Operation
	// lastMaint is the last finished maintenance action; it is kept apart
	// from last so a failed maintenance action cannot fail the application.
	lastMaint *Operation
	// pending: an explicit Activate changed the activation record under a
	// running worker. The next Start/Restart rebinds to the new activation;
	// nothing restarts the worker implicitly.
	pending bool
	closed  bool
	// stopped is set when the operator stopped the worker (Stop or Restart's
	// stop, Close) and cleared by the next Start/Restart. It is what
	// distinguishes an operator Stop from a crash.
	stopped bool
	subs    map[int]chan struct{}
	nextID  int
}

// New creates a controller. It starts nothing.
func New(cfg Config) *Controller {
	if cfg.Setup == nil {
		cfg.Setup = func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error {
			return setup.RunObserved(home.Home{Root: root}, device, model, log, onPhase)
		}
	}
	if cfg.Installed == nil {
		cfg.Installed = func(root string) bool {
			_, _, _, err := home.Home{Root: root}.LoadActive()
			return err == nil
		}
	}
	cfg.Maintenance = cfg.Maintenance.withDefaults()
	return &Controller{cfg: cfg, home: cleanHome(cfg.Home), subs: map[int]chan struct{}{}}
}

func cleanHome(root string) string {
	if root == "" {
		return ""
	}
	if abs, err := filepath.Abs(root); err == nil {
		return filepath.Clean(abs)
	}
	return filepath.Clean(root)
}

// Snapshot projects the current application state from the authorities.
func (c *Controller) Snapshot() Snapshot {
	c.mu.Lock()
	root, rt, stopped := c.home, c.rt, c.stopped
	op, last, maint, pending := c.op.clone(), c.last.clone(), c.lastMaint.clone(), c.pending
	c.mu.Unlock()

	s := Snapshot{Home: root, Operation: op, Last: last, Maintenance: maint}
	running := false
	if rt != nil {
		running = rt.Running()
		st := rt.Status()
		s.Status = &st
	}
	kind := ""
	if op != nil {
		kind = op.Kind
	}
	var lastFail *Failure
	if last != nil {
		lastFail = last.Failure
	}
	installed := false
	if root != "" && kind != OpSetup && !running {
		installed = c.cfg.Installed(root)
	}
	s.State, s.Failure = project(root, kind, running, s.Status, lastFail, installed)
	s.RestartRequired = pending && running
	s.OperatorStopped = stopped && !running && kind == ""
	if !s.OperatorStopped {
		s.Recovery = recoveryOf(kind, running, s.Status)
	}
	return s
}

// recoveryOf derives the recovery view from the supervisor's status. Only an
// exit after the worker was READY counts (a restart in progress, a spent
// restart budget, or a post-start failure class); startup failures such as a
// missing device are deterministic and reported as plain failures. It is
// absent while an application action is in flight.
func recoveryOf(op string, running bool, st *server.Status) *Recovery {
	if op != "" || st == nil {
		return nil
	}
	w := st.Worker
	var cause *Failure
	if w.LastFailure != nil {
		cause = workerFailure(w.LastFailure)
	}
	switch {
	case running && w.State == worker.StateRestarting:
		return &Recovery{State: RecoveryRecovering, Restarts: w.Restarts, Cause: cause,
			Message: "The worker exited unexpectedly and is being restarted automatically."}
	case w.State == worker.StateFailed && (w.Restarts > 0 || unexpectedExit(w.LastFailure)):
		return &Recovery{State: RecoveryGaveUp, Restarts: w.Restarts, Cause: cause,
			Message: "The worker kept exiting unexpectedly and automatic restarts have stopped. " +
				"Read the failure, then choose Restart Runtime. The requested device is unchanged; there is no fallback to another device."}
	}
	return nil
}

func unexpectedExit(f *worker.FailureView) bool {
	if f == nil {
		return false
	}
	switch f.Class {
	case worker.ClassCrash, worker.ClassUnresponsive, worker.ClassProtocolError:
		return true
	}
	return false
}

// Subscribe returns a channel that receives a (coalesced) signal whenever a
// controller action begins, enters a setup phase, or finishes. Worker-driven
// transitions (warming, ready, crash) are owned by the supervisor and are
// observed by polling Snapshot. Call cancel to unsubscribe.
func (c *Controller) Subscribe() (<-chan struct{}, func()) {
	ch := make(chan struct{}, 1)
	c.mu.Lock()
	id := c.nextID
	c.nextID++
	c.subs[id] = ch
	c.mu.Unlock()
	return ch, func() {
		c.mu.Lock()
		delete(c.subs, id)
		c.mu.Unlock()
	}
}

// notify must be called with c.mu held; it never blocks.
func (c *Controller) notify() {
	for _, ch := range c.subs {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

// begin records a new in-flight action; c.mu must be held.
func (c *Controller) begin(kind, device, model string) *Operation {
	op := &Operation{Kind: kind, Device: device, Model: model, Started: time.Now()}
	c.op, c.opDone = op, make(chan struct{})
	c.notify()
	return op
}

// finish ends the in-flight action; c.mu must be held.
func (c *Controller) finish(op *Operation, f *Failure) {
	op.Finished, op.Failure = time.Now(), f
	if isMaintenance(op.Kind) {
		c.lastMaint = op
	} else {
		c.last = op
	}
	c.op = nil
	close(c.opDone)
	c.opDone = nil
	c.notify()
}

// SetHome selects the home (explicit input; "" unresolves it). It is
// rejected while an action is in flight or the runtime is running.
func (c *Controller) SetHome(root string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ErrClosed
	}
	if c.op != nil {
		return ErrBusy
	}
	if c.rt != nil && c.rt.Running() {
		return ErrRuntimeBusy
	}
	c.home, c.rt, c.last, c.lastMaint, c.stopped, c.pending = cleanHome(root), nil, nil, nil, false, false
	c.notify()
	return nil
}

// Setup starts setup/materialization asynchronously with explicit params. It
// returns once the action is accepted; progress and outcome are observed via
// Snapshot/Subscribe.
func (c *Controller) Setup(p SetupParams) error {
	return c.async(OpSetup, p, true, func(root string, log io.Writer, onPhase func(setup.Phase)) error {
		return c.cfg.Setup(root, p.Device, p.Model, log, onPhase)
	}, func() {
		// The activation may have changed: the next Start rebinds.
		c.rt = nil
	})
}

// async runs one long action in the background. needStopped refuses it while
// the worker runs (it may change what the worker was started from); after runs
// with c.mu held on success.
func (c *Controller) async(kind string, p SetupParams, needStopped bool, run func(root string, log io.Writer, onPhase func(setup.Phase)) error, after func()) error {
	if p.Device == "" {
		return fmt.Errorf("%s: device is required", kind)
	}
	log := p.Log
	if log == nil {
		log = io.Discard
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.admit(); err != nil {
		if errors.Is(err, ErrBusy) && c.op.Kind == kind {
			if kind == OpSetup {
				return ErrSetupRunning
			}
			return ErrOperationRunning
		}
		return err
	}
	if needStopped && c.rt != nil && c.rt.Running() {
		return ErrRuntimeBusy
	}
	op := c.begin(kind, p.Device, p.Model)
	root := c.home
	go func() {
		err := run(root, log, func(ph setup.Phase) {
			c.mu.Lock()
			op.Phase = string(ph)
			op.Phases = append(op.Phases, string(ph))
			c.notify()
			c.mu.Unlock()
		})
		c.mu.Lock()
		defer c.mu.Unlock()
		var f *Failure
		if err != nil {
			f = &Failure{Source: SourceSetup, Message: err.Error()}
		} else if after != nil {
			after()
		}
		c.finish(op, f)
	}()
	return nil
}

// Materialize downloads/builds and verifies the catalog choice (device,
// model) through the staged setup authority without activating it. It may run
// beside a running worker: it never changes the activation record.
func (c *Controller) Materialize(p SetupParams) error {
	return c.async(OpMaterialize, p, false, func(root string, log io.Writer, onPhase func(setup.Phase)) error {
		return c.cfg.Maintenance.Materialize(root, p.Device, p.Model, log, onPhase)
	}, nil)
}

// Repair rebuilds an artifact of the catalog choice that fails verification.
// It is refused while the worker runs, since the artifact may be in use.
func (c *Controller) Repair(p SetupParams) error {
	return c.async(OpRepair, p, true, func(root string, log io.Writer, onPhase func(setup.Phase)) error {
		return c.cfg.Maintenance.Repair(root, p.Device, p.Model, log, onPhase)
	}, func() { c.rt = nil })
}

// sync runs one short maintenance action to completion. guard runs with c.mu
// held before the action begins; after runs with c.mu held when it succeeded.
func (c *Controller) sync(kind, device, model, target string, guard func() error, run func(root string) error, after func()) error {
	c.mu.Lock()
	if err := c.admit(); err != nil {
		c.mu.Unlock()
		return err
	}
	if guard != nil {
		if err := guard(); err != nil {
			c.mu.Unlock()
			return err
		}
	}
	op := c.begin(kind, device, model)
	op.Target = target
	root := c.home
	c.mu.Unlock()

	err := run(root)

	c.mu.Lock()
	defer c.mu.Unlock()
	var f *Failure
	if err != nil {
		f = &Failure{Source: SourceSetup, Message: err.Error()}
	} else if after != nil {
		after()
	}
	c.finish(op, f)
	return err
}

// Activate explicitly makes the already materialized, verified choice
// (device, model) the active runtime and model. It never restarts the
// worker: when a worker is running, Snapshot.RestartRequired is set and the
// running worker keeps serving what it started with until Restart. Failure
// leaves the current activation untouched.
func (c *Controller) Activate(p SetupParams) error {
	if p.Device == "" {
		return errors.New("activate: device is required")
	}
	changed := false
	log := p.Log
	if log == nil {
		log = io.Discard
	}
	return c.sync(OpActivate, p.Device, p.Model, "", nil, func(root string) (err error) {
		changed, err = c.cfg.Maintenance.Activate(root, p.Device, p.Model, log)
		return err
	}, func() {
		if !changed {
			return
		}
		if c.rt != nil && c.rt.Running() {
			c.pending = true
			return
		}
		c.rt = nil
	})
}

// Verify re-verifies one catalog artifact ("runtime" or "model") offline.
func (c *Controller) Verify(kind, id string) error {
	return c.sync(OpVerify, "", "", kind+" "+id, nil, func(root string) error {
		return c.cfg.Maintenance.Verify(root, kind, id)
	}, nil)
}

// Remove deletes one unused catalog artifact. The setup authority refuses
// the active runtime/model; the controller additionally refuses while a
// restart is required, because the running worker then still uses the
// previously active artifacts.
func (c *Controller) Remove(kind, id string) error {
	return c.sync(OpRemove, "", "", kind+" "+id, func() error {
		if c.pending && c.rt != nil && c.rt.Running() {
			return ErrRestartRequired
		}
		return nil
	}, func(root string) error {
		return c.cfg.Maintenance.Remove(root, kind, id)
	}, nil)
}

// Inventory reads the typed inventory of the selected home. It is read-only;
// verify additionally runs the full (expensive) verification.
func (c *Controller) Inventory(verify bool) (setup.Inventory, error) {
	c.mu.Lock()
	root := c.home
	c.mu.Unlock()
	if root == "" {
		return setup.Inventory{}, ErrUnconfigured
	}
	return c.cfg.Maintenance.Inspect(root, verify), nil
}

func isMaintenance(kind string) bool {
	switch kind {
	case OpMaterialize, OpRepair, OpActivate, OpVerify, OpRemove:
		return true
	}
	return false
}

// admit checks the preconditions shared by every action; c.mu must be held.
func (c *Controller) admit() error {
	switch {
	case c.closed:
		return ErrClosed
	case c.home == "":
		return ErrUnconfigured
	case c.op != nil:
		return ErrBusy
	}
	return nil
}

// Start starts the resident worker. It is a no-op (nil) when a start or
// restart is already in flight or the worker is running. The worker's own
// startup is observed through Snapshot (starting, warming, ready, failed).
func (c *Controller) Start() error { return c.run(OpStart) }

// Restart stops the worker (if running) and starts it again.
func (c *Controller) Restart() error { return c.run(OpRestart) }

func (c *Controller) run(kind string) error {
	c.mu.Lock()
	if c.op != nil && (c.op.Kind == OpStart || c.op.Kind == OpRestart) && !c.closed {
		c.mu.Unlock()
		return nil
	}
	if err := c.admit(); err != nil {
		c.mu.Unlock()
		return err
	}
	rt := c.rt
	if kind == OpStart && rt != nil && rt.Running() {
		c.mu.Unlock()
		return nil
	}
	var stale Runtime
	if c.pending && rt != nil {
		// An explicit Activate changed the activation under this binding:
		// the operator's Restart (or a Start after the worker went down)
		// stops the old binding and binds the new activation.
		stale, rt = rt, nil
	} else if rt != nil && !rt.Running() && rt.Status().Worker.State == worker.StateFailed {
		// The supervisor gave up (its restart budget is spent or a start
		// failed). An explicit operator Start/Restart begins from a fresh
		// supervisor, so the bounded budget resets by operator action and
		// only then. The same active runtime (and so the same requested
		// device) is reopened; nothing is set up or changed.
		rt = nil
	}
	if rt == nil && !c.cfg.Installed(c.home) {
		c.mu.Unlock()
		return ErrNotInstalled
	}
	op := c.begin(kind, "", "")
	c.stopped = false
	root := c.home
	c.mu.Unlock()

	var f *Failure
	if stale != nil && stale.Running() {
		stale.Stop()
	}
	if rt == nil {
		var err error
		if rt, err = c.cfg.Open(root); err != nil {
			f = &Failure{Source: SourceRuntime, Message: err.Error()}
		}
	}
	if f == nil {
		if kind == OpRestart {
			rt.Restart()
		} else {
			rt.Start()
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if f == nil {
		c.rt = rt
		c.pending = false
	}
	c.finish(op, f)
	if f != nil {
		return f
	}
	return nil
}

// Stop stops the resident worker and waits for it. It is a no-op when the
// worker is not running or a stop is already in flight. A worker that the
// supervisor gave up on keeps reporting failed (cause retained) until the
// next Start or Restart.
func (c *Controller) Stop() error {
	c.mu.Lock()
	if c.op != nil && c.op.Kind == OpStop && !c.closed {
		c.mu.Unlock()
		return nil
	}
	if err := c.admit(); err != nil {
		c.mu.Unlock()
		return err
	}
	rt := c.rt
	if rt == nil || !rt.Running() {
		c.mu.Unlock()
		return nil
	}
	op := c.begin(OpStop, "", "")
	c.mu.Unlock()
	rt.Stop()
	c.mu.Lock()
	c.stopped = true
	c.finish(op, nil)
	c.mu.Unlock()
	return nil
}

// Close stops accepting actions, waits for the in-flight action (setup is
// not interrupted), then stops the runtime. It returns an error naming what
// was still in progress if ctx ends first; it never leaves a second worker.
func (c *Controller) Close(ctx context.Context) error {
	c.mu.Lock()
	c.closed = true
	c.stopped = true
	done, op := c.opDone, c.op.clone()
	c.mu.Unlock()
	if done != nil {
		select {
		case <-done:
		case <-ctx.Done():
			return fmt.Errorf("shutdown: %s still in progress: %w", op.Kind, ctx.Err())
		}
	}
	c.mu.Lock()
	rt := c.rt
	c.mu.Unlock()
	if rt == nil {
		return nil
	}
	stopped := make(chan struct{})
	go func() { rt.Stop(); close(stopped) }()
	select {
	case <-stopped:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("shutdown: runtime still stopping: %w", ctx.Err())
	}
}
