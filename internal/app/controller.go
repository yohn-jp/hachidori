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

// Snapshot is the application view at one instant.
type Snapshot struct {
	State     State          `json:"state"`
	Home      string         `json:"home,omitempty"`
	Operation *Operation     `json:"operation,omitempty"`      // in flight
	Last      *Operation     `json:"last_operation,omitempty"` // most recently finished
	Failure   *Failure       `json:"failure,omitempty"`        // set iff State is Failed
	Status    *server.Status `json:"status,omitempty"`         // the /v1/status document, when a runtime is bound
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
	closed bool
	subs   map[int]chan struct{}
	nextID int
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
	root, rt := c.home, c.rt
	op, last := c.op.clone(), c.last.clone()
	c.mu.Unlock()

	s := Snapshot{Home: root, Operation: op, Last: last}
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
	return s
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
	c.last = op
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
	c.home, c.rt, c.last = cleanHome(root), nil, nil
	c.notify()
	return nil
}

// Setup starts setup/materialization asynchronously with explicit params. It
// returns once the action is accepted; progress and outcome are observed via
// Snapshot/Subscribe.
func (c *Controller) Setup(p SetupParams) error {
	if p.Device == "" {
		return errors.New("setup: device is required")
	}
	log := p.Log
	if log == nil {
		log = io.Discard
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.admit(); err != nil {
		if errors.Is(err, ErrBusy) && c.op.Kind == OpSetup {
			return ErrSetupRunning
		}
		return err
	}
	if c.rt != nil && c.rt.Running() {
		return ErrRuntimeBusy
	}
	op := c.begin(OpSetup, p.Device, p.Model)
	root := c.home
	go func() {
		err := c.cfg.Setup(root, p.Device, p.Model, log, func(ph setup.Phase) {
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
		} else {
			// The activation may have changed: the next Start rebinds.
			c.rt = nil
		}
		c.finish(op, f)
	}()
	return nil
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
	if rt == nil && !c.cfg.Installed(c.home) {
		c.mu.Unlock()
		return ErrNotInstalled
	}
	op := c.begin(kind, "", "")
	root := c.home
	c.mu.Unlock()

	var f *Failure
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
