package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
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
	// The System One model forge: building a variant and certifying it.
	OpOptimize = "optimize"
	OpCertify  = "certify"
	// Forge readiness: the preflight before expensive work and the probe of a
	// persisted variant. Neither certifies, activates or changes a resident.
	OpPreflight = "preflight"
	OpProbe     = "probe"
)

// Rejections of the maintenance actions.
var (
	ErrOperationRunning = errors.New("this action is already running")
	ErrRestartRequired  = errors.New("the active runtime/model changed; restart the runtime first")
)

// SetupFunc materializes and activates the runtime under root. It must report
// each real phase boundary it enters, and the progress of the work inside it,
// to obs (setup.RunObserved does).
type SetupFunc func(root, device, model string, log io.Writer, obs *setup.Observer) error

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
	// Residents reports the desired additional resident catalog models
	// (the settings authority's selection); nil means none are ever
	// requested. The controller never stores it: it only compares it with
	// the members of the bound runtime, so a change applies on the next
	// explicit Start/Restart and never mutates running workers.
	Residents func() []string
}

// Maintenance are the explicit model/runtime operations of the setup/home
// authority, addressed by home root.
type Maintenance struct {
	Inspect     func(root string, verify bool) setup.Inventory
	Materialize func(root, device, model string, log io.Writer, obs *setup.Observer) error
	Repair      func(root, device, model string, log io.Writer, obs *setup.Observer) error
	Activate    func(root, device, model string, log io.Writer, obs *setup.Observer) (changed bool, err error)
	Verify      func(root, kind, id string, obs *setup.Observer) error
	Remove      func(root, kind, id string, obs *setup.Observer) error
	// ActivateVariant is Activate for a source model plus one of its variants.
	ActivateVariant func(root, device, model, variant string, experimental bool, log io.Writer, obs *setup.Observer) (changed bool, err error)
	// Optimize builds a variant of a catalog model with a canonical recipe.
	Optimize func(ctx context.Context, root, model, recipe string, log io.Writer, obs *setup.Observer) error
	// Certify compares a reference run and a variant run and records the
	// certification.
	Certify func(root string, p CertifyParams, log io.Writer, obs *setup.Observer) error
	// Preflight inspects what Hachidori can know before an expensive Forge
	// operation and records the report.
	Preflight func(ctx context.Context, root string, p PreflightParams, obs *setup.Observer) (setup.PreflightReport, error)
	// Probe loads a persisted variant in an isolated worker and asks one typed
	// decision.
	Probe func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error)
}

// PreflightParams are the explicit inputs of a preflight: the operation it
// gates and its catalog identities.
type PreflightParams struct {
	Kind           string // setup.PreflightMaterialize | Optimize | Probe | Certify
	Model          string
	Recipe         string
	Variant        string
	Device         string
	Quick          bool
	ReferenceDType string
}

// CertifyParams are the explicit inputs of a certification: the variant, the
// two resident run files and, optionally, a policy file (the built-in profile
// when empty). Paths are read, never written.
type CertifyParams struct {
	Variant   string
	Reference string
	Candidate string
	Policy    string
}

func (m Maintenance) withDefaults() Maintenance {
	if m.Inspect == nil {
		m.Inspect = func(root string, verify bool) setup.Inventory { return setup.Inspect(home.Home{Root: root}, verify) }
	}
	if m.Materialize == nil {
		m.Materialize = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
			return setup.Materialize(home.Home{Root: root}, device, model, log, obs)
		}
	}
	if m.Repair == nil {
		m.Repair = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
			return setup.Repair(home.Home{Root: root}, device, model, log, obs)
		}
	}
	if m.Activate == nil {
		m.Activate = func(root, device, model string, log io.Writer, obs *setup.Observer) (bool, error) {
			return setup.Activate(home.Home{Root: root}, device, model, log, obs)
		}
	}
	if m.Verify == nil {
		m.Verify = func(root, kind, id string, obs *setup.Observer) error {
			return setup.Verify(home.Home{Root: root}, kind, id, obs)
		}
	}
	if m.Remove == nil {
		m.Remove = func(root, kind, id string, obs *setup.Observer) error {
			return setup.Remove(home.Home{Root: root}, kind, id, obs)
		}
	}
	if m.ActivateVariant == nil {
		m.ActivateVariant = func(root, device, model, variant string, experimental bool, log io.Writer, obs *setup.Observer) (bool, error) {
			return setup.ActivateTarget(home.Home{Root: root}, device, model, setup.ActivateOptions{Variant: variant, AllowUncertified: experimental}, log, obs)
		}
	}
	if m.Optimize == nil {
		m.Optimize = func(ctx context.Context, root, model, recipe string, log io.Writer, obs *setup.Observer) error {
			_, err := optimize.Build(ctx, home.Home{Root: root}, optimize.Request{Model: model, Recipe: recipe}, optimize.Deps{}, log, obs)
			return err
		}
	}
	if m.Certify == nil {
		m.Certify = certify
	}
	if m.Preflight == nil {
		m.Preflight = RunPreflight
	}
	if m.Probe == nil {
		m.Probe = func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
			return Probe(ctx, home.Home{Root: root}, p, ProbeDeps{}, log, obs)
		}
	}
	return m
}

// SetupParams are the explicit parameters of a setup action.
type SetupParams struct {
	Device string    // inference device, as for `hachidori setup --device`
	Model  string    // catalog model ID; empty selects setup.DefaultModel
	Log    io.Writer // setup's human-readable log; nil discards it
}

// Operation is one application action. A setup or maintenance action reports
// the phases it has actually entered and the progress of the work inside the
// current one. Progress carries a total only when the work has a measurable
// one (a download with a known size, a file being hashed); otherwise it names
// the step and is indeterminate, and no percentage is ever derived.
type Operation struct {
	// ID identifies this operation: it names it in its diagnostic.
	ID       string          `json:"id,omitempty"`
	Kind     string          `json:"kind"`
	Device   string          `json:"device,omitempty"`
	Model    string          `json:"model,omitempty"`
	Target   string          `json:"target,omitempty"`   // verify/remove: "<kind> <id>"
	Plan     []string        `json:"plan,omitempty"`     // the phases the action goes through, in order
	Phase    string          `json:"phase,omitempty"`    // the phase currently/last entered
	Phases   []string        `json:"phases,omitempty"`   // every phase entered, in order
	Progress *setup.Progress `json:"progress,omitempty"` // the current step within Phase
	Started  time.Time       `json:"started"`
	Finished time.Time       `json:"finished,omitzero"`
	Failure  *Failure        `json:"failure,omitempty"`
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
	c.Plan = append([]string(nil), o.Plan...)
	if o.Progress != nil {
		p := *o.Progress
		c.Progress = &p
	}
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
	// Checks is the outcome of the last explicit Verify of each artifact,
	// keyed "<kind> <id>".
	Checks map[string]Check `json:"checks,omitempty"`
	// RestartRequired is set while the running worker was started from a
	// runtime/model that is no longer the active one: an explicit Activate
	// changed the activation record and nothing has restarted the worker.
	// It is also set while the desired additional residents differ from the
	// members of the running set (ResidencyChanged); Restart applies both.
	RestartRequired bool `json:"restart_required,omitempty"`
	// ResidencyChanged is set while the desired resident selection differs
	// from the residents the running runtime was started with.
	ResidencyChanged bool           `json:"residency_changed,omitempty"`
	Status           *server.Status `json:"status,omitempty"` // the /v1/status document, when a runtime is bound
	// Recovery is set while an unexpected worker exit is being recovered or
	// recovery gave up. OperatorStopped is set while the worker is down
	// because the operator chose Stop (or Quit), so the two are never
	// confused.
	Recovery        *Recovery `json:"recovery,omitempty"`
	OperatorStopped bool      `json:"operator_stopped,omitempty"`
	// Residents is the status of every member of a bound resident set,
	// default first (nil for a single worker). State, Failure and Recovery
	// are projected over all of them: the application is Ready only when
	// every resident that is meant to be up is READY, and a resident's
	// failure names its model and provider. Status stays the default
	// resident's document.
	Residents []ResidentStatus `json:"residents,omitempty"`
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
	// checks is the outcome of the last explicit Verify per "<kind> <id>".
	checks map[string]Check
	closed bool
	// stopped is set when the operator stopped the worker (Stop or Restart's
	// stop, Close) and cleared by the next Start/Restart. It is what
	// distinguishes an operator Stop from a crash.
	stopped bool
	subs    map[int]chan struct{}
	nextID  int
	seq     int // operations begun; part of an operation's ID
}

// New creates a controller. It starts nothing.
func New(cfg Config) *Controller {
	if cfg.Setup == nil {
		cfg.Setup = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
			return setup.RunObserved(home.Home{Root: root}, device, model, log, obs)
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
	var checks map[string]Check
	if len(c.checks) > 0 {
		checks = make(map[string]Check, len(c.checks))
		for k, v := range c.checks {
			checks[k] = v
		}
	}
	c.mu.Unlock()

	s := Snapshot{Home: root, Operation: op, Last: last, Maintenance: maint, Checks: checks}
	running := false
	proj := s.Status
	var culprit *ResidentStatus
	drift := false
	if rt != nil {
		running = rt.Running()
		st := rt.Status()
		s.Status, proj = &st, &st
		drift = c.residencyDrift(st.Runtime.ModelID, rt)
		if rr, ok := rt.(ResidentRuntime); ok {
			// The status document already carries the residents of this
			// same read; asking the set again would be a second view.
			if s.Residents = st.Residents; s.Residents == nil {
				s.Residents = rr.ResidentStatuses()
			}
			agg, c := aggregateResidents(st, s.Residents)
			proj, culprit = &agg, c
		}
	}
	kind := ""
	if op != nil {
		kind = op.Kind
	}
	// An action on one resident is not an action on the application: the
	// other residents keep serving, so their state is not masked by it.
	projKind := kind
	if op != nil && op.Model != "" && s.Residents != nil {
		projKind = ""
	}
	var lastFail *Failure
	if last != nil {
		lastFail = last.Failure
	}
	installed := false
	if root != "" && kind != OpSetup && !running {
		installed = c.cfg.Installed(root)
	}
	s.State, s.Failure = project(root, projKind, running, proj, lastFail, installed)
	s.RestartRequired = (pending || drift) && running
	s.ResidencyChanged = drift && running
	s.OperatorStopped = stopped && !running && kind == ""
	if !s.OperatorStopped {
		s.Recovery = recoveryOf(projKind, running, proj)
	}
	if culprit != nil {
		if s.Failure != nil && s.Failure.Source == SourceWorker {
			s.Failure.Model, s.Failure.Provider = culprit.Model, culprit.Provider
		}
		if s.Recovery != nil && s.Recovery.Cause != nil {
			s.Recovery.Cause.Model, s.Recovery.Cause.Provider = culprit.Model, culprit.Provider
		}
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
		cause = workerFailure(w.LastFailure, w.Phase)
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
	c.seq++
	op := &Operation{ID: fmt.Sprintf("op-%d-%d", time.Now().UnixMilli(), c.seq), Kind: kind, Device: device, Model: model, Started: time.Now()}
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
	c.home, c.rt, c.last, c.lastMaint, c.stopped, c.pending, c.checks = cleanHome(root), nil, nil, nil, false, false, nil
	c.notify()
	return nil
}

// action describes one asynchronous setup or maintenance action.
type action struct {
	kind        string
	device      string
	model       string
	target      string
	needDevice  bool         // the action needs an explicit device
	needStopped bool         // refused while the worker runs (it may change what the worker started from)
	guard       func() error // c.mu held; a rejection starts nothing
	run         func(root string, log io.Writer, obs *setup.Observer) error
	// after runs with c.mu held once run returned, with its outcome.
	after func(err error)
	// forge, when set, makes a failure of this action leave a Forge
	// diagnostic. enrich completes it with what only the run knows (the
	// probe's record) when the failure is known.
	forge  *ForgeFailure
	enrich func(*ForgeFailure)
}

// Setup starts setup/materialization asynchronously with explicit params. It
// returns once the action is accepted; progress and outcome are observed via
// Snapshot/Subscribe.
func (c *Controller) Setup(p SetupParams) error {
	return c.async(p, action{kind: OpSetup, device: p.Device, model: p.Model, needDevice: true, needStopped: true,
		forge: forgeOf(OpSetup, p),
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Setup(root, p.Device, p.Model, log, obs)
		},
		after: func(err error) {
			if err == nil {
				// The activation may have changed: the next Start rebinds.
				c.rt = nil
			}
		}})
}

// async runs one long action in the background. It is the one path of every
// setup and maintenance action, so each of them is admitted the same way, runs
// off the caller's goroutine, and reports its phases and progress.
func (c *Controller) async(p SetupParams, a action) error {
	if a.needDevice && p.Device == "" {
		return fmt.Errorf("%s: device is required", a.kind)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.admit(); err != nil {
		if errors.Is(err, ErrBusy) && c.op.Kind == a.kind {
			if a.kind == OpSetup {
				return ErrSetupRunning
			}
			return ErrOperationRunning
		}
		return err
	}
	if a.needStopped && c.rt != nil && c.rt.Running() {
		return ErrRuntimeBusy
	}
	if a.guard != nil {
		if err := a.guard(); err != nil {
			return err
		}
	}
	op := c.begin(a.kind, a.device, a.model)
	op.Target = a.target
	op.Plan = plan(a.kind, a.target)
	root := c.home
	// Without a caller's log the action's output goes to the home's setup
	// log, where its failure can be diagnosed.
	log, closeLog := p.Log, func() {}
	if log == nil {
		log, closeLog = openSetupLog(root, a)
	}
	obs := &setup.Observer{
		OnPhase: func(ph setup.Phase) {
			c.mu.Lock()
			op.Phase = string(ph)
			op.Phases = append(op.Phases, string(ph))
			op.Progress = nil // a step belongs to the phase it was reported in
			c.notify()
			c.mu.Unlock()
		},
		OnProgress: func(pr setup.Progress) {
			c.mu.Lock()
			op.Progress = &pr
			c.notify()
			c.mu.Unlock()
		},
	}
	go func() {
		err := a.run(root, log, obs)
		closeLog()
		// A failed expensive Forge operation leaves a bounded, redacted
		// diagnostic. Recording it can fail in any way without changing the
		// outcome: the operation's own error is what is reported.
		diag := ""
		if err != nil && a.forge != nil {
			ff := *a.forge
			c.mu.Lock()
			ff.OperationID, ff.Started, ff.Finished, ff.Phase = op.ID, op.Started, time.Now(), op.Phase
			if op.Progress != nil {
				ff.Step = string(op.Progress.Step)
			}
			c.mu.Unlock()
			if a.enrich != nil {
				a.enrich(&ff)
			}
			diag = DiagnosticID(WithForgeDiagnostic(root, ff, err))
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		var f *Failure
		if err != nil {
			f = &Failure{Source: SourceSetup, Phase: op.Phase, Message: err.Error(), Diagnostic: diag}
			if op.Progress != nil {
				f.Step = string(op.Progress.Step)
			}
		}
		if a.after != nil {
			a.after(err)
		}
		c.finish(op, f)
	}()
	return nil
}

// plan is the phases an action goes through, in order: what the operator can
// expect to see it enter. A phase is reported only when it is really entered,
// so an action that fails or has nothing to do stops short of its plan.
func plan(kind, target string) []string {
	p := func(ph ...setup.Phase) []string {
		out := make([]string, len(ph))
		for i, x := range ph {
			out[i] = string(x)
		}
		return out
	}
	switch kind {
	case OpSetup:
		return p(setup.PhasePreparing, setup.PhaseRuntime, setup.PhaseModel, setup.PhaseActivation)
	case OpMaterialize, OpRepair:
		return p(setup.PhasePreparing, setup.PhaseRuntime, setup.PhaseModel, setup.PhasePublish)
	case OpActivate:
		if strings.HasPrefix(target, setup.KindVariant+" ") {
			return p(setup.PhaseRuntime, setup.PhaseModel, setup.PhaseVariant, setup.PhaseActivation)
		}
		return p(setup.PhaseRuntime, setup.PhaseModel, setup.PhaseActivation)
	case OpVerify, OpRemove:
		switch {
		case strings.HasPrefix(target, setup.KindRuntime+" "):
			return p(setup.PhaseRuntime)
		case strings.HasPrefix(target, setup.KindVariant+" "):
			return p(setup.PhaseVariant)
		}
		return p(setup.PhaseModel)
	case OpOptimize:
		return p(setup.PhaseModel, setup.PhasePreflight, setup.PhasePreparing, setup.PhaseRuntime, setup.PhaseStarting, setup.PhaseLoadingSource,
			setup.PhaseResolving, setup.PhaseQuantizing, setup.PhaseSerializing, setup.PhaseVerifying, setup.PhasePublish)
	case OpCertify:
		return p(setup.PhaseLoadingRuns, setup.PhaseComparing, setup.PhaseRecording)
	case OpPreflight:
		return p(setup.PhasePreflight)
	case OpProbe:
		return p(setup.PhasePreflight, setup.PhaseProbing)
	}
	return nil
}

// SetupLogPath is the log of setup and maintenance actions under a home.
func SetupLogPath(root string) string { return filepath.Join(root, "logs", "setup.log") }

// openSetupLog appends to the home's setup log, heading the entry with the
// action. When the log cannot be opened the output is discarded: logging never
// fails an action.
func openSetupLog(root string, a action) (io.Writer, func()) {
	return OpenSetupLog(root, a.kind, a.device, a.model, a.target)
}

// OpenSetupLog appends to the home's setup log, heading the entry with the
// operation. The CLI's Forge commands write the same section a controller
// action would, so a failure's diagnostic can quote its tail either way.
func OpenSetupLog(root, kind, device, model, target string) (io.Writer, func()) {
	if root == "" || os.MkdirAll(filepath.Join(root, "logs"), 0o755) != nil {
		return io.Discard, func() {}
	}
	f, err := os.OpenFile(SetupLogPath(root), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return io.Discard, func() {}
	}
	fmt.Fprintf(f, "== %s %s %s %s %s\n", time.Now().Format(time.RFC3339), kind, device, model, target)
	return f, func() { f.Close() }
}

// Materialize downloads/builds and verifies the catalog choice (device,
// model) through the staged setup authority without activating it. It may run
// beside a running worker: it never changes the activation record.
func (c *Controller) Materialize(p SetupParams) error {
	return c.async(p, action{kind: OpMaterialize, device: p.Device, model: p.Model, needDevice: true,
		forge: forgeOf(OpMaterialize, p),
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Maintenance.Materialize(root, p.Device, p.Model, log, obs)
		}})
}

// Repair rebuilds an artifact of the catalog choice that fails verification.
// It is refused while the worker runs, since the artifact may be in use.
func (c *Controller) Repair(p SetupParams) error {
	return c.async(p, action{kind: OpRepair, device: p.Device, model: p.Model, needDevice: true, needStopped: true,
		forge: forgeOf(OpRepair, p),
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Maintenance.Repair(root, p.Device, p.Model, log, obs)
		},
		after: func(err error) {
			if err == nil {
				c.rt = nil
			}
		}})
}

// Activate explicitly makes the already materialized, verified choice
// (device, model) the active runtime and model. It never restarts the
// worker: when a worker is running, Snapshot.RestartRequired is set and the
// running worker keeps serving what it started with until Restart. Failure
// leaves the current activation untouched. Verifying both artifacts can take
// a while, so it runs like the other maintenance actions: accepted here, its
// phases observed and its outcome read from Snapshot.Maintenance.
func (c *Controller) Activate(p SetupParams) error {
	changed := false
	return c.async(p, action{kind: OpActivate, device: p.Device, model: p.Model, needDevice: true,
		run: func(root string, log io.Writer, obs *setup.Observer) (err error) {
			changed, err = c.cfg.Maintenance.Activate(root, p.Device, p.Model, log, obs)
			return err
		},
		after: func(err error) {
			if err != nil || !changed {
				return
			}
			if c.rt != nil && c.rt.Running() {
				c.pending = true
				return
			}
			c.rt = nil
		}})
}

// ActivateVariant explicitly makes the already materialized choice (device,
// model) with one of its variants the active execution target. It is Activate
// for a source plus a variant: it validates the variant's manifest, source
// link, artifacts and certification before the activation record changes, never
// falls back to the source, and never restarts the worker. experimental
// requests the operator-only launch of a variant that has no certification
// record; it is reported as experimental/uncertified and never implicit.
func (c *Controller) ActivateVariant(p SetupParams, variant string, experimental bool) error {
	if variant == "" {
		return errors.New("activate variant: a variant ID is required")
	}
	changed := false
	return c.async(p, action{kind: OpActivate, device: p.Device, model: p.Model, target: setup.KindVariant + " " + variant, needDevice: true,
		run: func(root string, log io.Writer, obs *setup.Observer) (err error) {
			changed, err = c.cfg.Maintenance.ActivateVariant(root, p.Device, p.Model, variant, experimental, log, obs)
			return err
		},
		after: func(err error) {
			if err != nil || !changed {
				return
			}
			if c.rt != nil && c.rt.Running() {
				c.pending = true
				return
			}
			c.rt = nil
		}})
}

// Optimize builds a variant of a catalog model with a canonical recipe in the
// optimizer runtime. It reports the optimizer's real phases and never touches
// the active runtime, the active model or the source: a failed or interrupted
// build leaves nothing selectable. It may run beside a running worker.
func (c *Controller) Optimize(model, recipe string) error {
	return c.async(SetupParams{}, action{kind: OpOptimize, model: model, target: "recipe " + recipe,
		forge: &ForgeFailure{Kind: OpOptimize, Model: model, Recipe: recipe},
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Maintenance.Optimize(context.Background(), root, model, recipe, log, obs)
		}})
}

// Certify compares a reference run and a candidate run of a variant and
// records the certification, accepted or rejected, under the home. The
// verdict is the policy's; a refusal (mismatched identities, unaligned
// inputs) is a failed action.
func (c *Controller) Certify(p CertifyParams) error {
	return c.async(SetupParams{}, action{kind: OpCertify, target: setup.KindVariant + " " + p.Variant,
		forge: &ForgeFailure{Kind: OpCertify, Variant: p.Variant, Certify: &p},
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Maintenance.Certify(root, p, log, obs)
		}})
}

// forgeOf is the diagnostic context of a materialization-shaped action, or nil
// when the model is not a System One model: only Forge sources leave a
// diagnostic.
func forgeOf(kind string, p SetupParams) *ForgeFailure {
	model := p.Model
	if model == "" {
		model = setup.DefaultModel
	}
	if !IsForgeOperation(kind, model) {
		return nil
	}
	return &ForgeFailure{Kind: kind, Model: model, Device: p.Device}
}

// Preflight inspects what Hachidori can know before the expensive Forge
// operation p.Kind and records the report (LatestPreflight, Forge). A blocked
// report is the preflight's result, not its failure. It changes nothing: no
// download, no activation, no resident.
func (c *Controller) Preflight(p PreflightParams) error {
	target := p.Kind
	switch {
	case p.Variant != "":
		target += " " + setup.KindVariant + " " + p.Variant
	case p.Recipe != "":
		target += " recipe " + p.Recipe
	}
	return c.async(SetupParams{}, action{kind: OpPreflight, model: p.Model, device: p.Device, target: target,
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			_, err := c.cfg.Maintenance.Preflight(context.Background(), root, p, obs)
			return err
		}})
}

// Probe loads the persisted variant p.Variant on p.Device in an isolated worker
// and asks one typed decision (see Probe). It is not a certification and not
// an activation: it never touches the bound runtime, the resident set, the
// activation record or a certification record, and it may run beside a running
// worker. Its record is Forge().Probes; a failure leaves a diagnostic.
func (c *Controller) Probe(p ProbeParams) error {
	var rec ProbeRecord
	return c.async(SetupParams{Device: p.Device}, action{kind: OpProbe, device: p.Device, target: setup.KindVariant + " " + p.Variant, needDevice: true,
		forge:  &ForgeFailure{Kind: OpProbe, Variant: p.Variant, Device: p.Device},
		enrich: func(f *ForgeFailure) { f.Probe = &rec },
		run: func(root string, log io.Writer, obs *setup.Observer) (err error) {
			rec, err = c.cfg.Maintenance.Probe(context.Background(), root, p, log, obs)
			return err
		}})
}

// RunPreflight is the default Preflight: the optimize authority's checks over
// the home, recorded as the latest report of the target. The CLI runs the very
// same function the controller does.
func RunPreflight(ctx context.Context, root string, p PreflightParams, obs *setup.Observer) (setup.PreflightReport, error) {
	h := home.Home{Root: root}
	dtype := p.ReferenceDType
	if dtype == "" && p.Kind == setup.PreflightCertify {
		dtype = os.Getenv(server.EnvClefDType)
	}
	rep := optimize.Preflight(ctx, h, optimize.PreflightRequest{Kind: p.Kind, Model: p.Model, Recipe: p.Recipe, Variant: p.Variant, Device: p.Device,
		Quick: p.Quick, ReferenceDType: dtype}, optimize.PreflightDeps{}, obs)
	if err := SavePreflight(h, rep); err != nil {
		return rep, fmt.Errorf("the preflight ran but its report could not be saved: %w", err)
	}
	return rep, nil
}

// certify is the default Certify: the eval authority over the setup/home
// authority's variant and source.
func certify(root string, p CertifyParams, log io.Writer, obs *setup.Observer) error {
	h := home.Home{Root: root}
	obs.Phase(setup.PhaseLoadingRuns)
	src, v, err := setup.FindVariant(h, p.Variant)
	if err != nil {
		return err
	}
	ref, err := eval.LoadResidentRun(p.Reference)
	if err != nil {
		return err
	}
	cand, err := eval.LoadResidentRun(p.Candidate)
	if err != nil {
		return err
	}
	policy := eval.DefaultPolicy()
	if p.Policy != "" {
		if policy, err = eval.LoadPolicy(p.Policy); err != nil {
			return err
		}
	}
	obs.Phase(setup.PhaseComparing)
	cert, err := eval.Certify(eval.CertifyInput{Source: src, Variant: v, Reference: ref, Candidate: cand, Policy: policy})
	if err != nil {
		return err
	}
	obs.Phase(setup.PhaseRecording)
	rec, err := eval.SaveCertification(h, cert)
	if err != nil {
		return err
	}
	eval.CertificationSummary(log, cert)
	fmt.Fprintf(log, "certification recorded: %s verdict %s\n", rec.Report, rec.Verdict)
	return nil
}

// Check is the outcome of the last explicit Verify of one artifact. It is
// kept for as long as the controller lives, so it survives the binding of a
// new runtime.
type Check struct {
	OK      bool      `json:"ok"`
	Message string    `json:"message,omitempty"`
	Time    time.Time `json:"time"`
}

// Verify re-verifies one catalog artifact ("runtime" or "model") offline. It
// is accepted here and observed like the other maintenance actions; its
// outcome is Snapshot.Checks["<kind> <id>"].
func (c *Controller) Verify(kind, id string) error {
	target := kind + " " + id
	return c.async(SetupParams{}, action{kind: OpVerify, target: target,
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Maintenance.Verify(root, kind, id, obs)
		},
		after: func(err error) {
			ck := Check{OK: err == nil, Time: time.Now()}
			if err != nil {
				ck.Message = err.Error()
			}
			if c.checks == nil {
				c.checks = map[string]Check{}
			}
			c.checks[target] = ck
		}})
}

// Remove deletes one unused catalog artifact. The setup authority refuses
// the active runtime and model; the controller additionally refuses while a
// restart is required, because the running worker then still uses the
// previously active artifacts, and the model of any additional resident of
// the running set, which the setup authority does not know about.
func (c *Controller) Remove(kind, id string) error {
	target := kind + " " + id
	return c.async(SetupParams{}, action{kind: OpRemove, target: target,
		guard: func() error {
			if c.rt == nil || !c.rt.Running() {
				return nil
			}
			if c.pending {
				return ErrRestartRequired
			}
			if kind == setup.KindModel && slices.Contains(boundExtras(c.rt), id) {
				return fmt.Errorf("model %s is a resident of the running runtime: %w", id, ErrRuntimeBusy)
			}
			return nil
		},
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			return c.cfg.Maintenance.Remove(root, kind, id, obs)
		},
		after: func(err error) {
			if err == nil {
				delete(c.checks, target) // what it described no longer exists
			}
		}})
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
	case OpMaterialize, OpRepair, OpActivate, OpVerify, OpRemove, OpOptimize, OpCertify, OpPreflight, OpProbe:
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
	if rt != nil && (c.pending || c.residencyDrift(rt.Status().Runtime.ModelID, rt)) {
		// An explicit Activate changed the activation under this binding,
		// or the desired resident selection no longer matches its members:
		// the operator's Restart (or a Start after the worker went down)
		// stops the old binding and binds the new activation and residents.
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
			f = &Failure{Source: SourceRuntime, Phase: PhasePreflight, Message: err.Error()}
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

// StartResident starts one resident of the bound resident set; the others are
// untouched. It is a no-op (nil) when that resident is already running.
func (c *Controller) StartResident(model string) error { return c.residentAction(OpStart, model) }

// StopResident stops one resident and waits for it. The others keep serving;
// nothing restarts the stopped one until StartResident or Restart.
func (c *Controller) StopResident(model string) error { return c.residentAction(OpStop, model) }

// RestartResident restarts one resident (a resident the supervisor gave up on
// begins with a fresh restart budget). The others are untouched.
func (c *Controller) RestartResident(model string) error { return c.residentAction(OpRestart, model) }

// residentAction is the one path of the per-resident actions. It is admitted
// like Start/Stop/Restart (one action in flight, ErrBusy otherwise) but acts
// on the bound set only: it never binds a runtime, never rebinds after an
// Activate (ErrRestartRequired: restart the whole runtime), and never changes
// another resident.
func (c *Controller) residentAction(kind, model string) error {
	c.mu.Lock()
	if err := c.admit(); err != nil {
		c.mu.Unlock()
		return err
	}
	rr, ok := c.rt.(ResidentRuntime)
	if !ok {
		c.mu.Unlock()
		return ErrNoResidents
	}
	if c.pending {
		c.mu.Unlock()
		return ErrRestartRequired
	}
	known := false
	for _, r := range rr.ResidentStatuses() {
		known = known || r.Model == model
	}
	if !known {
		c.mu.Unlock()
		return fmt.Errorf("%w: %q", ErrUnknownResident, model)
	}
	op := c.begin(kind, "", model)
	c.mu.Unlock()

	var err error
	switch kind {
	case OpStart:
		_, err = rr.StartResident(model)
	case OpStop:
		err = rr.StopResident(model)
	case OpRestart:
		err = rr.RestartResident(model)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case kind != OpStop:
		c.stopped = false
	case !rr.Running():
		c.stopped = true // the last resident went down by the operator's hand
	}
	c.finish(op, nil)
	return err
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
