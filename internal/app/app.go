// Package app is the application-level lifecycle contract that a desktop
// surface consumes: one small, deterministic application state plus the
// actions a user can take (set up, start, stop, restart).
//
// It keeps no runtime state of its own. The application state is a
// projection over the existing authorities:
//
//   - setup/materialization is internal/setup (setup.RunObserved);
//   - the resident worker is driven through worker.Lifecycle and observed
//     through the /v1/status document (server.StatusBody), whose worker
//     section is worker.Supervisor's snapshot;
//   - "installed" is the activation record read by home.Home.LoadActive.
//
// The controller only adds what none of those own: which application action
// is in flight, the last action's outcome, and the rules that keep repeated
// or concurrent UI actions from creating duplicate setup or runtime
// ownership. Home discovery is not part of this package: the home is an
// explicit input ("" means unresolved).
package app

import (
	"errors"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// State is the application state. The set is closed and small; details stay
// in the authorities (Snapshot.Status carries the /v1/status document).
type State string

const (
	// Unconfigured: no home has been selected or resolved.
	Unconfigured State = "unconfigured"
	// NotInstalled: a home is known but it has no valid activation record.
	NotInstalled State = "not_installed"
	// Installing: setup/materialization is running.
	Installing State = "installing"
	// Installed: a valid active runtime exists but no worker is running.
	Installed State = "installed"
	// Starting: the worker is spawning, importing, loading, or being
	// restarted by the supervisor after a crash.
	Starting State = "starting"
	// Warming: the worker reported its warmup phase (observable through the
	// supervisor's phase). Never synthesized.
	Warming State = "warming"
	// Ready: the supervised worker is initialized, warmed and serving.
	Ready State = "ready"
	// Stopping: an application Stop or Restart is shutting the worker down.
	Stopping State = "stopping"
	// Failed: the last application action failed, or the supervisor gave up
	// on the worker. Snapshot.Failure carries the structured cause.
	Failed State = "failed"
	// Paused: serving is intentionally down because model engineering owns
	// the accelerator (Snapshot.Paused names the owner). It is neither an
	// operator Stop nor a failure, and the controller brings serving back by
	// itself when the owning operation ends.
	Paused State = "paused"
)

// Operation kinds.
const (
	OpSetup   = "setup"
	OpStart   = "start"
	OpStop    = "stop"
	OpRestart = "restart"
	// OpBind binds the serving runtime (its worker, API and dashboard) without
	// starting the worker: the desktop being open does not request serving.
	OpBind = "bind"
)

// Failure sources.
const (
	SourceSetup   = "setup"   // setup.RunObserved returned an error
	SourceRuntime = "runtime" // the active runtime could not be opened for serving
	SourceWorker  = "worker"  // the supervisor reported the worker failed
	SourceHome    = "home"    // the home is unresolved
)

// Errors returned by controller actions. They are rejections: the action was
// not performed and no state was changed.
var (
	ErrUnconfigured = errors.New("no Hachidori home is selected")
	ErrNotInstalled = errors.New("no valid active runtime; run setup first")
	ErrSetupRunning = errors.New("setup is already running")
	ErrBusy         = errors.New("another application action is in progress")
	ErrRuntimeBusy  = errors.New("the runtime is running; stop it first")
	ErrClosed       = errors.New("controller is closed")
)

// PhasePreflight is the failure phase of a runtime that could not be opened for
// serving: the active runtime was refused before any worker process started.
const PhasePreflight = "preflight"

// Failure is a structured application failure. Class and Message come from
// the owning authority (worker.Failure classes for worker failures). Phase
// names where it failed: the setup phase for a setup or maintenance action,
// the worker's last reported phase (spawning, importing, loading, warming)
// for a worker, PhasePreflight for a runtime that was never started. Step is
// the setup step that was running when a setup or maintenance action failed.
type Failure struct {
	Source   string   `json:"source"`
	Model    string   `json:"model,omitempty"`    // resident model that failed, for a resident set
	Provider string   `json:"provider,omitempty"` // its provider
	Class    string   `json:"class,omitempty"`
	Phase    string   `json:"phase,omitempty"`
	Step     string   `json:"step,omitempty"`
	Message  string   `json:"message"`
	Stderr   []string `json:"stderr_tail,omitempty"`
	// Diagnostic is the identity of the Forge diagnostic recorded for a failed
	// materialization, optimization, probe or certification ("" when none was
	// written). Inspect or export it with `hachidori forge diagnostics`.
	Diagnostic string `json:"diagnostic,omitempty"`
}

func (f *Failure) Error() string {
	where := f.Source
	if f.Model != "" {
		where += " " + f.Model
	}
	if f.Phase != "" {
		where += " (" + f.Phase + ")"
	}
	if f.Class != "" {
		return where + ": " + f.Class + ": " + f.Message
	}
	return where + ": " + f.Message
}

// Runtime is one serving binding of an active runtime: the worker.Lifecycle
// start/stop authority plus the /v1/status document of the same supervisor.
// WorkerRuntime builds the production binding; tests substitute fakes.
type Runtime interface {
	Start() bool
	Stop()
	Restart()
	Running() bool
	Status() server.Status
}

// project maps authority facts to the application state. It is pure so the
// mapping is deterministic and testable on its own.
//
// Precedence: unresolved home, then the in-flight action (installing,
// stopping), then the bound runtime (worker state from status), then the
// last action's failure, then installation state.
func project(home string, op string, running bool, st *server.Status, lastFail *Failure, installed bool) (State, *Failure) {
	if home == "" {
		return Unconfigured, nil
	}
	switch op {
	case OpSetup, OpReconcile:
		return Installing, nil
	case OpStop:
		return Stopping, nil
	}
	if running && st != nil {
		switch st.Worker.State {
		case worker.StateReady:
			return Ready, nil
		case worker.StateFailed:
			return Failed, workerFailure(st.Worker.LastFailure, st.Worker.Phase)
		case worker.StateStarting:
			if st.Worker.Phase == "warming" {
				return Warming, nil
			}
			return Starting, nil
		case worker.StateRestarting:
			return Starting, nil
		case worker.StateStopped:
			// The Run loop is returning; not serving.
			return Stopping, nil
		}
		return Starting, nil
	}
	if op == OpStart || op == OpRestart || op == OpApply {
		return Starting, nil
	}
	if lastFail != nil {
		return Failed, lastFail
	}
	// A supervisor that gave up (startup failure, restart budget exhausted)
	// is no longer running but still reports failed.
	if st != nil && st.Worker.State == worker.StateFailed {
		return Failed, workerFailure(st.Worker.LastFailure, st.Worker.Phase)
	}
	if installed {
		return Installed, nil
	}
	return NotInstalled, nil
}

// workerFailure is the application view of the supervisor's last failure.
// phase is the supervisor's phase, which stays at the last one the worker
// reported once it has failed.
func workerFailure(v *worker.FailureView, phase string) *Failure {
	if v == nil {
		return &Failure{Source: SourceWorker, Phase: phase, Message: "worker failed"}
	}
	return &Failure{Source: SourceWorker, Class: v.Class, Phase: phase, Message: v.Message, Stderr: v.Stderr}
}
