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
)

// Operation kinds.
const (
	OpSetup   = "setup"
	OpStart   = "start"
	OpStop    = "stop"
	OpRestart = "restart"
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

// Failure is a structured application failure. Class and Message come from
// the owning authority (worker.Failure classes for worker failures).
type Failure struct {
	Source  string   `json:"source"`
	Class   string   `json:"class,omitempty"`
	Message string   `json:"message"`
	Stderr  []string `json:"stderr_tail,omitempty"`
}

func (f *Failure) Error() string {
	if f.Class != "" {
		return f.Source + ": " + f.Class + ": " + f.Message
	}
	return f.Source + ": " + f.Message
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
	case OpSetup:
		return Installing, nil
	case OpStop:
		return Stopping, nil
	}
	if running && st != nil {
		switch st.Worker.State {
		case worker.StateReady:
			return Ready, nil
		case worker.StateFailed:
			return Failed, workerFailure(st.Worker.LastFailure)
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
	if op == OpStart || op == OpRestart {
		return Starting, nil
	}
	if lastFail != nil {
		return Failed, lastFail
	}
	// A supervisor that gave up (startup failure, restart budget exhausted)
	// is no longer running but still reports failed.
	if st != nil && st.Worker.State == worker.StateFailed {
		return Failed, workerFailure(st.Worker.LastFailure)
	}
	if installed {
		return Installed, nil
	}
	return NotInstalled, nil
}

func workerFailure(v *worker.FailureView) *Failure {
	if v == nil {
		return &Failure{Source: SourceWorker, Message: "worker failed"}
	}
	return &Failure{Source: SourceWorker, Class: v.Class, Message: v.Message, Stderr: v.Stderr}
}
