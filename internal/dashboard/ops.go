package dashboard

// Presentation of long-running work: model/runtime operations (phases, steps,
// byte progress) and the worker's own startup lifecycle. Everything here
// restates what the owning authorities reported (app.Operation through
// ModelOp, and the supervisor's snapshot); it decides nothing and keeps no
// state. A percentage is shown only for a step that reported a total.

import (
	"fmt"
	"slices"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Stage statuses.
const (
	stageDone    = "done"
	stageCurrent = "current"
	stagePending = "pending"
	stageFailed  = "failed"
	stageSkipped = "skipped" // a finished action never entered it (nothing to do)
)

// stage is one phase of an operation or one phase of a worker's startup, in
// order, with whether it is done, current, pending or failed. Label is a
// catalog message ID.
type stage struct {
	Name, Label, Status string
}

var phaseLabels = map[string]string{
	"preparing":  "Preparing",
	"runtime":    "Runtime",
	"model":      "Model",
	"publish":    "Publishing",
	"activation": "Activation",
	// System One variants: optimization and certification
	"variant":        "Variant",
	"starting":       "Starting optimizer",
	"loading_source": "Loading source model",
	"resolving":      "Resolving modules",
	"quantizing":     "Quantizing",
	"serializing":    "Serializing",
	"verifying":      "Verifying output",
	"loading_runs":   "Loading runs",
	"comparing":      "Comparing runs",
	"recording":      "Recording certification",
	// Forge readiness: the preflight and the probe of a persisted variant
	"preflight": "Preflight",
	"probing":   "Probing variant",
	// self-contained certification (forge_certify)
	"materializing": "Materializing runtime",
	"probe":         "Probing variant",
	"reference_run": "Reference run",
	"candidate_run": "Variant run",
	"aligning":      "Aligning runs",
	"certifying":    "Certifying",
	"persisting":    "Recording evidence",
	// certified variant apply (apply)
	"validating":     "Validating",
	"snapshotting":   "Snapshotting",
	"rebinding":      "Rebinding runtime",
	"awaiting_ready": "Waiting for READY",
	"proving":        "Proving provenance",
	"smoke":          "Typed decision",
	"finalizing":     "Finalizing",
	"rolling_back":   "Rolling back",
	// update check and download (internal/update)
	"releases": "Release list",
	"checksum": "Checksum file",
	"download": "Download",
	"verify":   "Verification",
}

var stepLabels = map[string]string{
	"fetching":      "Fetching",
	"downloading":   "Downloading",
	"verifying":     "Verifying",
	"materializing": "Materializing",
	"publishing":    "Publishing",
	"activating":    "Activating",
	"removing":      "Removing",
	"probing":       "Probing",
}

func phaseLabel(p string) string {
	if l, ok := phaseLabels[p]; ok {
		return l
	}
	return p
}

// kindPhaseLabels name a phase that means something else in one operation than
// in the others (the optimizer resolves modules; a certification resolves its
// exact inputs).
var kindPhaseLabels = map[string]map[string]string{
	"forge_certify": {"resolving": "Resolving inputs"},
	"forge_build_evaluate": {
		"resolve_inputs": "Resolving build intent",
		"provision":      "Provisioning prerequisites",
		"build":          "Building candidate",
		"resolving":      "Resolving evaluation inputs",
	},
}

// opPhaseLabel is the label of phase p of an operation of the given kind.
func opPhaseLabel(kind, p string) string {
	if l, ok := kindPhaseLabels[kind][p]; ok {
		return l
	}
	return phaseLabel(p)
}

func stepLabel(s string) string {
	if l, ok := stepLabels[s]; ok {
		return l
	}
	return s
}

// Operation detail belongs to one workspace. Every page shows the operation in
// flight as one compact headline in the shell; the phase strip, the position
// ("n of m"), the progress bar and the operation's own execution facts render
// only on the workspace that owns the operation. This is a rule of
// projection over the controller's one operation: no state is copied or kept
// per page.
//
// The composed Forge operations (and the low-level Forge steps they are built
// from) are owned by Forge. A kind that is not listed has no single owning
// workspace and is shown wherever it always was.
var opOwners = map[string]string{
	"forge_build_evaluate": "forge",
	"forge_certify":        "forge",
	"optimize":             "forge",
	"certify":              "forge",
	"preflight":            "forge",
	"probe":                "forge",
	"execute":              "forge",
}

// opOwner is the workspace (Chrome.Nav) that owns the detail of an operation of
// the given kind, "" when it has none.
func opOwner(kind string) string { return opOwners[kind] }

// opDetailOn reports whether the detail of an operation of kind renders on the
// workspace nav.
func opDetailOn(nav, kind string) bool {
	owner := opOwner(kind)
	return owner == "" || owner == nav
}

// scopedTo is the state as workspace nav may render it in detail. The
// operation in flight, when another workspace owns it, is left to the shell's
// headline; a finished one stays as a compact outcome line that points to the
// owner (so a failure is never hidden, but its phases and diagnostics are not
// repeated).
func (st ModelsState) scopedTo(nav string) ModelsState {
	if st.Busy != nil && !opDetailOn(nav, st.Busy.Kind) {
		st.Busy = nil
	}
	if st.Last != nil && !opDetailOn(nav, st.Last.Kind) {
		last := *st.Last
		last.Compact = true
		st.Last = &last
	}
	return st
}

var opKindLabels = map[string]string{
	"forge_build_evaluate": "Build & evaluate",
	"forge_certify":        "Certification",
	"optimize":             "Building candidate",
	"certify":              "Certifying",
	"preflight":            "Preflight",
	"probe":                "Probing variant",
	"execute":              "Evaluation run",
	"desired_state":        "Applying desired state",
	"apply":                "Applying variant",
	"activate":             "Activating",
	"materialize":          "Materializing",
	"repair":               "Repairing",
	"setup":                "Setting up",
	"verify":               "Verifying",
	"remove":               "Removing",
}

// opHeadline is the compact global statement that Hachidori is doing work: what
// the operation is and the phase it is in. It is what every workspace keeps
// while only the owning one shows the detail (the phase strip, the phase
// number, progress and the operation's execution facts).
type opHeadline struct {
	// Kind and Phase are catalog message IDs.
	Kind, Phase string
	// Href is the workspace holding the detail ("" when the operation has no
	// single owner).
	Href string
}

var navHrefs = map[string]string{"forge": "/forge", "models": "/models", "tuning": "/tuning"}

// OwnerHref is where the detail of an operation of this kind is shown ("" when
// it has no single owner).
func (o ModelOp) OwnerHref() string { return navHrefs[opOwner(o.Kind)] }

// headlineOf restates the operation in flight; nil when none is.
func headlineOf(o *ModelOp) *opHeadline {
	if o == nil {
		return nil
	}
	h := &opHeadline{Kind: o.Kind, Href: navHrefs[opOwner(o.Kind)]}
	if l, ok := opKindLabels[o.Kind]; ok {
		h.Kind = l
	}
	if o.Phase != "" {
		h.Phase = opPhaseLabel(o.Kind, o.Phase)
	}
	return h
}

// opStages places an operation among the phases it goes through. A phase is
// done or current only if the action really entered it; one that failed is
// marked failed; once the action finished, a planned phase it never entered
// is skipped rather than shown as pending forever.
func opStages(o ModelOp) []stage {
	plan := o.Plan
	if len(plan) == 0 {
		plan = o.Phases
	}
	running := o.Finished.IsZero()
	out := make([]stage, 0, len(plan))
	for _, p := range plan {
		st := stage{Name: p, Label: opPhaseLabel(o.Kind, p), Status: stagePending}
		switch i := slices.Index(o.Phases, p); {
		case i < 0 && !running && o.Failure == "":
			st.Status = stageSkipped
		case i < 0:
		case o.Failure != "" && p == o.FailurePhase:
			st.Status = stageFailed
		case running && p == o.Phase:
			st.Status = stageCurrent
		default:
			st.Status = stageDone
		}
		out = append(out, st)
	}
	return out
}

// phasePosition is "n of m" for the phase an operation is in, or "".
func phasePosition(o ModelOp) string {
	plan := o.Plan
	if i := slices.Index(plan, o.Phase); i >= 0 {
		return fmt.Sprintf("%d of %d", i+1, len(plan))
	}
	return ""
}

// workerPhases are the phases a worker reports while it starts, in order
// (internal/worker). They are measured by the worker itself; there is no
// total for any of them.
var workerPhases = []string{"spawning", "importing", "loading", "warming", "ready"}

var workerPhaseLabels = map[string]string{
	worker.PhasePreflight: "Checking the runtime",
	"spawning":            "Starting worker process",
	"importing":           "Importing provider",
	"loading":             "Loading model",
	"warming":             "Warming up",
	"ready":               "Ready",
}

// workerStages is the worker's startup as stages. It is empty unless a
// worker is starting, failed during startup or is ready; a stopped runtime has
// no lifecycle to show. The current stage of a failed worker is the one it
// failed in.
func workerStages(w worker.Snapshot) []stage {
	switch w.State {
	case worker.StateStarting, worker.StateRestarting, worker.StateFailed, worker.StateReady:
	default:
		return nil
	}
	if w.Phase == worker.PhasePreflight {
		return nil // refused before any process existed: there is no startup to show
	}
	at := slices.Index(workerPhases, w.Phase)
	if w.State == worker.StateRestarting || at < 0 {
		at = 0
	}
	out := make([]stage, len(workerPhases))
	for i, p := range workerPhases {
		st := stage{Name: p, Label: workerPhaseLabels[p], Status: stagePending}
		switch {
		case w.State == worker.StateReady:
			st.Status = stageDone
		case i < at:
			st.Status = stageDone
		case i == at && w.State == worker.StateFailed:
			st.Status = stageFailed
		case i == at:
			st.Status = stageCurrent
		}
		out[i] = st
	}
	return out
}

// workerPhaseWord is the operator's name of what the worker is doing, from
// its reported phase.
func workerPhaseWord(phase string) string {
	if l, ok := workerPhaseLabels[phase]; ok {
		return l
	}
	return phase
}

// bytesIn renders a byte count in binary units.
func bytesIn(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	f, i := float64(n)/unit, 0
	for ; f >= unit && i < 4; i++ {
		f /= unit
	}
	return fmt.Sprintf("%.1f %ciB", f, "KMGT"[i])
}

// since is how long ago t was, compactly; "" for the zero time.
func since(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return uptime(int(time.Since(t).Seconds()))
}

// took is the duration of a finished action.
func took(o ModelOp) string {
	if o.Started.IsZero() || o.Finished.IsZero() {
		return ""
	}
	return uptime(int(o.Finished.Sub(o.Started).Seconds()))
}

// failureView is a worker failure with where it happened and what to do
// about it. Hint is a catalog message ID.
type failureView struct {
	Class, Phase, PhaseWord, Message, Hint string
	Stderr                                 []string
}

// failureHints are the actionable meaning of each failure class.
var failureHints = map[string]string{
	worker.ClassPreflight: "The active runtime was refused before any process was started. The message names why and how to recover: " +
		"materialize the current runtime, activate it in Models and choose Restart.",
	worker.ClassStartup: "The worker process ended before it reported its first message. The last lines of its output below name why " +
		"(an interpreter, argument or environment error). If the runtime was materialized by an older Hachidori, materialize and activate the current one in Models.",
	worker.ClassProviderInit: "The provider packages could not be imported by the private Python. Verify the runtime in Models; Repair rebuilds one that fails verification.",
	worker.ClassDevice: "The requested device is not available. Hachidori never falls back to another device: fix the driver or the device, " +
		"or deliberately materialize and activate the cpu runtime in Models.",
	worker.ClassModelLoad:     "The model could not be loaded. Verify the model in Models; Repair rebuilds an artifact that fails verification.",
	worker.ClassWarmup:        "The model loaded on the requested device but its warm-up inference failed. The worker output below has the error.",
	worker.ClassStartTimeout:  "The worker did not become ready within its startup limit. Check the phase it stopped in and the worker output below.",
	worker.ClassCrash:         "The worker process exited. The last lines of its output below name why.",
	worker.ClassUnresponsive:  "The worker stopped answering and was ended. The last lines of its output below may name why.",
	worker.ClassProtocolError: "The worker wrote something other than protocol messages. The last lines of its output below show what.",
}

const failureTail = 8

// failureOf describes the worker's failure while it is failed; it is nil when
// there is nothing to act on: no failure, a worker that recovered, or one that
// is being started again (its earlier failure stays in the attention list and
// in Diagnostics).
func failureOf(w worker.Snapshot) *failureView {
	f := w.LastFailure
	if f == nil || w.State != worker.StateFailed {
		return nil
	}
	v := &failureView{Class: f.Class, Phase: w.Phase, PhaseWord: workerPhaseWord(w.Phase), Message: f.Message, Hint: failureHints[f.Class]}
	v.Stderr = f.Stderr
	if len(v.Stderr) > failureTail {
		v.Stderr = v.Stderr[len(v.Stderr)-failureTail:]
	}
	if v.Hint == "" {
		v.Hint = "Read the worker output below and in Diagnostics."
	}
	return v
}

// nextStart is the identity the next start of the runtime would serve, set
// beside the runtime this dashboard reports and the maintenance action in
// flight or last finished. It restates the maintenance authority's state.
type nextStart struct {
	Model, Device string
	// Differs: the activation record names a model or device other than the
	// one this runtime was configured with (an activation is waiting for a
	// restart).
	Differs bool
	// Variant is the variant the activation record selects ("" for the source
	// artifact) and VariantDiffers whether the runtime executes another one.
	Variant        string
	VariantDiffers bool
	// Problem is why the active pair cannot be started by this build.
	Problem    string
	Busy, Last *ModelOp
}

// nextOf builds the view from the maintenance state and the runtime the
// status document reports.
func nextOf(st ModelsState, rt server.Runtime) *nextStart {
	n := &nextStart{Busy: st.Busy, Last: st.Last, Problem: st.Inventory.ActiveProblem}
	if a := st.Inventory.Active; a != nil {
		n.Device = a.Device
		for _, m := range st.Inventory.Models {
			if m.Active {
				n.Model = m.ID
			}
		}
		n.Variant = a.Variant
		running := ""
		if rt.Variant != nil {
			running = rt.Variant.ID
		}
		n.VariantDiffers = n.Model != "" && n.Variant != running
		n.Differs = n.Model != "" && (n.Model != rt.ModelID || n.Device != rt.Device || n.VariantDiffers)
	}
	return n
}
