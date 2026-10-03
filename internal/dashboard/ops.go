package dashboard

// Presentation of long-running work: model/runtime operations (phases, steps,
// byte progress) and the worker's own startup lifecycle. Everything here
// restates what the owning authorities reported (app.Operation through
// ModelOp, and the supervisor's snapshot); it decides nothing and keeps no
// state. A percentage is shown only for a step that reported a total.

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/redact"
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
	VariantLabel   string // the operator's name of Variant, from the inventory's manifest
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
		n.VariantLabel = variantLabels(st.Inventory.Variants)[a.Variant]
		running := ""
		if rt.Variant != nil {
			running = rt.Variant.ID
		}
		n.VariantDiffers = n.Model != "" && n.Variant != running
		n.Differs = n.Model != "" && (n.Model != rt.ModelID || n.Device != rt.Device || n.VariantDiffers)
	}
	return n
}

// Workspaces that own a long-running operation. The owning workspace shows its
// full phases, progress and console while it runs; every other workspace shows
// only the compact headline in the shell and, once it ends, a compact summary.
const (
	ownerModels = "models"
	ownerForge  = "forge"
)

// opOwners is where each operation is started and followed. An operation kind
// that is not listed (an update download, which has its own panel) belongs to
// no maintenance workspace.
var opOwners = map[string]string{
	"setup": ownerModels, "materialize": ownerModels, "repair": ownerModels, "activate": ownerModels,
	"verify": ownerModels, "remove": ownerModels, "desired_state": ownerModels, "runtime_reconcile": ownerModels,
	"optimize": ownerForge, "certify": ownerForge, "forge_certify": ownerForge, "forge_build_evaluate": ownerForge,
	"preflight": ownerForge, "probe": ownerForge, "execute": ownerForge, "apply": ownerForge,
}

// opKindLabels name an operation kind for the operator. The raw kind stays on
// the element (data-kind) for evidence and tests.
var opKindLabels = map[string]string{
	"setup": "Set up", "materialize": "Materialize", "repair": "Repair", "activate": "Activate",
	"verify": "Verify", "remove": "Remove", "desired_state": "Apply desired state", "runtime_reconcile": "Reconcile runtime",
	"optimize": "Build variant", "certify": "Certify variant", "forge_certify": "Certify variant",
	"forge_build_evaluate": "Build and evaluate", "preflight": "Preflight", "probe": "Probe variant",
	"execute": "Execute", "apply": "Apply variant", "update": "Update",
}

// opOutcomes state what a successfully finished operation did, in the words the
// owning action already uses. They restate the plan the operation completed;
// they add no measurement.
var opOutcomes = map[string]string{
	"setup":                "Runtime and model are set up and activated.",
	"materialize":          "Materialized. It is not activated until you activate it.",
	"repair":               "Repaired and verified.",
	"activate":             "Activated. A running worker keeps its current runtime until you restart it.",
	"verify":               "Verified against its pinned digest.",
	"remove":               "Removed from HACHIDORI_HOME.",
	"desired_state":        "Desired state applied: the runtime serves the requested target.",
	"runtime_reconcile":    "Runtime reconciled: the required dependency runtime is active; models and variants were reused.",
	"optimize":             "Variant built.",
	"certify":              "Certification recorded.",
	"forge_certify":        "Certification recorded.",
	"forge_build_evaluate": "Candidate built and evaluated. Applying it stays your decision.",
	"preflight":            "Preflight recorded.",
	"probe":                "Probe recorded.",
	"apply":                "Applied: the runtime is READY on the variant and answered a typed decision.",
	"update":               "The update was downloaded and verified.",
}

func opOwner(kind string) string { return opOwners[kind] }

// OwnedBy reports whether the workspace nav owns the operation.
func (o ModelOp) OwnedBy(nav string) bool { return opOwners[o.Kind] == nav }

// opLabel is the operator's name of an operation kind.
func opLabel(kind string) string {
	if l, ok := opKindLabels[kind]; ok {
		return l
	}
	return kind
}

// opOutcome is what a successfully finished operation accomplished, or "".
func opOutcome(o ModelOp) string {
	if o.Failure != "" {
		return ""
	}
	return opOutcomes[o.Kind]
}

// opTarget is what an operation acted on: its device, model and target.
func opTarget(o ModelOp) string {
	return strings.Join(strings.Fields(o.Device+" "+o.Model+" "+o.Target), " ")
}

// opHref is where an operation is followed.
func opHref(kind string) string {
	switch opOwners[kind] {
	case ownerModels:
		return "/models"
	case ownerForge:
		return "/forge"
	}
	if kind == "update" {
		return "/settings/updates"
	}
	return "/"
}

// LastActivity is the latest real backend movement of the operation: the later
// of its last reported phase or step and the last write of its own log output.
// It is zero when the backend reported none; it is never the dashboard's clock.
func (o ModelOp) LastActivity() time.Time {
	if o.ConsoleAt.After(o.Activity) {
		return o.ConsoleAt
	}
	return o.Activity
}

// Console bounds. The adapter already scrubs and bounds the tail; the
// dashboard repeats both so that no source can widen what the console shows.
const (
	consoleLines     = 100
	consoleLineBytes = 512
)

// consoleTail is the bounded, redacted recent output of an operation.
func consoleTail(o ModelOp) []string {
	lines := o.Console
	if len(lines) > consoleLines {
		lines = lines[len(lines)-consoleLines:]
	}
	s := redact.New("")
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		out = append(out, s.Line(l, consoleLineBytes))
	}
	return out
}

// opHeadline is the compact line every workspace shows while an operation is
// running: what it is, where it is and how long it has run. It is the only
// view of a running operation outside the workspace that owns it.
type opHeadline struct {
	Label, Position, Elapsed, Href, Kind string
}

func headlineOf(o *ModelOp) *opHeadline {
	if o == nil || !o.Finished.IsZero() {
		return nil
	}
	return &opHeadline{Label: opLabel(o.Kind), Position: phasePosition(*o), Elapsed: since(o.Started), Href: opHref(o.Kind), Kind: o.Kind}
}
