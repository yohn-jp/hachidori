package setup

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

// Runtime continuity: the dependency runtime (the immutable Python/CUDA
// environment, identified by its RuntimeSpec) and the Hachidori worker (the
// script an application build delivers and runs in it) are separate things.
// A worker-only change leaves the dependency runtime identity unchanged and
// the materialized environment is reused; a change of the dependency contract
// yields a new runtime identity that is materialized and activated by
// reconciliation, beside the runtime that stays valid until then.

// WorkerIdentity is the Hachidori worker implementation an application build
// delivers. SHA256 is observable evidence of which worker ran; it is never a
// compatibility test. ABI is the explicit worker/runtime boundary: the
// environment contract the worker requires, compared with the one a runtime
// was derived for.
type WorkerIdentity struct {
	SHA256 string `json:"sha256"`
	ABI    string `json:"abi"`
}

// BuildWorker is the serving worker identity of this build.
func BuildWorker() WorkerIdentity {
	return WorkerIdentity{SHA256: digest(py.Script), ABI: home.WorkerABIServing}
}

// WorkerDigest is the digest of the serving worker script this build embeds.
func WorkerDigest() string { return BuildWorker().SHA256 }

// DeliveredWorker is a worker script written under HACHIDORI_HOME/workers/.
type DeliveredWorker struct {
	WorkerIdentity
	Path string // absolute path of the script
}

// DeliverWorker makes the serving worker script of this build available under
// workers/<digest>/ and returns it. The directory is named by the script's own
// digest and so is immutable: an existing script with that digest is reused
// untouched, anything else at that path is replaced atomically. Delivering a
// worker never touches a dependency runtime.
func DeliverWorker(h home.Home) (DeliveredWorker, error) {
	return deliver(h, kindOf(home.RuntimeSpec{}))
}

// DeliverOptimizer is DeliverWorker for the optimizer script.
func DeliverOptimizer(h home.Home) (DeliveredWorker, error) {
	return deliver(h, kindOf(home.RuntimeSpec{Role: home.RoleOptimizer}))
}

func deliver(h home.Home, k runtimeKind) (DeliveredWorker, error) {
	sum := digest(k.script)
	d := DeliveredWorker{
		WorkerIdentity: WorkerIdentity{SHA256: sum, ABI: home.WorkerABIFor(k.role)},
		Path:           filepath.Join(h.WorkerDir(sum), k.scriptName),
	}
	if got, err := FileSHA256(d.Path); err == nil && got == sum {
		return d, nil
	}
	if err := os.MkdirAll(filepath.Dir(d.Path), 0o755); err != nil {
		return d, fmt.Errorf("delivering worker %.12s: %w", sum, err)
	}
	if err := home.WriteFileAtomic(d.Path, k.script, 0o644); err != nil {
		return d, fmt.Errorf("delivering worker %.12s: %w", sum, err)
	}
	return d, nil
}

// FindRuntime returns the directory (name under runtime/) that holds the
// dependency runtime spec describes, and whether there is one.
//
// A runtime materialized under the current identity scheme is found by its
// identity. A runtime materialized before the environment was separated from
// the worker has an identity that includes a worker digest, so it is not
// found by name; it is found when its manifest is intact (its legacy spec
// derives the identity it carries) and its declared environment is exactly
// spec's. Both are only declarations: a caller that is going to use the runtime
// verifies it (verifyPublished). With several equivalent legacy runtimes the
// lexically first is chosen, so the choice is deterministic.
func FindRuntime(h home.Home, spec home.RuntimeSpec) (string, bool) {
	id := spec.ID()
	if fi, err := os.Stat(h.Path("runtime", id)); err == nil && fi.IsDir() {
		return id, true
	}
	entries, err := os.ReadDir(h.Path("runtime"))
	if err != nil {
		return "", false
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	for _, name := range names {
		var m home.RuntimeManifest
		if home.ReadJSON(h.Path("runtime", name, "manifest.json"), &m) != nil || m.Legacy == nil {
			continue
		}
		if m.CheckIdentity(name) == nil && m.Satisfies(spec) {
			return name, true
		}
	}
	return "", false
}

// RuntimeDirFor is the directory name under runtime/ where the runtime spec
// describes is, or will be materialized: an existing equivalent runtime's, or
// the identity spec derives.
func RuntimeDirFor(h home.Home, spec home.RuntimeSpec) string {
	if dir, ok := FindRuntime(h, spec); ok {
		return dir
	}
	return spec.ID()
}

// CheckRuntimeManifest checks that manifest m, read from runtime directory
// name, is intact and declares exactly the dependency environment spec
// describes. It reads nothing from the runtime itself.
func CheckRuntimeManifest(name string, m home.RuntimeManifest, spec home.RuntimeSpec) error {
	if err := m.CheckIdentity(name); err != nil {
		return err
	}
	if !m.Satisfies(spec) {
		return fmt.Errorf("runtime %s is dependency runtime %s (differs from the required %s in %s)",
			name, m.EnvironmentID(), spec.ID(), strings.Join(m.Environment().Differences(spec), ", "))
	}
	return nil
}

// Compatibility is the state of an activated dependency runtime relative to
// what this build requires.
type Compatibility string

const (
	// CompatCurrent: the active runtime has the required identity.
	CompatCurrent Compatibility = "current"
	// CompatEquivalent: the active runtime carries an identity of the scheme
	// that included the worker digest, and declares exactly the required
	// environment. It is used as is; nothing is rematerialized.
	CompatEquivalent Compatibility = "equivalent"
	// CompatStale: the active runtime is a different dependency environment
	// from the one this build requires. Reconciliation materializes the
	// required one and activates it when ready.
	CompatStale Compatibility = "stale"
	// CompatIncompatible: the active runtime was derived for another worker
	// ABI than this build's worker requires.
	CompatIncompatible Compatibility = "incompatible"
)

// Errors of the worker/runtime compatibility gate.
var (
	// ErrWorkerABI marks a runtime derived for a different worker ABI.
	ErrWorkerABI = errors.New("the runtime was derived for a different worker ABI")
	// ErrRuntimeStale marks an active runtime that is not the dependency
	// runtime this build requires.
	ErrRuntimeStale = errors.New("the active runtime is not the dependency runtime this build requires")
)

// Reconciliation is the observable state of the active dependency runtime
// against this build: the dependency runtime identities, the worker identity
// and why they do or do not fit. Dependency identities and the worker identity
// are separate fields: a worker-only update changes Worker and nothing else.
type Reconciliation struct {
	Device string        `json:"device"`
	State  Compatibility `json:"state"`
	// ActiveRuntime is the runtime directory the activation record names;
	// ActiveEnvironment the dependency runtime identity its manifest declares
	// (the same, unless the directory carries a legacy identity).
	ActiveRuntime     string `json:"active_runtime"`
	ActiveEnvironment string `json:"active_environment"`
	// RequiredEnvironment is the dependency runtime identity this build
	// requires for Device.
	RequiredEnvironment string `json:"required_environment"`
	// Differences names what separates the active environment from the
	// required one.
	Differences []string       `json:"differences,omitempty"`
	Worker      WorkerIdentity `json:"worker"`
	Detail      string         `json:"detail,omitempty"`
}

// Needs reports whether reconciliation has work to do.
func (r Reconciliation) Needs() bool { return r.State == CompatStale || r.State == CompatIncompatible }

// Assess compares an activated runtime with the dependency runtime this build
// requires for the activation's device. It reads nothing but its arguments.
func Assess(a home.Active, rm home.RuntimeManifest) (Reconciliation, error) {
	want, err := Desired(a.Device)
	if err != nil {
		return Reconciliation{}, err
	}
	have := rm.Environment()
	r := Reconciliation{Device: a.Device, ActiveRuntime: a.Runtime, ActiveEnvironment: rm.EnvironmentID(),
		RequiredEnvironment: want.ID(), Worker: BuildWorker()}
	switch {
	case rm.Satisfies(want) && rm.Legacy == nil:
		r.State = CompatCurrent
	case rm.Satisfies(want):
		r.State = CompatEquivalent
		r.Detail = fmt.Sprintf("runtime %s was materialized under the previous identity scheme; its declared environment is the required one and it is used as is", a.Runtime)
	default:
		r.Differences = have.Differences(want)
		r.State = CompatStale
		r.Detail = fmt.Sprintf("runtime %s differs from the required dependency runtime %s in %s", a.Runtime, want.ID(), strings.Join(r.Differences, ", "))
		if have.WorkerABI != want.WorkerABI {
			r.State = CompatIncompatible
			r.Detail = fmt.Sprintf("runtime %s was derived for worker ABI %q, this build's worker requires %q", a.Runtime, have.WorkerABI, want.WorkerABI)
		}
	}
	return r, nil
}

// CheckRuntimeCompatibility is the launch gate for an activated runtime. The
// runtime must be, or declare exactly, the dependency environment this build
// requires; the worker of this build is delivered separately and is not
// compared by digest. A mismatch is refused with the cause and what resolves
// it; nothing is bypassed.
func CheckRuntimeCompatibility(a home.Active, rm home.RuntimeManifest) error {
	r, err := Assess(a, rm)
	if err != nil {
		return err
	}
	switch r.State {
	case CompatCurrent, CompatEquivalent:
		return nil
	case CompatIncompatible:
		return fmt.Errorf("%w: %s. The desktop reconciles this on its own: it materializes the required runtime %s and activates it when it is ready; "+
			"or run `hachidori setup --device %s`. Installed models and variants are reused",
			ErrWorkerABI, r.Detail, r.RequiredEnvironment, a.Device)
	}
	return fmt.Errorf("%w: %s. The desktop reconciles this on its own: it materializes the required runtime and activates it when it is ready, keeping this one until then; "+
		"or run `hachidori setup --device %s`. Installed models and variants are reused",
		ErrRuntimeStale, r.Detail, a.Device)
}

// AssessHome is Assess for the activation record under h. A home with no
// readable activation, or an unreadable runtime manifest, has nothing to
// reconcile: those are reported by the start itself.
func AssessHome(h home.Home) (Reconciliation, bool) {
	var a home.Active
	var rm home.RuntimeManifest
	if home.ReadJSON(h.Path("state", "active-runtime.json"), &a) != nil ||
		home.ReadJSON(h.Path("runtime", a.Runtime, "manifest.json"), &rm) != nil || rm.CheckIdentity(a.Runtime) != nil {
		return Reconciliation{}, false
	}
	r, err := Assess(a, rm)
	return r, err == nil
}

// ReconcileActive brings the active runtime to the dependency runtime this
// build requires, for the activation's own device, model and variant. A
// current or equivalent runtime is left alone (nothing is rematerialized,
// downloaded or rebuilt). Otherwise the required runtime is materialized
// through the staged, verified and atomically published path (or reused when
// already materialized), the active model is verified and reused, and the
// activation record is replaced, atomically, only after the runtime, the model
// and the variant (with its certification) all verify. A failure at any step
// leaves the activation record and the runtime it names untouched; rerunning
// reconciles again.
func ReconcileActive(ctx context.Context, h home.Home, log io.Writer, obs *Observer) (Reconciliation, bool, error) {
	var a home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); err != nil {
		return Reconciliation{}, false, fmt.Errorf("no active runtime to reconcile (run `hachidori setup`): %w", err)
	}
	var rm home.RuntimeManifest
	if err := home.ReadJSON(h.Path("runtime", a.Runtime, "manifest.json"), &rm); err != nil {
		return Reconciliation{}, false, fmt.Errorf("runtime %s: invalid manifest: %w", a.Runtime, err)
	}
	if err := rm.CheckIdentity(a.Runtime); err != nil {
		return Reconciliation{}, false, fmt.Errorf("runtime %s: %w", a.Runtime, err)
	}
	rec, err := Assess(a, rm)
	if err != nil {
		return rec, false, err
	}
	if !rec.Needs() {
		return rec, false, nil
	}
	model, err := ActiveModel(a)
	if err != nil {
		return rec, false, err
	}
	fmt.Fprintf(log, "reconciling: %s\n", rec.Detail)
	next, err := reconcile(ctx, h, a.Device, model.ID, log, obs, PhasePublish)
	if err != nil {
		return rec, false, err
	}
	changed, err := ActivateTarget(h, a.Device, model.ID, ActivateOptions{Variant: a.Variant, AllowUncertified: a.Experimental}, log, obs)
	if err != nil {
		return rec, false, fmt.Errorf("activating runtime %s: %w", next.Runtime, err)
	}
	after, ok := AssessHome(h)
	if !ok || after.Needs() {
		return rec, changed, fmt.Errorf("runtime %s was activated but is still not the dependency runtime this build requires (%s)", next.Runtime, after.Detail)
	}
	// Report the state that was actually activated. Reconciliation may reuse a
	// verified schema-1 runtime whose dependency environment is equivalent to
	// the current spec; that is intentionally CompatEquivalent, not current.
	after.Detail = "reconciled"
	return after, changed, nil
}
