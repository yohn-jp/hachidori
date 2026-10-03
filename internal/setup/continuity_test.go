package setup_test

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

func swapWorker(t *testing.T) {
	t.Helper()
	old := py.Script
	t.Cleanup(func() { py.Script = old })
	py.Script = append(append([]byte{}, old...), []byte("\n# worker-only change\n")...)
}

func requiredRuntime(t *testing.T, device string) home.RuntimeSpec {
	t.Helper()
	spec, err := setup.Desired(device)
	if err != nil {
		t.Fatal(err)
	}
	return spec
}

// A runtime written under the identity scheme that included the worker digest
// declares exactly the environment this build requires. It is recognised by
// that declaration (after its identity and its interpreter verify) and reused:
// nothing is rematerialized, downloaded or rebuilt, the current worker runs in
// it, and the evidence names the dependency runtime and the worker separately.
func TestSchemaV1RuntimeIsReusedWithoutRematerialization(t *testing.T) {
	x := setup.NewFake(t)
	if err := x.Run("cpu", ""); err != nil {
		t.Fatal(err)
	}
	legacy := x.SchemaV1("cpu")
	spec := requiredRuntime(t, "cpu")
	if legacy == spec.ID() {
		t.Fatal("the fixture did not produce a runtime of the previous identity scheme")
	}
	activeBefore := rawActive(x.H)
	uv0, req0 := x.UVCalls(), x.Requests()

	inv := setup.Inspect(x.H, true)
	if inv.ActiveProblem != "" || inv.Reconciliation == nil || inv.Reconciliation.State != setup.CompatEquivalent ||
		inv.Reconciliation.ActiveRuntime != legacy || inv.Reconciliation.ActiveEnvironment != spec.ID() || inv.Reconciliation.RequiredEnvironment != spec.ID() {
		t.Fatalf("inventory problem %q reconciliation %+v", inv.ActiveProblem, inv.Reconciliation)
	}
	for _, r := range inv.Runtimes {
		if r.Device == "cpu" && (r.ID != spec.ID() || r.Directory != legacy || !r.Materialized || !r.Verified || !r.Active || r.Problem != "") {
			t.Fatalf("runtime entry %+v", r)
		}
	}

	cfg, rt, err := server.WorkerConfig(x.H, io.Discard)
	if err != nil {
		t.Fatalf("the equivalent runtime was refused: %v", err)
	}
	if err := cfg.Preflight(); err != nil {
		t.Fatal(err)
	}
	if rt.Runtime != spec.ID() || rt.RuntimeDirectory != legacy || rt.Worker == nil || rt.Worker.SHA256 != setup.WorkerDigest() {
		t.Fatalf("status runtime %+v", rt)
	}
	if want := filepath.Join(x.H.Root, "runtime", legacy, "env", "bin", "python"); cfg.Python != want {
		t.Fatalf("python %s, want %s", cfg.Python, want)
	}
	if cfg.Args[4] != filepath.Join(x.H.Root, "workers", setup.WorkerDigest(), "hachidori_worker.py") {
		t.Fatalf("the worker that runs is %s, not the delivered one of this build", cfg.Args[4])
	}

	// Setup reconciles to the same runtime and does nothing to it.
	if err := x.Run("cpu", ""); err != nil {
		t.Fatal(err)
	}
	if rawActive(x.H) != activeBefore || x.UVCalls() != uv0 || x.Requests() != req0 {
		t.Fatalf("setup rematerialized or redownloaded: uv %d->%d, requests %d->%d", uv0, x.UVCalls(), req0, x.Requests())
	}
	if _, err := os.Stat(x.H.Path("runtime", spec.ID())); !os.IsNotExist(err) {
		t.Fatalf("a second copy of the runtime was materialized: %v", err)
	}
	if _, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, nil); err != nil || changed {
		t.Fatalf("ReconcileActive = %v, %v", changed, err)
	}
	if x.UVCalls() != uv0 {
		t.Fatal("reconciliation of an equivalent runtime ran uv")
	}
}

// What cannot be verified is not reused: a runtime of the previous scheme whose
// declared environment differs, whose identity does not match its own spec, or
// whose interpreter no longer verifies is never taken for the required one.
func TestReconcileActiveReportsEquivalentWhenItReusesSchemaV1Runtime(t *testing.T) {
	x := setup.NewFake(t)
	if err := x.Run("cpu", ""); err != nil {
		t.Fatal(err)
	}
	// Keep a stale active runtime and a separate, verified schema-1 runtime
	// whose dependency environment exactly matches the current requirement.
	stale := x.Stale("cpu")
	if err := x.Run("cpu", ""); err != nil {
		t.Fatal(err)
	}
	legacy := x.SchemaV1("cpu")
	x.ActivateDir(stale)

	rec, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, nil)
	if err != nil || !changed {
		t.Fatalf("ReconcileActive = %+v, %v, %v", rec, changed, err)
	}
	if rec.State != setup.CompatEquivalent || rec.ActiveRuntime != legacy || rec.ActiveEnvironment != rec.RequiredEnvironment {
		t.Fatalf("reconciliation reported %+v; want the equivalent runtime actually activated", rec)
	}
	var a home.Active
	if err := home.ReadJSON(x.H.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	if a.Runtime != legacy {
		t.Fatalf("activation runtime %q, want reused legacy runtime %q", a.Runtime, legacy)
	}
}

func TestSchemaV1RuntimeThatIsNotVerifiablyEquivalentIsNotReused(t *testing.T) {
	rewrite := func(t *testing.T, x *setup.Fake, name string, edit func(m map[string]any)) {
		t.Helper()
		p := x.H.Path("runtime", name, "manifest.json")
		var m map[string]any
		if err := home.ReadJSON(p, &m); err != nil {
			t.Fatal(err)
		}
		edit(m)
		if err := home.WriteJSON(p, m); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("different environment", func(t *testing.T) {
		x := setup.NewFake(t)
		x.Run("cpu", "")
		spec := requiredRuntime(t, "cpu")
		legacy := x.SchemaV1("cpu")
		var rm home.RuntimeManifest
		home.ReadJSON(x.H.Path("runtime", legacy, "manifest.json"), &rm)
		l := *rm.Legacy
		l.Lock = strings.Repeat("0", 64) // the runtime was locked to other packages
		rewrite(t, x, legacy, func(m map[string]any) { m["spec"] = l; m["identity"] = l.ID() })
		if err := os.Rename(x.H.Path("runtime", legacy), x.H.Path("runtime", l.ID())); err != nil {
			t.Fatal(err)
		}
		x.ActivateDir(l.ID())
		if _, ok := setup.FindRuntime(x.H, spec); ok {
			t.Fatal("a runtime locked to other packages was found for the required one")
		}
		if r, ok := setup.AssessHome(x.H); !ok || r.State != setup.CompatStale || !slices.Contains(r.Differences, "lock_sha256") {
			t.Fatalf("assessment %+v", r)
		}
		uv0 := x.UVCalls()
		if err := x.Run("cpu", ""); err != nil {
			t.Fatal(err)
		}
		if x.UVCalls() == uv0 {
			t.Fatal("the required runtime was not materialized")
		}
		if _, err := os.Stat(x.H.Path("runtime", spec.ID(), "manifest.json")); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(x.H.Path("runtime", l.ID(), "manifest.json")); err != nil {
			t.Fatalf("the previous runtime was not preserved: %v", err)
		}
	})
	t.Run("identity does not match its spec", func(t *testing.T) {
		x := setup.NewFake(t)
		x.Run("cpu", "")
		spec := requiredRuntime(t, "cpu")
		legacy := x.SchemaV1("cpu")
		rewrite(t, x, legacy, func(m map[string]any) { m["identity"] = "cpu-0000000000000000" })
		if _, ok := setup.FindRuntime(x.H, spec); ok {
			t.Fatal("a runtime with a corrupt identity was found")
		}
	})
	t.Run("interpreter does not verify", func(t *testing.T) {
		x := setup.NewFake(t)
		x.Run("cpu", "")
		legacy := x.SchemaV1("cpu")
		if err := os.Remove(x.H.Path("runtime", legacy, "env", "bin", "python")); err != nil {
			t.Fatal(err)
		}
		if err := x.Run("cpu", ""); err == nil || !strings.Contains(err.Error(), "failed verification") || !strings.Contains(err.Error(), "never modified in place") {
			t.Fatalf("Run = %v", err)
		}
	})
}

func snapshotActive(t *testing.T, x *setup.Fake) (raw string, a home.Active) {
	t.Helper()
	raw = rawActive(x.H)
	if err := json.Unmarshal([]byte(raw), &a); err != nil {
		t.Fatal(err)
	}
	return raw, a
}

// When the required dependency runtime is absent the Controller's authority
// materializes it beside the active one and activates it only once it, the
// model and the variant verify. The activation record is replaced exactly once,
// at the end, atomically; the previous runtime stays; the model is not
// downloaded again.
func TestMissingDesiredRuntimeIsMaterializedAndActivatedAtomically(t *testing.T) {
	x := setup.NewFake(t)
	if err := x.Run("cpu", ""); err != nil {
		t.Fatal(err)
	}
	stale := x.Stale("cpu")
	spec := requiredRuntime(t, "cpu")
	if _, err := os.Stat(x.H.Path("runtime", spec.ID())); !os.IsNotExist(err) {
		t.Fatal("the required runtime exists before reconciliation")
	}
	prev, _ := snapshotActive(t, x)
	uv0, req0 := x.UVCalls(), x.Requests()

	inv := setup.Inspect(x.H, false)
	if inv.Reconciliation == nil || inv.Reconciliation.State != setup.CompatStale || inv.Reconciliation.ActiveRuntime != stale ||
		inv.Reconciliation.RequiredEnvironment != spec.ID() || !slices.Contains(inv.Reconciliation.Differences, "provider") ||
		!strings.Contains(inv.ActiveProblem, "not the dependency runtime") {
		t.Fatalf("inventory before: problem %q reconciliation %+v", inv.ActiveProblem, inv.Reconciliation)
	}

	var phases []setup.Phase
	obs := &setup.Observer{OnPhase: func(p setup.Phase) {
		phases = append(phases, p)
		if p != setup.PhaseActivation {
			if got := rawActive(x.H); got != prev {
				t.Errorf("the activation record changed during %s: %s", p, got)
			}
		}
	}}
	rec, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, obs)
	if err != nil || !changed {
		t.Fatalf("ReconcileActive = %v, %v", changed, err)
	}
	if rec.State != setup.CompatCurrent || rec.ActiveRuntime != spec.ID() {
		t.Fatalf("reconciliation %+v", rec)
	}
	if !slices.Contains(phases, setup.PhaseRuntime) || phases[len(phases)-1] != setup.PhaseActivation {
		t.Fatalf("phases %v", phases)
	}
	now, a := snapshotActive(t, x)
	if now == prev || a.Runtime != spec.ID() || a.ModelID != setup.DefaultModel || a.Device != "cpu" {
		t.Fatalf("active %+v", a)
	}
	if x.UVCalls() == uv0 {
		t.Fatal("the required runtime was not materialized")
	}
	if x.Requests() != req0 {
		t.Fatalf("reconciliation downloaded %d artifacts; the model and uv are reused", x.Requests()-req0)
	}
	if _, err := os.Stat(x.H.Path("runtime", stale, "manifest.json")); err != nil {
		t.Fatalf("the previous runtime was removed: %v", err)
	}
	inv = setup.Inspect(x.H, true)
	if inv.ActiveProblem != "" || inv.Reconciliation == nil || inv.Reconciliation.State != setup.CompatCurrent {
		t.Fatalf("inventory after: problem %q reconciliation %+v", inv.ActiveProblem, inv.Reconciliation)
	}

	// Reconciling again has nothing to do.
	uv1 := x.UVCalls()
	if _, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, nil); err != nil || changed || x.UVCalls() != uv1 {
		t.Fatalf("second reconcile: %v %v", changed, err)
	}
}

// A replacement that fails or is interrupted leaves the previous activation,
// the runtime it names and the model exactly as they were, and the next
// reconciliation completes it.
func TestFailedReplacementLeavesPreviousRuntimeIntact(t *testing.T) {
	x := setup.NewFake(t)
	if err := x.Run("cpu", ""); err != nil {
		t.Fatal(err)
	}
	stale := x.Stale("cpu")
	spec := requiredRuntime(t, "cpu")
	prev, _ := snapshotActive(t, x)
	modelsBefore := dirDigest(t, x.H.Path("models"))
	staleBefore := dirDigest(t, x.H.Path("runtime", stale))

	x.FailUV("sync")
	_, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, nil)
	if err == nil || changed || !strings.Contains(err.Error(), spec.ID()) {
		t.Fatalf("ReconcileActive = %v, %v", changed, err)
	}
	if now, _ := snapshotActive(t, x); now != prev {
		t.Fatalf("a failed replacement changed the activation record: %s", now)
	}
	if got := dirDigest(t, x.H.Path("runtime", stale)); got != staleBefore {
		t.Fatal("a failed replacement modified the previous runtime")
	}
	if got := dirDigest(t, x.H.Path("models")); got != modelsBefore {
		t.Fatal("a failed replacement modified the model")
	}
	if _, err := os.Stat(x.H.Path("runtime", spec.ID())); !os.IsNotExist(err) {
		t.Fatalf("a runtime that failed to materialize was published: %v", err)
	}
	// The previous runtime is still a valid, loadable activation.
	if _, _, _, err := x.H.LoadActive(); err != nil {
		t.Fatalf("the previous activation is no longer loadable: %v", err)
	}
	if r, ok := setup.AssessHome(x.H); !ok || r.ActiveRuntime != stale || !r.Needs() {
		t.Fatalf("the failure is not visible as a pending reconciliation: %+v", r)
	}

	// Recoverable: once uv works the same call completes the replacement.
	x.FailUV("")
	if _, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, nil); err != nil || !changed {
		t.Fatalf("retry = %v, %v", changed, err)
	}
	if _, a := snapshotActive(t, x); a.Runtime != spec.ID() {
		t.Fatalf("active %+v", a)
	}
}

// Immutable model and variant artifacts survive a runtime replacement
// untouched and stay selected; nothing is downloaded or rebuilt.
func TestRuntimeReplacementReusesModelAndVariantArtifacts(t *testing.T) {
	x := setup.NewFake(t)
	m := x.AddClef()
	if err := x.Run("cpu", setup.ClefFlash); err != nil {
		t.Fatal(err)
	}
	v := buildVariant(t, x.H)
	certify(t, x.H, m, v, true, time.Now())
	if _, err := setup.ActivateTarget(x.H, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	stale := x.Stale("cpu")
	modelsBefore, variantsBefore := dirDigest(t, x.H.Path("models")), dirDigest(t, x.H.Path("variants"))
	req0 := x.Requests()
	spec := requiredRuntime(t, "cpu")

	if _, changed, err := setup.ReconcileActive(t.Context(), x.H, io.Discard, nil); err != nil || !changed {
		t.Fatalf("ReconcileActive = %v, %v", changed, err)
	}
	a := activeRecord(t, x.H)
	if a.Runtime != spec.ID() || a.ModelID != setup.ClefFlash || a.Variant != v.ID || a.Experimental {
		t.Fatalf("active %+v (replaced runtime %s)", a, stale)
	}
	if dirDigest(t, x.H.Path("models")) != modelsBefore {
		t.Fatal("the model artifact changed across the runtime replacement")
	}
	if dirDigest(t, x.H.Path("variants")) != variantsBefore {
		t.Fatal("the tuned variant artifact changed across the runtime replacement")
	}
	if x.Requests() != req0 {
		t.Fatalf("%d artifacts were downloaded", x.Requests()-req0)
	}
	if got, ok, err := x.H.LoadVariant(a); err != nil || !ok || got.ID != v.ID {
		t.Fatalf("LoadVariant %v %v %v", got.ID, ok, err)
	}
}

// A worker-only application update leaves a physical benchmark environment
// materially the same: same dependency runtime, same model artifact, same
// variant artifact. Only the worker identity moves, and the evidence says so.
func TestWorkerOnlyUpdatePreservesBenchmarkEnvironment(t *testing.T) {
	x := setup.NewFake(t)
	m := x.AddClef()
	if err := x.Run("cpu", setup.ClefFlash); err != nil {
		t.Fatal(err)
	}
	v := buildVariant(t, x.H)
	certify(t, x.H, m, v, true, time.Now())
	if _, err := setup.ActivateTarget(x.H, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	launch := func() (server.Runtime, string) {
		cfg, rt, err := server.WorkerConfig(x.H, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		if err := cfg.Preflight(); err != nil {
			t.Fatal(err)
		}
		return rt, cfg.Args[4]
	}
	before, scriptBefore := launch()
	activeBefore := rawActive(x.H)
	runtimesBefore, modelsBefore, variantsBefore := dirDigest(t, x.H.Path("runtime")), dirDigest(t, x.H.Path("models")), dirDigest(t, x.H.Path("variants"))
	uv0, req0 := x.UVCalls(), x.Requests()

	swapWorker(t) // the application update

	after, scriptAfter := launch()
	if after.Runtime != before.Runtime || after.RuntimeDirectory != before.RuntimeDirectory || after.ModelID != before.ModelID ||
		after.Model != before.Model || after.Variant == nil || before.Variant == nil || *after.Variant != *before.Variant {
		t.Fatalf("the benchmark environment changed:\n before %+v\n after  %+v", before, after)
	}
	if before.Worker == nil || after.Worker == nil || after.Worker.SHA256 == before.Worker.SHA256 || after.Worker.ABI != before.Worker.ABI {
		t.Fatalf("worker identity before %+v after %+v", before.Worker, after.Worker)
	}
	if scriptBefore == scriptAfter || filepath.Base(scriptAfter) != "hachidori_worker.py" {
		t.Fatalf("the new worker was not delivered separately: %s -> %s", scriptBefore, scriptAfter)
	}
	if rawActive(x.H) != activeBefore || dirDigest(t, x.H.Path("runtime")) != runtimesBefore ||
		dirDigest(t, x.H.Path("models")) != modelsBefore || dirDigest(t, x.H.Path("variants")) != variantsBefore {
		t.Fatal("a worker-only update changed the runtime, model, variant or activation")
	}
	if x.UVCalls() != uv0 || x.Requests() != req0 {
		t.Fatalf("a worker-only update rematerialized or downloaded: uv %d->%d requests %d->%d", uv0, x.UVCalls(), req0, x.Requests())
	}

	// Diagnostics distinguish the two identities.
	f, _ := diagnostics.Collect(diagnostics.Source{Home: x.H.Root, Status: server.Status{Runtime: after}})
	b, _ := json.Marshal(f)
	if f.Runtime.Runtime != after.Runtime || f.Runtime.WorkerSHA256 != after.Worker.SHA256 || f.Runtime.WorkerSHA256 == f.Runtime.Runtime || f.Runtime.WorkerABI != after.Worker.ABI {
		t.Fatalf("diagnostics %s", b)
	}
	inv := setup.Inspect(x.H, false)
	if inv.Worker.SHA256 != after.Worker.SHA256 || inv.Reconciliation == nil || inv.Reconciliation.ActiveEnvironment != after.Runtime {
		t.Fatalf("inventory %+v %+v", inv.Worker, inv.Reconciliation)
	}
}
