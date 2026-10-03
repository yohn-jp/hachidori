package setup

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	optpy "github.com/yohn-jp/hachidori/internal/optimize/py"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

// A read-only interpreter probe is safely bounded; timing out must leave the
// active record and runtime reusable. Materialization/activation have no such
// deadline because interrupting their external work is not proven safe.
func TestRuntimeProbeDeadlinePreservesActivation(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	spec, _ := Desired("cpu")
	dir := f.H.Path("runtime", spec.ID())
	active := f.H.Path("state", "active-runtime.json")
	before, err := os.ReadFile(active)
	if err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(dir, "env", "stall-probe")
	if err := os.WriteFile(marker, nil, 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	started := time.Now()
	_, err = verifyRuntimeContext(ctx, f.H, dir, spec)
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(started) > 7*time.Second {
		t.Fatalf("unbounded or undiagnosable probe: %v", err)
	}
	after, err := os.ReadFile(active)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("timeout altered activation")
	}
	if err := os.Remove(marker); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyRuntime(f.H, dir, spec); err != nil {
		t.Fatalf("runtime no longer reusable: %v", err)
	}
}

// swapScripts replaces the embedded worker and optimizer sources for the
// duration of the test, as an application update that changed only them.
func swapScripts(t *testing.T) {
	t.Helper()
	oldW, oldO := py.Script, optpy.Script
	t.Cleanup(func() { py.Script, optpy.Script = oldW, oldO })
	py.Script = append(append([]byte{}, oldW...), []byte("\n# worker-only change\n")...)
	optpy.Script = append(append([]byte{}, oldO...), []byte("\n# optimizer script-only change\n")...)
}

// A worker-only change leaves every dependency runtime identity as it was; the
// worker identity changes and stays observable.
func TestWorkerOnlyChangeDoesNotChangeRuntimeIdentity(t *testing.T) {
	var before []string
	for _, d := range Devices {
		s, err := Desired(d)
		if err != nil {
			t.Fatal(err)
		}
		before = append(before, s.ID())
	}
	opt, _ := DesiredOptimizer()
	before = append(before, opt.ID())
	workerBefore := BuildWorker()

	swapScripts(t)

	var after []string
	for _, d := range Devices {
		s, _ := Desired(d)
		after = append(after, s.ID())
	}
	opt, _ = DesiredOptimizer()
	after = append(after, opt.ID())
	for i := range before {
		if before[i] != after[i] {
			t.Errorf("runtime identity %d changed with the worker script: %s -> %s", i, before[i], after[i])
		}
	}
	if w := BuildWorker(); w.SHA256 == workerBefore.SHA256 || w.ABI != workerBefore.ABI || w.SHA256 != WorkerDigest() {
		t.Fatalf("worker identity %+v -> %+v: the digest must change and the ABI must not", workerBefore, w)
	}
}

// Every material environment input changes the dependency runtime identity.
func TestMaterialInputsChangeRuntimeIdentity(t *testing.T) {
	base, err := desiredFor("cuda", "linux/amd64")
	if err != nil {
		t.Fatal(err)
	}
	win, err := desiredFor("cuda", "windows/amd64")
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := desiredFor("cpu", "linux/amd64")
	if err != nil {
		t.Fatal(err)
	}
	if base.ID() == win.ID() || base.ID() == cpu.ID() {
		t.Fatalf("platform/flavor do not change the identity: %s %s %s", base.ID(), win.ID(), cpu.ID())
	}
	for name, mut := range map[string]func(*home.RuntimeSpec){
		"pyproject":                  func(s *home.RuntimeSpec) { s.Project = digest([]byte("changed pyproject.toml")) },
		"uv.lock":                    func(s *home.RuntimeSpec) { s.Lock = digest([]byte("changed uv.lock")) },
		"python":                     func(s *home.RuntimeSpec) { s.Python = "3.12.12" },
		"torch":                      func(s *home.RuntimeSpec) { s.Torch = "2.12.0+cu128" },
		"transformers/provider pins": func(s *home.RuntimeSpec) { s.Provider += ",extra==1" },
		"uv":                         func(s *home.RuntimeSpec) { s.UV = "9.9.9" },
		"worker ABI":                 func(s *home.RuntimeSpec) { s.WorkerABI = "hachidori.worker-runtime/2" },
	} {
		s := base
		mut(&s)
		if s.ID() == base.ID() {
			t.Errorf("changing %s keeps the runtime identity %s", name, base.ID())
		}
		if len(base.Differences(s)) == 0 {
			t.Errorf("changing %s is not reported as a difference", name)
		}
	}
	if len(base.Differences(base)) != 0 {
		t.Fatal("a spec differs from itself")
	}
	// The pinned embedded project files are what the identity digests.
	if base.Project != digest(specFile("pyproject.toml")) || base.Lock != digest(specFile("uv.lock")) {
		t.Fatal("the serving identity is not derived from the embedded uv project")
	}
}

func TestCurrentRuntimeAssessmentSeparatesEnvironmentFromWorker(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	var a home.Active
	var rm home.RuntimeManifest
	if err := home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	if err := home.ReadJSON(f.H.Path("runtime", a.Runtime, "manifest.json"), &rm); err != nil {
		t.Fatal(err)
	}
	before, err := Assess(a, rm)
	if err != nil || before.State != CompatCurrent || before.Needs() {
		t.Fatalf("a freshly materialized runtime: %+v %v", before, err)
	}
	// The runtime carries no worker at all.
	if _, err := os.Stat(f.H.Path("runtime", a.Runtime, "worker")); !os.IsNotExist(err) {
		t.Fatalf("the runtime directory carries a worker: %v", err)
	}
	if len(rm.Worker) != 0 || rm.Spec.Schema != home.SpecSchema {
		t.Fatalf("manifest %+v", rm)
	}

	swapScripts(t)
	after, err := Assess(a, rm)
	if err != nil || after.State != CompatCurrent {
		t.Fatalf("a worker-only update made the runtime %q: %+v %v", after.State, after, err)
	}
	if after.ActiveEnvironment != before.ActiveEnvironment || after.RequiredEnvironment != before.RequiredEnvironment ||
		after.ActiveRuntime != before.ActiveRuntime {
		t.Fatalf("dependency runtime identity moved with the worker: %+v -> %+v", before, after)
	}
	if after.Worker.SHA256 == before.Worker.SHA256 {
		t.Fatalf("the worker identity did not move: %+v", after.Worker)
	}
	if err := CheckRuntimeCompatibility(a, rm); err != nil {
		t.Fatalf("a compatible new worker is refused: %v", err)
	}
}

// A runtime derived for another worker ABI is rejected explicitly, by the ABI
// and not by comparing worker source.
func TestIncompatibleWorkerABIIsRejectedExplicitly(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	var a home.Active
	var rm home.RuntimeManifest
	home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a)
	home.ReadJSON(f.H.Path("runtime", a.Runtime, "manifest.json"), &rm)

	rm.Spec.WorkerABI = "hachidori.worker-runtime/0"
	rm.Identity = rm.Spec.ID()
	a.Runtime = rm.Identity
	r, err := Assess(a, rm)
	if err != nil || r.State != CompatIncompatible || !r.Needs() {
		t.Fatalf("assessment %+v %v", r, err)
	}
	err = CheckRuntimeCompatibility(a, rm)
	if !errors.Is(err, ErrWorkerABI) || errors.Is(err, ErrRuntimeStale) {
		t.Fatalf("err = %v, want ErrWorkerABI", err)
	}
	for _, want := range []string{"hachidori.worker-runtime/0", home.WorkerABIServing, "hachidori setup --device cpu"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error lacks %q: %v", want, err)
		}
	}
	// The ABI is the only thing separating it: the same runtime with the
	// required ABI is current, whatever the worker source is.
	rm.Spec.WorkerABI = home.WorkerABIServing
	rm.Identity = rm.Spec.ID()
	a.Runtime = rm.Identity
	swapScripts(t)
	if err := CheckRuntimeCompatibility(a, rm); err != nil {
		t.Fatal(err)
	}
}

// A worker is delivered content addressed, beside the runtimes: delivering it
// again is a no-op, a changed worker lands in its own directory and a damaged
// copy is replaced.
func TestWorkerDeliveryIsContentAddressedAndLeavesRuntimesAlone(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	first, err := DeliverWorker(h)
	if err != nil {
		t.Fatal(err)
	}
	if first.Path != filepath.Join(h.Root, "workers", WorkerDigest(), "hachidori_worker.py") || first.SHA256 != WorkerDigest() || first.ABI != home.WorkerABIServing {
		t.Fatalf("delivery %+v", first)
	}
	st1, _ := os.Stat(first.Path)
	again, err := DeliverWorker(h)
	if err != nil || again != first {
		t.Fatalf("redelivery %+v %v", again, err)
	}
	if st2, _ := os.Stat(first.Path); !st2.ModTime().Equal(st1.ModTime()) {
		t.Fatal("an unchanged worker was rewritten")
	}
	os.WriteFile(first.Path, []byte("damaged"), 0o644)
	if healed, err := DeliverWorker(h); err != nil || healed != first {
		t.Fatalf("heal %+v %v", healed, err)
	}
	if got, _ := FileSHA256(first.Path); got != WorkerDigest() {
		t.Fatal("a damaged worker was not replaced")
	}

	swapScripts(t)
	next, err := DeliverWorker(h)
	if err != nil || next.SHA256 == first.SHA256 || filepath.Dir(next.Path) == filepath.Dir(first.Path) {
		t.Fatalf("the changed worker was not delivered beside the first: %+v %v", next, err)
	}
	if got, _ := FileSHA256(first.Path); got != first.SHA256 {
		t.Fatal("delivering a new worker changed the previous one")
	}
	if _, err := os.Stat(h.Path("runtime")); !os.IsNotExist(err) {
		t.Fatalf("delivering a worker touched runtime/: %v", err)
	}
}
