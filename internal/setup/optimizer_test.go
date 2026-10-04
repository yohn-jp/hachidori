package setup

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	optpy "github.com/yohn-jp/hachidori/internal/optimize/py"
)

// The optimizer runtime is a separate, deterministic, immutable runtime under
// the same home: its own Runtime Spec (role optimizer, engine pins), its own
// locked environment and script, never the serving runtime and never
// activated.
func TestOptimizerRuntimeIsSeparateAndDeterministic(t *testing.T) {
	f := newFixture(t)
	serving, err := Desired("cpu")
	if err != nil {
		t.Fatal(err)
	}
	spec, err := DesiredOptimizer("cpu")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Role != "optimizer" || spec.ID() == serving.ID() || !strings.HasPrefix(spec.ID(), "optimizer-cpu-") ||
		spec.Provider != "llmcompressor==0.14.0,compressed-tensors==0.19.0" || spec.WorkerABI != home.WorkerABIOptimizer ||
		spec.Project == serving.Project || spec.Lock == serving.Lock || spec.WorkerABI == serving.WorkerABI {
		t.Fatalf("optimizer spec %+v vs serving %+v", spec, serving)
	}
	if again, _ := DesiredOptimizer("cpu"); again.ID() != spec.ID() {
		t.Fatal("the optimizer identity is not deterministic")
	}
	// The serving runtime does not carry the compression stack.
	if serving.Provides("llmcompressor") || strings.Contains(serving.Provider, "llmcompressor") {
		t.Fatalf("serving runtime carries the optimizer: %s", serving.Provider)
	}
	// The guidance names only a command that exists (`variant` has no
	// `prepare` subcommand).
	if _, err := FindOptimizer(f.H, "cpu"); err == nil || !strings.Contains(err.Error(), "not materialized") ||
		!strings.Contains(err.Error(), "hachidori variant optimize") || strings.Contains(err.Error(), "variant prepare") {
		t.Fatalf("FindOptimizer before materialization: %v", err)
	}

	var log strings.Builder
	rt, err := EnsureOptimizer(f.H, "cpu", &log, nil)
	if err != nil {
		t.Fatalf("EnsureOptimizer: %v\n%s", err, log.String())
	}
	if rt.ID != spec.ID() || rt.Manifest.Spec != spec || !strings.HasSuffix(filepath.ToSlash(rt.Script), "workers/"+digest(optpy.Script)+"/hachidori_optimizer.py") {
		t.Fatalf("runtime %+v", rt)
	}
	if got, _ := FileSHA256(rt.Script); got != digest(optpy.Script) {
		t.Fatal("the optimizer script is not the embedded one")
	}
	pj, err := os.ReadFile(filepath.Join(rt.Dir, "spec", "pyproject.toml"))
	if err != nil || !strings.Contains(string(pj), "hachidori-optimizer") || !strings.Contains(string(pj), "llmcompressor==0.14.0") {
		t.Fatalf("optimizer project: %v", err)
	}
	// The locked sync used the cpu extra, offline from the model's point of view.
	var synced bool
	for _, c := range f.calls() {
		if len(c.Args) > 0 && c.Args[0] == "sync" && strings.Contains(strings.Join(c.Args, " "), "--extra cpu") {
			synced = true
		}
	}
	if !synced {
		t.Fatalf("uv calls %+v", f.calls())
	}
	calls := len(f.calls())
	if _, err := EnsureOptimizer(f.H, "cpu", io.Discard, nil); err != nil || len(f.calls()) != calls {
		t.Fatalf("a verified optimizer runtime was rebuilt: %v (%d -> %d uv calls)", err, calls, len(f.calls()))
	}
	if _, err := FindOptimizer(f.H, "cpu"); err != nil {
		t.Fatal(err)
	}

	// It is inventory-visible but never a serving runtime.
	inv := Inspect(f.H, false)
	if inv.Optimizer == nil || !inv.Optimizer.Materialized || inv.Optimizer.ID != spec.ID() {
		t.Fatalf("optimizer inventory %+v", inv.Optimizer)
	}
	for _, r := range inv.Runtimes {
		if r.ID == spec.ID() {
			t.Fatal("the optimizer runtime is listed as a serving runtime")
		}
	}
	if err := Remove(f.H, KindRuntime, spec.ID(), nil); err == nil {
		t.Fatal("the optimizer runtime was removable as a serving runtime")
	}
	if _, err := Activate(f.H, spec.Flavor, "", io.Discard, nil); err == nil {
		// Activating cpu needs the serving runtime, which was never materialized here.
		t.Fatal("activation succeeded without a serving runtime")
	}

	// The delivered optimizer script is content addressed: a tampered copy is
	// replaced by the exact script of this build, and the runtime is untouched.
	if err := os.WriteFile(rt.Script, []byte("import os"), 0o644); err != nil {
		t.Fatal(err)
	}
	healed, err := FindOptimizer(f.H, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := FileSHA256(healed.Script); got != digest(optpy.Script) {
		t.Fatal("a tampered optimizer script was launched")
	}

	// A damaged optimizer runtime is refused and never modified in place.
	if err := os.Remove(rt.Python); err != nil {
		t.Fatal(err)
	}
	if _, err := EnsureOptimizer(f.H, "cpu", io.Discard, nil); err == nil || !strings.Contains(err.Error(), "never modified in place") {
		t.Fatalf("a damaged optimizer runtime was reused: %v", err)
	}
}

// Both runtime specs pin the exact packages of their locks: the serving
// runtime gains the System One loading dependencies and nothing else, and the
// lock digests are part of the identities.
func TestRuntimeSpecsPinTheirProjects(t *testing.T) {
	serving, _ := Desired("cpu")
	if serving.Project != digest(specFile("pyproject.toml")) || serving.Lock != digest(specFile("uv.lock")) {
		t.Fatal("serving spec does not pin its project")
	}
	pj := string(specFile("pyproject.toml"))
	for _, want := range []string{"laya==0.3.21", "opendecider==0.3.0", "transformers==5.17.0", "accelerate==1.15.0", "compressed-tensors==0.19.0"} {
		if !strings.Contains(pj, want) {
			t.Errorf("serving project lacks %s", want)
		}
	}
	if strings.Contains(pj, "llmcompressor") {
		t.Error("the serving project carries the optimizer-only compression engine")
	}
	opt := string(kindOf(home.RuntimeSpec{Role: home.RoleOptimizer}).file("pyproject.toml"))
	if !strings.Contains(opt, "llmcompressor==0.14.0") || strings.Contains(opt, "laya") {
		t.Errorf("optimizer project:\n%s", opt)
	}
}

// The optimizer runtime has two flavors of the one locked optimizer project,
// selected by a concrete device: cpu and cuda (the cu128 extra). They are
// distinct runtime identities that differ in the torch build and flavor and in
// nothing else, each materializes through its own extra, and neither is ever
// substituted for the other. There is no automatic device.
func TestOptimizerRuntimeFlavorsAreDistinctIdentities(t *testing.T) {
	f := newFixture(t)
	cpu, err := DesiredOptimizer("cpu")
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := DesiredOptimizer("cuda")
	if err != nil {
		t.Fatal(err)
	}
	if cuda.Role != "optimizer" || cuda.Flavor != "cu128" || cuda.Torch != "2.11.0+cu128" || !strings.HasPrefix(cuda.ID(), "optimizer-cu128-") ||
		cpu.Flavor != "cpu" || cpu.Torch != "2.11.0+cpu" || !strings.HasPrefix(cpu.ID(), "optimizer-cpu-") || cuda.ID() == cpu.ID() {
		t.Fatalf("cpu %+v cuda %+v", cpu, cuda)
	}
	// One locked project: only the torch build differs.
	if d := cuda.Differences(cpu); strings.Join(d, ",") != "torch,flavor" {
		t.Fatalf("the flavors differ in %v, want only torch and flavor", d)
	}
	k := kindOf(home.RuntimeSpec{Role: home.RoleOptimizer})
	if cuda.Project != digest(k.file("pyproject.toml")) || cuda.Lock != digest(k.file("uv.lock")) || cuda.Provider != cpu.Provider || cuda.WorkerABI != cpu.WorkerABI {
		t.Fatal("the cuda optimizer is not the locked optimizer project")
	}
	if again, _ := DesiredOptimizer("cuda"); again.ID() != cuda.ID() {
		t.Fatal("the cuda optimizer identity is not deterministic")
	}
	for _, device := range []string{"", "auto", "gpu", "CUDA"} {
		if _, err := DesiredOptimizer(device); err == nil {
			t.Errorf("DesiredOptimizer(%q) resolved a runtime", device)
		}
	}

	// Device resolution is explicit: the empty request is the cpu, auto does not exist.
	for in, want := range map[string]string{"": "cpu", "cpu": "cpu", "cuda": "cuda"} {
		if got, err := ResolveOptimizerDevice(in); err != nil || got != want {
			t.Errorf("ResolveOptimizerDevice(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, in := range []string{"auto", "gpu", "cu128", "CUDA"} {
		if got, err := ResolveOptimizerDevice(in); err == nil {
			t.Errorf("ResolveOptimizerDevice(%q) = %q, want a refusal", in, got)
		}
	}

	// Each flavor materializes from its own extra; the cpu runtime is never reused for cuda.
	if _, err := FindOptimizer(f.H, "cuda"); err == nil || !strings.Contains(err.Error(), "--device cuda") {
		t.Fatalf("FindOptimizer(cuda) before materialization: %v", err)
	}
	var log strings.Builder
	if _, err := EnsureOptimizer(f.H, "cpu", &log, nil); err != nil {
		t.Fatalf("cpu: %v\n%s", err, log.String())
	}
	if _, err := FindOptimizer(f.H, "cuda"); err == nil {
		t.Fatal("the cpu optimizer runtime satisfied a cuda request")
	}
	rt, err := EnsureOptimizer(f.H, "cuda", &log, nil)
	if err != nil {
		t.Fatalf("cuda: %v\n%s", err, log.String())
	}
	if rt.ID != cuda.ID() || rt.Manifest.Spec != cuda || rt.Manifest.Spec.Flavor != "cu128" {
		t.Fatalf("cuda runtime %+v", rt)
	}
	extras := map[string]bool{}
	for _, c := range f.calls() {
		if len(c.Args) > 0 && c.Args[0] == "sync" {
			extras[flagValue(c.Args, "--extra")] = true
		}
	}
	if !extras["cpu"] || !extras["cu128"] {
		t.Fatalf("locked syncs used extras %v", extras)
	}
}
