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
	spec, err := DesiredOptimizer()
	if err != nil {
		t.Fatal(err)
	}
	if spec.Role != "optimizer" || spec.ID() == serving.ID() || !strings.HasPrefix(spec.ID(), "optimizer-cpu-") ||
		spec.Provider != "llmcompressor==0.14.0,compressed-tensors==0.19.0" || spec.Worker != digest(optpy.Script) ||
		spec.Project == serving.Project || spec.Lock == serving.Lock || spec.Worker == serving.Worker {
		t.Fatalf("optimizer spec %+v vs serving %+v", spec, serving)
	}
	if again, _ := DesiredOptimizer(); again.ID() != spec.ID() {
		t.Fatal("the optimizer identity is not deterministic")
	}
	// The serving runtime does not carry the compression stack.
	if serving.Provides("llmcompressor") || strings.Contains(serving.Provider, "llmcompressor") {
		t.Fatalf("serving runtime carries the optimizer: %s", serving.Provider)
	}
	// The guidance names only a command that exists (`variant` has no
	// `prepare` subcommand).
	if _, err := FindOptimizer(f.H); err == nil || !strings.Contains(err.Error(), "not materialized") ||
		!strings.Contains(err.Error(), "hachidori variant optimize") || strings.Contains(err.Error(), "variant prepare") {
		t.Fatalf("FindOptimizer before materialization: %v", err)
	}

	var log strings.Builder
	rt, err := EnsureOptimizer(f.H, &log, nil)
	if err != nil {
		t.Fatalf("EnsureOptimizer: %v\n%s", err, log.String())
	}
	if rt.ID != spec.ID() || rt.Manifest.Spec != spec || !strings.HasSuffix(filepath.ToSlash(rt.Script), "worker/hachidori_optimizer.py") {
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
	if _, err := EnsureOptimizer(f.H, io.Discard, nil); err != nil || len(f.calls()) != calls {
		t.Fatalf("a verified optimizer runtime was rebuilt: %v (%d -> %d uv calls)", err, calls, len(f.calls()))
	}
	if _, err := FindOptimizer(f.H); err != nil {
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

	// A tampered optimizer runtime is refused and never modified in place.
	if err := os.WriteFile(rt.Script, []byte("import os"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := EnsureOptimizer(f.H, io.Discard, nil); err == nil || !strings.Contains(err.Error(), "never modified in place") {
		t.Fatalf("a tampered optimizer runtime was reused: %v", err)
	}
	if _, err := FindOptimizer(f.H); err == nil {
		t.Fatal("FindOptimizer accepted a tampered script")
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
