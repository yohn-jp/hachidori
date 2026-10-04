package optimize_test

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// A cuda build records the concrete optimizer device in its manifest and in its
// build identity: it is a different contract and a different variant from the
// cpu build of the same source and recipe, so neither is mistaken for, reused
// as, or reproduced from the other.
func TestCUDABuildIsRecordedAndNeverConflatedWithCPU(t *testing.T) {
	h, _ := source(t)
	r := &optimizetest.Runner{}

	cpu, err := build(t, h, r, optimize.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if cpu.Variant.Optimizer.Device != "cpu" {
		t.Fatalf("the default optimizer device is %q, want cpu", cpu.Variant.Optimizer.Device)
	}
	cuda, err := build(t, h, r, optimize.Request{Device: "cuda"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if cuda.Existing {
		t.Fatal("a cuda request resolved to the existing cpu variant; nothing ran")
	}
	if cuda.Variant.Optimizer.Device != "cuda" || cuda.Variant.ID == cpu.Variant.ID || cuda.Variant.BuildID == cpu.Variant.BuildID {
		t.Fatalf("cpu %+v\ncuda %+v", cpu.Variant.Optimizer, cuda.Variant.Optimizer)
	}
	if got := strings.Join(r.Devices, ","); got != "cpu,cuda" {
		t.Fatalf("the optimizer was started with devices %q, want cpu then cuda", got)
	}
	// What is published says it: read back from disk, valid, device and runtime recorded.
	onDisk, err := home.ReadVariant(cuda.Dir)
	if err != nil || onDisk.Validate() != nil || onDisk.Optimizer.Device != "cuda" || onDisk.Optimizer.Runtime != "optimizer-test" {
		t.Fatalf("published cuda variant: %+v %v", onDisk.Optimizer, err)
	}
	if names := variantDirs(t, h); len(names) != 2 {
		t.Fatalf("variants %v, want the cpu and the cuda build side by side", names)
	}
	// Each contract resolves only to its own build and reproduces only against it.
	again, err := build(t, h, r, optimize.Request{Device: "cuda"}, nil)
	if err != nil || !again.Existing || again.Variant.ID != cuda.Variant.ID {
		t.Fatalf("cuda rebuild: existing=%v id=%s err=%v", again.Existing, again.Variant.ID, err)
	}
	if rep, err := build(t, h, r, optimize.Request{Device: "cuda", Reproduce: true}, nil); err != nil || !rep.Reproduced || rep.Variant.ID != cuda.Variant.ID {
		t.Fatalf("cuda reproduce: %+v %v", rep, err)
	}
	if got := strings.Join(r.Devices, ","); got != "cpu,cuda,cuda" {
		t.Fatalf("devices after the reproduction %q", got)
	}
}

// Once a build resolved to cuda, an accelerator failure surfaces as a typed
// failure of that build: the optimizer is started once, on cuda, never again
// on the cpu, and nothing is published.
func TestCUDAUnavailableFailsTypedWithoutCPUFallback(t *testing.T) {
	h, _ := source(t)
	r := &optimizetest.Runner{Fail: "cuda-unavailable"}
	_, err := build(t, h, r, optimize.Request{Device: "cuda"}, nil)
	var fatal *optimize.FatalError
	if !errors.As(err, &fatal) || fatal.Class != "cuda_unavailable" || !strings.Contains(err.Error(), "cuda_unavailable") {
		t.Fatalf("err = %v, want a typed cuda_unavailable failure", err)
	}
	if r.Calls != 1 || strings.Join(r.Devices, ",") != "cuda" {
		t.Fatalf("the optimizer was started %d times on %v; a cuda build must run once, on cuda", r.Calls, r.Devices)
	}
	if names := variantDirs(t, h); len(names) != 0 {
		t.Fatalf("a failed cuda build left %v", names)
	}
	// The same fake on the cpu is unaffected: the failure is the device's, not the recipe's.
	if _, err := build(t, h, r, optimize.Request{Device: "cpu"}, nil); err != nil {
		t.Fatalf("cpu build: %v", err)
	}
}

// The device is explicit: there is no automatic choice, and a typo is refused
// before anything runs rather than mapped to the cpu.
func TestOptimizerDeviceIsExplicit(t *testing.T) {
	h, _ := source(t)
	for _, device := range []string{"auto", "gpu", "CUDA", "cu128"} {
		r := &optimizetest.Runner{}
		if _, err := build(t, h, r, optimize.Request{Device: device}, nil); err == nil || !strings.Contains(err.Error(), "cpu or cuda") || r.Calls != 0 {
			t.Errorf("device %q: err=%v calls=%d", device, err, r.Calls)
		}
	}
}

// The optimizer reports the device its backend resolved before it loads
// anything; a build whose optimizer ran on another device than the one it
// resolved to is refused, never recorded under the requested device.
func TestBuildRefusesAnOptimizerThatRanOnAnotherDevice(t *testing.T) {
	h, _ := source(t)
	ran := runnerFunc(func(_ context.Context, args []string, events, _ io.Writer) error {
		io.WriteString(events, `{"event":"hello","protocol":"hachidori-optimizer.v1"}`+"\n")
		io.WriteString(events, `{"event":"engine","engine":"llmcompressor","version":"0.14.0","versions":{}}`+"\n")
		io.WriteString(events, `{"event":"device","device":"cpu","backend":"cpu"}`+"\n")
		io.WriteString(events, `{"event":"done"}`+"\n")
		return nil
	})
	_, err := build(t, h, ran, optimize.Request{Device: "cuda"}, nil)
	if err == nil || !strings.Contains(err.Error(), `reported device "cpu"`) {
		t.Fatalf("err = %v, want a refusal of a cpu run for a cuda build", err)
	}
	if names := variantDirs(t, h); len(names) != 0 {
		t.Fatalf("left %v", names)
	}
}

// materializedOptimizer writes the manifest of a materialized optimizer runtime
// of device, with the pinned engine installed.
func materializedOptimizer(t *testing.T, h home.Home, device string) home.RuntimeSpec {
	t.Helper()
	spec, err := setup.DesiredOptimizer(device)
	if err != nil {
		t.Fatal(err)
	}
	dir := h.Path("runtime", spec.ID())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	m := home.RuntimeManifest{Identity: spec.ID(), Spec: spec, PythonVersion: spec.Python, PythonRelPath: "env/bin/python",
		Installed: []string{"torch==" + spec.Torch, "transformers==5.17.0", "compressed-tensors==0.19.0", "llmcompressor==" + setup.OptimizerEngineVersion}}
	if err := home.WriteJSON(filepath.Join(dir, "manifest.json"), m); err != nil {
		t.Fatal(err)
	}
	return spec
}

func preOptimize(t *testing.T, h home.Home, device string) setup.PreflightReport {
	t.Helper()
	deps := optimize.PreflightDeps{Host: host(1<<40, true, ram(64*gib, 60*gib)), Now: func() time.Time { return time.Unix(0, 0) }}
	return optimize.Preflight(context.Background(), h,
		optimize.PreflightRequest{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16, Device: device, Verified: true}, deps, nil)
}

// The CUDA optimizer preflight uses the accelerator probe against the cuda
// optimizer runtime, never passes without a usable pinned CUDA torch, records
// the device name, CUDA version and VRAM it observed, and describes VRAM as the
// accelerator working set: the BF16 source (larger than the card) stays in host
// RAM and is not required to fit in VRAM.
func TestCUDAOptimizerPreflight(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	// A source larger than a 12 GiB card, as the 9B BF16 Clef-Flash is (sparse; never read).
	srcWeights := filepath.Join(h.Path("models", filepath.FromSlash(setup.ModelDirName(m))), "model.safetensors")
	if err := os.Truncate(srcWeights, 18*gib); err != nil {
		t.Skipf("cannot create a sparse file: %v", err)
	}
	cuda := materializedOptimizer(t, h, "cuda")
	var probed string
	accel := func(f setup.AcceleratorFacts, err error) optimize.PreflightDeps {
		return optimize.PreflightDeps{Accelerator: func(_ context.Context, _ home.Home, python string) (setup.AcceleratorFacts, error) {
			probed = python
			return f, err
		}}
	}
	run := func(f setup.AcceleratorFacts, err error) setup.PreflightReport {
		deps := accel(f, err)
		deps.Host, deps.Now = host(1<<40, true, ram(64*gib, 60*gib)), func() time.Time { return time.Unix(0, 0) }
		return optimize.Preflight(context.Background(), h,
			optimize.PreflightRequest{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16, Device: "cuda", Verified: true}, deps, nil)
	}

	good := setup.AcceleratorFacts{Torch: cuda.Torch, TorchCUDA: "12.8", CUDAAvailable: true, DeviceCount: 1, DeviceName: "NVIDIA GeForce RTX 3060",
		Capability: []int{8, 6}, VRAMTotal: 12 * gib, VRAMFree: 11 * gib}
	r := run(good, nil)
	if r.Blocked() {
		t.Fatalf("a 12 GiB card with an 18 GiB source was refused: %+v", r.Blockers())
	}
	if !strings.Contains(probed, cuda.ID()) {
		t.Fatalf("the accelerator was probed with %q, not the cuda optimizer runtime %s", probed, cuda.ID())
	}
	if f := find(t, r, "runtime.optimizer"); f.Status != setup.FindingPass || f.Facts["device"] != "cuda" || f.Facts["runtime"] != cuda.ID() || f.Facts["flavor"] != "cu128" {
		t.Fatalf("runtime.optimizer %s %+v", f.Status, f.Facts)
	}
	d := find(t, r, "accelerator.device")
	if d.Status != setup.FindingPass || d.Facts["device_name"] != "NVIDIA GeForce RTX 3060" || d.Facts["torch_cuda"] != "12.8" ||
		d.Facts["vram_total_bytes"] != uint64(12*gib) || d.Facts["vram_free_bytes"] != uint64(11*gib) {
		t.Fatalf("accelerator.device %s %+v", d.Status, d.Facts)
	}
	// VRAM describes the accelerator working set and never demands the source's size.
	v := find(t, r, "accelerator.vram")
	if v.Status != setup.FindingUnknown || v.Facts["source_weight_bytes"].(uint64) < 18*gib || !strings.Contains(v.Summary, "not required to fit in VRAM") {
		t.Fatalf("accelerator.vram %s %s %+v", v.Status, v.Summary, v.Facts)
	}
	// RAM still sizes the host-resident source.
	if mf := find(t, r, "memory.fit"); mf.Facts["lower_bound_bytes"].(uint64) < 18*gib {
		t.Fatalf("memory.fit %+v: the host-resident source working set is no longer the RAM lower bound", mf.Facts)
	}
	// The report is bound to the cuda optimizer runtime, not the cpu one.
	cpu, _ := setup.DesiredOptimizer("cpu")
	if r.Binding == nil || r.Binding.Device != "cuda" || r.Binding.Runtime != cuda.ID() || r.Binding.Runtime == cpu.ID() || r.Binding.RuntimeManifestSHA256 == "" {
		t.Fatalf("binding %+v", r.Binding)
	}

	// Unavailable CUDA is a blocker; the cpu is never substituted.
	for name, f := range map[string]setup.AcceleratorFacts{
		"no device":        {Torch: cuda.Torch, TorchCUDA: "12.8"},
		"cpu torch build":  {Torch: "2.11.0+cpu", CUDAAvailable: true, DeviceName: "x", VRAMTotal: gib},
		"not the pin":      {Torch: "2.12.0+cu128", TorchCUDA: "12.8", CUDAAvailable: true, DeviceName: "x", VRAMTotal: gib},
		"no device record": {Torch: cuda.Torch, TorchCUDA: "12.8", CUDAAvailable: true},
	} {
		r := run(f, nil)
		if f := find(t, r, "accelerator.device"); f.Status != setup.FindingBlocker || !r.Blocked() {
			t.Errorf("%s: accelerator.device = %s (%s)", name, f.Status, f.Summary)
		}
	}
	// A probe that cannot run is UNKNOWN, not a pass.
	if f := find(t, run(setup.AcceleratorFacts{}, errors.New("probe died")), "accelerator.device"); f.Status != setup.FindingUnknown {
		t.Fatalf("unobservable = %s", f.Status)
	}
}

// Without a materialized cuda optimizer runtime nothing can prove CUDA: the
// runtime is a warning (the build materializes it) and the accelerator stays
// UNKNOWN; the cpu optimizer runtime never stands in for it. The cpu request
// needs no accelerator, and an unknown device is a blocker.
func TestOptimizerPreflightDeviceSelection(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	materializedOptimizer(t, h, "cpu")

	r := preOptimize(t, h, "cuda")
	if f := find(t, r, "runtime.optimizer"); f.Status != setup.FindingWarning || f.Facts["device"] != "cuda" {
		t.Fatalf("cuda with only the cpu runtime: runtime.optimizer %s %+v", f.Status, f.Facts)
	}
	if f := find(t, r, "accelerator.device"); f.Status != setup.FindingUnknown {
		t.Fatalf("accelerator.device = %s without a cuda runtime to observe it with", f.Status)
	}

	cpu, _ := setup.DesiredOptimizer("cpu")
	for _, device := range []string{"", "cpu"} {
		r := preOptimize(t, h, device)
		if f := find(t, r, "runtime.optimizer"); f.Status != setup.FindingPass || f.Facts["runtime"] != cpu.ID() {
			t.Errorf("device %q: runtime.optimizer %s %+v", device, f.Status, f.Facts)
		}
		if f := find(t, r, "accelerator.device"); f.Status != setup.FindingPass || f.Facts["requested"] != "cpu" {
			t.Errorf("device %q: accelerator.device %s", device, f.Status)
		}
	}

	r = preOptimize(t, h, "auto")
	if f := find(t, r, "runtime.optimizer"); f.Status != setup.FindingBlocker || !r.Blocked() {
		t.Fatalf("device auto: runtime.optimizer %s", f.Status)
	}
}
