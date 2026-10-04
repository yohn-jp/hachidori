package optimize_test

import (
	"context"
	"encoding/json"
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

const gib = 1 << 30

// withIndex adds the checkpoint's weight map to the fixture source: one
// "<module>.weight" tensor per module of the synthetic graph plus a norm, as
// the real index does. skip drops modules from it.
func withIndex(t *testing.T, h home.Home, m home.ModelManifest, skip ...string) {
	t.Helper()
	wm := map[string]string{"model.language_model.norm.weight": "model.safetensors"}
	for _, mod := range optimizetest.Modules {
		wm[mod+".weight"] = "model.safetensors"
	}
	for _, s := range skip {
		delete(wm, s+".weight")
	}
	b, _ := json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": wm})
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if err := os.WriteFile(filepath.Join(dir, optimize.WeightMapFile), b, 0o644); err != nil {
		t.Fatal(err)
	}
	d, _ := setup.FileSHA256(filepath.Join(dir, optimize.WeightMapFile))
	m.Files[optimize.WeightMapFile] = d
	if err := home.WriteJSON(filepath.Join(dir, "hachidori-model.json"), m); err != nil {
		t.Fatal(err)
	}
}

func host(free uint64, freeOK bool, mem setup.Memory) *setup.Host {
	return &setup.Host{FreeDisk: func(string) (uint64, bool) { return free, freeOK }, Memory: func() setup.Memory { return mem }}
}

func ram(total, avail uint64) setup.Memory {
	return setup.Memory{Total: total, TotalKnown: true, Available: avail, AvailableKnown: true}
}

func pre(t *testing.T, h home.Home, req optimize.PreflightRequest, hst *setup.Host) setup.PreflightReport {
	t.Helper()
	if req.Kind == "" {
		req.Kind = setup.PreflightOptimize
	}
	if req.Model == "" {
		req.Model = setup.ClefFlash
	}
	if req.Recipe == "" && req.Kind == setup.PreflightOptimize {
		req.Recipe = optimize.RecipeClefFlashW4A16
	}
	return optimize.Preflight(context.Background(), h, req, optimize.PreflightDeps{Host: hst, Now: func() time.Time { return time.Unix(0, 0) }}, nil)
}

func find(t *testing.T, r setup.PreflightReport, id string) setup.Finding {
	t.Helper()
	for _, f := range r.Findings {
		if f.ID == id {
			return f
		}
	}
	t.Fatalf("no finding %q in %+v", id, r.Findings)
	return setup.Finding{}
}

// A healthy host is still not "ready": RAM fit is never established by
// installed RAM, so the report asks for attention and says what it did not
// measure.
func TestPreflightNeverTurnsInstalledRAMIntoAFit(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	r := pre(t, h, optimize.PreflightRequest{}, host(1<<40, true, ram(64*gib, 60*gib)))
	if r.Blocked() {
		t.Fatalf("blockers: %+v", r.Blockers())
	}
	if f := find(t, r, "memory.fit"); f.Status != setup.FindingUnknown {
		t.Fatalf("memory.fit = %s with 64 GiB installed: %s", f.Status, f.Summary)
	}
	if r.Outcome != setup.OutcomeAttention {
		t.Fatalf("outcome %s: a report with an unknown must ask for attention", r.Outcome)
	}
	for _, id := range []string{"source.manifest", "source.digests", "recipe.selected", "recipe.modules", "recipe.preserved", "recipe.carry", "storage.writable", "storage.capacity"} {
		if f := find(t, r, id); f.Status != setup.FindingPass {
			t.Errorf("%s = %s: %s", id, f.Status, f.Summary)
		}
	}
	// The optimizer runtime is not materialized in the fixture: the optimize
	// command materializes it, so this is a warning, never a pass.
	if f := find(t, r, "runtime.optimizer"); f.Status != setup.FindingWarning {
		t.Fatalf("runtime.optimizer = %s", f.Status)
	}
	if find(t, r, "accelerator.device").Facts["requested"] != "cpu" || len(r.NotMeasured) == 0 || r.Schema != setup.PreflightSchema {
		t.Fatalf("report %+v", r)
	}
	// The result is machine readable: it round-trips as JSON with the statuses.
	b, _ := json.Marshal(r)
	var back setup.PreflightReport
	if err := json.Unmarshal(b, &back); err != nil || back.Counts != r.Counts || back.Outcome != r.Outcome {
		t.Fatalf("report does not round-trip: %v %s", err, b)
	}
}

func TestPreflightRAMBoundaries(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	// The fixture's weight files are model.safetensors and the carried joint head.
	lower := uint64(len("fixture model.safetensors") + len("fixture joint_head.safetensors"))
	for name, tc := range map[string]struct {
		mem  setup.Memory
		want setup.FindingStatus
	}{
		"installed RAM below the proven lower bound": {ram(lower-1, lower-1), setup.FindingBlocker},
		"available RAM below the lower bound":        {ram(64*gib, lower-1), setup.FindingWarning},
		"RAM above the lower bound":                  {ram(64*gib, 60*gib), setup.FindingUnknown},
		"RAM that cannot be read":                    {setup.Memory{}, setup.FindingUnknown},
		"total known, available not reported":        {setup.Memory{Total: 64 * gib, TotalKnown: true}, setup.FindingUnknown},
	} {
		t.Run(name, func(t *testing.T) {
			f := find(t, pre(t, h, optimize.PreflightRequest{}, host(1<<40, true, tc.mem)), "memory.fit")
			if f.Status != tc.want {
				t.Fatalf("memory.fit = %s (%s), want %s", f.Status, f.Summary, tc.want)
			}
		})
	}
}

func TestPreflightDiskBoundaries(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	// Make the weights large without writing them: a sparse file.
	f, err := os.OpenFile(filepath.Join(dir, "model.safetensors"), os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(8000); err != nil {
		t.Fatal(err)
	}
	f.Close()
	d, _ := setup.FileSHA256(filepath.Join(dir, "model.safetensors"))
	m.Files["model.safetensors"] = d
	home.WriteJSON(filepath.Join(dir, "hachidori-model.json"), m)
	// The weight files are model.safetensors (8000) and the carried
	// joint_head.safetensors (31 bytes): upper bound 8031, lower bound 2007.
	for name, tc := range map[string]struct {
		free   uint64
		known  bool
		status setup.FindingStatus
	}{
		"free below the smallest possible variant": {2006, true, setup.FindingBlocker},
		"free between the bounds":                  {5000, true, setup.FindingWarning},
		"free above the largest possible variant":  {8031, true, setup.FindingPass},
		"free space not readable":                  {0, false, setup.FindingUnknown},
	} {
		t.Run(name, func(t *testing.T) {
			r := pre(t, h, optimize.PreflightRequest{Verified: true}, host(tc.free, tc.known, ram(64*gib, 60*gib)))
			f := find(t, r, "storage.capacity")
			if f.Status != tc.status {
				t.Fatalf("storage.capacity = %s (%s), want %s", f.Status, f.Summary, tc.status)
			}
			if tc.status == setup.FindingBlocker && (!r.Blocked() || r.Outcome != setup.OutcomeBlocked) {
				t.Fatal("a disk blocker did not block the report")
			}
		})
	}
}

// A preflight blocker stops Build before the optimizer runs: no optimizer
// process, no staging, no variant.
func TestBuildRefusesBeforeAnyExpensiveWorkOnABlocker(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	r := &optimizetest.Runner{}
	_, err := optimize.Build(context.Background(), h,
		optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: r, OptimizerRuntime: "optimizer-test", Preflight: optimize.PreflightDeps{Host: host(1, true, ram(64*gib, 60*gib))}},
		io.Discard, nil)
	pe, ok := setup.AsPreflightError(err)
	if !ok || find(t, pe.Report, "storage.capacity").Status != setup.FindingBlocker {
		t.Fatalf("err = %v, want a preflight refusal for the disk", err)
	}
	if r.Calls != 0 {
		t.Fatal("the optimizer ran although preflight refused the build")
	}
	if names := variantDirs(t, h); len(names) != 0 {
		t.Fatalf("a refused build left %v", names)
	}
}

func TestPreflightRecipeModuleMismatchIsABlocker(t *testing.T) {
	h, m := source(t)
	// The checkpoint has no lm_head tensor: the recipe's preserved selector
	// describes a module this source does not have.
	withIndex(t, h, m, "lm_head")
	r := pre(t, h, optimize.PreflightRequest{}, host(1<<40, true, ram(64*gib, 60*gib)))
	f := find(t, r, "recipe.modules")
	if f.Status != setup.FindingBlocker || !strings.Contains(f.Summary, "lm_head") || !r.Blocked() {
		t.Fatalf("recipe.modules = %s (%s)", f.Status, f.Summary)
	}
	if un, _ := f.Facts["unmatched"].([]string); len(un) != 1 || un[0] != "lm_head" {
		t.Fatalf("unmatched facts %+v", f.Facts)
	}

	// An unknown recipe, and an inventory that cannot be read.
	if f := find(t, pre(t, h, optimize.PreflightRequest{Recipe: "nope"}, host(1<<40, true, ram(64*gib, 60*gib))), "recipe.selected"); f.Status != setup.FindingBlocker {
		t.Fatalf("unknown recipe = %s", f.Status)
	}
	os.Remove(h.Path("models", filepath.FromSlash(setup.ModelDirName(m)), optimize.WeightMapFile))
	if f := find(t, pre(t, h, optimize.PreflightRequest{Verified: true}, host(1<<40, true, ram(64*gib, 60*gib))), "recipe.modules"); f.Status != setup.FindingUnknown {
		t.Fatalf("recipe.modules without an inventory = %s: modules must not be invented", f.Status)
	}
}

func TestPreflightPreservedModulesStayExcludedFromQuantization(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	f := find(t, pre(t, h, optimize.PreflightRequest{}, host(1<<40, true, ram(64*gib, 60*gib))), "recipe.preserved")
	if f.Status != setup.FindingPass {
		t.Fatalf("%s: %s", f.Status, f.Summary)
	}
	// The five preserved selectors resolve to the four synthetic modules they
	// name (lm_head, both gates, the visual tower's two modules).
	if f.Facts["preserved_modules"] != 5 {
		t.Fatalf("preserved modules %v", f.Facts["preserved_modules"])
	}
	mods, err := optimize.SourceModules(h.Path("models", filepath.FromSlash(setup.ModelDirName(m))))
	if err != nil || len(mods) != len(optimizetest.Modules)+1 {
		t.Fatalf("modules %v %v", mods, err)
	}
}

func TestPreflightSourceIntegrity(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	ok := host(1<<40, true, ram(64*gib, 60*gib))
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))

	if f := find(t, pre(t, h, optimize.PreflightRequest{Quick: true}, ok), "source.digests"); f.Status != setup.FindingUnknown {
		t.Fatalf("quick digests = %s: unhashed bytes must not pass", f.Status)
	}
	if f := find(t, pre(t, h, optimize.PreflightRequest{Verified: true}, ok), "source.digests"); f.Status != setup.FindingPass {
		t.Fatalf("caller-verified digests = %s", f.Status)
	}
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte("tampered"), 0o644)
	r := pre(t, h, optimize.PreflightRequest{}, ok)
	if f := find(t, r, "source.digests"); f.Status != setup.FindingBlocker || !strings.Contains(f.Summary, "tokenizer.json") || !r.Blocked() {
		t.Fatalf("tampered source: %s %s", f.Status, f.Summary)
	}
	os.RemoveAll(dir)
	if f := find(t, pre(t, h, optimize.PreflightRequest{}, ok), "source.manifest"); f.Status != setup.FindingBlocker || !strings.Contains(f.Summary, "setup --model") {
		t.Fatalf("missing source: %s %s", f.Status, f.Summary)
	}
}

// Variant-facing preflights (probe, certification) and the accelerator.

// servingRuntime writes a runtime manifest for device, as setup would have
// materialized it, and returns its Python path.
func servingRuntime(t *testing.T, h home.Home, device string) {
	t.Helper()
	spec, err := setup.Desired(device)
	if err != nil {
		t.Fatal(err)
	}
	dir := h.Path("runtime", spec.ID())
	os.MkdirAll(dir, 0o755)
	m := home.RuntimeManifest{Identity: spec.ID(), Spec: spec, PythonVersion: spec.Python, PythonRelPath: "env/bin/python",
		Installed: []string{"torch==" + spec.Torch, "transformers==5.17.0", "compressed-tensors==0.19.0"}}
	if err := home.WriteJSON(filepath.Join(dir, "manifest.json"), m); err != nil {
		t.Fatal(err)
	}
}

func builtVariant(t *testing.T, h home.Home) home.VariantManifest {
	t.Helper()
	res, err := build(t, h, &optimizetest.Runner{}, optimize.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res.Variant
}

func probePre(t *testing.T, h home.Home, v home.VariantManifest, device string, deps optimize.PreflightDeps) setup.PreflightReport {
	t.Helper()
	deps.Now = func() time.Time { return time.Unix(0, 0) }
	if deps.Host == nil {
		deps.Host = host(1<<40, true, ram(64*gib, 60*gib))
	}
	return optimize.Preflight(context.Background(), h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: device}, deps, nil)
}

func TestPreflightVariantIdentityAndLineage(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	v := builtVariant(t, h)
	servingRuntime(t, h, "cpu")
	r := probePre(t, h, v, "cpu", optimize.PreflightDeps{})
	for _, id := range []string{"variant.manifest", "variant.lineage", "variant.digests", "source.manifest", "source.digests", "runtime.serving", "storage.capacity"} {
		if f := find(t, r, id); f.Status != setup.FindingPass {
			t.Errorf("%s = %s: %s", id, f.Status, f.Summary)
		}
	}
	if f := find(t, r, "variant.manifest"); f.Facts["manifest_sha256"] != v.ManifestSHA256() || f.Facts["source_revision"] != m.Revision {
		t.Fatalf("variant facts %+v", f.Facts)
	}
	if f := find(t, r, "memory.fit"); f.Status != setup.FindingUnknown {
		t.Fatalf("cpu memory.fit = %s", f.Status)
	}
	if r.Variant != v.ID || r.Model != setup.ClefFlash || r.Device != "cpu" {
		t.Fatalf("report identity %+v", r)
	}

	// A corrupt variant artifact is a blocker.
	os.WriteFile(filepath.Join(h.VariantDir(setup.ClefFlash, v.ID), "model.safetensors"), []byte("corrupt"), 0o644)
	r = probePre(t, h, v, "cpu", optimize.PreflightDeps{})
	if f := find(t, r, "variant.digests"); f.Status != setup.FindingBlocker || !r.Blocked() {
		t.Fatalf("corrupt variant = %s", f.Status)
	}
	// An unknown variant is a blocker too.
	r = optimize.Preflight(context.Background(), h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: "clef-flash--nope--000000000000", Device: "cpu"},
		optimize.PreflightDeps{Host: host(1<<40, true, ram(64*gib, 60*gib))}, nil)
	if !r.Blocked() {
		t.Fatalf("unknown variant: %+v", r.Findings)
	}
}

func TestPreflightAcceleratorNeverFallsBackToCPU(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	v := builtVariant(t, h)
	servingRuntime(t, h, "cuda")
	accel := func(f setup.AcceleratorFacts, err error) optimize.PreflightDeps {
		return optimize.PreflightDeps{Accelerator: func(context.Context, home.Home, string) (setup.AcceleratorFacts, error) { return f, err }}
	}
	weights := uint64(len("fake-quantized::" + optimize.RecipeClefFlashW4A16))

	// No CUDA device: a blocker, not a cpu run.
	r := probePre(t, h, v, "cuda", accel(setup.AcceleratorFacts{Torch: "2.11.0+cu128", TorchCUDA: "12.8"}, nil))
	if f := find(t, r, "accelerator.device"); f.Status != setup.FindingBlocker || !strings.Contains(f.Summary, "never falls back") || !r.Blocked() {
		t.Fatalf("no cuda = %s %s", f.Status, f.Summary)
	}
	// A device that cannot be observed is UNKNOWN.
	r = probePre(t, h, v, "cuda", accel(setup.AcceleratorFacts{}, errors.New("probe died")))
	if f := find(t, r, "accelerator.device"); f.Status != setup.FindingUnknown {
		t.Fatalf("unobservable = %s", f.Status)
	}
	// An observed device with plenty of VRAM: the device passes, the fit stays unknown.
	good := setup.AcceleratorFacts{Torch: "2.11.0+cu128", TorchCUDA: "12.8", CUDAAvailable: true, DeviceCount: 1, DeviceName: "NVIDIA GeForce RTX 3060",
		Capability: []int{8, 6}, VRAMTotal: 12 * gib, VRAMFree: 11 * gib}
	r = probePre(t, h, v, "cuda", accel(good, nil))
	d, vr := find(t, r, "accelerator.device"), find(t, r, "accelerator.vram")
	if d.Status != setup.FindingPass || d.Facts["device_name"] != "NVIDIA GeForce RTX 3060" || vr.Status != setup.FindingUnknown {
		t.Fatalf("device %s vram %s (%s)", d.Status, vr.Status, vr.Summary)
	}
	// VRAM boundaries against the variant's known weight bytes.
	for name, tc := range map[string]struct {
		total, free uint64
		want        setup.FindingStatus
	}{
		"weights exceed total VRAM": {weights - 1, weights - 1, setup.FindingBlocker},
		"weights exceed free VRAM":  {12 * gib, weights - 1, setup.FindingWarning},
		"weights below VRAM":        {12 * gib, 11 * gib, setup.FindingUnknown},
	} {
		f := good
		f.VRAMTotal, f.VRAMFree = tc.total, tc.free
		if got := find(t, probePre(t, h, v, "cuda", accel(f, nil)), "accelerator.vram"); got.Status != tc.want {
			t.Errorf("%s: vram = %s (%s), want %s", name, got.Status, got.Summary, tc.want)
		}
	}
	// The device must be named; there is no default.
	if r := probePre(t, h, v, "", optimize.PreflightDeps{}); !r.Blocked() {
		t.Fatal("a missing device did not block")
	}
	// A cuda request with no runtime to observe it with is UNKNOWN, and the
	// missing runtime itself blocks a probe.
	os.RemoveAll(h.Path("runtime"))
	r = probePre(t, h, v, "cuda", accel(good, nil))
	if f := find(t, r, "runtime.serving"); f.Status != setup.FindingBlocker || !r.Blocked() {
		t.Fatalf("missing runtime = %s", f.Status)
	}
	if f := find(t, r, "accelerator.device"); f.Status != setup.FindingUnknown {
		t.Fatalf("accelerator without a runtime = %s", f.Status)
	}
}

func TestPreflightCertifyReferenceNeedsMoreRAMAtFloat32(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	v := builtVariant(t, h)
	servingRuntime(t, h, "cpu")
	// Weight files of the fixture source: model.safetensors and joint_head.safetensors.
	src := uint64(len("fixture model.safetensors") + len("fixture joint_head.safetensors"))
	run := func(dtype string) setup.Finding {
		r := optimize.Preflight(context.Background(), h, optimize.PreflightRequest{Kind: setup.PreflightCertify, Variant: v.ID, Device: "cpu", ReferenceDType: dtype},
			optimize.PreflightDeps{Host: host(1<<40, true, ram(src+10, src+10))}, nil)
		return find(t, r, "memory.fit")
	}
	if f := run(""); f.Status != setup.FindingUnknown {
		t.Fatalf("bf16 reference = %s", f.Status)
	}
	if f := run("float32"); f.Status != setup.FindingBlocker {
		t.Fatalf("float32 reference on %d bytes of RAM = %s (%s)", src+10, f.Status, f.Summary)
	}
}

func TestPreflightMaterializeReportsResumableState(t *testing.T) {
	h, m := source(t)
	// An unmaterialized model with an interrupted download kept for resume.
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	os.RemoveAll(dir)
	setup.Models = []home.ModelManifest{{ID: m.ID, Provider: m.Provider, Repo: m.Repo, Revision: m.Revision, Files: map[string]string{"big.safetensors": strings.Repeat("a", 64)}}}
	stage := dir + ".staging"
	os.MkdirAll(stage, 0o755)
	os.WriteFile(filepath.Join(stage, "big.safetensors.part"), make([]byte, 100), 0o644)
	os.WriteFile(filepath.Join(stage, "big.safetensors.part.json"), []byte(`{"schema":"hachidori.partial-download.v1","url":"`+
		setup.ModelFileURL(setup.Models[0], "big.safetensors")+`","sha256":"`+strings.Repeat("a", 64)+`","etag":"\"x\"","total":1000}`), 0o644)
	servingRuntime(t, h, "cpu")
	run := func(free uint64) setup.PreflightReport {
		return optimize.Preflight(context.Background(), h, optimize.PreflightRequest{Kind: setup.PreflightMaterialize, Model: m.ID, Device: "cpu"},
			optimize.PreflightDeps{Host: host(free, true, ram(64*gib, 60*gib))}, nil)
	}
	r := run(10_000)
	f := find(t, r, "storage.capacity")
	if f.Status != setup.FindingUnknown || f.Facts["remaining_known_bytes"] != uint64(900) || f.Facts["partial_bytes"] != int64(100) {
		t.Fatalf("storage.capacity = %s %+v: the catalog states no sizes, so the total is unknown, the resumable remainder is known", f.Status, f.Facts)
	}
	if r := run(500); find(t, r, "storage.capacity").Status != setup.FindingBlocker {
		t.Fatal("free space below the known remainder of an interrupted download did not block")
	}
}

func TestPreflightUnwritableTargetIsABlocker(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	// variants/<model> is a regular file: nothing can be staged under it.
	os.MkdirAll(h.Path("variants"), 0o755)
	os.WriteFile(h.VariantsDir(setup.ClefFlash), []byte("x"), 0o644)
	r := pre(t, h, optimize.PreflightRequest{Verified: true}, host(1<<40, true, ram(64*gib, 60*gib)))
	if f := find(t, r, "storage.writable"); f.Status != setup.FindingBlocker || !r.Blocked() {
		t.Fatalf("storage.writable = %s (%s)", f.Status, f.Summary)
	}
}

// Materialize preflight distinguishes an absent source from a present one, and
// checks a present one before materialization would reuse it: a corrupt
// manifest, another catalog identity, a missing or a corrupt pinned file is a
// blocker, never a pass because a manifest file exists. Quick mode reports the
// unhashed digests UNKNOWN.
func TestPreflightMaterializeChecksAnExistingSource(t *testing.T) {
	h, m := source(t)
	servingRuntime(t, h, "cpu")
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	manifest := filepath.Join(dir, "hachidori-model.json")
	good, _ := os.ReadFile(manifest)
	ok := host(1<<40, true, ram(64*gib, 60*gib))
	run := func(quick bool) setup.PreflightReport {
		t.Helper()
		return optimize.Preflight(context.Background(), h, optimize.PreflightRequest{Kind: setup.PreflightMaterialize, Model: m.ID, Device: "cpu", Quick: quick},
			optimize.PreflightDeps{Host: ok}, nil)
	}
	restore := func() {
		t.Helper()
		if err := os.WriteFile(manifest, good, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// Present and valid: the manifest matches and every digest is hashed.
	r := run(false)
	if f := find(t, r, "source.manifest"); f.Status != setup.FindingPass || f.Facts["condition"] != string(setup.SourcePresent) {
		t.Fatalf("valid source: %s %s %+v", f.Status, f.Summary, f.Facts)
	}
	if f := find(t, r, "source.digests"); f.Status != setup.FindingPass || r.Blocked() {
		t.Fatalf("valid source digests: %s %s", f.Status, f.Summary)
	}
	// Quick: the digests are not hashed, so they are UNKNOWN, not PASS.
	if f := find(t, run(true), "source.digests"); f.Status != setup.FindingUnknown {
		t.Fatalf("quick digests = %s", f.Status)
	}

	for name, tc := range map[string]struct {
		mutate func()
		cond   setup.SourceCondition
	}{
		"corrupt manifest": {func() { os.WriteFile(manifest, []byte("{not json"), 0o644) }, setup.SourceManifestCorrupt},
		"another catalog revision": {func() {
			other := m
			other.Revision = strings.Repeat("ab", 20)
			home.WriteJSON(manifest, other)
		}, setup.SourceIdentityMismatch},
		"another pinned digest": {func() {
			other := m
			other.Files = map[string]string{}
			for k, v := range m.Files {
				other.Files[k] = v
			}
			other.Files["config.json"] = strings.Repeat("0", 64)
			home.WriteJSON(manifest, other)
		}, setup.SourceIdentityMismatch},
		"missing pinned file": {func() { os.Remove(filepath.Join(dir, "config.json")) }, setup.SourceFileMissing},
	} {
		t.Run(name, func(t *testing.T) {
			defer os.WriteFile(filepath.Join(dir, "config.json"), []byte("fixture config.json"), 0o644)
			defer restore()
			tc.mutate()
			for _, quick := range []bool{false, true} {
				r := run(quick)
				f := find(t, r, "source.manifest")
				if f.Status != setup.FindingBlocker || f.Facts["condition"] != string(tc.cond) || !r.Blocked() {
					t.Fatalf("quick=%v: source.manifest = %s (%s) %+v, want a %s blocker", quick, f.Status, f.Summary, f.Facts, tc.cond)
				}
			}
		})
	}

	// A corrupt pinned artifact under a valid manifest: knowable by hashing,
	// so the normal preflight blocks; quick mode says UNKNOWN, not PASS.
	os.WriteFile(filepath.Join(dir, "config.json"), []byte("tampered"), 0o644)
	r = run(false)
	if f := find(t, r, "source.digests"); f.Status != setup.FindingBlocker || !strings.Contains(f.Summary, "config.json") || !r.Blocked() {
		t.Fatalf("corrupt artifact: %s %s", f.Status, f.Summary)
	}
	if r := run(true); find(t, r, "source.digests").Status != setup.FindingUnknown || r.Outcome == setup.OutcomeReady {
		t.Fatalf("quick with a corrupt artifact: %s", r.Outcome)
	}
	os.WriteFile(filepath.Join(dir, "config.json"), []byte("fixture config.json"), 0o644)

	// Absent: nothing to check; materialization will download it.
	os.RemoveAll(dir)
	r = run(false)
	if f := find(t, r, "source.manifest"); f.Status != setup.FindingPass || f.Facts["condition"] != string(setup.SourceAbsent) {
		t.Fatalf("absent source: %s %+v", f.Status, f.Facts)
	}
	for _, f := range r.Findings {
		if f.ID == "source.digests" {
			t.Fatalf("an absent source has no digests to report: %+v", f)
		}
	}
}

// Every report is bound to the identities its findings were drawn from,
// derived from the existing authorities.
func TestPreflightReportIsBoundToItsIdentities(t *testing.T) {
	h, m := source(t)
	withIndex(t, h, m)
	m = setup.Models[0]
	v := builtVariant(t, h)
	servingRuntime(t, h, "cpu")
	spec, _ := setup.Desired("cpu")
	rt, _ := setup.FileSHA256(h.Path("runtime", spec.ID(), "manifest.json"))
	srcManifest, _ := setup.FileSHA256(h.Path("models", filepath.FromSlash(setup.ModelDirName(m)), "hachidori-model.json"))
	src := home.SourceOf(m)

	r := probePre(t, h, v, "cpu", optimize.PreflightDeps{})
	want := setup.PreflightBinding{Kind: setup.PreflightProbe, Device: "cpu", Model: m.ID, Provider: m.Provider, SourceRepo: m.Repo, SourceRevision: m.Revision,
		SourceFilesSHA256: src.FilesSHA256, SourceManifestSHA256: srcManifest, Runtime: spec.ID(), RuntimeManifestSHA256: rt,
		Variant: v.ID, VariantManifestSHA256: v.ManifestSHA256(), Recipe: v.Recipe.Name, RecipeSHA256: v.RecipeSHA256}
	if r.Binding == nil || *r.Binding != want {
		t.Fatalf("probe binding\n got %+v\nwant %+v", r.Binding, want)
	}
	if again := optimize.PreflightBindingOf(h, optimize.PreflightRequestOf(r)); len(r.Binding.Mismatch(again)) != 0 {
		t.Fatalf("recomputed binding differs: %v", r.Binding.Mismatch(again))
	}

	recipe, _ := optimize.LookupRecipe(m.ID, optimize.RecipeClefFlashW4A16)
	opt, _ := setup.DesiredOptimizer("cpu")
	r = pre(t, h, optimize.PreflightRequest{Verified: true}, host(1<<40, true, ram(64*gib, 60*gib)))
	if b := r.Binding; b == nil || b.Kind != setup.PreflightOptimize || b.Recipe != recipe.Name || b.RecipeSHA256 != recipe.SHA256() || b.Runtime != opt.ID() ||
		b.RuntimeManifestSHA256 != "" || b.Variant != "" || b.Device != "" || b.SourceManifestSHA256 != srcManifest {
		t.Fatalf("optimize binding %+v", b)
	}

	// A materialize report of an absent source binds no source manifest; once
	// the source exists the recomputed binding differs.
	r = optimize.Preflight(context.Background(), h, optimize.PreflightRequest{Kind: setup.PreflightMaterialize, Model: m.ID, Device: "cpu", Quick: true},
		optimize.PreflightDeps{Host: host(1<<40, true, ram(64*gib, 60*gib))}, nil)
	if r.Binding.SourceManifestSHA256 != srcManifest {
		t.Fatalf("materialize binding %+v", r.Binding)
	}
	os.Remove(h.Path("models", filepath.FromSlash(setup.ModelDirName(m)), "hachidori-model.json"))
	if d := r.Binding.Mismatch(optimize.PreflightBindingOf(h, optimize.PreflightRequestOf(r))); len(d) != 1 || d[0] != "source_manifest_sha256" {
		t.Fatalf("mismatch %v", d)
	}
}
