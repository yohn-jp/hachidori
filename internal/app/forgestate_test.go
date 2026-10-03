package app

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// recordPreflight runs the optimize authority's preflight for req over a
// healthy fake host (no accelerator is ever observed) and records it, as
// RunPreflight does.
func recordPreflight(t *testing.T, h home.Home, req optimize.PreflightRequest, at time.Time) setup.PreflightReport {
	t.Helper()
	host := &setup.Host{FreeDisk: func(string) (uint64, bool) { return 1 << 40, true }, Memory: func() setup.Memory { return ram(64*gib, 60*gib) }}
	rep := optimize.Preflight(context.Background(), h, req, optimize.PreflightDeps{Host: host, Now: func() time.Time { return at },
		Accelerator: func(context.Context, home.Home, string) (setup.AcceleratorFacts, error) {
			return setup.AcceleratorFacts{Torch: "2.11.0+cu128", TorchCUDA: "12.8", CUDAAvailable: true, DeviceCount: 1, DeviceName: "gpu", VRAMTotal: 12 * gib, VRAMFree: 11 * gib}, nil
		}}, nil)
	if rep.Binding == nil || rep.Schema != setup.PreflightSchema {
		t.Fatalf("report is not bound: %+v", rep)
	}
	if err := SavePreflight(h, rep); err != nil {
		t.Fatal(err)
	}
	return rep
}

// clefAnalysis is the semantic analysis of a minimal clef-flash layout.
func clefAnalysis(t *testing.T, source home.ModelManifest) tuning.Analysis {
	t.Helper()
	analysis, err := tuning.Analyze(source, tuning.DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text",
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes:    []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{
			"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj",
		},
		CarriedFiles: []string{"joint_head.safetensors"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return analysis
}

func TestOptimizeProfileResolvesExactProfileAndRefusesAnotherSource(t *testing.T) {
	source, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	analysis := clefAnalysis(t, source)
	profile, err := tuning.NewDefaultProfile(analysis, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	h := home.Home{Root: t.TempDir()}
	if err := tuning.SaveProfile(h, profile, analysis); err != nil {
		t.Fatal(err)
	}

	req, err := tunedBuildRequest(h, source, profile.ID())
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != source.ID || req.Recipe != optimize.RecipeClefFlashW4A16 || req.CompiledRecipe == nil || req.Tuning == nil || req.Plan == nil {
		t.Fatalf("profile did not resolve to a tuned optimizer request: %+v", req)
	}
	if req.Tuning.ProfileID != profile.ID() || req.Tuning.ProfileSHA256 != profile.SHA256() ||
		req.Tuning.AnalysisID != analysis.ID() || req.Tuning.AnalysisSHA256 != analysis.SHA256() ||
		req.Tuning.CompilerVersion != profile.CompilerVersion {
		t.Fatalf("optimizer request lost exact tuning provenance: %+v", req.Tuning)
	}
	if req.Tuning.PlanSHA256 != req.Plan.SHA256() || req.Plan.RecipeSHA256 != req.CompiledRecipe.SHA256() {
		t.Fatalf("optimizer request does not bind its plan: %+v", req.Tuning)
	}
	var built optimize.Request
	c := New(Config{Home: h.Root, Maintenance: Maintenance{Build: func(_ context.Context, _ home.Home, got optimize.Request, _ io.Writer, _ *setup.Observer) (optimize.Result, error) {
		built = got
		return optimize.Result{Variant: home.VariantManifest{ID: "fixture"}}, nil
	}}})
	if err := c.OptimizeProfile(source.ID, profile.ID()); err != nil {
		t.Fatal(err)
	}
	operation := waitIdle(t, c).Maintenance
	if operation == nil || operation.Failure != nil || operation.Target != "tuning profile "+profile.ID() {
		t.Fatalf("profile build operation did not complete with the exact target: %+v", operation)
	}
	if built.Tuning == nil || built.Tuning.ProfileID != profile.ID() || built.CompiledRecipe == nil || built.Recipe != req.Recipe {
		t.Fatalf("controller did not pass the resolved tuning request to the build authority: %+v", built)
	}

	otherSource := source
	otherSource.Revision = strings.Repeat("0", len(source.Revision))
	if _, err := tunedBuildRequest(h, otherSource, profile.ID()); err == nil {
		t.Fatal("Forge accepted a tuning profile bound to another source")
	}
}

// recorded is the one recorded preflight of the home, judged now.
func recorded(t *testing.T, h home.Home) RecordedPreflight {
	t.Helper()
	st := ReadForgeState(h)
	if len(st.Preflights) != 1 {
		t.Fatalf("%d recorded preflights", len(st.Preflights))
	}
	return st.Preflights[0]
}

// rewriteRecorded edits the one recorded preflight file in place.
func rewriteRecorded(t *testing.T, h home.Home, edit func(*setup.PreflightReport)) {
	t.Helper()
	files, _ := filepath.Glob(filepath.Join(forgeDir(h, "preflight"), "*.json"))
	if len(files) != 1 {
		t.Fatalf("recorded files %v", files)
	}
	var r setup.PreflightReport
	if err := home.ReadJSON(files[0], &r); err != nil {
		t.Fatal(err)
	}
	edit(&r)
	if err := home.WriteJSON(files[0], r); err != nil {
		t.Fatal(err)
	}
}

// A recorded preflight is current only while every identity it was bound to is
// still the identity of its target: a changed runtime, source, variant
// manifest or recipe, another device or another operation kind makes it
// stale, and it is then never returned as the readiness of its target.
func TestRecordedPreflightGoesStaleWhenItsIdentitiesChange(t *testing.T) {
	probe := func(h home.Home, v home.VariantManifest) optimize.PreflightRequest {
		return optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: "cpu", Quick: true}
	}
	target := func(v home.VariantManifest) PreflightTarget {
		return PreflightTarget{Kind: setup.PreflightProbe, Model: setup.ClefFlash, Variant: v.ID, Recipe: v.Recipe.Name, Device: "cpu"}
	}
	for name, tc := range map[string]struct {
		change func(t *testing.T, h home.Home, v home.VariantManifest)
		stale  []string
	}{
		"runtime identity (the runtime is rematerialized differently)": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			spec, _ := setup.Desired("cpu")
			path := h.Path("runtime", spec.ID(), "manifest.json")
			var rm home.RuntimeManifest
			home.ReadJSON(path, &rm)
			rm.Installed = append(rm.Installed, "numpy==9.9.9")
			home.WriteJSON(path, rm)
		}, []string{"runtime_manifest_sha256"}},
		"runtime spec (another build's Runtime Spec)": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			rewriteRecorded(t, h, func(r *setup.PreflightReport) { r.Binding.Runtime = "cpu-0000000000000000" })
		}, []string{"runtime"}},
		"source revision and files (the catalog pin moved)": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			m := setup.Models[0]
			files := map[string]string{}
			for k, d := range m.Files {
				files[k] = d
			}
			files["config.json"] = strings.Repeat("0", 64)
			m.Files = files
			setup.Models = []home.ModelManifest{m}
		}, nil},
		"source manifest (the materialized source was replaced)": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			m := setup.Models[0]
			path := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)), "hachidori-model.json")
			b, _ := os.ReadFile(path)
			os.WriteFile(path, append(b, ' '), 0o644)
		}, []string{"source_manifest_sha256"}},
		"variant manifest": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			v.Creation.Command = "another build"
			v.Seal()
			b, _ := json.MarshalIndent(v, "", "  ")
			os.WriteFile(filepath.Join(h.VariantDir(setup.ClefFlash, v.ID), home.VariantManifestFile), b, 0o644)
		}, []string{"variant_manifest_sha256"}},
		"recipe digest (the report was made under another recipe definition)": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			rewriteRecorded(t, h, func(r *setup.PreflightReport) { r.Binding.RecipeSHA256 = strings.Repeat("a", 64) })
		}, []string{"recipe_sha256"}},
		"device": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			rewriteRecorded(t, h, func(r *setup.PreflightReport) { r.Binding.Device = "cuda" })
		}, []string{"device"}},
		"operation kind": {func(t *testing.T, h home.Home, v home.VariantManifest) {
			rewriteRecorded(t, h, func(r *setup.PreflightReport) { r.Binding.Kind = setup.PreflightCertify })
		}, []string{"kind"}},
	} {
		t.Run(name, func(t *testing.T) {
			h, v := forgeHome(t)
			rep := recordPreflight(t, h, probe(h, v), time.Unix(100, 0))
			if r := recorded(t, h); r.Evidence != setup.EvidenceCurrent || len(r.Stale) != 0 {
				t.Fatalf("fresh report: %s %v", r.Evidence, r.Stale)
			}
			if got, ok := LatestPreflight(h, target(v)); !ok || got.CreatedAt != rep.CreatedAt {
				t.Fatal("the fresh report is not the latest of its target")
			}
			tc.change(t, h, v)
			r := recorded(t, h)
			if r.Evidence != setup.EvidenceStale || r.Current() {
				t.Fatalf("after the change: %s %v", r.Evidence, r.Stale)
			}
			for _, d := range tc.stale {
				if !contains(r.Stale, d) {
					t.Fatalf("stale dimensions %v, want %s", r.Stale, d)
				}
			}
			if name == "source revision and files (the catalog pin moved)" && !contains(r.Stale, "source_files_sha256") && !contains(r.Stale, "variant_manifest_sha256") {
				t.Fatalf("stale dimensions %v", r.Stale)
			}
			if _, ok := LatestPreflight(h, target(v)); ok {
				t.Fatal("a stale report was returned as the latest of its target")
			}
		})
	}
}

// A source revision change (a new catalog pin) is a different source: the
// recorded binding names the old revision.
func TestRecordedPreflightGoesStaleOnASourceRevisionChange(t *testing.T) {
	h, _ := forgeHome(t)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16, Quick: true}, time.Unix(100, 0))
	if r := recorded(t, h); !r.Current() {
		t.Fatalf("fresh: %s %v", r.Evidence, r.Stale)
	}
	m := setup.Models[0]
	m.Revision = strings.Repeat("cd", 20)
	setup.Models = []home.ModelManifest{m}
	r := recorded(t, h)
	if r.Current() || !contains(r.Stale, "source_revision") || !contains(r.Stale, "source_manifest_sha256") {
		t.Fatalf("after the revision change: %s %v", r.Evidence, r.Stale)
	}
}

// A report from before bindings existed stays readable but is legacy: never
// current, never the latest of a target, and never shown as READY.
func TestLegacyPreflightReportIsNeverCurrent(t *testing.T) {
	h, v := forgeHome(t)
	legacy := map[string]any{"schema": setup.LegacyPreflightSchema, "kind": setup.PreflightProbe, "model": setup.ClefFlash, "variant": v.ID, "device": "cpu",
		"created_at": "2026-10-01T00:00:00Z", "outcome": setup.OutcomeReady, "findings": []any{}, "counts": map[string]int{"pass": 3}}
	os.MkdirAll(forgeDir(h, "preflight"), 0o755)
	b, _ := json.Marshal(legacy)
	os.WriteFile(filepath.Join(forgeDir(h, "preflight"), "probe--clef-flash--"+v.ID+"--cpu.json"), b, 0o644)
	r := recorded(t, h)
	if r.Evidence != setup.EvidenceLegacy || r.Current() || r.Outcome != setup.OutcomeReady {
		t.Fatalf("legacy report: %+v", r)
	}
	if _, ok := LatestPreflight(h, PreflightTarget{Kind: setup.PreflightProbe, Model: setup.ClefFlash, Variant: v.ID, Recipe: v.Recipe.Name, Device: "cpu"}); ok {
		t.Fatal("a legacy report was returned as current evidence")
	}
	// A report that claims the current schema but carries no binding cannot
	// prove what it is about either.
	legacy["schema"] = setup.PreflightSchema
	b, _ = json.Marshal(legacy)
	os.WriteFile(filepath.Join(forgeDir(h, "preflight"), "probe--clef-flash--"+v.ID+"--cpu.json"), b, 0o644)
	if r := recorded(t, h); r.Evidence != setup.EvidenceLegacy {
		t.Fatalf("unbound report: %+v", r)
	}
}

// LatestPreflight selects by the whole target: kind, model, variant, recipe
// and device, an empty one included. A CPU report never answers for CUDA, a
// probe report never for a certification, and two reports of the same target
// recorded at the same instant answer for neither.
func TestLatestPreflightSelectsTheExactTarget(t *testing.T) {
	h, v := forgeHome(t)
	at := time.Unix(100, 0)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: "cpu", Quick: true}, at)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16, Quick: true}, at)
	probe := PreflightTarget{Kind: setup.PreflightProbe, Model: setup.ClefFlash, Variant: v.ID, Recipe: v.Recipe.Name, Device: "cpu"}
	if _, ok := LatestPreflight(h, probe); !ok {
		t.Fatal("exact probe target not found")
	}
	for name, tgt := range map[string]PreflightTarget{
		"cuda instead of cpu":        {Kind: setup.PreflightProbe, Model: setup.ClefFlash, Variant: v.ID, Recipe: v.Recipe.Name, Device: "cuda"},
		"certify instead of probe":   {Kind: setup.PreflightCertify, Model: setup.ClefFlash, Variant: v.ID, Recipe: v.Recipe.Name, Device: "cpu"},
		"another variant":            {Kind: setup.PreflightProbe, Model: setup.ClefFlash, Variant: "clef-flash--other--000000000000", Recipe: v.Recipe.Name, Device: "cpu"},
		"another model":              {Kind: setup.PreflightProbe, Model: setup.DefaultModel, Variant: v.ID, Recipe: v.Recipe.Name, Device: "cpu"},
		"another recipe":             {Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: "clef-flash-other-recipe"},
		"optimize on a named device": {Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16, Device: "cuda"},
		"no device named":            {Kind: setup.PreflightProbe, Model: setup.ClefFlash, Variant: v.ID, Recipe: v.Recipe.Name},
	} {
		if r, ok := LatestPreflight(h, tgt); ok {
			t.Errorf("%s: got the report of %+v", name, r.Binding)
		}
	}
	// The same resolved target requested twice (the recipe named and not):
	// two reports of one instant have no order.
	rep := recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: "cpu"}, at)
	rep.Recipe = "renamed-request"
	if err := SavePreflight(h, rep); err != nil {
		t.Fatal(err)
	}
	if _, ok := LatestPreflight(h, probe); ok {
		t.Fatal("two reports of the same instant were ordered")
	}
	later := recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: "cpu"}, at.Add(time.Millisecond))
	if got, ok := LatestPreflight(h, probe); !ok || got.CreatedAt != later.CreatedAt {
		t.Fatalf("the later report did not decide: %+v %v", got.CreatedAt, ok)
	}
}
