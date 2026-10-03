package trial

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// optimizerLayout is the declared layout of the fake optimizer's module graph,
// so that a finalist can be built end to end without Python.
func optimizerLayout() tuning.DeclaredLayout {
	return tuning.DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text", Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes: []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj"},
		CarriedFiles: []string{"joint_head.safetensors"},
	}
}

const (
	attn0 = "block.00.linear-attn"
	attn3 = "block.03.full-attn"
)

// finalistFixture measures a profile in a session, saves the profile and the
// candidate with its evidence.
func finalistFixture(t *testing.T) (world, Candidate, tuning.Profile) {
	t.Helper()
	w := newWorldWith(t, optimizerLayout())
	// attn0 is kept at source precision by an operator override; mlp0 and attn3
	// are AUTO (W4A16).
	profile := w.profile(t, attn0)
	if err := tuning.SaveProfile(w.h, profile, w.analysis); err != nil {
		t.Fatal(err)
	}
	s, _ := openSession(t, w, Options{})
	plan, err := tuningCompile(w, profile)
	if err != nil {
		t.Fatal(err)
	}
	r := mustRun(t, s, plan, &evaluator{})
	c, err := NewCandidate(w.source, w.ref(t, profile), r, fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	e, err := NewEvidence(c, r, fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	if err := Save(w.h, c, e); err != nil {
		t.Fatal(err)
	}
	return w, c, profile
}

func TestFinalistRequestReproducesTheCandidatesExactResolvedPlan(t *testing.T) {
	w, c, profile := finalistFixture(t)
	f, err := PrepareFinalist(w.h, w.model, c.ID)
	if err != nil {
		t.Fatal(err)
	}
	if f.Request.Plan == nil || string(f.Request.Plan.Canonical()) != string(c.Plan.Canonical()) || f.Request.Tuning.PlanSHA256 != c.PlanSHA256 {
		t.Fatal("the finalist request is not the candidate's plan")
	}
	if f.Request.Tuning.ProfileID != profile.ID() || f.Request.Tuning.ProfileSHA256 != profile.SHA256() {
		t.Fatalf("provenance %+v", f.Request.Tuning)
	}
	// It is exactly what a Forge build of that profile resolves, whatever profile
	// is currently open elsewhere.
	direct, err := tuning.BuildRequest(w.h, w.model, profile.ID())
	if err != nil || direct.Plan.SHA256() != f.Request.Plan.SHA256() || direct.Tuning.PlanSHA256 != f.Request.Tuning.PlanSHA256 {
		t.Fatalf("direct build request differs (%v)", err)
	}
}

func TestFinalistIsRefusedWhenTheProfileNoLongerResolvesToTheMeasuredPlan(t *testing.T) {
	w, c, _ := finalistFixture(t)
	// A candidate that names a profile whose plan is a different one: it must
	// not be materialized as "the measured plan".
	otherProfile := w.profile(t, attn0, mlp0)
	if err := tuning.SaveProfile(w.h, otherProfile, w.analysis); err != nil {
		t.Fatal(err)
	}
	s, _ := openSession(t, w, Options{})
	plan, _ := tuningCompile(w, w.profile(t, attn0))
	r := mustRun(t, s, plan, &evaluator{})
	liar, err := NewCandidate(w.source, w.ref(t, otherProfile), r, fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := NewEvidence(liar, r, fixedNow)
	if err := Save(w.h, liar, e); err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareFinalist(w.h, w.model, liar.ID); err == nil || !strings.Contains(err.Error(), "no longer resolves") {
		t.Fatalf("err = %v", err)
	}
	if _, err := PrepareFinalist(w.h, w.model, c.ID); err != nil {
		t.Fatalf("the honest candidate: %v", err)
	}
	if _, err := PrepareFinalist(w.h, w.model, strings.Repeat("a", 64)); err == nil {
		t.Fatal("an unknown candidate was prepared")
	}
}

func TestFinalistMaterializesAsAnIndependentImmutableVariant(t *testing.T) {
	w, c, _ := finalistFixture(t)
	f, err := PrepareFinalist(w.h, w.model, c.ID)
	if err != nil {
		t.Fatal(err)
	}
	res, err := optimize.Build(context.Background(), w.h, f.Request, optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The variant is the normal Forge artifact: it carries the candidate's plan
	// in its provenance and per-group evidence, and reads back with nothing but
	// its own directory (no session, no RAM cache, no candidate record).
	got, err := home.ReadVariant(res.Dir)
	if err != nil || got.Tuning == nil || got.Tuning.PlanSHA256 != c.PlanSHA256 {
		t.Fatalf("variant %+v (%v)", got.Tuning, err)
	}
	ev, err := home.ReadTuningEvidence(res.Dir)
	if err != nil || ev.PlanSHA256 != c.PlanSHA256 {
		t.Fatalf("variant evidence %+v (%v)", ev, err)
	}
	for _, g := range ev.Groups {
		want := ""
		for _, pg := range c.Plan.Groups {
			if pg.ID == g.ID {
				want = pg.Effective
			}
		}
		if g.Applied != want {
			t.Errorf("group %s applied %s, the candidate's plan says %s", g.ID, g.Applied, want)
		}
	}
	before, _ := os.ReadFile(filepath.Join(res.Dir, home.VariantManifestFile))
	if err := RecordMaterialization(w.h, c.ID, res.Variant, fixedNow); err != nil {
		t.Fatal(err)
	}
	if err := RecordMaterialization(w.h, c.ID, res.Variant, fixedNow.Add(1)); err != nil {
		t.Fatalf("recording twice: %v", err)
	}
	after, _ := os.ReadFile(filepath.Join(res.Dir, home.VariantManifestFile))
	if string(before) != string(after) {
		t.Fatal("materialization record modified the variant")
	}
	st, err := StatusOf(w.h, c.ID)
	if err != nil || st.Stage() != "materialized" || len(st.Variants) != 1 || st.Variants[0] != res.Variant.ID {
		t.Fatalf("status %+v (%v)", st, err)
	}
}

func TestMaterializationRefusesAVariantNotBuiltFromTheCandidate(t *testing.T) {
	w, c, _ := finalistFixture(t)
	// A variant of the canonical recipe (no tuning provenance) is not it.
	canonical, err := optimize.Build(context.Background(), w.h, optimize.Request{Model: w.model.ID, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := RecordMaterialization(w.h, c.ID, canonical.Variant, fixedNow); err == nil {
		t.Fatal("a variant without the candidate's plan was recorded as its materialization")
	}
	// A variant of another plan of the same source is not it either.
	other := w.profile(t, attn0, mlp0)
	if err := tuning.SaveProfile(w.h, other, w.analysis); err != nil {
		t.Fatal(err)
	}
	req, err := tuning.BuildRequest(w.h, w.model, other.ID())
	if err != nil {
		t.Fatal(err)
	}
	built, err := optimize.Build(context.Background(), w.h, req, optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := RecordMaterialization(w.h, c.ID, built.Variant, fixedNow); err == nil {
		t.Fatal("a variant of another plan was recorded")
	}
	if st, _ := StatusOf(w.h, c.ID); st.Stage() != "measured" {
		t.Fatalf("status %+v", st)
	}
}

func TestLegacyForgeBuildsAreUnaffectedByTheTrialSubstrate(t *testing.T) {
	w := newWorldWith(t, optimizerLayout())
	// The canonical, untuned Forge path needs nothing from trials: it builds
	// from the catalog recipe, with no provenance or plan.
	res, err := optimize.Build(context.Background(), w.h, optimize.Request{Model: w.model.ID, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Variant.Tuning != nil {
		t.Fatalf("a canonical build gained tuning provenance: %+v", res.Variant.Tuning)
	}
	if entries, _ := os.ReadDir(filepath.Join(w.h.Root, "state", "tuning-trials")); len(entries) != 0 {
		t.Fatal("a Forge build created trial state")
	}
}

func TestMeasurementOfAResidentRun(t *testing.T) {
	acc, conf := 0.8, 0.7
	alloc, rss := int64(5), int64(6)
	run := eval.ResidentRun{Labelled: true, Run: eval.ModelRun{
		Model: "clef-flash", DatasetSHA256: strings64("d"), InputSHA256: strings64("i"), SentSHA256: strings64("s"),
		Questions: []eval.QuestionRecord{{ID: "q", SHA256: strings64("q")}}, Cases: 3, WarmupRequests: 1, Passes: 1, Requests: 3, Succeeded: 3,
		ResidentStable: true, Quality: eval.Quality{Accuracy: &acc, MeanConfidence: &conf},
		RequestLatency: eval.Latency{N: 3, P50: 10, P95: 12},
		Memory:         eval.Memory{PeakAlloc: &alloc, PeakHostRSS: &rss},
		Identity:       eval.Served{Provider: map[string]any{"execution": ExecutionTrial, "device": "cuda", "dtype": "torch.bfloat16"}, WorkerPID: 42}}}
	m, err := MeasurementOf(run)
	if err != nil {
		t.Fatal(err)
	}
	if m.Mode != EvalModeTrial || m.Accuracy == nil || *m.Accuracy != 0.8 || *m.LatencyP50MS != 10 || m.Served.Execution != ExecutionTrial || m.Served.WorkerPID != 42 || *m.GPUAllocatedBytes != 5 || *m.HostRSSBytes != 6 {
		t.Fatalf("measurement %+v", m)
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	unlabelled := run
	unlabelled.Labelled = false
	if m, _ := MeasurementOf(unlabelled); m.Accuracy != nil {
		t.Fatal("accuracy reported for a run without labels")
	}
	unstable := run
	unstable.Run.ResidentStable = false
	if _, err := MeasurementOf(unstable); err == nil {
		t.Fatal("an unstable run is a measurement")
	}
	empty := run
	empty.Run.Succeeded = 0
	if _, err := MeasurementOf(empty); err == nil {
		t.Fatal("a run that answered nothing is a measurement")
	}
	served := run
	served.Run.Identity.Provider = map[string]any{"execution": "variant", "variant_id": "v"}
	if m, err := MeasurementOf(served); err != nil || m.Validate() == nil {
		t.Fatalf("a measurement of a served variant must not validate as trial evidence (%v)", err)
	}
}
