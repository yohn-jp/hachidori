package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// storeVariant publishes a sealed variant of the fixture source with the given
// recipe and optional tuning provenance, with real artifact bytes of size
// bytes, and returns it with its directory.
func storeVariant(t *testing.T, h home.Home, m home.ModelManifest, recipe home.Recipe, prov *home.TuningProvenance, bytes int) (home.VariantManifest, string) {
	t.Helper()
	v := home.VariantManifest{Source: home.SourceOf(m), Provider: m.Provider, Optimizer: home.Optimizer{Engine: recipe.Engine, Version: "1"}, Recipe: recipe, Tuning: prov,
		Weights:  home.WeightPrecision{Scheme: recipe.Scheme, Bits: 4, GroupSize: 128, Format: "compressed-tensors", DType: "bfloat16"},
		Files:    map[string]string{"model.safetensors": strings.Repeat("ab", 32)},
		Creation: home.Creation{CreatedAt: "2026-10-03T00:00:00Z"}}
	v.Seal()
	dir := h.VariantDir(m.ID, v.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), make([]byte, bytes), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := home.WriteJSON(filepath.Join(dir, home.VariantManifestFile), v); err != nil {
		t.Fatal(err)
	}
	return v, dir
}

// certify records an accepted certification of v on the given dataset with
// labelled accuracy, request latency and peak VRAM.
func certify(t *testing.T, h home.Home, m home.ModelManifest, v home.VariantManifest, dataset string, accuracy float64, p50 float64, vram int64) {
	t.Helper()
	pol := eval.DefaultPolicy()
	c := eval.Certification{
		Schema: eval.CertificationSchema, CreatedAt: "2026-10-03T01:00:00Z", Source: v.Source,
		Variant: eval.CertifiedVariant{ID: v.ID, ManifestSHA256: v.ManifestSHA256(), BuildID: v.BuildID, Recipe: v.Recipe.Name,
			RecipeSHA256: v.RecipeSHA256, Scheme: v.Weights.Scheme, Engine: v.Optimizer.Engine, EngineVersion: v.Optimizer.Version},
		Reference: eval.Execution{Role: "reference", ModelID: m.ID, Provider: m.Provider, Revision: m.Revision, Device: "cpu", DType: "torch.bfloat16", IdentitySHA256: "ref", ResidentStable: true},
		Candidate: eval.Execution{Role: "candidate", ModelID: m.ID, Provider: m.Provider, Revision: m.Revision, VariantID: v.ID, Device: "cuda", DType: "torch.bfloat16", IdentitySHA256: "cand", ResidentStable: true},
		Dataset:   eval.CertifiedDataset{SHA256: dataset, Cases: 100, Observations: 100, QuestionsSHA256: "questions", InputSHA256: "inputs", SentSHA256: "sent"},
		Fidelity:  eval.Fidelity{Paired: 200, ChoiceFlips: 2, FlipRate: 0.01}, Policy: pol, PolicySHA256: pol.SHA256(),
		Labelled: &eval.LabelledEvidence{Candidate: eval.Quality{N: 100, Accuracy: &accuracy}},
	}
	c.Resources.Candidate.RequestLatency = eval.Latency{N: 100, P50: p50, P95: p50 * 2, Mean: p50}
	c.Resources.Candidate.Memory = eval.Memory{Available: true, PeakRsrv: &vram}
	c.Verdict = pol.Evaluate(c)
	if c.Verdict.Status != eval.VerdictAccepted {
		t.Fatalf("fixture verdict %s", c.Verdict.Status)
	}
	if _, err := eval.SaveCertification(h, c); err != nil {
		t.Fatal(err)
	}
}

func activate(t *testing.T, h home.Home, m home.ModelManifest, variant string) {
	t.Helper()
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), home.Active{Runtime: "rt", ModelID: m.ID, Model: setup.ModelDirName(m), Device: "cuda", Variant: variant}); err != nil {
		t.Fatal(err)
	}
}

func TestTuningStoreReportsTheCandidateItsEvidenceAndTheAcceptedBaseline(t *testing.T) {
	store, h, m, _ := tuningFixture(t)
	var _ dashboard.Tuning = store
	a, err := store.Analysis(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	canonical, _ := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	p := pinnedProfile(t, a, "balanced", tuning.RegionFeedForward)
	if err := store.SaveProfile(p, a); err != nil {
		t.Fatal(err)
	}
	compiled, err := tuning.Compile(p, a)
	if err != nil {
		t.Fatal(err)
	}

	// Nothing built, nothing active: nothing is invented.
	got, err := store.Candidate(p, compiled)
	if err != nil || got.Variant != "" || got.Baseline != "" || !strings.Contains(got.BaselineWhy, "no runtime is active") || len(got.Figures) != 0 {
		t.Fatalf("empty home: %+v %v", got, err)
	}

	// A candidate built from the profile, with its per-group evidence.
	prov := tuning.Provenance(p, a, compiled)
	cand, cdir := storeVariant(t, h, m, compiled.Recipe, &prov, 3<<20)
	ev := home.TuningEvidence{Schema: home.TuningEvidenceSchema, PlanSHA256: compiled.Plan.SHA256(), AutoPolicy: tuning.AutoPolicyVersion}
	for _, g := range compiled.Plan.Groups {
		ev.Groups = append(ev.Groups, home.TuningGroupApplied{ID: g.ID, Selection: g.Selection, Requested: g.Requested, Effective: g.Effective, Required: g.Required,
			Applied: g.Effective, Preserved: g.Effective == home.PolicySourcePrecision})
	}
	if err := os.WriteFile(filepath.Join(cdir, home.TuningEvidenceFile), ev.Canonical(), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err = store.Candidate(p, compiled)
	if err != nil || got.Variant != cand.ID || got.Evidence == nil || got.Evidence.PlanSHA256 != compiled.Plan.SHA256() || got.EvidenceErr != "" || len(got.Figures) != 0 {
		t.Fatalf("candidate without a baseline: %+v %v", got, err)
	}
	if err := os.WriteFile(filepath.Join(cdir, home.TuningEvidenceFile), []byte("{broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, _ = store.Candidate(p, compiled); got.Evidence != nil || got.EvidenceErr == "" {
		t.Errorf("unreadable evidence is not reported: %+v", got)
	}
	if err := os.WriteFile(filepath.Join(cdir, home.TuningEvidenceFile), ev.Canonical(), 0o644); err != nil {
		t.Fatal(err)
	}

	// The active variant is the baseline only once its certification is accepted.
	base, bdir := storeVariant(t, h, m, canonical, nil, 5<<20)
	activate(t, h, m, base.ID)
	if got, _ = store.Candidate(p, compiled); got.Baseline != "" || !strings.Contains(got.BaselineWhy, "not certified accepted") || len(got.Figures) != 0 {
		t.Fatalf("uncertified active variant: %+v", got)
	}
	certify(t, h, m, base, "dataset", 0.95, 11, 6<<30)
	got, err = store.Candidate(p, compiled)
	if err != nil || got.Baseline != base.ID || !got.BaselineCanonical || got.BaselineEvidence != nil || got.BaselineWhy != "" {
		t.Fatalf("accepted baseline: %+v %v", got, err)
	}
	byKey := func(c dashboard.TuningCandidate) map[string]dashboard.TuningFigure {
		out := map[string]dashboard.TuningFigure{}
		for _, f := range c.Figures {
			out[f.Key] = f
		}
		return out
	}
	figs := byKey(got)
	if len(figs) != 5 || figs["size"].Evaluated || !figs["accuracy"].Evaluated {
		t.Fatalf("figures %+v", got.Figures)
	}
	// Artifact size is measured for both from their files; the candidate has no
	// certification, so its evaluated figures are NOT_CHECKED, never estimated.
	if f := figs["size"]; f.Baseline.State != dashboard.Measured || f.Baseline.Value != gib(5<<20) || f.Candidate.State != dashboard.Measured || f.Candidate.Value != gib(3<<20) {
		t.Errorf("size %+v", f)
	}
	for _, key := range []string{"accuracy", "fidelity", "latency", "vram"} {
		f := figs[key]
		if f.Baseline.State != dashboard.Measured || f.Baseline.Value == "" || f.Candidate.State != dashboard.NotChecked || f.Candidate.Value != "" || f.Candidate.Basis == "" {
			t.Errorf("%s baseline %+v candidate %+v", key, f.Baseline, f.Candidate)
		}
	}
	if got.Comparable || !strings.Contains(got.ComparableWhy, "no certification record") {
		t.Errorf("a pair with one certification is comparable: %+v", got)
	}
	if f := figs["accuracy"]; !strings.Contains(f.Baseline.Value, "0.9500") || !strings.Contains(f.Baseline.Basis, "certification of variant "+base.ID) {
		t.Errorf("baseline accuracy %+v", f.Baseline)
	}
	if f := figs["vram"]; f.Baseline.Value != "peak VRAM reserved "+gib(6<<30) || figs["latency"].Baseline.Value != "p50 11.0 ms · p95 22.0 ms over 100 requests" {
		t.Errorf("baseline resources %+v / %+v", f, figs["latency"])
	}

	// Both certified on the same dataset and questions: comparable, measured.
	certify(t, h, m, cand, "dataset", 0.90, 12, 5<<30)
	got, _ = store.Candidate(p, compiled)
	figs = byKey(got)
	if !got.Comparable || got.ComparableWhy != "" {
		t.Errorf("same evaluation is not comparable: %+v", got)
	}
	for _, key := range []string{"accuracy", "fidelity", "latency", "vram", "size"} {
		if f := figs[key]; f.Baseline.State != dashboard.Measured || f.Candidate.State != dashboard.Measured || f.Candidate.Value == "" {
			t.Errorf("%s: %+v", key, f)
		}
	}
	if !strings.Contains(figs["accuracy"].Candidate.Value, "0.9000") || figs["vram"].Candidate.Value != "peak VRAM reserved "+gib(5<<30) {
		t.Errorf("candidate figures %+v", figs)
	}

	// Different datasets are not comparable, and the reason is stated.
	if err := os.RemoveAll(h.Path("state", "certifications", cand.ID)); err != nil {
		t.Fatal(err)
	}
	certify(t, h, m, cand, "another-dataset", 0.90, 12, 5<<30)
	if got, _ = store.Candidate(p, compiled); got.Comparable || !strings.Contains(got.ComparableWhy, "different datasets or questions") {
		t.Errorf("different evaluations were compared: %+v", got)
	}

	// The candidate itself active and accepted is the baseline: no self comparison.
	activate(t, h, m, cand.ID)
	if got, _ = store.Candidate(p, compiled); got.Baseline != cand.ID || !strings.Contains(got.BaselineWhy, "is the accepted baseline") || len(got.Figures) != 0 {
		t.Errorf("self comparison: %+v", got)
	}

	// A baseline that records per-group evidence is reported with it.
	if err := os.WriteFile(filepath.Join(bdir, home.TuningEvidenceFile), ev.Canonical(), 0o644); err != nil {
		t.Fatal(err)
	}
	activate(t, h, m, base.ID)
	certify(t, h, m, base, "dataset", 0.95, 11, 6<<30)
	if got, _ = store.Candidate(p, compiled); got.BaselineEvidence == nil || !got.BaselineCanonical {
		t.Errorf("baseline evidence: %+v", got)
	}
	// A variant of another source, or one not of this exact source, is no baseline.
	activate(t, h, home.ModelManifest{ID: "laya-base"}, "")
	if got, _ = store.Candidate(p, compiled); got.Baseline != "" || !strings.Contains(got.BaselineWhy, "not a variant of") {
		t.Errorf("another model active: %+v", got)
	}
}
