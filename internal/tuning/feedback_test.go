package tuning

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// feedbackAnalysis has three linear-attention projection modules against one
// full-attention module, so the smallest unpreserved supported region is
// unique. (clefAnalysis ties them at one module each.)
func feedbackAnalysis(t *testing.T) Analysis {
	t.Helper()
	layout := clefLayout()
	layout.LinearModules = append(layout.LinearModules,
		"model.language_model.layers.0.linear_attn.in_proj_z", "model.language_model.layers.0.linear_attn.out_proj")
	analysis, err := Analyze(clefSource(t), layout)
	if err != nil {
		t.Fatal(err)
	}
	return analysis
}

func defaultProfile(t *testing.T, a Analysis) Profile {
	t.Helper()
	p, err := NewDefaultProfile(a, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// regressedContext is a compatible, bound, error-free measured regression for
// the exact profile.
func regressedContext(p Profile) EvidenceContext {
	return EvidenceContext{
		ProfileID: p.ID(), DatasetSHA256: strings.Repeat("d", 64), QuestionsSHA256: strings.Repeat("e", 64),
		Baseline:  &EvidenceRun{EvidenceSHA256: strings.Repeat("1", 64), ModelID: p.Source.ID, Revision: p.Source.Revision},
		Candidate: EvidenceRun{EvidenceSHA256: strings.Repeat("2", 64), ModelID: p.Source.ID, Revision: p.Source.Revision, VariantID: p.Source.ID + "--r--aaaaaaaaaaaa"},
		Regression: &MeasuredRegression{Cases: 20, BaselineAccuracy: 0.95, CandidateAccuracy: 0.80,
			Questions: []QuestionRegression{{Question: "q1", N: 20, Baseline: 0.95, Candidate: 0.80}}},
	}
}

func TestRecommendExplainsEvidenceBasisAndRegionChange(t *testing.T) {
	a := feedbackAnalysis(t)
	p := defaultProfile(t, a)
	ctx := regressedContext(p)
	rec, why := Recommend(ctx, p, a)
	if rec == nil {
		t.Fatalf("no recommendation: %s", why)
	}
	if rec.ProfileID != p.ID() || rec.RegionID != RegionFullAttention || rec.From != PreservationAuto || rec.To != PreservationPinned {
		t.Fatalf("recommendation %+v", rec)
	}
	basis := strings.Join(rec.Basis, "\n")
	for _, want := range []string{ctx.Baseline.EvidenceSHA256[:12], ctx.Candidate.EvidenceSHA256[:12], ctx.Candidate.VariantID,
		ctx.DatasetSHA256[:12], ctx.QuestionsSHA256[:12], "0.9500 to 0.8000 over 20 cases", "Question q1"} {
		if !strings.Contains(basis, want) {
			t.Errorf("basis lacks %q:\n%s", want, basis)
		}
	}
	for _, want := range []string{RegionFullAttention, "1 modules", RegionFullAttention + ", " + RegionLinearAttention, "does not show that this region caused it"} {
		if !strings.Contains(rec.Rationale, want) {
			t.Errorf("rationale lacks %q: %s", want, rec.Rationale)
		}
	}
	if strings.Contains(strings.ToLower(rec.Rationale+basis), "confidence") {
		t.Error("the recommendation states a confidence")
	}
	again, _ := Recommend(ctx, p, a)
	if !reflect.DeepEqual(rec, again) {
		t.Error("recommendation is not deterministic")
	}
}

func TestRecommendRefusesContextNotBoundToProfile(t *testing.T) {
	a := feedbackAnalysis(t)
	p := defaultProfile(t, a)
	other := defaultProfile(t, a)
	other.Preservation[RegionLinearAttention] = PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}
	mutate := map[string]func(*EvidenceContext){
		"other profile":       func(c *EvidenceContext) { c.ProfileID = other.ID() },
		"candidate is source": func(c *EvidenceContext) { c.Candidate.VariantID = "" },
		"candidate model":     func(c *EvidenceContext) { c.Candidate.ModelID = "laya-base" },
		"candidate revision":  func(c *EvidenceContext) { c.Candidate.Revision = "other" },
		"baseline model":      func(c *EvidenceContext) { c.Baseline.ModelID = "laya-base" },
		"baseline variant":    func(c *EvidenceContext) { c.Baseline.VariantID = "v0" },
		"no dataset":          func(c *EvidenceContext) { c.DatasetSHA256 = "" },
		"no questions":        func(c *EvidenceContext) { c.QuestionsSHA256 = "" },
	}
	for name, f := range mutate {
		ctx := regressedContext(p)
		f(&ctx)
		if err := ctx.Check(p); err == nil {
			t.Errorf("%s: context accepted", name)
		}
		if rec, why := Recommend(ctx, p, a); rec != nil || why == "" {
			t.Errorf("%s: recommendation %+v (%q) for a mismatched context", name, rec, why)
		}
	}
}

func TestRecommendReturnsNothingWhenUnsupportedOrAmbiguous(t *testing.T) {
	a := feedbackAnalysis(t)
	p := defaultProfile(t, a)
	pinned := func(regions ...string) Profile {
		q := defaultProfile(t, a)
		for _, r := range regions {
			q.Preservation[r] = PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}
		}
		return q
	}
	tie := clefAnalysis(t) // one module in each supported region
	cases := map[string]struct {
		profile  Profile
		analysis Analysis
		edit     func(*EvidenceContext)
		want     string
	}{
		"candidate evidence only":   {p, a, func(c *EvidenceContext) { c.Baseline = nil }, "needs a compatible baseline"},
		"no regression":             {p, a, func(c *EvidenceContext) { c.Regression = nil }, "no accuracy regression"},
		"accuracy did not fall":     {p, a, func(c *EvidenceContext) { c.Regression.CandidateAccuracy = c.Regression.BaselineAccuracy }, "no accuracy regression"},
		"no regressed question":     {p, a, func(c *EvidenceContext) { c.Regression.Questions = nil }, "no accuracy regression"},
		"candidate request errors":  {p, a, func(c *EvidenceContext) { c.Regression.CandidateErrors = 1 }, "confound"},
		"baseline request errors":   {p, a, func(c *EvidenceContext) { c.Regression.BaselineErrors = 2 }, "confound"},
		"supported regions applied": {pinned(RegionFullAttention, RegionLinearAttention), a, nil, "already applied"},
		"equally small regions":     {defaultProfile(t, tie), tie, nil, "equally small"},
	}
	for name, c := range cases {
		ctx := regressedContext(c.profile)
		if c.edit != nil {
			c.edit(&ctx)
		}
		rec, why := Recommend(ctx, c.profile, c.analysis)
		if rec != nil || !strings.Contains(why, c.want) {
			t.Errorf("%s: recommendation %+v, reason %q, want none mentioning %q", name, rec, why, c.want)
		}
	}
	// One supported region applied leaves the other as the only choice, even
	// where the two were tied.
	only := pinned(RegionFullAttention)
	if rec, why := Recommend(regressedContext(only), only, a); rec == nil || rec.RegionID != RegionLinearAttention {
		t.Errorf("remaining supported region: %+v %q", rec, why)
	}
	// The feed-forward projections are never proposed, however the evidence reads.
	for _, r := range []string{RegionFullAttention, RegionLinearAttention} {
		q := pinned(r)
		if rec, _ := Recommend(regressedContext(q), q, a); rec != nil && rec.RegionID == RegionFeedForward {
			t.Error("feed-forward recommended")
		}
	}
}

func TestAcceptCreatesDistinctProfileAndLeavesTheOriginalUntouched(t *testing.T) {
	a := feedbackAnalysis(t)
	p := defaultProfile(t, a)
	before := string(p.Canonical())
	rec, why := Recommend(regressedContext(p), p, a)
	if rec == nil {
		t.Fatal(why)
	}
	next, err := Accept(*rec, p, a)
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Canonical()) != before || p.Preservation[rec.RegionID].Mode != PreservationAuto {
		t.Fatal("accepting changed the profile it was derived from")
	}
	if next.ID() == p.ID() {
		t.Fatal("the accepted profile is not distinct")
	}
	if next.Preservation[rec.RegionID] != (PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}) || next.Objective != p.Objective ||
		next.Source != p.Source || next.AnalysisSHA256 != p.AnalysisSHA256 {
		t.Fatalf("accepted profile %+v", next)
	}
	for id, c := range p.Preservation {
		if id != rec.RegionID && next.Preservation[id] != c {
			t.Errorf("accepting changed region %s", id)
		}
	}
	c, err := Compile(next, a)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range c.Evidence.Regions {
		if m.RegionID == rec.RegionID && !m.Preserved {
			t.Error("the accepted region is not preserved by the compiled profile")
		}
	}
	// Refusals: another profile, an already-applied change, an unsupported region.
	if _, err := Accept(*rec, next, a); err == nil {
		t.Error("recommendation applied to a profile it was not made for")
	}
	if _, err := Accept(Recommendation{ProfileID: next.ID(), RegionID: rec.RegionID, From: PreservationAuto, To: PreservationPinned}, next, a); err == nil {
		t.Error("already-pinned region accepted again")
	}
	if _, err := Accept(Recommendation{ProfileID: p.ID(), RegionID: RegionFeedForward, From: PreservationAuto, To: PreservationPinned}, p, a); err == nil {
		t.Error("unsupported region accepted")
	}
}

// writeTunedVariant stores a sealed variant manifest whose provenance names
// the profile.
func writeTunedVariant(t *testing.T, h home.Home, p Profile, a Analysis, tuned bool) home.VariantManifest {
	t.Helper()
	c, err := Compile(p, a)
	if err != nil {
		t.Fatal(err)
	}
	v := home.VariantManifest{Source: p.Source, Provider: p.Source.Provider,
		Optimizer: home.Optimizer{Engine: c.Recipe.Engine, Version: "0.14.0", Runtime: "optimizer-cpu-x", Device: "cpu"}, Recipe: c.Recipe,
		Weights:  home.WeightPrecision{Scheme: "W4A16", Bits: 4, GroupSize: 128, Symmetric: true, Format: "compressed-tensors/pack-quantized", DType: "bfloat16"},
		Files:    map[string]string{"model.safetensors": strings.Repeat("4", 64)},
		Creation: home.Creation{CreatedAt: "2026-01-01T00:00:00Z", Platform: "linux/amd64", Command: "hachidori variant optimize"}}
	if tuned {
		v.Tuning = &home.TuningProvenance{Schema: home.TuningProvenanceSchema, Source: p.Source, ProfileID: p.ID(), ProfileSHA256: p.ID(),
			AnalysisID: a.ID(), AnalysisSHA256: a.SHA256(), CompilerVersion: RecipeCompilerVersion}
	}
	v.Seal()
	dir := h.VariantDir(p.Source.ID, v.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := home.WriteJSON(filepath.Join(dir, home.VariantManifestFile), v); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestVariantProfileResolvesExactProvenance(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	a := feedbackAnalysis(t)
	p := defaultProfile(t, a)
	tuned := writeTunedVariant(t, h, p, a, true)
	got, err := VariantProfile(h, p.Source.ID, tuned.ID)
	if err != nil || got != p.ID() {
		t.Fatalf("profile %q, %v", got, err)
	}
	untuned := writeTunedVariant(t, h, p, a, false)
	if _, err := VariantProfile(h, p.Source.ID, untuned.ID); err == nil || !strings.Contains(err.Error(), "not built from a tuning profile") {
		t.Errorf("recipe-only variant: %v", err)
	}
	if _, err := VariantProfile(h, p.Source.ID, "clef-flash--r--missing00000"); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Errorf("missing variant: %v", err)
	}
	if _, err := VariantProfile(h, "other-model", tuned.ID); err == nil {
		t.Error("variant resolved under another source")
	}
	for _, bad := range []string{"../" + tuned.ID, "a/b", `a\b`, ""} {
		if _, err := VariantProfile(h, p.Source.ID, bad); err == nil {
			t.Errorf("variant id %q accepted", bad)
		}
	}
}
