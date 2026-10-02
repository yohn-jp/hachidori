package eval

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
)

const certModel = "clef-flash"

// certSource is the catalog model of the fixture (revision rev1 is the one the
// fake endpoint reports).
func certSource() home.ModelManifest {
	return home.ModelManifest{ID: certModel, Provider: "clef", Repo: "o/clef-flash", Revision: "rev1",
		Files: map[string]string{"config.json": strings.Repeat("1", 64), "joint_head.safetensors": strings.Repeat("2", 64)}}
}

func certRecipe() home.Recipe {
	return home.Recipe{Schema: home.RecipeSchema, Name: "t-w4a16", Engine: "llmcompressor", Scheme: "W4A16", Algorithm: "rtn", Targets: []string{"Linear"},
		Preserved: []home.PreservedModule{{Pattern: "lm_head", Scope: home.ScopeBackbone, Precision: "bfloat16", Reason: "r"}}}
}

func certVariant() home.VariantManifest {
	v := home.VariantManifest{Source: home.SourceOf(certSource()), Provider: "clef",
		Optimizer: home.Optimizer{Engine: "llmcompressor", Version: "0.14.0", Runtime: "optimizer-cpu-x", Device: "cpu"}, Recipe: certRecipe(),
		Weights: home.WeightPrecision{Scheme: "W4A16", Bits: 4, GroupSize: 128, Symmetric: true, Format: "compressed-tensors/pack-quantized", DType: "bfloat16"},
		Files:   map[string]string{"config.json": strings.Repeat("3", 64)}, Creation: home.Creation{CreatedAt: "2026-01-01T00:00:00Z", Platform: "linux/amd64", Command: "c"}}
	v.Seal()
	return v
}

// scripted answers: probabilities of three options as a function of the case index.
type script func(i int) map[string]float64

func distribution(a, b, c float64) map[string]float64 {
	return map[string]float64{"x": a, "y": b, "z": c}
}

func topOf(p map[string]float64) (string, float64) {
	best := ""
	for _, k := range []string{"x", "y", "z"} {
		if best == "" || p[k] > p[best] {
			best = k
		}
	}
	return best, p[best]
}

func certCases(n int, labelled bool) []Case {
	var cs []Case
	q := api.Question{ID: "q1", Type: "choice", Instructions: "pick", Choices: []string{"x", "y", "z"}}
	for i := 0; i < n; i++ {
		c := Case{ID: fmt.Sprintf("c%03d", i), State: fmt.Sprintf("state %d", i), Questions: []api.Question{q}, Expected: map[string]string{}}
		if labelled {
			c.Expected["q1"] = []string{"x", "y", "z"}[i%3]
		}
		cs = append(cs, c)
	}
	return cs
}

// endpoint is a one-resident fake System One endpoint answering from a script.
type scriptedEndpoint struct {
	*fakeResidents
	cases map[string]int // state -> case index
	s     script
}

func newScripted(role string, v home.VariantManifest, cases []Case, s script, mutate func(*fakeResident)) *scriptedEndpoint {
	idx := map[string]int{}
	for i, c := range cases {
		idx[c.State] = i
	}
	r := &fakeResident{id: certModel, provider: "clef", running: true, pid: 10, starts: 1, uptime: 30, loadMS: 1000, warmupMS: 100,
		alloc: 4 << 30, rsrv: 5 << 30, dtype: "torch.float32", device: "cpu", noAccel: true}
	if role == "candidate" {
		r.dtype, r.device, r.noAccel = "torch.bfloat16", "cuda", false
		r.provExtra = map[string]any{"variant_id": v.ID, "quantized_execution": "compressed-tensors/pack-quantized W4A16 weights, bfloat16 compute"}
	}
	if mutate != nil {
		mutate(r)
	}
	e := &scriptedEndpoint{fakeResidents: &fakeResidents{residents: []*fakeResident{r}}, cases: idx, s: s}
	r.answer = func(state string, q api.Question) (string, float64) { return topOf(s(idx[state])) }
	return e
}

// Decide is overridden so the probabilities are exactly the script's.
func (e *scriptedEndpoint) Decide(req api.DecideRequest) (api.DecideResponse, error) {
	resp, err := e.fakeResidents.Decide(req)
	if err != nil {
		return resp, err
	}
	for i := range resp.Results {
		p := e.s(e.cases[req.State])
		_, conf := topOf(p)
		resp.Results[i].Probabilities, resp.Results[i].Confidence = p, conf
	}
	return resp, nil
}

func runRole(t *testing.T, role string, v home.VariantManifest, cases []Case, labelled bool, s script, mutate func(*fakeResident)) ResidentRun {
	t.Helper()
	e := newScripted(role, v, cases, s, mutate)
	r, err := RunResident(e, cases, "dataset-sha", labelled, certModel, ResidentOptions{Options: Options{Warmup: 0, Passes: 1}})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func certify(t *testing.T, ref, cand ResidentRun, policy CertPolicy) (Certification, error) {
	t.Helper()
	return Certify(CertifyInput{Source: certSource(), Variant: certVariant(), Reference: ref, Candidate: cand, Policy: policy, Now: time.Unix(1_700_000_000, 0)})
}

// refScript: the reference is confident in the expected-looking option.
func refScript(i int) map[string]float64 {
	switch i % 3 {
	case 0:
		return distribution(0.8, 0.15, 0.05)
	case 1:
		return distribution(0.1, 0.8, 0.1)
	}
	return distribution(0.05, 0.15, 0.8)
}

// A candidate that flips the first k decisions to another option and shifts
// probabilities slightly elsewhere.
func driftScript(flips int) script {
	return func(i int) map[string]float64 {
		p := refScript(i)
		if i < flips {
			top, _ := topOf(p)
			other := map[string]string{"x": "y", "y": "z", "z": "x"}[top]
			q := distribution(0.1, 0.1, 0.1)
			q[other], q[top] = 0.8, 0.1
			return q
		}
		q := map[string]float64{}
		for k, v := range p {
			q[k] = v
		}
		q["x"] += 0.01
		q["y"] -= 0.01
		return q
	}
}

func TestJSDivergenceIsBoundedAndSymmetric(t *testing.T) {
	p, q := map[string]float64{"a": 0.5, "b": 0.5}, map[string]float64{"a": 1, "b": 0}
	got := jsBits(p, q)
	if math.Abs(got-0.31127812445913283) > 1e-12 || math.Abs(jsBits(q, p)-got) > 1e-15 {
		t.Fatalf("JS = %v", got)
	}
	if jsBits(p, p) != 0 {
		t.Fatal("identical distributions diverge")
	}
	if d := jsBits(map[string]float64{"a": 1, "b": 0}, map[string]float64{"a": 0, "b": 1}); math.Abs(d-1) > 1e-12 {
		t.Fatalf("disjoint distributions have JS %v, want 1 bit", d)
	}
}

// Without labels: flips, the transition matrix, probability and confidence
// drift, divergence, rank preservation and high-confidence flips are all
// measured, and no labelled section exists.
func TestCertifyGroundTruthFreeFidelity(t *testing.T) {
	v, cases := certVariant(), certCases(100, false)
	ref := runRole(t, "reference", v, cases, false, refScript, nil)
	cand := runRole(t, "candidate", v, cases, false, driftScript(6), nil)
	c, err := certify(t, ref, cand, DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	f := c.Fidelity
	if f.Paired != 100 || f.Unpaired != 0 || f.ChoiceFlips != 6 || math.Abs(f.FlipRate-0.06) > 1e-12 || math.Abs(f.TopChoicePreservedRate-0.94) > 1e-12 {
		t.Fatalf("fidelity %+v", f)
	}
	if len(f.Flips) != 6 || f.Flips[0].CaseID != "c000" || f.Flips[0].Effect != "" {
		t.Fatalf("flips %+v", f.Flips)
	}
	if len(f.Transitions) != 1 || f.Transitions[0].Question != "q1" || !reflect.DeepEqual(f.Transitions[0].Choices, []string{"x", "y", "z"}) {
		t.Fatalf("transitions %+v", f.Transitions)
	}
	m, diag, off := f.Transitions[0].Counts, 0, 0
	for i := range m {
		for j := range m[i] {
			if i == j {
				diag += m[i][j]
			} else {
				off += m[i][j]
			}
		}
	}
	if diag != 94 || off != 6 {
		t.Fatalf("transition matrix %v: agreement %d, disagreement %d", m, diag, off)
	}
	// Cases 0..5 are x,y,z,x,y,z -> flipped to y,z,x,y,z,x.
	if m[0][1] != 2 || m[1][2] != 2 || m[2][0] != 2 {
		t.Fatalf("transition matrix off-diagonal %v", m)
	}
	if f.JS.N != 100 || f.JS.Mean <= 0 || f.JS.Max <= 0.3 || f.JS.Max > 1 {
		t.Fatalf("js %+v", f.JS)
	}
	if f.ProbabilityMaxAbs.Max < 0.6 || f.ProbabilityMeanAbs.Mean <= 0 || f.Confidence.Abs.Max <= 0 {
		t.Fatalf("probability %+v / %+v / %+v", f.ProbabilityMaxAbs, f.ProbabilityMeanAbs, f.Confidence)
	}
	if f.RankPreserved >= 100 || f.RankPreservedRate >= 1 {
		t.Fatalf("rank preservation %d of 100 with 6 flips", f.RankPreserved)
	}
	// Reference confidence is 0.8 everywhere: below the 0.9 high-confidence line.
	if f.HighConfidenceRef.N != 0 || f.HighConfidenceRef.FlipRate != nil {
		t.Fatalf("high confidence %+v", f.HighConfidenceRef)
	}
	if c.Labelled != nil || c.Dataset.Labelled {
		t.Fatal("an unlabelled run reports labelled quality")
	}
	if c.Reference.VariantID != "" || c.Candidate.VariantID != v.ID || c.Reference.DType != "torch.float32" || c.Candidate.Device != "cuda" {
		t.Fatalf("executions %+v %+v", c.Reference, c.Candidate)
	}
	if c.NoulScoreDriftSupported() || !strings.Contains(f.NoulScoreDrift, "not measurable") || f.QuestionTypes["choice"] != 100 {
		t.Fatalf("question types %+v %q", f.QuestionTypes, f.NoulScoreDrift)
	}
}

// NoulScoreDriftSupported documents that the schema can carry noul/score drift
// once those question types are served; today it never claims it.
func (c Certification) NoulScoreDriftSupported() bool {
	return c.Fidelity.QuestionTypes["noul"] > 0 || c.Fidelity.QuestionTypes["score"] > 0
}

func TestHighConfidenceReferenceFlips(t *testing.T) {
	v, cases := certVariant(), certCases(100, false)
	confident := func(i int) map[string]float64 { return distribution(0.95, 0.03, 0.02) }
	flipFive := func(i int) map[string]float64 {
		if i < 5 {
			return distribution(0.2, 0.7, 0.1)
		}
		return confident(i)
	}
	ref := runRole(t, "reference", v, cases, false, confident, nil)
	cand := runRole(t, "candidate", v, cases, false, flipFive, nil)
	c, err := certify(t, ref, cand, DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	h := c.Fidelity.HighConfidenceRef
	if h.N != 100 || h.Flips != 5 || h.FlipRate == nil || math.Abs(*h.FlipRate-0.05) > 1e-12 || len(h.Items) != 5 {
		t.Fatalf("high-confidence flips %+v", h)
	}
	if c.Verdict.Status != VerdictRejected || !contains(c.Verdict.Failed(), "high_confidence_flip_rate") || !contains(c.Verdict.Failed(), "flip_rate") {
		t.Fatalf("verdict %+v", c.Verdict)
	}
}

func TestIdenticalRunsAreAccepted(t *testing.T) {
	v, cases := certVariant(), certCases(100, true)
	ref := runRole(t, "reference", v, cases, true, refScript, nil)
	cand := runRole(t, "candidate", v, cases, true, refScript, nil)
	c, err := certify(t, ref, cand, DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if c.Fidelity.ChoiceFlips != 0 || c.Fidelity.JS.Max != 0 || c.Fidelity.RankPreservedRate != 1 || c.Verdict.Status != VerdictAccepted {
		t.Fatalf("identical runs: %+v / %+v", c.Fidelity, c.Verdict)
	}
	if c.Labelled == nil || c.Labelled.Delta.Accuracy == nil || *c.Labelled.Delta.Accuracy != 0 {
		t.Fatalf("labelled %+v", c.Labelled)
	}
}

// Labelled runs additionally report quality and calibration deltas, computed
// by the existing metrics, with slices; the flips carry their effect on
// correctness.
func TestCertifyLabelledQualityDeltas(t *testing.T) {
	v, cases := certVariant(), certCases(99, true) // expected labels x,y,z,x,...
	ref := runRole(t, "reference", v, cases, true, refScript, nil)
	cand := runRole(t, "candidate", v, cases, true, driftScript(9), nil)
	c, err := certify(t, ref, cand, DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	l := c.Labelled
	if l == nil {
		t.Fatal("no labelled evidence")
	}
	// The reference is right everywhere, the candidate wrong on 9 flips.
	if *l.Reference.Accuracy != 1 || math.Abs(*l.Candidate.Accuracy-90.0/99) > 1e-12 || math.Abs(*l.Delta.Accuracy-(90.0/99-1)) > 1e-12 {
		t.Fatalf("accuracy %v %v %v", *l.Reference.Accuracy, *l.Candidate.Accuracy, *l.Delta.Accuracy)
	}
	if l.FlipsBroken != 9 || l.FlipsFixed != 0 || l.FlipsBothWrong != 0 {
		t.Fatalf("flip effects %+v", l)
	}
	for _, e := range c.Fidelity.Flips {
		if e.Effect != FlipBroken {
			t.Fatalf("flip effect %q", e.Effect)
		}
	}
	// The deltas are the existing QualityOf measures, not a second formula.
	if want := QualityOf(ref.Run.Observations2(), DefaultHighConfidence, ref.Declared.Thresholds); *want.Calibration.Brier != *l.Reference.Calibration.Brier {
		t.Fatal("reference quality differs from QualityOf")
	}
	if l.Delta.ECE == nil || l.Delta.Brier == nil || l.Delta.NLL == nil || l.Delta.MacroF1 == nil || *l.Delta.Brier <= 0 || *l.Delta.NLL <= 0 {
		t.Fatalf("calibration deltas %+v", l.Delta)
	}
	if len(l.Thresholds) != len(DefaultThresholds) || len(l.PerQuestion) != 1 || l.PerQuestion[0].Key != "q1" || len(l.PerFamily) != 1 || l.PerFamily[0].Key != UnassignedFamily {
		t.Fatalf("slices %+v %+v %+v", l.Thresholds, l.PerQuestion, l.PerFamily)
	}
	if l.Delta.HighConfidenceErrors != l.Candidate.HighConfidence.Errors-l.Reference.HighConfidence.Errors {
		t.Fatalf("high-confidence error delta %+v", l.Delta)
	}
	if c.Verdict.Status != VerdictRejected || !contains(c.Verdict.Failed(), "accuracy_drop") {
		t.Fatalf("verdict %+v", c.Verdict)
	}
}

// Observations2 is the plain observations of a run, for recomputing quality.
func (m ModelRun) Observations2() []Observation {
	out := make([]Observation, len(m.Observations))
	for i, o := range m.Observations {
		out[i] = o.Observation
	}
	return out
}

// Evidence and policy are separate: two policies over the same runs produce
// the same evidence and, at most, different verdicts.
func TestPolicyDoesNotChangeEvidence(t *testing.T) {
	v, cases := certVariant(), certCases(100, true)
	ref := runRole(t, "reference", v, cases, true, refScript, nil)
	cand := runRole(t, "candidate", v, cases, true, driftScript(4), nil)
	strict, lenient := DefaultPolicy(), DefaultPolicy()
	strict.ID, strict.MaxFlipRate = "strict/1", 0.01
	lenient.ID, lenient.MaxFlipRate, lenient.MaxProbabilityDelta, lenient.MaxMeanJS = "lenient/1", 0.10, 1, 0.1
	lenient.Labelled.MaxAccuracyDrop, lenient.Labelled.MaxMacroF1Drop, lenient.Labelled.MaxBrierIncrease, lenient.Labelled.MaxECEIncrease, lenient.Labelled.MaxNLLIncrease = 0.2, 0.2, 0.5, 0.5, 1
	a, err := certify(t, ref, cand, strict)
	if err != nil {
		t.Fatal(err)
	}
	b, err := certify(t, ref, cand, lenient)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Fidelity, b.Fidelity) {
		t.Fatal("the policy changed the fidelity evidence")
	}
	if !reflect.DeepEqual(a.Labelled, b.Labelled) {
		t.Fatal("the policy changed the labelled evidence")
	}
	if !reflect.DeepEqual(a.Resources, b.Resources) {
		t.Fatal("the policy changed the resource evidence")
	}
	if a.Verdict.Status != VerdictRejected || b.Verdict.Status != VerdictAccepted || a.PolicySHA256 == b.PolicySHA256 {
		t.Fatalf("verdicts %s %s (lenient failed %v)", a.Verdict.Status, b.Verdict.Status, b.Verdict.Failed())
	}
}

// Thresholds are inclusive: a value equal to its limit passes, the next one fails.
func TestPolicyThresholdBoundaries(t *testing.T) {
	v, cases := certVariant(), certCases(100, false)
	ref := runRole(t, "reference", v, cases, false, refScript, nil)
	cand := runRole(t, "candidate", v, cases, false, driftScript(3), nil) // flip rate exactly 0.03
	p := DefaultPolicy()
	p.MaxMeanJS, p.MaxP95ProbabilityDelta, p.MaxProbabilityDelta, p.MaxHighConfidenceFlipRate = 1, 1, 1, 1
	c, err := certify(t, ref, cand, p)
	if err != nil {
		t.Fatal(err)
	}
	if c.Fidelity.FlipRate != 0.03 || c.Verdict.Status != VerdictAccepted {
		t.Fatalf("flip rate %v at the limit: %+v", c.Fidelity.FlipRate, c.Verdict)
	}
	p.MaxFlipRate = math.Nextafter(0.03, 0)
	c, _ = certify(t, ref, cand, p)
	if c.Verdict.Status != VerdictRejected || !reflect.DeepEqual(c.Verdict.Failed(), []string{"flip_rate"}) {
		t.Fatalf("just over the limit: %+v", c.Verdict)
	}
	// Too little evidence is not acceptance.
	p = DefaultPolicy()
	p.MinObservations = 101
	if c, _ := certify(t, ref, cand, p); !contains(c.Verdict.Failed(), "min_observations") {
		t.Fatalf("min observations: %+v", c.Verdict)
	}
	// Labelled criteria are not applied (and not failed) without labels.
	for _, cr := range c.Verdict.Criteria {
		if cr.Name == "accuracy_drop" && cr.Applied {
			t.Fatal("a labelled criterion was applied to an unlabelled run")
		}
	}
	// Invalid policies are refused.
	bad := DefaultPolicy()
	bad.ID = "no-version"
	if _, err := certify(t, ref, cand, bad); err == nil {
		t.Fatal("an unversioned policy was accepted")
	}
	bad = DefaultPolicy()
	bad.MaxFlipRate = 2
	if err := bad.Validate(); err == nil {
		t.Fatal("an out-of-range threshold was accepted")
	}
}

// Missing or mismatched identities are refused, every reason listed, and no
// deltas are computed.
func TestCertifyRefusesMismatchedIdentities(t *testing.T) {
	v, cases := certVariant(), certCases(40, false)
	good := func() (ResidentRun, ResidentRun) {
		return runRole(t, "reference", v, cases, false, refScript, nil), runRole(t, "candidate", v, cases, false, refScript, nil)
	}
	other := certVariant()
	other.Calibration = &home.Calibration{ID: "x", SHA256: strings.Repeat("a", 64)}
	other.Seal()
	for name, tc := range map[string]struct {
		build func() (ResidentRun, ResidentRun)
		want  string
	}{
		"candidate is another variant": {func() (ResidentRun, ResidentRun) {
			return runRole(t, "reference", v, cases, false, refScript, nil), runRole(t, "candidate", other, cases, false, refScript, nil)
		}, "executed variant"},
		"candidate is not a variant": {func() (ResidentRun, ResidentRun) {
			return runRole(t, "reference", v, cases, false, refScript, nil), runRole(t, "reference", v, cases, false, refScript, nil)
		}, "executed variant"},
		"reference is a variant": {func() (ResidentRun, ResidentRun) {
			return runRole(t, "candidate", v, cases, false, refScript, nil), runRole(t, "candidate", v, cases, false, refScript, nil)
		}, "the reference must be the source model"},
		"reference is not high precision": {func() (ResidentRun, ResidentRun) {
			return runRole(t, "reference", v, cases, false, refScript, func(r *fakeResident) { r.dtype = "torch.int8" }), runRole(t, "candidate", v, cases, false, refScript, nil)
		}, "high-precision"},
		"candidate dtype is not the declared execution": {func() (ResidentRun, ResidentRun) {
			return runRole(t, "reference", v, cases, false, refScript, nil), runRole(t, "candidate", v, cases, false, refScript, func(r *fakeResident) { r.dtype = "torch.float32" })
		}, "declares torch.bfloat16"},
		"wrong model": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			r.Run.Identity.Runtime["model_id"] = "laya-base"
			return r, c
		}, "not the source model"},
		"wrong revision": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			c.Run.Identity.Provider["model_revision"] = "rev2"
			return r, c
		}, "not the source revision"},
		"device unreported": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			delete(r.Run.Identity.Provider, "dtype")
			return r, c
		}, "does not report the device and dtype"},
		"labelled mismatch": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			r.Labelled = true
			return r, c
		}, "one run is labelled"},
		"dataset differs": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			c.Run.DatasetSHA256, c.DatasetSHA256 = "other", "other"
			return r, c
		}, "dataset"},
		"normalized inputs differ": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			c.Run.InputSHA256 = "other"
			return r, c
		}, "normalized input digest"},
		"requests sent differ": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			c.Run.SentSHA256 = "other"
			return r, c
		}, "actually sent"},
		"question identity differs": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			c.Run.Questions = []QuestionRecord{{ID: "q1", SHA256: "different"}}
			return r, c
		}, "question"},
		"declared controls differ": {func() (ResidentRun, ResidentRun) {
			r, c := good()
			c.Declared.HighConfidence = 0.5
			return r, c
		}, "declared"},
	} {
		t.Run(name, func(t *testing.T) {
			ref, cand := tc.build()
			c, err := certify(t, ref, cand, DefaultPolicy())
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err %v, want %q", err, tc.want)
			}
			if c.Schema != "" || len(c.Fidelity.Flips) != 0 {
				t.Fatal("a refused certification carries evidence")
			}
		})
	}
	// A variant that does not derive from the catalog source is refused.
	ref, cand := good()
	src := certSource()
	src.Revision = "rev9"
	if _, err := Certify(CertifyInput{Source: src, Variant: v, Reference: ref, Candidate: cand, Policy: DefaultPolicy()}); err == nil {
		t.Fatal("a variant was certified against a source it does not derive from")
	}
	// Both runs must be valid documents.
	ref.Run.Identity.Digest = ""
	if _, err := certify(t, ref, cand, DefaultPolicy()); err == nil {
		t.Fatal("a run without an identity was certified")
	}
}

// Resource evidence is separate and honest: load/warmup, latency and memory
// where reported, NOT_CHECKED where not, never invented.
func TestCertifyResourceEvidenceIsHonest(t *testing.T) {
	v, cases := certVariant(), certCases(30, false)
	ref := runRole(t, "reference", v, cases, false, refScript, nil) // CPU: no accelerator statistics
	cand := runRole(t, "candidate", v, cases, false, refScript, nil)
	c, err := certify(t, ref, cand, DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	r := c.Resources
	if r.Checks["reference_vram"] != FactNotChecked || r.Checks["candidate_vram"] != FactMeasured {
		t.Fatalf("vram checks %v", r.Checks)
	}
	if r.Checks["reference_load_time"] != FactMeasured || *r.Reference.LoadMS != 1000 || r.Checks["reference_request_latency"] != FactMeasured {
		t.Fatalf("checks %v", r.Checks)
	}
	if r.Checks["reference_host_ram"] != FactNotChecked || r.Checks["candidate_host_ram"] != FactNotChecked {
		t.Fatalf("host ram was not reported but is marked: %v", r.Checks)
	}
	if r.Memory.PeakAllocated != nil || r.HostRSS != nil {
		t.Fatalf("a ratio was invented from a side that reported nothing: %+v", r.Memory)
	}
	if r.Reference.Requests != 30 || r.Candidate.ErrorCount != 0 {
		t.Fatalf("counts %+v", r)
	}
	// Host RAM reported by the worker is carried through when present.
	ref2 := runRole(t, "reference", v, cases, false, refScript, nil)
	ref2.Run.Memory.PeakHostRSS = new(int64)
	*ref2.Run.Memory.PeakHostRSS = 20 << 30
	cand2 := runRole(t, "candidate", v, cases, false, refScript, nil)
	cand2.Run.Memory.PeakHostRSS = new(int64)
	*cand2.Run.Memory.PeakHostRSS = 4 << 30
	c2, _ := certify(t, ref2, cand2, DefaultPolicy())
	if c2.Resources.Checks["reference_host_ram"] != FactMeasured || c2.Resources.HostRSS == nil || *c2.Resources.HostRSS.Ratio != 0.2 {
		t.Fatalf("host ram %+v %v", c2.Resources.HostRSS, c2.Resources.Checks)
	}
}

// Request errors and unpaired observations count against acceptance; the
// evidence is still produced.
func TestCertifyErrorsAndUnpairedBlockAcceptance(t *testing.T) {
	v, cases := certVariant(), certCases(100, false)
	ref := runRole(t, "reference", v, cases, false, refScript, nil)
	cand := runRole(t, "candidate", v, cases, false, refScript, nil)
	cand.Run.Observations = cand.Run.Observations[:99]
	cand.Run.Quality.N = 99
	c, err := certify(t, ref, cand, DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if c.Fidelity.Unpaired != 1 || c.Verdict.Status != VerdictRejected || !contains(c.Verdict.Failed(), "unpaired_and_errors") {
		t.Fatalf("unpaired %d, verdict %+v", c.Fidelity.Unpaired, c.Verdict)
	}
	cand = runRole(t, "candidate", v, cases, false, refScript, func(r *fakeResident) {})
	cand.Run.Memory.Available = true
	unstable := cand
	unstable.Run.ResidentStable = false
	c, _ = certify(t, ref, unstable, DefaultPolicy())
	if c.Verdict.Status != VerdictRejected || !contains(c.Verdict.Failed(), "stable_residents") {
		t.Fatalf("an unstable resident was accepted: %+v", c.Verdict)
	}
}

func TestRunResidentIsDirectAndSingleModel(t *testing.T) {
	f := newFake()
	cases := certCases(5, true)
	r, err := RunResident(f, cases, "sum", true, "laya-base", ResidentOptions{Options: Options{Passes: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schema != ResidentRunSchema || r.Run.Model != "laya-base" || len(r.Run.Observations) != 5 || !r.Labelled || f.find("opendecider-nano").decides != 0 {
		t.Fatalf("run %+v (other resident answered %d)", r.Run.Model, f.find("opendecider-nano").decides)
	}
	if err := ValidateResidentRun(r); err != nil {
		t.Fatal(err)
	}
	// A model that is not resident fails and nothing answers in its place.
	if _, err := RunResident(f, cases, "sum", true, "clef-flash", ResidentOptions{Options: Options{Passes: 1}}); err == nil {
		t.Fatal("a missing resident was recorded")
	}
	// Unlabelled: nothing about correctness is recorded.
	u, err := RunResident(f, certCases(5, false), "sum", false, "laya-base", ResidentOptions{Options: Options{Passes: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if u.Labelled || u.Run.Quality.Accuracy != nil || u.Run.Observations[0].Expected != "" || u.Run.Observations[0].Correct {
		t.Fatalf("unlabelled run carries label-based quality: %+v", u.Run.Quality)
	}
	// Round trip through a file.
	path := filepath.Join(t.TempDir(), "run.json")
	b, _ := marshalIndent(r)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	back, err := LoadResidentRun(path)
	if err != nil || back.Run.Identity.Digest != r.Run.Identity.Digest {
		t.Fatalf("round trip: %v", err)
	}
	os.WriteFile(path, []byte(strings.Replace(string(b), ResidentRunSchema, "other/1", 1)), 0o644)
	if _, err := LoadResidentRun(path); err == nil {
		t.Fatal("another schema decoded")
	}
}

func TestLoadAnyLabelling(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		os.WriteFile(p, []byte(body), 0o644)
		return p
	}
	q := `"questions":[{"id":"q","type":"choice","instructions":"i","choices":["a","b"]}]`
	un := write("u.jsonl", `{"id":"1","state":"s",`+q+`}`+"\n"+`{"id":"2","state":"t",`+q+`}`+"\n")
	if cs, _, labelled, err := LoadAny(un, nil); err != nil || labelled || len(cs) != 2 {
		t.Fatalf("unlabelled dataset: %v %v", labelled, err)
	}
	if _, _, err := Load(un, nil); err == nil {
		t.Fatal("Load accepted a dataset without labels (existing behavior must stay strict)")
	}
	lab := write("l.jsonl", `{"id":"1","state":"s",`+q+`,"expected":{"q":"a"}}`+"\n")
	if _, _, labelled, err := LoadAny(lab, nil); err != nil || !labelled {
		t.Fatalf("labelled dataset: %v %v", labelled, err)
	}
	mixed := write("m.jsonl", `{"id":"1","state":"s",`+q+`,"expected":{"q":"a"}}`+"\n"+`{"id":"2","state":"t",`+q+`}`+"\n")
	if _, _, _, err := LoadAny(mixed, nil); err == nil || !strings.Contains(err.Error(), "for every question or for none") {
		t.Fatalf("partly labelled dataset: %v", err)
	}
}

func marshalIndent(v any) ([]byte, error) { return jsonIndent(v) }

// A real certification is recorded, then resolved from its records only; the
// record binds the variant manifest digest, so another manifest does not
// inherit it, and a forged verdict is never trusted.
func TestSaveAndResolveCertification(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	v, cases := certVariant(), certCases(100, true)
	ref := runRole(t, "reference", v, cases, true, refScript, nil)
	good := runRole(t, "candidate", v, cases, true, refScript, nil)
	bad := runRole(t, "candidate", v, cases, true, driftScript(30), nil)
	if st := ResolveCertification(h, v); st.State != StateUncertified || len(st.Problems) != 0 {
		t.Fatalf("no records: %+v", st)
	}
	c1, err := Certify(CertifyInput{Source: certSource(), Variant: v, Reference: ref, Candidate: bad, Policy: DefaultPolicy(), Now: time.Now().Add(-time.Hour)})
	if err != nil {
		t.Fatal(err)
	}
	rec1, err := SaveCertification(h, c1)
	if err != nil {
		t.Fatal(err)
	}
	if rec1.Verdict != VerdictRejected || len(rec1.Failed) == 0 || rec1.VariantID != v.ID || rec1.VariantManifestSHA256 != v.ManifestSHA256() {
		t.Fatalf("record %+v", rec1)
	}
	if st := ResolveCertification(h, v); st.State != StateRejected || st.Record == nil {
		t.Fatalf("rejecting record: %+v", st)
	}
	loaded, err := LoadCertification(h, rec1)
	if err != nil || loaded.Verdict.Status != VerdictRejected || !reflect.DeepEqual(loaded.Fidelity, c1.Fidelity) {
		t.Fatalf("the rejected certification was not kept whole: %v", err)
	}
	c2, err := Certify(CertifyInput{Source: certSource(), Variant: v, Reference: ref, Candidate: good, Policy: DefaultPolicy(), Now: time.Now()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SaveCertification(h, c2); err != nil {
		t.Fatal(err)
	}
	if st := ResolveCertification(h, v); st.State != StateAccepted {
		t.Fatalf("the later accepted record does not decide: %+v", st)
	}
	// The same evidence under a different manifest digest (another variant manifest) is not inherited.
	other := v
	other.Creation.Command = "another"
	if st := ResolveCertification(h, other); st.State != StateUncertified || len(st.Problems) == 0 {
		t.Fatalf("a different manifest inherited the certification: %+v", st)
	}
	// Deleting the report makes its record untrusted, not accepted.
	st := ResolveCertification(h, v)
	os.Remove(filepath.Join(h.Root, filepath.FromSlash(home.CertificationDir(v.ID)), st.Record.Report))
	if st := ResolveCertification(h, v); st.State != StateRejected {
		t.Fatalf("with the accepted report gone the older rejecting record should decide: %+v", st)
	}
}

func TestPolicyDecodeIsStrict(t *testing.T) {
	b, _ := jsonIndent(DefaultPolicy())
	if p, err := DecodePolicy(b); err != nil || p.SHA256() != DefaultPolicy().SHA256() {
		t.Fatalf("default policy round trip: %v", err)
	}
	if _, err := DecodePolicy([]byte(strings.Replace(string(b), `"min_observations"`, `"min_obs"`, 1))); err == nil {
		t.Fatal("an unknown field was accepted")
	}
	if _, err := DecodePolicy(append(b, []byte(`{}`)...)); err == nil {
		t.Fatal("trailing data was accepted")
	}
	// Any threshold change is a different policy identity digest.
	p := DefaultPolicy()
	p.MaxFlipRate = 0.04
	if p.SHA256() == DefaultPolicy().SHA256() {
		t.Fatal("a threshold change kept the policy digest")
	}
}
