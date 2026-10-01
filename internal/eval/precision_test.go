package eval

import (
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
)

// precisionRun runs the fake comparison with OpenDecider-nano on dtype. The
// answer script flips one question and shifts the confidence of every answer.
func precisionRun(t *testing.T, dtype string, mod func(*fakeResidents)) ComparisonReport {
	t.Helper()
	f := newFake()
	f.find("opendecider-nano").dtype = dtype
	f.find("laya-base").dtype = "torch.float32"
	if dtype == "torch.bfloat16" {
		od := f.find("opendecider-nano")
		od.alloc, od.rsrv = od.alloc/2, od.rsrv/2
		od.inferenceMS = func(string) float64 { return 2.5 }
		od.answer = func(state string, q api.Question) (string, float64) {
			if state == "short state" && q.ID == "a" {
				return "no", 0.85 // flips a baseline-correct "yes"
			}
			return "yes", 0.88
		}
	} else {
		f.find("opendecider-nano").answer = func(state string, q api.Question) (string, float64) { return "yes", 0.9 }
	}
	if mod != nil {
		mod(f)
	}
	rep, err := RunResidents(f, residentCases(), "d", ResidentOptions{Options: Options{Warmup: 1, Passes: 2}, Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	return rep
}

func TestComparePrecisionReportsFlipsAndDeltasNotJustAggregates(t *testing.T) {
	base, cand := precisionRun(t, "torch.float32", nil), precisionRun(t, "torch.bfloat16", nil)
	r, err := ComparePrecision(base, cand, "opendecider-nano")
	if err != nil {
		t.Fatal(err)
	}
	if r.Schema != PrecisionSchema || r.Baseline.DType != "torch.float32" || r.Candidate.DType != "torch.bfloat16" || r.Baseline.Device != "cuda" {
		t.Fatalf("identity: %+v %+v", r.Baseline, r.Candidate)
	}
	// cases: short (a, b), mid (a), long (a, b) = 5 paired observations.
	if r.Paired != 5 || r.Unpaired != 0 || r.QuestionCount != 2 || r.Cases != 3 {
		t.Fatalf("pairing: %+v", r)
	}
	if r.ChoiceFlips != 1 || len(r.Flips) != 1 {
		t.Fatalf("flips = %d %+v", r.ChoiceFlips, r.Flips)
	}
	f := r.Flips[0]
	if f.CaseID != "short" || f.QuestionID != "a" || f.BaselineChoice != "yes" || f.CandidateChoice != "no" ||
		f.Expected != "yes" || f.Effect != FlipBroken || r.FlipsBroken != 1 || r.FlipsFixed != 0 || r.FlipsBothWrong != 0 {
		t.Fatalf("flip = %+v", f)
	}
	if r.FlipRate != 0.2 {
		t.Fatalf("flip rate %v", r.FlipRate)
	}
	// Probabilities are scripted: baseline yes 0.9, candidate yes 0.88; the flipped
	// observation moves from 0.9/0.1 to 0.15/0.85.
	if r.ProbabilityDiff.N != 5 || r.ProbabilityDiff.Max < 0.749 || r.ProbabilityDiff.Max > 0.751 || r.ProbabilityDiff.Mean <= 0 {
		t.Fatalf("probability delta: %+v", r.ProbabilityDiff)
	}
	if r.Confidence.Abs.N != 5 || r.Confidence.Abs.Max < 0.049 || r.Confidence.MeanSigned >= 0 {
		t.Fatalf("confidence delta: %+v", r.Confidence)
	}
	if r.QualityDelta.Accuracy == nil || *r.QualityDelta.Accuracy >= 0 || r.QualityDelta.Brier == nil || r.QualityDelta.ECE == nil || r.QualityDelta.NLL == nil {
		t.Fatalf("quality delta: %+v", r.QualityDelta)
	}
	if r.InferenceLatency.P50.Ratio == nil || *r.InferenceLatency.P50.Ratio != 0.5 {
		t.Fatalf("inference ratio: %+v", r.InferenceLatency)
	}
	if r.Memory.ResidentAllocated == nil || *r.Memory.ResidentAllocated.Ratio != 0.5 || r.Memory.PeakReserved == nil {
		t.Fatalf("memory: %+v", r.Memory)
	}
	if r.LoadMS == nil || r.WarmupMS == nil || len(r.LengthBuckets) != len(DefaultLengthEdges)+1 {
		t.Fatalf("startup/buckets: %+v %d", r.LoadMS, len(r.LengthBuckets))
	}
	if r.Baseline.Requests != 6 || r.Baseline.ErrorCount != 0 || r.Candidate.ErrorCount != 0 {
		t.Fatalf("request behavior: %+v", r.Baseline)
	}
	var sb strings.Builder
	PrecisionSummary(&sb, r)
	if !strings.Contains(sb.String(), "choice flips   1") || !strings.Contains(sb.String(), "flip short/a") {
		t.Fatal(sb.String())
	}
}

func TestComparePrecisionRefusesRunsThatAreNotDTypeOnly(t *testing.T) {
	base := precisionRun(t, "torch.float32", nil)
	cand := precisionRun(t, "torch.bfloat16", nil)
	for name, tc := range map[string]struct {
		cand ComparisonReport
		want string
	}{
		"same dtype": {precisionRun(t, "torch.float32", nil), "nothing to compare"},
		"no dtype":   {precisionRun(t, "", nil), "does not report the dtype"},
		"other provider": {precisionRun(t, "torch.bfloat16", func(f *fakeResidents) {
			f.find("opendecider-nano").provider = "other"
		}), "identities differ"},
	} {
		if _, err := ComparePrecision(base, tc.cand, "opendecider-nano"); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s: err = %v", name, err)
		}
	}
	// A different dataset is not the same inputs.
	other := cand
	other.Runs = append([]ModelRun{}, cand.Runs...)
	for i := range other.Runs {
		other.Runs[i].DatasetSHA256 = "different"
	}
	if _, err := ComparePrecision(base, other, "opendecider-nano"); err == nil || !strings.Contains(err.Error(), "dataset") {
		t.Fatalf("dataset: %v", err)
	}
	// Other declared controls make the buckets and thresholds incomparable.
	other = cand
	other.Declared.HighConfidence = 0.5
	if _, err := ComparePrecision(base, other, "opendecider-nano"); err == nil {
		t.Fatal("declared controls differ")
	}
	if _, err := ComparePrecision(base, cand, "nope"); err == nil {
		t.Fatal("model missing from the comparisons")
	}
}

func TestComparePrecisionRecordsUnpairedAndOptionMismatch(t *testing.T) {
	base, cand := precisionRun(t, "torch.float32", nil), precisionRun(t, "torch.bfloat16", nil)
	cand.Runs[1].Observations = cand.Runs[1].Observations[1:] // one request has no candidate answer
	r, err := ComparePrecision(base, cand, "opendecider-nano")
	if err != nil || r.Paired != 4 || r.Unpaired != 1 {
		t.Fatalf("unpaired: %+v %v", r, err)
	}
	base, cand = precisionRun(t, "torch.float32", nil), precisionRun(t, "torch.bfloat16", nil)
	delete(cand.Runs[1].Observations[0].Probabilities, "no")
	if _, err := ComparePrecision(base, cand, "opendecider-nano"); err == nil || !strings.Contains(err.Error(), "option sets") {
		t.Fatalf("option mismatch: %v", err)
	}
}
