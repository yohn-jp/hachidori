package eval

import (
	"math"
	"testing"
)

func near(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil || math.Abs(*got-want) > 1e-9 {
		t.Fatalf("%s = %v, want %v", name, deref(got), want)
	}
}

func deref(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

func ob(q, exp, choice string, conf float64, probs map[string]float64) Observation {
	return Observation{CaseID: "c", QuestionID: q, Expected: exp, Choice: choice, Confidence: conf, Probabilities: probs,
		Correct: exp == choice}
}

// Hand-computed fixture over two binary questions.
//
//	q1 yes/no:  (yes, yes 0.9)  (yes, no 0.8)  (no, no 0.6)
//	q2 a/b:     (a, b 0.95)
func calibrationFixture() []Observation {
	return []Observation{
		ob("q1", "yes", "yes", 0.9, map[string]float64{"yes": 0.9, "no": 0.1}),
		ob("q1", "yes", "no", 0.8, map[string]float64{"yes": 0.2, "no": 0.8}),
		ob("q1", "no", "no", 0.6, map[string]float64{"yes": 0.4, "no": 0.6}),
		ob("q2", "a", "b", 0.95, map[string]float64{"a": 0.05, "b": 0.95}),
	}
}

func TestQualityCalibrationFixture(t *testing.T) {
	q := QualityOf(calibrationFixture(), 0.9, []float64{0.5, 0.9, 0.99})
	near(t, "accuracy", q.Accuracy, 0.5)
	near(t, "mean confidence", q.MeanConfidence, (0.9+0.8+0.6+0.95)/4)
	// Brier: per observation sum of squared errors over both choices.
	brier := (2*0.1*0.1 + 2*0.8*0.8 + 2*0.4*0.4 + 2*0.95*0.95) / 4
	near(t, "brier", q.Calibration.Brier, brier)
	nll := (-math.Log(0.9) - math.Log(0.2) - math.Log(0.6) - math.Log(0.05)) / 4
	near(t, "nll", q.Calibration.NLL, nll)
	// ECE, 15 bins: 0.9, 0.8, 0.6 and 0.95 fall in different bins.
	ece := (math.Abs(0.9-1) + math.Abs(0.8-0) + math.Abs(0.6-1) + math.Abs(0.95-0)) / 4
	near(t, "ece", q.Calibration.ECE, ece)
	if q.Calibration.ProbabilityN != 4 || q.Calibration.Undefined != 0 || q.Calibration.ECEBins != 15 {
		t.Fatalf("%+v", q.Calibration)
	}
	// Macro F1 over classes (q1,yes) (q1,no) (q2,a) (q2,b):
	// q1/yes tp1 fn1 fp0 -> 2/3; q1/no tp1 fp1 fn0 -> 2/3; q2/a fn1 -> 0; q2/b fp1 -> 0.
	near(t, "macro f1", q.MacroF1, (2.0/3+2.0/3)/4)
	// High-confidence errors at 0.9: wrong and confidence >= 0.9 is only q2.
	hc := q.HighConfidence
	if hc.Threshold != 0.9 || hc.HighConfidenceN != 2 || hc.Errors != 1 {
		t.Fatalf("%+v", hc)
	}
	near(t, "hc rate of all", hc.RateOfObservations, 0.25)
	near(t, "hc rate of hc", hc.RateOfHighConfidence, 0.5)
}

func TestThresholdCoverageTable(t *testing.T) {
	q := QualityOf(calibrationFixture(), 0.9, []float64{0, 0.5, 0.8, 0.9, 0.99})
	type row struct {
		covered int
		cov     float64
		acc     *float64
	}
	half := 0.5
	want := []row{{4, 1, &half}, {4, 1, &half}, {3, 0.75, ptr(1.0 / 3)}, {2, 0.5, &half}, {0, 0, nil}}
	for i, w := range want {
		g := q.Thresholds[i]
		if g.Covered != w.covered {
			t.Fatalf("row %d: covered %d want %d", i, g.Covered, w.covered)
		}
		near(t, "coverage", g.Coverage, w.cov)
		if w.acc == nil {
			if g.ConditionalAccuracy != nil {
				t.Fatalf("row %d: conditional accuracy defined with nothing covered", i)
			}
			continue
		}
		near(t, "conditional accuracy", g.ConditionalAccuracy, *w.acc)
	}
	// Threshold is inclusive: confidence exactly at the threshold is covered.
	if QualityOf(calibrationFixture(), 0.9, []float64{0.95}).Thresholds[0].Covered != 1 {
		t.Fatal("threshold must be >=")
	}
}

func TestUndefinedMetricsAreNullNotZero(t *testing.T) {
	q := QualityOf(nil, 0.9, []float64{0.5})
	if q.N != 0 || q.Accuracy != nil || q.MacroF1 != nil || q.MeanConfidence != nil || q.Calibration.ECE != nil ||
		q.Calibration.Brier != nil || q.Calibration.NLL != nil || q.HighConfidence.RateOfObservations != nil ||
		q.Thresholds[0].Coverage != nil || q.Thresholds[0].ConditionalAccuracy != nil {
		t.Fatalf("%+v", q)
	}
	// Probabilities that do not cover the expected label make Brier and NLL
	// undefined for that observation, while accuracy and ECE stay defined.
	obs := []Observation{
		ob("q", "yes", "yes", 0.9, nil),
		ob("q", "yes", "no", 0.7, map[string]float64{"no": 1}),
		ob("q", "yes", "yes", 0.8, map[string]float64{"yes": 0.8, "no": 0.2}),
		ob("q", "yes", "yes", 0.8, map[string]float64{"yes": math.NaN()}),
	}
	q = QualityOf(obs, 0.9, nil)
	if q.Calibration.ProbabilityN != 1 || q.Calibration.Undefined != 3 {
		t.Fatalf("%+v", q.Calibration)
	}
	near(t, "brier", q.Calibration.Brier, 0.2*0.2+0.2*0.2)
	near(t, "accuracy", q.Accuracy, 0.75)
	if q.Calibration.ECE == nil {
		t.Fatal("ece must stay defined")
	}
	// No observation has usable probabilities.
	q = QualityOf(obs[:2], 0.9, nil)
	if q.Calibration.Brier != nil || q.Calibration.NLL != nil || q.Calibration.ProbabilityN != 0 {
		t.Fatalf("%+v", q.Calibration)
	}
}

func TestNLLClipsZeroProbability(t *testing.T) {
	q := QualityOf([]Observation{ob("q", "yes", "no", 1, map[string]float64{"yes": 0, "no": 1})}, 0.9, nil)
	if q.Calibration.NLL == nil || math.IsInf(*q.Calibration.NLL, 0) || math.Abs(*q.Calibration.NLL-(-math.Log(1e-15))) > 1e-9 {
		t.Fatalf("%v", deref(q.Calibration.NLL))
	}
}

func TestMacroF1PerfectAndDegenerate(t *testing.T) {
	perfect := []Observation{ob("q", "yes", "yes", 1, nil), ob("q", "no", "no", 1, nil)}
	near(t, "perfect", QualityOf(perfect, 0.9, nil).MacroF1, 1)
	wrong := []Observation{ob("q", "yes", "no", 1, nil), ob("q", "no", "yes", 1, nil)}
	near(t, "all wrong", QualityOf(wrong, 0.9, nil).MacroF1, 0)
	// A class that is only ever predicted counts with F1 0.
	one := []Observation{ob("q", "yes", "yes", 1, nil), ob("q", "yes", "no", 1, nil)}
	near(t, "predicted only", QualityOf(one, 0.9, nil).MacroF1, 1.0/3)
}

func TestSlicesAreIndependent(t *testing.T) {
	obs := calibrationFixture()
	s := slices(obs, func(o Observation) string { return o.QuestionID }, 0.9, []float64{0.5})
	if len(s) != 2 || s[0].Key != "q1" || s[1].Key != "q2" || s[0].N != 3 || s[1].N != 1 {
		t.Fatalf("%+v", s)
	}
	near(t, "q1 accuracy", s[0].Accuracy, 2.0/3)
	near(t, "q2 accuracy", s[1].Accuracy, 0)
	if s[1].HighConfidence.Errors != 1 || s[0].HighConfidence.Errors != 0 {
		t.Fatalf("high confidence by slice: %+v %+v", s[0].HighConfidence, s[1].HighConfidence)
	}
}

func TestValidateDeclaredControls(t *testing.T) {
	for _, ok := range [][]int{nil, {1}, {256, 512}} {
		if ValidateEdges(ok) != nil {
			t.Fatalf("%v", ok)
		}
	}
	for _, bad := range [][]int{{0}, {-1}, {5, 5}, {9, 3}} {
		if ValidateEdges(bad) == nil {
			t.Fatalf("%v accepted", bad)
		}
	}
	for _, bad := range [][]float64{{1.1}, {-0.1}, {0.5, 0.5}, {0.9, 0.5}, {math.NaN()}} {
		if ValidateThresholds(bad) == nil {
			t.Fatalf("%v accepted", bad)
		}
	}
	if ValidateThresholds(DefaultThresholds) != nil || ValidateEdges(DefaultLengthEdges) != nil {
		t.Fatal("defaults must be valid")
	}
	if bucketIndex([]int{256, 512}, 255) != 0 || bucketIndex([]int{256, 512}, 256) != 1 || bucketIndex([]int{256, 512}, 9999) != 2 {
		t.Fatal("bucket edges are exclusive upper bounds")
	}
}
