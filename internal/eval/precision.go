package eval

import (
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/worker"
)

// PrecisionSchema identifies the paired dtype comparison of one resident model.
// It is a separate, additive document derived from two resident comparisons
// (hachidori.resident-comparison.v1); no existing schema changes.
const PrecisionSchema = "hachidori.precision-comparison.v1"

// Effects of a choice flip on correctness, from the baseline to the candidate.
const (
	FlipFixed     = "fixed"      // baseline wrong, candidate right
	FlipBroken    = "broken"     // baseline right, candidate wrong
	FlipBothWrong = "both_wrong" // both wrong, different choices
)

// PrecisionSide is one precision of the compared model: its identity as the
// resident itself reported it (including the dtype it is actually on) and
// everything the run recorded about it.
type PrecisionSide struct {
	Model          string `json:"model"`
	DType          string `json:"dtype"`
	Device         string `json:"device"`
	IdentitySHA256 string `json:"identity_sha256"`
	ResidentStable bool   `json:"resident_stable"`

	Startup       Startup        `json:"startup"`
	Memory        Memory         `json:"memory"`
	Requests      int            `json:"requests"`
	Succeeded     int            `json:"succeeded"`
	ErrorCount    int            `json:"error_count"`
	ErrorsByClass map[string]int `json:"errors_by_class"`

	Quality          Quality        `json:"quality"`
	RequestLatency   Latency        `json:"request_latency"`
	InferenceLatency Latency        `json:"inference_latency"`
	LengthBuckets    []LengthBucket `json:"length_buckets"`
}

// Flip is one observation whose chosen option differs between the precisions.
type Flip struct {
	CaseID              string  `json:"case_id"`
	QuestionID          string  `json:"question_id"`
	Expected            string  `json:"expected"`
	BaselineChoice      string  `json:"baseline_choice"`
	CandidateChoice     string  `json:"candidate_choice"`
	BaselineConfidence  float64 `json:"baseline_confidence"`
	CandidateConfidence float64 `json:"candidate_confidence"`
	Effect              string  `json:"effect"`
}

// Spread summarizes a non-negative per-observation difference.
type Spread struct {
	N    int     `json:"n"`
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
	Max  float64 `json:"max"`
}

// ConfidenceDelta is candidate minus baseline answer confidence over the
// paired observations: the signed mean (a bias) and the spread of the
// absolute difference.
type ConfidenceDelta struct {
	MeanSigned float64 `json:"mean_signed"`
	Abs        Spread  `json:"abs"`
}

// QualityDelta is candidate minus baseline of the quality measures. A value is
// null when either side has none.
type QualityDelta struct {
	Accuracy             *float64 `json:"accuracy"`
	MacroF1              *float64 `json:"macro_f1"`
	ECE                  *float64 `json:"ece"`
	Brier                *float64 `json:"brier"`
	NLL                  *float64 `json:"nll"`
	HighConfidenceErrors int      `json:"high_confidence_errors"`
}

// Ratio is candidate divided by baseline; null when the baseline is zero. A
// ratio below 1 means the candidate is smaller (lower latency, less memory).
type Ratio struct {
	Baseline  float64  `json:"baseline"`
	Candidate float64  `json:"candidate"`
	Ratio     *float64 `json:"ratio"`
}

// LatencyRatios compares the percentiles of warm requests.
type LatencyRatios struct {
	P50 Ratio `json:"p50"`
	P95 Ratio `json:"p95"`
}

// BucketRatios is the latency comparison inside one input-length bucket.
type BucketRatios struct {
	Label     string        `json:"label"`
	Requests  CountDelta    `json:"requests"`
	Accuracy  *float64      `json:"accuracy_delta"`
	Request   LatencyRatios `json:"request_latency"`
	Inference LatencyRatios `json:"inference_latency"`
}

// MemoryDeltas compares resident and peak accelerator memory in bytes. Null
// when either side reports no accelerator statistics.
type MemoryDeltas struct {
	ResidentAllocated *Ratio `json:"resident_allocated"`
	ResidentReserved  *Ratio `json:"resident_reserved"`
	PeakAllocated     *Ratio `json:"peak_allocated"`
	PeakReserved      *Ratio `json:"peak_reserved"`
}

// PrecisionReport is the descriptive, paired evidence of one catalog model
// run at two precisions on identical inputs. It states no verdict: whether a
// default dtype may change is decided against docs/certification.md.
type PrecisionReport struct {
	Schema        string `json:"schema"`
	Model         string `json:"model"`
	DatasetSHA256 string `json:"dataset_sha256"`
	InputSHA256   string `json:"input_sha256"`
	SentSHA256    string `json:"sent_sha256"`
	Cases         int    `json:"cases"`
	// QuestionCount is the number of distinct questions of the dataset: the
	// question-count scaling is the series of reports over datasets that differ
	// in it.
	QuestionCount int `json:"question_count"`

	Baseline  PrecisionSide `json:"baseline"`
	Candidate PrecisionSide `json:"candidate"`

	Paired   int `json:"paired_observations"`
	Unpaired int `json:"unpaired_observations"`

	ChoiceFlips     int             `json:"choice_flips"`
	FlipRate        float64         `json:"flip_rate"`
	FlipsFixed      int             `json:"flips_fixed"`
	FlipsBroken     int             `json:"flips_broken"`
	FlipsBothWrong  int             `json:"flips_both_wrong"`
	Flips           []Flip          `json:"flips"`
	ProbabilityDiff Spread          `json:"probability_abs_delta"` // per observation, the largest |p_candidate - p_baseline| over the options
	Confidence      ConfidenceDelta `json:"confidence_delta"`

	QualityDelta     QualityDelta   `json:"quality_delta"`
	RequestLatency   LatencyRatios  `json:"request_latency"`
	InferenceLatency LatencyRatios  `json:"inference_latency"`
	LengthBuckets    []BucketRatios `json:"length_buckets"`
	Memory           MemoryDeltas   `json:"memory"`
	LoadMS           *Ratio         `json:"load_ms"`
	WarmupMS         *Ratio         `json:"warmup_ms"`
}

type pairKey struct{ caseID, questionID string }

// ComparePrecision pairs the run of model in baseline with the run of the same
// model in candidate. It refuses, with every reason listed, unless the two
// runs are the same pinned model on the same device and runtime that differ in
// nothing but dtype, and received identical datasets, protocol, question
// identities and normalized inputs. It never computes deltas across runs that
// are not like for like.
func ComparePrecision(baseline, candidate ComparisonReport, model string) (PrecisionReport, error) {
	find := func(c ComparisonReport) *ModelRun {
		for i := range c.Runs {
			if c.Runs[i].Model == model {
				return &c.Runs[i]
			}
		}
		return nil
	}
	a, b := find(baseline), find(candidate)
	if a == nil || b == nil {
		return PrecisionReport{}, fmt.Errorf("model %q is not in both comparisons", model)
	}
	var why []string
	if al := Align([]ModelRun{*a, *b}); al.Status != AlignAligned {
		for _, i := range al.Incompatibility {
			why = append(why, fmt.Sprintf("%s %s: %s", i.Code, i.Question, i.Detail))
		}
	}
	if !reflect.DeepEqual(baseline.Declared.Thresholds, candidate.Declared.Thresholds) ||
		!reflect.DeepEqual(baseline.Declared.LengthEdges, candidate.Declared.LengthEdges) ||
		baseline.Declared.HighConfidence != candidate.Declared.HighConfidence {
		why = append(why, "the declared high-confidence threshold, coverage thresholds or length-bucket edges differ")
	}
	da, db := dtypeOf(a.Identity), dtypeOf(b.Identity)
	switch {
	case da == "" || db == "":
		why = append(why, "a run does not report the dtype its resident is on")
	case da == db:
		why = append(why, fmt.Sprintf("both runs are %s: nothing to compare", da))
	}
	if diff := identityDiff(a.Identity, b.Identity); len(diff) > 0 {
		why = append(why, "the resident identities differ in more than the dtype: "+strings.Join(diff, ", "))
	}
	if len(why) > 0 {
		return PrecisionReport{}, errors.New("runs are not comparable: " + strings.Join(why, "; "))
	}

	r := PrecisionReport{Schema: PrecisionSchema, Model: model, DatasetSHA256: a.DatasetSHA256, InputSHA256: a.InputSHA256,
		SentSHA256: a.SentSHA256, Cases: a.Cases, QuestionCount: len(questionIndex(a.Questions)),
		Baseline: side(*a), Candidate: side(*b), Flips: []Flip{}}

	index := map[pairKey]ResidentObservation{}
	for _, o := range a.Observations {
		k := pairKey{o.CaseID, o.QuestionID}
		if _, dup := index[k]; dup {
			return PrecisionReport{}, fmt.Errorf("baseline has two observations of case %q question %q", k.caseID, k.questionID)
		}
		index[k] = o
	}
	seen := map[pairKey]bool{}
	var probMax, confAbs []float64
	var confSigned float64
	for _, o := range b.Observations {
		k := pairKey{o.CaseID, o.QuestionID}
		if seen[k] {
			return PrecisionReport{}, fmt.Errorf("candidate has two observations of case %q question %q", k.caseID, k.questionID)
		}
		seen[k] = true
		p, ok := index[k]
		if !ok {
			r.Unpaired++
			continue
		}
		delete(index, k)
		if !sameKeys(p.Probabilities, o.Probabilities) {
			return PrecisionReport{}, fmt.Errorf("case %q question %q: the runs scored different option sets", k.caseID, k.questionID)
		}
		r.Paired++
		if p.Choice != o.Choice {
			f := Flip{CaseID: k.caseID, QuestionID: k.questionID, Expected: o.Expected, BaselineChoice: p.Choice,
				CandidateChoice: o.Choice, BaselineConfidence: p.Confidence, CandidateConfidence: o.Confidence}
			switch {
			case !p.Correct && o.Correct:
				f.Effect = FlipFixed
				r.FlipsFixed++
			case p.Correct && !o.Correct:
				f.Effect = FlipBroken
				r.FlipsBroken++
			default:
				f.Effect = FlipBothWrong
				r.FlipsBothWrong++
			}
			r.Flips = append(r.Flips, f)
		}
		worst := 0.0
		for opt, pb := range p.Probabilities {
			worst = math.Max(worst, math.Abs(o.Probabilities[opt]-pb))
		}
		probMax = append(probMax, worst)
		confSigned += o.Confidence - p.Confidence
		confAbs = append(confAbs, math.Abs(o.Confidence-p.Confidence))
	}
	r.Unpaired += len(index)
	r.ChoiceFlips = len(r.Flips)
	sort.Slice(r.Flips, func(i, j int) bool {
		x, y := r.Flips[i], r.Flips[j]
		if x.CaseID != y.CaseID {
			return x.CaseID < y.CaseID
		}
		return x.QuestionID < y.QuestionID
	})
	if r.Paired > 0 {
		r.FlipRate = float64(r.ChoiceFlips) / float64(r.Paired)
		r.Confidence = ConfidenceDelta{MeanSigned: confSigned / float64(r.Paired), Abs: spread(confAbs)}
	}
	r.ProbabilityDiff = spread(probMax)

	qa, qb := a.Quality, b.Quality
	r.QualityDelta = QualityDelta{Accuracy: sub(qb.Accuracy, qa.Accuracy), MacroF1: sub(qb.MacroF1, qa.MacroF1),
		ECE: sub(qb.Calibration.ECE, qa.Calibration.ECE), Brier: sub(qb.Calibration.Brier, qa.Calibration.Brier),
		NLL: sub(qb.Calibration.NLL, qa.Calibration.NLL), HighConfidenceErrors: qb.HighConfidence.Errors - qa.HighConfidence.Errors}
	r.RequestLatency, r.InferenceLatency = latencyRatios(a.RequestLatency, b.RequestLatency), latencyRatios(a.InferenceLatency, b.InferenceLatency)
	for i, x := range a.LengthBuckets {
		if i >= len(b.LengthBuckets) || b.LengthBuckets[i].Label != x.Label {
			return PrecisionReport{}, errors.New("length buckets do not line up")
		}
		y := b.LengthBuckets[i]
		r.LengthBuckets = append(r.LengthBuckets, BucketRatios{Label: x.Label, Requests: countDelta(x.Requests, y.Requests),
			Accuracy: sub(y.Accuracy, x.Accuracy), Request: latencyRatios(x.RequestLatency, y.RequestLatency),
			Inference: latencyRatios(x.InferenceLatency, y.InferenceLatency)})
	}
	if a.Memory.Available && b.Memory.Available {
		r.Memory = MemoryDeltas{ResidentAllocated: byteRatio(a.Memory.ResidentAlloc, b.Memory.ResidentAlloc),
			ResidentReserved: byteRatio(a.Memory.ResidentRsrv, b.Memory.ResidentRsrv),
			PeakAllocated:    byteRatio(a.Memory.PeakAlloc, b.Memory.PeakAlloc), PeakReserved: byteRatio(a.Memory.PeakRsrv, b.Memory.PeakRsrv)}
	}
	r.LoadMS, r.WarmupMS = msRatio(a.Startup.LoadMS, b.Startup.LoadMS), msRatio(a.Startup.WarmupMS, b.Startup.WarmupMS)
	return r, nil
}

func side(m ModelRun) PrecisionSide {
	dev, _ := m.Identity.Provider["device"].(string)
	return PrecisionSide{Model: m.Model, DType: dtypeOf(m.Identity), Device: dev, IdentitySHA256: m.Identity.Digest,
		ResidentStable: m.ResidentStable, Startup: m.Startup, Memory: m.Memory, Requests: m.Requests, Succeeded: m.Succeeded,
		ErrorCount: m.ErrorCount, ErrorsByClass: m.ErrorsByClass, Quality: m.Quality, RequestLatency: m.RequestLatency,
		InferenceLatency: m.InferenceLatency, LengthBuckets: m.LengthBuckets}
}

// dtypeOf is the dtype the resident reported being on, "" when it reports none.
func dtypeOf(s Served) string {
	d, _ := s.Provider["dtype"].(string)
	return d
}

// identityDiff lists the identity fields, outside the dtype, in which two
// resident identities differ: same pinned model, revision, provider, runtime,
// device and torch build are required for a dtype-only comparison.
func identityDiff(a, b Served) []string {
	var out []string
	for _, p := range [][2]map[string]any{{a.Runtime, b.Runtime}, {a.Provider, b.Provider}} {
		keys := map[string]bool{}
		for k := range p[0] {
			keys[k] = true
		}
		for k := range p[1] {
			keys[k] = true
		}
		for k := range keys {
			if k != "dtype" && !reflect.DeepEqual(p[0][k], p[1][k]) {
				out = append(out, k)
			}
		}
	}
	sort.Strings(out)
	return out
}

func sameKeys(a, b map[string]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if _, ok := b[k]; !ok {
			return false
		}
	}
	return true
}

func spread(xs []float64) Spread {
	s := Spread{N: len(xs)}
	if len(xs) == 0 {
		return s
	}
	sum := 0.0
	for _, x := range xs {
		sum += x
		s.Max = math.Max(s.Max, x)
	}
	s.Mean, s.P50, s.P95 = sum/float64(len(xs)), worker.Percentile(xs, 50), worker.Percentile(xs, 95)
	return s
}

func sub(b, a *float64) *float64 {
	if a == nil || b == nil {
		return nil
	}
	return ptr(*b - *a)
}

func ratio(a, b float64) Ratio {
	r := Ratio{Baseline: a, Candidate: b}
	if a != 0 {
		r.Ratio = ptr(b / a)
	}
	return r
}

func latencyRatios(a, b Latency) LatencyRatios {
	if a.N == 0 || b.N == 0 {
		return LatencyRatios{}
	}
	return LatencyRatios{P50: ratio(a.P50, b.P50), P95: ratio(a.P95, b.P95)}
}

func byteRatio(a, b *int64) *Ratio {
	if a == nil || b == nil {
		return nil
	}
	r := ratio(float64(*a), float64(*b))
	return &r
}

func msRatio(a, b *float64) *Ratio {
	if a == nil || b == nil {
		return nil
	}
	r := ratio(*a, *b)
	return &r
}

// PrecisionSummary renders the headline numbers. It states measurements only.
func PrecisionSummary(w io.Writer, r PrecisionReport) {
	f := func(p *float64) string {
		if p == nil {
			return "n/a"
		}
		return fmt.Sprintf("%+.4f", *p)
	}
	x := func(p Ratio) string {
		if p.Ratio == nil {
			return "n/a"
		}
		return fmt.Sprintf("%.3fx (%.1f -> %.1f)", *p.Ratio, p.Baseline, p.Candidate)
	}
	fmt.Fprintf(w, "model          %s  %s (baseline) vs %s (candidate) on %s\n", r.Model, r.Baseline.DType, r.Candidate.DType, r.Baseline.Device)
	fmt.Fprintf(w, "inputs         dataset sha256 %.12s, %d cases, %d questions, inputs sha256 %.12s\n", r.DatasetSHA256, r.Cases, r.QuestionCount, r.InputSHA256)
	fmt.Fprintf(w, "paired         %d observations (%d unpaired)\n", r.Paired, r.Unpaired)
	fmt.Fprintf(w, "choice flips   %d (%.2f%%): %d fixed, %d broken, %d both wrong\n", r.ChoiceFlips, 100*r.FlipRate, r.FlipsFixed, r.FlipsBroken, r.FlipsBothWrong)
	fmt.Fprintf(w, "probability    max |dp| per observation: mean %.5f p95 %.5f max %.5f\n", r.ProbabilityDiff.Mean, r.ProbabilityDiff.P95, r.ProbabilityDiff.Max)
	fmt.Fprintf(w, "confidence     signed mean %+.5f, |d| p95 %.5f max %.5f\n", r.Confidence.MeanSigned, r.Confidence.Abs.P95, r.Confidence.Abs.Max)
	q := r.QualityDelta
	fmt.Fprintf(w, "quality delta  accuracy %s  ECE %s  Brier %s  NLL %s  high-confidence errors %+d\n", f(q.Accuracy), f(q.ECE), f(q.Brier), f(q.NLL), q.HighConfidenceErrors)
	fmt.Fprintf(w, "request p50    %s\n", x(r.RequestLatency.P50))
	fmt.Fprintf(w, "request p95    %s\n", x(r.RequestLatency.P95))
	fmt.Fprintf(w, "inference p50  %s\n", x(r.InferenceLatency.P50))
	fmt.Fprintf(w, "inference p95  %s\n", x(r.InferenceLatency.P95))
	if r.Memory.ResidentAllocated != nil {
		fmt.Fprintf(w, "resident alloc %s\n", x(*r.Memory.ResidentAllocated))
	}
	if r.Memory.PeakReserved != nil {
		fmt.Fprintf(w, "peak reserved  %s\n", x(*r.Memory.PeakReserved))
	}
	fmt.Fprintf(w, "errors         baseline %d, candidate %d of %d requests\n", r.Baseline.ErrorCount, r.Candidate.ErrorCount, r.Baseline.Requests)
	for _, f := range r.Flips {
		fmt.Fprintf(w, "  flip %s/%s expected %s: %s (%.3f) -> %s (%.3f) %s\n", f.CaseID, f.QuestionID, f.Expected,
			f.BaselineChoice, f.BaselineConfidence, f.CandidateChoice, f.CandidateConfidence, f.Effect)
	}
}
