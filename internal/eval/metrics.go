package eval

import (
	"fmt"
	"math"
	"sort"
)

// Declared analysis controls. They are recorded in every comparison report so
// a number is never separated from the threshold or bucket edges it used.
const (
	// DefaultHighConfidence is the declared high-confidence threshold: an
	// observation is a high-confidence error when it is wrong and its
	// confidence is at least this value (the same rule as the Errors
	// workspace).
	DefaultHighConfidence = 0.9
	// ECEBins is the number of equal-width bins of every reported ECE.
	ECEBins = 15
	// probFloor clips the probability of the expected label in NLL so a model
	// that gives it probability 0 yields a large finite value, not +Inf.
	probFloor = 1e-15
)

// DefaultThresholds are the confidence thresholds of the coverage table.
var DefaultThresholds = []float64{0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}

// DefaultLengthEdges are the exclusive upper edges, in characters of the
// request state, of the input-length buckets; the last bucket is open.
var DefaultLengthEdges = []int{256, 512, 1024, 2048, 4096, 8192, 16384}

// Calibration is probability-quality evidence. A metric that is not defined
// for the observations (none, or no usable probabilities) is null, never 0.
//
// ECE uses Confidence and correctness with ECEBins equal-width bins. Brier is
// the multi-class Brier score: the mean over observations of the sum over
// choices of (p - 1[choice is the expected label])^2. NLL is the mean of
// -ln p(expected label), with p clipped below at 1e-15. Brier and NLL cover
// only observations whose probabilities include the expected label;
// ProbabilityN counts them and Undefined counts the rest.
type Calibration struct {
	ECEBins      int      `json:"ece_bins"`
	ECE          *float64 `json:"ece"`
	Brier        *float64 `json:"brier"`
	NLL          *float64 `json:"nll"`
	ProbabilityN int      `json:"probability_n"`
	Undefined    int      `json:"probability_undefined"`
}

// HighConfidence is the high-confidence error evidence at one declared
// threshold. RateOfObservations is errors over all observations;
// RateOfHighConfidence is errors over the observations whose confidence
// reached the threshold (null when there are none).
type HighConfidence struct {
	Threshold            float64  `json:"threshold"`
	HighConfidenceN      int      `json:"high_confidence_n"`
	Errors               int      `json:"errors"`
	RateOfObservations   *float64 `json:"rate_of_observations"`
	RateOfHighConfidence *float64 `json:"rate_of_high_confidence"`
}

// ThresholdRow is one row of the confidence-threshold table: the
// observations answered with confidence >= Threshold, their share of all
// observations (Coverage) and their accuracy (ConditionalAccuracy, null when
// nothing is covered).
type ThresholdRow struct {
	Threshold           float64  `json:"threshold"`
	Covered             int      `json:"covered"`
	Coverage            *float64 `json:"coverage"`
	ConditionalAccuracy *float64 `json:"conditional_accuracy"`
}

// Quality is the label-based evidence of a set of observations.
//
// MacroF1 is the unweighted mean of per-class F1 over every (question id,
// label) class that occurs as an expected label or as a choice: a class's F1
// is 2tp/(2tp+fp+fn). Classes never expected and never chosen do not exist
// and are not counted.
type Quality struct {
	N              int            `json:"n"`
	Accuracy       *float64       `json:"accuracy"`
	MacroF1        *float64       `json:"macro_f1"`
	MeanConfidence *float64       `json:"mean_confidence"`
	Calibration    Calibration    `json:"calibration"`
	HighConfidence HighConfidence `json:"high_confidence"`
	Thresholds     []ThresholdRow `json:"thresholds"`
}

// Slice is the Quality of the observations of one question or family.
type Slice struct {
	Key string `json:"key"`
	Quality
}

func ptr(v float64) *float64 { return &v }

// QualityOf computes Quality for obs at the declared high-confidence
// threshold and coverage thresholds. It is a pure function of its arguments.
func QualityOf(obs []Observation, highConfidence float64, thresholds []float64) Quality {
	q := Quality{N: len(obs), Thresholds: make([]ThresholdRow, 0, len(thresholds)),
		Calibration:    Calibration{ECEBins: ECEBins},
		HighConfidence: HighConfidence{Threshold: highConfidence}}
	for _, t := range thresholds {
		q.Thresholds = append(q.Thresholds, ThresholdRow{Threshold: t})
	}
	if len(obs) == 0 {
		return q
	}
	n := float64(len(obs))
	conf := make([]float64, len(obs))
	ok := make([]bool, len(obs))
	var correct int
	var sumConf, brier, nll float64
	covered := make([]int, len(thresholds))
	coveredOK := make([]int, len(thresholds))
	for i, o := range obs {
		conf[i], ok[i] = o.Confidence, o.Correct
		sumConf += o.Confidence
		if o.Correct {
			correct++
		}
		for j, t := range thresholds {
			if o.Confidence >= t {
				covered[j]++
				if o.Correct {
					coveredOK[j]++
				}
			}
		}
		if o.Confidence >= highConfidence {
			q.HighConfidence.HighConfidenceN++
			if !o.Correct {
				q.HighConfidence.Errors++
			}
		}
		if b, l, defined := probabilityLoss(o); defined {
			brier += b
			nll += l
			q.Calibration.ProbabilityN++
		} else {
			q.Calibration.Undefined++
		}
	}
	q.Accuracy = ptr(float64(correct) / n)
	q.MeanConfidence = ptr(sumConf / n)
	q.MacroF1 = ptr(macroF1(obs))
	q.Calibration.ECE = ptr(ECE(conf, ok, ECEBins))
	if m := q.Calibration.ProbabilityN; m > 0 {
		q.Calibration.Brier, q.Calibration.NLL = ptr(brier/float64(m)), ptr(nll/float64(m))
	}
	q.HighConfidence.RateOfObservations = ptr(float64(q.HighConfidence.Errors) / n)
	if h := q.HighConfidence.HighConfidenceN; h > 0 {
		q.HighConfidence.RateOfHighConfidence = ptr(float64(q.HighConfidence.Errors) / float64(h))
	}
	for j := range q.Thresholds {
		q.Thresholds[j].Covered = covered[j]
		q.Thresholds[j].Coverage = ptr(float64(covered[j]) / n)
		if covered[j] > 0 {
			q.Thresholds[j].ConditionalAccuracy = ptr(float64(coveredOK[j]) / float64(covered[j]))
		}
	}
	return q
}

// probabilityLoss is the Brier and NLL contribution of one observation. It is
// undefined when the probabilities are missing, not finite, or do not include
// the expected label.
func probabilityLoss(o Observation) (brier, nll float64, defined bool) {
	pe, has := o.Probabilities[o.Expected]
	if len(o.Probabilities) == 0 || !has {
		return 0, 0, false
	}
	for k, p := range o.Probabilities {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			return 0, 0, false
		}
		y := 0.0
		if k == o.Expected {
			y = 1
		}
		brier += (p - y) * (p - y)
	}
	return brier, -math.Log(math.Min(1, math.Max(pe, probFloor))), true
}

func macroF1(obs []Observation) float64 {
	type class struct{ q, label string }
	tp, fp, fn := map[class]int{}, map[class]int{}, map[class]int{}
	seen := map[class]bool{}
	for _, o := range obs {
		e, c := class{o.QuestionID, o.Expected}, class{o.QuestionID, o.Choice}
		seen[e], seen[c] = true, true
		if o.Expected == o.Choice {
			tp[e]++
		} else {
			fn[e]++
			fp[c]++
		}
	}
	if len(seen) == 0 {
		return 0
	}
	sum := 0.0
	for c := range seen {
		if d := 2*tp[c] + fp[c] + fn[c]; d > 0 {
			sum += float64(2*tp[c]) / float64(d)
		}
	}
	return sum / float64(len(seen))
}

// LengthBucket is the behaviour of one input-length range of the dataset.
// Input length is the number of characters (runes) of the case state; the
// range is [MinChars, MaxChars) and MaxChars is null for the open last
// bucket. Latency is over warm requests of every pass (warmup excluded),
// request being the client round trip and inference the server-reported time.
// Every bucket is always present, empty ones with zero counts, so reports of
// different models have the same shape.
type LengthBucket struct {
	Label            string   `json:"label"`
	MinChars         int      `json:"min_chars"`
	MaxChars         *int     `json:"max_chars"`
	Cases            int      `json:"cases"`
	Requests         int      `json:"requests"`
	Errors           int      `json:"errors"`
	Observations     int      `json:"observations"`
	Accuracy         *float64 `json:"accuracy"`
	RequestLatency   Latency  `json:"request_latency"`
	InferenceLatency Latency  `json:"inference_latency"`
}

// ValidateEdges checks input-length bucket edges: positive and strictly
// ascending.
func ValidateEdges(edges []int) error {
	prev := 0
	for _, e := range edges {
		if e <= prev {
			return fmt.Errorf("length bucket edges must be positive and strictly ascending, got %v", edges)
		}
		prev = e
	}
	return nil
}

// ValidateThresholds checks coverage thresholds: within [0, 1], strictly
// ascending.
func ValidateThresholds(ts []float64) error {
	prev := -1.0
	for _, t := range ts {
		if math.IsNaN(t) || t < 0 || t > 1 || t <= prev {
			return fmt.Errorf("thresholds must be within [0, 1] and strictly ascending, got %v", ts)
		}
		prev = t
	}
	return nil
}

func bucketIndex(edges []int, chars int) int {
	return sort.Search(len(edges), func(i int) bool { return chars < edges[i] })
}

func bucketLabel(edges []int, i int) (string, int, *int) {
	lo := 0
	if i > 0 {
		lo = edges[i-1]
	}
	if i == len(edges) {
		return fmt.Sprintf("%d+ chars", lo), lo, nil
	}
	hi := edges[i]
	return fmt.Sprintf("%d-%d chars", lo, hi-1), lo, &hi
}

// lengthBuckets slices a run by the input length of its cases.
func lengthBuckets(edges []int, inputs []inputInfo, r Report, det runDetail) []LengthBucket {
	n := len(edges) + 1
	bs := make([]LengthBucket, n)
	req, inf := make([][]float64, n), make([][]float64, n)
	obs := make([][]Observation, n)
	caseBucket := map[string]int{}
	for i := range bs {
		bs[i].Label, bs[i].MinChars, bs[i].MaxChars = bucketLabel(edges, i)
	}
	for _, in := range inputs {
		b := bucketIndex(edges, in.Chars)
		caseBucket[in.CaseID] = b
		bs[b].Cases++
	}
	for _, s := range det.samples {
		b := bucketIndex(edges, inputs[s.caseIdx].Chars)
		bs[b].Requests++
		req[b] = append(req[b], s.requestMS)
		if s.inferenceMS != nil {
			inf[b] = append(inf[b], *s.inferenceMS)
		}
	}
	for _, e := range r.Errors {
		if e.Phase != "pass" {
			continue
		}
		if b, ok := caseBucket[e.CaseID]; ok {
			bs[b].Errors++
		}
	}
	for i, o := range r.Results {
		b := bucketIndex(edges, inputs[det.caseIdx[i]].Chars)
		obs[b] = append(obs[b], o)
	}
	for i := range bs {
		bs[i].Observations = len(obs[i])
		if len(obs[i]) > 0 {
			acc, _, _ := score(obs[i])
			bs[i].Accuracy = ptr(acc)
		}
		bs[i].RequestLatency, bs[i].InferenceLatency = summarize(req[i]), summarize(inf[i])
	}
	return bs
}

// slices groups obs by key (question id, or family) and computes the Quality
// of each group, ordered by key.
func slices(obs []Observation, key func(Observation) string, hc float64, ts []float64) []Slice {
	groups := map[string][]Observation{}
	for _, o := range obs {
		k := key(o)
		groups[k] = append(groups[k], o)
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]Slice, 0, len(keys))
	for _, k := range keys {
		out = append(out, Slice{Key: k, Quality: QualityOf(groups[k], hc, ts)})
	}
	return out
}
