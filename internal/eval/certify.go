package eval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
)

// CertificationSchema identifies the System One variant certification
// artifact: the audited comparison of one high-precision reference run and one
// derived variant run over identical typed-decision inputs.
//
// It states evidence first (what changed) and a verdict second (whether that
// change is acceptable under an explicit, versioned, digest-recorded policy).
// The verdict is never a ranking: it says "accepted" or "rejected" against
// thresholds and nothing about which model is better. The reference is the
// authority for fidelity only; it is never treated as ground truth for
// correctness, which only expected labels can measure.
const CertificationSchema = "hachidori.system-one-certification/1"

// Verdict statuses.
const (
	VerdictAccepted = "accepted"
	VerdictRejected = "rejected"
)

// Resource facts a certification reports as MEASURED or NOT_CHECKED.
const (
	FactMeasured   = "MEASURED"
	FactNotChecked = "NOT_CHECKED"
)

// Execution is the identity of what executed one side of the comparison,
// taken from the resident's own status: the semantic model and revision, the
// provider, the variant (absent for the reference), and the device, dtype and
// quantized execution the resident actually reported.
type Execution struct {
	Role           string `json:"role"` // reference | candidate
	ModelID        string `json:"model_id"`
	Provider       string `json:"provider"`
	Revision       string `json:"revision"`
	VariantID      string `json:"variant_id,omitempty"`
	Device         string `json:"device"`
	DType          string `json:"dtype"`
	Quantization   string `json:"quantized_execution,omitempty"`
	IdentitySHA256 string `json:"identity_sha256"`
	ResidentStable bool   `json:"resident_stable"`
}

func executionOf(role string, run ModelRun) Execution {
	str := func(m map[string]any, k string) string { v, _ := m[k].(string); return v }
	return Execution{Role: role, ModelID: str(run.Identity.Runtime, "model_id"), Provider: str(run.Identity.Provider, "provider"),
		Revision: str(run.Identity.Provider, "model_revision"), VariantID: str(run.Identity.Provider, "variant_id"),
		Device: str(run.Identity.Provider, "device"), DType: str(run.Identity.Provider, "dtype"),
		Quantization: str(run.Identity.Provider, "quantized_execution"), IdentitySHA256: run.Identity.Digest,
		ResidentStable: run.ResidentStable}
}

// CertifiedVariant is the variant the candidate must be: its identity as the
// manifest states it.
type CertifiedVariant struct {
	ID             string `json:"id"`
	ManifestSHA256 string `json:"manifest_sha256"`
	BuildID        string `json:"build_id"`
	Recipe         string `json:"recipe"`
	RecipeSHA256   string `json:"recipe_sha256"`
	Scheme         string `json:"scheme"`
	Engine         string `json:"engine"`
	EngineVersion  string `json:"engine_version"`
}

// CertifiedDataset is the shared input of both runs.
type CertifiedDataset struct {
	SHA256       string              `json:"sha256"`
	Cases        int                 `json:"cases"`
	Observations int                 `json:"observations"`
	Labelled     bool                `json:"labelled"`
	Definitions  []question.Identity `json:"question_definitions,omitempty"`
	Questions    []QuestionRecord    `json:"questions"`
	// QuestionsSHA256 digests the exact question identities as sent.
	QuestionsSHA256 string `json:"questions_sha256"`
	// InputSHA256 is the digest of the normalized requests; SentSHA256 that of
	// every request actually sent, in order. Both runs carry the same values.
	InputSHA256 string `json:"input_sha256"`
	SentSHA256  string `json:"sent_sha256"`
}

// Transition is the per-choice transition matrix of one question: Counts[i][j]
// is the number of observations whose reference choice was Choices[i] and
// candidate choice Choices[j]. The diagonal is agreement.
type Transition struct {
	Question string   `json:"question"`
	Choices  []string `json:"choices"`
	Counts   [][]int  `json:"counts"`
}

// FidelityFlip is one observation whose chosen option differs between the
// reference and the candidate. Effect is set only when labels exist.
type FidelityFlip struct {
	CaseID              string  `json:"case_id"`
	QuestionID          string  `json:"question_id"`
	ReferenceChoice     string  `json:"reference_choice"`
	CandidateChoice     string  `json:"candidate_choice"`
	ReferenceConfidence float64 `json:"reference_confidence"`
	CandidateConfidence float64 `json:"candidate_confidence"`
	Effect              string  `json:"effect,omitempty"`
}

// HighConfidenceFlips is the subset of reference decisions the reference made
// with confidence at or above Threshold, and how many of them flipped.
type HighConfidenceFlips struct {
	Threshold float64        `json:"threshold"`
	N         int            `json:"n"`
	Flips     int            `json:"flips"`
	FlipRate  *float64       `json:"flip_rate"`
	Items     []FidelityFlip `json:"items"`
}

// QuestionFidelity is the fidelity of one question.
type QuestionFidelity struct {
	Question           string  `json:"question"`
	Paired             int     `json:"paired"`
	Flips              int     `json:"flips"`
	FlipRate           float64 `json:"flip_rate"`
	MeanJS             float64 `json:"mean_js_divergence"`
	MaxProbabilityDiff float64 `json:"max_probability_delta"`
}

// Fidelity is the ground-truth-free evidence: how much compression changed the
// typed decisions, measured against the reference alone. It needs no labels.
//
// JS is the Jensen-Shannon divergence between the two option distributions of
// an observation, in bits (log base 2), so it is bounded by [0, 1]; 0 means
// identical distributions. Probability deltas are absolute differences of
// option probabilities: Mean is per observation the mean over options, Max the
// largest over options. RankPreserved counts observations whose complete option
// ranking (descending probability, ties by option name) is unchanged.
type Fidelity struct {
	Paired                 int                 `json:"paired_observations"`
	Unpaired               int                 `json:"unpaired_observations"`
	ChoiceFlips            int                 `json:"choice_flips"`
	FlipRate               float64             `json:"flip_rate"`
	TopChoicePreservedRate float64             `json:"top_choice_preserved_rate"`
	RankPreserved          int                 `json:"rank_preserved"`
	RankPreservedRate      float64             `json:"rank_preserved_rate"`
	Transitions            []Transition        `json:"transitions"`
	Flips                  []FidelityFlip      `json:"flips"`
	Confidence             ConfidenceDelta     `json:"confidence_delta"`
	ProbabilityMeanAbs     Spread              `json:"probability_mean_abs_delta"`
	ProbabilityMaxAbs      Spread              `json:"probability_max_abs_delta"`
	JS                     Spread              `json:"js_divergence_bits"`
	HighConfidenceRef      HighConfidenceFlips `json:"high_confidence_reference"`
	PerQuestion            []QuestionFidelity  `json:"per_question"`
	// QuestionTypes counts paired observations by question type. Only the
	// choice type exists in the hachidori.v1 decide contract; noul and score
	// drift are not measurable until those types are served, and are reported
	// as such rather than as zero.
	QuestionTypes  map[string]int `json:"question_types"`
	NoulScoreDrift string         `json:"noul_score_drift"`
}

// ThresholdDelta is the candidate-minus-reference change of one row of the
// confidence-threshold table.
type ThresholdDelta struct {
	Threshold           float64  `json:"threshold"`
	CoveredDelta        int      `json:"covered_delta"`
	CoverageDelta       *float64 `json:"coverage_delta"`
	ConditionalAccuracy *float64 `json:"conditional_accuracy_delta"`
}

// SliceDelta is the QualityDelta of one question or family.
type SliceDelta struct {
	Key      string   `json:"key"`
	N        int      `json:"n"`
	Accuracy *float64 `json:"accuracy_delta"`
	MacroF1  *float64 `json:"macro_f1_delta"`
	ECE      *float64 `json:"ece_delta"`
	Brier    *float64 `json:"brier_delta"`
	NLL      *float64 `json:"nll_delta"`
}

// LabelledEvidence is present only when the dataset carries expected labels.
// It is separate from Fidelity: correctness and calibration of each side
// against labels, and the change between them. Deltas are candidate minus
// reference and are descriptive; the reference is not the truth here either,
// the labels are.
type LabelledEvidence struct {
	HighConfidence float64 `json:"high_confidence_threshold"`
	Reference      Quality `json:"reference"`
	Candidate      Quality `json:"candidate"`
	Delta          QualityDelta
	// HighConfidenceErrorRate is the change of high-confidence errors over all
	// observations.
	HighConfidenceErrorRate *float64         `json:"high_confidence_error_rate_delta"`
	Thresholds              []ThresholdDelta `json:"threshold_deltas"`
	PerQuestion             []SliceDelta     `json:"per_question"`
	PerFamily               []SliceDelta     `json:"per_family"`
	FlipsFixed              int              `json:"flips_fixed"`
	FlipsBroken             int              `json:"flips_broken"`
	FlipsBothWrong          int              `json:"flips_both_wrong"`
}

// ResourceSide is the resource evidence of one run. Nothing is estimated: a
// value the resident did not report is null and its fact NOT_CHECKED.
type ResourceSide struct {
	LoadMS           *float64       `json:"load_ms"`
	WarmupMS         *float64       `json:"warmup_ms"`
	RequestLatency   Latency        `json:"request_latency"`
	InferenceLatency Latency        `json:"inference_latency"`
	Memory           Memory         `json:"memory"`
	Requests         int            `json:"requests"`
	Succeeded        int            `json:"succeeded"`
	ErrorCount       int            `json:"error_count"`
	ErrorsByClass    map[string]int `json:"errors_by_class"`
}

// ResourceEvidence is separate from fidelity and quality: load and warmup,
// inference and round-trip latency, and RAM and VRAM where reported. Checks
// records, per fact, whether it was MEASURED or is NOT_CHECKED (for example no
// VRAM for a CPU reference, or a run on a machine that was not measured).
type ResourceEvidence struct {
	Reference        ResourceSide      `json:"reference"`
	Candidate        ResourceSide      `json:"candidate"`
	RequestLatency   LatencyRatios     `json:"request_latency_ratio"`
	InferenceLatency LatencyRatios     `json:"inference_latency_ratio"`
	LoadMS           *Ratio            `json:"load_ratio"`
	WarmupMS         *Ratio            `json:"warmup_ratio"`
	Memory           MemoryDeltas      `json:"accelerator_memory_ratio"`
	HostRSS          *Ratio            `json:"host_rss_ratio"`
	Checks           map[string]string `json:"checks"`
}

// Certification is one auditable certification artifact.
type Certification struct {
	Schema    string             `json:"schema"`
	CreatedAt string             `json:"created_at"`
	Source    home.VariantSource `json:"source"`
	Variant   CertifiedVariant   `json:"variant"`
	Reference Execution          `json:"reference"`
	Candidate Execution          `json:"candidate"`
	Dataset   CertifiedDataset   `json:"dataset"`

	// Evidence: what changed.
	Fidelity  Fidelity          `json:"fidelity"`
	Labelled  *LabelledEvidence `json:"labelled,omitempty"`
	Resources ResourceEvidence  `json:"resources"`

	// Policy and verdict: whether that is acceptable. The policy is embedded
	// whole and its digest recorded; the verdict is a pure function of the
	// policy and the evidence above and is recomputed wherever it is trusted.
	Policy       CertPolicy `json:"policy"`
	PolicySHA256 string     `json:"policy_sha256"`
	Verdict      Verdict    `json:"verdict"`
}

// CertifyInput is everything Certify binds: the exact source and variant, the
// two runs, and the policy.
type CertifyInput struct {
	Source    home.ModelManifest
	Variant   home.VariantManifest
	Reference ResidentRun
	Candidate ResidentRun
	Policy    CertPolicy
	Now       time.Time
}

// Certify compares a reference run and a candidate run of one variant. It
// refuses, listing every reason, unless the runs are the pinned source model
// at high precision and the exact variant, over identical datasets, question
// identities and normalized requests. Deltas are never computed across runs
// that are not like for like. The returned artifact carries the evidence and,
// separately, the verdict of the policy; a rejected verdict is still a complete
// artifact.
func Certify(in CertifyInput) (Certification, error) {
	var why []string
	add := func(f string, a ...any) { why = append(why, fmt.Sprintf(f, a...)) }
	if err := in.Policy.Validate(); err != nil {
		return Certification{}, fmt.Errorf("certification policy: %w", err)
	}
	if err := in.Variant.Validate(); err != nil {
		return Certification{}, fmt.Errorf("variant manifest: %w", err)
	}
	if err := in.Variant.CheckSource(in.Source); err != nil {
		return Certification{}, err
	}
	ref, cand := in.Reference, in.Candidate
	for _, x := range []struct {
		role string
		run  ResidentRun
	}{{"reference", ref}, {"candidate", cand}} {
		if err := ValidateResidentRun(x.run); err != nil {
			add("%s run: %v", x.role, err)
		}
	}
	if len(why) > 0 {
		return Certification{}, errors.New("runs cannot be certified: " + strings.Join(why, "; "))
	}
	if ref.Labelled != cand.Labelled {
		add("one run is labelled and the other is not")
	}
	if al := Align([]ModelRun{ref.Run, cand.Run}); al.Status != AlignAligned {
		for _, i := range al.Incompatibility {
			add("%s %s: %s", i.Code, i.Question, i.Detail)
		}
	}
	if !equalControls(ref.Declared, cand.Declared) {
		add("the declared high-confidence threshold, coverage thresholds or length-bucket edges differ")
	}
	re, ce := executionOf("reference", ref.Run), executionOf("candidate", cand.Run)
	src := in.Source
	for _, e := range []Execution{re, ce} {
		switch {
		case e.ModelID != src.ID:
			add("%s run served model %q, not the source model %q", e.Role, e.ModelID, src.ID)
		case e.Revision != src.Revision:
			add("%s run served revision %q, not the source revision %q", e.Role, e.Revision, src.Revision)
		case e.Provider != src.Provider:
			add("%s run reports provider %q, not %q", e.Role, e.Provider, src.Provider)
		}
		if e.Device == "" || e.DType == "" {
			add("%s run does not report the device and dtype its resident is on", e.Role)
		}
	}
	if re.VariantID != "" {
		add("the reference run executed variant %q: the reference must be the source model", re.VariantID)
	}
	if re.DType != "torch.float32" && re.DType != "torch.bfloat16" {
		add("the reference run is %q, not a high-precision (float32 or bfloat16) execution", re.DType)
	}
	if ce.VariantID != in.Variant.ID {
		add("the candidate run executed variant %q, not %q", ce.VariantID, in.Variant.ID)
	}
	if want := "torch." + in.Variant.Weights.DType; ce.DType != want {
		add("the candidate run is %s, but the variant declares %s execution", ce.DType, want)
	}
	if len(why) > 0 {
		return Certification{}, errors.New("runs cannot be certified: " + strings.Join(why, "; "))
	}

	fid, flips, err := fidelityOf(ref.Run, cand.Run, in.Policy.HighConfidence, ref.Labelled)
	if err != nil {
		return Certification{}, err
	}
	now := in.Now
	if now.IsZero() {
		now = time.Now()
	}
	v := in.Variant
	c := Certification{
		Schema: CertificationSchema, CreatedAt: now.UTC().Format(time.RFC3339), Source: v.Source,
		Variant: CertifiedVariant{ID: v.ID, ManifestSHA256: v.ManifestSHA256(), BuildID: v.BuildID, Recipe: v.Recipe.Name,
			RecipeSHA256: v.RecipeSHA256, Scheme: v.Weights.Scheme, Engine: v.Optimizer.Engine, EngineVersion: v.Optimizer.Version},
		Reference: re, Candidate: ce,
		Dataset: CertifiedDataset{SHA256: ref.DatasetSHA256, Cases: ref.Run.Cases, Observations: len(ref.Run.Observations),
			Labelled: ref.Labelled, Definitions: ref.Definitions, Questions: ref.Run.Questions,
			QuestionsSHA256: digestJSON(ref.Run.Questions), InputSHA256: ref.Run.InputSHA256, SentSHA256: ref.Run.SentSHA256},
		Fidelity: fid, Resources: resourcesOf(ref.Run, cand.Run),
		Policy: in.Policy, PolicySHA256: in.Policy.SHA256(),
	}
	if ref.Labelled {
		c.Labelled = labelledOf(ref.Run, cand.Run, flips)
	}
	c.Verdict = in.Policy.Evaluate(c)
	return c, nil
}

func equalControls(a, b Declared) bool {
	return a.HighConfidence == b.HighConfidence && fmt.Sprint(a.Thresholds) == fmt.Sprint(b.Thresholds) &&
		fmt.Sprint(a.LengthEdges) == fmt.Sprint(b.LengthEdges)
}

func digestJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// fidelityOf pairs the two runs' observations by (case, question) and
// measures how the candidate's typed decisions differ from the reference's.
func fidelityOf(a, b ModelRun, highConf float64, labelled bool) (Fidelity, []Flip, error) {
	f := Fidelity{Transitions: []Transition{}, Flips: []FidelityFlip{}, QuestionTypes: map[string]int{},
		HighConfidenceRef: HighConfidenceFlips{Threshold: highConf, Items: []FidelityFlip{}},
		NoulScoreDrift:    "not measurable: the hachidori.v1 decide contract serves choice questions only"}
	index := map[pairKey]ResidentObservation{}
	for _, o := range a.Observations {
		k := pairKey{o.CaseID, o.QuestionID}
		if _, dup := index[k]; dup {
			return f, nil, fmt.Errorf("the reference has two observations of case %q question %q", k.caseID, k.questionID)
		}
		index[k] = o
	}
	type qacc struct {
		paired, flips int
		js, maxDiff   float64
		choices       map[string]bool
		pairs         [][2]string
	}
	perQ := map[string]*qacc{}
	var meanAbs, maxAbs, js, confAbs []float64
	var confSigned float64
	var flips []Flip
	seen := map[pairKey]bool{}
	for _, o := range b.Observations {
		k := pairKey{o.CaseID, o.QuestionID}
		if seen[k] {
			return f, nil, fmt.Errorf("the candidate has two observations of case %q question %q", k.caseID, k.questionID)
		}
		seen[k] = true
		p, ok := index[k]
		if !ok {
			f.Unpaired++
			continue
		}
		delete(index, k)
		if !sameKeys(p.Probabilities, o.Probabilities) {
			return f, nil, fmt.Errorf("case %q question %q: the runs scored different option sets", k.caseID, k.questionID)
		}
		f.Paired++
		f.QuestionTypes["choice"]++
		q := perQ[k.questionID]
		if q == nil {
			q = &qacc{choices: map[string]bool{}}
			perQ[k.questionID] = q
		}
		q.paired++
		for opt := range p.Probabilities {
			q.choices[opt] = true
		}
		q.pairs = append(q.pairs, [2]string{p.Choice, o.Choice})
		sum, worst := 0.0, 0.0
		for _, opt := range optionOrder(p.Probabilities) {
			d := math.Abs(o.Probabilities[opt] - p.Probabilities[opt])
			sum += d
			worst = math.Max(worst, d)
		}
		meanAbs = append(meanAbs, sum/float64(len(p.Probabilities)))
		maxAbs = append(maxAbs, worst)
		d := jsBits(p.Probabilities, o.Probabilities)
		js = append(js, d)
		q.js += d
		q.maxDiff = math.Max(q.maxDiff, worst)
		confSigned += o.Confidence - p.Confidence
		confAbs = append(confAbs, math.Abs(o.Confidence-p.Confidence))
		if sameRanking(p.Probabilities, o.Probabilities) {
			f.RankPreserved++
		}
		high := p.Confidence >= highConf
		if high {
			f.HighConfidenceRef.N++
		}
		if p.Choice != o.Choice {
			ff := FidelityFlip{CaseID: k.caseID, QuestionID: k.questionID, ReferenceChoice: p.Choice, CandidateChoice: o.Choice,
				ReferenceConfidence: p.Confidence, CandidateConfidence: o.Confidence}
			fl := Flip{CaseID: k.caseID, QuestionID: k.questionID, Expected: p.Expected, BaselineChoice: p.Choice,
				CandidateChoice: o.Choice, BaselineConfidence: p.Confidence, CandidateConfidence: o.Confidence}
			if labelled {
				switch {
				case !p.Correct && o.Correct:
					fl.Effect = FlipFixed
				case p.Correct && !o.Correct:
					fl.Effect = FlipBroken
				default:
					fl.Effect = FlipBothWrong
				}
				ff.Effect = fl.Effect
			}
			f.Flips = append(f.Flips, ff)
			flips = append(flips, fl)
			q.flips++
			if high {
				f.HighConfidenceRef.Flips++
				f.HighConfidenceRef.Items = append(f.HighConfidenceRef.Items, ff)
			}
		}
	}
	f.Unpaired += len(index)
	f.ChoiceFlips = len(f.Flips)
	sort.Slice(f.Flips, func(i, j int) bool {
		x, y := f.Flips[i], f.Flips[j]
		if x.CaseID != y.CaseID {
			return x.CaseID < y.CaseID
		}
		return x.QuestionID < y.QuestionID
	})
	sort.Slice(f.HighConfidenceRef.Items, func(i, j int) bool {
		x, y := f.HighConfidenceRef.Items[i], f.HighConfidenceRef.Items[j]
		if x.CaseID != y.CaseID {
			return x.CaseID < y.CaseID
		}
		return x.QuestionID < y.QuestionID
	})
	if f.Paired > 0 {
		n := float64(f.Paired)
		f.FlipRate = float64(f.ChoiceFlips) / n
		f.TopChoicePreservedRate = 1 - f.FlipRate
		f.RankPreservedRate = float64(f.RankPreserved) / n
		f.Confidence = ConfidenceDelta{MeanSigned: confSigned / n, Abs: spread(confAbs)}
	}
	if f.HighConfidenceRef.N > 0 {
		f.HighConfidenceRef.FlipRate = ptr(float64(f.HighConfidenceRef.Flips) / float64(f.HighConfidenceRef.N))
	}
	f.ProbabilityMeanAbs, f.ProbabilityMaxAbs, f.JS = spread(meanAbs), spread(maxAbs), spread(js)
	ids := make([]string, 0, len(perQ))
	for id := range perQ {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		q := perQ[id]
		choices := make([]string, 0, len(q.choices))
		for c := range q.choices {
			choices = append(choices, c)
		}
		sort.Strings(choices)
		pos := map[string]int{}
		for i, c := range choices {
			pos[c] = i
		}
		counts := make([][]int, len(choices))
		for i := range counts {
			counts[i] = make([]int, len(choices))
		}
		for _, pr := range q.pairs {
			counts[pos[pr[0]]][pos[pr[1]]]++
		}
		f.Transitions = append(f.Transitions, Transition{Question: id, Choices: choices, Counts: counts})
		f.PerQuestion = append(f.PerQuestion, QuestionFidelity{Question: id, Paired: q.paired, Flips: q.flips,
			FlipRate: float64(q.flips) / float64(q.paired), MeanJS: q.js / float64(q.paired), MaxProbabilityDiff: q.maxDiff})
	}
	return f, flips, nil
}

// jsBits is the Jensen-Shannon divergence of two option distributions in
// bits, bounded by [0, 1].
func jsBits(p, q map[string]float64) float64 {
	var d float64
	for _, k := range optionOrder(p) {
		pv, qv := p[k], q[k]
		m := (pv + qv) / 2
		if pv > 0 && m > 0 {
			d += 0.5 * pv * math.Log2(pv/m)
		}
		if qv > 0 && m > 0 {
			d += 0.5 * qv * math.Log2(qv/m)
		}
	}
	return math.Min(1, math.Max(0, d))
}

// optionOrder is the options of a distribution in a fixed order: floating
// point sums over them are then reproducible, which an audited artifact needs.
func optionOrder(p map[string]float64) []string {
	r := make([]string, 0, len(p))
	for k := range p {
		r = append(r, k)
	}
	sort.Strings(r)
	return r
}

func ranking(p map[string]float64) []string {
	r := make([]string, 0, len(p))
	for k := range p {
		r = append(r, k)
	}
	sort.Slice(r, func(i, j int) bool {
		if p[r[i]] != p[r[j]] {
			return p[r[i]] > p[r[j]]
		}
		return r[i] < r[j]
	})
	return r
}

func sameRanking(a, b map[string]float64) bool {
	ra, rb := ranking(a), ranking(b)
	for i := range ra {
		if ra[i] != rb[i] {
			return false
		}
	}
	return true
}

// labelledOf is the label-based evidence. It reuses the Quality of each run,
// computed by QualityOf at the declared controls, and the QualityDelta and
// slice arithmetic of the precision comparison; no formula is duplicated.
func labelledOf(a, b ModelRun, flips []Flip) *LabelledEvidence {
	qa, qb := a.Quality, b.Quality
	l := &LabelledEvidence{Reference: qa, Candidate: qb,
		Delta: QualityDelta{Accuracy: sub(qb.Accuracy, qa.Accuracy), MacroF1: sub(qb.MacroF1, qa.MacroF1),
			ECE: sub(qb.Calibration.ECE, qa.Calibration.ECE), Brier: sub(qb.Calibration.Brier, qa.Calibration.Brier),
			NLL: sub(qb.Calibration.NLL, qa.Calibration.NLL), HighConfidenceErrors: qb.HighConfidence.Errors - qa.HighConfidence.Errors},
		HighConfidenceErrorRate: sub(qb.HighConfidence.RateOfObservations, qa.HighConfidence.RateOfObservations),
		HighConfidence:          qa.HighConfidence.Threshold,
		Thresholds:              []ThresholdDelta{}, PerQuestion: sliceDeltas(a.PerQuestion, b.PerQuestion), PerFamily: sliceDeltas(a.PerFamily, b.PerFamily)}
	for i, ra := range qa.Thresholds {
		if i >= len(qb.Thresholds) {
			break
		}
		rb := qb.Thresholds[i]
		l.Thresholds = append(l.Thresholds, ThresholdDelta{Threshold: ra.Threshold, CoveredDelta: rb.Covered - ra.Covered,
			CoverageDelta: sub(rb.Coverage, ra.Coverage), ConditionalAccuracy: sub(rb.ConditionalAccuracy, ra.ConditionalAccuracy)})
	}
	for _, f := range flips {
		switch f.Effect {
		case FlipFixed:
			l.FlipsFixed++
		case FlipBroken:
			l.FlipsBroken++
		case FlipBothWrong:
			l.FlipsBothWrong++
		}
	}
	return l
}

func sliceDeltas(a, b []Slice) []SliceDelta {
	out := []SliceDelta{}
	idx := map[string]Slice{}
	for _, s := range b {
		idx[s.Key] = s
	}
	for _, x := range a {
		y, ok := idx[x.Key]
		if !ok {
			continue
		}
		out = append(out, SliceDelta{Key: x.Key, N: x.N, Accuracy: sub(y.Accuracy, x.Accuracy), MacroF1: sub(y.MacroF1, x.MacroF1),
			ECE: sub(y.Calibration.ECE, x.Calibration.ECE), Brier: sub(y.Calibration.Brier, x.Calibration.Brier),
			NLL: sub(y.Calibration.NLL, x.Calibration.NLL)})
	}
	return out
}

func resourcesOf(a, b ModelRun) ResourceEvidence {
	side := func(m ModelRun) ResourceSide {
		return ResourceSide{LoadMS: m.Startup.LoadMS, WarmupMS: m.Startup.WarmupMS, RequestLatency: m.RequestLatency,
			InferenceLatency: m.InferenceLatency, Memory: m.Memory, Requests: m.Requests, Succeeded: m.Succeeded,
			ErrorCount: m.ErrorCount, ErrorsByClass: m.ErrorsByClass}
	}
	r := ResourceEvidence{Reference: side(a), Candidate: side(b),
		RequestLatency: latencyRatios(a.RequestLatency, b.RequestLatency), InferenceLatency: latencyRatios(a.InferenceLatency, b.InferenceLatency),
		LoadMS: msRatio(a.Startup.LoadMS, b.Startup.LoadMS), WarmupMS: msRatio(a.Startup.WarmupMS, b.Startup.WarmupMS),
		HostRSS: byteRatio(a.Memory.PeakHostRSS, b.Memory.PeakHostRSS), Checks: map[string]string{}}
	if a.Memory.Available && b.Memory.Available {
		r.Memory = MemoryDeltas{ResidentAllocated: byteRatio(a.Memory.ResidentAlloc, b.Memory.ResidentAlloc),
			ResidentReserved: byteRatio(a.Memory.ResidentRsrv, b.Memory.ResidentRsrv),
			PeakAllocated:    byteRatio(a.Memory.PeakAlloc, b.Memory.PeakAlloc), PeakReserved: byteRatio(a.Memory.PeakRsrv, b.Memory.PeakRsrv)}
	}
	mark := func(name string, measured bool) {
		if measured {
			r.Checks[name] = FactMeasured
		} else {
			r.Checks[name] = FactNotChecked
		}
	}
	for role, m := range map[string]ModelRun{"reference": a, "candidate": b} {
		mark(role+"_load_time", m.Startup.LoadMS != nil)
		mark(role+"_warmup_time", m.Startup.WarmupMS != nil)
		mark(role+"_request_latency", m.RequestLatency.N > 0)
		mark(role+"_inference_latency", m.InferenceLatency.N > 0)
		mark(role+"_vram", m.Memory.Available)
		mark(role+"_host_ram", m.Memory.PeakHostRSS != nil)
	}
	return r
}
