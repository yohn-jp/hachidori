package eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
)

// PolicySchema identifies a certification policy: an explicit, versioned set
// of thresholds that turns the evidence of a certification into accepted or
// rejected. A policy is a declaration, not a ranking: it says nothing about
// which model is better, only whether this variant's change from its
// reference stays within the stated bounds.
const PolicySchema = "hachidori.certification-policy/1"

var policyIDRe = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*/[1-9][0-9]*$`)

// CertPolicy is a certification profile. Its ID carries its version
// ("name/N"); changing any threshold requires a new ID. Its SHA-256 is
// recorded in every certification and record that applied it.
//
// The fidelity thresholds apply to every certification. Labelled, when
// present, additionally bounds the label-based degradation and is applied
// only to labelled runs; a nil Labelled applies no label criteria at all.
type CertPolicy struct {
	Schema string `json:"schema"`
	ID     string `json:"id"`
	// MinObservations is the least number of paired observations the evidence
	// must rest on.
	MinObservations int `json:"min_observations"`
	// HighConfidence defines "high-confidence reference decisions": those the
	// reference made with at least this confidence.
	HighConfidence            float64 `json:"high_confidence"`
	MaxFlipRate               float64 `json:"max_flip_rate"`
	MaxHighConfidenceFlipRate float64 `json:"max_high_confidence_flip_rate"`
	MaxMeanJS                 float64 `json:"max_mean_js_bits"`
	MaxP95ProbabilityDelta    float64 `json:"max_p95_probability_delta"`
	MaxProbabilityDelta       float64 `json:"max_probability_delta"`
	// MaxRequestErrors bounds the failed requests of each run, and the
	// observations left unpaired between them.
	MaxRequestErrors int `json:"max_request_errors"`
	// RequireStable requires that neither resident was stopped, restarted or
	// reloaded while it was measured.
	RequireStable bool            `json:"require_stable_residents"`
	Labelled      *LabelledPolicy `json:"labelled,omitempty"`
}

// LabelledPolicy bounds the degradation of label-based quality, candidate
// minus reference: accuracy and macro-F1 may fall by at most the stated
// amounts, calibration error and loss may rise by at most theirs.
type LabelledPolicy struct {
	MaxAccuracyDrop                float64 `json:"max_accuracy_drop"`
	MaxMacroF1Drop                 float64 `json:"max_macro_f1_drop"`
	MaxECEIncrease                 float64 `json:"max_ece_increase"`
	MaxBrierIncrease               float64 `json:"max_brier_increase"`
	MaxNLLIncrease                 float64 `json:"max_nll_increase"`
	MaxHighConfidenceErrorIncrease float64 `json:"max_high_confidence_error_rate_increase"`
}

// DefaultPolicyID is the identity of the built-in profile.
const DefaultPolicyID = "system-one-fidelity/1"

// DefaultPolicy is the built-in certification profile.
func DefaultPolicy() CertPolicy {
	return CertPolicy{Schema: PolicySchema, ID: DefaultPolicyID, MinObservations: 100, HighConfidence: DefaultHighConfidence,
		MaxFlipRate: 0.03, MaxHighConfidenceFlipRate: 0.01, MaxMeanJS: 0.01, MaxP95ProbabilityDelta: 0.10,
		MaxProbabilityDelta: 0.50, MaxRequestErrors: 0, RequireStable: true,
		Labelled: &LabelledPolicy{MaxAccuracyDrop: 0.02, MaxMacroF1Drop: 0.03, MaxECEIncrease: 0.03, MaxBrierIncrease: 0.03,
			MaxNLLIncrease: 0.10, MaxHighConfidenceErrorIncrease: 0.01}}
}

// Canonical is the policy's canonical encoding.
func (p CertPolicy) Canonical() []byte {
	b, err := json.Marshal(p)
	if err != nil {
		panic(err)
	}
	return b
}

// SHA256 is the policy digest.
func (p CertPolicy) SHA256() string {
	sum := sha256.Sum256(p.Canonical())
	return hex.EncodeToString(sum[:])
}

// Validate checks the policy is complete and its thresholds are meaningful.
func (p CertPolicy) Validate() error {
	if p.Schema != PolicySchema {
		return fmt.Errorf("policy schema %q, want %q", p.Schema, PolicySchema)
	}
	if !policyIDRe.MatchString(p.ID) {
		return fmt.Errorf("policy id %q must be a versioned identity such as name/1", p.ID)
	}
	if p.MinObservations < 1 {
		return errors.New("policy min_observations must be at least 1")
	}
	unit := func(name string, v float64) error {
		if math.IsNaN(v) || v < 0 || v > 1 {
			return fmt.Errorf("policy %s must be within [0, 1], got %v", name, v)
		}
		return nil
	}
	if p.HighConfidence <= 0 || p.HighConfidence > 1 || math.IsNaN(p.HighConfidence) {
		return fmt.Errorf("policy high_confidence must be within (0, 1], got %v", p.HighConfidence)
	}
	for name, v := range map[string]float64{"max_flip_rate": p.MaxFlipRate, "max_high_confidence_flip_rate": p.MaxHighConfidenceFlipRate,
		"max_mean_js_bits": p.MaxMeanJS, "max_p95_probability_delta": p.MaxP95ProbabilityDelta, "max_probability_delta": p.MaxProbabilityDelta} {
		if err := unit(name, v); err != nil {
			return err
		}
	}
	if p.MaxRequestErrors < 0 {
		return errors.New("policy max_request_errors must not be negative")
	}
	if l := p.Labelled; l != nil {
		for name, v := range map[string]float64{"max_accuracy_drop": l.MaxAccuracyDrop, "max_macro_f1_drop": l.MaxMacroF1Drop,
			"max_ece_increase": l.MaxECEIncrease, "max_brier_increase": l.MaxBrierIncrease,
			"max_high_confidence_error_rate_increase": l.MaxHighConfidenceErrorIncrease} {
			if err := unit("labelled."+name, v); err != nil {
				return err
			}
		}
		if math.IsNaN(l.MaxNLLIncrease) || l.MaxNLLIncrease < 0 {
			return fmt.Errorf("policy labelled.max_nll_increase must not be negative, got %v", l.MaxNLLIncrease)
		}
	}
	return nil
}

// DecodePolicy parses a policy strictly (unknown fields and trailing data are
// rejected) and validates it.
func DecodePolicy(data []byte) (CertPolicy, error) {
	var p CertPolicy
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return CertPolicy{}, fmt.Errorf("malformed %s: %w", PolicySchema, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return CertPolicy{}, fmt.Errorf("malformed %s: trailing data", PolicySchema)
	}
	return p, p.Validate()
}

// LoadPolicy reads a policy file.
func LoadPolicy(path string) (CertPolicy, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return CertPolicy{}, err
	}
	p, err := DecodePolicy(b)
	if err != nil {
		return p, fmt.Errorf("%s: %w", path, err)
	}
	return p, nil
}

// Criterion is one threshold of a policy applied to the evidence. Applied is
// false when the evidence the criterion needs does not exist (labelled
// criteria on an unlabelled run); an unapplied criterion neither passes nor
// fails the verdict and is reported as such. Limit is the bound and Observed
// the measured value; Pass is meaningful only when Applied.
type Criterion struct {
	Name     string   `json:"name"`
	Limit    *float64 `json:"limit,omitempty"`
	Observed *float64 `json:"observed,omitempty"`
	Applied  bool     `json:"applied"`
	Pass     bool     `json:"pass"`
	Detail   string   `json:"detail,omitempty"`
}

// Verdict is the policy's decision about one certification's evidence. It is
// a pure function of (policy, evidence): Evaluate recomputes it identically
// wherever it is trusted.
type Verdict struct {
	Status   string      `json:"status"` // accepted | rejected
	Criteria []Criterion `json:"criteria"`
}

// Failed lists the criteria that were applied and not passed.
func (v Verdict) Failed() []string {
	var out []string
	for _, c := range v.Criteria {
		if c.Applied && !c.Pass {
			out = append(out, c.Name)
		}
	}
	return out
}

// Evaluate applies the policy to the evidence of c (its Fidelity, Labelled and
// Resources sections; its own Policy and Verdict fields are not read).
// Boundaries are inclusive: a value equal to its limit passes.
func (p CertPolicy) Evaluate(c Certification) Verdict {
	v := Verdict{Status: VerdictAccepted}
	upper := func(name string, observed, limit float64, detail string) {
		cr := Criterion{Name: name, Limit: ptr(limit), Observed: ptr(observed), Applied: true, Pass: observed <= limit, Detail: detail}
		v.Criteria = append(v.Criteria, cr)
	}
	none := func(name string, limit float64, detail string) {
		v.Criteria = append(v.Criteria, Criterion{Name: name, Limit: ptr(limit), Applied: false, Detail: detail})
	}
	f := c.Fidelity
	enough := f.Paired >= p.MinObservations
	v.Criteria = append(v.Criteria, Criterion{Name: "min_observations", Limit: ptr(float64(p.MinObservations)),
		Observed: ptr(float64(f.Paired)), Applied: true, Pass: enough, Detail: "paired observations the evidence rests on"})
	upper("unpaired_and_errors", float64(f.Unpaired+maxInt(c.Resources.Reference.ErrorCount, c.Resources.Candidate.ErrorCount)),
		float64(p.MaxRequestErrors), "observations unpaired between the runs plus the larger request error count of the two runs")
	if p.RequireStable {
		stable := c.Reference.ResidentStable && c.Candidate.ResidentStable
		v.Criteria = append(v.Criteria, Criterion{Name: "stable_residents", Applied: true, Pass: stable,
			Detail: "neither resident was stopped, restarted or reloaded while measured"})
	}
	upper("flip_rate", f.FlipRate, p.MaxFlipRate, "share of paired observations whose choice changed")
	if f.HighConfidenceRef.FlipRate != nil {
		upper("high_confidence_flip_rate", *f.HighConfidenceRef.FlipRate, p.MaxHighConfidenceFlipRate,
			"share of high-confidence reference decisions whose choice changed")
	} else {
		none("high_confidence_flip_rate", p.MaxHighConfidenceFlipRate, "the reference made no high-confidence decision")
	}
	upper("mean_js_bits", f.JS.Mean, p.MaxMeanJS, "mean Jensen-Shannon divergence between option distributions, in bits")
	upper("p95_probability_delta", f.ProbabilityMaxAbs.P95, p.MaxP95ProbabilityDelta, "95th percentile of the largest option probability change")
	upper("max_probability_delta", f.ProbabilityMaxAbs.Max, p.MaxProbabilityDelta, "largest option probability change of any observation")
	if l := p.Labelled; l != nil {
		if c.Labelled == nil {
			for _, x := range []struct {
				name string
				lim  float64
			}{{"accuracy_drop", l.MaxAccuracyDrop}, {"macro_f1_drop", l.MaxMacroF1Drop}, {"ece_increase", l.MaxECEIncrease},
				{"brier_increase", l.MaxBrierIncrease}, {"nll_increase", l.MaxNLLIncrease},
				{"high_confidence_error_rate_increase", l.MaxHighConfidenceErrorIncrease}} {
				none(x.name, x.lim, "the dataset carries no expected labels")
			}
		} else {
			d := c.Labelled.Delta
			drop := func(name string, delta *float64, limit float64, detail string) {
				if delta == nil {
					none(name, limit, "not defined for this run")
					return
				}
				upper(name, -*delta, limit, detail)
			}
			rise := func(name string, delta *float64, limit float64, detail string) {
				if delta == nil {
					none(name, limit, "not defined for this run")
					return
				}
				upper(name, *delta, limit, detail)
			}
			drop("accuracy_drop", d.Accuracy, l.MaxAccuracyDrop, "reference accuracy minus candidate accuracy")
			drop("macro_f1_drop", d.MacroF1, l.MaxMacroF1Drop, "reference macro-F1 minus candidate macro-F1")
			rise("ece_increase", d.ECE, l.MaxECEIncrease, "candidate ECE minus reference ECE")
			rise("brier_increase", d.Brier, l.MaxBrierIncrease, "candidate Brier minus reference Brier")
			rise("nll_increase", d.NLL, l.MaxNLLIncrease, "candidate NLL minus reference NLL")
			rise("high_confidence_error_rate_increase", c.Labelled.HighConfidenceErrorRate, l.MaxHighConfidenceErrorIncrease,
				"candidate minus reference high-confidence errors over all observations")
		}
	}
	for _, cr := range v.Criteria {
		if cr.Applied && !cr.Pass {
			v.Status = VerdictRejected
		}
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
