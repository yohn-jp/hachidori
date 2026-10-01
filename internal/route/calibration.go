package route

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/yohn-jp/hachidori/internal/eval"
)

// Calibration is the evidence a policy was verified against: the digest of
// the resident comparison report and the dataset it was measured on, and the
// slice of that report behind each rule that routes on confidence or margin.
type Calibration struct {
	EvidenceSHA256 string           `json:"evidence_sha256"`
	DatasetSHA256  string           `json:"dataset_sha256"`
	Rules          []CalibratedRule `json:"rules"`
}

// CalibratedRule names the evidence slice behind one threshold rule: the
// rule's family ("" for the default rule), its first-path model and the
// number of observations of that model in that family.
type CalibratedRule struct {
	Family       string `json:"family,omitempty"`
	Model        string `json:"model"`
	Observations int    `json:"observations"`
}

// Verify checks that a policy's confidence and margin thresholds are backed
// by resident comparison evidence (internal/eval), and returns what backs
// them. The evidence must be a valid, aligned comparison, and for every rule
// that routes on confidence_below or margin_below it must contain
// observations of the rule's first-path model in the rule's family (the whole
// run for the default rule). Verify does not choose or adjust a threshold:
// the thresholds stay the operator's, and the report's threshold-by-coverage
// and calibration tables are what they were derived from.
//
// data is the exact comparison document; its digest is recorded.
func Verify(p Policy, data []byte) (Calibration, error) {
	rep, err := eval.DecodeComparison(data)
	if err != nil {
		return Calibration{}, fmt.Errorf("routing evidence: %w", err)
	}
	if rep.Alignment.Status != eval.AlignAligned {
		return Calibration{}, fmt.Errorf("routing evidence: the comparison is not aligned (%s); its slices are not comparable like for like", rep.Alignment.Status)
	}
	sum := sha256.Sum256(data)
	c := Calibration{EvidenceSHA256: "sha256:" + hex.EncodeToString(sum[:]), DatasetSHA256: rep.DatasetSHA256}
	for _, r := range append([]Rule{p.Default}, p.Rules...) {
		if r.Handoff == nil || (r.Handoff.When.ConfidenceBelow == nil && r.Handoff.When.MarginBelow == nil) {
			continue
		}
		n, err := observations(rep, r)
		if err != nil {
			return Calibration{}, err
		}
		c.Rules = append(c.Rules, CalibratedRule{Family: r.Family, Model: r.First, Observations: n})
	}
	return c, nil
}

func observations(rep eval.ComparisonReport, r Rule) (int, error) {
	what := "default rule"
	if r.Family != "" {
		what = "family " + r.Family
	}
	for _, run := range rep.Runs {
		if run.Model != r.First {
			continue
		}
		n := run.Quality.N
		if r.Family != "" {
			n = 0
			for _, s := range run.PerFamily {
				if s.Key == r.Family {
					n = s.N
				}
			}
		}
		if n == 0 {
			return 0, fmt.Errorf("routing evidence has no %s observations of model %s: the threshold has no calibration behind it", what, r.First)
		}
		return n, nil
	}
	return 0, fmt.Errorf("routing evidence has no run of model %s (%s)", r.First, what)
}
