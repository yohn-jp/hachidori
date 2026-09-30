package eval

import (
	"fmt"
	"sort"
	"strings"
)

// Comparison states. None of them is a verdict about which run is better: they
// only say how much of the two reports can be compared like for like.
const (
	CompareCompatible = "compatible" // same dataset, every question identity-aligned
	ComparePartial    = "partial"    // same dataset; some questions cannot be aligned
	CompareRefused    = "refused"    // reports are not comparable; no deltas
)

// Incompatibility codes.
const (
	IncompatSchema           = "schema_mismatch"
	IncompatDataset          = "dataset_mismatch"
	IncompatCases            = "case_count_mismatch"
	IncompatOnlyInA          = "question_only_in_a"
	IncompatOnlyInB          = "question_only_in_b"
	IncompatQuestionIdentity = "question_identity_mismatch"
)

// Incompatibility is one explicit reason the reports are not fully
// comparable. Question is set for question-level reasons.
type Incompatibility struct {
	Code     string
	Question string
	Detail   string
}

// Delta is one metric of both reports and B minus A. The comparison is
// directional and descriptive: a positive Delta means the value is larger in
// B, never that B is better.
type Delta struct {
	A, B, Diff float64
}

// CountDelta is Delta for integer counts.
type CountDelta struct {
	A, B, Diff int
}

// LatencyDelta compares two latency summaries with the report's own
// definitions (Latency). Available is false when either side has no samples;
// the deltas are then not computed rather than compared against zero.
type LatencyDelta struct {
	Available bool
	N         CountDelta
	P50       Delta
	P95       Delta
	Mean      Delta
}

// Aggregate holds the whole-report deltas, reusing the metrics stored in the
// reports (choice accuracy, mean confidence, ECE, latency summaries, counts).
type Aggregate struct {
	Accuracy        Delta
	MeanConfidence  Delta
	ECE             Delta
	Cases           CountDelta
	Observations    CountDelta
	RequestErrors   CountDelta
	RequestLatency  LatencyDelta
	ServerInference LatencyDelta
}

// QuestionDelta is the per-question delta of one identity-aligned question,
// from the reports' own per-question statistics.
type QuestionDelta struct {
	ID             string
	N              CountDelta
	Accuracy       Delta
	MeanConfidence Delta
	ECE            Delta
}

// RunContext is descriptive context of one run that may explain a difference.
// It never gates the comparison.
type RunContext struct {
	StartedAt      string
	ServedModel    string
	ServedIdentity string
	WarmupRequests int
	Passes         int
}

// Comparison is the descriptive comparison of two reports A and B.
type Comparison struct {
	Status          string
	Incompatibility []Incompatibility
	DatasetA        string // dataset sha256 of each report
	DatasetB        string
	A, B            RunContext
	Aggregate       *Aggregate      // nil when refused
	Questions       []QuestionDelta // identity-aligned questions only, by id
}

// Compare compares two strictly decoded reports. It is a pure function of its
// arguments: it reads no file, contacts no endpoint, does not depend on either
// report's endpoint or display names, and modifies neither report.
//
// Reports are comparable only for the same dataset (identical dataset sha256).
// Questions are aligned by question id together with the exact question
// identity recorded in the evidence (question sha256 values and Question
// Definition identity); a question present in only one report, or whose
// identity differs, is listed as incompatible and gets no per-question delta.
func Compare(a, b Report) Comparison {
	c := Comparison{DatasetA: a.DatasetSHA256, DatasetB: b.DatasetSHA256, A: contextOf(a), B: contextOf(b)}
	if a.Schema != EvidenceSchema || b.Schema != EvidenceSchema {
		c.Status = CompareRefused
		c.Incompatibility = append(c.Incompatibility, Incompatibility{Code: IncompatSchema,
			Detail: fmt.Sprintf("schemas %q and %q; both must be %q", a.Schema, b.Schema, EvidenceSchema)})
		return c
	}
	if a.DatasetSHA256 != b.DatasetSHA256 {
		c.Status = CompareRefused
		c.Incompatibility = append(c.Incompatibility, Incompatibility{Code: IncompatDataset,
			Detail: fmt.Sprintf("dataset sha256 %s differs from %s", a.DatasetSHA256, b.DatasetSHA256)})
		return c
	}
	if a.Cases != b.Cases {
		c.Incompatibility = append(c.Incompatibility, Incompatibility{Code: IncompatCases,
			Detail: fmt.Sprintf("%d cases in A, %d in B", a.Cases, b.Cases)})
	}
	ia, ib := questionIdentities(a), questionIdentities(b)
	ids := map[string]bool{}
	for id := range ia {
		ids[id] = true
	}
	for id := range ib {
		ids[id] = true
	}
	sorted := make([]string, 0, len(ids))
	for id := range ids {
		sorted = append(sorted, id)
	}
	sort.Strings(sorted)
	for _, id := range sorted {
		x, inA := ia[id]
		y, inB := ib[id]
		switch {
		case !inB:
			c.Incompatibility = append(c.Incompatibility, Incompatibility{Code: IncompatOnlyInA, Question: id,
				Detail: "question is only in A; no per-question delta"})
		case !inA:
			c.Incompatibility = append(c.Incompatibility, Incompatibility{Code: IncompatOnlyInB, Question: id,
				Detail: "question is only in B; no per-question delta"})
		case x != y:
			c.Incompatibility = append(c.Incompatibility, Incompatibility{Code: IncompatQuestionIdentity, Question: id,
				Detail: "same question id but a different question or Question Definition identity; not aligned"})
		default:
			p, q := a.PerQuestion[id], b.PerQuestion[id]
			c.Questions = append(c.Questions, QuestionDelta{ID: id, N: countDelta(p.N, q.N),
				Accuracy: delta(p.Accuracy, q.Accuracy), MeanConfidence: delta(p.MeanConfidence, q.MeanConfidence),
				ECE: delta(p.ECE, q.ECE)})
		}
	}
	c.Status = CompareCompatible
	if len(c.Incompatibility) > 0 {
		c.Status = ComparePartial
	}
	c.Aggregate = &Aggregate{
		Accuracy: delta(a.ChoiceAccuracy, b.ChoiceAccuracy), MeanConfidence: delta(a.MeanConfidence, b.MeanConfidence),
		ECE: delta(a.ECE, b.ECE), Cases: countDelta(a.Cases, b.Cases),
		Observations: countDelta(a.Observations, b.Observations), RequestErrors: countDelta(len(a.Errors), len(b.Errors)),
		RequestLatency:  latencyDelta(a.RequestLatency, b.RequestLatency),
		ServerInference: latencyDelta(a.ServerInference, b.ServerInference),
	}
	return c
}

func contextOf(r Report) RunContext {
	c := RunContext{StartedAt: r.StartedAt, ServedModel: r.Served.Model(), WarmupRequests: r.WarmupRequests, Passes: r.Passes}
	if r.Served != nil {
		c.ServedIdentity = r.Served.Digest
	}
	return c
}

// questionIdentities maps each question id in the report's observations to a
// canonical signature of every distinct identity recorded for it.
func questionIdentities(r Report) map[string]string {
	sets := map[string]map[string]bool{}
	for _, o := range r.Results {
		s := o.QuestionSHA256
		if o.QuestionDefinition != nil {
			s += "|" + o.QuestionDefinition.String() + "#" + o.QuestionDefinition.Digest
		}
		if sets[o.QuestionID] == nil {
			sets[o.QuestionID] = map[string]bool{}
		}
		sets[o.QuestionID][s] = true
	}
	out := make(map[string]string, len(sets))
	for id, set := range sets {
		parts := make([]string, 0, len(set))
		for s := range set {
			parts = append(parts, s)
		}
		sort.Strings(parts)
		out[id] = strings.Join(parts, "\n")
	}
	return out
}

func delta(a, b float64) Delta { return Delta{A: a, B: b, Diff: b - a} }

func countDelta(a, b int) CountDelta { return CountDelta{A: a, B: b, Diff: b - a} }

func latencyDelta(a, b Latency) LatencyDelta {
	d := LatencyDelta{N: countDelta(a.N, b.N)}
	if a.N == 0 || b.N == 0 {
		return d
	}
	d.Available = true
	d.P50, d.P95, d.Mean = delta(a.P50, b.P50), delta(a.P95, b.P95), delta(a.Mean, b.Mean)
	return d
}
