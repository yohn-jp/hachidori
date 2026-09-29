package eval

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/yohn-jp/hachidori/internal/api"
)

// ReplaySchema identifies the replay output format.
const ReplaySchema = "hachidori.replay.v1"

// LoadReport reads a Decision Evidence report and rejects other schemas.
func LoadReport(path string) (Report, error) {
	var r Report
	b, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return r, fmt.Errorf("%s: %w", path, err)
	}
	if r.Schema != EvidenceSchema {
		return r, fmt.Errorf("%s: evidence schema %q, want %q", path, r.Schema, EvidenceSchema)
	}
	return r, nil
}

// Selection limits a replay to case and/or question ids; empty means all.
type Selection struct {
	Cases     []string
	Questions []string
}

// ReplayItem is one reconstructed inference request and the recorded
// observations it reproduces, in request question order.
type ReplayItem struct {
	CaseID   string            `json:"case_id"`
	Request  api.DecideRequest `json:"request"`
	Recorded []Observation     `json:"-"`
}

// ReplayRequests reconstructs, deterministically and without contacting the
// endpoint, the /v1/decide requests behind the selected observations of r.
//
// It refuses when the dataset digest differs from the recorded one, when a
// recorded case or question is absent from the dataset, or when a
// question's wire digest differs from the recorded one. The request is the
// case's full original request (all questions, as sent); expected labels
// are never part of it.
func ReplayRequests(r Report, cases []Case, datasetSHA256 string, sel Selection) ([]ReplayItem, error) {
	if r.DatasetSHA256 == "" || datasetSHA256 != r.DatasetSHA256 {
		return nil, fmt.Errorf("dataset sha256 %s does not match recorded %s; refusing to replay", datasetSHA256, r.DatasetSHA256)
	}
	byID := map[string]Case{}
	for _, c := range cases {
		byID[c.ID] = c
	}
	wantCase, wantQ := set(sel.Cases), set(sel.Questions)
	var items []ReplayItem
	idx := map[string]int{}
	for _, o := range r.Results {
		if (len(wantCase) > 0 && !wantCase[o.CaseID]) || (len(wantQ) > 0 && !wantQ[o.QuestionID]) {
			continue
		}
		c, ok := byID[o.CaseID]
		if !ok {
			return nil, fmt.Errorf("case %q is not in the dataset; refusing to replay", o.CaseID)
		}
		var q *api.Question
		for i := range c.Questions {
			if c.Questions[i].ID == o.QuestionID {
				q = &c.Questions[i]
			}
		}
		if q == nil {
			return nil, fmt.Errorf("case %q has no question %q; refusing to replay", o.CaseID, o.QuestionID)
		}
		if got := QuestionSHA256(*q); got != o.QuestionSHA256 {
			return nil, fmt.Errorf("case %q question %q digest %s does not match recorded %s; refusing to replay",
				o.CaseID, o.QuestionID, got, o.QuestionSHA256)
		}
		i, ok := idx[c.ID]
		if !ok {
			i = len(items)
			idx[c.ID] = i
			items = append(items, ReplayItem{CaseID: c.ID, Request: c.Request()})
		}
		items[i].Recorded = append(items[i].Recorded, o)
	}
	for _, id := range sel.Cases {
		if _, ok := idx[id]; !ok {
			return nil, fmt.Errorf("case %q has no selected recorded observation", id)
		}
	}
	for _, id := range sel.Questions {
		found := false
		for _, it := range items {
			for _, o := range it.Recorded {
				found = found || o.QuestionID == id
			}
		}
		if !found {
			return nil, fmt.Errorf("question %q has no selected recorded observation", id)
		}
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("no recorded observations match the selection")
	}
	return items, nil
}

// ReplayResult compares one recorded observation with its replay.
// MaxProbDelta is the largest absolute probability difference over the
// union of choices; Error is set, and nothing is compared, when the replay
// request failed or returned no result for the question.
type ReplayResult struct {
	CaseID         string             `json:"case_id"`
	QuestionID     string             `json:"question_id"`
	RecordedChoice string             `json:"recorded_choice"`
	RecordedProbs  map[string]float64 `json:"recorded_probabilities"`
	ReplayedChoice string             `json:"replayed_choice,omitempty"`
	ReplayedProbs  map[string]float64 `json:"replayed_probabilities,omitempty"`
	SameChoice     bool               `json:"same_choice"`
	MaxProbDelta   float64            `json:"max_probability_delta"`
	Error          *RequestError      `json:"error,omitempty"`
}

// ReplayReport is the output of a replay. ServedMatches is false when the
// endpoint now serves a different identity than the one recorded (a
// cross-model replay), so the comparison is never silently mixed.
type ReplayReport struct {
	Schema         string         `json:"schema"`
	Endpoint       string         `json:"endpoint"`
	DatasetSHA256  string         `json:"dataset_sha256"`
	RecordedServed *Served        `json:"recorded_served"`
	Served         *Served        `json:"served"`
	ServedMatches  bool           `json:"served_matches"`
	Results        []ReplayResult `json:"results"`
}

// Replay sends the reconstructed requests through the existing /v1/decide
// contract and compares the results with the recorded observations.
func Replay(d Endpoint, r Report, items []ReplayItem) (ReplayReport, error) {
	out := ReplayReport{Schema: ReplaySchema, DatasetSHA256: r.DatasetSHA256, RecordedServed: r.Served, Results: []ReplayResult{}}
	now, err := Snapshot(d)
	if err != nil {
		return out, fmt.Errorf("cannot identify served runtime: %w", err)
	}
	out.Served = &now
	out.ServedMatches = r.Served != nil && r.Served.Digest == now.Digest
	for _, it := range items {
		resp, err := d.Decide(it.Request)
		got := map[string]api.Result{}
		if err == nil {
			for _, res := range resp.Results {
				got[res.ID] = res
			}
		}
		for _, o := range it.Recorded {
			rr := ReplayResult{CaseID: o.CaseID, QuestionID: o.QuestionID, RecordedChoice: o.Choice, RecordedProbs: o.Probabilities}
			res, ok := got[o.QuestionID]
			switch {
			case err != nil:
				cls, msg := classify(err)
				rr.Error = &RequestError{Phase: "replay", CaseID: o.CaseID, Class: cls, Message: msg}
			case !ok:
				rr.Error = &RequestError{Phase: "replay", CaseID: o.CaseID, QuestionID: o.QuestionID,
					Class: ErrClassMissingResult, Message: "response has no result for this question"}
			default:
				rr.ReplayedChoice, rr.ReplayedProbs = res.Choice, res.Probabilities
				rr.SameChoice = res.Choice == o.Choice
				rr.MaxProbDelta = maxDelta(o.Probabilities, res.Probabilities)
			}
			out.Results = append(out.Results, rr)
		}
	}
	return out, nil
}

func maxDelta(a, b map[string]float64) float64 {
	d := 0.0
	for k, v := range a {
		d = math.Max(d, math.Abs(v-b[k]))
	}
	for k, v := range b {
		d = math.Max(d, math.Abs(a[k]-v))
	}
	return d
}

func set(xs []string) map[string]bool {
	m := map[string]bool{}
	for _, x := range xs {
		m[x] = true
	}
	return m
}
