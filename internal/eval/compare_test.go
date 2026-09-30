package eval

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/yohn-jp/hachidori/internal/question"
)

func cmpReport(dataset string, qs map[string]string, acc, ece float64) Report {
	r := Report{Schema: EvidenceSchema, Endpoint: "http://one", Dataset: "d.jsonl", DatasetSHA256: dataset, Cases: 2,
		ChoiceAccuracy: acc, ECE: ece, MeanConfidence: 0.8, Passes: 1, Errors: []RequestError{},
		PerQuestion:    map[string]QuestionStats{},
		RequestLatency: Latency{N: 2, P50: 10, P95: 20, Mean: 12}}
	for id, sha := range qs {
		r.Results = append(r.Results, Observation{CaseID: "c", QuestionID: id, QuestionSHA256: sha})
		r.PerQuestion[id] = QuestionStats{N: 1, Accuracy: acc, MeanConfidence: 0.8, ECE: ece}
	}
	r.Observations = len(r.Results)
	return r
}

func TestCompareCompatibleDeltas(t *testing.T) {
	a := cmpReport("ds", map[string]string{"x": "s1", "y": "s2"}, 0.5, 0.2)
	b := cmpReport("ds", map[string]string{"x": "s1", "y": "s2"}, 0.75, 0.1)
	b.RequestLatency = Latency{N: 4, P50: 15, P95: 40, Mean: 20}
	b.Errors = []RequestError{{Phase: "pass", Class: "transport"}}
	b.Endpoint, b.Dataset = "http://two", "renamed.jsonl" // display names never matter
	c := Compare(a, b)
	if c.Status != CompareCompatible || len(c.Incompatibility) != 0 {
		t.Fatalf("%+v", c)
	}
	ag := c.Aggregate
	if ag.Accuracy != (Delta{0.5, 0.75, 0.25}) || ag.ECE.Diff != 0.1-0.2 {
		t.Fatalf("aggregate %+v", ag)
	}
	if !ag.RequestLatency.Available || ag.RequestLatency.P95.Diff != 20 || ag.RequestLatency.N.Diff != 2 {
		t.Fatalf("latency %+v", ag.RequestLatency)
	}
	if ag.ServerInference.Available {
		t.Fatal("inference latency delta computed without samples")
	}
	if ag.RequestErrors != (CountDelta{0, 1, 1}) || ag.Observations.Diff != 0 {
		t.Fatalf("counts %+v", ag)
	}
	if len(c.Questions) != 2 || c.Questions[0].ID != "x" || c.Questions[1].ID != "y" || c.Questions[0].Accuracy.Diff != 0.25 {
		t.Fatalf("questions %+v", c.Questions)
	}
}

func TestCompareDeterministicAndPure(t *testing.T) {
	a := cmpReport("ds", map[string]string{"x": "s1", "y": "s2", "z": "s3"}, 0.5, 0.2)
	b := cmpReport("ds", map[string]string{"x": "s1", "y": "s9", "w": "s4"}, 0.6, 0.3)
	before, _ := json.Marshal([]Report{a, b})
	first := Compare(a, b)
	for i := 0; i < 20; i++ {
		if got := Compare(a, b); !reflect.DeepEqual(got, first) {
			t.Fatal("comparison is not deterministic")
		}
	}
	after, _ := json.Marshal([]Report{a, b})
	if string(before) != string(after) {
		t.Fatal("comparison modified a report")
	}
	// Different endpoints do not change the comparison.
	b2 := b
	b2.Endpoint = "http://elsewhere"
	if got := Compare(a, b2); !reflect.DeepEqual(got, first) {
		t.Fatal("comparison depends on the endpoint")
	}
}

func TestCompareRefusesDifferentDataset(t *testing.T) {
	a := cmpReport("ds1", map[string]string{"x": "s1"}, 0.5, 0.2)
	b := cmpReport("ds2", map[string]string{"x": "s1"}, 0.6, 0.2)
	b.Dataset = a.Dataset // same display name, different digest
	c := Compare(a, b)
	if c.Status != CompareRefused || c.Aggregate != nil || len(c.Questions) != 0 ||
		len(c.Incompatibility) != 1 || c.Incompatibility[0].Code != IncompatDataset {
		t.Fatalf("%+v", c)
	}
}

func TestCompareRefusesOtherSchema(t *testing.T) {
	a := cmpReport("ds", map[string]string{"x": "s1"}, 0.5, 0.2)
	b := a
	b.Schema = "hachidori.evidence.v2"
	c := Compare(a, b)
	if c.Status != CompareRefused || c.Aggregate != nil || c.Incompatibility[0].Code != IncompatSchema {
		t.Fatalf("%+v", c)
	}
}

func TestComparePartialQuestionAlignment(t *testing.T) {
	a := cmpReport("ds", map[string]string{"x": "s1", "y": "s2", "onlya": "s3"}, 0.5, 0.2)
	b := cmpReport("ds", map[string]string{"x": "s1", "y": "CHANGED", "onlyb": "s4"}, 0.6, 0.3)
	c := Compare(a, b)
	if c.Status != ComparePartial || c.Aggregate == nil {
		t.Fatalf("%+v", c)
	}
	if len(c.Questions) != 1 || c.Questions[0].ID != "x" {
		t.Fatalf("only identity-aligned questions get deltas: %+v", c.Questions)
	}
	got := map[string]string{}
	for _, in := range c.Incompatibility {
		got[in.Question] = in.Code
	}
	want := map[string]string{"onlya": IncompatOnlyInA, "onlyb": IncompatOnlyInB, "y": IncompatQuestionIdentity}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("incompatibilities %v, want %v", got, want)
	}
}

func TestCompareDefinitionIdentityAlignsQuestions(t *testing.T) {
	a := cmpReport("ds", map[string]string{"x": "s1"}, 0.5, 0.2)
	b := cmpReport("ds", map[string]string{"x": "s1"}, 0.5, 0.2)
	a.Results[0].QuestionDefinition = &question.Identity{ID: "x", Version: 1, Digest: "d1"}
	b.Results[0].QuestionDefinition = &question.Identity{ID: "x", Version: 1, Digest: "d1"}
	if c := Compare(a, b); c.Status != CompareCompatible {
		t.Fatalf("%+v", c)
	}
	b.Results[0].QuestionDefinition = &question.Identity{ID: "x", Version: 2, Digest: "d2"}
	if c := Compare(a, b); c.Status != ComparePartial || len(c.Questions) != 0 || c.Incompatibility[0].Code != IncompatQuestionIdentity {
		t.Fatalf("%+v", c)
	}
	// Inline in one report, definition-backed in the other: not aligned.
	b.Results[0].QuestionDefinition = nil
	if c := Compare(a, b); c.Status != ComparePartial {
		t.Fatalf("%+v", c)
	}
}

func TestCompareCaseCountMismatchIsPartial(t *testing.T) {
	a := cmpReport("ds", map[string]string{"x": "s1"}, 0.5, 0.2)
	b := cmpReport("ds", map[string]string{"x": "s1"}, 0.5, 0.2)
	b.Cases = 3
	c := Compare(a, b)
	if c.Status != ComparePartial || c.Incompatibility[0].Code != IncompatCases || len(c.Questions) != 1 {
		t.Fatalf("%+v", c)
	}
}
