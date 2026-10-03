package eval

import (
	"encoding/json"
	"reflect"
	"strings"
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

func servedReport(r Report, model, revision, variant string) Report {
	prov := map[string]any{"model_revision": revision}
	if variant != "" {
		prov["variant_id"] = variant
	}
	r.Served = &Served{Runtime: map[string]any{"model_id": model}, Provider: prov, Digest: "served-" + model + variant}
	r.ServedConsistent = true
	return r
}

// A source baseline and a variant candidate of one model over one dataset and
// identical question identities bind to their exact recorded identities.
func TestBindExactIdentities(t *testing.T) {
	qs := map[string]string{"x": "s1", "y": "s2"}
	base := servedReport(cmpReport("ds", qs, 0.9, 0.1), "clef-flash", "rev1", "")
	cand := servedReport(cmpReport("ds", qs, 0.7, 0.2), "clef-flash", "rev1", "clef-flash--r--aaaaaaaaaaaa")
	b, err := Bind(base, cand, "sha-base", "sha-cand")
	if err != nil {
		t.Fatal(err)
	}
	if b.DatasetSHA256 != "ds" || b.QuestionsSHA256 != QuestionIdentitiesSHA256(base) || b.QuestionsSHA256 != QuestionIdentitiesSHA256(cand) {
		t.Fatalf("dataset/question identity %+v", b)
	}
	if b.Baseline != (RunIdentity{EvidenceSHA256: "sha-base", ModelID: "clef-flash", Revision: "rev1", ServedIdentity: "served-clef-flash"}) ||
		b.Candidate != (RunIdentity{EvidenceSHA256: "sha-cand", ModelID: "clef-flash", Revision: "rev1", VariantID: "clef-flash--r--aaaaaaaaaaaa", ServedIdentity: "served-clef-flashclef-flash--r--aaaaaaaaaaaa"}) {
		t.Fatalf("run identity %+v %+v", b.Baseline, b.Candidate)
	}
	if b.Comparison.Status != CompareCompatible || b.Comparison.Aggregate.Accuracy.Diff >= 0 {
		t.Fatalf("comparison %+v", b.Comparison)
	}
	// A different question identity changes the question digest.
	other := servedReport(cmpReport("ds", map[string]string{"x": "s1", "y": "other"}, 0.7, 0.2), "clef-flash", "rev1", "v")
	if QuestionIdentitiesSHA256(other) == b.QuestionsSHA256 {
		t.Fatal("question identity digest ignores the question identity")
	}
}

// Anything that is not like for like is refused with its reason, never
// compared as equivalent.
func TestBindRefusesMismatchedIdentity(t *testing.T) {
	qs := map[string]string{"x": "s1"}
	source := func() Report { return servedReport(cmpReport("ds", qs, 0.9, 0.1), "clef-flash", "rev1", "") }
	variant := func() Report { return servedReport(cmpReport("ds", qs, 0.7, 0.2), "clef-flash", "rev1", "v1") }
	cases := map[string]struct {
		base, cand func() Report
		want       string
	}{
		"different model":         {source, func() Report { return servedReport(cmpReport("ds", qs, 0.7, 0.2), "laya-base", "rev1", "v1") }, "different source models"},
		"different revision":      {source, func() Report { return servedReport(cmpReport("ds", qs, 0.7, 0.2), "clef-flash", "rev2", "v1") }, "different source models"},
		"baseline is a variant":   {func() Report { return servedReport(cmpReport("ds", qs, 0.9, 0.1), "clef-flash", "rev1", "v0") }, variant, "baseline must be the source"},
		"candidate is the source": {source, source, "does not name an executed variant"},
		"different dataset":       {source, func() Report { return servedReport(cmpReport("other", qs, 0.7, 0.2), "clef-flash", "rev1", "v1") }, "dataset"},
		"different question identity": {source, func() Report {
			return servedReport(cmpReport("ds", map[string]string{"x": "changed"}, 0.7, 0.2), "clef-flash", "rev1", "v1")
		}, "question"},
		"extra question": {source, func() Report {
			return servedReport(cmpReport("ds", map[string]string{"x": "s1", "y": "s2"}, 0.7, 0.2), "clef-flash", "rev1", "v1")
		}, "question"},
		"served identity changed": {source, func() Report { r := variant(); r.ServedConsistent = false; return r }, "served_consistent=false"},
		"no served identity":      {source, func() Report { r := variant(); r.Served = nil; return r }, "no served identity"},
	}
	for name, c := range cases {
		b, err := Bind(c.base(), c.cand(), "sha-base", "sha-cand")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want %q", name, err, c.want)
		}
		if b.Comparison.Aggregate != nil || b.DatasetSHA256 != "" {
			t.Errorf("%s: refused binding still carries deltas", name)
		}
	}
	if _, err := Bind(source(), variant(), "", "sha-cand"); err == nil {
		t.Error("evidence without a digest was bound")
	}
}
