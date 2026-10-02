package api

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func valid() DecideRequest {
	return DecideRequest{Schema: SchemaV1, State: "s", Questions: []Question{
		{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}, Descriptions: map[string]string{"yes": "y"}}}}
}

func TestValidate(t *testing.T) {
	if err := func() *DecideRequest { r := valid(); return &r }().Validate(); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*DecideRequest){
		"schema":           func(r *DecideRequest) { r.Schema = "v0" },
		"empty state":      func(r *DecideRequest) { r.State = " " },
		"large state":      func(r *DecideRequest) { r.State = strings.Repeat("x", MaxStateBytes+1) },
		"no questions":     func(r *DecideRequest) { r.Questions = nil },
		"type":             func(r *DecideRequest) { r.Questions[0].Type = "score" },
		"one choice":       func(r *DecideRequest) { r.Questions[0].Choices = []string{"yes"} },
		"dup choice":       func(r *DecideRequest) { r.Questions[0].Choices = []string{"a", "a"} },
		"no instructions":  func(r *DecideRequest) { r.Questions[0].Instructions = "" },
		"dup id":           func(r *DecideRequest) { r.Questions = append(r.Questions, r.Questions[0]) },
		"unknown describe": func(r *DecideRequest) { r.Questions[0].Descriptions = map[string]string{"maybe": "m"} },
	}
	for name, mut := range cases {
		r := valid()
		r.Questions = append([]Question(nil), r.Questions...)
		mut(&r)
		if r.Validate() == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestBatchValidate(t *testing.T) {
	r := valid()
	r.Schema = ""
	b := BatchRequest{Schema: SchemaV1, Requests: []DecideRequest{r}}
	if err := b.Validate(); err != nil {
		t.Fatal(err)
	}
	if (&BatchRequest{Schema: SchemaV1}).Validate() == nil {
		t.Fatal("empty batch accepted")
	}
}

// TestWireSchemaV1 pins the v1 JSON field set of requests and results.
func TestWireSchemaV1(t *testing.T) {
	req := DecideRequest{Schema: SchemaV1, State: "s", Questions: []Question{
		{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}, Descriptions: map[string]string{"yes": "y"}}}}
	b, _ := json.Marshal(req)
	if want := `{"schema":"hachidori.v1","state":"s","questions":[{"id":"q","type":"choice","instructions":"i","choices":["yes","no"],"descriptions":{"yes":"y"}}]}`; string(b) != want {
		t.Fatalf("request wire form\n got %s\nwant %s", b, want)
	}
	resp := DecideResponse{Schema: SchemaV1, Results: []Result{{ID: "q", Type: "choice", Choice: "yes", Confidence: 0.9,
		Probabilities: map[string]float64{"no": 0.1, "yes": 0.9}}}, Timing: &Timing{InferenceMS: 1, TotalMS: 2}}
	b, _ = json.Marshal(resp)
	if want := `{"schema":"hachidori.v1","results":[{"id":"q","type":"choice","choice":"yes","confidence":0.9,"probabilities":{"no":0.1,"yes":0.9}}],"timing":{"inference_ms":1,"total_ms":2}}`; string(b) != want {
		t.Fatalf("response wire form\n got %s\nwant %s", b, want)
	}
}

// The model selector is optional and additive: absent it the wire form is the
// v1 one above; present it must be a catalog-ID-shaped token, never a
// repository or revision reference.
func TestModelSelector(t *testing.T) {
	r := valid()
	if b, _ := json.Marshal(r); strings.Contains(string(b), "model") {
		t.Fatalf("an unrouted request carries a model field: %s", b)
	}
	if err := r.Validate(); err != nil {
		t.Errorf("omitted model: %v", err)
	}
	for _, ok := range []string{"laya-base", "opendecider-nano", "a_b.c-1"} {
		r.Model = &ok
		if err := r.Validate(); err != nil {
			t.Errorf("model %q: %v", ok, err)
		}
	}
	// An explicitly empty selector is invalid; it is not an omitted one.
	for _, bad := range []string{"", " ", " laya-base", "org/repo", "org/repo@rev", "a b", "https://x/y", "mé", strings.Repeat("x", 129)} {
		r.Model = &bad
		if r.Validate() == nil {
			t.Errorf("model %q accepted", bad)
		}
	}
	// omitted is the zero selector; "-" marks nothing, "" marks an explicit empty string.
	sel := func(m string) *string {
		if m == "-" {
			return nil
		}
		return &m
	}
	for _, tc := range []struct {
		batch string
		items []string
		want  string
		ok    bool
	}{
		{"-", []string{"-", "-"}, "", true},
		{"a", []string{"-", "-"}, "a", true},
		{"-", []string{"a", "a"}, "a", true},
		{"-", []string{"-", "a"}, "a", true},
		{"a", []string{"a", "-"}, "a", true},
		{"a", []string{"b"}, "", false},
		{"-", []string{"a", "b"}, "", false},
		{"", []string{"-"}, "", false},
		{"-", []string{""}, "", false},
		{"-", []string{"-", ""}, "", false},
		{"a", []string{""}, "", false},
		{"-", []string{"a", ""}, "", false},
		{"-", []string{" "}, "", false},
	} {
		b := BatchRequest{Schema: SchemaV1, Model: sel(tc.batch)}
		for _, m := range tc.items {
			r := valid()
			r.Model = sel(m)
			b.Requests = append(b.Requests, r)
		}
		if (b.Validate() == nil) != tc.ok {
			t.Errorf("batch %+v: validate %v", tc, b.Validate())
		}
		if tc.ok {
			if got, err := b.Target(); err != nil || got != tc.want {
				t.Errorf("batch %+v: target %q err %v", tc, got, err)
			}
		}
	}
}

// The wire form distinguishes an omitted selector from an explicit empty one.
func TestModelSelectorWireForm(t *testing.T) {
	var r DecideRequest
	if err := json.Unmarshal([]byte(`{"schema":"hachidori.v1"}`), &r); err != nil || r.Model != nil {
		t.Fatalf("omitted decoded as %v, err %v", r.Model, err)
	}
	if err := json.Unmarshal([]byte(`{"schema":"hachidori.v1","model":""}`), &r); err != nil || r.Model == nil || *r.Model != "" {
		t.Fatalf("empty decoded as %v, err %v", r.Model, err)
	}
	if b, _ := json.Marshal(r); !strings.Contains(string(b), `"model":""`) {
		t.Fatalf("explicit empty selector lost on the wire: %s", b)
	}
}

func TestResultValidateAppliesTheTypedDecisionContract(t *testing.T) {
	q := Question{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}
	ok := Result{ID: "q", Type: "choice", Choice: "yes", Confidence: 0.75, Probabilities: map[string]float64{"yes": 0.75, "no": 0.25}}
	if err := ok.Validate(q); err != nil {
		t.Fatal(err)
	}
	mod := func(f func(*Result)) Result {
		r := ok
		r.Probabilities = map[string]float64{"yes": 0.75, "no": 0.25}
		f(&r)
		return r
	}
	nan := math.NaN()
	for name, r := range map[string]Result{
		"another question":          mod(func(r *Result) { r.ID = "x" }),
		"a choice not offered":      mod(func(r *Result) { r.Choice = "maybe" }),
		"a missing probability":     mod(func(r *Result) { r.Probabilities = map[string]float64{"yes": 1} }),
		"an unknown option":         mod(func(r *Result) { r.Probabilities = map[string]float64{"yes": 0.75, "maybe": 0.25} }),
		"a NaN probability":         mod(func(r *Result) { r.Probabilities["no"] = nan }),
		"an infinite probability":   mod(func(r *Result) { r.Probabilities["no"] = math.Inf(1) }),
		"a negative probability":    mod(func(r *Result) { r.Probabilities["no"] = -0.25; r.Probabilities["yes"] = 1.25 }),
		"probabilities not summing": mod(func(r *Result) { r.Probabilities["no"] = 0.5 }),
		"a NaN confidence":          mod(func(r *Result) { r.Confidence = nan }),
		"confidence not max p":      mod(func(r *Result) { r.Confidence = 0.5 }),
		"a non-argmax choice":       mod(func(r *Result) { r.Choice, r.Confidence = "no", 0.25 }),
	} {
		if err := r.Validate(q); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
}
