package eval

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
)

func TestECE(t *testing.T) {
	// Two bins: conf 0.9 (1/2 correct) and conf 0.6 (2/2 correct).
	got := ECE([]float64{0.9, 0.9, 0.6, 0.6}, []bool{true, false, true, true}, 15)
	want := 0.5*math.Abs(0.9-0.5) + 0.5*math.Abs(0.6-1.0)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("ECE = %v, want %v", got, want)
	}
	if ECE([]float64{0}, []bool{false}, 15) != 0 {
		t.Fatal("zero confidence must land in the first bin")
	}
}

type stub struct {
	answers map[string]string // state -> choice for every question
	seen    []api.DecideRequest
}

func (s *stub) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	s.seen = append(s.seen, r)
	if r.State == "fail" {
		return api.DecideResponse{}, errors.New("worker_failure")
	}
	var rs []api.Result
	for _, q := range r.Questions {
		rs = append(rs, api.Result{ID: q.ID, Choice: s.answers[r.State], Confidence: 0.8})
	}
	return api.DecideResponse{Results: rs, Timing: &api.Timing{InferenceMS: 5}}, nil
}

func q(id string) api.Question {
	return api.Question{ID: id, Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}
}

func TestRun(t *testing.T) {
	cases := []Case{
		{ID: "c1", State: "a", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "no"}},
		{ID: "c2", State: "b", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
	}
	s := &stub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := Run(s, cases, Options{Warmup: 1, Passes: 2})
	if r.Observations != 3 || math.Abs(r.ChoiceAccuracy-2.0/3) > 1e-12 {
		t.Fatalf("obs %d acc %v", r.Observations, r.ChoiceAccuracy)
	}
	if r.PerQuestion["x"].Accuracy != 1 || r.PerQuestion["y"].Accuracy != 0 {
		t.Fatalf("per question %+v", r.PerQuestion)
	}
	if len(s.seen) != 1+2*2 || r.RequestLatency.N != 4 || r.ServerInference.P95 != 5 {
		t.Fatalf("calls %d latency %+v", len(s.seen), r.RequestLatency)
	}
	if r.Results[0].CaseID != "c1" || r.Results[0].QuestionID != "x" {
		t.Fatalf("ids not preserved: %+v", r.Results[0])
	}

	r = Run(s, []Case{{ID: "c3", State: "fail", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}}}, Options{})
	if len(r.Errors) != 1 || r.Observations != 0 {
		t.Fatalf("errors must not be scored: %+v", r)
	}
}

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.jsonl")
	os.WriteFile(good, []byte(`{"id":"a","state":"s","questions":[{"id":"x","type":"choice","instructions":"i","choices":["yes","no"]}],"expected":{"x":"yes"}}`+"\n"), 0o644)
	cases, sum, err := Load(good)
	if err != nil || len(cases) != 1 || len(sum) != 64 {
		t.Fatal(cases, sum, err)
	}
	for name, line := range map[string]string{
		"bad label":  `{"id":"a","state":"s","questions":[{"id":"x","type":"choice","instructions":"i","choices":["yes","no"]}],"expected":{"x":"maybe"}}`,
		"no label":   `{"id":"a","state":"s","questions":[{"id":"x","type":"choice","instructions":"i","choices":["yes","no"]}],"expected":{}}`,
		"no case id": `{"state":"s","questions":[{"id":"x","type":"choice","instructions":"i","choices":["yes","no"]}],"expected":{"x":"yes"}}`,
	} {
		p := filepath.Join(dir, "bad.jsonl")
		os.WriteFile(p, []byte(line+"\n"), 0o644)
		if _, _, err := Load(p); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestContractExampleFixture(t *testing.T) {
	if _, _, err := Load("../../testdata/eval/contract-example.jsonl"); err != nil {
		t.Fatal(err)
	}
}
