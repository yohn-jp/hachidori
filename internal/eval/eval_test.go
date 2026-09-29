package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
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
	answers  map[string]string // state -> choice for every question
	seen     []api.DecideRequest
	statuses []string // status documents served in order; the last repeats; "" fails
	nstatus  int
}

func statusDoc(model string, uptime int) string {
	return fmt.Sprintf(`{"schema":"hachidori.v1","runtime":{"home":"/h","runtime":"cpu-abc","model":%q,"device":"cpu"},`+
		`"uptime_s":%d,"worker":{"state":"ready","ready":true,"pid":42,"starts":1,`+
		`"provider":{"provider":"laya","laya_version":"0.3.21","device":"cpu","load_ms":%d,"warmup_ms":3}}}`, model, uptime, 100+uptime)
}

func (s *stub) Status() (json.RawMessage, error) {
	if len(s.statuses) == 0 {
		return json.RawMessage(statusDoc("org/laya@rev1", 10)), nil
	}
	d := s.statuses[min(s.nstatus, len(s.statuses)-1)]
	s.nstatus++
	if d == "" {
		return nil, errors.New("connection refused")
	}
	return json.RawMessage(d), nil
}

func (s *stub) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	s.seen = append(s.seen, r)
	switch r.State {
	case "fail":
		return api.DecideResponse{}, &client.APIError{Status: 500, ErrorInfo: api.ErrorInfo{Class: api.ErrInferenceFailed, Message: "boom"}}
	case "down":
		return api.DecideResponse{}, errors.New("connection refused")
	}
	var rs []api.Result
	for _, q := range r.Questions {
		if r.State == "partial" && q.ID == "y" {
			continue
		}
		c := s.answers[r.State]
		p := map[string]float64{}
		for _, ch := range q.Choices {
			p[ch] = 0.2 / float64(len(q.Choices)-1)
		}
		p[c] = 0.8
		rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: c, Confidence: 0.8, Probabilities: p})
	}
	return api.DecideResponse{Schema: api.SchemaV1, Results: rs, Timing: &api.Timing{InferenceMS: 5}}, nil
}

func q(id string) api.Question {
	return api.Question{ID: id, Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}
}

func baseCases() []Case {
	return []Case{
		{ID: "c1", State: "a", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "no"}},
		{ID: "c2", State: "b", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
	}
}

func run(t *testing.T, s *stub, cases []Case, opt Options) Report {
	t.Helper()
	r, err := Run(s, cases, opt)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// Verification 12: aggregate metrics are unchanged by the evidence work.
func TestRunAggregates(t *testing.T) {
	s := &stub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := run(t, s, baseCases(), Options{Warmup: 1, Passes: 2})
	if r.Observations != 3 || math.Abs(r.ChoiceAccuracy-2.0/3) > 1e-12 || math.Abs(r.MeanConfidence-0.8) > 1e-12 {
		t.Fatalf("obs %d acc %v conf %v", r.Observations, r.ChoiceAccuracy, r.MeanConfidence)
	}
	if want := ECE([]float64{0.8, 0.8, 0.8}, []bool{true, false, true}, 15); math.Abs(r.ECE-want) > 1e-12 {
		t.Fatalf("ECE %v want %v", r.ECE, want)
	}
	if r.PerQuestion["x"].Accuracy != 1 || r.PerQuestion["y"].Accuracy != 0 || r.PerQuestion["x"].N != 2 {
		t.Fatalf("per question %+v", r.PerQuestion)
	}
	if len(s.seen) != 1+2*2 || r.RequestLatency.N != 4 || r.ServerInference.P95 != 5 {
		t.Fatalf("calls %d latency %+v", len(s.seen), r.RequestLatency)
	}
	if r.Results[0].CaseID != "c1" || r.Results[0].QuestionID != "x" || r.Results[2].CaseID != "c2" {
		t.Fatalf("ids not preserved: %+v", r.Results)
	}
}

// Verification 1 and 2: full distributions and an explicit schema.
func TestEvidenceRetainsProbabilitiesAndSchema(t *testing.T) {
	cases := []Case{{ID: "c", State: "a", Expected: map[string]string{"m": "b"},
		Questions: []api.Question{{ID: "m", Type: "choice", Instructions: "i", Choices: []string{"a", "b", "c"}}}}}
	s := &stub{answers: map[string]string{"a": "a"}}
	r := run(t, s, cases, Options{})
	o := r.Results[0]
	if len(o.Probabilities) != 3 || o.Probabilities["a"] != 0.8 || o.Probabilities["b"] != 0.1 || o.Probabilities["c"] != 0.1 {
		t.Fatalf("probabilities lost: %+v", o.Probabilities)
	}
	if o.Correct || o.InferenceMS == nil || *o.InferenceMS != 5 {
		t.Fatalf("observation %+v", o)
	}
	b, _ := json.Marshal(r)
	var m map[string]any
	json.Unmarshal(b, &m)
	if m["schema"] != EvidenceSchema || EvidenceSchema != "hachidori.evidence.v1" {
		t.Fatalf("schema %v", m["schema"])
	}
	res := m["results"].([]any)[0].(map[string]any)
	if p := res["probabilities"].(map[string]any); len(p) != 3 {
		t.Fatalf("serialized probabilities %v", p)
	}
}

// Verification 4: served identity comes from endpoint status only, and a
// change during the run is marked rather than hidden.
func TestServedIdentityFromStatus(t *testing.T) {
	answers := map[string]string{"a": "yes", "b": "no"}
	s := &stub{answers: answers}
	r := run(t, s, baseCases(), Options{})
	if r.Served.Model() != "org/laya@rev1" || r.Served.Runtime["runtime"] != "cpu-abc" || r.Served.Provider["laya_version"] != "0.3.21" {
		t.Fatalf("served %+v", r.Served)
	}
	if _, ok := r.Served.Provider["load_ms"]; ok {
		t.Fatal("per-start timings are not identity")
	}
	if !r.ServedConsistent || r.ServedEnd == nil || r.ServedEnd.Digest != r.Served.Digest || len(r.Served.Digest) != 64 {
		t.Fatalf("consistent run marked inconsistent: %+v %+v", r.Served, r.ServedEnd)
	}

	// Model changes between the start and end snapshots.
	s = &stub{answers: answers, statuses: []string{statusDoc("org/laya@rev1", 10), statusDoc("org/laya@rev2", 20)}}
	r = run(t, s, baseCases(), Options{})
	if r.ServedConsistent || r.ServedEnd.Model() != "org/laya@rev2" || r.Served.Model() != "org/laya@rev1" {
		t.Fatalf("model change not marked: %+v", r)
	}
	// Same identity, but the runtime restarted (uptime went backwards).
	s = &stub{answers: answers, statuses: []string{statusDoc("m", 100), statusDoc("m", 3)}}
	if r = run(t, s, baseCases(), Options{}); r.ServedConsistent {
		t.Fatal("runtime restart not marked")
	}
	// End status unavailable: consistency cannot be claimed.
	s = &stub{answers: answers, statuses: []string{statusDoc("m", 1), ""}}
	if r = run(t, s, baseCases(), Options{}); r.ServedConsistent || r.ServedEnd != nil || r.Errors[len(r.Errors)-1].Class != ErrClassStatus {
		t.Fatalf("unverifiable end status: %+v", r)
	}

	// No identity: nothing is sent and no evidence is produced.
	for name, doc := range map[string]string{
		"unreachable": "",
		"no model":    `{"schema":"hachidori.v1","runtime":{"runtime":"r"},"worker":{"ready":true,"provider":{"provider":"laya"}}}`,
		"not ready":   `{"schema":"hachidori.v1","runtime":{"model":"m"},"worker":{"ready":false}}`,
		"schema":      `{"schema":"other","runtime":{"model":"m"},"worker":{"ready":true,"provider":{"provider":"laya"}}}`,
	} {
		s = &stub{statuses: []string{doc}}
		if _, err := Run(s, baseCases(), Options{}); err == nil || len(s.seen) != 0 {
			t.Errorf("%s: err %v, %d requests sent", name, err, len(s.seen))
		}
	}
}

// Verification 5 and 6: question-definition identity is recorded when the
// seam supplies it; inline questions still have an unambiguous wire digest.
func TestQuestionIdentity(t *testing.T) {
	s := &stub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := run(t, s, baseCases(), Options{})
	b, _ := json.Marshal(r.Results[0])
	if strings.Contains(string(b), "question_definition") {
		t.Fatalf("inline question claims a definition: %s", b)
	}
	if r.Results[0].QuestionSHA256 != QuestionSHA256(q("x")) || len(r.Results[0].QuestionSHA256) != 64 {
		t.Fatalf("inline digest %+v", r.Results[0])
	}
	revised := q("x")
	revised.Instructions = "i, revised"
	if QuestionSHA256(revised) == QuestionSHA256(q("x")) {
		t.Fatal("question revisions with the same id must be distinguishable")
	}

	r = run(t, s, baseCases(), Options{QuestionDefinition: func(c Case, q api.Question) string {
		if q.ID == "x" {
			return "def:" + q.ID + "@sha256:abc"
		}
		return ""
	}})
	if r.Results[0].QuestionDefinition != "def:x@sha256:abc" || r.Results[1].QuestionDefinition != "" {
		t.Fatalf("definition identity %+v", r.Results)
	}
}

// Verification 7: expected labels never enter inference requests, neither
// during eval (including warmup and latency passes) nor during replay.
func TestExpectedNeverSent(t *testing.T) {
	s := &stub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := run(t, s, baseCases(), Options{Warmup: 2, Passes: 2})
	for _, req := range s.seen {
		if b := mustJSON(req); strings.Contains(b, "expected") {
			t.Fatalf("request carries expected labels: %s", b)
		}
	}
	r.DatasetSHA256 = "d"
	items, err := ReplayRequests(r, baseCases(), "d", Selection{})
	if err != nil {
		t.Fatal(err)
	}
	for _, it := range items {
		if b := mustJSON(it.Request); strings.Contains(b, "expected") {
			t.Fatalf("replay request carries expected labels: %s", b)
		}
	}
}

// Verification 8: serialization is deterministic and round-trips.
func TestReportSerializationDeterministic(t *testing.T) {
	s := &stub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := run(t, s, baseCases(), Options{QuestionDefinition: func(Case, api.Question) string { return "d" }})
	r.DatasetSHA256 = strings.Repeat("0", 64)
	a, _ := json.MarshalIndent(r, "", "  ")
	for i := 0; i < 20; i++ {
		b, _ := json.MarshalIndent(r, "", "  ")
		if !bytes.Equal(a, b) {
			t.Fatal("serialization is not deterministic")
		}
	}
	p := filepath.Join(t.TempDir(), "r.json")
	os.WriteFile(p, a, 0o644)
	back, err := LoadReport(p)
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := json.MarshalIndent(back, "", "  "); !bytes.Equal(a, b) {
		t.Fatalf("round trip differs:\n%s\n%s", a, b)
	}
	os.WriteFile(p, []byte(`{"schema":"hachidori.evidence.v0"}`), 0o644)
	if _, err := LoadReport(p); err == nil {
		t.Fatal("other evidence schema accepted")
	}
	os.WriteFile(p, []byte(`{"endpoint":"x"}`), 0o644)
	if _, err := LoadReport(p); err == nil {
		t.Fatal("unversioned report accepted")
	}
}

// Verification 11: request errors are never scored as incorrect.
func TestRequestErrorsDistinctFromIncorrect(t *testing.T) {
	cases := append(baseCases(),
		Case{ID: "c3", State: "fail", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
		Case{ID: "c4", State: "down", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
		Case{ID: "c5", State: "partial", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "yes"}},
	)
	s := &stub{answers: map[string]string{"a": "yes", "b": "no", "partial": "yes"}}
	r := run(t, s, cases, Options{})
	if r.Observations != 4 || math.Abs(r.ChoiceAccuracy-3.0/4) > 1e-12 {
		t.Fatalf("obs %d acc %v", r.Observations, r.ChoiceAccuracy)
	}
	want := []RequestError{
		{Phase: "pass", Pass: 1, CaseID: "c3", Class: api.ErrInferenceFailed, Message: "boom"},
		{Phase: "pass", Pass: 1, CaseID: "c4", Class: ErrClassTransport, Message: "connection refused"},
		{Phase: "pass", Pass: 1, CaseID: "c5", QuestionID: "y", Class: ErrClassMissingResult, Message: "response has no result for this question"},
	}
	if fmt.Sprint(r.Errors) != fmt.Sprint(want) {
		t.Fatalf("errors\n got %+v\nwant %+v", r.Errors, want)
	}
	for _, o := range r.Results {
		if o.CaseID == "c3" || o.CaseID == "c4" || (o.CaseID == "c5" && o.QuestionID == "y") {
			t.Fatalf("errored request scored: %+v", o)
		}
	}
}

func writeDataset(t *testing.T, path string, cases []Case) string {
	t.Helper()
	var buf bytes.Buffer
	for _, c := range cases {
		buf.WriteString(mustJSON(c) + "\n")
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	_, sum, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return sum
}

// Verification 3 and 9: replay checks the recorded dataset digest and
// refuses changed or missing source material.
func TestReplay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "d.jsonl")
	sum := writeDataset(t, path, baseCases())
	s := &stub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := run(t, s, baseCases(), Options{})
	r.Dataset, r.DatasetSHA256 = path, sum

	cases, got, _ := Load(path)
	sel := Selection{Cases: []string{"c1"}, Questions: []string{"y"}}
	items, err := ReplayRequests(r, cases, got, sel)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].CaseID != "c1" || len(items[0].Recorded) != 1 || items[0].Recorded[0].QuestionID != "y" {
		t.Fatalf("selection %+v", items)
	}
	if a, b := mustJSON(items[0].Request), mustJSON(baseCases()[0].Request()); a != b {
		t.Fatalf("request not reconstructed exactly:\n%s\n%s", a, b)
	}
	again, _ := ReplayRequests(r, cases, got, sel)
	if mustJSON(items[0].Request) != mustJSON(again[0].Request) {
		t.Fatal("reconstruction is not deterministic")
	}

	s.seen = nil
	rep, err := Replay(s, r, items)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Schema != ReplaySchema || !rep.ServedMatches || len(rep.Results) != 1 || !rep.Results[0].SameChoice ||
		rep.Results[0].MaxProbDelta != 0 || len(s.seen) != 1 {
		t.Fatalf("replay %+v", rep)
	}
	s.statuses = []string{statusDoc("org/other", 1)}
	s.answers["a"] = "no"
	if rep, _ = Replay(s, r, items); rep.ServedMatches || rep.Results[0].SameChoice || math.Abs(rep.Results[0].MaxProbDelta-0.6) > 1e-12 {
		t.Fatalf("cross-model replay not flagged %+v", rep)
	}

	// Changed source: the dataset digest differs.
	changed := baseCases()
	changed[1].State = "b, edited"
	sum2 := writeDataset(t, path, changed)
	cases2, _, _ := Load(path)
	if _, err := ReplayRequests(r, cases2, sum2, Selection{}); err == nil || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("changed dataset accepted: %v", err)
	}
	// Missing source: the recorded dataset path is gone.
	os.Remove(path)
	if _, _, err := Load(r.Dataset); err == nil {
		t.Fatal("missing dataset loaded")
	}
	// Tampered evidence: digests agree but a recorded question or case is
	// not what the dataset holds.
	bad := r
	bad.Results = append([]Observation{}, r.Results...)
	bad.Results[0].QuestionSHA256 = strings.Repeat("f", 64)
	if _, err := ReplayRequests(bad, cases, sum, Selection{}); err == nil {
		t.Fatal("question digest mismatch accepted")
	}
	bad.Results = []Observation{{CaseID: "ghost", QuestionID: "x"}}
	if _, err := ReplayRequests(bad, cases, sum, Selection{}); err == nil {
		t.Fatal("unknown case accepted")
	}
	if _, err := ReplayRequests(r, cases, sum, Selection{Cases: []string{"nope"}}); err == nil {
		t.Fatal("unknown selected case accepted")
	}
	if _, err := ReplayRequests(r, cases, "", Selection{}); err == nil {
		t.Fatal("replay without a dataset digest accepted")
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
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
