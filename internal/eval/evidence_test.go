package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/question"
)

// evStub is an evidence-capable endpoint: decisions plus a scripted status
// authority.
type evStub struct {
	answers  map[string]string // state -> choice for every question
	seen     []api.DecideRequest
	statuses []string // status documents served in order; the last repeats; "" fails
	nstatus  int
}

func evStatusDoc(model string, uptime int) string {
	return fmt.Sprintf(`{"schema":"hachidori.v1","runtime":{"home":"/h","runtime":"cpu-abc","model_id":"laya-base","model":%q,"device":"cpu"},`+
		`"uptime_s":%d,"worker":{"state":"ready","ready":true,"pid":42,"starts":1,`+
		`"provider":{"provider":"laya","laya_version":"0.3.21","device":"cpu","load_ms":%d,"warmup_ms":3}}}`, model, uptime, 100+uptime)
}

func (s *evStub) Status() (json.RawMessage, error) {
	if len(s.statuses) == 0 {
		return json.RawMessage(evStatusDoc("org/laya@rev1", 10)), nil
	}
	d := s.statuses[min(s.nstatus, len(s.statuses)-1)]
	s.nstatus++
	if d == "" {
		return nil, errors.New("connection refused")
	}
	return json.RawMessage(d), nil
}

func (s *evStub) Decide(r api.DecideRequest) (api.DecideResponse, error) {
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

func evCases() []Case {
	return []Case{
		{ID: "c1", State: "a", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "no"}},
		{ID: "c2", State: "b", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
	}
}

func evRun(t *testing.T, s *evStub, cases []Case, opt Options) Report {
	t.Helper()
	r, err := RunEvidence(s, cases, opt)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func asJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// Full distributions and an explicit evidence schema are recorded.
func TestEvidenceRetainsProbabilitiesAndSchema(t *testing.T) {
	cases := []Case{{ID: "c", State: "a", Expected: map[string]string{"m": "b"},
		Questions: []api.Question{{ID: "m", Type: "choice", Instructions: "i", Choices: []string{"a", "b", "c"}}}}}
	r := evRun(t, &evStub{answers: map[string]string{"a": "a"}}, cases, Options{})
	o := r.Results[0]
	if len(o.Probabilities) != 3 || o.Probabilities["a"] != 0.8 || o.Probabilities["b"] != 0.1 || o.Probabilities["c"] != 0.1 {
		t.Fatalf("probabilities lost: %+v", o.Probabilities)
	}
	if o.Correct || o.InferenceMS == nil || *o.InferenceMS != 5 {
		t.Fatalf("observation %+v", o)
	}
	var m map[string]any
	json.Unmarshal([]byte(asJSON(r)), &m)
	if m["schema"] != EvidenceSchema || EvidenceSchema != "hachidori.evidence.v1" {
		t.Fatalf("schema %v", m["schema"])
	}
}

// Model catalog -> status -> Decision Evidence: the served identity is taken
// from the canonical status document only, including the catalog model_id,
// and a change during the run is marked rather than hidden.
func TestServedIdentityFromStatus(t *testing.T) {
	answers := map[string]string{"a": "yes", "b": "no"}
	r := evRun(t, &evStub{answers: answers}, evCases(), Options{})
	if r.Served.Model() != "org/laya@rev1" || r.Served.Runtime["model_id"] != "laya-base" ||
		r.Served.Runtime["runtime"] != "cpu-abc" || r.Served.Provider["laya_version"] != "0.3.21" {
		t.Fatalf("served %+v", r.Served)
	}
	if _, ok := r.Served.Provider["load_ms"]; ok {
		t.Fatal("per-start timings are not identity")
	}
	if !r.ServedConsistent || r.ServedEnd == nil || r.ServedEnd.Digest != r.Served.Digest || len(r.Served.Digest) != 64 {
		t.Fatalf("consistent run marked inconsistent: %+v %+v", r.Served, r.ServedEnd)
	}

	// The caller has no input that supplies or overrides the identity.
	var opt []string
	ot := reflect.TypeOf(Options{})
	for i := 0; i < ot.NumField(); i++ {
		opt = append(opt, ot.Field(i).Name)
	}
	if !reflect.DeepEqual(opt, []string{"Warmup", "Passes"}) {
		t.Fatalf("Options gained fields that could carry identity: %v", opt)
	}
	// Model changes between the start and end snapshots.
	r = evRun(t, &evStub{answers: answers, statuses: []string{evStatusDoc("org/laya@rev1", 10), evStatusDoc("org/laya@rev2", 20)}}, evCases(), Options{})
	if r.ServedConsistent || r.ServedEnd.Model() != "org/laya@rev2" || r.Served.Model() != "org/laya@rev1" {
		t.Fatalf("model change not marked: %+v", r)
	}
	// Same identity, but the runtime restarted (uptime went backwards).
	if r = evRun(t, &evStub{answers: answers, statuses: []string{evStatusDoc("m", 100), evStatusDoc("m", 3)}}, evCases(), Options{}); r.ServedConsistent {
		t.Fatal("runtime restart not marked")
	}
	// End status unavailable: consistency cannot be claimed.
	if r = evRun(t, &evStub{answers: answers, statuses: []string{evStatusDoc("m", 1), ""}}, evCases(), Options{}); r.ServedConsistent || r.ServedEnd != nil || r.Errors[len(r.Errors)-1].Class != ErrClassStatus {
		t.Fatalf("unverifiable end status: %+v", r)
	}

	// No identity: nothing is sent and no evidence is produced.
	for name, doc := range map[string]string{
		"unreachable": "",
		"no model":    `{"schema":"hachidori.v1","runtime":{"runtime":"r"},"worker":{"ready":true,"provider":{"provider":"laya"}}}`,
		"not ready":   `{"schema":"hachidori.v1","runtime":{"model":"m"},"worker":{"ready":false}}`,
		"schema":      `{"schema":"other","runtime":{"model":"m"},"worker":{"ready":true,"provider":{"provider":"laya"}}}`,
	} {
		s := &evStub{statuses: []string{doc}}
		if _, err := RunEvidence(s, evCases(), Options{}); err == nil || len(s.seen) != 0 {
			t.Errorf("%s: err %v, %d requests sent", name, err, len(s.seen))
		}
	}
}

func TestServedCapacityIsRecordedInDecisionEvidence(t *testing.T) {
	status := `{"schema":"hachidori.v1","runtime":{"home":"/h","runtime":"rt","model_id":"clef-flash","model":"org/clef@rev","device":"cuda"},` +
		`"uptime_s":10,"worker":{"state":"ready","ready":true,"pid":42,"starts":1,"provider":{"provider":"clef","capacity":{"requested_max_input_tokens":700,"effective_max_input_tokens":500}}}}`
	r := evRun(t, &evStub{answers: map[string]string{"a": "yes"}, statuses: []string{status}}, evCases(), Options{})
	capacity, ok := r.Served.Provider["capacity"].(map[string]any)
	if !ok || capacity["requested_max_input_tokens"] != float64(700) || capacity["effective_max_input_tokens"] != float64(500) {
		t.Fatalf("served capacity evidence = %#v", r.Served.Provider["capacity"])
	}
}

// Question Definitions -> eval -> Decision Evidence: every observation of a
// definition-backed dataset records the exact definition identity/digest, and
// inline questions claim no definition but keep an unambiguous wire digest.
func TestEvidenceRecordsDefinitionIdentity(t *testing.T) {
	defs := loadDefs(t)
	cases, _, err := Load(refsFixture, defs)
	if err != nil {
		t.Fatal(err)
	}
	s := &evStub{answers: map[string]string{}}
	for _, c := range cases {
		s.answers[c.State] = "yes"
	}
	r := evRun(t, s, cases, Options{})
	if len(r.Results) != 6 || len(r.Errors) != 0 {
		t.Fatalf("results %d errors %v", len(r.Results), r.Errors)
	}
	for _, o := range r.Results {
		if o.QuestionDefinition == nil {
			t.Fatalf("observation %s/%s lost its definition identity", o.CaseID, o.QuestionID)
		}
		d, err := defs.Resolve(question.Ref{ID: o.QuestionID, Version: o.QuestionDefinition.Version})
		if err != nil || *o.QuestionDefinition != d.Identity() || o.QuestionDefinition.Digest != d.Digest() {
			t.Fatalf("observation %s/%s definition %+v, want %+v (%v)", o.CaseID, o.QuestionID, o.QuestionDefinition, d.Identity(), err)
		}
		if o.QuestionSHA256 != QuestionSHA256(d.Compile()) {
			t.Fatalf("wire digest %s does not match the compiled definition", o.QuestionSHA256)
		}
	}
	if len(r.Definitions) != 2 {
		t.Fatalf("report definitions %+v", r.Definitions)
	}

	inline := evRun(t, &evStub{answers: map[string]string{"a": "yes", "b": "no"}}, evCases(), Options{})
	if b := asJSON(inline.Results[0]); strings.Contains(b, "question_definition") || inline.Definitions != nil {
		t.Fatalf("inline question claims a definition: %s", b)
	}
	revised := q("x")
	revised.Instructions = "i, revised"
	if QuestionSHA256(revised) == QuestionSHA256(q("x")) {
		t.Fatal("question revisions with the same id must be distinguishable")
	}
}

// Expected labels never enter inference requests, neither during eval
// (including warmup and latency passes) nor during replay.
func TestExpectedNeverSent(t *testing.T) {
	s := &evStub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := evRun(t, s, evCases(), Options{Warmup: 2, Passes: 2})
	for _, req := range s.seen {
		if b := asJSON(req); strings.Contains(b, "expected") {
			t.Fatalf("request carries expected labels: %s", b)
		}
	}
	r.DatasetSHA256 = "d"
	items, err := ReplayRequests(r, evCases(), "d", Selection{})
	if err != nil {
		t.Fatal(err)
	}
	for _, it := range items {
		if b := asJSON(it.Request); strings.Contains(b, "expected") {
			t.Fatalf("replay request carries expected labels: %s", b)
		}
	}
}

// Serialization is deterministic and round-trips; other schemas are refused.
func TestReportSerializationDeterministic(t *testing.T) {
	r := evRun(t, &evStub{answers: map[string]string{"a": "yes", "b": "no"}}, evCases(), Options{})
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

// Request errors are never scored as incorrect.
func TestRequestErrorsDistinctFromIncorrect(t *testing.T) {
	cases := append(evCases(),
		Case{ID: "c3", State: "fail", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
		Case{ID: "c4", State: "down", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
		Case{ID: "c5", State: "partial", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "yes"}},
	)
	r := evRun(t, &evStub{answers: map[string]string{"a": "yes", "b": "no", "partial": "yes"}}, cases, Options{})
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
}

func writeDataset(t *testing.T, path string, lines string, defs *question.Set) (string, []Case) {
	t.Helper()
	if err := os.WriteFile(path, []byte(lines), 0o644); err != nil {
		t.Fatal(err)
	}
	cases, sum, err := Load(path, defs)
	if err != nil {
		t.Fatal(err)
	}
	return sum, cases
}

// Replay of a definition-backed dataset succeeds with the same definitions
// and reconstructs exactly the original requests; a changed dataset or a
// changed question digest is refused.
func TestReplayDefinitionBackedDataset(t *testing.T) {
	defs := loadDefs(t)
	src, err := os.ReadFile(refsFixture)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "d.jsonl")
	sum, cases := writeDataset(t, path, string(src), defs)
	s := &evStub{answers: map[string]string{}}
	for _, c := range cases {
		s.answers[c.State] = "yes"
	}
	r := evRun(t, s, cases, Options{})
	r.Dataset, r.DatasetSHA256 = path, sum

	// Same dataset, same definitions: reconstructed requests equal the sent ones.
	cases2, sum2, err := Load(path, defs)
	if err != nil {
		t.Fatal(err)
	}
	items, err := ReplayRequests(r, cases2, sum2, Selection{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != len(cases) {
		t.Fatalf("%d items for %d cases", len(items), len(cases))
	}
	for i, it := range items {
		if asJSON(it.Request) != asJSON(cases[i].Request()) {
			t.Fatalf("request %d not reconstructed exactly:\n%s\n%s", i, asJSON(it.Request), asJSON(cases[i].Request()))
		}
		for _, o := range it.Recorded {
			if o.QuestionDefinition == nil {
				t.Fatalf("recorded observation lost its definition identity: %+v", o)
			}
		}
	}
	rep, err := Replay(s, r, items)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.ServedMatches || len(rep.Results) != 6 {
		t.Fatalf("replay %+v", rep)
	}
	for _, res := range rep.Results {
		if res.Error != nil || !res.SameChoice || res.MaxProbDelta != 0 {
			t.Fatalf("replay result %+v", res)
		}
	}

	// Changed dataset: the dataset digest differs.
	edited := strings.Replace(string(src), "example-001", "example-001b", 1)
	sum3, cases3 := writeDataset(t, path, edited, defs)
	if _, err := ReplayRequests(r, cases3, sum3, Selection{}); err == nil || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("changed dataset accepted: %v", err)
	}

	// Same dataset bytes, but the definition behind the reference changed
	// (same id and version, different instructions): the question digest no
	// longer matches the evidence.
	writeDataset(t, path, string(src), defs)
	changed := &question.Set{}
	d, _ := defs.Resolve(question.Ref{ID: "scope_expansion", Version: 1})
	d.Instructions = "Did the agent leave the requested scope?"
	if err := changed.Add(d, "test"); err != nil {
		t.Fatal(err)
	}
	other, _ := defs.Resolve(question.Ref{ID: "ready_to_finalize", Version: 1})
	if err := changed.Add(other, "test"); err != nil {
		t.Fatal(err)
	}
	cases4, sum4, err := Load(path, changed)
	if err != nil {
		t.Fatal(err)
	}
	if sum4 != sum {
		t.Fatal("dataset bytes unexpectedly changed")
	}
	if _, err := ReplayRequests(r, cases4, sum4, Selection{}); err == nil || !strings.Contains(err.Error(), "digest") {
		t.Fatalf("changed question digest accepted: %v", err)
	}
}

// Replay selection, tampered evidence and missing digests are refused.
func TestReplayRefusals(t *testing.T) {
	path := filepath.Join(t.TempDir(), "d.jsonl")
	var lines strings.Builder
	for _, c := range evCases() {
		lines.WriteString(asJSON(c) + "\n")
	}
	sum, cases := writeDataset(t, path, lines.String(), nil)
	s := &evStub{answers: map[string]string{"a": "yes", "b": "no"}}
	r := evRun(t, s, cases, Options{})
	r.Dataset, r.DatasetSHA256 = path, sum

	items, err := ReplayRequests(r, cases, sum, Selection{Cases: []string{"c1"}, Questions: []string{"y"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].CaseID != "c1" || len(items[0].Recorded) != 1 || items[0].Recorded[0].QuestionID != "y" {
		t.Fatalf("selection %+v", items)
	}
	s.statuses = []string{evStatusDoc("org/other", 1)}
	s.answers["a"] = "no"
	if rep, _ := Replay(s, r, items); rep.ServedMatches || rep.Results[0].SameChoice {
		t.Fatalf("cross-model replay not flagged %+v", rep)
	}

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
