package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/question"
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
	cases, sum, err := Load(good, nil)
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
		if _, _, err := Load(p, nil); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestContractExampleFixture(t *testing.T) {
	if _, _, err := Load("../../testdata/eval/contract-example.jsonl", nil); err != nil {
		t.Fatal(err)
	}
}

const (
	inlineFixture = "../../testdata/eval/contract-example.jsonl"
	refsFixture   = "../../testdata/eval/contract-example-refs.jsonl"
	defsDir       = "../../testdata/questions"
)

func loadDefs(t *testing.T) *question.Set {
	t.Helper()
	s, err := question.Load(defsDir)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// Referenced definitions resolve locally to exactly the questions of the
// equivalent inline dataset.
func TestLoadResolvesDefinitions(t *testing.T) {
	defs := loadDefs(t)
	inline, _, err := Load(inlineFixture, nil)
	if err != nil {
		t.Fatal(err)
	}
	refs, _, err := Load(refsFixture, defs)
	if err != nil {
		t.Fatal(err)
	}
	if len(inline) != len(refs) {
		t.Fatalf("%d vs %d cases", len(inline), len(refs))
	}
	for i := range refs {
		if !reflect.DeepEqual(refs[i].Request(), inline[i].Request()) {
			t.Fatalf("case %s: resolved request %+v differs from inline %+v", refs[i].ID, refs[i].Request(), inline[i].Request())
		}
		if refs[i].QuestionRefs != nil || len(refs[i].Definitions) != len(refs[i].Questions) {
			t.Fatalf("case %s not fully resolved: %+v", refs[i].ID, refs[i])
		}
		for j, id := range refs[i].Definitions {
			d, _ := defs.Resolve(question.Ref{ID: id.ID, Version: id.Version})
			if id != d.Identity() || refs[i].Questions[j].ID != id.ID {
				t.Fatalf("identity %+v does not match definition %+v", id, d.Identity())
			}
		}
	}
	if inline[0].Definitions != nil {
		t.Fatal("inline cases must carry no definition identity")
	}
}

// Every reference problem fails in Load, i.e. before any request exists.
func TestLoadReferenceFailures(t *testing.T) {
	defs := loadDefs(t)
	scope, _ := defs.Resolve(question.Ref{ID: "scope_expansion", Version: 1})
	ready, _ := defs.Resolve(question.Ref{ID: "ready_to_finalize", Version: 1})
	v2 := scope
	v2.Version, v2.Instructions = 2, "Did the agent leave the requested scope?"
	if err := defs.Add(v2, "test"); err != nil {
		t.Fatal(err)
	}
	inlineScope, _ := json.Marshal(scope.Compile())
	dir := t.TempDir()
	for name, tc := range map[string]struct {
		defs  *question.Set
		lines []string
	}{
		"no definitions loaded":   {nil, []string{`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":1}],"expected":{"scope_expansion":"yes"}}`}},
		"unknown id":              {defs, []string{`{"id":"a","state":"s","question_refs":[{"id":"nope","version":1}],"expected":{"nope":"yes"}}`}},
		"unknown version":         {defs, []string{`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":7}],"expected":{"scope_expansion":"yes"}}`}},
		"digest mismatch":         {defs, []string{`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":1,"digest":"` + ready.Digest() + `"}],"expected":{"scope_expansion":"yes"}}`}},
		"duplicate ref":           {defs, []string{`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":1},{"id":"scope_expansion","version":2}],"expected":{"scope_expansion":"yes"}}`}},
		"mixed inline and refs":   {defs, []string{`{"id":"a","state":"s","questions":[` + string(inlineScope) + `],"question_refs":[{"id":"ready_to_finalize","version":1}],"expected":{"scope_expansion":"yes","ready_to_finalize":"no"}}`}},
		"label not in definition": {defs, []string{`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":1}],"expected":{"scope_expansion":"maybe"}}`}},
		"versions differ across cases": {defs, []string{
			`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":1}],"expected":{"scope_expansion":"yes"}}`,
			`{"id":"b","state":"s","question_refs":[{"id":"scope_expansion","version":2}],"expected":{"scope_expansion":"yes"}}`}},
		"inline and definition share id": {defs, []string{
			`{"id":"a","state":"s","questions":[` + string(inlineScope) + `],"expected":{"scope_expansion":"yes"}}`,
			`{"id":"b","state":"s","question_refs":[{"id":"scope_expansion","version":1}],"expected":{"scope_expansion":"yes"}}`}},
	} {
		p := filepath.Join(dir, "d.jsonl")
		os.WriteFile(p, []byte(strings.Join(tc.lines, "\n")+"\n"), 0o644)
		if _, _, err := Load(p, tc.defs); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// A matching digest pin resolves.
	p := filepath.Join(dir, "pinned.jsonl")
	os.WriteFile(p, []byte(`{"id":"a","state":"s","question_refs":[{"id":"scope_expansion","version":1,"digest":"`+scope.Digest()+`"}],"expected":{"scope_expansion":"yes"}}`+"\n"), 0o644)
	if _, _, err := Load(p, defs); err != nil {
		t.Fatal(err)
	}
}

// fakeEndpoint is an in-process v1 endpoint recording raw request bodies.
type fakeEndpoint struct {
	mu     sync.Mutex
	bodies [][]byte
}

func (f *fakeEndpoint) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	f.mu.Unlock()
	var req api.DecideRequest
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if r.URL.Path != "/v1/decide" || dec.Decode(&req) != nil || req.Validate() != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(api.ErrorBody{Schema: api.SchemaV1, Error: api.ErrorInfo{Class: api.ErrRequestInvalid, Message: "bad"}})
		return
	}
	resp := api.DecideResponse{Schema: api.SchemaV1, Timing: &api.Timing{InferenceMS: 1, TotalMS: 1}}
	for _, q := range req.Questions {
		resp.Results = append(resp.Results, api.Result{ID: q.ID, Type: q.Type, Choice: q.Choices[1], Confidence: 0.7,
			Probabilities: map[string]float64{q.Choices[0]: 0.3, q.Choices[1]: 0.7}})
	}
	json.NewEncoder(w).Encode(resp)
}

// Over real HTTP, a definition-based dataset sends ordinary v1 requests only:
// no expected labels, references, versions, digests or paths.
func TestRunDefinitionsOverHTTP(t *testing.T) {
	defs := loadDefs(t)
	cases, _, err := Load(refsFixture, defs)
	if err != nil {
		t.Fatal(err)
	}
	fe := &fakeEndpoint{}
	srv := httptest.NewServer(fe)
	defer srv.Close()
	r := Run(client.New(srv.URL), cases, Options{})
	if len(r.Errors) != 0 || r.Observations != 6 {
		t.Fatalf("errors %v observations %d", r.Errors, r.Observations)
	}
	if len(fe.bodies) != len(cases) {
		t.Fatalf("%d requests for %d cases", len(fe.bodies), len(cases))
	}
	for i, body := range fe.bodies {
		var got map[string]any
		json.Unmarshal(body, &got)
		for k := range got {
			if k != "schema" && k != "state" && k != "questions" {
				t.Fatalf("request carries non-v1 field %q: %s", k, body)
			}
		}
		for _, bad := range []string{"expected", "question_refs", "version", "digest", "sha256:", "testdata", ".json", "hachidori.question"} {
			if bytes.Contains(body, []byte(bad)) {
				t.Fatalf("request leaks %q: %s", bad, body)
			}
		}
		want, _ := json.Marshal(cases[i].Request())
		if !bytes.Equal(body, append(want, '\n')) && !bytes.Equal(body, want) {
			t.Fatalf("request body\n got %s\nwant %s", body, want)
		}
	}
	// The report identifies the exact definitions used.
	var want []question.Identity
	for _, d := range defs.Definitions() {
		want = append(want, d.Identity())
	}
	if !reflect.DeepEqual(r.Definitions, want) {
		t.Fatalf("report definitions %+v, want %+v", r.Definitions, want)
	}
	b, _ := json.Marshal(r)
	if !bytes.Contains(b, []byte(`"question_definitions":[{"id":"ready_to_finalize","version":1,"digest":"sha256:`)) {
		t.Fatalf("report JSON lacks definition identity: %s", b)
	}
	var sum bytes.Buffer
	r.DatasetSHA256 = strings.Repeat("0", 64)
	Summary(&sum, r)
	if !strings.Contains(sum.String(), "definition     scope_expansion@1 ("+want[1].Digest+")") {
		t.Fatalf("summary lacks definition identity:\n%s", sum.String())
	}
}

// Inline-question datasets keep working unchanged and report no definitions.
func TestRunInlineOverHTTP(t *testing.T) {
	cases, _, err := Load(inlineFixture, nil)
	if err != nil {
		t.Fatal(err)
	}
	fe := &fakeEndpoint{}
	srv := httptest.NewServer(fe)
	defer srv.Close()
	r := Run(client.New(srv.URL), cases, Options{})
	if len(r.Errors) != 0 || r.Observations != 6 || r.Definitions != nil {
		t.Fatalf("report %+v", r)
	}
	for _, body := range fe.bodies {
		if bytes.Contains(body, []byte("expected")) {
			t.Fatalf("expected label leaked: %s", body)
		}
	}
	b, _ := json.Marshal(r)
	if bytes.Contains(b, []byte("question_definitions")) {
		t.Fatal("inline report must not claim definitions")
	}
}
