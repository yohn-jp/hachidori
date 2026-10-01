package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// fakeSet is a two-resident runtime behind the real HTTP handler: the real
// direct-selection, served provenance and status document shapes, with no
// worker process and no model.
type fakeSet struct {
	mu      sync.Mutex
	decides map[string]int
	running map[string]bool
}

func (s *fakeSet) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	return s.DecideOn("", items)
}

func (s *fakeSet) DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if model == "" {
		model = "laya-base"
	}
	if !s.running[model] {
		return nil, 0, &worker.RequestError{Class: api.ErrRequestInvalid, Message: "model " + model + " is not resident"}
	}
	s.decides[model]++
	var out [][]api.Result
	for _, it := range items {
		var rs []api.Result
		for _, q := range it.Questions {
			p := map[string]float64{}
			for _, c := range q.Choices {
				p[c] = 0.2 / float64(len(q.Choices)-1)
			}
			p[q.Choices[0]] = 0.8
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.8, Probabilities: p})
		}
		out = append(out, rs)
	}
	return out, 3, nil
}

func (s *fakeSet) Identity(model string) (api.Served, bool) {
	switch model {
	case "laya-base":
		return api.Served{Model: model, Provider: "laya"}, true
	case "opendecider-nano":
		return api.Served{Model: model, Provider: "opendecider"}, true
	}
	return api.Served{}, false
}

func (s *fakeSet) Ready() bool   { return true }
func (s *fakeSet) State() string { return worker.StateReady }
func (s *fakeSet) Snapshot() worker.Snapshot {
	return s.resident("laya-base", "laya", 100).Status.Worker
}

func (s *fakeSet) resident(id, provider string, pid int) server.ResidentStatus {
	return server.ResidentStatus{Model: id, Provider: provider, Default: id == "laya-base", Running: s.running[id],
		Status: server.Status{Schema: api.SchemaV1, UptimeS: 60,
			Runtime: server.Runtime{Home: "h", Runtime: "rt", ModelID: id, Model: "org/" + id + "@rev", Device: "cuda"},
			Worker: worker.Snapshot{State: worker.StateReady, Ready: true, PID: pid, Starts: 1,
				Info:        worker.Info{"provider": provider, "device": "cuda", "load_ms": 1234.5, "warmup_ms": 99.5},
				Accelerator: map[string]any{"memory_allocated": float64(1 << 30), "memory_reserved": float64(2 << 30), "memory_free": float64(4 << 30), "memory_total": float64(12 << 30)}}}}
}

func (s *fakeSet) ResidentStatuses() []server.ResidentStatus {
	return []server.ResidentStatus{s.resident("laya-base", "laya", 100), s.resident("opendecider-nano", "opendecider", 200)}
}

func residentEndpoint(t *testing.T, set *fakeSet) (*httptest.Server, *[]string) {
	t.Helper()
	var mu sync.Mutex
	var seen []string
	h := server.Handler(set, server.Runtime{Home: "h", Runtime: "rt", ModelID: "laya-base", Model: "org/laya-base@rev", Device: "cuda"})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Method+" "+r.URL.Path)
		mu.Unlock()
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(ts.Close)
	return ts, &seen
}

// `benchmark -models a,b` evaluates both residents on the same dataset and
// Question Definitions through the real HTTP handler, using only decide and
// status: nothing is started, restarted or reloaded.
func TestBenchmarkModelsComparesResidentsWithoutLifecycleCalls(t *testing.T) {
	set := &fakeSet{decides: map[string]int{}, running: map[string]bool{"laya-base": true, "opendecider-nano": true}}
	ts, seen := residentEndpoint(t, set)
	dataset := filepath.Join(t.TempDir(), "d.jsonl")
	src, err := os.ReadFile("../../testdata/eval/contract-example-refs.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(dataset, src, 0o644)
	out := filepath.Join(t.TempDir(), "cmp.json")
	err = cmdEval("benchmark", []string{"-endpoint", ts.URL, "-questions", "../../testdata/questions", "-models", "laya-base, opendecider-nano",
		"-warmup", "1", "-passes", "2", "-thresholds", "0.5,0.9", "-length-edges", "64,128", "-high-confidence", "0.75",
		"-family", "scope_expansion=scope", "-family", "ready_to_finalize=readiness", "-out", out, dataset})
	if err != nil {
		t.Fatal(err)
	}
	rep, err := eval.LoadComparison(out)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Alignment.Status != eval.AlignAligned || !rep.ResidentsStable || len(rep.Runs) != 2 || rep.Dataset != dataset || rep.Endpoint != ts.URL {
		t.Fatalf("%+v", rep.Alignment)
	}
	if rep.Declared.HighConfidence != 0.75 || len(rep.Declared.Thresholds) != 2 || len(rep.Declared.LengthEdges) != 2 || len(rep.Definitions) != 2 {
		t.Fatalf("%+v", rep.Declared)
	}
	for i, id := range []string{"laya-base", "opendecider-nano"} {
		r := rep.Runs[i]
		if r.Model != id || r.Identity.Runtime["model_id"] != id || r.Identity.Runtime["model"] != "org/"+id+"@rev" || r.ErrorCount != 0 ||
			r.Startup.LoadMS == nil || *r.Startup.LoadMS != 1234.5 || *r.Startup.WarmupMS != 99.5 || !r.Memory.Available ||
			*r.Memory.ResidentRsrv != 2<<30 || len(r.PerFamily) != 2 || r.PerFamily[0].Key != "readiness" {
			t.Fatalf("%s: %+v", id, r)
		}
		if set.decides[id] != 1+2*3 {
			t.Fatalf("%s answered %d requests", id, set.decides[id])
		}
		for _, o := range r.Observations {
			if o.QuestionDefinition == nil || o.ServedModel != id {
				t.Fatalf("%+v", o)
			}
		}
	}
	if rep.Runs[0].SentSHA256 != rep.Runs[1].SentSHA256 {
		t.Fatal("the two residents did not receive identical requests")
	}
	for _, line := range *seen {
		if line != "POST /v1/decide" && line != "GET /v1/status" {
			t.Fatalf("comparison used %q; only decide and status are allowed", line)
		}
	}
}

func TestBenchmarkModelsRefusesStoppedResidentBeforeAnyDecision(t *testing.T) {
	set := &fakeSet{decides: map[string]int{}, running: map[string]bool{"laya-base": true}}
	ts, seen := residentEndpoint(t, set)
	dataset := filepath.Join(t.TempDir(), "d.jsonl")
	src, _ := os.ReadFile("../../testdata/eval/contract-example-refs.jsonl")
	os.WriteFile(dataset, src, 0o644)
	err := cmdEval("eval", []string{"-endpoint", ts.URL, "-questions", "../../testdata/questions", "-models", "laya-base,opendecider-nano", dataset})
	if err == nil || !strings.Contains(err.Error(), "opendecider-nano") {
		t.Fatalf("%v", err)
	}
	for _, l := range *seen {
		if strings.Contains(l, "decide") {
			t.Fatalf("a request was sent although a resident is not running: %v", *seen)
		}
	}
}

func TestResidentFlagsAreValidatedBeforeAnyRequest(t *testing.T) {
	set := &fakeSet{decides: map[string]int{}, running: map[string]bool{"laya-base": true, "opendecider-nano": true}}
	ts, seen := residentEndpoint(t, set)
	dataset := filepath.Join(t.TempDir(), "d.jsonl")
	src, _ := os.ReadFile("../../testdata/eval/contract-example-refs.jsonl")
	os.WriteFile(dataset, src, 0o644)
	for _, extra := range [][]string{
		{"-thresholds", "0.9,x"}, {"-thresholds", "0.9,0.5"}, {"-length-edges", "10,a"}, {"-length-edges", "10,10"},
		{"-family", "nope"}, {"-family", "ghost=f"}, {"-family", "scope_expansion=a", "-family", "scope_expansion=b"},
		{"-high-confidence", "2"},
	} {
		args := append([]string{"-endpoint", ts.URL, "-questions", "../../testdata/questions", "-models", "laya-base,opendecider-nano"}, extra...)
		if err := cmdEval("eval", append(args, dataset)); err == nil {
			t.Fatalf("%v accepted", extra)
		}
	}
	for _, l := range *seen {
		if strings.Contains(l, "decide") {
			t.Fatalf("a request was sent for invalid flags: %v", *seen)
		}
	}
}
