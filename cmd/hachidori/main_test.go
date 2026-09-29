package main

import (
	"io"
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

type hostWorker struct{}

func (hostWorker) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	var out [][]api.Result
	for _, it := range items {
		var rs []api.Result
		for _, q := range it.Questions {
			p := map[string]float64{}
			for _, c := range q.Choices {
				p[c] = 0.3 / float64(len(q.Choices)-1)
			}
			p[q.Choices[0]] = 0.7
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.7, Probabilities: p})
		}
		out = append(out, rs)
	}
	return out, 1, nil
}
func (hostWorker) Ready() bool   { return true }
func (hostWorker) State() string { return worker.StateReady }
func (hostWorker) Snapshot() worker.Snapshot {
	return worker.Snapshot{State: worker.StateReady, Ready: true, Starts: 1,
		Info: worker.Info{"provider": "laya", "laya_version": "0.3.21", "device": "cpu"}}
}

// Evidence is produced and kept only on the caller side: the host sees only
// the public inference/status contract without labels or evidence, and its
// home stays untouched (verification 7 and 10, end to end).
func TestEvalEvidenceStaysCallerSide(t *testing.T) {
	hostHome, caller := t.TempDir(), t.TempDir()
	var mu sync.Mutex
	var seen []string
	h := server.Handler(hostWorker{}, server.Runtime{Home: hostHome, Runtime: "cpu-abc", Model: "org/laya@rev", Device: "cpu"})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, r.Method+" "+r.URL.Path+" "+string(b))
		mu.Unlock()
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		h.ServeHTTP(w, r)
	}))
	defer ts.Close()

	src, err := os.ReadFile("../../testdata/eval/contract-example.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	dataset, out := filepath.Join(caller, "d.jsonl"), filepath.Join(caller, "report.json")
	os.WriteFile(dataset, src, 0o644)
	if err := cmdEval("eval", []string{"-endpoint", ts.URL, "-out", out, dataset}); err != nil {
		t.Fatal(err)
	}
	r, err := eval.LoadReport(out)
	if err != nil {
		t.Fatal(err)
	}
	if r.Served.Model() != "org/laya@rev" || !r.ServedConsistent || r.Dataset != dataset || len(r.DatasetSHA256) != 64 {
		t.Fatalf("report %+v", r)
	}
	if len(r.Results) == 0 || len(r.Results[0].Probabilities) != 2 {
		t.Fatalf("results %+v", r.Results)
	}

	replayOut := filepath.Join(caller, "replay.json")
	if err := cmdReplay([]string{"-endpoint", ts.URL, "-out", replayOut, "-case", "example-001", out}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(replayOut); err != nil {
		t.Fatal(err)
	}

	for _, s := range seen {
		allowed := strings.HasPrefix(s, "GET /health ") || strings.HasPrefix(s, "GET /v1/status ") || strings.HasPrefix(s, "POST /v1/decide ")
		if !allowed || strings.Contains(s, "expected") || strings.Contains(s, "dataset") || strings.Contains(s, "probabilities") ||
			strings.Contains(s, eval.EvidenceSchema) {
			t.Fatalf("host received more than the inference/status contract: %s", s)
		}
	}
	if ents, _ := os.ReadDir(hostHome); len(ents) != 0 {
		t.Fatalf("evidence persisted on the host: %v", ents)
	}

	// Replay refuses changed and missing source material.
	os.WriteFile(dataset, []byte(strings.Replace(string(src), "example-001", "example-001b", 1)), 0o644)
	if err := cmdReplay([]string{"-endpoint", ts.URL, "-print", out}); err == nil || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("changed dataset replayed: %v", err)
	}
	os.Remove(dataset)
	if err := cmdReplay([]string{"-endpoint", ts.URL, "-print", out}); err == nil {
		t.Fatal("missing dataset replayed")
	}
}
