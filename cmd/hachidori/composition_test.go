package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/question"
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

func copyDir(t *testing.T, src, dst string) {
	t.Helper()
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// Question Definitions -> eval -> Decision Evidence -> replay, and model
// catalog -> status -> Decision Evidence, through the real commands against
// a real status/decide handler. The host sees only the public inference and
// status contract.
func TestDefinitionEvalEvidenceReplay(t *testing.T) {
	const defsDir = "../../testdata/questions"
	hostHome, caller := t.TempDir(), t.TempDir()
	var mu sync.Mutex
	var seen []string
	h := server.Handler(hostWorker{}, server.Runtime{Home: hostHome, Runtime: "cpu-abc", ModelID: "laya-base", Model: "org/laya@rev", Device: "cpu"})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, r.Method+" "+r.URL.Path+" "+string(b))
		mu.Unlock()
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		h.ServeHTTP(w, r)
	}))
	defer ts.Close()

	src, err := os.ReadFile("../../testdata/eval/contract-example-refs.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	dataset, out := filepath.Join(caller, "d.jsonl"), filepath.Join(caller, "report.json")
	os.WriteFile(dataset, src, 0o644)
	if err := cmdEval("eval", []string{"-endpoint", ts.URL, "-questions", defsDir, "-out", out, dataset}); err != nil {
		t.Fatal(err)
	}
	r, err := eval.LoadReport(out)
	if err != nil {
		t.Fatal(err)
	}

	// Served identity is the status authority's, including the catalog ID.
	if r.Served.Model() != "org/laya@rev" || r.Served.Runtime["model_id"] != "laya-base" || !r.ServedConsistent {
		t.Fatalf("served %+v consistent=%v", r.Served, r.ServedConsistent)
	}
	// Evidence records the exact definition identity and digest.
	defs, err := question.Load(defsDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Definitions) != 2 || len(r.Results) != 6 {
		t.Fatalf("definitions %+v results %d", r.Definitions, len(r.Results))
	}
	for _, o := range r.Results {
		if o.QuestionDefinition == nil {
			t.Fatalf("observation without definition identity: %+v", o)
		}
		d, err := defs.Resolve(question.Ref{ID: o.QuestionID, Version: o.QuestionDefinition.Version})
		if err != nil || *o.QuestionDefinition != d.Identity() || !strings.HasPrefix(o.QuestionDefinition.Digest, "sha256:") ||
			o.QuestionSHA256 != eval.QuestionSHA256(d.Compile()) {
			t.Fatalf("observation %+v does not carry the exact definition %+v (%v)", o, d.Identity(), err)
		}
	}

	// Replay of the same dataset with the same definitions succeeds.
	replayOut := filepath.Join(caller, "replay.json")
	if err := cmdReplay([]string{"-endpoint", ts.URL, "-questions", defsDir, "-out", replayOut, out}); err != nil {
		t.Fatalf("replay with the same dataset and definitions: %v", err)
	}
	if _, err := os.Stat(replayOut); err != nil {
		t.Fatal(err)
	}

	// The host received nothing beyond decide/status/health, and no labels,
	// references, digests or evidence.
	for _, s := range seen {
		allowed := strings.HasPrefix(s, "GET /health ") || strings.HasPrefix(s, "GET /v1/status ") || strings.HasPrefix(s, "POST /v1/decide ")
		if !allowed || strings.Contains(s, "expected") || strings.Contains(s, "question_refs") || strings.Contains(s, "sha256:") ||
			strings.Contains(s, "probabilities") || strings.Contains(s, eval.EvidenceSchema) {
			t.Fatalf("host received more than the inference/status contract: %s", s)
		}
	}
	if ents, _ := os.ReadDir(hostHome); len(ents) != 0 {
		t.Fatalf("evidence persisted on the host: %v", ents)
	}

	// A changed question digest (same id and version, edited instructions)
	// is refused although the dataset bytes are unchanged.
	edited := t.TempDir()
	copyDir(t, defsDir, edited)
	p := filepath.Join(edited, "scope_expansion.json")
	b, _ := os.ReadFile(p)
	changed := strings.Replace(string(b), "Did the agent modify files", "Did the agent touch files", 1)
	if changed == string(b) {
		t.Fatal("fixture edit did not apply")
	}
	os.WriteFile(p, []byte(changed), 0o644)
	if err := cmdReplay([]string{"-endpoint", ts.URL, "-questions", edited, "-print", out}); err == nil || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("changed question digest replayed: %v", err)
	}

	// A changed dataset is refused.
	os.WriteFile(dataset, []byte(strings.Replace(string(src), "example-001", "example-001b", 1)), 0o644)
	if err := cmdReplay([]string{"-endpoint", ts.URL, "-questions", defsDir, "-print", out}); err == nil || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("changed dataset replayed: %v", err)
	}
}

// The composed CLI keeps every command dispatchable and documented: desktop,
// question, replay and model-selecting setup coexist with the earlier ones.
func TestCommandDispatchAndHelpCoexist(t *testing.T) {
	cmds := commands()
	documented := map[string]bool{}
	for _, m := range regexp.MustCompile(`(?m)^  ([a-z]+)  `).FindAllStringSubmatch(usage, -1) {
		documented[m[1]] = true
	}
	for _, name := range []string{"setup", "serve", "dashboard", "desktop", "doctor", "status", "decide", "eval", "benchmark", "question", "replay"} {
		if cmds[name] == nil {
			t.Errorf("command %q is not dispatched", name)
		}
		if !documented[name] {
			t.Errorf("command %q is not in the usage text", name)
		}
	}
	for name := range documented {
		if cmds[name] == nil {
			t.Errorf("usage documents %q but it is not dispatched", name)
		}
	}
	for name := range cmds {
		if !documented[name] {
			t.Errorf("command %q is dispatched but undocumented", name)
		}
	}
	if !strings.Contains(usage, "--model") {
		t.Error("usage does not mention setup model selection")
	}
}
