package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// scriptedSystemOne is a System One resident behind the real HTTP handler: it
// reports the status facts the clef provider reports (revision, device, dtype,
// variant) and answers from a distribution that depends on the state.
type scriptedSystemOne struct {
	info    worker.Info
	runtime server.Runtime
	flip    func(i int) bool // candidate flips case i
}

func (s *scriptedSystemOne) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	var out [][]api.Result
	for _, it := range items {
		var i int
		fmt.Sscanf(it.State, "state %d", &i)
		var rs []api.Result
		for _, q := range it.Questions {
			top, other := q.Choices[0], q.Choices[1]
			if s.flip != nil && s.flip(i) {
				top, other = other, top
			}
			p := map[string]float64{top: 0.8, other: 0.2}
			for _, c := range q.Choices[2:] {
				p[c] = 0
			}
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: top, Confidence: 0.8, Probabilities: p})
		}
		out = append(out, rs)
	}
	return out, 4, nil
}
func (s *scriptedSystemOne) Ready() bool   { return true }
func (s *scriptedSystemOne) State() string { return worker.StateReady }
func (s *scriptedSystemOne) Snapshot() worker.Snapshot {
	return worker.Snapshot{State: worker.StateReady, Ready: true, PID: 7, Starts: 1, Info: s.info}
}

func forgeSource(t *testing.T) (home.Home, home.ModelManifest) {
	t.Helper()
	files := map[string]string{}
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	m := home.ModelManifest{ID: setup.ClefFlash, Provider: home.ProviderClef, Repo: "test/clef", Revision: strings.Repeat("ef", 20), Files: files}
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	os.MkdirAll(dir, 0o755)
	for _, rel := range []string{"LICENSE", "chat_template.jinja", "config.json", "generation_config.json", "joint_head.safetensors", "joint_head_config.json",
		"joint_schema_model.py", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json"} {
		os.WriteFile(filepath.Join(dir, rel), []byte("fixture "+rel), 0o644)
		files[rel], _ = setup.FileSHA256(filepath.Join(dir, rel))
	}
	if err := home.WriteJSON(filepath.Join(dir, "hachidori-model.json"), m); err != nil {
		t.Fatal(err)
	}
	old := setup.Models
	setup.Models = []home.ModelManifest{m}
	t.Cleanup(func() { setup.Models = old })
	return h, m
}

func endpointOf(t *testing.T, s *scriptedSystemOne) string {
	t.Helper()
	ts := httptest.NewServer(server.Handler(s, s.runtime))
	t.Cleanup(ts.Close)
	return ts.URL
}

func writeDataset(t *testing.T, n int) string {
	t.Helper()
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, `{"id":"c%03d","state":"state %d","questions":[{"id":"q","type":"choice","instructions":"pick","choices":["a","b","c"]}]}`+"\n", i, i)
	}
	p := filepath.Join(t.TempDir(), "d.jsonl")
	os.WriteFile(p, []byte(b.String()), 0o644)
	return p
}

// The whole operator workflow through the CLI, offline against fakes:
// optimize -> record the reference and the variant -> certify -> inspect ->
// activation gate. A variant that drifts is rejected, its evidence is kept and
// it is not activatable.
func TestForgeWorkflowThroughTheCLI(t *testing.T) {
	h, m := forgeSource(t)
	res, err := optimize.Build(context.Background(), h, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-fake"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	v := res.Variant
	dataset := writeDataset(t, 20)
	tmp := t.TempDir()

	ref := &scriptedSystemOne{
		info:    worker.Info{"provider": "clef", "model_id": m.ID, "model_revision": m.Revision, "device": "cpu", "dtype": "torch.float32"},
		runtime: server.Runtime{Home: "h", Runtime: "rt", ModelID: m.ID, Model: "test--clef/rev", Device: "cpu"}}
	cand := func(flip func(int) bool) *scriptedSystemOne {
		return &scriptedSystemOne{flip: flip,
			info: worker.Info{"provider": "clef", "model_id": m.ID, "model_revision": m.Revision, "device": "cuda", "dtype": "torch.bfloat16",
				"variant_id": v.ID, "quantized_execution": "compressed-tensors/pack-quantized W4A16 weights, bfloat16 compute"},
			runtime: server.Runtime{Home: "h", Runtime: "rt", ModelID: m.ID, Model: "test--clef/rev", Device: "cuda"}}
	}
	exec := func(args ...string) int { return run(args, nil) }

	refRun, candRun, badRun := filepath.Join(tmp, "ref.json"), filepath.Join(tmp, "cand.json"), filepath.Join(tmp, "bad.json")
	if code := exec("certify", "run", "-endpoint", endpointOf(t, ref), "-model", m.ID, "-warmup", "0", "-out", refRun, dataset); code != 0 {
		t.Fatalf("certify run (reference) exit %d", code)
	}
	if code := exec("certify", "run", "-endpoint", endpointOf(t, cand(nil)), "-model", m.ID, "-warmup", "0", "-out", candRun, dataset); code != 0 {
		t.Fatalf("certify run (candidate) exit %d", code)
	}
	if code := exec("certify", "run", "-endpoint", endpointOf(t, cand(func(i int) bool { return i < 8 })), "-model", m.ID, "-warmup", "0", "-out", badRun, dataset); code != 0 {
		t.Fatalf("certify run (drifting candidate) exit %d", code)
	}
	r, err := eval.LoadResidentRun(refRun)
	if err != nil || r.Labelled || len(r.Run.Observations) != 20 || r.Run.Identity.Provider["dtype"] != "torch.float32" {
		t.Fatalf("recorded reference: %v %+v", err, r.Run.Identity.Provider)
	}

	// The policy is an explicit, versioned file; the built-in profile needs 100 observations.
	policy := eval.DefaultPolicy()
	policy.ID, policy.MinObservations = "smoke-test/1", 10
	pb, _ := json.Marshal(policy)
	policyFile := filepath.Join(tmp, "policy.json")
	os.WriteFile(policyFile, pb, 0o644)

	// Before certification the variant is not activatable.
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err == nil {
		t.Fatal("an uncertified variant was activatable")
	}
	// A drifting variant: evidence recorded, verdict rejected, non-zero exit.
	if code := exec("certify", "evaluate", "-home", h.Root, "-variant", v.ID, "-reference", refRun, "-candidate", badRun, "-policy", policyFile); code != 1 {
		t.Fatalf("a rejected certification exited %d, want 1", code)
	}
	if st := eval.ResolveCertification(h, v); st.State != eval.StateRejected || st.Record == nil || !contains(st.Record.Failed, "flip_rate") {
		t.Fatalf("after rejection: %+v", st)
	}
	// The faithful variant is accepted.
	out := filepath.Join(tmp, "report.json")
	if code := exec("certify", "evaluate", "-home", h.Root, "-variant", v.ID, "-reference", refRun, "-candidate", candRun, "-policy", policyFile, "-out", out); code != 0 {
		t.Fatalf("an accepted certification exited %d", code)
	}
	if st := eval.ResolveCertification(h, v); st.State != eval.StateAccepted {
		t.Fatalf("after acceptance: %+v", st)
	}
	var report eval.Certification
	b, _ := os.ReadFile(out)
	if err := json.Unmarshal(b, &report); err != nil || report.Schema != eval.CertificationSchema || report.Variant.ID != v.ID ||
		report.Verdict.Status != eval.VerdictAccepted || report.Fidelity.ChoiceFlips != 0 || report.Policy.ID != "smoke-test/1" {
		t.Fatalf("report: %v %+v", err, report.Verdict)
	}
	// A certification of run files stays what it was: no producer linkage.
	if st := eval.ResolveCertification(h, v); report.Producer != nil || st.Record == nil || st.Record.Producer != nil {
		t.Fatalf("a run-file certification carries producer evidence: %+v", st.Record)
	}
	for _, args := range [][]string{{"certify", "show", "-home", h.Root, v.ID}, {"certify", "show", "-json", "-home", h.Root, v.ID},
		{"variant", "list", "-home", h.Root}, {"variant", "show", "-home", h.Root, v.ID}, {"variant", "verify", "-home", h.Root, v.ID}, {"variant", "recipes"}} {
		if code := exec(args...); code != 0 {
			t.Errorf("%v exited %d", args, code)
		}
	}
	// A variant ID or identity mismatch is refused: the reference is not a variant run.
	if code := exec("certify", "evaluate", "-home", h.Root, "-variant", v.ID, "-reference", candRun, "-candidate", candRun, "-policy", policyFile); code != 1 {
		t.Fatalf("a variant used as the reference exited %d", code)
	}
	if code := exec("certify", "evaluate", "-home", h.Root, "-variant", "clef-flash--none--000000000000", "-reference", refRun, "-candidate", candRun); code != 1 {
		t.Fatalf("an unknown variant exited %d", code)
	}
	// Corrupt a variant artifact: verification fails.
	os.WriteFile(filepath.Join(h.VariantDir(setup.ClefFlash, v.ID), "model.safetensors"), []byte("corrupt"), 0o644)
	if code := exec("variant", "verify", "-home", h.Root, v.ID); code != 1 {
		t.Fatalf("a corrupt variant verified (exit %d)", code)
	}
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// Argument errors are refusals, not partial actions.
func TestForgeCommandsRefuseBadArguments(t *testing.T) {
	h, _ := forgeSource(t)
	for _, args := range [][]string{
		{"variant"}, {"variant", "nope"}, {"certify"}, {"certify", "nope"},
		{"variant", "show", "-home", h.Root},
		{"variant", "optimize", "-home", h.Root, "-model", "laya-base"},
		{"variant", "optimize", "-home", h.Root, "-recipe", "free-form"},
		{"activate", "-home", h.Root, "-experimental"},
		{"certify", "evaluate", "-home", h.Root},
	} {
		if code := run(args, nil); code == 0 {
			t.Errorf("%v succeeded", args)
		}
	}
	if code := run([]string{"variant", "recipes"}, nil); code != 0 {
		t.Fatalf("recipes exit %d", code)
	}
}
