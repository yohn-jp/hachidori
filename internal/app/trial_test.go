package app

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/trial"
	"github.com/yohn-jp/hachidori/internal/tuning"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const fakeTrialEnv = "HACHIDORI_APP_FAKE_TRIAL"

// fakeTrialWorker is a resident tuning trial session: it keeps the
// representation of every module in memory, speaks the trial protocol and
// answers decisions. It proves what the controller-side orchestration asks of a
// worker and how it reacts to the worker's answers; it says nothing about torch
// or the accelerator. spec is "mode|model|revision|device|marker-dir".
func fakeTrialWorker(spec string) {
	f := strings.SplitN(spec, "|", 5)
	mode, model, rev, device, marker := f[0], f[1], f[2], f[3], f[4]
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	note := func(name, text string) {
		if fh, err := os.OpenFile(filepath.Join(marker, name), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
			_, _ = fh.WriteString(text + "\n")
			_ = fh.Close()
		}
	}
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	info := map[string]any{"provider": "clef", "provider_version": "1", "model_id": model, "model_revision": rev, "device": device, "dtype": "torch.bfloat16",
		"execution": "trial", "trial_session": true, "trial_transform": trial.TransformImplementation, "weights_quantized_modules": 0, "load_ms": 1.0, "warmup_ms": 1.0}
	if mode == "plain" {
		info["execution"], info["trial_session"] = "source", false
	}
	emit(map[string]any{"event": "ready", "info": info})

	reps := map[string]string{}
	groups := map[string][]string{}
	dec := json.NewDecoder(os.Stdin)
	for {
		var req struct {
			ID       int64               `json:"id"`
			Op       string              `json:"op"`
			Groups   map[string][]string `json:"groups"`
			Identity struct {
				ID      string `json:"id"`
				Members []struct {
					Module string  `json:"module"`
					Shape  []int64 `json:"shape"`
				} `json:"members"`
			} `json:"identity"`
			Replacements []struct {
				Group          string   `json:"group"`
				Modules        []string `json:"modules"`
				Transformation struct {
					Representation string `json:"representation"`
				} `json:"transformation"`
			} `json:"replacements"`
			Items []worker.Item `json:"items"`
		}
		if dec.Decode(&req) != nil {
			return
		}
		ok := func(result map[string]any) { emit(map[string]any{"id": req.ID, "ok": true, "result": result}) }
		switch req.Op {
		case "shutdown":
			note("shutdown", "x")
			emit(map[string]any{"id": req.ID, "ok": true})
			return
		case "stats":
			emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{"host_rss_bytes": 1e9, "memory_allocated": 5e9, "memory_reserved": 6e9, "memory_free": 5e9, "memory_total": 12e9}})
		case "trial_open":
			groups = req.Groups
			var mods []map[string]any
			n := 0
			for _, ms := range groups {
				for _, m := range ms {
					reps[m] = "dense"
					mods = append(mods, map[string]any{"module": m, "shape": []int{256, 256}, "dtype": "bfloat16"})
					n++
				}
			}
			note("ops", "open")
			ok(map[string]any{"device": device, "dtype": "bfloat16", "backend": "fake-ct 1", "modules": mods, "canonical_bytes": n * 256 * 256 * 2, "transform": trial.TransformImplementation})
		case "trial_transform":
			var total int64
			for _, m := range req.Identity.Members {
				out, in := m.Shape[0], m.Shape[1]
				total += out*((in*4+31)/32)*4 + out*(in/128)*2 + 16
			}
			note("ops", "transform "+req.Identity.ID[:8])
			ok(map[string]any{"bytes": total, "digest": "digest-" + req.Identity.ID[:16]})
		case "trial_release":
			note("ops", "release")
			ok(map[string]any{})
		case "trial_validate":
			ok(map[string]any{"ok": true})
		case "trial_apply", "trial_reconstruct":
			var names []string
			for _, r := range req.Replacements {
				names = append(names, r.Group)
				for _, m := range r.Modules {
					reps[m] = r.Transformation.Representation
				}
			}
			note("ops", strings.TrimPrefix(req.Op, "trial_")+" "+strings.Join(names, ","))
			ok(map[string]any{"bytes_to_gpu": 1000, "bytes_released": 2000, "modules": len(names)})
		case "trial_state":
			state := map[string]string{}
			for g, ms := range groups {
				set := map[string]bool{}
				for _, m := range ms {
					set[reps[m]] = true
				}
				switch {
				case len(set) == 1:
					for r := range set {
						state[g] = r
					}
				case len(set) == 0:
					state[g] = "dense"
				default:
					state[g] = "mixed"
				}
			}
			ok(map[string]any{"groups": state, "gpu_allocated": 123, "host_rss": 456})
		case "decide":
			note("decides", "x")
			if mode == "slow" {
				_ = os.WriteFile(filepath.Join(marker, "deciding"), nil, 0o644)
				time.Sleep(1500 * time.Millisecond)
			}
			if mode == "decide_error" {
				emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": api.ErrInferenceFailed, "message": "boom"}})
				continue
			}
			var results [][]api.Result
			for _, it := range req.Items {
				var rs []api.Result
				for _, q := range it.Questions {
					rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.8, Probabilities: map[string]float64{q.Choices[0]: 0.8, q.Choices[1]: 0.2}})
				}
				results = append(results, rs)
			}
			emit(map[string]any{"id": req.ID, "ok": true, "results": results, "inference_ms": 2.5})
		}
	}
}

// trialWorld is a fixture home with a saved set of layer-wise profiles.
type trialWorld struct {
	h        home.Home
	model    home.ModelManifest
	analysis tuning.Analysis
	dataset  string
	marker   string
}

func newTrialWorld(t *testing.T) trialWorld {
	t.Helper()
	h := forgeSource(t)
	m, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	a, err := tuning.Analyze(m, tuning.DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text", Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes: []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj"},
		CarriedFiles: []string{"joint_head.safetensors"}})
	if err != nil {
		t.Fatal(err)
	}
	dataset := filepath.Join(t.TempDir(), "cases.jsonl")
	lines := []string{
		`{"id":"c1","state":"one","questions":[{"id":"q","type":"choice","instructions":"Is it?","choices":["yes","no"]}],"expected":{"q":"yes"}}`,
		`{"id":"c2","state":"two","questions":[{"id":"q","type":"choice","instructions":"Is it?","choices":["yes","no"]}],"expected":{"q":"no"}}`,
	}
	if err := os.WriteFile(dataset, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return trialWorld{h: h, model: m, analysis: a, dataset: dataset, marker: t.TempDir()}
}

// profile saves the profile that keeps the named groups at source precision.
func (w trialWorld) profile(t *testing.T, keep ...string) string {
	t.Helper()
	p, err := tuning.NewDefaultProfile(w.analysis, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	if len(keep) > 0 {
		if p, err = tuning.SetPolicy(p, w.analysis, keep, home.PolicySourcePrecision); err != nil {
			t.Fatal(err)
		}
	}
	if err := tuning.SaveProfile(w.h, p, w.analysis); err != nil {
		t.Fatal(err)
	}
	return p.ID()
}

func (w trialWorld) params(profiles ...string) TrialParams {
	return TrialParams{Source: setup.ClefFlash, Device: "cuda", Profiles: profiles, Dataset: w.dataset, BudgetBytes: 1 << 30, Passes: 1}
}

func (w trialWorld) deps(t *testing.T, mode string) TrialDeps {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return TrialDeps{Config: func(h home.Home, model, device string, log io.Writer) (worker.Config, server.Runtime, error) {
		m, err := setup.LookupModel(model)
		if err != nil {
			return worker.Config{}, server.Runtime{}, err
		}
		spec := strings.Join([]string{mode, m.ID, m.Revision, device, w.marker}, "|")
		rt := server.Runtime{Runtime: "fake", ModelID: m.ID, Model: setup.ModelDirName(m), Device: device}
		return worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeTrialEnv + "=" + spec},
			StartTimeout: 20 * time.Second, RequestTimeout: 10 * time.Second}, rt, nil
	}}
}

func (w trialWorld) ops(t *testing.T) []string {
	t.Helper()
	b, _ := os.ReadFile(filepath.Join(w.marker, "ops"))
	return strings.Fields(strings.ReplaceAll(strings.TrimSpace(string(b)), "\n", ";"))
}

func (w trialWorld) opLines(t *testing.T) []string {
	t.Helper()
	b, _ := os.ReadFile(filepath.Join(w.marker, "ops"))
	if len(b) == 0 {
		return nil
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

func TestRunTrialsAppliesDifferentialPlansMeasuresAndRecordsCandidates(t *testing.T) {
	w := newTrialWorld(t)
	// A quantizes the full attention of block 3; B also quantizes the MLP of
	// block 0: the second trial differs from the first in exactly one group.
	a := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	b := w.profile(t, "block.00.linear-attn")
	run, err := RunTrials(context.Background(), w.h, w.params(a, b), w.deps(t, "ok"), io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(run.Outcomes) != 2 || run.Outcomes[0].Result == nil || run.Outcomes[1].Result == nil {
		t.Fatalf("outcomes %+v", run.Outcomes)
	}
	first, second := run.Outcomes[0].Result, run.Outcomes[1].Result
	if len(first.Assembly.Changed) != 1 || len(second.Assembly.Changed) != 1 || second.Assembly.Changed[0].Group != "block.00.mlp" || second.Assembly.Reused != len(second.Plan.Groups)-1 {
		t.Fatalf("assemblies: first %+v, second %+v", first.Assembly, second.Assembly)
	}
	// The worker saw one open, two transforms, and a bounded delta per trial.
	lines := w.opLines(t)
	var transforms, applies, opens int
	for _, l := range lines {
		switch {
		case strings.HasPrefix(l, "transform"):
			transforms++
		case strings.HasPrefix(l, "apply"):
			applies++
		case l == "open":
			opens++
		}
	}
	if opens != 1 || transforms != 2 || applies != 2 {
		t.Fatalf("worker operations %v", lines)
	}
	if !strings.HasPrefix(lines[len(lines)-1], "apply block.00.mlp") {
		t.Fatalf("the second trial replaced more than its delta: %v", lines)
	}
	// Measurements are trial measurements of the ephemeral model, bound to the dataset.
	for _, o := range run.Outcomes {
		m := o.Result.Measurement
		if m.Mode != trial.EvalModeTrial || m.Served.Execution != trial.ExecutionTrial || m.Served.VariantID != "" || m.DatasetSHA256 == "" || m.Accuracy == nil {
			t.Fatalf("measurement %+v", m)
		}
		if o.CandidateID == "" || o.EvidenceID == "" {
			t.Fatalf("outcome not recorded: %+v", o)
		}
	}
	// Candidates and evidence are on disk; no Variant and no certification were made.
	list, err := trial.List(w.h, setup.ClefFlash)
	if err != nil || len(list) != 2 {
		t.Fatalf("candidates %d (%v)", len(list), err)
	}
	for _, c := range list {
		st, _ := trial.StatusOf(w.h, c.ID)
		if st.Stage() != "measured" || st.Measurements != 1 {
			t.Fatalf("status %+v", st)
		}
	}
	if entries, _ := os.ReadDir(w.h.VariantsDir(setup.ClefFlash)); len(entries) != 0 {
		t.Fatalf("trials produced %d variant artifacts", len(entries))
	}
	if _, err := os.Stat(w.h.Path("state", "certifications")); !os.IsNotExist(err) {
		t.Fatal("a trial wrote certification state")
	}
	if _, err := os.Stat(w.h.Path("state", "active-runtime.json")); !os.IsNotExist(err) {
		t.Fatal("a trial wrote an activation record")
	}
	if _, err := os.Stat(filepath.Join(w.marker, "shutdown")); err != nil {
		t.Fatal("the trial worker was not shut down")
	}
	if run.Session.Trials != 2 || run.Session.Cache.Misses != 2 || run.Session.Cache.TransformedBytes == 0 || run.StartMS < 0 || run.SessionMS <= 0 {
		t.Fatalf("session %+v start %v total %v", run.Session, run.StartMS, run.SessionMS)
	}
}

func TestRunTrialsReusesCachedComponentsAcrossTrialsOfOneSession(t *testing.T) {
	w := newTrialWorld(t)
	a := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	b := w.profile(t, "block.00.linear-attn")
	run, err := RunTrials(context.Background(), w.h, w.params(b, a, b), w.deps(t, "ok"), io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	// B, then A (block 0's MLP back to source), then B again: the MLP of block 0
	// is not transformed a second time.
	third := run.Outcomes[2].Result
	if third == nil || third.Assembly.CacheHits != 1 || third.Assembly.CacheMisses != 0 || len(third.Assembly.Transformed) != 0 {
		t.Fatalf("the repeated profile re-transformed: %+v", third)
	}
	transforms := 0
	for _, l := range w.opLines(t) {
		if strings.HasPrefix(l, "transform") {
			transforms++
		}
	}
	if transforms != 2 {
		t.Fatalf("worker transformed %d components for three trials", transforms)
	}
}

func TestRunTrialsEvaluationFailureRestoresTheModelAndRecordsNothing(t *testing.T) {
	w := newTrialWorld(t)
	a := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	run, err := RunTrials(context.Background(), w.h, w.params(a), w.deps(t, "decide_error"), io.Discard, nil)
	if err != nil {
		t.Fatalf("a restored failure is an outcome, not a session failure: %v", err)
	}
	o := run.Outcomes[0]
	if o.Result != nil || o.Failure == "" || !o.Restored || o.CandidateID != "" {
		t.Fatalf("outcome %+v", o)
	}
	lines := w.opLines(t)
	if len(lines) < 4 || !strings.HasPrefix(lines[len(lines)-1], "apply ") {
		t.Fatalf("the model was not put back: %v", lines)
	}
	if list, _ := trial.List(w.h, ""); len(list) != 0 {
		t.Fatal("a failed trial recorded a candidate")
	}
}

func TestRunTrialsCancellationRestoresTheModelAndRecordsNothing(t *testing.T) {
	w := newTrialWorld(t)
	a := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for i := 0; i < 400; i++ {
			if _, err := os.Stat(filepath.Join(w.marker, "deciding")); err == nil {
				cancel()
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
		cancel()
	}()
	run, err := RunTrials(ctx, w.h, w.params(a), w.deps(t, "slow"), io.Discard, nil)
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if len(run.Outcomes) != 1 || run.Outcomes[0].Result != nil || !run.Outcomes[0].Restored {
		t.Fatalf("outcomes %+v", run.Outcomes)
	}
	if list, _ := trial.List(w.h, ""); len(list) != 0 {
		t.Fatal("a cancelled trial recorded a candidate")
	}
	if _, err := os.Stat(filepath.Join(w.marker, "shutdown")); err != nil {
		t.Fatal("the worker was not shut down after cancellation")
	}
}

func TestRunTrialsRefusesAWorkerThatIsNotATrialSession(t *testing.T) {
	w := newTrialWorld(t)
	a := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	_, err := RunTrials(context.Background(), w.h, w.params(a), w.deps(t, "plain"), io.Discard, nil)
	if err == nil || !strings.Contains(err.Error(), "trial provenance") {
		t.Fatalf("err = %v", err)
	}
	if lines := w.opLines(t); len(lines) != 0 {
		t.Fatalf("a trial was driven on a worker that is not a trial session: %v", lines)
	}
}

func TestRunTrialsRefusesBadInputsBeforeAnythingStarts(t *testing.T) {
	w := newTrialWorld(t)
	good := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	for name, mutate := range map[string]func(*TrialParams){
		"no budget":       func(p *TrialParams) { p.BudgetBytes = 0 },
		"no device":       func(p *TrialParams) { p.Device = "" },
		"unknown device":  func(p *TrialParams) { p.Device = "tpu" },
		"no profiles":     func(p *TrialParams) { p.Profiles = nil },
		"unknown profile": func(p *TrialParams) { p.Profiles = []string{strings.Repeat("a", 64)} },
		"no dataset":      func(p *TrialParams) { p.Dataset = "" },
		"no source":       func(p *TrialParams) { p.Source = "" },
		"another model":   func(p *TrialParams) { p.Source = "laya" },
	} {
		p := w.params(good)
		mutate(&p)
		started := false
		deps := w.deps(t, "ok")
		inner := deps.Config
		deps.Config = func(h home.Home, model, device string, log io.Writer) (worker.Config, server.Runtime, error) {
			started = true
			return inner(h, model, device, log)
		}
		if _, err := RunTrials(context.Background(), w.h, p, deps, io.Discard, nil); err == nil {
			t.Errorf("%s: accepted", name)
		}
		if started {
			t.Errorf("%s: the worker was configured before the inputs were refused", name)
		}
	}
}

func TestResolveTrialPlansRejectsALegacyProfileWithoutAPlan(t *testing.T) {
	w := newTrialWorld(t)
	legacyAnalysis, err := tuning.AnalyzeLegacy(w.model, tuning.DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text", Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes: []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj"},
		CarriedFiles: []string{"joint_head.safetensors"}})
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := tuning.NewLegacyProfile(legacyAnalysis, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	if err := tuning.SaveProfile(w.h, legacy, legacyAnalysis); err != nil {
		t.Fatal(err)
	}
	_, _, _, err = resolveTrialPlans(w.h, w.params(legacy.ID()))
	if err == nil || !strings.Contains(err.Error(), "legacy coarse profile") {
		t.Fatalf("err = %v", err)
	}
}

func TestBuildingATunedProfileLinksItsTrialCandidatesToTheVariant(t *testing.T) {
	w := newTrialWorld(t)
	prof := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	run, err := RunTrials(context.Background(), w.h, w.params(prof), w.deps(t, "ok"), io.Discard, nil)
	if err != nil || run.Outcomes[0].CandidateID == "" {
		t.Fatalf("trial: %v %+v", err, run.Outcomes)
	}
	cand := run.Outcomes[0].CandidateID
	// The Forge build of that profile, through the composed operation's build
	// step, publishes a variant and links the candidate to it.
	req, err := tuning.BuildRequest(w.h, w.model, prof)
	if err != nil {
		t.Fatal(err)
	}
	res, err := buildWithFake(t, w.h, req)
	if err != nil {
		t.Fatal(err)
	}
	linkTrialCandidates(w.h, res.Variant, io.Discard)
	st, err := trial.StatusOf(w.h, cand)
	if err != nil || st.Stage() != "materialized" || len(st.Variants) != 1 || st.Variants[0] != res.Variant.ID {
		t.Fatalf("status %+v (%v)", st, err)
	}
	// A variant of another plan links nothing.
	other := w.profile(t)
	oreq, _ := tuning.BuildRequest(w.h, w.model, other)
	ores, err := buildWithFake(t, w.h, oreq)
	if err != nil {
		t.Fatal(err)
	}
	var log strings.Builder
	linkTrialCandidates(w.h, ores.Variant, &log)
	if log.Len() != 0 {
		t.Fatalf("a variant of another plan was linked: %s", log.String())
	}
	if st, _ := trial.StatusOf(w.h, cand); len(st.Variants) != 1 {
		t.Fatalf("status %+v", st)
	}
}

func TestControllerRunTrialsOwnsTheAcceleratorAndRunsTheSession(t *testing.T) {
	w := newTrialWorld(t)
	prof := w.profile(t, "block.00.linear-attn", "block.00.mlp")
	var got TrialParams
	c := New(Config{Maintenance: Maintenance{
		Trials: func(ctx context.Context, root string, p TrialParams, log io.Writer, obs *setup.Observer) (TrialRun, error) {
			got = p
			return TrialRun{}, nil
		}}})
	if err := c.SetHome(w.h.Root); err != nil {
		t.Fatal(err)
	}
	if err := c.RunTrials(w.params(prof)); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, c)
	snap := c.Snapshot()
	if got.Source != setup.ClefFlash || got.Device != "cuda" || len(got.Profiles) != 1 || snap.Maintenance == nil || snap.Maintenance.Kind != OpTuningTrial || snap.Maintenance.Failure != nil {
		t.Fatalf("params %+v snapshot %+v", got, snap.Maintenance)
	}
	// An input that cannot work is refused by the operation, before any session.
	bad := w.params(prof)
	bad.BudgetBytes = 0
	if err := c.RunTrials(bad); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, c)
	if last := c.Snapshot().Maintenance; last == nil || last.Failure == nil || !strings.Contains(last.Failure.Message, "budget") {
		t.Fatalf("last %+v", last)
	}
}

// buildWithFake publishes a variant of req with the fake optimizer.
func buildWithFake(t *testing.T, h home.Home, req optimize.Request) (optimize.Result, error) {
	t.Helper()
	return optimize.Build(context.Background(), h, req, optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
}
