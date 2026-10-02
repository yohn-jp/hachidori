package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const fakeExecEnv = "HACHIDORI_APP_FAKE_EXEC"

// fakeExecWorker reports the provenance of a Clef worker and answers every
// question of every item. spec is
// "mode|model|revision|variant|scheme|variant-dtype|device|dtype|marker-dir".
// The modes bend one provenance fact each.
func fakeExecWorker(spec string) {
	f := strings.SplitN(spec, "|", 9)
	mode, model, rev, variant, scheme, vdtype, device, dtype, marker := f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	info := map[string]any{"provider": "clef", "provider_version": "1", "model_id": model, "model_revision": rev, "device": device, "dtype": dtype,
		"execution": "source", "weights_quantized_modules": 0, "load_ms": 12.5, "warmup_ms": 3.5}
	if variant != "" {
		info["dtype"] = "torch." + vdtype
		info["variant_id"], info["execution"], info["quantization_scheme"], info["weights_quantized_modules"] = variant, "variant", scheme, 7
		info["quantized_execution"] = "compressed-tensors/pack-quantized " + scheme + " weights, " + vdtype + " compute"
	}
	switch mode {
	case "source": // the variant was requested; the source loaded
		for _, k := range []string{"variant_id", "quantization_scheme", "quantized_execution"} {
			delete(info, k)
		}
		info["execution"], info["weights_quantized_modules"] = "source", 0
	case "wrong_variant":
		info["variant_id"] = "clef-flash--w4a16--000000000000"
	case "cpu_on_cuda":
		info["device"] = "cpu"
	case "wrong_dtype":
		info["dtype"] = "float16"
	case "unquantized":
		info["weights_quantized_modules"] = 0
	}
	emit(map[string]any{"event": "ready", "info": info})
	dec := json.NewDecoder(os.Stdin)
	for {
		var req struct {
			ID    int64         `json:"id"`
			Op    string        `json:"op"`
			Items []worker.Item `json:"items"`
		}
		if dec.Decode(&req) != nil {
			return
		}
		switch req.Op {
		case "shutdown":
			_ = os.WriteFile(filepath.Join(marker, "shutdown"), nil, 0o644)
			emit(map[string]any{"id": req.ID, "ok": true})
			return
		case "stats":
			emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{"host_rss_bytes": 1e9, "memory_allocated": 5e9, "memory_reserved": 6e9, "memory_free": 5e9, "memory_total": 12e9}})
		case "decide":
			if mode == "slow" {
				_ = os.WriteFile(filepath.Join(marker, "deciding"), nil, 0o644)
				time.Sleep(1500 * time.Millisecond)
			}
			var results [][]api.Result
			for _, it := range req.Items {
				var rs []api.Result
				for _, q := range it.Questions {
					probs := map[string]float64{q.Choices[0]: 0.8, q.Choices[1]: 0.2}
					rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.8, Probabilities: probs})
				}
				results = append(results, rs)
			}
			emit(map[string]any{"id": req.ID, "ok": true, "results": results, "inference_ms": 2.5})
		}
	}
}

func execInput() ExecutionInput {
	q := api.Question{ID: "scope_expansion", Type: "choice", Instructions: "Did the agent leave scope?", Choices: []string{"yes", "no"}}
	def := []question.Identity{{ID: "scope_expansion", Version: 1, Digest: strings.Repeat("a", 64)}}
	var cases []eval.Case
	for i := range 3 {
		cases = append(cases, eval.Case{ID: fmt.Sprintf("c%d", i), State: fmt.Sprintf("state %d", i), Questions: []api.Question{q},
			Expected: map[string]string{"scope_expansion": "yes"}, Definitions: def})
	}
	return ExecutionInput{Dataset: "fixture.jsonl", DatasetSHA256: strings.Repeat("d", 64), Cases: cases, Labelled: true, Passes: 1}
}

// execDeps launches the fake worker in mode as whatever the target is.
func execDeps(t *testing.T, mode, marker string) ExecutionDeps {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return ExecutionDeps{
		Config: func(h home.Home, tg ExecutionTarget, log io.Writer) (worker.Config, server.Runtime, error) {
			model, v, err := resolveTarget(h, tg)
			if err != nil {
				return worker.Config{}, server.Runtime{}, err
			}
			rt := server.Runtime{Runtime: "rt-test", ModelID: model.ID, Model: setup.ModelDirName(model), Device: tg.Device}
			variant, scheme, vdtype := "", "", ""
			if v != nil {
				variant, scheme, vdtype = v.ID, v.Weights.Scheme, v.Weights.DType
				rt.Variant = &server.Variant{ID: v.ID, ManifestSHA256: v.ManifestSHA256()}
			}
			spec := strings.Join([]string{mode, model.ID, model.Revision, variant, scheme, vdtype, tg.Device, tg.DType, marker}, "|")
			return worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeExecEnv + "=" + spec},
				StartTimeout: 20 * time.Second, RequestTimeout: 10 * time.Second}, rt, nil
		},
		Preflight: probeDeps(t, "ok", marker, "uncertified").Preflight,
	}
}

func sourceTarget(device, dtype string) ExecutionTarget {
	return ExecutionTarget{Kind: eval.ForgeTargetSource, Model: setup.ClefFlash, Device: device, DType: dtype}
}

func variantTarget(v home.VariantManifest, device string) ExecutionTarget {
	return ExecutionTarget{Kind: eval.ForgeTargetVariant, Variant: v.ID, Device: device}
}

func runsOnDisk(h home.Home) []string {
	es, _ := os.ReadDir(eval.ForgeRunsDir(h))
	var out []string
	for _, e := range es {
		out = append(out, e.Name())
	}
	return out
}

func TestExecuteSourceRecordsABoundResidentRun(t *testing.T) {
	h, _ := forgeHome(t)
	marker := t.TempDir()
	res, err := RunExecution(context.Background(), h, ExecuteParams{Target: sourceTarget("cuda", "float32"), Input: execInput()}, execDeps(t, "ok", marker), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if res.Observations != 3 || res.ErrorCount != 0 || !strings.HasPrefix(res.EvidenceID, "fr-") {
		t.Fatalf("result %+v", res)
	}
	rec, err := eval.LoadForgeRun(h, res.EvidenceID)
	if err != nil {
		t.Fatal(err)
	}
	tg := rec.Target
	if tg.Kind != eval.ForgeTargetSource || tg.Model != setup.ClefFlash || tg.Revision != strings.Repeat("ef", 20) || tg.Provider != "clef" || tg.SourceFilesSHA256 == "" ||
		tg.Variant != "" || tg.Runtime != "rt-test" || tg.RequestedDevice != "cuda" || tg.Device != "cuda" || tg.RequestedDType != "float32" || tg.DType != "float32" {
		t.Fatalf("target %+v", tg)
	}
	// The dataset and the Question Definitions are bound to the stored run.
	if rec.DatasetSHA256 != strings.Repeat("d", 64) || rec.Run.Dataset != "fixture.jsonl" || rec.QuestionDefinitionsSHA256 == "" || len(rec.Run.Definitions) != 1 || rec.RunSHA256 == "" {
		t.Fatalf("dataset/definition binding: %+v", rec)
	}
	if r := rec.Run.Run; r.Model != setup.ClefFlash || len(r.Observations) != 3 || !r.ResidentStable || r.Identity.Provider["variant_id"] != nil {
		t.Fatalf("run %+v", r)
	}
	if _, err := os.Stat(filepath.Join(marker, "shutdown")); err != nil {
		t.Fatal("the execution worker was not shut down")
	}
	if _, err := os.Stat(h.Path("state", "active-runtime.json")); !os.IsNotExist(err) {
		t.Fatal("an execution session wrote an activation record")
	}
}

func TestExecuteVariantNeedsNoCertificationOrActivation(t *testing.T) {
	h, v := forgeHome(t)
	res, err := RunExecution(context.Background(), h, ExecuteParams{Target: variantTarget(v, "cuda"), Input: execInput()}, execDeps(t, "ok", t.TempDir()), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := eval.LoadForgeRun(h, res.EvidenceID)
	if err != nil {
		t.Fatal(err)
	}
	tg := rec.Target
	if tg.Kind != eval.ForgeTargetVariant || tg.Model != setup.ClefFlash || tg.Variant != v.ID || tg.VariantManifestSHA256 != v.ManifestSHA256() || tg.Scheme != "W4A16" ||
		tg.QuantizedModules != 7 || !strings.Contains(tg.Quantization, "W4A16") || tg.DType != v.Weights.DType || tg.Device != "cuda" {
		t.Fatalf("target %+v", tg)
	}
	if rec.Run.Run.Identity.Provider["variant_id"] != v.ID {
		t.Fatalf("the run does not carry the variant's execution identity: %v", rec.Run.Run.Identity.Provider)
	}
	if st := eval.ResolveCertification(h, v); st.State != eval.StateUncertified {
		t.Fatalf("the executed variant became %s", st.State)
	}
	if _, err := os.Stat(h.Path("state", "active-runtime.json")); !os.IsNotExist(err) {
		t.Fatal("an execution session wrote an activation record")
	}
}

func TestExecuteRefusesAnythingButTheExactTarget(t *testing.T) {
	h, v := forgeHome(t)
	for name, tc := range map[string]struct {
		mode   string
		target ExecutionTarget
		want   string
	}{
		"the variant worker reports the source": {"source", variantTarget(v, "cuda"), "did not load the persisted variant"},
		"another variant":                       {"wrong_variant", variantTarget(v, "cuda"), "the worker reports"},
		"a variant with no quantized execution": {"unquantized", variantTarget(v, "cuda"), "quantized W4A16 execution"},
		"cpu instead of cuda (variant)":         {"cpu_on_cuda", variantTarget(v, "cuda"), "no fallback to another device"},
		"cpu instead of cuda (source)":          {"cpu_on_cuda", sourceTarget("cuda", "float32"), "no fallback to another device"},
		"another dtype":                         {"wrong_dtype", sourceTarget("cuda", "float32"), `dtype "float32" was requested`},
		"the variant worker for a source":       {"ok", ExecutionTarget{Kind: eval.ForgeTargetSource, Model: setup.ClefFlash, Variant: v.ID, Device: "cuda", DType: "float32"}, "names no variant"},
		"a source with no explicit dtype":       {"ok", sourceTarget("cuda", ""), "dtype"},
		"no device":                             {"ok", sourceTarget("", "float32"), "no default and no fallback"},
		"a missing variant":                     {"ok", ExecutionTarget{Kind: eval.ForgeTargetVariant, Variant: "clef-flash--w4a16--ffffffffffff", Device: "cuda"}, "not in this home"},
		"a variant of another model":            {"ok", ExecutionTarget{Kind: eval.ForgeTargetVariant, Model: setup.DefaultModel, Variant: v.ID, Device: "cuda"}, "not " + setup.DefaultModel},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := RunExecution(context.Background(), h, ExecuteParams{Target: tc.target, Input: execInput()}, execDeps(t, tc.mode, t.TempDir()), io.Discard)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
			if got := runsOnDisk(h); len(got) != 0 {
				t.Fatalf("a refused execution left evidence %v", got)
			}
		})
	}
	t.Run("a corrupt variant", func(t *testing.T) {
		h, v := forgeHome(t)
		os.WriteFile(filepath.Join(h.VariantDir(setup.ClefFlash, v.ID), "model.safetensors"), []byte("corrupt"), 0o644)
		_, err := RunExecution(context.Background(), h, ExecuteParams{Target: variantTarget(v, "cuda"), Input: execInput()}, execDeps(t, "ok", t.TempDir()), io.Discard)
		var pe *setup.PreflightError
		if !errors.As(err, &pe) || len(runsOnDisk(h)) != 0 {
			t.Fatalf("err = %v, evidence %v", err, runsOnDisk(h))
		}
	})
}

func TestExecuteCancellationRecordsNothingAndTearsDown(t *testing.T) {
	h, _ := forgeHome(t)
	marker := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		for {
			if _, err := os.Stat(filepath.Join(marker, "deciding")); err == nil {
				cancel()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()
	_, err := RunExecution(ctx, h, ExecuteParams{Target: sourceTarget("cuda", "float32"), Input: execInput()}, execDeps(t, "slow", marker), io.Discard)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if got := runsOnDisk(h); len(got) != 0 {
		t.Fatalf("an interrupted run left evidence %v", got)
	}
}

func TestForgeRunStoreIsAtomicAndStrict(t *testing.T) {
	h, _ := forgeHome(t)
	res, err := RunExecution(context.Background(), h, ExecuteParams{Target: sourceTarget("cpu", "float32"), Input: execInput()}, execDeps(t, "ok", t.TempDir()), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(eval.ForgeRunsDir(h), res.EvidenceID+".json")
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if entries, _ := os.ReadDir(eval.ForgeRunsDir(h)); len(entries) != 1 {
		t.Fatalf("a published run leaves only its document, got %d entries", len(entries))
	}
	rec, _ := eval.LoadForgeRun(h, res.EvidenceID)
	id := rec.ID

	// A write interrupted before publication: only a temporary file exists.
	os.Remove(path)
	os.WriteFile(filepath.Join(eval.ForgeRunsDir(h), "."+id+".json.123.tmp"), good[:len(good)/2], 0o644)
	if _, err := eval.LoadForgeRun(h, id); err == nil {
		t.Fatal("an unpublished run was loadable")
	}
	// A document that is not complete, or was edited, is not evidence.
	for name, body := range map[string][]byte{
		"truncated":       good[:len(good)/2],
		"edited dataset":  []byte(strings.Replace(string(good), strings.Repeat("d", 64), strings.Repeat("e", 64), 2)),
		"edited device":   []byte(strings.Replace(string(good), `"device": "cpu"`, `"device": "cuda"`, 1)),
		"trailing data":   append(append([]byte{}, good...), []byte(`{}`)...),
		"different model": []byte(strings.Replace(string(good), `"model": "`+setup.ClefFlash+`"`, `"model": "other"`, 1)),
	} {
		os.WriteFile(path, body, 0o644)
		if _, err := eval.LoadForgeRun(h, id); err == nil {
			t.Fatalf("%s: loaded", name)
		}
	}
	os.WriteFile(path, good, 0o644)
	if _, err := eval.LoadForgeRun(h, id); err != nil {
		t.Fatal(err)
	}
	if _, err := eval.LoadForgeRun(h, "../fr-x"); err == nil {
		t.Fatal("a path-shaped ID was accepted")
	}
}

// ---- the maintenance lease ----

type leaseEnv struct {
	c        *Controller
	set      *ResidentSet
	h        home.Home
	activate []byte
	desired  []string
	ran      func() // Execute's body, run between quiesce and restore
}

// newLeaseEnv binds a running two-resident set (modelA on the accelerator,
// modelB on bDevice) to a controller whose home has an activation record, a
// desired resident selection and a routing policy, none of which a session may change.
func newLeaseEnv(t *testing.T, bDevice string, exec func(ctx context.Context) (ExecutionResult, error)) *leaseEnv {
	t.Helper()
	a, b := residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok")
	b.Info.Device = bDevice
	set := newResidentSet(t, noRestart, a, b)
	if _, err := set.SetRouting(keepPolicy()); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	h := home.Home{Root: root}
	os.MkdirAll(h.Path("state"), 0o755)
	act := []byte(`{"runtime":"rt","model_id":"` + modelA + `","device":"cuda"}`)
	os.WriteFile(h.Path("state", "active-runtime.json"), act, 0o644)
	desired := []string{modelB}
	c := New(Config{Home: root, Installed: func(string) bool { return true }, Open: func(string) (Runtime, error) { return set, nil },
		Residents: func() []string { return desired }, RestoreTimeout: 10 * time.Second,
		Maintenance: Maintenance{Execute: func(ctx context.Context, _ string, _ ExecuteParams, _ io.Writer) (ExecutionResult, error) {
			return exec(ctx)
		}}})
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = c.Close(ctx)
	})
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	bothReady(t, set)
	return &leaseEnv{c: c, set: set, h: h, activate: act, desired: desired}
}

// residentsNow is one coherent reading of every member: PID, state and running, from a
// single ResidentStatuses call.
func residentsNow(s *ResidentSet) map[string]ResidentStatus {
	out := map[string]ResidentStatus{}
	for _, r := range s.ResidentStatuses() {
		out[r.Model] = r
	}
	return out
}

func (e *leaseEnv) assertUntouched(t *testing.T) {
	t.Helper()
	got, _ := os.ReadFile(e.h.Path("state", "active-runtime.json"))
	if string(got) != string(e.activate) {
		t.Fatalf("activation record %q, was %q", got, e.activate)
	}
	if st := e.set.RoutingStatus(); st == nil || st.Policy.ID != "keep" {
		t.Fatalf("routing %+v", st)
	}
	if def := e.set.Default().Model; def != modelA {
		t.Fatalf("default resident %s", def)
	}
}

func TestLeaseWithNothingRunning(t *testing.T) {
	ran := false
	c := New(Config{Home: t.TempDir(), Maintenance: Maintenance{Execute: func(context.Context, string, ExecuteParams, io.Writer) (ExecutionResult, error) {
		ran = true
		return ExecutionResult{EvidenceID: "fr-x"}, nil
	}}})
	res, err := c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cuda", "float32")})
	if err != nil || !ran || res.EvidenceID != "fr-x" || len(res.Quiesced) != 0 || len(res.Restored) != 0 {
		t.Fatalf("res %+v err %v ran %v", res, err, ran)
	}
	if s := c.Snapshot(); s.Operation != nil || s.Maintenance == nil || s.Maintenance.Kind != OpExecute || s.Maintenance.Failure != nil {
		t.Fatalf("snapshot %+v", s)
	}
}

func TestLeaseQuiescesTheAcceleratorResidentsAndRestoresThem(t *testing.T) {
	var during map[string]ResidentStatus
	var e *leaseEnv
	e = newLeaseEnv(t, "cuda", func(context.Context) (ExecutionResult, error) {
		during = residentsNow(e.set)
		return ExecutionResult{EvidenceID: "fr-1"}, nil
	})
	before := residentsNow(e.set)
	res, err := e.c.Execute(context.Background(), ExecuteParams{Target: variantTarget(home.VariantManifest{ID: "v"}, "cuda")})
	if err != nil || res.EvidenceID != "fr-1" {
		t.Fatalf("res %+v err %v", res, err)
	}
	for _, m := range []string{modelA, modelB} {
		if during[m].Running || during[m].Status.Worker.State == worker.StateReady {
			t.Fatalf("%s was still running during the session: %+v", m, during[m])
		}
	}
	bothReady(t, e.set)
	after := residentsNow(e.set)
	for _, m := range []string{modelA, modelB} {
		if !after[m].Running || after[m].Status.Worker.State != worker.StateReady || after[m].Status.Worker.PID == before[m].Status.Worker.PID {
			t.Fatalf("%s: before %+v after %+v: want a fresh READY worker", m, before[m].Status.Worker, after[m].Status.Worker)
		}
	}
	if len(res.Quiesced) != 2 || len(res.Restored) != 2 {
		t.Fatalf("result %+v", res)
	}
	e.assertUntouched(t)
}

func TestLeaseLeavesNonConflictingResidentsRunning(t *testing.T) {
	var e *leaseEnv
	var during map[string]ResidentStatus
	e = newLeaseEnv(t, "cpu", func(context.Context) (ExecutionResult, error) {
		during = residentsNow(e.set)
		return ExecutionResult{}, nil
	})
	before := residentsNow(e.set)
	if _, err := e.c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cuda", "float32")}); err != nil {
		t.Fatal(err)
	}
	if !during[modelB].Running || during[modelB].Status.Worker.PID != before[modelB].Status.Worker.PID || during[modelA].Running {
		t.Fatalf("only the accelerator resident is quiesced: %+v", during)
	}
	// A CPU session shares nothing with the accelerator: nothing is stopped.
	e2 := newLeaseEnv(t, "cuda", func(context.Context) (ExecutionResult, error) { return ExecutionResult{}, nil })
	b2 := residentsNow(e2.set)
	res, err := e2.c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cpu", "float32")})
	if err != nil || len(res.Quiesced) != 0 || residentsNow(e2.set)[modelA].Status.Worker.PID != b2[modelA].Status.Worker.PID {
		t.Fatalf("res %+v err %v", res, err)
	}
	e.assertUntouched(t)
}

func TestLeaseRestoresAfterFailureAndCancellation(t *testing.T) {
	boom := errors.New("evaluation failed")
	for name, tc := range map[string]struct {
		cancel bool
		fail   error
	}{"execution or evaluation failure": {fail: boom}, "cancellation": {cancel: true}} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			e := newLeaseEnv(t, "cuda", func(ctx context.Context) (ExecutionResult, error) {
				if tc.cancel {
					cancel()
					<-ctx.Done()
					return ExecutionResult{}, ctx.Err()
				}
				return ExecutionResult{}, tc.fail
			})
			_, err := e.c.Execute(ctx, ExecuteParams{Target: sourceTarget("cuda", "float32")})
			want := tc.fail
			if tc.cancel {
				want = context.Canceled
			}
			var ee *ExecutionError
			if !errors.Is(err, want) || errors.As(err, &ee) {
				t.Fatalf("err = %v (%T), want the primary failure alone", err, err)
			}
			bothReady(t, e.set)
			e.assertUntouched(t)
			if s := e.c.Snapshot(); s.Operation != nil || s.Maintenance == nil || s.Maintenance.Failure == nil {
				t.Fatalf("snapshot %+v", s.Maintenance)
			}
			if err := e.c.StopResident(modelB); err != nil {
				t.Fatalf("the controller is not idle after the session: %v", err)
			}
		})
	}
}

func TestLeaseRestoreFailureIsSecondaryNeverReplacingThePrimary(t *testing.T) {
	var occupied atomic.Bool
	boom := errors.New("evaluation failed")
	for name, tc := range map[string]struct{ primary error }{"after a failure": {boom}, "after a success": {nil}} {
		t.Run(name, func(t *testing.T) {
			occupied.Store(false)
			a, b := residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok")
			b.Config.Preflight = func() error {
				if occupied.Load() {
					return errors.New("the accelerator is occupied by another process")
				}
				return nil
			}
			set := newResidentSet(t, noRestart, a, b)
			c := New(Config{Home: t.TempDir(), Installed: func(string) bool { return true }, Open: func(string) (Runtime, error) { return set, nil }, RestoreTimeout: 10 * time.Second,
				Maintenance: Maintenance{Execute: func(context.Context, string, ExecuteParams, io.Writer) (ExecutionResult, error) {
					occupied.Store(true)
					return ExecutionResult{EvidenceID: "fr-kept"}, tc.primary
				}}})
			t.Cleanup(func() { _ = c.Close(context.Background()) })
			if err := c.Start(); err != nil {
				t.Fatal(err)
			}
			bothReady(t, set)
			res, err := c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cuda", "float32")})
			var ee *ExecutionError
			if !errors.As(err, &ee) || ee.Restore == nil || !strings.Contains(ee.Restore.Error(), modelB) || !errors.Is(ee.Primary, tc.primary) {
				t.Fatalf("err = %v (%T)", err, err)
			}
			if tc.primary != nil && (!errors.Is(err, tc.primary) || !strings.Contains(err.Error(), "evaluation failed")) {
				t.Fatalf("the primary failure is hidden: %v", err)
			}
			if tc.primary == nil && res.EvidenceID != "fr-kept" {
				t.Fatalf("the recorded evidence reference was lost: %+v", res)
			}
			if got := residentsNow(set); !got[modelA].Running || got[modelA].Status.Worker.State != worker.StateReady || len(res.Restored) != 1 {
				t.Fatalf("the resident that could come back did not: %+v restored %v", got[modelA], res.Restored)
			}
		})
	}
}

func TestExecuteIsOneControllerAction(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	e := newLeaseEnv(t, "cuda", func(context.Context) (ExecutionResult, error) {
		close(entered)
		<-release
		return ExecutionResult{}, nil
	})
	done := make(chan error, 1)
	go func() {
		_, err := e.c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cuda", "float32")})
		done <- err
	}()
	<-entered
	if err := e.c.Restart(); !errors.Is(err, ErrBusy) {
		t.Fatalf("Restart during a session = %v, want ErrBusy", err)
	}
	if _, err := e.c.Execute(context.Background(), ExecuteParams{}); !errors.Is(err, ErrBusy) {
		t.Fatalf("a second session = %v, want ErrBusy", err)
	}
	if s := e.c.Snapshot(); s.Operation == nil || s.Operation.Kind != OpExecute || s.Operation.Phase != PhaseExecRun {
		t.Fatalf("operation %+v", s.Operation)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	bothReady(t, e.set)
}

// A session that changes the activation record is a failure, whatever else it did.
func TestLeaseReportsAnActivationChange(t *testing.T) {
	var e *leaseEnv
	e = newLeaseEnv(t, "cuda", func(context.Context) (ExecutionResult, error) {
		return ExecutionResult{EvidenceID: "fr-1"}, os.WriteFile(e.h.Path("state", "active-runtime.json"), []byte(`{"variant":"x"}`), 0o644)
	})
	_, err := e.c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cuda", "float32")})
	var ee *ExecutionError
	if !errors.As(err, &ee) || ee.Primary != nil || !strings.Contains(err.Error(), "activation record changed") {
		t.Fatalf("err = %v", err)
	}
}
