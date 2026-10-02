package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const fakeProbeEnv = "HACHIDORI_APP_FAKE_PROBE"

// fakeProbeWorker is a worker that speaks the protocol and reports a variant
// provenance like the real one. spec is "mode|variant-id|marker-dir|device".
func fakeProbeWorker(spec string) {
	parts := strings.SplitN(spec, "|", 4)
	mode, variant, marker, device := parts[0], parts[1], parts[2], parts[3]
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	if mode == "load_fails" {
		fmt.Fprintln(os.Stderr, "loading weights with HF_TOKEN=hf_AbCdEf0123456789secrettoken from "+marker)
		emit(map[string]any{"event": "phase", "phase": "loading"})
		emit(map[string]any{"event": "fatal", "class": worker.ClassModelLoad, "message": "no weights"})
		os.Exit(3)
	}
	for _, ph := range []string{"importing", "loading", "warming"} {
		emit(map[string]any{"event": "phase", "phase": ph})
	}
	info := map[string]any{"provider": "clef", "provider_version": "1", "model_id": setup.ClefFlash, "variant_id": variant, "device": device,
		"dtype": "bfloat16", "torch_version": "2.11.0", "python_version": "3.12.0", "load_ms": 1234.5, "warmup_ms": 67.8}
	switch mode {
	case "source":
		delete(info, "variant_id") // the source loaded, not the variant
	case "cpu_on_cuda":
		info["device"] = "cpu"
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
			if mode == "crash_on_decide" {
				fmt.Fprintln(os.Stderr, "CUDA out of memory. Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payloadpart.sigpart")
				os.Exit(7)
			}
			if mode == "decide_error" {
				emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": api.ErrInferenceFailed, "message": "boom"}})
				continue
			}
			q := req.Items[0].Questions[0]
			probs := map[string]float64{q.Choices[0]: 0.8, q.Choices[1]: 0.2}
			if mode == "bad_probabilities" {
				probs[q.Choices[1]] = 0.5
			}
			res := []api.Result{{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.8, Probabilities: probs}}
			emit(map[string]any{"id": req.ID, "ok": true, "results": [][]api.Result{res}, "inference_ms": 12.5})
		}
	}
}

const gib = 1 << 30

// forgeHome is a home with a fixture System One source under the catalog
// identity, a variant built by the fake optimizer and the cpu and cuda
// serving runtimes materialized.
func forgeHome(t *testing.T) (home.Home, home.VariantManifest) {
	t.Helper()
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{}
	m := home.ModelManifest{ID: setup.ClefFlash, Provider: home.ProviderClef, Repo: "test/clef", Revision: strings.Repeat("ef", 20), Files: files}
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	carried := []string{"LICENSE", "chat_template.jinja", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py", "processor_config.json", "tokenizer.json", "tokenizer_config.json"}
	wm := map[string]string{}
	for _, mod := range optimizetest.Modules {
		wm[mod+".weight"] = "model.safetensors"
	}
	idx, _ := json.Marshal(map[string]any{"weight_map": wm})
	for rel, body := range map[string][]byte{"config.json": []byte("c"), "generation_config.json": []byte("g"), "model.safetensors": []byte("fixture weights"), optimize.WeightMapFile: idx} {
		os.WriteFile(filepath.Join(dir, rel), body, 0o644)
	}
	for _, rel := range carried {
		os.WriteFile(filepath.Join(dir, rel), []byte("fixture "+rel), 0o644)
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		d, _ := setup.FileSHA256(filepath.Join(dir, e.Name()))
		files[e.Name()] = d
	}
	if err := home.WriteJSON(filepath.Join(dir, "hachidori-model.json"), m); err != nil {
		t.Fatal(err)
	}
	old := setup.Models
	setup.Models = []home.ModelManifest{m}
	t.Cleanup(func() { setup.Models = old })
	for _, device := range []string{"cpu", "cuda"} {
		spec, err := setup.Desired(device)
		if err != nil {
			t.Fatal(err)
		}
		rdir := h.Path("runtime", spec.ID())
		os.MkdirAll(rdir, 0o755)
		home.WriteJSON(filepath.Join(rdir, "manifest.json"), home.RuntimeManifest{Identity: spec.ID(), Spec: spec, PythonVersion: spec.Python, PythonRelPath: "env/bin/python",
			Installed: []string{"torch==" + spec.Torch, "transformers==5.17.0", "compressed-tensors==0.19.0"}})
	}
	res, err := optimize.Build(context.Background(), h, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	return h, res.Variant
}

func ram(total, avail uint64) setup.Memory {
	return setup.Memory{Total: total, TotalKnown: true, Available: avail, AvailableKnown: true}
}

// probeDeps launches the fake worker in mode and observes a healthy host.
func probeDeps(t *testing.T, mode, marker string, vstate string) ProbeDeps {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	host := &setup.Host{FreeDisk: func(string) (uint64, bool) { return 1 << 40, true }, Memory: func() setup.Memory { return ram(64*gib, 60*gib) }}
	return ProbeDeps{
		Config: func(h home.Home, device, variant string, log io.Writer) (worker.Config, server.Runtime, error) {
			if _, _, err := setup.FindVariant(h, variant); err != nil {
				return worker.Config{}, server.Runtime{}, err
			}
			return worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeProbeEnv + "=" + mode + "|" + variant + "|" + marker + "|" + device},
					StartTimeout: 20 * time.Second, RequestTimeout: 10 * time.Second},
				server.Runtime{Variant: &server.Variant{ID: variant, Certification: vstate}}, nil
		},
		Preflight: optimize.PreflightDeps{Host: host, Accelerator: func(context.Context, home.Home, string) (setup.AcceleratorFacts, error) {
			return setup.AcceleratorFacts{Torch: "2.11.0+cu128", TorchCUDA: "12.8", CUDAAvailable: true, DeviceCount: 1, DeviceName: "NVIDIA GeForce RTX 3060", VRAMTotal: 12 * gib, VRAMFree: 11 * gib}, nil
		}},
	}
}

func TestProbeLoadsThePersistedVariantDecidesOnceAndTearsDown(t *testing.T) {
	h, v := forgeHome(t)
	marker := t.TempDir()
	var phases []setup.Phase
	obs := &setup.Observer{OnPhase: func(p setup.Phase) { phases = append(phases, p) }}
	rec, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cuda"}, probeDeps(t, "ok", marker, "uncertified"), io.Discard, obs)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Result != ProbePassed || rec.Phase != ProbePhaseDone || rec.Model != setup.ClefFlash || rec.Variant != v.ID || rec.Provider != "clef" ||
		rec.Device != "cuda" || rec.Runtime == "" || rec.SourceRevision == "" || rec.VariantManifestSHA256 != v.ManifestSHA256() || rec.Scheme != "W4A16" {
		t.Fatalf("record identity %+v", rec)
	}
	if rec.Execution.VariantID != v.ID || rec.Execution.DType != "bfloat16" || rec.Execution.Quantization != "W4A16" || rec.Execution.PID == 0 {
		t.Fatalf("execution %+v", rec.Execution)
	}
	if rec.Timing.LoadMS != 1234.5 || rec.Timing.WarmupMS != 67.8 || rec.Timing.StartupMS <= 0 || rec.Timing.RequestMS <= 0 || rec.Timing.Inference != 12.5 {
		t.Fatalf("timing %+v: load/warmup are the worker's, startup and request are measured here", rec.Timing)
	}
	d := rec.Decision
	if d == nil || !d.Valid || d.Choice == "" || d.Confidence != 0.8 || len(d.Probabilities) != 2 {
		t.Fatalf("decision %+v", d)
	}
	if r := rec.Resources; r.HostRSSBytes != 1e9 || r.VRAMAllocated != 5e9 || r.VRAMTotal != 12e9 || r.HostRAMTotal != 64*gib || r.WeightFileBytes == 0 {
		t.Fatalf("resources %+v", r)
	}
	if rec.Certification != "uncertified" || !strings.Contains(rec.Effect, "not a certification") || !rec.TornDown {
		t.Fatalf("effect %q cert %q down=%v", rec.Effect, rec.Certification, rec.TornDown)
	}
	if _, err := os.Stat(filepath.Join(marker, "shutdown")); err != nil {
		t.Fatal("the probe worker was not shut down")
	}
	if len(phases) < 2 || phases[0] != setup.PhasePreflight || phases[len(phases)-1] != setup.PhaseProbing {
		t.Fatalf("phases %v", phases)
	}
	// The record is persisted and read back as the variant's latest probe.
	got, ok := LatestProbe(h, v.ID)
	if !ok || got.Result != ProbePassed || got.FinishedAt == "" || !got.TornDown {
		t.Fatalf("persisted probe %+v ok=%v", got, ok)
	}
	if pf, ok := LatestPreflight(h, setup.PreflightProbe, setup.ClefFlash, v.ID); !ok || pf.Blocked() {
		t.Fatal("the probe's preflight was not recorded")
	}
}

func TestProbeFailuresAreTypedAndNeverFallBack(t *testing.T) {
	h, v := forgeHome(t)
	for name, tc := range map[string]struct {
		mode, device, phase, want string
	}{
		"the worker cannot load":                    {"load_fails", "cpu", ProbePhaseStart, "no weights"},
		"the source loaded instead of the variant":  {"source", "cpu", ProbePhaseProvenance, "did not load the persisted variant"},
		"cpu instead of the requested cuda":         {"cpu_on_cuda", "cuda", ProbePhaseProvenance, "no fallback to another device"},
		"the worker dies in the decision":           {"crash_on_decide", "cpu", ProbePhaseDecide, "worker"},
		"the decision request errors":               {"decide_error", "cpu", ProbePhaseDecide, "boom"},
		"probabilities that are not a distribution": {"bad_probabilities", "cpu", ProbePhaseValidate, "sum to"},
	} {
		t.Run(name, func(t *testing.T) {
			marker := t.TempDir()
			rec, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: tc.device}, probeDeps(t, tc.mode, marker, "accepted"), io.Discard, nil)
			var pe *ProbeError
			if !errors.As(err, &pe) || pe.Phase != tc.phase || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v (%T), want phase %s containing %q", err, err, tc.phase, tc.want)
			}
			if rec.Result != ProbeFailed || rec.Phase != tc.phase || rec.Error == "" || rec.Decision != nil && rec.Decision.Valid {
				t.Fatalf("record %+v", rec)
			}
			if got, ok := LatestProbe(h, v.ID); !ok || got.Result != ProbeFailed || got.Device != tc.device {
				t.Fatalf("a failed probe was not recorded as failed: %+v", got)
			}
			// A started worker is always shut down; one that never started has nothing to stop.
			if tc.phase != ProbePhaseStart {
				if !rec.TornDown {
					t.Fatal("a failed probe left its worker running")
				}
			}
		})
	}
}

// Worker stderr evidence is bounded and redacted in the record itself.
func TestProbeRecordRedactsWorkerStderr(t *testing.T) {
	h, v := forgeHome(t)
	rec, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cpu"}, probeDeps(t, "load_fails", h.Root, "accepted"), io.Discard, nil)
	if err == nil || rec.WorkerClass != worker.ClassModelLoad {
		t.Fatalf("err %v class %q", err, rec.WorkerClass)
	}
	b, _ := json.Marshal(rec)
	if strings.Contains(string(b), "hf_AbCdEf0123456789secrettoken") || strings.Contains(string(b), h.Root) || len(rec.StderrTail) == 0 {
		t.Fatalf("stderr evidence %v in %s", rec.StderrTail, b)
	}
}

func TestProbePreflightBlockerStopsBeforeAnyWorker(t *testing.T) {
	h, v := forgeHome(t)
	started := false
	deps := probeDeps(t, "ok", t.TempDir(), "accepted")
	deps.Start = func(context.Context, worker.Config, func(string)) (*worker.Process, error) {
		started = true
		return nil, errors.New("must not start")
	}
	// A corrupt variant artifact: the integrity check is the probe's first step.
	os.WriteFile(filepath.Join(h.VariantDir(setup.ClefFlash, v.ID), "model.safetensors"), []byte("corrupt"), 0o644)
	rec, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cpu"}, deps, io.Discard, nil)
	pe, ok := setup.AsPreflightError(err)
	if !ok || started || rec.Phase != ProbePhasePreflight || rec.Result != ProbeFailed {
		t.Fatalf("err %v started=%v rec %+v", err, started, rec)
	}
	if f := findingOf(pe.Report, "variant.digests"); f.Status != setup.FindingBlocker {
		t.Fatalf("variant.digests = %s", f.Status)
	}
	// No CUDA device: refused before any worker, never run on cpu.
	h2, v2 := forgeHome(t)
	deps = probeDeps(t, "ok", t.TempDir(), "accepted")
	deps.Preflight.Accelerator = func(context.Context, home.Home, string) (setup.AcceleratorFacts, error) {
		return setup.AcceleratorFacts{Torch: "2.11.0+cu128"}, nil
	}
	deps.Start = func(context.Context, worker.Config, func(string)) (*worker.Process, error) {
		started = true
		return nil, errors.New("must not start")
	}
	if _, err := Probe(context.Background(), h2, ProbeParams{Variant: v2.ID, Device: "cuda"}, deps, io.Discard, nil); err == nil || started {
		t.Fatalf("a probe without a CUDA device ran: err=%v started=%v", err, started)
	}
}

func findingOf(r setup.PreflightReport, id string) setup.Finding {
	for _, f := range r.Findings {
		if f.ID == id {
			return f
		}
	}
	return setup.Finding{}
}

// A record that cannot be written never replaces the probe's own failure; for
// a pass it is reported, since the pass was lost.
func TestProbeRecordFailureNeverReplacesThePrimaryError(t *testing.T) {
	h, v := forgeHome(t)
	// state/forge is a file: no record can be written.
	if err := os.MkdirAll(h.Path("state"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(h.Path("state", "forge"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cpu"}, probeDeps(t, "decide_error", t.TempDir(), "accepted"), io.Discard, nil)
	var pe *ProbeError
	if !errors.As(err, &pe) || pe.Phase != ProbePhaseDecide || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("the primary failure was replaced: %v", err)
	}
	if _, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cpu"}, probeDeps(t, "ok", t.TempDir(), "accepted"), io.Discard, nil); err == nil || !strings.Contains(err.Error(), "could not be saved") {
		t.Fatalf("a pass that could not be recorded was reported as a plain pass: %v", err)
	}
}

// The probe is not a certification and not an activation, and the resident
// runtime it runs beside is untouched: same worker process, same status, the
// activation record and the certification records byte for byte.
func TestProbeLeavesResidentsActivationAndCertificationUntouched(t *testing.T) {
	h, v := forgeHome(t)
	active := home.Active{Runtime: "rt", ModelID: setup.ClefFlash, Model: "m", Device: "cpu"}
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), active); err != nil {
		t.Fatal(err)
	}
	activeBefore, _ := os.ReadFile(h.Path("state", "active-runtime.json"))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var opens atomicCounter
	gate := make(chan struct{})
	c := New(Config{Home: h.Root, Installed: func(string) bool { return true }, Open: realOpenCounting(t, ctx, "ok", &opens),
		Maintenance: Maintenance{Probe: func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
			<-gate
			return Probe(ctx, h, p, probeDeps(t, "ok", t.TempDir(), "uncertified"), log, obs)
		}}})
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "ready", func() bool { return c.Snapshot().State == Ready })
	before := c.Snapshot()

	if err := c.Probe(ProbeParams{Variant: v.ID, Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	// Conflicting activation follows the exclusivity of every other action.
	if err := c.Probe(ProbeParams{Variant: v.ID, Device: "cuda"}); !errors.Is(err, ErrOperationRunning) && !errors.Is(err, ErrBusy) {
		t.Fatalf("a second probe while one runs: %v", err)
	}
	close(gate)
	s := waitIdle(t, c)
	if m := s.Maintenance; m == nil || m.Kind != OpProbe || m.Failure != nil || m.Target != "variant "+v.ID ||
		strings.Join(m.Phases, ",") != "preflight,probing" || m.Cancellable {
		t.Fatalf("probe operation %+v", m)
	}
	if st := s.Status.Worker; st.PID != before.Status.Worker.PID || st.Starts != before.Status.Worker.Starts || st.State != before.Status.Worker.State || !st.Ready {
		t.Fatalf("the resident worker changed: before %+v after %+v", before.Status.Worker, st)
	}
	if s.State != Ready || s.RestartRequired || s.ResidencyChanged || opens.n() != 1 {
		t.Fatalf("state %s restart=%v residency=%v opens=%d", s.State, s.RestartRequired, s.ResidencyChanged, opens.n())
	}
	if after, _ := os.ReadFile(h.Path("state", "active-runtime.json")); !bytes.Equal(after, activeBefore) {
		t.Fatal("the activation record changed")
	}
	if _, err := os.Stat(filepath.Join(h.Root, filepath.FromSlash(home.CertificationDir(v.ID)))); err == nil {
		t.Fatal("a probe created certification records")
	}
	st := c.Forge()
	if len(st.Probes) != 1 || st.Probes[0].Result != ProbePassed || len(st.Preflights) != 1 {
		t.Fatalf("forge state %+v", st)
	}

	// A failing probe is the operation's failure, leaves a diagnostic, and the
	// resident is still untouched.
	c.cfg.Maintenance.Probe = func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
		return Probe(ctx, h, p, probeDeps(t, "crash_on_decide", t.TempDir(), "uncertified"), log, obs)
	}
	if err := c.Probe(ProbeParams{Variant: v.ID, Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	s = waitIdle(t, c)
	if f := s.Maintenance.Failure; f == nil || f.Diagnostic == "" || s.State != Ready || s.Status.Worker.PID != before.Status.Worker.PID {
		t.Fatalf("failed probe: %+v state %s", s.Maintenance.Failure, s.State)
	}
	if err := c.Close(ctx); err != nil {
		t.Fatal(err)
	}
}

type atomicCounter struct {
	mu sync.Mutex
	v  int
}

func (a *atomicCounter) inc() { a.mu.Lock(); a.v++; a.mu.Unlock() }
func (a *atomicCounter) n() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.v
}

// realOpenCounting binds the standard fake resident worker and counts opens.
func realOpenCounting(t *testing.T, ctx context.Context, mode string, opens *atomicCounter) OpenFunc {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return func(root string) (Runtime, error) {
		opens.inc()
		cfg := worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeWorkerEnv + "=" + mode},
			StartTimeout: 10 * time.Second, RequestTimeout: 5 * time.Second}
		sup := worker.NewSupervisor(cfg, worker.Policy{MaxRestarts: 0, Window: time.Minute, QueueDepth: 4})
		return &WorkerBinding{Lifecycle: worker.NewLifecycle(ctx, sup), Supervisor: sup, Info: server.Runtime{Home: root}, Started: time.Now()}, nil
	}
}

// A probe, passing or failing, leaves a live ResidentSet exactly as it found
// it: the same members, the same worker processes (no restart, no reload), the
// same default route, and the residents keep answering from the same sequence.
func TestProbeLeavesTheResidentSetUnchanged(t *testing.T) {
	// The set first: building the forge home narrows the catalog to Clef-Flash.
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	s.Start()
	bothReady(t, s)
	h, v := forgeHome(t)
	type snap struct {
		pid, starts int
		state       string
	}
	take := func() map[string]snap {
		out := map[string]snap{}
		for _, m := range s.Models() {
			w := residentState(s, m)
			out[m] = snap{w.PID, w.Starts, w.State}
		}
		return out
	}
	_, pidA, seqA := decideOn(t, s, modelA, "before")
	before, models, def := take(), s.Models(), s.Default().Model

	for _, mode := range []string{"ok", "crash_on_decide", "source"} {
		rec, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cuda"}, probeDeps(t, mode, t.TempDir(), "uncertified"), io.Discard, nil)
		if (mode == "ok") != (err == nil) || rec.Variant != v.ID {
			t.Fatalf("mode %s: err %v rec %+v", mode, err, rec)
		}
	}

	if after := take(); fmt.Sprint(after) != fmt.Sprint(before) {
		t.Fatalf("the resident set changed: before %v after %v", before, after)
	}
	if fmt.Sprint(s.Models()) != fmt.Sprint(models) || s.Default().Model != def {
		t.Fatalf("members %v default %s, before %v %s", s.Models(), s.Default().Model, models, def)
	}
	if _, pid, seq := decideOn(t, s, modelA, "after"); pid != pidA || seq != seqA+1 {
		t.Fatalf("the default resident was reloaded: pid %d->%d seq %d->%d", pidA, pid, seqA, seq)
	}
}
