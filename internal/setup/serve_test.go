package setup_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Proofs 13 and 14: after materialization the serving path needs neither uv
// nor package resolution, and still launches the absolute private interpreter
// with the existing isolation and worker digest verification.
func TestServeIndependentOfUV(t *testing.T) {
	h := setup.MaterializeFake(t, "cuda")
	for _, p := range []string{h.Path("tools", "uv"), h.Path("cache", "uv"), h.Path("packages")} {
		if err := os.RemoveAll(p); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", t.TempDir())

	cfg, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatalf("serve needs something removed with uv: %v", err)
	}
	id, _ := setup.RuntimeName("cuda")
	wantPython := filepath.Join(h.Root, "runtime", id, "env", "bin", "python")
	if cfg.Python != wantPython || !filepath.IsAbs(cfg.Python) || rt.Runtime != id || rt.Device != "cuda" {
		t.Fatalf("worker python %s (runtime %+v), want %s", cfg.Python, rt, wantPython)
	}
	if _, err := os.Stat(cfg.Python); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cfg.Args[:4], []string{"-I", "-B", "-X", "utf8"}) || cfg.Args[4] != filepath.Join(h.Root, "runtime", id, "worker", "hachidori_worker.py") {
		t.Fatalf("args %v", cfg.Args)
	}
	for _, kv := range cfg.Env {
		k, v, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(k, "UV_") || strings.Contains(v, filepath.Join(h.Root, "tools", "uv")+string(filepath.Separator)+"bin") {
			t.Errorf("worker env references uv: %s", kv)
		}
		if k == "PATH" && v != filepath.Dir(wantPython) {
			t.Errorf("worker PATH=%s", v)
		}
	}
	for _, want := range []string{"PYTHONNOUSERSITE=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"} {
		if !slices.Contains(cfg.Env, want) {
			t.Errorf("worker env lacks %s", want)
		}
	}

	// Worker digest verification is still enforced.
	os.WriteFile(filepath.Join(h.Root, "runtime", id, "worker", "hachidori_worker.py"), []byte("import os"), 0o644)
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("tampered worker accepted")
	}
}

// Issue #117: a runtime materialized by an older build is self-consistent (its
// identity, manifest and worker digest all agree) but carries a worker script
// that does not know the arguments this build passes. Launching it ended in
// the interpreter's argument parser ("exit status 2") with the cause only in
// the stderr tail. It is refused before any process starts, with the cause and
// the recovery, and the inventory says the active pair cannot start.
func TestOlderWorkerRuntimeIsRefusedBeforeSpawn(t *testing.T) {
	h, name := setup.MaterializeFakeOlderWorker(t, "cpu")
	cfg, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatalf("the binding must still come up so the operator can recover: %v", err)
	}
	if rt.Runtime != name || rt.ModelID != setup.DefaultModel {
		t.Fatalf("runtime %+v", rt)
	}
	err = cfg.Preflight()
	if !errors.Is(err, setup.ErrWorkerContract) {
		t.Fatalf("Preflight = %v, want ErrWorkerContract", err)
	}
	// The supervisor refuses the launch before any process exists.
	cfg.Python = filepath.Join(t.TempDir(), "must-not-be-started")
	sup := worker.NewSupervisor(cfg, worker.DefaultPolicy)
	sup.Run(context.Background())
	snap := sup.Snapshot()
	if snap.State != worker.StateFailed || snap.Phase != worker.PhasePreflight || snap.LastFailure == nil ||
		snap.LastFailure.Class != worker.ClassPreflight || !strings.Contains(snap.LastFailure.Message, "older Hachidori") || snap.PID != 0 {
		t.Fatalf("snapshot %+v failure %+v", snap, snap.LastFailure)
	}
	for _, want := range []string{name, "older Hachidori", "Materialize", "hachidori setup --device cpu"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error lacks %q: %v", want, err)
		}
	}
	var report strings.Builder
	if doctor.Run(h.Root, &report) || !strings.Contains(report.String(), "FAIL runtime") || !strings.Contains(report.String(), "older Hachidori") {
		t.Fatalf("doctor accepted a runtime this build cannot start:\n%s", report.String())
	}
	inv := setup.Inspect(h, false)
	if inv.Active == nil || inv.Active.Runtime != name || !strings.Contains(inv.ActiveProblem, "older Hachidori") {
		t.Fatalf("inventory active %+v problem %q", inv.Active, inv.ActiveProblem)
	}
	for _, r := range inv.Runtimes {
		if r.Active {
			t.Errorf("an older runtime is not a catalog identity, yet %s is marked active", r.ID)
		}
	}

	// Materializing and activating the current runtime recovers it: the model
	// is reused and the same call that was refused now succeeds.
	if err := setup.Materialize(h, "cpu", "", io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	if changed, err := setup.Activate(h, "cpu", "", io.Discard, nil); err != nil || !changed {
		t.Fatalf("Activate = %v, %v", changed, err)
	}
	cfg, _, err = server.WorkerConfig(h, io.Discard)
	if err != nil || cfg.Preflight() != nil {
		t.Fatalf("current runtime refused: %v / %v", err, cfg.Preflight())
	}
	if inv := setup.Inspect(h, false); inv.ActiveProblem != "" {
		t.Fatalf("problem %q after recovery", inv.ActiveProblem)
	}
}

// Issue #117 through the application: with a runtime from an older build
// active, Start still binds the runtime (so Settings is reachable) and reports a
// preflight failure naming the cause, without spawning anything. Materialize and
// Activate of the current runtime, then Restart, rebind the application to the
// activation; the failed binding is not restarted again.
func TestOlderWorkerRuntimeRecoversThroughTheApplication(t *testing.T) {
	h, old := setup.MaterializeFakeOlderWorker(t, "cpu")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctl := app.New(app.Config{Home: h.Root, Open: app.WorkerRuntime(ctx, io.Discard, worker.DefaultPolicy, nil)})
	defer func() {
		cctx, ccancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer ccancel()
		ctl.Close(cctx)
	}()
	waitSnap := func(what string, ok func(app.Snapshot) bool) app.Snapshot {
		t.Helper()
		for i := 0; i < 1000; i++ {
			if s := ctl.Snapshot(); ok(s) {
				return s
			}
			time.Sleep(5 * time.Millisecond)
		}
		t.Fatalf("%s: last snapshot %+v", what, ctl.Snapshot())
		return app.Snapshot{}
	}

	if err := ctl.Start(); err != nil {
		t.Fatalf("Start must bind the runtime even when it cannot be started: %v", err)
	}
	s := waitSnap("preflight failure", func(s app.Snapshot) bool { return s.State == app.Failed })
	if f := s.Failure; f == nil || f.Source != app.SourceWorker || f.Class != worker.ClassPreflight || f.Phase != worker.PhasePreflight ||
		!strings.Contains(f.Message, "older Hachidori") || s.Status == nil || s.Status.Worker.PID != 0 || s.Status.Runtime.Runtime != old {
		t.Fatalf("failure %+v status %+v", s.Failure, s.Status)
	}
	if s.Status.Worker.Starts != 1 || s.Status.Worker.LastFailure.Stderr != nil {
		t.Fatalf("a process was started: %+v", s.Status.Worker)
	}

	// Recover through the same application the Settings manager drives.
	for _, act := range []func() error{
		func() error { return ctl.Materialize(app.SetupParams{Device: "cpu"}) },
		func() error { return ctl.Activate(app.SetupParams{Device: "cpu"}) },
	} {
		if err := act(); err != nil {
			t.Fatal(err)
		}
		s = waitSnap("maintenance", func(s app.Snapshot) bool { return s.Operation == nil })
		if s.Maintenance == nil || s.Maintenance.Failure != nil {
			t.Fatalf("maintenance %+v", s.Maintenance)
		}
	}
	current, _ := setup.RuntimeName("cpu")
	if inv, err := ctl.Inventory(false); err != nil || inv.Active == nil || inv.Active.Runtime != current || inv.ActiveProblem != "" {
		t.Fatalf("inventory active %+v problem %q (%v)", inv.Active, inv.ActiveProblem, err)
	}
	if err := ctl.Restart(); err != nil {
		t.Fatal(err)
	}
	s = waitSnap("rebound to the activation", func(s app.Snapshot) bool {
		return s.Status != nil && s.Status.Runtime.Runtime == current && s.Status.Worker.Starts >= 1 && s.State != app.Starting
	})
	if s.Failure != nil && s.Failure.Class == worker.ClassPreflight {
		t.Fatalf("restart ran the old runtime again: %+v", s.Failure)
	}
}

// Issue #9 proofs 9 and 10: serve resolves only the already materialized and
// activated catalog model, never downloads, and exposes its exact identity in
// status; doctor verifies that same identity.
func TestServeResolvesActivatedModel(t *testing.T) {
	h, requests := setup.MaterializeFakeCounting(t, "cpu", setup.TunedModel)
	var a home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	model, err := setup.LookupModel(setup.TunedModel)
	if err != nil {
		t.Fatal(err)
	}
	cfg, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	modelDir := filepath.Join(h.Root, "models", filepath.FromSlash(setup.ModelDirName(model)))
	if rt.ModelID != setup.TunedModel || rt.Model != setup.ModelDirName(model) || a.ModelID != setup.TunedModel {
		t.Fatalf("runtime %+v, active %+v", rt, a)
	}
	if !slices.Equal(cfg.Args[5:], []string{"--model-dir", modelDir, "--device", "cpu", "--manifest", filepath.Join(modelDir, "hachidori-model.json"), "--provider", "opendecider"}) {
		t.Fatalf("worker args %v", cfg.Args)
	}
	b, _ := json.Marshal(server.StatusBody(idle{}, rt, time.Now()))
	var st struct {
		Runtime map[string]any `json:"runtime"`
	}
	json.Unmarshal(b, &st)
	if st.Runtime["model_id"] != setup.TunedModel || st.Runtime["model"] != setup.ModelDirName(model) {
		t.Fatalf("status runtime %v", st.Runtime)
	}

	var out strings.Builder
	doctor.Run(h.Root, &out)
	if want := "PASS model              [hachidori] " + setup.TunedModel + " (" + model.Repo + "@" + model.Revision + "), 1 files verified"; !strings.Contains(out.String(), want) {
		t.Fatalf("doctor does not verify the selected model:\n%s", out.String())
	}

	activePath := h.Path("state", "active-runtime.json")
	bad := map[string]home.Active{
		"unknown id":       {Runtime: a.Runtime, ModelID: "laya-unknown", Model: a.Model, Device: a.Device},
		"id/dir mismatch":  {Runtime: a.Runtime, ModelID: setup.DefaultModel, Model: a.Model, Device: a.Device},
		"not materialized": {Runtime: a.Runtime, ModelID: setup.DefaultModel, Model: "test--model/" + strings.Repeat("ab", 20), Device: a.Device},
	}
	for name, act := range bad {
		home.WriteJSON(activePath, act)
		if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
			t.Errorf("%s: serve accepted %+v", name, act)
		}
		out.Reset()
		if doctor.Run(h.Root, &out) || !strings.Contains(out.String(), "FAIL") {
			t.Errorf("%s: doctor passed:\n%s", name, out.String())
		}
	}
	home.WriteJSON(activePath, a)

	// A materialized manifest that is not the catalog entry is refused.
	mf := filepath.Join(modelDir, "hachidori-model.json")
	orig, _ := os.ReadFile(mf)
	tampered := model
	tampered.Revision = strings.Repeat("0", 40)
	home.WriteJSON(mf, tampered)
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Error("serve accepted a manifest that is not the catalog entry")
	}
	os.WriteFile(mf, orig, 0o644)

	// A tampered model file fails doctor's model check.
	os.WriteFile(filepath.Join(modelDir, "config.json"), []byte("tampered"), 0o644)
	out.Reset()
	if doctor.Run(h.Root, &out) || !strings.Contains(out.String(), "FAIL model") || !strings.Contains(out.String(), "class: "+doctor.ModelUnavailable) {
		t.Fatalf("doctor accepted a tampered model:\n%s", out.String())
	}

	// Serving from a home whose activated model was removed fails instead of
	// fetching it.
	os.RemoveAll(modelDir)
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("serve started without the materialized model")
	}
	if n := requests(); n != 0 {
		t.Fatalf("serve/doctor made %d artifact requests", n)
	}
}

// Issue #107: serve/worker startup, the desktop's installed decision and
// doctor apply one runtime-validity rule. For every manifest state they all
// accept or all reject, and a state is never READY-capable while doctor calls
// the runtime invalid.
func TestActivationAndDoctorAgreeOnRuntimeIdentity(t *testing.T) {
	type mutate func(t *testing.T, h home.Home, id string)
	edit := func(f func(m map[string]any, id string) string) mutate {
		return func(t *testing.T, h home.Home, id string) {
			var m map[string]any
			if err := home.ReadJSON(h.Path("runtime", id, "manifest.json"), &m); err != nil {
				t.Fatal(err)
			}
			name := f(m, id)
			if err := home.WriteJSON(h.Path("runtime", id, "manifest.json"), m); err != nil {
				t.Fatal(err)
			}
			if name != id {
				if err := os.Rename(h.Path("runtime", id), h.Path("runtime", name)); err != nil {
					t.Fatal(err)
				}
				var a home.Active
				home.ReadJSON(h.Path("state", "active-runtime.json"), &a)
				a.Runtime = name
				home.WriteJSON(h.Path("state", "active-runtime.json"), a)
			}
		}
	}
	for name, tc := range map[string]struct {
		change mutate
		valid  bool
	}{
		"valid current identity": {change: func(*testing.T, home.Home, string) {}, valid: true},
		"identity key missing, spec missing": {change: edit(func(m map[string]any, id string) string {
			delete(m, "identity")
			delete(m, "spec")
			return id
		})},
		"empty identity, spec missing": {change: edit(func(m map[string]any, id string) string {
			m["identity"] = ""
			delete(m, "spec")
			return id
		})},
		"empty identity, spec present": {change: edit(func(m map[string]any, id string) string {
			m["identity"] = ""
			return id
		})},
		"incorrect non-empty identity": {change: edit(func(m map[string]any, id string) string {
			m["identity"] = "cu128-0000000000000000"
			return id
		})},
		"identity differs from the activated directory": {change: edit(func(m map[string]any, id string) string {
			return "0.1.0-cu128"
		})},
	} {
		t.Run(name, func(t *testing.T) {
			h := setup.MaterializeFake(t, "cuda")
			id, _ := setup.RuntimeName("cuda")
			tc.change(t, h, id)

			_, _, serveErr := server.WorkerConfig(h, io.Discard)
			var out strings.Builder
			doctor.Run(h.Root, &out)
			doctorRuntimeFails := strings.Contains(out.String(), "FAIL runtime")
			installed := firstrun.Env{}.IsInstalled(h.Root)

			if tc.valid {
				if serveErr != nil || doctorRuntimeFails || !installed || !strings.Contains(out.String(), "PASS runtime") {
					t.Fatalf("valid runtime rejected: serve=%v installed=%v\n%s", serveErr, installed, out.String())
				}
				return
			}
			if serveErr == nil || !doctorRuntimeFails || installed {
				t.Fatalf("activation and doctor disagree: serve=%v doctorFails=%v installed=%v\n%s", serveErr, doctorRuntimeFails, installed, out.String())
			}
			if !strings.Contains(out.String(), "class: "+doctor.RuntimeInvalid) || !strings.Contains(serveErr.Error(), "hachidori setup") {
				t.Fatalf("rejection is not actionable: %v\n%s", serveErr, out.String())
			}
		})
	}
}

// Issue #107: the physical state (a legacy runtime that reaches READY-capable
// activation, requested device cuda) is rejected by serve and doctor alike,
// and `setup` for the same device is the repair that makes both pass the
// runtime check, without selecting the CPU.
func TestLegacyRuntimeRejectedThenRepairedForCUDA(t *testing.T) {
	h := setup.MaterializeFakeLegacy(t, "cuda")
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("serve accepted a runtime without a Runtime Spec identity")
	}
	var out strings.Builder
	if doctor.Run(h.Root, &out) || !strings.Contains(out.String(), "FAIL runtime") || !strings.Contains(out.String(), "run `hachidori setup`") {
		t.Fatalf("doctor:\n%s", out.String())
	}
	if err := setup.Run(h, "cuda", "", io.Discard); err != nil {
		t.Fatal(err)
	}
	cfg, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if rt.Device != "cuda" || !slices.Contains(cfg.Args, "cuda") || slices.Contains(cfg.Args, "cpu") {
		t.Fatalf("requested CUDA not preserved: %+v %v", rt, cfg.Args)
	}
	out.Reset()
	doctor.Run(h.Root, &out)
	if !strings.Contains(out.String(), "PASS runtime") || !strings.Contains(out.String(), "PASS model") {
		t.Fatalf("doctor did not reach the checks after runtime:\n%s", out.String())
	}
}

type idle struct{}

func (idle) Decide([]worker.Item) ([][]api.Result, float64, error) { return nil, 0, nil }
func (idle) Ready() bool                                           { return false }
func (idle) State() string                                         { return worker.StateStarting }
func (idle) Snapshot() worker.Snapshot                             { return worker.Snapshot{} }

// A model is served only by a runtime that carries its provider: an
// activation pairing a model with a runtime built without that provider is
// refused before any worker starts, with the action that fixes it.
func TestServeRefusesRuntimeWithoutModelProvider(t *testing.T) {
	h := setup.MaterializeFakeModel(t, "cpu", setup.TunedModel)
	var a home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := server.WorkerConfig(h, io.Discard); err != nil {
		t.Fatal(err)
	}

	var rm home.RuntimeManifest
	if err := home.ReadJSON(h.Path("runtime", a.Runtime, "manifest.json"), &rm); err != nil {
		t.Fatal(err)
	}
	rm.Spec.Provider = "laya==0.3.21" // a runtime from before OpenDecider was carried
	rm.Identity = rm.Spec.ID()
	if err := os.Rename(h.Path("runtime", a.Runtime), h.Path("runtime", rm.Identity)); err != nil {
		t.Fatal(err)
	}
	if err := home.WriteJSON(h.Path("runtime", rm.Identity, "manifest.json"), rm); err != nil {
		t.Fatal(err)
	}
	a.Runtime = rm.Identity
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), a); err != nil {
		t.Fatal(err)
	}
	_, _, err := server.WorkerConfig(h, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "does not carry provider opendecider") || !strings.Contains(err.Error(), "hachidori setup") {
		t.Fatalf("err = %v", err)
	}
}
