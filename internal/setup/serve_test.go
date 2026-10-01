package setup_test

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
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
	if cfg.Args[0] != "-I" || cfg.Args[3] != filepath.Join(h.Root, "runtime", id, "worker", "hachidori_worker.py") {
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
	if !slices.Equal(cfg.Args[4:], []string{"--model-dir", modelDir, "--device", "cpu", "--manifest", filepath.Join(modelDir, "hachidori-model.json")}) {
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
