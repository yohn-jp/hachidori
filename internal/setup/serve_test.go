package setup_test

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
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
