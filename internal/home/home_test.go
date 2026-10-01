package home

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestEnvIsExplicit(t *testing.T) {
	t.Setenv("PYTHONPATH", "/user/site")
	t.Setenv("VIRTUAL_ENV", "/user/venv")
	t.Setenv("HF_HOME", "/user/hf")
	root := t.TempDir()
	h := Home{Root: root}
	env := map[string]string{}
	for _, kv := range h.Env(filepath.Join(root, "runtime", "py"), true) {
		k, v, _ := strings.Cut(kv, "=")
		env[k] = v
	}
	for _, k := range []string{"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"} {
		if _, ok := env[k]; ok {
			t.Errorf("%s leaked into worker env", k)
		}
	}
	if env["PYTHONNOUSERSITE"] != "1" || env["HF_HUB_OFFLINE"] != "1" {
		t.Errorf("isolation flags missing: %v", env)
	}
	for _, k := range []string{"HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "HOME", "TMPDIR", "CUDA_CACHE_PATH", "PIP_CACHE_DIR"} {
		if !strings.HasPrefix(env[k], root) {
			t.Errorf("%s=%q not under HACHIDORI_HOME", k, env[k])
		}
	}
	if !strings.HasPrefix(env["PATH"], filepath.Join(root, "runtime", "py")) {
		t.Errorf("PATH=%q", env["PATH"])
	}
}

func TestResolve(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	if _, err := Resolve(""); err == nil {
		t.Fatal("unset home accepted")
	}
	t.Setenv("HACHIDORI_HOME", "rel")
	h, err := Resolve("")
	if err != nil || !filepath.IsAbs(h.Root) {
		t.Fatal(h, err)
	}
}

func TestWriteJSONReplacesAtomicallyAndLeavesNothingBehind(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	if err := WriteJSON(path, map[string]int{"v": 1}); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSON(path, map[string]int{"v": 2}); err != nil {
		t.Fatal(err)
	}
	var got map[string]int
	if err := ReadJSON(path, &got); err != nil || got["v"] != 2 {
		t.Fatalf("read %v %v", got, err)
	}
	if fi, err := os.Stat(path); err != nil || (runtime.GOOS != "windows" && fi.Mode().Perm() != 0o644) {
		t.Fatalf("mode %v %v, want 0644", fi.Mode().Perm(), err)
	}

	// A write that cannot be published must not leave its temporary file or
	// damage what is there: the target here is a directory.
	blocked := filepath.Join(dir, "blocked")
	if err := os.Mkdir(blocked, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSON(blocked, map[string]int{"v": 3}); err == nil {
		t.Fatal("WriteJSON over a directory succeeded")
	}
	ents, _ := os.ReadDir(dir)
	if len(ents) != 2 {
		t.Fatalf("failed write left stray files: %v", ents)
	}
}
