package home

import (
	"path/filepath"
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
