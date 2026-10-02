package home

import (
	"encoding/json"
	"errors"
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

// legacyManifest is a runtime manifest as the procedural pip-based setup wrote
// it (Hachidori 0.1.0): no identity and no Runtime Spec.
const legacyManifest = `{"version":"0.1.0","flavor":"cu128","platform":"windows/amd64","python_version":"3.12.11",
"python_archive":{"url":"https://example.invalid/python.zip","sha256":"00"},"python":"python/python.exe",
"packages":["torch==2.11.0+cu128"],"package_indexes":["https://download.pytorch.org/whl/cu128"],
"installed":["laya==0.3.21","torch==2.11.0+cu128"],"worker":{"worker/hachidori_worker.py":"00"}}`

func testSpec() RuntimeSpec {
	return RuntimeSpec{Schema: "hachidori.runtime-spec/1", Platform: "linux/amd64", Python: "3.12.11",
		Provider: "laya==0.3.21", Torch: "2.11.0+cu128", Flavor: "cu128", UV: "0.12.19",
		UVSHA256: "a", Project: "b", Lock: "c", Worker: "d"}
}

// CheckIdentity is the one runtime-validity rule. Only an absent identity
// with no Runtime Spec is the missing-identity (legacy) class; a wrong,
// foreign or partial identity is corruption and is never classified as legacy.
func TestCheckIdentity(t *testing.T) {
	spec := testSpec()
	id := spec.ID()
	other := spec
	other.Torch = "2.11.0+cpu"

	for name, tc := range map[string]struct {
		m       RuntimeManifest
		runtime string
		missing bool // ErrRuntimeIdentityMissing
		invalid bool // any other rejection
	}{
		"valid current identity":            {m: RuntimeManifest{Identity: id, Spec: spec}, runtime: id},
		"missing identity and spec":         {m: RuntimeManifest{}, runtime: "0.1.0-cu128", missing: true},
		"empty identity, spec absent":       {m: RuntimeManifest{Identity: ""}, runtime: id, missing: true},
		"empty identity with a spec":        {m: RuntimeManifest{Spec: spec}, runtime: id, invalid: true},
		"wrong non-empty identity":          {m: RuntimeManifest{Identity: "cu128-0000000000000000", Spec: spec}, runtime: "cu128-0000000000000000", invalid: true},
		"identity without a spec":           {m: RuntimeManifest{Identity: id}, runtime: id, invalid: true},
		"identity of a different spec":      {m: RuntimeManifest{Identity: id, Spec: other}, runtime: id, invalid: true},
		"identity not the activated record": {m: RuntimeManifest{Identity: id, Spec: spec}, runtime: "0.1.0-cu128", invalid: true},
	} {
		t.Run(name, func(t *testing.T) {
			err := tc.m.CheckIdentity(tc.runtime)
			switch {
			case tc.missing:
				if !errors.Is(err, ErrRuntimeIdentityMissing) {
					t.Fatalf("got %v, want ErrRuntimeIdentityMissing", err)
				}
			case tc.invalid:
				if err == nil || errors.Is(err, ErrRuntimeIdentityMissing) {
					t.Fatalf("got %v, want a non-legacy identity rejection", err)
				}
			default:
				if err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

// LoadActive applies CheckIdentity, so every consumer of the activation
// record agrees: a legacy manifest is recognized as missing its identity, a
// corrupted non-empty identity is rejected as such, a valid one loads.
func TestLoadActiveAppliesRuntimeIdentity(t *testing.T) {
	spec := testSpec()
	id := spec.ID()
	load := func(runtime, manifest string) error {
		h := Home{Root: t.TempDir()}
		if err := h.Ensure(); err != nil {
			t.Fatal(err)
		}
		write := func(p, s string) {
			t.Helper()
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write(h.Path("runtime", runtime, "manifest.json"), manifest)
		write(h.Path("models", "m", "r", "hachidori-model.json"), `{}`)
		if err := WriteJSON(h.Path("state", "active-runtime.json"), Active{Runtime: runtime, Model: "m/r", Device: "cuda"}); err != nil {
			t.Fatal(err)
		}
		_, _, _, err := h.LoadActive()
		return err
	}
	current, _ := json.Marshal(RuntimeManifest{Identity: id, Spec: spec})
	corrupt, _ := json.Marshal(RuntimeManifest{Identity: "cu128-0000000000000000", Spec: spec})

	if err := load(id, string(current)); err != nil {
		t.Fatalf("current manifest: %v", err)
	}
	for name, manifest := range map[string]string{
		"legacy, identity absent": legacyManifest,
		"legacy, identity empty":  `{"identity":"","version":"0.1.0"}`,
		"legacy, empty object":    `{}`,
	} {
		if err := load("0.1.0-cu128", manifest); !errors.Is(err, ErrRuntimeIdentityMissing) {
			t.Errorf("%s: got %v, want ErrRuntimeIdentityMissing", name, err)
		}
	}
	if err := load("cu128-0000000000000000", string(corrupt)); err == nil || errors.Is(err, ErrRuntimeIdentityMissing) {
		t.Errorf("corrupt identity: got %v, want a non-legacy rejection", err)
	}
}

// The activation record is restored byte for byte, atomically.
func TestRestoreActiveRecordIsByteExact(t *testing.T) {
	h := Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	if _, err := h.ReadActiveRecord(); !os.IsNotExist(err) {
		t.Fatalf("read of a missing record: %v", err)
	}
	// Not what WriteJSON would produce: indentation, key order and a trailing
	// blank line survive a restore.
	raw := []byte("{\"device\":\"cuda\",   \"runtime\":\"rt\",\n \"model\":\"m\"}\n\n")
	if err := h.RestoreActiveRecord(raw); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSON(h.Path("state", "active-runtime.json"), Active{Runtime: "other", Model: "m", Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	if err := h.RestoreActiveRecord(raw); err != nil {
		t.Fatal(err)
	}
	got, err := h.ReadActiveRecord()
	if err != nil || string(got) != string(raw) {
		t.Fatalf("restored %q, want %q (%v)", got, raw, err)
	}
	if es, _ := os.ReadDir(h.Path("state")); len(es) != 1 {
		t.Fatalf("a restore leaves only the record, got %d entries", len(es))
	}
}
