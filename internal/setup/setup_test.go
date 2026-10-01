package setup

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

func TestFetchVerifiesDigest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "payload") }))
	defer srv.Close()
	sum := sha256.Sum256([]byte("payload"))
	dst := filepath.Join(t.TempDir(), "a", "f")
	if err := fetch(srv.URL, dst, hex.EncodeToString(sum[:]), io.Discard); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(t.TempDir(), "g")
	if err := fetch(srv.URL, bad, strings.Repeat("0", 64), io.Discard); err == nil {
		t.Fatal("digest mismatch accepted")
	}
	if _, err := os.Stat(bad); !os.IsNotExist(err) {
		t.Fatal("unverified file left behind")
	}
}

func TestFetchAbandonsAStalledDownload(t *testing.T) {
	old := downloadStall
	downloadStall = 150 * time.Millisecond
	t.Cleanup(func() { downloadStall = old })
	sum := sha256.Sum256([]byte("payload"))
	want := hex.EncodeToString(sum[:])
	for name, handler := range map[string]http.HandlerFunc{
		"no response headers": func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() },
		"body stops midway": func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, "pay")
			w.(http.Flusher).Flush()
			<-r.Context().Done()
		},
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(handler)
			defer srv.Close()
			dst := filepath.Join(t.TempDir(), "f")
			err := fetch(srv.URL, dst, want, io.Discard)
			if err == nil || !strings.Contains(err.Error(), "stalled") {
				t.Fatalf("err = %v, want a stall failure", err)
			}
			for _, p := range []string{dst, dst + ".part"} {
				if _, err := os.Stat(p); !os.IsNotExist(err) {
					t.Fatalf("%s left behind after a stalled download", p)
				}
			}
		})
	}
}

// The bound is on progress, not on the whole download: a transfer that takes
// much longer than the stall period but keeps arriving completes.
func TestFetchKeepsASlowButProgressingDownload(t *testing.T) {
	old := downloadStall
	downloadStall = 500 * time.Millisecond
	t.Cleanup(func() { downloadStall = old })
	const chunks = 8
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for i := 0; i < chunks; i++ {
			io.WriteString(w, "0123456789")
			w.(http.Flusher).Flush()
			time.Sleep(100 * time.Millisecond)
		}
	}))
	defer srv.Close()
	sum := sha256.Sum256([]byte(strings.Repeat("0123456789", chunks)))
	dst := filepath.Join(t.TempDir(), "f")
	t0 := time.Now()
	if err := fetch(srv.URL, dst, hex.EncodeToString(sum[:]), io.Discard); err != nil {
		t.Fatal(err)
	}
	if time.Since(t0) <= downloadStall {
		t.Fatal("the transfer was not longer than the stall period; the test proves nothing")
	}
}

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Proof 1: the uv artifact and version are an exact, deterministic pin per platform.
func TestUVArtifactSelection(t *testing.T) {
	want := map[string]string{
		"linux/amd64":   "https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz",
		"windows/amd64": "https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-pc-windows-msvc.zip",
	}
	if uvVersion != "0.12.19" {
		t.Fatalf("uv version %s", uvVersion)
	}
	for plat, url := range want {
		a, err := uvFor(plat)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := uvFor(plat)
		if a != b || a.URL != url || !strings.Contains(a.URL, "/"+uvVersion+"/") {
			t.Errorf("%s: %+v", plat, a)
		}
		if !hex64.MatchString(a.SHA256) || !hex64.MatchString(a.BinarySHA256) || a.Member == "" {
			t.Errorf("%s: incomplete pin %+v", plat, a)
		}
	}
	if len(uvArtifacts) != len(want) {
		t.Errorf("unexpected platforms: %v", uvArtifacts)
	}
	for _, plat := range []string{"darwin/arm64", "linux/arm64", "windows/386"} {
		if _, err := uvFor(plat); err == nil {
			t.Errorf("%s: unpinned platform accepted", plat)
		}
		if _, err := desiredFor("cpu", plat); err == nil {
			t.Errorf("%s: spec for unsupported platform", plat)
		}
	}
}

func TestExtractMember(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("uv-binary")
	tgz := filepath.Join(dir, "a.tar.gz")
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	tw.WriteHeader(&tar.Header{Name: "../uv", Typeflag: tar.TypeReg, Mode: 0o755, Size: 4})
	tw.Write([]byte("evil"))
	tw.WriteHeader(&tar.Header{Name: "d/uv", Typeflag: tar.TypeReg, Mode: 0o755, Size: int64(len(payload))})
	tw.Write(payload)
	tw.Close()
	gz.Close()
	os.WriteFile(tgz, buf.Bytes(), 0o644)

	zp := filepath.Join(dir, "a.zip")
	zf, _ := os.Create(zp)
	zw := zip.NewWriter(zf)
	w, _ := zw.Create("uv.exe")
	w.Write(payload)
	zw.Close()
	zf.Close()

	for archive, member := range map[string]string{tgz: "d/uv", zp: "uv.exe"} {
		dst := filepath.Join(dir, "out")
		if err := extractMember(archive, member, dst); err != nil {
			t.Fatal(err)
		}
		if b, _ := os.ReadFile(dst); !bytes.Equal(b, payload) {
			t.Errorf("%s: got %q", archive, b)
		}
		if err := extractMember(archive, "missing", dst); err == nil {
			t.Errorf("%s: missing member accepted", archive)
		}
	}
}

// Proof 2: uv integrity is verified at bootstrap (archive and executable)
// and again before every execution.
func TestUVBootstrapVerifiesDigest(t *testing.T) {
	f := newFixture(t)
	f.H.Ensure()
	good := uvArtifacts[platform()]

	uvArtifacts[platform()] = uvArtifact{URL: good.URL, SHA256: strings.Repeat("0", 64), Member: good.Member, BinarySHA256: good.BinarySHA256}
	if _, err := ensureUV(f.H, io.Discard); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("archive digest mismatch: %v", err)
	}
	uvArtifacts[platform()] = uvArtifact{URL: good.URL, SHA256: good.SHA256, Member: good.Member, BinarySHA256: strings.Repeat("0", 64)}
	if _, err := ensureUV(f.H, io.Discard); err == nil || !strings.Contains(err.Error(), "extracted executable") {
		t.Fatalf("executable digest mismatch: %v", err)
	}
	exe := filepath.Join(uvDir(f.H), "uv")
	if _, err := os.Stat(exe); !os.IsNotExist(err) {
		t.Fatal("unverified uv left in place")
	}

	uvArtifacts[platform()] = good
	uv, err := ensureUV(f.H, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if uv.exe != exe || !filepath.IsAbs(uv.exe) {
		t.Fatalf("uv at %s, want %s", uv.exe, exe)
	}
	// Tampering after bootstrap: execution is refused ...
	if err := os.WriteFile(exe, []byte("#!/bin/sh\ntouch "+f.H.Path("tampered")+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := uv.run(f.H.Root, uv.env(), "--version"); err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("tampered uv executed: %v", err)
	}
	if _, err := os.Stat(f.H.Path("tampered")); err == nil {
		t.Fatal("tampered uv ran")
	}
	// ... and the next bootstrap restores the pinned executable from the verified archive.
	if _, err := ensureUV(f.H, io.Discard); err != nil {
		t.Fatal(err)
	}
	if got, _ := FileSHA256(exe); got != good.BinarySHA256 {
		t.Fatal("pinned uv not restored")
	}
}

// Proofs 3 and 9: uv from PATH is never used; the managed uv is invoked by
// absolute path with a constructed PATH.
func TestNoPathUVFallback(t *testing.T) {
	f := newFixture(t)
	pathDir := t.TempDir()
	marker := filepath.Join(pathDir, "system-uv-ran")
	script := "#!/bin/sh\ntouch " + marker + "\nexit 0\n"
	for _, n := range []string{"uv", "python", "python3", "pip"} {
		os.WriteFile(filepath.Join(pathDir, n), []byte(script), 0o755)
	}
	t.Setenv("PATH", pathDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("UV_INDEX_URL", "https://attacker.invalid/simple")
	t.Setenv("UV_PYTHON", "/usr/bin/python3")

	f.setDown("/uv/uv-fake.tar.gz", true)
	if _, err := f.run("cpu"); err == nil || !strings.Contains(err.Error(), "private uv") {
		t.Fatalf("setup without private uv: %v", err)
	}
	if _, err := os.Stat(marker); err == nil {
		t.Fatal("uv/python from PATH was executed")
	}
	if len(f.calls()) != 0 || len(f.active()) != 0 {
		t.Fatal("materialization proceeded without the private uv")
	}

	f.setDown("/uv/uv-fake.tar.gz", false)
	f.mustRun("cpu")
	if _, err := os.Stat(marker); err == nil {
		t.Fatal("uv/python from PATH was executed")
	}
	calls := f.calls()
	if len(calls) == 0 {
		t.Fatal("no uv calls recorded")
	}
	want := filepath.Join(f.H.Root, "tools", "uv", uvVersion, "uv")
	for _, c := range calls {
		if c.Exe != want {
			t.Errorf("uv invoked as %q, want absolute %q", c.Exe, want)
		}
		if c.Path != filepath.Dir(want) {
			t.Errorf("uv PATH=%q, want only the private uv dir", c.Path)
		}
		for _, kv := range c.Env {
			if strings.HasPrefix(kv, "UV_INDEX_URL=") || strings.HasPrefix(kv, "UV_PYTHON=") {
				t.Errorf("ambient uv configuration leaked: %s", kv)
			}
		}
		if !slices.Contains(c.Env, "UV_NO_CONFIG=1") || !slices.Contains(c.Env, "UV_MANAGED_PYTHON=1") {
			t.Errorf("uv not isolated from user config/system python: %v", c.Env)
		}
	}
}

// Proofs 4, 5 and 11: the Runtime Spec identity is deterministic, changes
// with every semantic field, and separates CUDA from CPU.
func TestRuntimeSpecIdentity(t *testing.T) {
	a, err := desiredFor("cuda", "windows/amd64")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := desiredFor("cuda", "windows/amd64")
	if a.ID() != b.ID() || a.ID() != a.ID() {
		t.Fatal("identity not deterministic")
	}
	if !regexp.MustCompile(`^cu128-[0-9a-f]{16}$`).MatchString(a.ID()) {
		t.Fatalf("identity %q", a.ID())
	}
	cpu, _ := desiredFor("cpu", "windows/amd64")
	if cpu.ID() == a.ID() || !strings.HasPrefix(cpu.ID(), "cpu-") || cpu.Torch != "2.11.0+cpu" || a.Torch != "2.11.0+cu128" {
		t.Fatalf("cuda %s / cpu %s", a.ID(), cpu.ID())
	}
	// Same digest part with only the flavor label changed must still differ.
	relabeled := cpu
	relabeled.Flavor = "cu128"
	if relabeled.ID() == a.ID() {
		t.Fatal("CUDA/CPU identities collide")
	}
	mutations := map[string]func(*home.RuntimeSpec){
		"schema":   func(s *home.RuntimeSpec) { s.Schema = "hachidori.runtime-spec/2" },
		"platform": func(s *home.RuntimeSpec) { s.Platform = "linux/amd64" },
		"python":   func(s *home.RuntimeSpec) { s.Python = "3.12.12" },
		"provider": func(s *home.RuntimeSpec) { s.Provider = "laya==0.3.22,opendecider==0.3.0" },
		"torch":    func(s *home.RuntimeSpec) { s.Torch = "2.11.1+cu128" },
		"flavor":   func(s *home.RuntimeSpec) { s.Flavor = "cu129" },
		"uv":       func(s *home.RuntimeSpec) { s.UV = "0.12.20" },
		"uv_sha":   func(s *home.RuntimeSpec) { s.UVSHA256 = strings.Repeat("1", 64) },
		"project":  func(s *home.RuntimeSpec) { s.Project = strings.Repeat("2", 64) },
		"lock":     func(s *home.RuntimeSpec) { s.Lock = strings.Repeat("3", 64) },
		"worker":   func(s *home.RuntimeSpec) { s.Worker = strings.Repeat("4", 64) },
	}
	seen := map[string]string{a.ID(): "base"}
	for name, mut := range mutations {
		s := a
		mut(&s)
		if prev, dup := seen[s.ID()]; dup {
			t.Errorf("%s change keeps identity of %s", name, prev)
		}
		seen[s.ID()] = name
	}
	lin, _ := desiredFor("cuda", "linux/amd64")
	if lin.ID() == a.ID() {
		t.Error("platforms share an identity")
	}
	// The model is not part of the Python runtime identity.
	old := Models
	t.Cleanup(func() { Models = old })
	Models = []home.ModelManifest{{ID: DefaultModel, Provider: "laya", Repo: "x/y", Revision: strings.Repeat("f", 40)}}
	if c, _ := desiredFor("cuda", "windows/amd64"); c.ID() != a.ID() {
		t.Error("model catalog changed the runtime identity")
	}
}

// The embedded uv project is the environment authority, and the spec's
// verification targets agree with it.
func TestSpecMatchesProject(t *testing.T) {
	proj, lock := string(specFile("pyproject.toml")), string(specFile("uv.lock"))
	for _, s := range []string{
		`requires-python = "==` + pythonVersion + `"`,
		`"` + providerLaya + "==" + layaVersion + `"`,
		`"` + providerOpenDecider + "==" + openDeciderVersion + `"`,
		`cpu = ["torch==` + torchVersion + `+cpu"]`,
		`cu128 = ["torch==` + torchVersion + `+cu128"]`,
		`https://download.pytorch.org/whl/cu128`,
	} {
		if !strings.Contains(proj, s) {
			t.Errorf("pyproject.toml lacks %s", s)
		}
	}
	for _, s := range []string{
		`requires-python = "==` + pythonVersion + `"`,
		"name = \"laya\"\nversion = \"" + layaVersion + "\"",
		"name = \"opendecider\"\nversion = \"" + openDeciderVersion + "\"",
		"name = \"torch\"\nversion = \"" + torchVersion + "+cpu\"",
		"name = \"torch\"\nversion = \"" + torchVersion + "+cu128\"",
		`{ package = "hachidori-runtime", extra = "cpu" }`,
	} {
		if !strings.Contains(lock, s) {
			t.Errorf("uv.lock lacks %q", s)
		}
	}
	for dev, flavor := range flavors {
		if !strings.Contains(proj, "\n"+flavor+" = [") {
			t.Errorf("%s: no uv extra %s", dev, flavor)
		}
	}
	for _, m := range Models {
		for rel, sum := range m.Files {
			if !hex64.MatchString(sum) {
				t.Errorf("%s %s: bad digest", m.ID, rel)
			}
		}
	}
	if _, err := RuntimeName("rocm"); err == nil {
		t.Error("unknown device accepted")
	}
}

func subcommands(calls []uvCall) []string {
	var out []string
	for _, c := range calls {
		out = append(out, strings.Join(c.Args[:min(2, len(c.Args))], " "))
	}
	return out
}

// Proofs 6 and 12: setup materializes through uv into an immutable runtime,
// reuses it (and the model) on rerun, and the model is materialized
// independently of uv.
func TestRunMaterializesAndReuses(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	spec, _ := Desired("cpu")
	id := spec.ID()
	var a home.Active
	if err := home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	if a != (home.Active{Runtime: id, ModelID: DefaultModel, Model: ModelDirName(f.model("")), Device: "cpu"}) {
		t.Fatalf("active %+v", a)
	}
	_, rm, _, err := f.H.LoadActive()
	if err != nil {
		t.Fatal(err)
	}
	if rm.Identity != id || rm.Spec != spec || rm.PythonVersion != "3.12.11" || !slices.Contains(rm.Installed, "torch==2.11.0+cpu") {
		t.Fatalf("manifest %+v", rm)
	}
	if !within(uvDir(f.H), rm.BasePython) {
		t.Fatalf("base python %s outside the private uv tree", rm.BasePython)
	}
	if _, err := os.Stat(f.H.Path("runtime", ".staging-"+id)); !os.IsNotExist(err) {
		t.Fatal("staging left behind after publish")
	}
	calls := f.calls()
	if got := subcommands(calls); !slices.Equal(got, []string{"python install", "venv --relocatable", "sync --locked"}) {
		t.Fatalf("uv calls %v", got)
	}
	sync := calls[2]
	if flagValue(sync.Args, "--extra") != "cpu" || !slices.Contains(sync.Args, "--no-build") ||
		!slices.Contains(sync.Env, "UV_PROJECT_ENVIRONMENT="+filepath.Join(f.H.Root, "runtime", ".staging-"+id, "env")) {
		t.Fatalf("sync %+v", sync.Args)
	}
	for _, c := range calls {
		for _, arg := range c.Args {
			if strings.Contains(arg, "huggingface") || strings.Contains(arg, f.model("").Repo) || strings.Contains(arg, "models") {
				t.Fatalf("uv involved in model materialization: %v", c.Args)
			}
		}
	}
	modelPath := "/test/model/resolve/" + f.model("").Revision + "/config.json"
	if f.hitCount(modelPath) != 1 {
		t.Fatalf("model fetched %d times", f.hitCount(modelPath))
	}

	before := snapshot(t, f.H.Path("runtime", id))
	log, err := f.run("cpu")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(log, "runtime "+id+" verified, reusing") || !strings.Contains(log, "verified, reusing") {
		t.Fatalf("no reuse:\n%s", log)
	}
	if len(f.calls()) != len(calls) {
		t.Fatalf("rerun invoked uv again: %v", subcommands(f.calls()))
	}
	if f.hitCount(modelPath) != 1 || f.hitCount("/uv/uv-fake.tar.gz") != 1 {
		t.Fatal("rerun downloaded again")
	}
	if after := snapshot(t, f.H.Path("runtime", id)); !maps(before, after) {
		t.Fatal("reused runtime was modified")
	}

	// A different spec (CUDA) materializes side by side and reuses the model.
	f.mustRun("cuda")
	cuda, _ := Desired("cuda")
	if cuda.ID() == id {
		t.Fatal("cpu/cuda share a runtime")
	}
	if after := snapshot(t, f.H.Path("runtime", id)); !maps(before, after) {
		t.Fatal("cpu runtime modified by cuda setup")
	}
	if f.hitCount(modelPath) != 1 {
		t.Fatal("model re-fetched for a different runtime")
	}
}

func snapshot(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().IsRegular() {
			sum, _ := FileSHA256(p)
			out[p] = sum + info.ModTime().String()
		}
		return nil
	})
	return out
}

func maps(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

// Proof 8: failed materialization leaves the active runtime unchanged, and
// setup is safely rerunnable.
func TestFailedMaterializationPreservesActive(t *testing.T) {
	for _, fail := range []string{"python install", "venv", "sync"} {
		t.Run(fail, func(t *testing.T) {
			f := newFixture(t)
			f.mustRun("cpu")
			prev := f.active()
			f.control(fakeControl{Fail: fail})
			if _, err := f.run("cuda"); err == nil {
				t.Fatal("injected failure not reported")
			}
			if !bytes.Equal(f.active(), prev) {
				t.Fatal("active runtime changed by failed setup")
			}
			cuda, _ := Desired("cuda")
			if _, err := os.Stat(f.H.Path("runtime", cuda.ID())); !os.IsNotExist(err) {
				t.Fatal("failed runtime published")
			}
			if _, _, _, err := f.H.LoadActive(); err != nil {
				t.Fatalf("previous runtime no longer loads: %v", err)
			}
			f.control(fakeControl{})
			f.mustRun("cuda")
			var a home.Active
			home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a)
			if a.Runtime != cuda.ID() {
				t.Fatalf("rerun did not converge: %+v", a)
			}
		})
	}
}

// A model failure also leaves both the new runtime unpublished and the active
// state unchanged.
func TestModelFailurePreservesActive(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	os.RemoveAll(f.H.Path("models"))
	f.setDown("/test/model/resolve/"+f.model("").Revision+"/config.json", true)
	if _, err := f.run("cuda"); err == nil || !strings.Contains(err.Error(), "model") {
		t.Fatalf("model failure: %v", err)
	}
	cuda, _ := Desired("cuda")
	if _, err := os.Stat(f.H.Path("runtime", cuda.ID())); !os.IsNotExist(err) {
		t.Fatal("runtime published although the model failed")
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("active changed")
	}
	// Corrupted materialized model is an explicit failure, not silently repaired.
	f.setDown("/test/model/resolve/"+f.model("").Revision+"/config.json", false)
	f.mustRun("cpu")
	cfg := filepath.Join(f.H.Path("models"), filepath.FromSlash(ModelDirName(f.model(""))), "config.json")
	os.WriteFile(cfg, []byte("tampered"), 0o644)
	if _, err := f.run("cpu"); err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("tampered model: %v", err)
	}
}

// Proof 7: an incomplete or invalid runtime never becomes active.
func TestInvalidRuntimeNotActivated(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	cuda, _ := Desired("cuda")

	// uv "succeeds" but the environment does not satisfy the spec.
	f.control(fakeControl{Torch: "2.10.0+cu128"})
	if _, err := f.run("cuda"); err == nil || !strings.Contains(err.Error(), "torch==2.11.0+cu128 not installed") {
		t.Fatalf("invalid environment: %v", err)
	}
	if _, err := os.Stat(f.H.Path("runtime", cuda.ID())); !os.IsNotExist(err) || !bytes.Equal(f.active(), prev) {
		t.Fatal("invalid runtime published or activated")
	}

	// An incomplete runtime directory with the desired identity is refused, not repaired.
	f.control(fakeControl{})
	partial := f.H.Path("runtime", cuda.ID())
	os.MkdirAll(filepath.Join(partial, "env"), 0o755)
	if _, err := f.run("cuda"); err == nil || !strings.Contains(err.Error(), "incomplete runtime") {
		t.Fatalf("incomplete runtime: %v", err)
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("incomplete runtime activated")
	}
	if entries, _ := os.ReadDir(partial); len(entries) != 1 {
		t.Fatal("incomplete runtime was modified in place")
	}

	// A published runtime whose content no longer verifies is refused too.
	os.RemoveAll(partial)
	f.mustRun("cuda")
	os.WriteFile(filepath.Join(partial, "worker", "hachidori_worker.py"), []byte("print('x')"), 0o644)
	now := f.active()
	if _, err := f.run("cuda"); err == nil || !strings.Contains(err.Error(), "worker script digest mismatch") {
		t.Fatalf("tampered runtime: %v", err)
	}
	if !bytes.Equal(f.active(), now) {
		t.Fatal("active changed")
	}
}

// Migration: a runtime created by the procedural pip-based setup never
// satisfies the declarative identity and is left untouched.
func TestLegacyRuntimeNotReused(t *testing.T) {
	f := newFixture(t)
	f.H.Ensure()
	legacy := f.H.Path("runtime", "0.1.0-cpu")
	os.MkdirAll(filepath.Join(legacy, "worker"), 0o755)
	os.WriteFile(filepath.Join(legacy, "manifest.json"), []byte(`{"version":"0.1.0","flavor":"cpu","packages":["torch==2.11.0+cpu"],"installed":["pip==25.2"]}`), 0o644)
	home.WriteJSON(f.H.Path("state", "active-runtime.json"), home.Active{Runtime: "0.1.0-cpu", Model: ModelDirName(f.model("")), Device: "cpu"})
	before := snapshot(t, legacy)

	f.mustRun("cpu")
	spec, _ := Desired("cpu")
	var a home.Active
	home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a)
	if a.Runtime != spec.ID() || a.Runtime == "0.1.0-cpu" {
		t.Fatalf("active %+v", a)
	}
	if !maps(before, snapshot(t, legacy)) {
		t.Fatal("legacy runtime mutated")
	}
}

// Issue #107: a legacy runtime that is still activated (identity and spec
// missing, a valid independent model present) is repaired only by the
// declarative setup. Setup materializes the current runtime beside it, reuses
// the verified model without any download, never edits the legacy directory,
// switches the activation only after everything verified, keeps the requested
// device, and is idempotent.
func TestLegacyActiveRuntimeRepairedBySetup(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cuda")
	// The model of a pre-catalog install carries no id/provider.
	model := f.model("")
	modelDir := f.H.Path("models", filepath.FromSlash(ModelDirName(model)))
	if err := home.WriteJSON(filepath.Join(modelDir, "hachidori-model.json"),
		home.ModelManifest{Repo: model.Repo, Revision: model.Revision, Files: model.Files}); err != nil {
		t.Fatal(err)
	}
	legacy := f.legacyize("cuda")
	legacyDir := f.H.Path("runtime", legacy)
	legacyBefore, modelBefore := snapshot(t, legacyDir), snapshot(t, modelDir)
	activeBefore := f.active()
	modelHits := func() int { return f.hitCount("/test/model/resolve/" + model.Revision + "/config.json") }
	hitsBefore, callsBefore := modelHits(), len(f.calls())

	if _, _, _, err := f.H.LoadActive(); err == nil {
		t.Fatal("legacy runtime is still a valid active runtime")
	}

	// A failed repair changes nothing: the active record stays on the legacy
	// runtime, nothing is published, and no other device is substituted.
	f.control(fakeControl{Fail: "sync"})
	if _, err := f.run("cuda"); err == nil {
		t.Fatal("injected failure not reported")
	}
	f.control(fakeControl{})
	cuda, _ := Desired("cuda")
	cpu, _ := Desired("cpu")
	if !bytes.Equal(f.active(), activeBefore) {
		t.Fatal("failed repair changed the activation record")
	}
	for _, id := range []string{cuda.ID(), cpu.ID()} {
		if _, err := os.Stat(f.H.Path("runtime", id)); !os.IsNotExist(err) {
			t.Fatalf("runtime %s published by a failed repair", id)
		}
	}

	f.mustRun("cuda")
	var a home.Active
	if err := home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	if a != (home.Active{Runtime: cuda.ID(), ModelID: DefaultModel, Model: ModelDirName(model), Device: "cuda"}) {
		t.Fatalf("active %+v", a)
	}
	if _, rm, _, err := f.H.LoadActive(); err != nil || rm.Spec != cuda || rm.Spec.Flavor != "cu128" || !slices.Contains(rm.Installed, "torch==2.11.0+cu128") {
		t.Fatalf("repaired runtime does not load as the current CUDA spec: %v %+v", err, rm)
	}
	if !maps(legacyBefore, snapshot(t, legacyDir)) {
		t.Fatal("legacy runtime modified in place")
	}
	if !maps(modelBefore, snapshot(t, modelDir)) {
		t.Fatal("independent model artifact rewritten by runtime repair")
	}
	if modelHits() != hitsBefore {
		t.Fatalf("model downloaded again by runtime repair (%d requests)", modelHits()-hitsBefore)
	}
	// Both the failed and the successful attempt materialized the runtime via
	// uv exactly as a first setup does; nothing else was invoked.
	for _, c := range f.calls()[callsBefore:] {
		if c.Args[0] != "python" && c.Args[0] != "venv" && c.Args[0] != "sync" {
			t.Fatalf("unexpected uv call %v", c.Args)
		}
	}
	for _, c := range f.calls()[callsBefore:] {
		if sync := c.Args[0] == "sync"; sync && flagValue(c.Args, "--extra") != "cu128" {
			t.Fatalf("CUDA repair resolved %q, not cu128", flagValue(c.Args, "--extra"))
		}
	}

	// Idempotent: a second setup reuses everything and changes nothing.
	activeAfter, calls, hits := f.active(), len(f.calls()), modelHits()
	repaired := snapshot(t, f.H.Path("runtime", cuda.ID()))
	f.mustRun("cuda")
	if !bytes.Equal(f.active(), activeAfter) || len(f.calls()) != calls || modelHits() != hits {
		t.Fatal("second setup was not idempotent")
	}
	if !maps(repaired, snapshot(t, f.H.Path("runtime", cuda.ID()))) {
		t.Fatal("verified runtime rewritten by a second setup")
	}
}

// Proof 10: Hachidori no longer owns a pip-install / requirements /
// package-index construction path; environment construction goes through uv.
func TestNoPipInstallPath(t *testing.T) {
	files, _ := filepath.Glob("*.go")
	forbidden := regexp.MustCompile(`"pip"|"-m", "pip"|requirements\.txt|--index-url|--extra-index-url|pip freeze|pip install|python-build-standalone`)
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if m := forbidden.Find(b); m != nil {
			t.Errorf("%s still contains procedural environment construction: %s", name, m)
		}
	}
	f := newFixture(t)
	f.mustRun("cpu")
	for _, c := range f.calls() {
		if slices.Contains(c.Args, "pip") {
			t.Errorf("uv pip interface used: %v", c.Args)
		}
	}
	spec, _ := Desired("cpu")
	if _, err := os.Stat(f.H.Path("runtime", spec.ID(), "requirements.txt")); err == nil {
		t.Error("requirements.txt written")
	}
	for _, n := range []string{"pyproject.toml", "uv.lock"} {
		b, _ := os.ReadFile(f.H.Path("runtime", spec.ID(), "spec", n))
		if !bytes.Equal(b, specFile(n)) {
			t.Errorf("%s: runtime does not carry the exact spec file it was materialized from", n)
		}
	}
}

// RunObserved reports real setup phases in order and stops at the failing phase.
func TestRunObservedPhases(t *testing.T) {
	f := newFixture(t)
	var got []Phase
	if err := RunObserved(f.H, "cpu", DefaultModel, io.Discard, func(p Phase) { got = append(got, p) }); err != nil {
		t.Fatal(err)
	}
	want := []Phase{PhasePreparing, PhaseRuntime, PhaseModel, PhaseActivation}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("phases %v, want %v", got, want)
	}

	f2 := newFixture(t)
	f2.control(fakeControl{Fail: "venv"})
	got = nil
	if err := RunObserved(f2.H, "cpu", DefaultModel, io.Discard, func(p Phase) { got = append(got, p) }); err == nil {
		t.Fatal("injected uv failure did not fail setup")
	}
	if want := []Phase{PhasePreparing, PhaseRuntime}; !reflect.DeepEqual(got, want) {
		t.Fatalf("phases on runtime failure %v, want %v", got, want)
	}
}
