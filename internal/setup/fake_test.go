package setup

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// TestMain doubles as a fake uv and a fake private python, selected by the
// executable name. Setup constructs the environment of both processes, so
// the fakes are configured through files under HACHIDORI_HOME instead:
//
//	fake-uv.json    {"fail": "<uv subcommand>", "torch": "<installed torch version>"}
//	uv-calls.jsonl  one record per uv invocation (argv[0], args, PATH, cwd)
func TestMain(m *testing.M) {
	// syscall.Exit skips the race detector's exit delay in the fake processes.
	switch strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") {
	case "uv":
		syscall.Exit(fakeUV(os.Args[1:]))
	case "python":
		syscall.Exit(fakePython())
	}
	os.Exit(m.Run())
}

type uvCall struct {
	Exe  string   `json:"exe"`
	Args []string `json:"args"`
	Path string   `json:"path"`
	Cwd  string   `json:"cwd"`
	Env  []string `json:"env"`
}

type fakeControl struct {
	Fail  string `json:"fail"`
	Torch string `json:"torch"`
}

func fakeUV(args []string) int {
	root := os.Getenv("HACHIDORI_HOME")
	cwd, _ := os.Getwd()
	b, _ := json.Marshal(uvCall{Exe: os.Args[0], Args: args, Path: os.Getenv("PATH"), Cwd: cwd, Env: os.Environ()})
	f, err := os.OpenFile(filepath.Join(root, "uv-calls.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return 3
	}
	f.Write(append(b, '\n'))
	f.Close()
	var ctl fakeControl
	home.ReadJSON(filepath.Join(root, "fake-uv.json"), &ctl)
	if len(args) == 0 {
		return 2
	}
	sub := args[0]
	if sub == "python" && len(args) > 1 {
		sub = "python " + args[1]
	}
	if ctl.Fail == sub {
		fmt.Fprintln(os.Stderr, "fake uv: injected failure in", sub)
		return 1
	}
	switch sub {
	case "python install":
		dir := filepath.Join(os.Getenv("UV_PYTHON_INSTALL_DIR"), "cpython-"+args[2]+"-fake")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return 1
		}
		return writeOr1(filepath.Join(dir, "version"), []byte(args[2]))
	case "venv":
		env, ver := args[len(args)-1], flagValue(args, "--python")
		base := filepath.Join(os.Getenv("UV_PYTHON_INSTALL_DIR"), "cpython-"+ver+"-fake")
		if _, err := os.Stat(base); err != nil {
			fmt.Fprintln(os.Stderr, "fake uv: no managed python", ver)
			return 1
		}
		self, _ := os.Executable()
		dst := filepath.Join(filepath.Dir(env), filepath.FromSlash(pythonRelPath()))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil || copyFile(self, dst) != nil {
			return 1
		}
		return writeOr1(filepath.Join(env, "pyvenv.cfg"), []byte("home = "+base+"\n"))
	case "sync":
		for _, f := range []string{"pyproject.toml", "uv.lock"} {
			if _, err := os.Stat(f); err != nil {
				return 1
			}
		}
		torch := "2.11.0+" + flagValue(args, "--extra")
		if ctl.Torch != "" {
			torch = ctl.Torch
		}
		pkgs := []string{"laya==0.3.21", "opendecider==0.3.0", "numpy==2.5.3", "torch==" + torch, "transformers==5.17.0", "safetensors==0.8.0", "tokenizers==0.23.2", "accelerate==1.15.0", "compressed-tensors==0.19.0"}
		if flagValue(args, "--extra") == "cu128" {
			pkgs = append(pkgs, clefKernelDistributions(home.RuntimeSpec{Flavor: "cu128", Platform: platform()})...)
		}
		if pj, _ := os.ReadFile("pyproject.toml"); strings.Contains(string(pj), "hachidori-optimizer") {
			pkgs = []string{"llmcompressor==0.14.0", "compressed-tensors==0.19.0", "numpy==2.5.3", "torch==" + torch, "transformers==5.17.0"}
		}
		b, _ := json.Marshal(pkgs)
		return writeOr1(filepath.Join(os.Getenv("UV_PROJECT_ENVIRONMENT"), "installed.json"), b)
	}
	return 2
}

// fakePython answers the runtime verification probe from the files the fake
// uv wrote around it.
func fakePython() int {
	self, _ := os.Executable()
	env := filepath.Dir(filepath.Dir(self))
	cfg, _ := os.ReadFile(filepath.Join(env, "pyvenv.cfg"))
	base := strings.TrimSpace(strings.TrimPrefix(string(cfg), "home = "))
	ver, _ := os.ReadFile(filepath.Join(base, "version"))
	var installed []string
	home.ReadJSON(filepath.Join(env, "installed.json"), &installed)
	b, _ := json.Marshal(runtimeProbe{Python: string(ver), Prefix: env, BasePrefix: base, Installed: installed})
	fmt.Println(string(b))
	return 0
}

func flagValue(args []string, name string) string {
	for i, a := range args {
		if a == name && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func writeOr1(p string, b []byte) int {
	if os.WriteFile(p, b, 0o644) != nil {
		return 1
	}
	return 0
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	return err
}

// fixture is a HACHIDORI_HOME plus a local artifact server hosting a fake uv
// release (the test binary) and a small pinned model.
type fixture struct {
	t    *testing.T
	H    home.Home
	URL  string
	mu   sync.Mutex
	hits map[string]int
	down map[string]bool // paths answering 404
	// clef holds the files of the System One fixture model once addClef was
	// called: rel path -> content, served under /test/clef/resolve/.
	clef map[string][]byte
}

const fakeUVMember = "uv-fake/uv"

var fakeRelease struct {
	once         sync.Once
	bin, archive []byte
	err          error
}

// fakeUVRelease packages the test binary as a uv release archive (built once;
// stored uncompressed to keep -race runs fast).
func fakeUVRelease(t *testing.T) (bin, archive []byte) {
	fakeRelease.once.Do(func() {
		self, err := os.Executable()
		if err != nil {
			fakeRelease.err = err
			return
		}
		b, err := os.ReadFile(self)
		if err != nil {
			fakeRelease.err = err
			return
		}
		var buf bytes.Buffer
		gz, _ := gzip.NewWriterLevel(&buf, gzip.NoCompression)
		tw := tar.NewWriter(gz)
		tw.WriteHeader(&tar.Header{Name: "uv-fake/", Typeflag: tar.TypeDir, Mode: 0o755})
		tw.WriteHeader(&tar.Header{Name: fakeUVMember, Typeflag: tar.TypeReg, Mode: 0o755, Size: int64(len(b))})
		tw.Write(b)
		tw.Close()
		gz.Close()
		fakeRelease.bin, fakeRelease.archive = b, buf.Bytes()
	})
	if fakeRelease.err != nil {
		t.Fatal(fakeRelease.err)
	}
	return fakeRelease.bin, fakeRelease.archive
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake uv/python fixtures are POSIX-only")
	}
	bin, archive := fakeUVRelease(t)

	f := &fixture{t: t, hits: map[string]int{}, down: map[string]bool{}}
	modelFile, tunedFile := []byte(`{"model":"fake"}`), []byte(`{"model":"fake-tuned"}`)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.hits[r.URL.Path]++
		down := f.down[r.URL.Path]
		f.mu.Unlock()
		switch {
		case down:
			http.NotFound(w, r)
		case r.URL.Path == "/uv/uv-fake.tar.gz":
			w.Write(archive)
		case strings.HasPrefix(r.URL.Path, "/test/model/resolve/"):
			w.Write(modelFile)
		case strings.HasPrefix(r.URL.Path, "/test/tuned/resolve/"):
			w.Write(tunedFile)
		case strings.HasPrefix(r.URL.Path, "/test/clef/resolve/"):
			rel := strings.TrimPrefix(r.URL.Path, "/test/clef/resolve/")
			rel = rel[strings.Index(rel, "/")+1:]
			f.mu.Lock()
			body, ok := f.clef[rel]
			f.mu.Unlock()
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Write(body)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	f.URL = srv.URL

	oldVer, oldArts, oldModels, oldBase := uvVersion, uvArtifacts, Models, modelBaseURL
	t.Cleanup(func() { uvVersion, uvArtifacts, Models, modelBaseURL = oldVer, oldArts, oldModels, oldBase })
	uvVersion = "0.0.0-test"
	uvArtifacts = map[string]uvArtifact{platform(): {
		URL: srv.URL + "/uv/uv-fake.tar.gz", SHA256: digest(archive), Member: fakeUVMember, BinarySHA256: digest(bin),
	}}
	// A two-entry catalog: the default identity and a second checkpoint loaded
	// by the other provider, declared the same way.
	Models = []home.ModelManifest{
		{ID: DefaultModel, Provider: "laya", Repo: "test/model", Revision: strings.Repeat("ab", 20),
			Files: map[string]string{"config.json": digest(modelFile)}},
		{ID: tunedModel, Provider: providerOpenDecider, Repo: "test/tuned", Revision: strings.Repeat("cd", 20),
			Files: map[string]string{"config.json": digest(tunedFile)}},
	}
	modelBaseURL = srv.URL + "/"

	f.H = home.Home{Root: t.TempDir()}
	return f
}

func (f *fixture) hitCount(path string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.hits[path]
}

func (f *fixture) setDown(path string, down bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.down[path] = down
}

func (f *fixture) control(c fakeControl) {
	f.t.Helper()
	if err := home.WriteJSON(f.H.Path("fake-uv.json"), c); err != nil {
		f.t.Fatal(err)
	}
}

func (f *fixture) calls() []uvCall {
	f.t.Helper()
	b, err := os.ReadFile(f.H.Path("uv-calls.jsonl"))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		f.t.Fatal(err)
	}
	var out []uvCall
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var c uvCall
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			f.t.Fatal(err)
		}
		out = append(out, c)
	}
	return out
}

// tunedModel is the second catalog entry of the fixture: a model of the other
// provider, so every selection test crosses the provider boundary.
const tunedModel = "opendecider-test"

func (f *fixture) run(device string) (string, error) { return f.runModel(device, "") }

func (f *fixture) runModel(device, model string) (string, error) {
	var log strings.Builder
	err := Run(f.H, device, model, &log)
	return log.String(), err
}

func (f *fixture) mustRun(device string) { f.t.Helper(); f.mustRunModel(device, "") }

func (f *fixture) mustRunModel(device, model string) {
	f.t.Helper()
	if log, err := f.runModel(device, model); err != nil {
		f.t.Fatalf("setup %s %s: %v\n%s", device, model, err, log)
	}
}

// model is the fixture catalog entry for id ("" is the default).
func (f *fixture) model(id string) home.ModelManifest {
	f.t.Helper()
	m, err := LookupModel(id)
	if err != nil {
		f.t.Fatal(err)
	}
	return m
}

func (f *fixture) active() []byte {
	b, _ := os.ReadFile(f.H.Path("state", "active-runtime.json"))
	return b
}

// legacyize turns the runtime setup just materialized for device into the
// state of a runtime written by the procedural pip-based setup: a version-named
// directory whose manifest carries no identity and no Runtime Spec, still
// activated. Models are left exactly as setup materialized them. It returns
// the legacy runtime directory name.
func (f *fixture) legacyize(device string) string {
	f.t.Helper()
	spec, err := Desired(device)
	if err != nil {
		f.t.Fatal(err)
	}
	var rm home.RuntimeManifest
	if err := home.ReadJSON(f.H.Path("runtime", spec.ID(), "manifest.json"), &rm); err != nil {
		f.t.Fatal(err)
	}
	name := "0.1.0-" + spec.Flavor
	if err := os.Rename(f.H.Path("runtime", spec.ID()), f.H.Path("runtime", name)); err != nil {
		f.t.Fatal(err)
	}
	legacy := map[string]any{
		"version": "0.1.0", "flavor": spec.Flavor, "platform": spec.Platform, "python_version": rm.PythonVersion,
		"python": rm.PythonRelPath, "packages": []string{"torch==" + spec.Torch},
		"package_indexes": []string{"https://download.pytorch.org/whl/" + spec.Flavor},
		"installed":       rm.Installed, "worker": rm.Worker,
	}
	if err := home.WriteJSON(f.H.Path("runtime", name, "manifest.json"), legacy); err != nil {
		f.t.Fatal(err)
	}
	var a home.Active
	if err := home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a); err != nil {
		f.t.Fatal(err)
	}
	a.Runtime = name
	if err := home.WriteJSON(f.H.Path("state", "active-runtime.json"), a); err != nil {
		f.t.Fatal(err)
	}
	return name
}

// olderWorker rewrites the active runtime as an older build would have
// materialized it: a self-consistent runtime (its identity is the one its own
// Runtime Spec derives and its manifest agrees with its files) whose worker
// script is not the one this build embeds. It returns the new runtime name.
func (f *fixture) olderWorker(device string) string {
	f.t.Helper()
	spec, err := Desired(device)
	if err != nil {
		f.t.Fatal(err)
	}
	var rm home.RuntimeManifest
	if err := home.ReadJSON(f.H.Path("runtime", spec.ID(), "manifest.json"), &rm); err != nil {
		f.t.Fatal(err)
	}
	script := []byte("# worker of an older build: no --provider argument\n")
	rm.Spec.Worker = digest(script)
	rm.Spec.Provider = "laya==0.3.21"
	rm.Identity = rm.Spec.ID()
	rm.Worker = map[string]string{"worker/hachidori_worker.py": rm.Spec.Worker}
	if err := os.Rename(f.H.Path("runtime", spec.ID()), f.H.Path("runtime", rm.Identity)); err != nil {
		f.t.Fatal(err)
	}
	if err := os.WriteFile(f.H.Path("runtime", rm.Identity, "worker", "hachidori_worker.py"), script, 0o644); err != nil {
		f.t.Fatal(err)
	}
	if err := home.WriteJSON(f.H.Path("runtime", rm.Identity, "manifest.json"), rm); err != nil {
		f.t.Fatal(err)
	}
	var a home.Active
	if err := home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a); err != nil {
		f.t.Fatal(err)
	}
	a.Runtime = rm.Identity
	if err := home.WriteJSON(f.H.Path("state", "active-runtime.json"), a); err != nil {
		f.t.Fatal(err)
	}
	return rm.Identity
}

// clefFixtureFiles are the files of the System One fixture model: the file
// names of the real release, with tiny contents.
var clefFixtureFiles = []string{"LICENSE", "chat_template.jinja", "config.json", "generation_config.json", "joint_head.safetensors",
	"joint_head_config.json", "joint_schema_model.py", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json"}

// addClef adds a System One model to the fixture catalog, under the real
// Clef-Flash ID so that variants and the canonical recipe apply, and serves
// its files from the fixture server. It returns the catalog entry.
func (f *fixture) addClef() home.ModelManifest {
	f.t.Helper()
	f.clef = map[string][]byte{}
	files := map[string]string{}
	for _, rel := range clefFixtureFiles {
		b := []byte("fixture " + rel)
		f.clef[rel] = b
		files[rel] = digest(b)
	}
	m := home.ModelManifest{ID: ClefFlash, Provider: providerClef, Repo: "test/clef", Revision: strings.Repeat("ef", 20), Files: files}
	Models = append(Models, m)
	return m
}
