// Package harness is the runtime-lifecycle and recovery proof harness of the
// Windows appliance E2E certification (#245): a deterministic lightweight
// runtime/model fixture, a launcher for the real packaged hachidori.exe in a
// disposable profile, observers of its API, dashboard and processes, and the
// bounded-wait helper every scenario synchronizes with.
//
// What the fixture replaces, and what it does not. The fixture replaces only
// the heavy model payload: the private Python environment's "torch" and "laya"
// packages and the model weights are tiny deterministic stand-ins (stubs/).
// Everything that is certified is the production code path: the executable,
// the application controller, the supervisor, the worker script the executable
// itself delivers and runs (hachidori_worker.py), the private NDJSON protocol,
// the HTTP API and the dashboard. No production-scale download happens.
//
// The portable parts (fixture construction, stub rule, process-tree and page
// parsing, polling) are tested on Linux; only launching hachidori.exe, ending
// processes and Quit are Windows specific.
package harness

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

//go:embed all:stubs
var stubFS embed.FS

// EnvPython selects the host interpreter the fixture's private environment is
// created from. Without it python3 (python on Windows) on PATH is used.
const EnvPython = "HACHIDORI_E2E_PYTHON"

// Fixture is a disposable HACHIDORI_HOME that looks like the result of a
// finished setup and runs the real worker on stub torch/laya packages.
type Fixture struct {
	Home      home.Home
	Device    string
	RuntimeID string
	ModelID   string
	Python    string // the private interpreter inside the runtime
}

// HostPython finds the interpreter the fixture environment is created from and
// checks that it runs.
func HostPython() (string, error) {
	var candidates []string
	if p := os.Getenv(EnvPython); p != "" {
		candidates = []string{p}
	} else if runtime.GOOS == "windows" {
		candidates = []string{"python", "py"}
	} else {
		candidates = []string{"python3", "python"}
	}
	var last error
	for _, c := range candidates {
		path, err := exec.LookPath(c)
		if err != nil {
			last = err
			continue
		}
		args := []string{"-c", "import sys; assert sys.version_info >= (3, 9); print(sys.version_info[0])"}
		if filepath.Base(path) == "py.exe" || filepath.Base(path) == "py" {
			args = append([]string{"-3"}, args...)
		}
		if out, err := exec.Command(path, args...).CombinedOutput(); err != nil {
			last = fmt.Errorf("%s does not run as Python 3.9+: %v: %s", path, err, strings.TrimSpace(string(out)))
			continue
		}
		return path, nil
	}
	return "", fmt.Errorf("no usable host Python for the fixture environment (set %s): %v", EnvPython, last)
}

// privatePython is the interpreter path inside a runtime directory, as setup
// derives it.
func privatePython(runtimeDir string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(runtimeDir, "env", "Scripts", "python.exe")
	}
	return filepath.Join(runtimeDir, "env", "bin", "python")
}

// PythonRelPath is the runtime manifest's interpreter path.
func PythonRelPath() string {
	if runtime.GOOS == "windows" {
		return "env/Scripts/python.exe"
	}
	return "env/bin/python"
}

// NewHome builds a disposable home under root (created if needed) for device
// ("cpu" or "cuda") and the default catalog model: the private environment is a
// venv of hostPython carrying the stub packages, the runtime manifest, model
// manifest and activation record are written with the production types and
// identities (setup.Desired, setup.LookupModel), so the real launch gate and
// worker configuration accept them as they would a materialized home.
func NewHome(root, device, hostPython string) (Fixture, error) {
	spec, err := setup.Desired(device)
	if err != nil {
		return Fixture{}, err
	}
	model, err := setup.LookupModel(setup.DefaultModel)
	if err != nil {
		return Fixture{}, err
	}
	h := home.Home{Root: root}
	if err := h.Ensure(); err != nil {
		return Fixture{}, err
	}
	runtimeID := spec.ID()
	runtimeDir := h.Path("runtime", runtimeID)
	envDir := filepath.Join(runtimeDir, "env")
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		return Fixture{}, err
	}
	args := []string{"-m", "venv", "--without-pip", envDir}
	if filepath.Base(hostPython) == "py.exe" || filepath.Base(hostPython) == "py" {
		args = append([]string{"-3"}, args...)
	}
	if out, err := exec.Command(hostPython, args...).CombinedOutput(); err != nil {
		return Fixture{}, fmt.Errorf("create the fixture environment: %v: %s", err, strings.TrimSpace(string(out)))
	}
	python := privatePython(runtimeDir)
	site, err := runOut(python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))")
	if err != nil {
		return Fixture{}, fmt.Errorf("locate the fixture site-packages: %w", err)
	}
	if err := installStubs(site); err != nil {
		return Fixture{}, err
	}
	ver, err := runOut(python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])")
	if err != nil {
		return Fixture{}, fmt.Errorf("read the fixture interpreter version: %w", err)
	}

	m := home.RuntimeManifest{
		Identity: runtimeID, Spec: spec, PythonVersion: ver, PythonRelPath: PythonRelPath(),
		Installed: []string{"laya==" + stubVersion(site, "laya"), "torch==" + stubVersion(site, "torch")},
	}
	if err := home.WriteJSON(filepath.Join(runtimeDir, "manifest.json"), m); err != nil {
		return Fixture{}, err
	}
	modelDir := h.ModelDir(home.Active{Model: setup.ModelDirName(model)})
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return Fixture{}, err
	}
	// The model manifest is the catalog entry itself; the pinned weight files
	// are not materialized (the fixture's laya loads none).
	if err := home.WriteJSON(filepath.Join(modelDir, "hachidori-model.json"), model); err != nil {
		return Fixture{}, err
	}
	a := home.Active{Runtime: runtimeID, ModelID: model.ID, Model: setup.ModelDirName(model), Device: device}
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), a); err != nil {
		return Fixture{}, err
	}
	if _, _, _, err := h.LoadActive(); err != nil {
		return Fixture{}, fmt.Errorf("the fixture home is not a valid active runtime: %w", err)
	}
	return Fixture{Home: h, Device: device, RuntimeID: runtimeID, ModelID: model.ID, Python: python}, nil
}

func runOut(exe string, args ...string) (string, error) {
	var stderr bytes.Buffer
	cmd := exec.Command(exe, args...)
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%v: %s", err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(string(out)), nil
}

func installStubs(site string) error {
	return fs.WalkDir(stubFS, "stubs", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel := strings.TrimPrefix(strings.TrimPrefix(p, "stubs"), "/")
		if rel == "" {
			return nil
		}
		dst := filepath.Join(site, filepath.FromSlash(rel))
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		b, err := stubFS.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(dst, b, 0o644)
	})
}

// stubVersion reads the __version__ a stub package declares.
func stubVersion(site, pkg string) string {
	b, err := os.ReadFile(filepath.Join(site, pkg, "__init__.py"))
	if err != nil {
		return "unknown"
	}
	for _, line := range strings.Split(string(b), "\n") {
		if v, ok := strings.CutPrefix(line, "__version__ = "); ok {
			return strings.Trim(strings.TrimSpace(v), `"'`)
		}
	}
	return "unknown"
}

// PersistedState is the digest of the persisted files whose consistency every
// failure scenario asserts: the activation record, the runtime manifest and the
// model manifest. A failure that is "recovered" must leave these byte for byte
// as they were.
type PersistedState map[string]string

// persistedFiles lists the files, relative to the home, of a fixture.
func (f Fixture) persistedFiles() []string {
	model, _ := setup.LookupModel(f.ModelID)
	return []string{
		filepath.Join("state", "active-runtime.json"),
		filepath.Join("runtime", f.RuntimeID, "manifest.json"),
		filepath.Join("models", filepath.FromSlash(setup.ModelDirName(model)), "hachidori-model.json"),
	}
}

// Persisted reads the digest of every persisted file; a missing file is an
// error, never an empty digest.
func (f Fixture) Persisted() (PersistedState, error) {
	out := PersistedState{}
	for _, rel := range f.persistedFiles() {
		sum, _, err := fileSHA256(filepath.Join(f.Home.Root, rel))
		if err != nil {
			return nil, err
		}
		out[filepath.ToSlash(rel)] = sum
	}
	return out, nil
}

// Diff describes how b differs from a, "" when they are identical.
func (a PersistedState) Diff(b PersistedState) string {
	var d []string
	for k, v := range a {
		if b[k] != v {
			d = append(d, k)
		}
	}
	for k := range b {
		if _, ok := a[k]; !ok {
			d = append(d, k)
		}
	}
	if len(d) == 0 {
		return ""
	}
	sort.Strings(d)
	return "persisted state changed: " + strings.Join(d, ", ")
}

// Leftovers lists staging directories and partial files under the home that an
// interrupted materialization would leave behind. A fixture home has none, and
// no lifecycle scenario may create one.
func (f Fixture) Leftovers() ([]string, error) {
	var out []string
	err := filepath.WalkDir(f.Home.Root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if strings.Contains(name, ".staging") || strings.HasSuffix(name, ".part") || strings.HasSuffix(name, ".part.json") {
			rel, _ := filepath.Rel(f.Home.Root, p)
			out = append(out, filepath.ToSlash(rel))
			if d.IsDir() {
				return fs.SkipDir
			}
		}
		return nil
	})
	return out, err
}

// ReadJSONFile decodes a JSON file of the home.
func ReadJSONFile(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	return nil
}

// ErrDeadline is returned by Eventually when the condition did not hold in time.
var ErrDeadline = errors.New("deadline exceeded")

// Eventually polls cond every interval until it reports done, returns an error,
// or the deadline passes. It is the only way a scenario waits: on an observable
// condition with a bounded deadline, never on a fixed sleep as the oracle.
// cond returns (done, detail); detail describes what was last observed and is
// part of the deadline error.
func Eventually(timeout, interval time.Duration, what string, cond func() (bool, string, error)) error {
	deadline := time.Now().Add(timeout)
	last := ""
	for {
		done, detail, err := cond()
		if err != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		if done {
			return nil
		}
		last = detail
		if time.Now().After(deadline) {
			return fmt.Errorf("%s: %w after %s (last observed: %s)", what, ErrDeadline, timeout, last)
		}
		time.Sleep(interval)
	}
}

// fileSHA256 returns the lower-case hex SHA-256 and the size of a file.
func fileSHA256(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// FileSHA256 is the digest of a file, for scenarios that compare a delivered
// file with the digest it is named by.
func FileSHA256(path string) (string, error) {
	sum, _, err := fileSHA256(path)
	return sum, err
}

// Holds watches, for the bounded duration, that an invariant keeps holding: the
// observation of something that must NOT happen (a worker restarted by itself
// after Stop, recovery continuing after it gave up). The window is long enough
// to cover the event were it going to occur, and the condition is observed on
// every tick, so it is a bounded negative observation rather than a sleep.
func Holds(d, interval time.Duration, what string, inv func() (bool, string, error)) error {
	deadline := time.Now().Add(d)
	for {
		ok, detail, err := inv()
		if err != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		if !ok {
			return fmt.Errorf("%s: violated: %s", what, detail)
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(interval)
	}
}
