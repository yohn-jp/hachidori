package setup

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

// Run reconciles HACHIDORI_HOME with the desired Runtime Spec for device and
// the catalog model modelID (empty selects DefaultModel):
//
//	desired spec -> private uv -> runtime identity
//	  -> reuse the verified immutable runtime, or materialize it into staging and verify it
//	  -> materialize/verify the selected model independently
//	  -> publish the runtime (atomic rename) -> update state/active-runtime.json
//
// Any failure leaves the active runtime and model untouched; rerunning
// reconciles again.
func Run(h home.Home, device, modelID string, log io.Writer) error {
	return RunObserved(h, device, modelID, log, nil)
}

// Phase is a real setup boundary reported by RunObserved. There is no
// percentage-based synthetic progress.
type Phase string

const (
	PhasePreparing  Phase = "preparing"
	PhaseRuntime    Phase = "runtime"
	PhaseModel      Phase = "model"
	PhaseActivation Phase = "activation"
)

// RunObserved is Run with an optional synchronous phase callback. It preserves
// the same materialization/verification/activation semantics as Run.
func RunObserved(h home.Home, device, modelID string, log io.Writer, onPhase func(Phase)) error {
	a, err := reconcile(h, device, modelID, log, onPhase, PhaseActivation)
	if err != nil {
		return err
	}
	return writeActive(h, a, log)
}

// PhasePublish is the real boundary at which Materialize publishes a
// verified runtime. Materialize never activates.
const PhasePublish Phase = "publish"

// Materialize is Run without activation: it materializes and verifies the
// runtime for device and the catalog model modelID through the same staged
// and verified path, publishes them, and leaves state/active-runtime.json
// untouched. Activation is the separate, explicit Activate.
func Materialize(h home.Home, device, modelID string, log io.Writer, onPhase func(Phase)) error {
	_, err := reconcile(h, device, modelID, log, onPhase, PhasePublish)
	return err
}

// reconcile materializes and verifies the desired runtime and model and
// publishes the runtime, entering publishPhase before the publish. It never
// writes the activation record; it returns the record that would activate
// exactly what it materialized.
func reconcile(h home.Home, device, modelID string, log io.Writer, onPhase func(Phase), publishPhase Phase) (home.Active, error) {
	enter := func(p Phase) {
		if onPhase != nil {
			onPhase(p)
		}
	}
	enter(PhasePreparing)
	spec, model, err := choose(device, modelID)
	if err != nil {
		return home.Active{}, err
	}
	if err := h.Ensure(); err != nil {
		return home.Active{}, err
	}
	uv, err := ensureUV(h, log)
	if err != nil {
		return home.Active{}, fmt.Errorf("private uv: %w", err)
	}
	enter(PhaseRuntime)
	id := spec.ID()
	final := h.Path("runtime", id)
	stage := ""
	if _, err := os.Stat(final); err == nil {
		if err := verifyPublished(h, final, spec); err != nil {
			return home.Active{}, fmt.Errorf("runtime %s exists but failed verification; it is never modified in place (repair it, or remove %s to rematerialize): %w", id, final, err)
		}
		fmt.Fprintf(log, "runtime %s verified, reusing\n", id)
	} else if !errors.Is(err, os.ErrNotExist) {
		return home.Active{}, err
	} else if stage, err = materializeRuntime(h, uv, spec, log); err != nil {
		return home.Active{}, fmt.Errorf("runtime %s: %w", id, err)
	}
	enter(PhaseModel)
	if err := materializeModel(h, model, log); err != nil {
		return home.Active{}, fmt.Errorf("model %s: %w", model.ID, err)
	}
	enter(publishPhase)
	if stage != "" {
		if err := os.Rename(stage, final); err != nil {
			return home.Active{}, fmt.Errorf("runtime %s: publish: %w", id, err)
		}
		if err := verifyPublished(h, final, spec); err != nil {
			return home.Active{}, fmt.Errorf("runtime %s: published runtime failed verification: %w", id, err)
		}
		fmt.Fprintf(log, "runtime %s published\n", id)
	}
	return home.Active{Runtime: id, ModelID: model.ID, Model: ModelDirName(model), Device: device}, nil
}

// choose resolves an explicit device and catalog model to the desired
// runtime spec and model. Only catalog identities resolve; a device is
// never substituted for another.
func choose(device, modelID string) (home.RuntimeSpec, home.ModelManifest, error) {
	spec, err := Desired(device)
	if err != nil {
		return spec, home.ModelManifest{}, err
	}
	model, err := LookupModel(modelID)
	if err != nil {
		return spec, model, err
	}
	if model.Provider != providerName {
		return spec, model, fmt.Errorf("model %s: provider %q is not supported by runtime provider %s", model.ID, model.Provider, spec.Provider)
	}
	return spec, model, nil
}

func writeActive(h home.Home, a home.Active, log io.Writer) error {
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), a); err != nil {
		return err
	}
	fmt.Fprintf(log, "active: runtime=%s model=%s (%s) device=%s\n", a.Runtime, a.ModelID, a.Model, a.Device)
	return nil
}

// pythonRelPath is the private interpreter inside a runtime directory.
func pythonRelPath() string {
	if runtime.GOOS == "windows" {
		return "env/Scripts/python.exe"
	}
	return "env/bin/python"
}

// materializeRuntime lets the private uv materialize spec into a fresh
// staging directory and verifies the result. It returns the staging path; the
// runtime becomes valid only once Run publishes it.
func materializeRuntime(h home.Home, uv uvTool, spec home.RuntimeSpec, log io.Writer) (string, error) {
	stage := h.Path("runtime", ".staging-"+spec.ID())
	if err := os.RemoveAll(stage); err != nil {
		return "", err
	}
	specDir := filepath.Join(stage, "spec")
	if err := os.MkdirAll(specDir, 0o755); err != nil {
		return "", err
	}
	for _, name := range []string{"pyproject.toml", "uv.lock"} {
		if err := os.WriteFile(filepath.Join(specDir, name), specFile(name), 0o644); err != nil {
			return "", err
		}
	}
	envDir := filepath.Join(stage, "env")
	fmt.Fprintf(log, "materializing runtime %s (python %s, %s, torch %s) with private uv %s\n",
		spec.ID(), spec.Python, spec.Provider, spec.Torch, spec.UV)
	// uv acquires the CPython build it pins for this version into
	// tools/uv/<version>/python; no system interpreter is considered.
	if err := uv.run(stage, uv.env(), "python", "install", spec.Python, "--no-bin", "--no-registry"); err != nil {
		return "", err
	}
	if err := uv.run(stage, uv.env(), "venv", "--relocatable", "--no-python-downloads", "--python", spec.Python, envDir); err != nil {
		return "", err
	}
	if err := uv.run(specDir, uv.env("UV_PROJECT_ENVIRONMENT="+envDir),
		"sync", "--locked", "--no-build", "--no-install-project", "--no-python-downloads", "--extra", spec.Flavor); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Join(stage, "worker"), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(stage, "worker", "hachidori_worker.py"), py.Script, 0o644); err != nil {
		return "", err
	}
	p, err := verifyRuntime(h, stage, spec)
	if err != nil {
		return "", fmt.Errorf("verification: %w", err)
	}
	m := home.RuntimeManifest{
		Identity: spec.ID(), Spec: spec, PythonVersion: p.Python, PythonRelPath: pythonRelPath(),
		BasePython: p.BasePrefix, Installed: p.Installed,
		Worker: map[string]string{"worker/hachidori_worker.py": spec.Worker},
	}
	// The manifest is written last: its presence marks a complete runtime.
	if err := home.WriteJSON(filepath.Join(stage, "manifest.json"), m); err != nil {
		return "", err
	}
	return stage, nil
}

// verifyPublished checks that dir is a complete runtime for exactly spec.
func verifyPublished(h home.Home, dir string, spec home.RuntimeSpec) error {
	var m home.RuntimeManifest
	if err := home.ReadJSON(filepath.Join(dir, "manifest.json"), &m); err != nil {
		return fmt.Errorf("incomplete runtime: %w", err)
	}
	if m.Identity != spec.ID() || m.Spec != spec {
		return fmt.Errorf("manifest identity %q does not match desired spec %s", m.Identity, spec.ID())
	}
	if m.PythonRelPath != pythonRelPath() {
		return fmt.Errorf("unexpected interpreter path %q", m.PythonRelPath)
	}
	_, err := verifyRuntime(h, dir, spec)
	return err
}

type runtimeProbe struct {
	Python     string   `json:"python"`
	Prefix     string   `json:"prefix"`
	BasePrefix string   `json:"base_prefix"`
	Installed  []string `json:"installed"`
}

// runtimeProbeCode reports the interpreter and installed distributions
// without importing the ML stack.
const runtimeProbeCode = `import json, sys, importlib.metadata as md
print(json.dumps({"python": "%d.%d.%d" % sys.version_info[:3], "prefix": sys.prefix, "base_prefix": sys.base_prefix,
 "installed": sorted({"%s==%s" % (d.metadata["Name"], d.version) for d in md.distributions()})}))`

// verifyRuntime is Hachidori's own check of a materialized environment: the
// worker digest, and, from the private interpreter run isolated and offline,
// the exact Python, provider and torch versions and that the environment and
// its base interpreter live under HACHIDORI_HOME.
func verifyRuntime(h home.Home, dir string, spec home.RuntimeSpec) (runtimeProbe, error) {
	var p runtimeProbe
	if got, err := FileSHA256(filepath.Join(dir, "worker", "hachidori_worker.py")); err != nil || got != spec.Worker {
		return p, fmt.Errorf("worker script digest mismatch")
	}
	python := filepath.Join(dir, filepath.FromSlash(pythonRelPath()))
	if _, err := os.Stat(python); err != nil {
		return p, fmt.Errorf("private python missing: %w", err)
	}
	cmd := exec.Command(python, "-I", "-c", runtimeProbeCode)
	cmd.Env = h.Env(filepath.Dir(python), true)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return p, fmt.Errorf("private python probe: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	if err := json.Unmarshal(out, &p); err != nil {
		return p, fmt.Errorf("private python probe: unexpected output %q", strings.TrimSpace(string(out)))
	}
	if p.Python != spec.Python {
		return p, fmt.Errorf("python %s, want %s", p.Python, spec.Python)
	}
	if !within(filepath.Join(dir, "env"), p.Prefix) {
		return p, fmt.Errorf("interpreter prefix %s is not the runtime environment", p.Prefix)
	}
	if !within(filepath.Join(uvDir(h), "python"), p.BasePrefix) {
		return p, fmt.Errorf("base interpreter %s is not the Hachidori-managed CPython", p.BasePrefix)
	}
	have := map[string]bool{}
	for _, d := range p.Installed {
		have[normalizeDist(d)] = true
	}
	for _, want := range []string{spec.Provider, "torch==" + spec.Torch} {
		if !have[normalizeDist(want)] {
			return p, fmt.Errorf("%s not installed", want)
		}
	}
	return p, nil
}

func normalizeDist(d string) string {
	name, ver, _ := strings.Cut(d, "==")
	return strings.ReplaceAll(strings.ToLower(name), "_", "-") + "==" + ver
}

// materializeModel materializes the catalog model m independently of the
// Python runtime. A present model is reused only if every file still matches
// its pinned digest.
func materializeModel(h home.Home, m home.ModelManifest, log io.Writer) error {
	final := h.Path("models", filepath.FromSlash(ModelDirName(m)))
	if _, err := os.Stat(filepath.Join(final, "hachidori-model.json")); err == nil {
		if err := VerifyModel(final, m); err != nil {
			return fmt.Errorf("%s exists but failed verification: %w", final, err)
		}
		fmt.Fprintf(log, "model %s (%s@%s) verified, reusing\n", m.ID, m.Repo, m.Revision[:12])
		return nil
	}
	stage := final + ".staging"
	if err := os.RemoveAll(stage); err != nil {
		return err
	}
	for rel, want := range m.Files {
		url := modelBaseURL + m.Repo + "/resolve/" + m.Revision + "/" + rel
		if err := fetch(url, filepath.Join(stage, filepath.FromSlash(rel)), want, log); err != nil {
			return err
		}
	}
	if err := home.WriteJSON(filepath.Join(stage, "hachidori-model.json"), m); err != nil {
		return err
	}
	return os.Rename(stage, final)
}

// VerifyModel checks that dir holds exactly the catalog model m: its
// manifest names m's repository and revision (and ID, when recorded) with
// m's digests, and every file matches its pinned digest.
func VerifyModel(dir string, m home.ModelManifest) error {
	var mm home.ModelManifest
	if err := home.ReadJSON(filepath.Join(dir, "hachidori-model.json"), &mm); err != nil {
		return err
	}
	if mm.Repo != m.Repo || mm.Revision != m.Revision || (mm.ID != "" && mm.ID != m.ID) {
		return fmt.Errorf("model manifest %s %s@%s is not catalog model %s (%s@%s)", mm.ID, mm.Repo, mm.Revision, m.ID, m.Repo, m.Revision)
	}
	for rel, want := range m.Files {
		if mm.Files[rel] != want {
			return fmt.Errorf("model manifest does not match pinned digest for %s", rel)
		}
		got, err := FileSHA256(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("%s: sha256 %s, want %s", rel, got, want)
		}
	}
	return nil
}

// fetch downloads url to dst and verifies its SHA-256. A present file with
// the right digest is reused.
func fetch(url, dst, want string, log io.Writer) error {
	if got, err := FileSHA256(dst); err == nil && got == want {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	fmt.Fprintf(log, "downloading %s\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	hash := sha256.New()
	_, err = io.Copy(io.MultiWriter(f, hash), resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != want {
		os.Remove(tmp)
		return fmt.Errorf("%s: sha256 mismatch: got %s want %s", url, got, want)
	}
	return os.Rename(tmp, dst)
}

// FileSHA256 hashes a file.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func within(root, p string) bool {
	rel, err := filepath.Rel(root, p)
	return err == nil && filepath.IsAbs(p) && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
