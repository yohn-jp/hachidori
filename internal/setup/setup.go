package setup

import (
	"context"
	"crypto/sha256"
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
	"github.com/yohn-jp/hachidori/internal/subprocess"
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

// Phase is a real setup boundary reported by RunObserved. Progress inside a
// phase is reported separately (Progress); a percentage exists only where a
// total is actually known.
type Phase string

const (
	PhasePreparing  Phase = "preparing"
	PhaseRuntime    Phase = "runtime"
	PhaseModel      Phase = "model"
	PhaseActivation Phase = "activation"
)

// RunObserved is Run with an optional Observer of its phases and progress. It
// preserves the same materialization/verification/activation semantics as Run.
func RunObserved(h home.Home, device, modelID string, log io.Writer, obs *Observer) error {
	return RunContext(context.Background(), h, device, modelID, log, obs)
}

// RunContext is RunObserved that ctx can cancel. A cancelled model download
// keeps its resumable partial; nothing is activated.
func RunContext(ctx context.Context, h home.Home, device, modelID string, log io.Writer, obs *Observer) error {
	a, err := reconcile(ctx, h, device, modelID, log, obs, PhaseActivation)
	if err != nil {
		return err
	}
	obs.step(StepActivate, "activation record")
	return writeActive(h, a, log)
}

// PhasePublish is the real boundary at which Materialize publishes a
// verified runtime. Materialize never activates.
const PhasePublish Phase = "publish"

// Materialize is Run without activation: it materializes and verifies the
// runtime for device and the catalog model modelID through the same staged
// and verified path, publishes them, and leaves state/active-runtime.json
// untouched. Activation is the separate, explicit Activate.
func Materialize(h home.Home, device, modelID string, log io.Writer, obs *Observer) error {
	return MaterializeContext(context.Background(), h, device, modelID, log, obs)
}

// MaterializeContext is Materialize that ctx can cancel. A cancelled model
// download keeps its resumable partial (see fetchResumable); nothing else is
// left behind and nothing is published.
func MaterializeContext(ctx context.Context, h home.Home, device, modelID string, log io.Writer, obs *Observer) error {
	_, err := reconcile(ctx, h, device, modelID, log, obs, PhasePublish)
	return err
}

// reconcile materializes and verifies the desired runtime and model and
// publishes the runtime, entering publishPhase before the publish. It never
// writes the activation record; it returns the record that would activate
// exactly what it materialized.
func reconcile(ctx context.Context, h home.Home, device, modelID string, log io.Writer, obs *Observer, publishPhase Phase) (home.Active, error) {
	enter := obs.phase
	enter(PhasePreparing)
	spec, model, err := choose(device, modelID)
	if err != nil {
		return home.Active{}, err
	}
	if err := h.Ensure(); err != nil {
		return home.Active{}, err
	}
	uv, err := ensureUV(h, log, obs)
	if err != nil {
		return home.Active{}, fmt.Errorf("private uv: %w", err)
	}
	enter(PhaseRuntime)
	// The dependency runtime is found by its identity, or as a runtime of the
	// previous identity scheme that declares exactly this environment: either
	// is verified and reused, never rematerialized.
	id := RuntimeDirFor(h, spec)
	final := h.Path("runtime", id)
	stage := ""
	if _, err := os.Stat(final); err == nil {
		obs.step(StepVerify, "runtime "+id)
		if err := verifyPublished(h, final, spec); err != nil {
			return home.Active{}, fmt.Errorf("runtime %s exists but failed verification; it is never modified in place (repair it, or remove %s to rematerialize): %w", id, final, err)
		}
		if id != spec.ID() {
			fmt.Fprintf(log, "runtime %s (previous identity scheme) declares the required dependency runtime %s, verified, reusing\n", id, spec.ID())
		} else {
			fmt.Fprintf(log, "runtime %s verified, reusing\n", id)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return home.Active{}, err
	} else if stage, err = materializeRuntime(h, uv, spec, log, obs); err != nil {
		return home.Active{}, fmt.Errorf("runtime %s: %w", id, err)
	}
	enter(PhaseModel)
	if err := materializeModel(ctx, h, model, log, obs); err != nil {
		return home.Active{}, fmt.Errorf("model %s: %w", model.ID, err)
	}
	enter(publishPhase)
	if stage != "" {
		obs.step(StepPublish, "runtime "+id)
		if err := os.Rename(stage, final); err != nil {
			return home.Active{}, fmt.Errorf("runtime %s: publish: %w", id, err)
		}
		obs.step(StepVerify, "published runtime "+id)
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
	if !spec.Provides(model.Provider) {
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
func materializeRuntime(h home.Home, uv uvTool, spec home.RuntimeSpec, log io.Writer, obs *Observer) (string, error) {
	stage := h.Path("runtime", ".staging-"+spec.ID())
	if err := os.RemoveAll(stage); err != nil {
		return "", err
	}
	specDir := filepath.Join(stage, "spec")
	if err := os.MkdirAll(specDir, 0o755); err != nil {
		return "", err
	}
	kind := kindOf(spec)
	for _, name := range []string{"pyproject.toml", "uv.lock"} {
		if err := os.WriteFile(filepath.Join(specDir, name), kind.file(name), 0o644); err != nil {
			return "", err
		}
	}
	envDir := filepath.Join(stage, "env")
	fmt.Fprintf(log, "materializing runtime %s (python %s, %s, torch %s) with private uv %s\n",
		spec.ID(), spec.Python, spec.Provider, spec.Torch, spec.UV)
	// uv acquires the CPython build it pins for this version into
	// tools/uv/<version>/python; no system interpreter is considered.
	// uv reports no measurable total, so each step is indeterminate and only
	// names what it is doing.
	obs.step(StepMaterialize, "installing Python "+spec.Python)
	if err := uv.run(stage, uv.env(), "python", "install", spec.Python, "--no-bin", "--no-registry"); err != nil {
		return "", err
	}
	obs.step(StepMaterialize, "creating the private environment")
	if err := uv.run(stage, uv.env(), "venv", "--relocatable", "--no-python-downloads", "--python", spec.Python, envDir); err != nil {
		return "", err
	}
	obs.step(StepMaterialize, "installing the locked packages ("+spec.Flavor+")")
	if err := uv.run(specDir, uv.env("UV_PROJECT_ENVIRONMENT="+envDir),
		"sync", "--locked", "--no-build", "--no-install-project", "--no-python-downloads", "--extra", spec.Flavor); err != nil {
		return "", err
	}
	obs.step(StepVerify, "runtime "+spec.ID())
	p, err := verifyRuntime(h, stage, spec)
	if err != nil {
		return "", fmt.Errorf("verification: %w", err)
	}
	m := home.RuntimeManifest{
		Identity: spec.ID(), Spec: spec, PythonVersion: p.Python, PythonRelPath: pythonRelPath(),
		BasePython: p.BasePrefix, Installed: p.Installed,
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
	if err := CheckRuntimeManifest(filepath.Base(dir), m, spec); err != nil {
		return err
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

// verifyRuntime is Hachidori's own check of a materialized environment: from
// the private interpreter run isolated and offline, the exact Python, provider
// and torch versions and that the environment and its base interpreter live
// under HACHIDORI_HOME. The worker is not part of the environment and is not
// checked here (see DeliverWorker).
func verifyRuntime(h home.Home, dir string, spec home.RuntimeSpec) (runtimeProbe, error) {
	var p runtimeProbe
	python := filepath.Join(dir, filepath.FromSlash(pythonRelPath()))
	if _, err := os.Stat(python); err != nil {
		return p, fmt.Errorf("private python missing: %w", err)
	}
	cmd := exec.Command(python, "-I", "-c", runtimeProbeCode)
	subprocess.Configure(cmd)
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
	var required []string
	for _, pin := range spec.ProviderPins() {
		if name, _, _ := strings.Cut(pin, "=="); carriedProviders[name] {
			required = append(required, clefDistributions...)
			required = append(required, clefKernelDistributions(spec)...)
			continue
		}
		required = append(required, pin)
	}
	for _, want := range append(required, "torch=="+spec.Torch) {
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
func materializeModel(ctx context.Context, h home.Home, m home.ModelManifest, log io.Writer, obs *Observer) error {
	final := h.Path("models", filepath.FromSlash(ModelDirName(m)))
	if _, err := os.Stat(filepath.Join(final, "hachidori-model.json")); err == nil {
		if err := verifyModel(final, m, obs); err != nil {
			return fmt.Errorf("%s exists but failed verification: %w", final, err)
		}
		fmt.Fprintf(log, "model %s (%s@%s) verified, reusing\n", m.ID, m.Repo, m.Revision[:12])
		return nil
	}
	// The staging directory is the resumable partial of this model: files
	// already complete and verified, and an interrupted file's partial. It is
	// never a materialized model (it has no manifest, and only the atomic
	// rename below publishes anything). Whatever is not part of this model's
	// pinned file set is removed.
	stage := final + ".staging"
	rels := sortedFiles(m.Files)
	if err := pruneStage(stage, rels); err != nil {
		return err
	}
	for i, rel := range rels {
		url := ModelFileURL(m, rel)
		at := Progress{Step: StepDownload, Detail: rel, Item: i + 1, Items: len(rels)}
		if err := fetchResumable(ctx, url, filepath.Join(stage, filepath.FromSlash(rel)), m.Files[rel], log, obs, at); err != nil {
			return err
		}
	}
	obs.step(StepPublish, "model "+m.ID)
	if err := home.WriteJSON(filepath.Join(stage, "hachidori-model.json"), m); err != nil {
		return err
	}
	return os.Rename(stage, final)
}

// pruneStage keeps in a model's staging directory only what may be continued:
// the pinned files (complete or not) and their partial records. Anything else,
// including a manifest, is removed, so a published model holds exactly its
// pinned files.
func pruneStage(stage string, rels []string) error {
	keep := map[string]bool{}
	for _, rel := range rels {
		keep[filepath.ToSlash(rel)] = true
		keep[filepath.ToSlash(rel)+".part"] = true
		keep[filepath.ToSlash(rel)+".part.json"] = true
	}
	if err := os.MkdirAll(stage, 0o755); err != nil {
		return err
	}
	var drop []string
	err := filepath.WalkDir(stage, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, rerr := filepath.Rel(stage, p)
		if rerr != nil || rel == "." {
			return rerr
		}
		slash := filepath.ToSlash(rel)
		if d.IsDir() {
			for k := range keep {
				if strings.HasPrefix(k, slash+"/") {
					return nil // holds a pinned file
				}
			}
			drop = append(drop, p)
			return filepath.SkipDir
		}
		if !keep[slash] || !d.Type().IsRegular() {
			drop = append(drop, p)
		}
		return nil
	})
	if err != nil {
		return err
	}
	for _, p := range drop {
		if err := os.RemoveAll(p); err != nil {
			return err
		}
	}
	return nil
}

// PresentModel checks that dir holds the catalog model m's manifest and every
// pinned file, without hashing them.
func PresentModel(dir string, m home.ModelManifest) error { return presentModel(dir, m) }

// ModelFileURL is the pinned download URL of one file of the catalog model m.
func ModelFileURL(m home.ModelManifest, rel string) string {
	return modelBaseURL + m.Repo + "/resolve/" + m.Revision + "/" + rel
}

// sortedFiles lists a model's files in a fixed order, so progress is
// reported as "file i of n" in a stable sequence.
func sortedFiles(files map[string]string) []string {
	rels := make([]string, 0, len(files))
	for rel := range files {
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	return rels
}

// VerifyModel checks that dir holds exactly the catalog model m: its
// manifest names m's repository and revision (and ID, when recorded) with
// m's digests, and every file matches its pinned digest.
func VerifyModel(dir string, m home.ModelManifest) error { return verifyModel(dir, m, nil) }

// verifyModel is VerifyModel reporting each file's hashing as determinate
// byte progress (the file size is known).
func verifyModel(dir string, m home.ModelManifest, obs *Observer) error {
	var mm home.ModelManifest
	if err := home.ReadJSON(filepath.Join(dir, "hachidori-model.json"), &mm); err != nil {
		return err
	}
	if mm.Repo != m.Repo || mm.Revision != m.Revision || (mm.ID != "" && mm.ID != m.ID) {
		return fmt.Errorf("model manifest %s %s@%s is not catalog model %s (%s@%s)", mm.ID, mm.Repo, mm.Revision, m.ID, m.Repo, m.Revision)
	}
	rels := sortedFiles(m.Files)
	for i, rel := range rels {
		want := m.Files[rel]
		if mm.Files[rel] != want {
			return fmt.Errorf("model manifest does not match pinned digest for %s", rel)
		}
		got, err := fileSHA256Observed(filepath.Join(dir, filepath.FromSlash(rel)), obs,
			Progress{Step: StepVerify, Detail: rel, Item: i + 1, Items: len(rels)})
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("%s: sha256 %s, want %s", rel, got, want)
		}
	}
	return nil
}

// downloadStall is how long a download may go without receiving a byte
// (waiting for the response headers counts) before it is abandoned. It is a
// progress bound, not a total one: a large model that keeps arriving is never
// cut off. Connecting and the TLS handshake are already bounded by
// http.DefaultTransport.
var downloadStall = 2 * time.Minute

// progressReader restarts the stall watchdog whenever data arrives.
type progressReader struct {
	r io.Reader
	w *time.Timer
}

func (p progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	if n > 0 {
		p.w.Reset(downloadStall)
	}
	return n, err
}

// FileSHA256 hashes a file.
func FileSHA256(path string) (string, error) { return fileSHA256Observed(path, nil, Progress{}) }

// fileSHA256Observed hashes a file, reporting determinate byte progress
// (the file's size is the total) under at when obs observes it.
func fileSHA256Observed(path string, obs *Observer, at Progress) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	var w io.Writer = h
	var counter *byteCounter
	if obs != nil && obs.OnProgress != nil {
		if fi, err := f.Stat(); err == nil {
			at.Total = fi.Size()
		}
		counter = newByteCounter(obs, at)
		w = io.MultiWriter(h, counter)
	}
	if _, err := io.Copy(w, f); err != nil {
		return "", err
	}
	if counter != nil {
		counter.flush()
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func within(root, p string) bool {
	rel, err := filepath.Rel(root, p)
	return err == nil && filepath.IsAbs(p) && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
