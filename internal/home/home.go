// Package home owns the HACHIDORI_HOME layout, manifests and the explicitly
// constructed worker environment.
package home

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Subdirectories of HACHIDORI_HOME (architecture §6.2).
var Dirs = []string{"runtime", "packages", "models", "cache", "logs", "state"}

// Home is a resolved HACHIDORI_HOME.
type Home struct{ Root string }

// Resolve picks the home from an explicit flag value or HACHIDORI_HOME.
func Resolve(flag string) (Home, error) {
	root := flag
	if root == "" {
		root = os.Getenv("HACHIDORI_HOME")
	}
	if root == "" {
		return Home{}, errors.New("HACHIDORI_HOME is not set; pass --home or set HACHIDORI_HOME")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return Home{}, err
	}
	return Home{Root: filepath.Clean(abs)}, nil
}

// Path joins elements under the home root.
func (h Home) Path(elem ...string) string {
	return filepath.Join(append([]string{h.Root}, elem...)...)
}

// Ensure creates the layout and checks it is writable.
func (h Home) Ensure() error {
	for _, d := range Dirs {
		if err := os.MkdirAll(h.Path(d), 0o755); err != nil {
			return err
		}
	}
	for _, d := range []string{"tmp", "home", "huggingface", "torch", "pip", "xdg", "nv"} {
		if err := os.MkdirAll(h.Path("cache", d), 0o755); err != nil {
			return err
		}
	}
	probe := h.Path("state", ".write-probe")
	if err := os.WriteFile(probe, []byte("ok"), 0o644); err != nil {
		return fmt.Errorf("home not writable: %w", err)
	}
	return os.Remove(probe)
}

// Active is state/active-runtime.json: the activation record.
type Active struct {
	Runtime string `json:"runtime"` // directory name under runtime/
	Model   string `json:"model"`   // directory under models/, slash separated
	Device  string `json:"device"`  // cuda | cpu
}

// RuntimeManifest is runtime/<version>/manifest.json, written once at materialization.
type RuntimeManifest struct {
	Version        string            `json:"version"`
	Flavor         string            `json:"flavor"`
	Platform       string            `json:"platform"`
	PythonVersion  string            `json:"python_version"`
	PythonArchive  Artifact          `json:"python_archive"`
	PythonRelPath  string            `json:"python"` // relative to runtime dir, slash separated
	Packages       []string          `json:"packages"`
	PackageIndexes []string          `json:"package_indexes"`
	Installed      []string          `json:"installed"` // pip freeze after install
	Worker         map[string]string `json:"worker"`    // relpath -> sha256
}

// ModelManifest is models/<id>/<revision>/hachidori-model.json.
type ModelManifest struct {
	Repo     string            `json:"repo"`
	Revision string            `json:"revision"`
	Files    map[string]string `json:"files"` // relpath -> sha256
}

// Artifact is a pinned downloadable artifact.
type Artifact struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
}

// ReadJSON decodes a JSON file.
func ReadJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// WriteJSON writes a JSON file atomically.
func WriteJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadActive reads the activation record and both manifests.
func (h Home) LoadActive() (Active, RuntimeManifest, ModelManifest, error) {
	var a Active
	var rm RuntimeManifest
	var mm ModelManifest
	if err := ReadJSON(h.Path("state", "active-runtime.json"), &a); err != nil {
		return a, rm, mm, fmt.Errorf("no active runtime (run `hachidori setup`): %w", err)
	}
	if err := ReadJSON(h.Path("runtime", a.Runtime, "manifest.json"), &rm); err != nil {
		return a, rm, mm, fmt.Errorf("runtime %s: invalid manifest: %w", a.Runtime, err)
	}
	if err := ReadJSON(h.ModelDir(a)+string(filepath.Separator)+"hachidori-model.json", &mm); err != nil {
		return a, rm, mm, fmt.Errorf("model %s: invalid manifest: %w", a.Model, err)
	}
	return a, rm, mm, nil
}

// RuntimeDir is the active runtime's directory.
func (h Home) RuntimeDir(a Active) string { return h.Path("runtime", a.Runtime) }

// ModelDir is the active model's directory.
func (h Home) ModelDir(a Active) string {
	return h.Path(append([]string{"models"}, strings.Split(a.Model, "/")...)...)
}

// PythonExe is the absolute path of the private interpreter.
func (h Home) PythonExe(a Active, rm RuntimeManifest) string {
	return filepath.Join(h.RuntimeDir(a), filepath.FromSlash(rm.PythonRelPath))
}

// WorkerScript is the absolute path of the materialized worker.
func (h Home) WorkerScript(a Active) string {
	return filepath.Join(h.RuntimeDir(a), "worker", "hachidori_worker.py")
}

// Env constructs the complete environment for the private Python process.
// Nothing from the parent environment is inherited except OS essentials
// needed to load system libraries and the host GPU driver.
func (h Home) Env(pythonDir string, offline bool) []string {
	c := func(p ...string) string { return h.Path(append([]string{"cache"}, p...)...) }
	env := []string{
		"HACHIDORI_HOME=" + h.Root,
		"PYTHONNOUSERSITE=1",
		"PYTHONDONTWRITEBYTECODE=1",
		"PYTHONUNBUFFERED=1",
		"PYTHONUTF8=1",
		"PYTHONIOENCODING=utf-8",
		"PATH=" + pythonDir,
		"HOME=" + c("home"),
		"TMPDIR=" + c("tmp"),
		"TMP=" + c("tmp"),
		"TEMP=" + c("tmp"),
		"XDG_CACHE_HOME=" + c("xdg"),
		"HF_HOME=" + c("huggingface"),
		"HF_HUB_CACHE=" + c("huggingface", "hub"),
		"HF_XET_CACHE=" + c("huggingface", "xet"),
		"HF_HUB_DISABLE_TELEMETRY=1",
		"TORCH_HOME=" + c("torch"),
		"TORCHINDUCTOR_CACHE_DIR=" + c("torch", "inductor"),
		"TRITON_CACHE_DIR=" + c("torch", "triton"),
		"CUDA_CACHE_PATH=" + c("nv"),
		"PIP_CACHE_DIR=" + c("pip"),
		"PIP_CONFIG_FILE=" + os.DevNull,
		"PIP_DISABLE_PIP_VERSION_CHECK=1",
		"PIP_NO_INPUT=1",
	}
	if offline {
		env = append(env, "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1")
	}
	pass := []string{"CUDA_VISIBLE_DEVICES", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"}
	if runtime.GOOS == "windows" {
		sys := os.Getenv("SystemRoot")
		env = append(env,
			"PATH="+pythonDir+string(os.PathListSeparator)+filepath.Join(sys, "System32"),
			"USERPROFILE="+c("home"),
			"APPDATA="+c("home", "AppData", "Roaming"),
			"LOCALAPPDATA="+c("home", "AppData", "Local"))
		pass = append(pass, "SystemRoot", "SystemDrive", "WINDIR", "COMSPEC", "PATHEXT",
			"NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")
	} else {
		// Host GPU driver libraries may live outside the default loader path
		// (for example /run/opengl-driver/lib on NixOS). The driver is a host
		// prerequisite, so its location is taken from the host.
		pass = append(pass, "LD_LIBRARY_PATH")
	}
	for _, k := range pass {
		if v, ok := os.LookupEnv(k); ok {
			env = append(env, k+"="+v)
		}
	}
	return dedupe(env)
}

// dedupe keeps the last value for each key, preserving first-seen order.
func dedupe(env []string) []string {
	idx := map[string]int{}
	var out []string
	for _, kv := range env {
		k := kv[:strings.IndexByte(kv, '=')]
		if runtime.GOOS == "windows" {
			k = strings.ToUpper(k)
		}
		if i, ok := idx[k]; ok {
			out[i] = kv
			continue
		}
		idx[k] = len(out)
		out = append(out, kv)
	}
	return out
}
