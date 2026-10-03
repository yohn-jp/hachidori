// Package home owns the HACHIDORI_HOME layout, manifests and the explicitly
// constructed worker environment.
package home

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Subdirectories of HACHIDORI_HOME (architecture §6.2).
// tools/ holds Hachidori-managed materializer tooling (the pinned private uv
// and the CPython installations it manages).
//
// workers/ holds the Hachidori worker scripts delivered by application builds,
// each in a directory named by its own digest. A worker is delivered beside
// the dependency runtimes, never inside one: a runtime is the immutable
// executable environment and does not carry the worker that runs in it.
var Dirs = []string{"runtime", "workers", "tools", "packages", "models", "variants", "cache", "logs", "state"}

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
	for _, d := range []string{"tmp", "home", "huggingface", "torch", "pip", "uv", "xdg", "nv"} {
		if err := os.MkdirAll(h.Path("cache", d), 0o755); err != nil {
			return err
		}
	}
	probe := h.Path("state", ".write-probe")
	if err := os.WriteFile(probe, []byte("ok"), 0o644); err != nil {
		return fmt.Errorf("home not writable: %w", err)
	}
	if err := os.Remove(probe); err != nil {
		return err
	}
	return h.EnsureEvaluationResources()
}

// Active is state/active-runtime.json: the activation record. The execution
// target is the source model plus, optionally, one immutable variant of it.
type Active struct {
	Runtime string `json:"runtime"`            // directory name under runtime/
	ModelID string `json:"model_id,omitempty"` // catalog model identity (absent in records from before model selection)
	Model   string `json:"model"`              // directory under models/, slash separated
	Device  string `json:"device"`             // cuda | cpu
	// Variant is the ID of the variant of ModelID that executes instead of
	// the source artifact. Absent in every record written before variants
	// existed, where it means the upstream source artifact.
	Variant string `json:"variant,omitempty"`
	// Experimental marks a variant activated without an accepted
	// certification record, by an explicit operator request. It is never set
	// implicitly and is reported as experimental/uncertified in status.
	Experimental bool `json:"experimental,omitempty"`
}

// RuntimeSpec is the declarative desired state of a private dependency
// runtime: the immutable executable environment (interpreter, torch/CUDA
// flavor, locked packages, kernels). Its canonical encoding determines the
// runtime identity: any semantic change yields a different identity and
// therefore a different immutable runtime directory. The locked package set
// is identified by the digests of the uv project files it was materialized
// from.
//
// The spec deliberately does not contain the Hachidori worker script. The
// worker is delivered by the application build and is observed separately
// (see WorkerABI): a worker-only change leaves the runtime identity, and so
// the materialized environment, unchanged.
type RuntimeSpec struct {
	Schema   string `json:"schema"`
	Role     string `json:"role,omitempty"` // "" serving runtime | optimizer runtime
	Platform string `json:"platform"`       // GOOS/GOARCH
	Python   string `json:"python"`         // exact CPython version
	Provider string `json:"provider"`       // name==version of every model provider (optimizer: engine) comma separated
	Torch    string `json:"torch"`          // exact torch version including local flavor
	Flavor   string `json:"flavor"`         // uv extra selecting the torch build: cu128 | cpu
	UV       string `json:"uv"`             // pinned private uv version
	UVSHA256 string `json:"uv_sha256"`      // pinned uv executable digest for Platform
	Project  string `json:"project_sha256"` // runtimespec/pyproject.toml
	Lock     string `json:"lock_sha256"`    // runtimespec/uv.lock
	// WorkerABI is the worker/runtime compatibility boundary: the version of
	// the environment contract the worker scripts of this runtime role require
	// of the interpreter and packages around them. It is bumped by hand when a
	// worker needs an environment that no runtime of the previous contract
	// provides, and only then; it is not derived from worker source.
	WorkerABI string `json:"worker_abi"`
}

// RoleOptimizer is the Role of the optimizer runtime: the separate, bounded
// environment that builds variants. It never serves.
const RoleOptimizer = "optimizer"

// Worker ABI contracts: one per runtime role. A build embeds the ABI its
// worker requires; a runtime records the ABI it was derived for.
const (
	WorkerABIServing   = "hachidori.worker-runtime/1"
	WorkerABIOptimizer = "hachidori.optimizer-runtime/1"
)

// WorkerABIFor is the ABI a build's worker requires of a runtime of role.
func WorkerABIFor(role string) string {
	if role == RoleOptimizer {
		return WorkerABIOptimizer
	}
	return WorkerABIServing
}

// ProviderPins lists the pinned model providers (name==version) the runtime
// carries.
func (s RuntimeSpec) ProviderPins() []string { return strings.Split(s.Provider, ",") }

// Provides reports whether the runtime carries the named model provider.
func (s RuntimeSpec) Provides(name string) bool {
	for _, p := range s.ProviderPins() {
		if n, _, _ := strings.Cut(p, "=="); n == name {
			return true
		}
	}
	return false
}

// ID is the runtime identity: the flavor plus a digest of the canonical
// (field-ordered JSON) encoding of the spec.
func (s RuntimeSpec) ID() string {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return idOf(s.Role, s.Flavor, b)
}

func idOf(role, flavor string, canonical []byte) string {
	sum := sha256.Sum256(canonical)
	prefix := flavor
	if role != "" {
		prefix = role + "-" + flavor
	}
	return prefix + "-" + hex.EncodeToString(sum[:8])
}

// Differences names, in field order, what differs between two specs. It is the
// evidence behind a reconciliation: an empty result means the specs are the
// same dependency environment.
func (s RuntimeSpec) Differences(want RuntimeSpec) []string {
	var d []string
	add := func(name, have, need string) {
		if have != need {
			d = append(d, name)
		}
	}
	add("schema", s.Schema, want.Schema)
	add("role", s.Role, want.Role)
	add("platform", s.Platform, want.Platform)
	add("python", s.Python, want.Python)
	add("provider", s.Provider, want.Provider)
	add("torch", s.Torch, want.Torch)
	add("flavor", s.Flavor, want.Flavor)
	add("uv", s.UV, want.UV)
	add("uv_sha256", s.UVSHA256, want.UVSHA256)
	add("project_sha256", s.Project, want.Project)
	add("lock_sha256", s.Lock, want.Lock)
	add("worker_abi", s.WorkerABI, want.WorkerABI)
	return d
}

// SchemaV1 is the Runtime Spec schema of runtimes materialized before the
// dependency environment was separated from the worker. Their identity
// included the digest of the worker script of the build that created them.
const SchemaV1 = "hachidori.runtime-spec/1"

// LegacyRuntimeSpec is the Runtime Spec of schema SchemaV1: today's RuntimeSpec
// plus the worker script digest that was part of the identity. It exists only
// to verify the integrity of such a runtime's manifest and to derive the
// dependency environment it materially is; nothing is ever materialized from
// it.
type LegacyRuntimeSpec struct {
	Schema   string `json:"schema"`
	Role     string `json:"role,omitempty"`
	Platform string `json:"platform"`
	Python   string `json:"python"`
	Provider string `json:"provider"`
	Torch    string `json:"torch"`
	Flavor   string `json:"flavor"`
	UV       string `json:"uv"`
	UVSHA256 string `json:"uv_sha256"`
	Project  string `json:"project_sha256"`
	Lock     string `json:"lock_sha256"`
	Worker   string `json:"worker_sha256"`
}

// ID is the legacy runtime identity, derived exactly as it was when the
// runtime was materialized.
func (l LegacyRuntimeSpec) ID() string {
	b, err := json.Marshal(l)
	if err != nil {
		panic(err)
	}
	return idOf(l.Role, l.Flavor, b)
}

// Environment is the dependency environment a legacy runtime materially is:
// its spec without the worker digest, under the worker ABI every runtime of
// its role carried before the ABI was declared. It is a statement about the
// runtime's declared contract; whether the installed environment really is that
// is established by verifying the runtime, never by this derivation.
func (l LegacyRuntimeSpec) Environment() RuntimeSpec {
	return RuntimeSpec{Schema: SpecSchema, Role: l.Role, Platform: l.Platform, Python: l.Python, Provider: l.Provider,
		Torch: l.Torch, Flavor: l.Flavor, UV: l.UV, UVSHA256: l.UVSHA256, Project: l.Project, Lock: l.Lock,
		WorkerABI: WorkerABIFor(l.Role)}
}

// SpecSchema versions the Runtime Spec contract and its identity derivation.
const SpecSchema = "hachidori.runtime-spec/2"

// RuntimeManifest is runtime/<identity>/manifest.json, written once when the
// runtime is materialized and verified. Its presence marks a complete runtime.
type RuntimeManifest struct {
	Identity      string      `json:"identity"`
	Spec          RuntimeSpec `json:"spec"`
	PythonVersion string      `json:"python_version"` // verified
	PythonRelPath string      `json:"python"`         // relative to runtime dir, slash separated
	BasePython    string      `json:"base_python"`    // uv-managed CPython prefix under tools/
	Installed     []string    `json:"installed"`      // verified distributions, name==version
	// Worker is the worker script a runtime of schema SchemaV1 carried inside
	// itself (relpath -> sha256). Current runtimes carry none.
	Worker map[string]string `json:"worker,omitempty"`
	// Legacy is the schema-1 spec this manifest was read with, when it is one.
	// It is never written.
	Legacy *LegacyRuntimeSpec `json:"-"`
}

// UnmarshalJSON reads a manifest of either spec schema. A schema-1 manifest
// keeps its own spec in Legacy so its identity can still be checked, and its
// Spec holds the schema-1 fields (it never equals a desired spec: the desired
// spec is of the current schema).
func (m *RuntimeManifest) UnmarshalJSON(b []byte) error {
	type plain RuntimeManifest
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*m = RuntimeManifest(p)
	if m.Spec.Schema == SchemaV1 {
		var raw struct {
			Spec LegacyRuntimeSpec `json:"spec"`
		}
		if err := json.Unmarshal(b, &raw); err != nil {
			return err
		}
		m.Legacy = &raw.Spec
	}
	return nil
}

// Environment is the dependency environment the manifest declares: its Spec,
// or, for a schema-1 manifest, the environment its legacy spec materially is.
func (m RuntimeManifest) Environment() RuntimeSpec {
	if m.Legacy != nil {
		return m.Legacy.Environment()
	}
	return m.Spec
}

// EnvironmentID is the dependency runtime identity of the manifest's
// environment. For a current runtime it is the directory name; for a legacy
// runtime it is the identity the same environment has under the current
// scheme, which is what lets a worker-only upgrade recognise it.
func (m RuntimeManifest) EnvironmentID() string { return m.Environment().ID() }

// Satisfies reports whether the manifest declares exactly the dependency
// environment spec describes.
func (m RuntimeManifest) Satisfies(spec RuntimeSpec) bool {
	return m.Environment() == spec
}

// ErrRuntimeIdentityMissing marks a runtime manifest that carries neither a
// declarative identity nor a Runtime Spec: the shape of a runtime written
// before declarative setup existed. Its contents cannot be derived from the
// current Runtime Spec, so it is not trusted and not completed in place;
// `hachidori setup` materializes the current runtime beside it.
var ErrRuntimeIdentityMissing = errors.New("manifest has no declarative Runtime Spec identity " +
	"(a runtime from before declarative setup is never trusted or modified in place; " +
	"run `hachidori setup` to materialize the current runtime, which reuses valid models)")

// CheckIdentity is the runtime-validity authority for an activated runtime:
// the manifest must carry the identity named by the activation record, and
// that identity must be the one its own Runtime Spec derives (for a schema-1
// manifest, the one its own legacy spec derives). A missing identity (with no
// Spec) is reported as ErrRuntimeIdentityMissing; any other disagreement,
// including a non-empty wrong identity, is corruption.
func (m RuntimeManifest) CheckIdentity(runtime string) error {
	if m.Identity == "" && m.Spec == (RuntimeSpec{}) {
		return ErrRuntimeIdentityMissing
	}
	own := m.Spec.ID()
	if m.Legacy != nil {
		own = m.Legacy.ID()
	}
	if m.Identity != runtime || own != m.Identity {
		return fmt.Errorf("manifest identity %q does not match its Runtime Spec (not materialized by declarative setup; run `hachidori setup`)", m.Identity)
	}
	return nil
}

// ModelManifest is an immutable model identity: an entry of the model
// catalog, and, once materialized, models/<repo>/<revision>/hachidori-model.json.
// Manifests written before the catalog existed carry no ID or provider.
type ModelManifest struct {
	ID          string            `json:"id,omitempty"`          // stable Hachidori model ID
	Provider    string            `json:"provider,omitempty"`    // provider kind that loads it (laya | opendecider)
	Repo        string            `json:"repo"`                  // upstream repository
	Revision    string            `json:"revision"`              // immutable upstream revision
	Description string            `json:"description,omitempty"` // descriptive only
	Files       map[string]string `json:"files"`                 // relpath -> sha256
}

// ReadJSON decodes a JSON file from a path selected by Hachidori's
// filesystem authorities. Callers must not pass an unvalidated external path.
func ReadJSON(path string, v any) error {
	// #nosec G304 -- path is not a public/user-controlled file selector. It is
	// constructed by Home.Path or another repository-owned filesystem
	// authority; those authorities validate any external identity components
	// before this generic JSON decoder is reached.
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
	return WriteFileAtomic(path, append(b, '\n'), 0o644)
}

// LoadActive reads the activation record and both manifests. It is the one
// definition of a valid active runtime shared by activation, serve, the
// desktop and doctor: the runtime manifest must satisfy CheckIdentity.
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
	if err := rm.CheckIdentity(a.Runtime); err != nil {
		return a, rm, mm, fmt.Errorf("runtime %s: %w", a.Runtime, err)
	}
	if err := ReadJSON(h.ModelDir(a)+string(filepath.Separator)+"hachidori-model.json", &mm); err != nil {
		return a, rm, mm, fmt.Errorf("model %s: invalid manifest: %w", a.Model, err)
	}
	return a, rm, mm, nil
}

// ReadActiveRecord returns the activation record exactly as stored, for a
// caller that must be able to put it back byte for byte.
func (h Home) ReadActiveRecord() ([]byte, error) {
	return os.ReadFile(h.Path("state", "active-runtime.json"))
}

// RestoreActiveRecord atomically puts back an activation record previously read
// with ReadActiveRecord. It is the same atomic write the activation uses; it
// exists so that a failed transaction restores exactly the record it replaced.
func (h Home) RestoreActiveRecord(raw []byte) error {
	return WriteFileAtomic(h.Path("state", "active-runtime.json"), raw, 0o644)
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

// WorkerDir is the directory a delivered worker script lives in: workers/
// <digest of the script>/. A delivered worker is immutable and named by its
// content.
func (h Home) WorkerDir(digest string) string { return h.Path("workers", digest) }

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
