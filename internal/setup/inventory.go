package setup

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Artifact kinds addressed by Verify and Remove.
const (
	KindRuntime = "runtime"
	KindModel   = "model"
)

// Devices lists the runtime flavors a catalog choice may request, in display
// order. A device is always explicit: nothing substitutes one for the other.
var Devices = []string{"cuda", "cpu"}

// Inventory is the read-only, typed view of what HACHIDORI_HOME holds for the
// catalog-pinned runtime and model identities. Artifacts that are not
// catalog identities are neither listed nor manageable.
type Inventory struct {
	Active    *home.Active   `json:"active,omitempty"` // the activation record, when one is readable
	ActiveErr string         `json:"active_error,omitempty"`
	Runtimes  []RuntimeEntry `json:"runtimes"`
	Models    []ModelEntry   `json:"models"`
}

// RuntimeEntry is one supported runtime identity (one per device).
type RuntimeEntry struct {
	ID           string `json:"id"` // runtime identity (directory under runtime/)
	Device       string `json:"device"`
	Platform     string `json:"platform"`
	Python       string `json:"python"`
	Provider     string `json:"provider"`
	Torch        string `json:"torch"`
	Supported    bool   `json:"supported"`    // this platform has a pinned materializer for it
	Materialized bool   `json:"materialized"` // a manifest for exactly this identity is present
	Verified     bool   `json:"verified"`     // full verification passed (only when requested)
	Active       bool   `json:"active"`
	Problem      string `json:"problem,omitempty"`
}

// ModelEntry is one catalog model identity.
type ModelEntry struct {
	ID           string `json:"id"`
	Provider     string `json:"provider"`
	Repo         string `json:"repo"`
	Revision     string `json:"revision"`
	Description  string `json:"description,omitempty"`
	Files        int    `json:"files"`
	Materialized bool   `json:"materialized"` // manifest matches the catalog and every pinned file is present
	Verified     bool   `json:"verified"`     // every file matches its pinned digest (only when requested)
	Active       bool   `json:"active"`
	Problem      string `json:"problem,omitempty"`
}

// Inspect reports the inventory of h. It only reads: it never creates or
// changes anything and performs no network access. With verify it also runs
// the full verification of each materialized artifact (the private
// interpreter probe and the file digests), which is expensive.
func Inspect(h home.Home, verify bool) Inventory {
	var inv Inventory
	var a home.Active
	switch err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); {
	case err == nil:
		inv.Active = &a
	case !errors.Is(err, fs.ErrNotExist):
		inv.ActiveErr = err.Error()
	}
	activeModel := ""
	if inv.Active != nil {
		if m, err := ActiveModel(a); err == nil {
			activeModel = m.ID
		}
	}

	for _, device := range Devices {
		e := RuntimeEntry{Device: device}
		spec, err := Desired(device)
		if err != nil {
			e.Problem = err.Error()
			inv.Runtimes = append(inv.Runtimes, e)
			continue
		}
		e.ID, e.Platform, e.Python, e.Provider, e.Torch, e.Supported = spec.ID(), spec.Platform, spec.Python, spec.Provider, spec.Torch, true
		e.Active = inv.Active != nil && a.Runtime == e.ID
		dir := h.Path("runtime", e.ID)
		if _, err := os.Stat(dir); err == nil {
			var m home.RuntimeManifest
			switch err := home.ReadJSON(filepath.Join(dir, "manifest.json"), &m); {
			case err != nil:
				e.Problem = "incomplete runtime: " + err.Error()
			case m.Identity != e.ID || m.Spec != spec:
				e.Problem = "manifest does not match the desired runtime spec"
			default:
				e.Materialized = true
				if verify {
					if err := verifyPublished(h, dir, spec); err != nil {
						e.Problem = err.Error()
					} else {
						e.Verified = true
					}
				}
			}
		}
		inv.Runtimes = append(inv.Runtimes, e)
	}

	for _, m := range Models {
		e := ModelEntry{ID: m.ID, Provider: m.Provider, Repo: m.Repo, Revision: m.Revision, Description: m.Description, Files: len(m.Files)}
		e.Active = activeModel == m.ID
		dir := h.Path("models", filepath.FromSlash(ModelDirName(m)))
		if _, err := os.Stat(dir); err == nil {
			if err := presentModel(dir, m); err != nil {
				e.Problem = err.Error()
			} else {
				e.Materialized = true
				if verify {
					if err := VerifyModel(dir, m); err != nil {
						e.Problem = err.Error()
					} else {
						e.Verified = true
					}
				}
			}
		}
		inv.Models = append(inv.Models, e)
	}
	return inv
}

// presentModel is the cheap check behind "materialized": the manifest names
// exactly the catalog identity and every pinned file exists.
func presentModel(dir string, m home.ModelManifest) error {
	var mm home.ModelManifest
	if err := home.ReadJSON(filepath.Join(dir, "hachidori-model.json"), &mm); err != nil {
		return fmt.Errorf("incomplete model: %w", err)
	}
	if mm.Repo != m.Repo || mm.Revision != m.Revision || (mm.ID != "" && mm.ID != m.ID) {
		return fmt.Errorf("manifest %s@%s is not catalog model %s", mm.Repo, mm.Revision, m.ID)
	}
	for rel, want := range m.Files {
		if mm.Files[rel] != want {
			return fmt.Errorf("manifest does not match pinned digest for %s", rel)
		}
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel))); err != nil {
			return fmt.Errorf("missing %s", rel)
		}
	}
	return nil
}

// runtimeDevice resolves a catalog runtime identity to its device.
func runtimeDevice(id string) (string, bool) {
	for _, d := range Devices {
		if s, err := Desired(d); err == nil && id != "" && s.ID() == id {
			return d, true
		}
	}
	return "", false
}

// modelByID is LookupModel for an explicit ID: the empty ID does not select
// the default.
func modelByID(id string) (home.ModelManifest, error) {
	if id == "" {
		return home.ModelManifest{}, errors.New("a catalog model ID is required")
	}
	return LookupModel(id)
}

// Verify fully re-verifies one materialized catalog artifact (offline).
func Verify(h home.Home, kind, id string) error {
	switch kind {
	case KindRuntime:
		device, ok := runtimeDevice(id)
		if !ok {
			return fmt.Errorf("runtime %q is not a supported catalog runtime", id)
		}
		spec, _ := Desired(device)
		dir := h.Path("runtime", id)
		if _, err := os.Stat(dir); err != nil {
			return fmt.Errorf("runtime %s is not materialized", id)
		}
		return verifyPublished(h, dir, spec)
	case KindModel:
		m, err := modelByID(id)
		if err != nil {
			return err
		}
		dir := h.Path("models", filepath.FromSlash(ModelDirName(m)))
		if _, err := os.Stat(dir); err != nil {
			return fmt.Errorf("model %s is not materialized", id)
		}
		return VerifyModel(dir, m)
	}
	return fmt.Errorf("unknown artifact kind %q", kind)
}

// Activate makes the already materialized catalog choice (device, modelID)
// the active runtime and model. It materializes nothing and touches no
// network: both artifacts must already be published and pass verification,
// and only then is state/active-runtime.json replaced (atomically). Any
// failure leaves the current activation exactly as it was. changed reports
// whether the activation record differs from the one it replaced.
func Activate(h home.Home, device, modelID string, log io.Writer) (changed bool, err error) {
	spec, m, err := choose(device, modelID)
	if err != nil {
		return false, err
	}
	dir := h.Path("runtime", spec.ID())
	if _, err := os.Stat(dir); err != nil {
		return false, fmt.Errorf("runtime %s (%s) is not materialized; materialize it first", spec.ID(), device)
	}
	if err := verifyPublished(h, dir, spec); err != nil {
		return false, fmt.Errorf("runtime %s failed verification: %w", spec.ID(), err)
	}
	mdir := h.Path("models", filepath.FromSlash(ModelDirName(m)))
	if _, err := os.Stat(mdir); err != nil {
		return false, fmt.Errorf("model %s is not materialized; materialize it first", m.ID)
	}
	if err := VerifyModel(mdir, m); err != nil {
		return false, fmt.Errorf("model %s failed verification: %w", m.ID, err)
	}
	next := home.Active{Runtime: spec.ID(), ModelID: m.ID, Model: ModelDirName(m), Device: device}
	var prev home.Active
	if home.ReadJSON(h.Path("state", "active-runtime.json"), &prev) == nil && prev == next {
		return false, nil
	}
	if err := writeActive(h, next, log); err != nil {
		return false, err
	}
	return true, nil
}

// Repair re-establishes the catalog choice (device, modelID) after a failed
// verification. An artifact that is present but fails verification is moved
// aside (never modified in place), Materialize rebuilds whatever is missing
// through the staged and verified path, and the moved artifact is discarded
// only once the rebuild succeeded; on any failure it is put back. The
// activation record is never written. The caller must ensure no worker is
// running from the artifact.
func Repair(h home.Home, device, modelID string, log io.Writer, onPhase func(Phase)) error {
	spec, m, err := choose(device, modelID)
	if err != nil {
		return err
	}
	type moved struct{ final, old string }
	var aside []moved
	restore := func() {
		for _, mv := range aside {
			_ = os.RemoveAll(mv.final)
			_ = os.Rename(mv.old, mv.final)
		}
	}
	setAside := func(final string, bad error) error {
		old := final + ".repair-old"
		if err := os.RemoveAll(old); err != nil {
			return err
		}
		if err := os.Rename(final, old); err != nil {
			return err
		}
		fmt.Fprintf(log, "%s failed verification (%v); rebuilding\n", final, bad)
		aside = append(aside, moved{final, old})
		return nil
	}
	rdir := h.Path("runtime", spec.ID())
	if _, err := os.Stat(rdir); err == nil {
		if bad := verifyPublished(h, rdir, spec); bad != nil {
			if err := setAside(rdir, bad); err != nil {
				restore()
				return err
			}
		}
	}
	mdir := h.Path("models", filepath.FromSlash(ModelDirName(m)))
	if _, err := os.Stat(mdir); err == nil {
		if bad := VerifyModel(mdir, m); bad != nil {
			if err := setAside(mdir, bad); err != nil {
				restore()
				return err
			}
		}
	}
	if err := Materialize(h, device, modelID, log, onPhase); err != nil {
		restore()
		return err
	}
	for _, mv := range aside {
		_ = os.RemoveAll(mv.old)
	}
	return nil
}

// ErrActive is returned by Remove for an artifact the activation record uses.
var ErrActive = errors.New("the active runtime/model cannot be removed")

// Remove deletes one materialized, unused, catalog-pinned artifact: a
// runtime by identity or a model by catalog ID. It refuses the active
// runtime and model, anything that is not a catalog identity, and any path
// that does not resolve to a real directory beneath HACHIDORI_HOME. The
// caller must ensure no worker is running from the artifact.
func Remove(h home.Home, kind, id string) error {
	var a home.Active
	hasActive := false
	switch err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); {
	case err == nil:
		hasActive = true
	case !errors.Is(err, fs.ErrNotExist):
		return fmt.Errorf("activation record is unreadable, so an unused artifact cannot be proven: %w", err)
	}
	var base, target string
	switch kind {
	case KindRuntime:
		if _, ok := runtimeDevice(id); !ok {
			return fmt.Errorf("runtime %q is not a supported catalog runtime", id)
		}
		if hasActive && a.Runtime == id {
			return fmt.Errorf("runtime %s: %w", id, ErrActive)
		}
		base, target = h.Path("runtime"), h.Path("runtime", id)
	case KindModel:
		m, err := modelByID(id)
		if err != nil {
			return err
		}
		if hasActive && (a.ModelID == m.ID || a.Model == ModelDirName(m)) {
			return fmt.Errorf("model %s: %w", id, ErrActive)
		}
		base, target = h.Path("models"), h.Path("models", filepath.FromSlash(ModelDirName(m)))
	default:
		return fmt.Errorf("unknown artifact kind %q", kind)
	}
	if err := removeConfined(h, base, target); err != nil {
		return err
	}
	if kind == KindModel {
		// Drop the now-empty <repo> directory; never a non-empty one.
		_ = os.Remove(filepath.Dir(target))
	}
	return nil
}

// removeConfined deletes target only if, after resolving symbolic links, both
// base and target lie beneath HACHIDORI_HOME, target is a real directory
// strictly inside base, and h is a plausible home (an absolute, non-root
// directory holding the layout).
func removeConfined(h home.Home, base, target string) error {
	root := h.Root
	if root == "" || !filepath.IsAbs(root) || filepath.Clean(root) != root || filepath.Dir(root) == root {
		return fmt.Errorf("refusing to remove under %q: not a Hachidori home", root)
	}
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return fmt.Errorf("home %s: %w", root, err)
	}
	for _, d := range []string{"runtime", "models", "state"} {
		if fi, err := os.Lstat(filepath.Join(root, d)); err != nil || !fi.IsDir() {
			return fmt.Errorf("refusing to remove under %s: not a Hachidori home (no %s/)", root, d)
		}
	}
	if !within(root, base) || !within(base, target) || filepath.Clean(base) == filepath.Clean(target) {
		return fmt.Errorf("refusing to remove %s: not beneath %s", target, base)
	}
	fi, err := os.Lstat(target)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("%s is not materialized", target)
		}
		return err
	}
	if !fi.IsDir() || fi.Mode()&fs.ModeSymlink != 0 {
		return fmt.Errorf("refusing to remove %s: not a real directory", target)
	}
	realBase, err := filepath.EvalSymlinks(base)
	if err != nil {
		return err
	}
	realTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		return err
	}
	if !within(realRoot, realBase) || !within(realBase, realTarget) || realBase == realTarget {
		return fmt.Errorf("refusing to remove %s: resolves outside %s", target, realRoot)
	}
	return os.RemoveAll(target)
}
