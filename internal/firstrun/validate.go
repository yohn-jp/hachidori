package firstrun

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/yohn-jp/hachidori/internal/home"
)

// HomeSubdir is the folder Hachidori creates inside a chosen storage root
// when that root is a non-empty folder that is not already a Hachidori home.
// The user still chooses only the root; this keeps Hachidori's runtime, model
// and cache directories from being mixed into unrelated files.
const HomeSubdir = "Hachidori"

// Installed is the identity of a valid active runtime found in a home.
type Installed struct {
	Runtime string `json:"runtime"`
	ModelID string `json:"model_id,omitempty"`
	Model   string `json:"model"`
	Device  string `json:"device"`
}

// Validation is the outcome of validating a picked storage root. Problem is
// non-empty when the location cannot be used; everything else is informative.
type Validation struct {
	Picked    string     `json:"picked"`
	Home      string     `json:"home,omitempty"`       // the effective HACHIDORI_HOME
	Nested    bool       `json:"nested,omitempty"`     // Home is Picked/Hachidori
	Exists    bool       `json:"exists"`               // Home exists already
	Existing  *Installed `json:"existing,omitempty"`   // a valid installation lives there
	FreeBytes *uint64    `json:"free_bytes,omitempty"` // only when the OS reports it
	Problem   string     `json:"problem,omitempty"`
}

// OK reports whether the location can be used.
func (v Validation) OK() bool { return v.Problem == "" && v.Home != "" }

// Env is the filesystem knowledge Validate needs; tests substitute fakes.
type Env struct {
	// FreeSpace reports the free bytes available to the user on the volume of
	// path. ok=false means unknown; nothing is then displayed or estimated.
	FreeSpace func(path string) (free uint64, ok bool)
	// Load reads a home's activation record (home.Home.LoadActive).
	Load func(root string) (home.Active, error)
}

func (e Env) load(root string) (home.Active, error) {
	if e.Load != nil {
		return e.Load(root)
	}
	a, _, _, err := home.Home{Root: root}.LoadActive()
	return a, err
}

func (e Env) installed(root string) *Installed {
	a, err := e.load(root)
	if err != nil {
		return nil
	}
	return &Installed{Runtime: a.Runtime, ModelID: a.ModelID, Model: a.Model, Device: a.Device}
}

// IsInstalled reports whether root has a valid activation record.
func (e Env) IsInstalled(root string) bool { return e.installed(root) != nil }

// Validate checks a storage root picked by the user without leaving anything
// behind: writability is probed with a temporary file that is removed, and the
// root itself is created only later, by the install action.
func Validate(picked string, env Env) Validation {
	v := Validation{Picked: picked}
	if picked == "" {
		v.Problem = "No folder selected."
		return v
	}
	if !filepath.IsAbs(picked) {
		v.Problem = "The location must be an absolute path."
		return v
	}
	picked = filepath.Clean(picked)
	v.Picked = picked

	fi, err := os.Stat(picked)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		v.Home = picked
		anc, ok := nearestDir(picked)
		if !ok {
			v.Problem = "The location is not on a usable folder or drive."
			return v
		}
		return finish(v, anc, env)
	case err != nil:
		v.Problem = "The location cannot be accessed: " + err.Error()
		return v
	case !fi.IsDir():
		v.Problem = "The location is a file, not a folder."
		return v
	}

	switch kind, err := classify(picked); {
	case err != nil:
		v.Problem = "The folder cannot be read: " + err.Error()
		return v
	case kind == kindForeign:
		nested := filepath.Join(picked, HomeSubdir)
		v.Home, v.Nested = nested, true
		nfi, err := os.Stat(nested)
		switch {
		case errors.Is(err, fs.ErrNotExist):
			v.Exists = false
		case err != nil:
			v.Problem = "The location cannot be accessed: " + err.Error()
			return v
		case !nfi.IsDir():
			v.Problem = fmt.Sprintf("%s exists and is a file.", nested)
			return v
		default:
			k, err := classify(nested)
			if err != nil {
				v.Problem = "The folder cannot be read: " + err.Error()
				return v
			}
			if k == kindForeign {
				v.Problem = fmt.Sprintf("%s exists, is not empty and is not a Hachidori home. Choose another folder.", nested)
				return v
			}
			v.Exists = true
		}
	default:
		v.Home, v.Exists = picked, true
	}
	return finish(v, nearestExisting(v), env)
}

func nearestExisting(v Validation) string {
	if v.Exists {
		return v.Home
	}
	if v.Nested {
		return v.Picked
	}
	return v.Home
}

func finish(v Validation, probeDir string, env Env) Validation {
	if v.Exists {
		v.Existing = env.installed(v.Home)
	}
	if err := probeWritable(probeDir); err != nil {
		v.Problem = "Hachidori cannot write to this location: " + err.Error()
		return v
	}
	if env.FreeSpace != nil {
		if free, ok := env.FreeSpace(probeDir); ok {
			v.FreeBytes = &free
		}
	}
	return v
}

type dirKind int

const (
	kindEmpty   dirKind = iota // empty directory
	kindHome                   // looks like a Hachidori home
	kindForeign                // non-empty, not a Hachidori home
)

// classify tells an empty folder, an existing Hachidori home (recognised by
// its state and runtime directories) and unrelated content apart.
func classify(dir string) (dirKind, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return kindForeign, err
	}
	if len(ents) == 0 {
		return kindEmpty, nil
	}
	for _, sub := range []string{"state", "runtime"} {
		fi, err := os.Stat(filepath.Join(dir, sub))
		if err != nil || !fi.IsDir() {
			return kindForeign, nil
		}
	}
	return kindHome, nil
}

// nearestDir returns the closest existing ancestor directory of path.
func nearestDir(path string) (string, bool) {
	for p := filepath.Dir(path); ; p = filepath.Dir(p) {
		fi, err := os.Stat(p)
		if err == nil {
			return p, fi.IsDir()
		}
		if !errors.Is(err, fs.ErrNotExist) || filepath.Dir(p) == p {
			return "", false
		}
	}
}

func probeWritable(dir string) error {
	f, err := os.CreateTemp(dir, ".hachidori-probe-*")
	if err != nil {
		return err
	}
	name := f.Name()
	f.Close()
	return os.Remove(name)
}
