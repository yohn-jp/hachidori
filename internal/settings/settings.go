// Package settings is the typed application-settings authority behind the
// Settings workspace. It composes the existing desktop preference authority
// (the per-user startup entry and desktop.json, reached through Desktop) and
// owns only one new record: schema-versioned, non-secret runtime defaults.
//
// Runtime defaults are stored and read with no side effects. Saving them never
// touches the running worker, the active model, setup or the bootstrap
// locator; they are only the values a later, explicit setup may propose.
// Explicit CLI flags and environment inputs always win (see Defaults.Resolve).
package settings

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Schema versions the settings record.
const Schema = "hachidori.settings/1"

// Inference devices a default may name (the same set as the --device flag).
const (
	DeviceCUDA = "cuda"
	DeviceCPU  = "cpu"
)

// Desktop is the existing desktop preference authority (desktop.Manager).
type Desktop interface {
	Prefs() (startAtSignIn, startMinimized bool, err error)
	Set(startAtSignIn, startMinimized bool) error
}

// Defaults are the saved runtime defaults. An empty field means "no saved
// default". They hold no secrets.
type Defaults struct {
	Device string `json:"device,omitempty"` // cuda | cpu
	Model  string `json:"model,omitempty"`  // catalog model ID
}

// Validate reports whether d names only a supported device and catalog model.
func (d Defaults) Validate() error {
	switch d.Device {
	case "", DeviceCUDA, DeviceCPU:
	default:
		return fmt.Errorf("device %q is not supported (cuda or cpu)", d.Device)
	}
	if d.Model != "" {
		if _, err := setup.LookupModel(d.Model); err != nil {
			return err
		}
	}
	return nil
}

// Resolve applies explicit input over the saved defaults: a non-empty explicit
// value (a CLI flag or environment input the caller actually received) always
// wins; the saved default fills only what was not given.
func (d Defaults) Resolve(explicitDevice, explicitModel string) Defaults {
	if explicitDevice != "" {
		d.Device = explicitDevice
	}
	if explicitModel != "" {
		d.Model = explicitModel
	}
	return d
}

// Models lists the catalog model IDs a default may select.
func Models() []string {
	ids := make([]string, 0, len(setup.Models))
	for _, m := range setup.Models {
		ids = append(ids, m.ID)
	}
	return ids
}

// record is the on-disk document. Field order is fixed, so the same values
// always produce the same bytes.
type record struct {
	Schema          string   `json:"schema"`
	RuntimeDefaults Defaults `json:"runtime_defaults"`
}

// Store is the settings authority. An empty Path keeps defaults in memory.
type Store struct {
	// Path is settings.json beside the desktop preferences file.
	Path string
	// Desktop is the desktop preference authority; nil means none.
	Desktop Desktop

	mu  sync.Mutex
	mem Defaults
}

// Defaults reads the saved runtime defaults. A missing file is not an error;
// an unreadable one yields no defaults plus the error.
func (s *Store) Defaults() (Defaults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.load()
}

func (s *Store) load() (Defaults, error) {
	if s.Path == "" {
		return s.mem, nil
	}
	var r record
	err := home.ReadJSON(s.Path, &r)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return Defaults{}, nil
	case err != nil:
		return Defaults{}, fmt.Errorf("settings %s: %w", s.Path, err)
	case r.Schema != Schema:
		return Defaults{}, fmt.Errorf("settings %s: unknown schema %q", s.Path, r.Schema)
	}
	if err := r.RuntimeDefaults.Validate(); err != nil {
		return Defaults{}, fmt.Errorf("settings %s: %w", s.Path, err)
	}
	return r.RuntimeDefaults, nil
}

// SetDefaults validates and stores the runtime defaults. It changes nothing
// else: no worker, model, setup or startup state is read or written.
func (s *Store) SetDefaults(d Defaults) error {
	if err := d.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Path == "" {
		s.mem = d
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return err
	}
	return home.WriteJSON(s.Path, record{Schema: Schema, RuntimeDefaults: d})
}

// Prefs reports the desktop preferences through the desktop authority.
func (s *Store) Prefs() (startAtSignIn, startMinimized bool, err error) {
	if s.Desktop == nil {
		return false, false, nil
	}
	return s.Desktop.Prefs()
}

// Set applies the desktop preferences through the desktop authority.
func (s *Store) Set(startAtSignIn, startMinimized bool) error {
	if s.Desktop == nil {
		return errors.New("desktop preferences are not available")
	}
	return s.Desktop.Set(startAtSignIn, startMinimized)
}
