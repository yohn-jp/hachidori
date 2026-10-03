// Package settings is the typed application-settings authority behind the
// Settings workspace. It composes the existing desktop preference authority
// (the per-user startup entry and desktop.json, reached through Desktop) and
// owns only one new record: schema-versioned, non-secret runtime defaults.
//
// It also persists named, non-secret Development Connection profiles (see
// Connection); the live tunnel stays in tunnel.Manager.
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
	"regexp"
	"sort"
	"sync"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/update"
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

// NormalizeResidents validates the additional resident selection and returns
// it in catalog order without duplicates. Only stable Hachidori catalog model
// IDs are accepted; there is no way to name a repository or revision. The
// active model is not special-cased here: it is always the default resident
// whatever is selected, and selecting it again adds nothing.
func NormalizeResidents(ids []string) ([]string, error) {
	want := make(map[string]bool, len(ids))
	for _, id := range ids {
		if id == "" {
			return nil, errors.New("a resident model must be named by its catalog ID")
		}
		if _, err := setup.LookupModel(id); err != nil {
			return nil, err
		}
		want[id] = true
	}
	var out []string
	for _, id := range Models() {
		if want[id] {
			out = append(out, id)
		}
	}
	return out, nil
}

// ValidateLocale accepts an unset locale ("") or a supported one (en, ja).
func ValidateLocale(l string) error {
	if l != "" && !i18n.Valid(l) {
		return fmt.Errorf("locale %q is not supported (en or ja)", l)
	}
	return nil
}

// TransportMode records whether a connection transport value is automatically
// resolved or explicitly pinned by the operator.
type TransportMode string

const (
	ConnectionAuto   TransportMode = "auto"
	ConnectionPinned TransportMode = "pinned"
)

// Connection is one named Development Connection profile: a declarative,
// non-secret description of the SSH reverse tunnel that carries a development
// host's loopback HACHIDORI_ENDPOINT to this host's loopback API. Transport
// values retain their Auto/pinned intent; Resolve produces the concrete
// tunnel.Spec used by tunnel.Manager. Live connection state stays in
// tunnel.Manager. There is no field for keys, passwords, agents or known_hosts.
type Connection struct {
	Name           string        `json:"name"`
	Destination    string        `json:"destination"`
	RemoteBind     string        `json:"remote_bind"`
	RemoteBindMode TransportMode `json:"remote_bind_mode,omitempty"`
	RemotePort     int           `json:"remote_port"`
	RemotePortMode TransportMode `json:"remote_port_mode,omitempty"`
	LocalPort      int           `json:"local_port"`
	LocalPortMode  TransportMode `json:"local_port_mode,omitempty"`
	localEndpoint  tunnel.LocalEndpoint
}

// MaxConnections bounds the saved profile list.
const MaxConnections = 32

var connectionNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// ResolvedConnection keeps the stored operator intent separate from the
// concrete, validated tunnel spec used by the live manager.
type ResolvedConnection struct {
	Intent Connection  `json:"intent"`
	Spec   tunnel.Spec `json:"spec"`
}

var defaultLocalEndpoint = tunnel.LocalEndpoint{Host: tunnel.DefaultRemoteBind, Port: tunnel.DefaultPort}

// Resolve interprets legacy values without a mode as pinned, resolves Auto
// against the current managed loopback API endpoint, and validates the full
// concrete spec before it can be passed to tunnel.Manager.
func (c Connection) Resolve(endpoint tunnel.LocalEndpoint) (ResolvedConnection, error) {
	c, err := c.normalized()
	if err != nil {
		return ResolvedConnection{}, err
	}
	if !connectionNameRe.MatchString(c.Name) {
		return ResolvedConnection{}, fmt.Errorf("invalid connection name %q: use letters, digits, '.', '_' or '-' (up to 64, starting with a letter or digit)", c.Name)
	}
	if err := endpoint.Validate(); err != nil {
		return ResolvedConnection{}, fmt.Errorf("resolve connection %q: %w", c.Name, err)
	}
	if c.LocalPortMode == ConnectionAuto && endpoint.Host != "127.0.0.1" {
		return ResolvedConnection{}, fmt.Errorf("resolve connection %q: local endpoint host %q cannot be represented by the tunnel spec's 127.0.0.1 target", c.Name, endpoint.Host)
	}

	spec := tunnel.Spec{Destination: c.Destination}
	if c.RemoteBindMode == ConnectionAuto {
		spec.RemoteBind = tunnel.DefaultRemoteBind
	} else {
		spec.RemoteBind = c.RemoteBind
	}
	if c.RemotePortMode == ConnectionAuto {
		spec.RemotePort = endpoint.Port
	} else {
		spec.RemotePort = c.RemotePort
	}
	if c.LocalPortMode == ConnectionAuto {
		spec.LocalPort = endpoint.Port
	} else {
		spec.LocalPort = c.LocalPort
	}
	if err := spec.Validate(); err != nil {
		return ResolvedConnection{}, err
	}
	return ResolvedConnection{Intent: c, Spec: spec}, nil
}

// Spec is the concrete tunnel description used by the dashboard's existing
// connection seam. Settings attaches the current endpoint to profiles it
// returns; standalone and legacy callers resolve against the safe default.
func (c Connection) Spec() tunnel.Spec {
	endpoint := c.localEndpoint
	if endpoint.Port == 0 {
		endpoint = defaultLocalEndpoint
	}
	resolved, err := c.Resolve(endpoint)
	if err != nil {
		return tunnel.Spec{}
	}
	return resolved.Spec
}

// Endpoint is the HACHIDORI_ENDPOINT value a development host uses.
func (c Connection) Endpoint() string { return c.Spec().CallerEndpoint() }

// Validate checks the intent and its resolution against the safe default
// endpoint. Store uses the actual managed endpoint when one is available.
func (c Connection) Validate() error {
	_, err := c.Resolve(defaultLocalEndpoint)
	return err
}

func (c Connection) normalized() (Connection, error) {
	var err error
	if c.RemoteBindMode, err = normalizeStringMode("remote bind", c.RemoteBindMode, c.RemoteBind); err != nil {
		return Connection{}, err
	}
	if c.RemotePortMode, err = normalizeIntMode("remote port", c.RemotePortMode, c.RemotePort); err != nil {
		return Connection{}, err
	}
	if c.LocalPortMode, err = normalizeIntMode("local port", c.LocalPortMode, c.LocalPort); err != nil {
		return Connection{}, err
	}
	return c, nil
}

func normalizeStringMode(name string, mode TransportMode, value string) (TransportMode, error) {
	switch mode {
	case "": // Existing settings stored explicit values before mode fields existed.
		if value == "" {
			return "", fmt.Errorf("%s is missing both a mode and a pinned value", name)
		}
		return ConnectionPinned, nil
	case ConnectionAuto:
		if value != "" {
			return "", fmt.Errorf("%s marked Auto also contains a pinned value", name)
		}
	case ConnectionPinned:
	default:
		return "", fmt.Errorf("%s has unknown mode %q", name, mode)
	}
	return mode, nil
}

func normalizeIntMode(name string, mode TransportMode, value int) (TransportMode, error) {
	switch mode {
	case "": // Existing settings stored explicit values before mode fields existed.
		if value == 0 {
			return "", fmt.Errorf("%s is missing both a mode and a pinned value", name)
		}
		return ConnectionPinned, nil
	case ConnectionAuto:
		if value != 0 {
			return "", fmt.Errorf("%s marked Auto also contains a pinned value", name)
		}
	case ConnectionPinned:
	default:
		return "", fmt.Errorf("%s has unknown mode %q", name, mode)
	}
	return mode, nil
}

// record is the on-disk document. Field order is fixed, so the same values
// always produce the same bytes. Connections is additive: a file without it
// (written before profiles existed) is the same schema with no profiles.
// Locale is additive in the same way: absent means no explicit selection.
// Updates is the update subsystem's channel, last explicit check and installed
// identity (internal/update); absent means Stable and nothing remembered.
// Residents is the desired set of additional resident catalog models for the
// next start; absent means only the active model is resident.
type record struct {
	Schema          string           `json:"schema"`
	RuntimeDefaults Defaults         `json:"runtime_defaults"`
	Connections     []Connection     `json:"connections,omitempty"`
	Locale          string           `json:"locale,omitempty"`
	Updates         *update.Settings `json:"updates,omitempty"`
	Residents       []string         `json:"resident_models,omitempty"`
}

// Store is the settings authority. An empty Path keeps its values in memory.
type Store struct {
	// Path is settings.json beside the desktop preferences file.
	Path string
	// Desktop is the desktop preference authority; nil means none.
	Desktop Desktop

	mu            sync.Mutex
	mem           record
	localEndpoint tunnel.LocalEndpoint
}

// SetLocalEndpoint supplies the currently bound loopback API endpoint used
// when a saved connection has Auto local or remote ports. It is process state,
// not a second persisted settings authority.
func (s *Store) SetLocalEndpoint(endpoint tunnel.LocalEndpoint) error {
	if err := endpoint.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.localEndpoint = endpoint
	return nil
}

func (s *Store) currentLocalEndpoint() tunnel.LocalEndpoint {
	if s.localEndpoint.Port == 0 {
		return defaultLocalEndpoint
	}
	return s.localEndpoint
}

// Defaults reads the saved runtime defaults. A missing file is not an error;
// an unreadable one yields no defaults plus the error.
func (s *Store) Defaults() (Defaults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.load()
	return r.RuntimeDefaults, err
}

// Connections lists the saved profiles ordered by name. A missing file is not
// an error; an unreadable one yields no profiles plus the error.
func (s *Store) Connections() ([]Connection, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.load()
	if err != nil {
		return nil, err
	}
	for i := range r.Connections {
		r.Connections[i].localEndpoint = s.currentLocalEndpoint()
	}
	return r.Connections, nil
}

func (s *Store) load() (record, error) {
	if s.Path == "" {
		r := s.mem
		r.Connections = append([]Connection(nil), r.Connections...)
		r.Updates = cloneUpdates(r.Updates)
		r.Residents = append([]string(nil), r.Residents...)
		return r, nil
	}
	var r record
	err := home.ReadJSON(s.Path, &r)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return record{}, nil
	case err != nil:
		return record{}, fmt.Errorf("settings %s: %w", s.Path, err)
	case r.Schema != Schema:
		return record{}, fmt.Errorf("settings %s: unknown schema %q", s.Path, r.Schema)
	}
	if err := r.RuntimeDefaults.Validate(); err != nil {
		return record{}, fmt.Errorf("settings %s: %w", s.Path, err)
	}
	if err := ValidateLocale(r.Locale); err != nil {
		return record{}, fmt.Errorf("settings %s: %w", s.Path, err)
	}
	if _, err := NormalizeResidents(r.Residents); err != nil {
		return record{}, fmt.Errorf("settings %s: resident_models: %w", s.Path, err)
	}
	if r.Updates != nil {
		if err := r.Updates.Validate(); err != nil {
			return record{}, fmt.Errorf("settings %s: updates: %w", s.Path, err)
		}
	}
	seen := map[string]bool{}
	for i, c := range r.Connections {
		normalized, err := c.normalized()
		if err != nil {
			return record{}, fmt.Errorf("settings %s: connection %q: %w", s.Path, c.Name, err)
		}
		if _, err := normalized.Resolve(s.currentLocalEndpoint()); err != nil {
			return record{}, fmt.Errorf("settings %s: connection %q: %w", s.Path, c.Name, err)
		}
		if seen[c.Name] {
			return record{}, fmt.Errorf("settings %s: duplicate connection %q", s.Path, c.Name)
		}
		seen[c.Name] = true
		r.Connections[i] = normalized
	}
	return r, nil
}

// update applies fn to the current record and stores the result. Nothing is
// written when fn fails.
func (s *Store) update(fn func(*record) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.load()
	if err != nil {
		return err
	}
	if err := fn(&r); err != nil {
		return err
	}
	sort.Slice(r.Connections, func(i, j int) bool { return r.Connections[i].Name < r.Connections[j].Name })
	if s.Path == "" {
		s.mem = r
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return err
	}
	r.Schema = Schema
	return home.WriteJSON(s.Path, r)
}

// SetDefaults validates and stores the runtime defaults. It changes nothing
// else: no worker, model, setup, startup or connection state is touched.
func (s *Store) SetDefaults(d Defaults) error {
	if err := d.Validate(); err != nil {
		return err
	}
	return s.update(func(r *record) error { r.RuntimeDefaults = d; return nil })
}

// Residents reads the desired additional resident models (catalog order). It
// is the selection the next start of the runtime honors, not what is running.
// A missing file is not an error; an unreadable one yields none plus the error.
func (s *Store) Residents() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.load()
	return r.Residents, err
}

// SetResidents validates and stores the desired additional resident models.
// It only stores: nothing is downloaded, materialized, activated, started,
// stopped or restarted, and the active model does not change. The selection
// applies at the next start of the runtime.
func (s *Store) SetResidents(ids []string) error {
	norm, err := NormalizeResidents(ids)
	if err != nil {
		return err
	}
	return s.update(func(r *record) error { r.Residents = norm; return nil })
}

// Locale reads the explicit operator UI locale selection; "" means none was
// made. A missing file is not an error.
func (s *Store) Locale() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.load()
	return r.Locale, err
}

// SetLocale validates and stores the operator UI locale ("" clears the
// selection). It changes presentation only: no runtime, CLI or environment
// behavior depends on it.
func (s *Store) SetLocale(l string) error {
	if err := ValidateLocale(l); err != nil {
		return err
	}
	return s.update(func(r *record) error { r.Locale = l; return nil })
}

// ResolvedLocale is the operator UI locale: the explicit selection, else the
// host locale, else English. An unreadable settings file resolves as unset.
func (s *Store) ResolvedLocale() i18n.Locale {
	l, _ := s.Locale()
	return i18n.Resolve(l, i18n.HostLocales()...)
}

func cloneUpdates(u *update.Settings) *update.Settings {
	if u == nil {
		return nil
	}
	c := u.Clone()
	return &c
}

// UpdateSettings reads the update subsystem's saved channel, last explicit
// check and installed identity. A missing file is not an error.
func (s *Store) UpdateSettings() (update.Settings, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.load()
	if r.Updates == nil {
		return update.Settings{}, err
	}
	return r.Updates.Clone(), err
}

// ModifyUpdateSettings applies fn to the saved update settings and stores the
// validated result; nothing is written when fn or validation fails. It only
// stores: no update check, download or installation is started by it.
func (s *Store) ModifyUpdateSettings(fn func(*update.Settings) error) error {
	return s.update(func(r *record) error {
		var u update.Settings
		if r.Updates != nil {
			u = r.Updates.Clone()
		}
		if err := fn(&u); err != nil {
			return err
		}
		if err := u.Validate(); err != nil {
			return err
		}
		r.Updates = &u
		return nil
	})
}

// SaveConnection validates and stores a profile, creating it or replacing the
// profile of the same name. It never starts, stops or inspects a tunnel.
func (s *Store) SaveConnection(c Connection) error {
	var err error
	if c, err = c.normalized(); err != nil {
		return err
	}
	// A profile returned by Connections carries a transient endpoint for Spec;
	// never retain it as part of the stored intent.
	c.localEndpoint = tunnel.LocalEndpoint{}
	return s.update(func(r *record) error {
		if _, err := c.Resolve(s.currentLocalEndpoint()); err != nil {
			return err
		}
		for i := range r.Connections {
			if r.Connections[i].Name == c.Name {
				r.Connections[i] = c
				return nil
			}
		}
		if len(r.Connections) >= MaxConnections {
			return fmt.Errorf("at most %d connections can be saved", MaxConnections)
		}
		r.Connections = append(r.Connections, c)
		return nil
	})
}

// RemoveConnection deletes the named profile.
func (s *Store) RemoveConnection(name string) error {
	return s.update(func(r *record) error {
		for i := range r.Connections {
			if r.Connections[i].Name == name {
				r.Connections = append(r.Connections[:i], r.Connections[i+1:]...)
				return nil
			}
		}
		return fmt.Errorf("connection %q does not exist", name)
	})
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
