package home

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// BootstrapSchema versions the bootstrap locator record.
const BootstrapSchema = "hachidori.bootstrap/1"

// Bootstrap is the complete bootstrap locator record. It owns only discovery
// metadata: the location of the selected HACHIDORI_HOME. It is the single
// narrow exception to "all mutable Hachidori state lives under
// HACHIDORI_HOME" (architecture §2.2, §6.2, §21) and must never carry
// runtime/model manifests, device selection, worker state, logs, caches,
// credentials or evaluation state.
type Bootstrap struct {
	Schema string `json:"schema"`
	Home   string `json:"home"`
}

// Sentinel errors reported by the bootstrap locator. They are matched with
// errors.Is; the concrete error is a *BootstrapError.
var (
	// ErrBootstrapMalformed: the locator exists but is not a valid record.
	ErrBootstrapMalformed = errors.New("bootstrap locator is malformed")
	// ErrBootstrapUnknownSchema: the locator declares an unsupported schema.
	ErrBootstrapUnknownSchema = errors.New("bootstrap locator has an unknown schema")
	// ErrStoredHomeMissing: the locator is valid but the home it names no
	// longer exists as a directory. This is distinct from never configured.
	ErrStoredHomeMissing = errors.New("bootstrap locator names a home that no longer exists")
	// ErrBootstrapUnsupported: this platform has no bootstrap locator.
	ErrBootstrapUnsupported = errors.New("bootstrap locator is only supported on Windows")
)

// BootstrapError describes a locator failure. Kind is one of the sentinel
// errors above (or nil for I/O failures); Cause is the underlying error.
type BootstrapError struct {
	Path  string // locator file
	Home  string // stored home, when known
	Kind  error
	Cause error
}

func (e *BootstrapError) Error() string {
	msg := "bootstrap locator " + e.Path
	if e.Kind != nil {
		msg = e.Kind.Error() + " (" + e.Path + ")"
	}
	if e.Home != "" {
		msg += ": home " + e.Home
	}
	if e.Cause != nil {
		msg += ": " + e.Cause.Error()
	}
	return msg
}

func (e *BootstrapError) Unwrap() []error {
	var errs []error
	if e.Kind != nil {
		errs = append(errs, e.Kind)
	}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// Locator is a bootstrap locator file. The platform-neutral logic lives here;
// DefaultLocator supplies the per-user path on Windows only.
type Locator struct{ Path string }

// Load reads the locator. It returns found=false and no error when the
// locator does not exist (never configured, or forgotten). A malformed or
// unknown-schema record is an error, never a first run. Load does not check
// that the stored home exists; see Lookup.
func (l Locator) Load() (b Bootstrap, found bool, err error) {
	data, err := readBootstrapFile(l.Path)
	if errors.Is(err, fs.ErrNotExist) {
		return Bootstrap{}, false, nil
	}
	if err != nil {
		return Bootstrap{}, false, &BootstrapError{Path: l.Path, Cause: err}
	}
	b, err = parseBootstrap(data)
	if err != nil {
		if be, ok := err.(*BootstrapError); ok {
			be.Path = l.Path
		}
		return Bootstrap{}, false, err
	}
	return b, true, nil
}


// readBootstrapFile tolerates the narrow Windows sharing/lock window that can
// occur while another process atomically replaces the locator. It never retries
// malformed data or permanent I/O failures, and the total delay is bounded.
func readBootstrapFile(path string) ([]byte, error) {
	const attempts = 10
	var data []byte
	var err error
	for i := 0; i < attempts; i++ {
		data, err = os.ReadFile(path)
		if err == nil || !retryableBootstrapReadError(err) {
			return data, err
		}
		time.Sleep(time.Millisecond)
	}
	return nil, err
}

func parseBootstrap(data []byte) (Bootstrap, error) {
	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return Bootstrap{}, &BootstrapError{Kind: ErrBootstrapMalformed, Cause: err}
	}
	if probe.Schema != BootstrapSchema {
		return Bootstrap{}, &BootstrapError{Kind: ErrBootstrapUnknownSchema, Cause: fmt.Errorf("schema %q, want %q", probe.Schema, BootstrapSchema)}
	}
	var b Bootstrap
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&b); err != nil {
		return Bootstrap{}, &BootstrapError{Kind: ErrBootstrapMalformed, Cause: err}
	}
	if _, err := dec.Token(); err != io.EOF {
		return Bootstrap{}, &BootstrapError{Kind: ErrBootstrapMalformed, Cause: errors.New("trailing data")}
	}
	if b.Home == "" || !filepath.IsAbs(b.Home) || filepath.Clean(b.Home) != b.Home {
		return Bootstrap{}, &BootstrapError{Kind: ErrBootstrapMalformed, Home: b.Home, Cause: errors.New("home must be a clean absolute path")}
	}
	return b, nil
}

// Lookup resolves the home named by the locator. found=false with no error
// means never configured. A stored home that is no longer a directory is
// reported as ErrStoredHomeMissing (the error carries the stored path).
func (l Locator) Lookup() (h Home, found bool, err error) {
	b, found, err := l.Load()
	if err != nil || !found {
		return Home{}, found, err
	}
	fi, err := os.Stat(b.Home)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return Home{}, true, &BootstrapError{Path: l.Path, Home: b.Home, Kind: ErrStoredHomeMissing}
	case err != nil:
		return Home{}, true, &BootstrapError{Path: l.Path, Home: b.Home, Cause: err}
	case !fi.IsDir():
		return Home{}, true, &BootstrapError{Path: l.Path, Home: b.Home, Kind: ErrStoredHomeMissing, Cause: errors.New("not a directory")}
	}
	return Home{Root: b.Home}, true, nil
}

// Save records root as the selected home and returns it normalized. root must
// be an existing directory; it need not contain a runtime, and selecting a
// home does not run setup or write anything under it. The parent directory of
// the locator is created only here, and the record is replaced atomically.
func (l Locator) Save(root string) (Home, error) {
	h, err := normalizeHome(root)
	if err != nil {
		return Home{}, err
	}
	fi, err := os.Stat(h.Root)
	if err != nil {
		return Home{}, fmt.Errorf("select home %s: %w", h.Root, err)
	}
	if !fi.IsDir() {
		return Home{}, fmt.Errorf("select home %s: not a directory", h.Root)
	}
	data, err := json.MarshalIndent(Bootstrap{Schema: BootstrapSchema, Home: h.Root}, "", "  ")
	if err != nil {
		return Home{}, err
	}
	if err := os.MkdirAll(filepath.Dir(l.Path), 0o755); err != nil {
		return Home{}, &BootstrapError{Path: l.Path, Cause: err}
	}
	if err := writeFileAtomic(l.Path, append(data, '\n')); err != nil {
		return Home{}, &BootstrapError{Path: l.Path, Cause: err}
	}
	return h, nil
}

// Forget removes the locator file and nothing else: never the home it names,
// never the locator's directory. Forgetting an absent locator is not an error.
func (l Locator) Forget() error {
	err := os.Remove(l.Path)
	if err == nil || errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return &BootstrapError{Path: l.Path, Cause: err}
}

// normalizeHome makes a platform-native (on Windows: Windows) absolute,
// cleaned path without touching the filesystem.
func normalizeHome(root string) (Home, error) {
	if root == "" {
		return Home{}, errors.New("empty home path")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return Home{}, err
	}
	return Home{Root: filepath.Clean(abs)}, nil
}

// writeFileAtomic writes data to a temporary file in path's directory, syncs
// it and renames it over path, so readers see either the old or the new
// record, never a partial one.
func writeFileAtomic(path string, data []byte) (err error) {
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer func() {
		if err != nil {
			os.Remove(tmp)
		}
	}()
	if _, err = f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Source says which authority resolved a desktop home.
type Source string

const (
	SourceExplicit     Source = "explicit"     // supplied by the desktop/controller
	SourceEnv          Source = "env"          // HACHIDORI_HOME
	SourceLocator      Source = "locator"      // bootstrap locator
	SourceUnconfigured Source = "unconfigured" // first run: nothing selected
)

// Discovery is the result of desktop home discovery. Home is zero when
// Source is SourceUnconfigured.
type Discovery struct {
	Home   Home
	Source Source
}

// discover implements desktop precedence: explicit > HACHIDORI_HOME > valid
// locator > unconfigured. A nil locator (non-Windows) skips step 3. Lower
// authorities are not consulted once a higher one applies.
func discover(explicit string, getenv func(string) string, loc *Locator) (Discovery, error) {
	if explicit != "" {
		h, err := normalizeHome(explicit)
		return Discovery{Home: h, Source: SourceExplicit}, err
	}
	if env := getenv("HACHIDORI_HOME"); env != "" {
		h, err := normalizeHome(env)
		return Discovery{Home: h, Source: SourceEnv}, err
	}
	if loc != nil {
		h, found, err := loc.Lookup()
		if err != nil {
			return Discovery{Source: SourceLocator}, err
		}
		if found {
			return Discovery{Home: h, Source: SourceLocator}, nil
		}
	}
	return Discovery{Source: SourceUnconfigured}, nil
}

// Discover resolves the home for the desktop entry point. Unlike Resolve,
// which stays strict for the CLI, it consults the per-user bootstrap locator
// on Windows and reports an unconfigured first run instead of an error. On
// other platforms it resolves only explicit and HACHIDORI_HOME and never
// touches bootstrap state.
func Discover(explicit string) (Discovery, error) {
	var loc *Locator
	if explicit == "" && os.Getenv("HACHIDORI_HOME") == "" {
		l, err := DefaultLocator()
		switch {
		case err == nil:
			loc = &l
		case !errors.Is(err, ErrBootstrapUnsupported):
			return Discovery{}, err
		}
	}
	return discover(explicit, os.Getenv, loc)
}

// Remember persists root as the selected home in the per-user bootstrap
// locator (Windows only). It does not run setup.
func Remember(root string) (Home, error) {
	l, err := DefaultLocator()
	if err != nil {
		return Home{}, err
	}
	return l.Save(root)
}

// Forget removes the per-user bootstrap locator (Windows only). The selected
// home is never deleted; the desktop simply asks for a home again.
func Forget() error {
	l, err := DefaultLocator()
	if err != nil {
		return err
	}
	return l.Forget()
}
