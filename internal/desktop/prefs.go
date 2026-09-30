package desktop

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"

	"github.com/yohn-jp/hachidori/internal/home"
)

// PrefsSchema versions the desktop preferences record.
const PrefsSchema = "hachidori.desktop/1"

// Preferences are the user-visible desktop preferences.
type Preferences struct {
	// StartAtSignIn is read from the per-user startup entry itself (the
	// entry is the single source of truth; nothing mirrors it in a file).
	StartAtSignIn bool
	// StartMinimized: a sign-in launch starts in the tray, window hidden.
	StartMinimized bool
}

// prefsFile is the on-disk record: small integration metadata that lives
// beside the Windows bootstrap locator (%LOCALAPPDATA%\Hachidori\desktop.json).
// It holds no runtime or model state and no secrets.
type prefsFile struct {
	Schema           string `json:"schema"`
	StartMinimized   bool   `json:"start_minimized"`
	CloseNoticeShown bool   `json:"close_notice_shown"`
}

// PrefsPath is the desktop preferences file beside the bootstrap locator.
// It does not touch the filesystem; it fails where there is no locator
// (non-Windows).
func PrefsPath() (string, error) {
	l, err := home.DefaultLocator()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(l.Path), "desktop.json"), nil
}

// SettingsPath is the application settings file (settings.json) that lives
// beside the given desktop preferences file. Empty stays empty (in memory).
func SettingsPath(prefsPath string) string {
	if prefsPath == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(prefsPath), "settings.json")
}

// Manager owns the desktop preferences: the startup entry (through Startup)
// and the small preferences file. Every operation is idempotent.
type Manager struct {
	// Path is the preferences file; empty keeps values in memory only.
	Path string
	// Startup is the per-user startup mechanism; nil means none.
	Startup Startup
	// Command is the exact command line the startup entry runs.
	Command string

	mu  sync.Mutex
	mem prefsFile
}

func (m *Manager) load() (prefsFile, error) {
	if m.Path == "" {
		f := m.mem
		f.Schema = PrefsSchema
		return f, nil
	}
	var f prefsFile
	err := home.ReadJSON(m.Path, &f)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return prefsFile{Schema: PrefsSchema}, nil
	case err != nil:
		return prefsFile{Schema: PrefsSchema}, fmt.Errorf("desktop preferences %s: %w", m.Path, err)
	case f.Schema != PrefsSchema:
		return prefsFile{Schema: PrefsSchema}, fmt.Errorf("desktop preferences %s: unknown schema %q", m.Path, f.Schema)
	}
	return f, nil
}

func (m *Manager) save(f prefsFile) error {
	f.Schema = PrefsSchema
	if m.Path == "" {
		m.mem = f
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(m.Path), 0o755); err != nil {
		return err
	}
	return home.WriteJSON(m.Path, f)
}

// Get reads the current preferences. A malformed preferences file yields the
// defaults plus an error; the startup entry state is still reported.
func (m *Manager) Get() (Preferences, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	f, ferr := m.load()
	p := Preferences{StartMinimized: f.StartMinimized}
	var serr error
	if m.Startup != nil {
		p.StartAtSignIn, serr = m.Startup.Enabled()
	}
	return p, errors.Join(ferr, serr)
}

// SetStartAtSignIn enables or removes the per-user startup entry. Enabling
// twice leaves one identical entry; disabling an absent entry succeeds.
func (m *Manager) SetStartAtSignIn(on bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.Startup == nil {
		return ErrUnsupported
	}
	if on {
		if m.Command == "" {
			return errors.New("no startup command is configured")
		}
		return m.Startup.Enable(m.Command)
	}
	return m.Startup.Disable()
}

// SetStartMinimized stores the Start minimized preference.
func (m *Manager) SetStartMinimized(on bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	f, err := m.load()
	if err != nil {
		return err
	}
	f.StartMinimized = on
	return m.save(f)
}

// Set applies both preferences (the dashboard form).
func (m *Manager) Set(startAtSignIn, startMinimized bool) error {
	return errors.Join(m.SetStartAtSignIn(startAtSignIn), m.SetStartMinimized(startMinimized))
}

// Prefs reports both preferences (the dashboard's view of the Manager).
func (m *Manager) Prefs() (startAtSignIn, startMinimized bool, err error) {
	p, err := m.Get()
	return p.StartAtSignIn, p.StartMinimized, err
}

// ShouldShowCloseNotice reports, once, that the first close-to-tray should
// be explained to the user, and records that it was. If the record cannot be
// read or written the notice is skipped or may repeat; it never blocks close.
func (m *Manager) ShouldShowCloseNotice() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	f, err := m.load()
	if err != nil || f.CloseNoticeShown {
		return false
	}
	f.CloseNoticeShown = true
	_ = m.save(f)
	return true
}
