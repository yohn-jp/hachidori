package desktop

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSettingsPathIsBesideDesktopPrefs(t *testing.T) {
	p := filepath.Join("x", "Hachidori", "desktop.json")
	if got, want := SettingsPath(p), filepath.Join("x", "Hachidori", "settings.json"); got != want {
		t.Fatalf("%q, want %q", got, want)
	}
	if SettingsPath("") != "" {
		t.Fatal("empty prefs path must stay in memory")
	}
}

// An existing desktop.json stays readable as is: Start minimized and the
// close-notice record keep their meaning after the settings authority exists.
func TestExistingDesktopJSONKeepsSemantics(t *testing.T) {
	path := filepath.Join(t.TempDir(), "desktop.json")
	legacy := `{"schema":"hachidori.desktop/1","start_minimized":true,"close_notice_shown":true}`
	writeFile(t, path, legacy)
	m := &Manager{Path: path}
	if _, min, err := m.Prefs(); err != nil || !min {
		t.Fatalf("start minimized %v %v", min, err)
	}
	if m.ShouldShowCloseNotice() {
		t.Fatal("close notice repeated for an existing record")
	}
	if b, _ := os.ReadFile(path); string(b) != legacy {
		t.Fatalf("reading rewrote desktop.json: %s", b)
	}
	// Changing a preference keeps the close-notice record.
	if err := m.SetStartMinimized(false); err != nil {
		t.Fatal(err)
	}
	var f prefsFile
	if err := readJSONFile(path, &f); err != nil || f.StartMinimized || !f.CloseNoticeShown || f.Schema != PrefsSchema {
		t.Fatalf("%+v %v", f, err)
	}
}
