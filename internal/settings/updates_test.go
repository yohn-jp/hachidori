package settings

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/update"
)

func TestUpdateSettingsPersistAdditivelyAcrossRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "settings.json")
	s := &Store{Path: path}
	if err := s.SaveConnection(nixos); err != nil {
		t.Fatal(err)
	}
	if err := s.SetLocale("ja"); err != nil {
		t.Fatal(err)
	}
	if u, err := s.UpdateSettings(); err != nil || u.Channel != "" || u.LastCheck != nil || u.Installed != nil {
		t.Fatalf("defaults %+v %v", u, err)
	}
	at := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	err := s.ModifyUpdateSettings(func(u *update.Settings) error {
		u.Channel = update.Development
		u.LastCheck = &update.LastCheck{Time: at, Channel: update.Development, OK: true, Latest: "0.2.5-dev"}
		u.Installed = &update.Installed{Version: "0.2.5-dev", SHA256: strings.Repeat("a", 64)}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := (&Store{Path: path}).UpdateSettings() // a new process reads the file
	if err != nil || got.Channel != update.Development || got.LastCheck == nil || !got.LastCheck.Time.Equal(at) || got.Installed.Version != "0.2.5-dev" {
		t.Fatalf("after restart %+v %v", got, err)
	}
	// The other settings are untouched.
	r := &Store{Path: path}
	if l, _ := r.Locale(); l != "ja" {
		t.Errorf("locale %q", l)
	}
	if cs, _ := r.Connections(); len(cs) != 1 {
		t.Errorf("connections %+v", cs)
	}
}

func TestUpdateSettingsAreValidatedBeforeWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := &Store{Path: path}
	for name, fn := range map[string]func(*update.Settings) error{
		"channel": func(u *update.Settings) error { u.Channel = "beta"; return nil },
		"version": func(u *update.Settings) error {
			u.Installed = &update.Installed{Version: "dev-24", SHA256: strings.Repeat("a", 64)}
			return nil
		},
		"callback": func(u *update.Settings) error { u.Channel = update.Stable; return os.ErrInvalid },
	} {
		if err := s.ModifyUpdateSettings(fn); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("invalid input wrote a file: %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"schema":"hachidori.settings/1","runtime_defaults":{},"updates":{"channel":"nightly"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := s.UpdateSettings(); err == nil {
		t.Fatal("invalid stored channel accepted")
	}
}

func TestUpdateSettingsInMemory(t *testing.T) {
	s := &Store{}
	if err := s.ModifyUpdateSettings(func(u *update.Settings) error { u.Channel = update.Development; return nil }); err != nil {
		t.Fatal(err)
	}
	u, _ := s.UpdateSettings()
	u.Channel = update.Stable // a copy: it must not change the store
	if again, _ := s.UpdateSettings(); again.Channel != update.Development {
		t.Fatalf("%+v", again)
	}
}
