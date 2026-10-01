package settings

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
)

// The desired additional residents persist across a relaunch (a new Store over
// the same file), in catalog order and without duplicates, as intent only.
func TestResidentSelectionPersistsAcrossRelaunch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if got, err := (&Store{Path: path}).Residents(); err != nil || len(got) != 0 {
		t.Fatalf("no file: %v %v", got, err)
	}
	if err := (&Store{Path: path}).SetResidents([]string{setup.OpenDeciderNano, setup.DefaultModel, setup.OpenDeciderNano}); err != nil {
		t.Fatal(err)
	}
	got, err := (&Store{Path: path}).Residents()
	if err != nil || !slices.Equal(got, []string{setup.DefaultModel, setup.OpenDeciderNano}) {
		t.Fatalf("after relaunch: %v %v", got, err)
	}
	first, _ := os.ReadFile(path)
	if err := (&Store{Path: path}).SetResidents(got); err != nil {
		t.Fatal(err)
	}
	if second, _ := os.ReadFile(path); string(first) != string(second) || !strings.Contains(string(first), `"resident_models"`) {
		t.Fatalf("same selection wrote different bytes:\n%s\n%s", first, second)
	}
	if err := (&Store{Path: path}).SetResidents(nil); err != nil {
		t.Fatal(err)
	}
	if got, err := (&Store{Path: path}).Residents(); err != nil || len(got) != 0 {
		t.Fatalf("cleared: %v %v", got, err)
	}
}

// Saving the selection stores only it: the saved runtime defaults, locale and
// connections are untouched, and an in-memory store behaves the same.
func TestResidentSelectionTouchesNothingElse(t *testing.T) {
	for name, s := range map[string]*Store{"file": {Path: filepath.Join(t.TempDir(), "settings.json")}, "memory": {}} {
		def := Defaults{Device: DeviceCPU, Model: setup.OpenDeciderNano}
		if err := s.SetDefaults(def); err != nil {
			t.Fatal(err)
		}
		if err := s.SetLocale("ja"); err != nil {
			t.Fatal(err)
		}
		if err := s.SetResidents([]string{setup.DefaultModel}); err != nil {
			t.Fatal(err)
		}
		if got, _ := s.Defaults(); got != def {
			t.Errorf("%s: defaults changed to %+v", name, got)
		}
		if l, _ := s.Locale(); l != "ja" {
			t.Errorf("%s: locale changed to %q", name, l)
		}
		if got, _ := s.Residents(); !slices.Equal(got, []string{setup.DefaultModel}) {
			t.Errorf("%s: residents %v", name, got)
		}
	}
}

// Only stable catalog identities are accepted: nothing that names a repository
// or revision, and a rejected selection writes nothing.
func TestResidentSelectionAcceptsOnlyCatalogModels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := &Store{Path: path}
	for _, bad := range [][]string{{"someone/else"}, {"laya-base@main"}, {""}, {setup.DefaultModel, "not-in-catalog"}} {
		if err := s.SetResidents(bad); err == nil {
			t.Errorf("accepted %v", bad)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("a rejected selection wrote a file: %v", err)
	}
}

// A settings file from before the selection existed is the same schema with no
// additional residents; a file naming an uncataloged resident is reported, not
// half-applied, and left untouched.
func TestResidentSelectionIsAdditiveAndValidatedOnRead(t *testing.T) {
	dir := t.TempDir()
	old := filepath.Join(dir, "old.json")
	if err := os.WriteFile(old, []byte(`{"schema":"hachidori.settings/1","runtime_defaults":{"device":"cuda"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := (&Store{Path: old}).Residents(); err != nil || len(got) != 0 {
		t.Fatalf("pre-existing file: %v %v", got, err)
	}
	bad := filepath.Join(dir, "bad.json")
	body := `{"schema":"hachidori.settings/1","runtime_defaults":{},"resident_models":["someone/else"]}`
	if err := os.WriteFile(bad, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := (&Store{Path: bad}).Residents(); err == nil || len(got) != 0 {
		t.Fatalf("uncataloged resident read as %v (%v)", got, err)
	}
	if b, _ := os.ReadFile(bad); string(b) != body {
		t.Fatal("reading modified the file")
	}
}
