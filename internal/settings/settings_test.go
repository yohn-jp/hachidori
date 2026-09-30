package settings

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
)

type fakeDesktop struct {
	signIn, minimized bool
	sets              int
}

func (f *fakeDesktop) Prefs() (bool, bool, error) { return f.signIn, f.minimized, nil }
func (f *fakeDesktop) Set(a, b bool) error        { f.sets++; f.signIn, f.minimized = a, b; return nil }

func TestDefaultsRoundTripAndSurviveRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "settings.json")
	s := &Store{Path: path}
	if d, err := s.Defaults(); err != nil || d != (Defaults{}) {
		t.Fatalf("fresh store: %+v %v", d, err)
	}
	want := Defaults{Device: DeviceCPU, Model: setup.DefaultModel}
	if err := s.SetDefaults(want); err != nil {
		t.Fatal(err)
	}
	got, err := (&Store{Path: path}).Defaults() // a new process reads the file
	if err != nil || got != want {
		t.Fatalf("after restart: %+v %v", got, err)
	}
}

func TestMemoryStoreWithoutPath(t *testing.T) {
	s := &Store{}
	if err := s.SetDefaults(Defaults{Device: DeviceCUDA}); err != nil {
		t.Fatal(err)
	}
	if d, _ := s.Defaults(); d.Device != DeviceCUDA {
		t.Fatalf("%+v", d)
	}
}

func TestRecordIsVersionedDeterministicAndNonSecret(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := &Store{Path: path}
	d := Defaults{Device: DeviceCUDA, Model: setup.DefaultModel}
	if err := s.SetDefaults(d); err != nil {
		t.Fatal(err)
	}
	first, _ := os.ReadFile(path)
	if err := s.SetDefaults(d); err != nil {
		t.Fatal(err)
	}
	second, _ := os.ReadFile(path)
	if string(first) != string(second) {
		t.Fatal("same values wrote different bytes")
	}
	for _, want := range []string{`"schema": "hachidori.settings/1"`, `"runtime_defaults"`, `"device": "cuda"`} {
		if !strings.Contains(string(first), want) {
			t.Errorf("record lacks %s:\n%s", want, first)
		}
	}
	for _, bad := range []string{"secret", "token", "password"} {
		if strings.Contains(strings.ToLower(string(first)), bad) {
			t.Errorf("record mentions %q", bad)
		}
	}
}

func TestValidationRejectsUnknownValuesWithoutWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := &Store{Path: path}
	for _, d := range []Defaults{{Device: "tpu"}, {Model: "not-in-catalog"}} {
		if err := s.SetDefaults(d); err == nil {
			t.Errorf("accepted %+v", d)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("invalid input wrote a file: %v", err)
	}
}

func TestUnknownSchemaAndCorruptFileAreReportedAndLeftUntouched(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string]string{"schema.json": `{"schema":"other/9"}`, "bad.json": `{`} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		d, err := (&Store{Path: p}).Defaults()
		if err == nil || d != (Defaults{}) {
			t.Errorf("%s: %+v %v", name, d, err)
		}
		if b, _ := os.ReadFile(p); string(b) != body {
			t.Errorf("%s: read modified the file", name)
		}
	}
}

func TestExplicitInputWinsOverSavedDefaults(t *testing.T) {
	saved := Defaults{Device: DeviceCPU, Model: setup.DefaultModel}
	if got := saved.Resolve("cuda", ""); got.Device != "cuda" || got.Model != setup.DefaultModel {
		t.Fatalf("%+v", got)
	}
	if got := saved.Resolve("", ""); got != saved {
		t.Fatalf("%+v", got)
	}
	if got := (Defaults{}).Resolve("cpu", "laya-base"); got != (Defaults{Device: "cpu", Model: "laya-base"}) {
		t.Fatalf("%+v", got)
	}
}

func TestDesktopPreferencesGoThroughTheDesktopAuthority(t *testing.T) {
	f := &fakeDesktop{}
	s := &Store{Path: filepath.Join(t.TempDir(), "settings.json"), Desktop: f}
	if err := s.Set(true, true); err != nil {
		t.Fatal(err)
	}
	if a, b, err := s.Prefs(); err != nil || !a || !b || f.sets != 1 {
		t.Fatalf("%v %v %v sets=%d", a, b, err, f.sets)
	}
	// Saving runtime defaults does not touch desktop preferences.
	if err := s.SetDefaults(Defaults{Device: DeviceCPU}); err != nil || f.sets != 1 {
		t.Fatalf("%v sets=%d", err, f.sets)
	}
	if err := (&Store{}).Set(true, false); err == nil {
		t.Fatal("Set without a desktop authority succeeded")
	}
}

func TestModelsAreTheCatalog(t *testing.T) {
	if got := Models(); len(got) != len(setup.Models) || got[0] != setup.Models[0].ID {
		t.Fatalf("%v", got)
	}
}
