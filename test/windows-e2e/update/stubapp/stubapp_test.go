package stubapp

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestOrdinaryArgumentsAreNotARole(t *testing.T) {
	for _, args := range [][]string{nil, {}, {"-test.v"}, {"-test.run=X"}, {"apply-update"}, {"decide"}} {
		if code, ok := RunAs("x", args); ok || code != 0 {
			t.Errorf("%v was handled as a role", args)
		}
		if SelectsRole(args) {
			t.Errorf("%v selects a role", args)
		}
	}
}

func TestReopenWritesAMarkerBesideTheExecutableOnly(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "hachidori.exe")
	if err := os.WriteFile(exe, []byte("MZ new release"), 0o755); err != nil {
		t.Fatal(err)
	}
	home := filepath.Join(dir, "home")
	code, ok := RunAs(exe, []string{"desktop", "--home", home})
	if !ok || code != 0 {
		t.Fatalf("code %d handled %v", code, ok)
	}
	m, err := ReadMarker(exe)
	if err != nil {
		t.Fatal(err)
	}
	if m.Exe != exe || m.Home != home || len(m.SHA256) != 64 || len(m.Args) != 3 {
		t.Fatalf("%+v", m)
	}
	ents, _ := os.ReadDir(dir)
	if len(ents) != 2 {
		t.Fatalf("unexpected files beside the executable: %v", ents)
	}
	if _, err := os.Stat(home); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("the stand-in must not create or touch the home")
	}
}

func TestReopenRejectsUnknownFlags(t *testing.T) {
	exe := filepath.Join(t.TempDir(), "x.exe")
	os.WriteFile(exe, []byte("MZ"), 0o755)
	if code, ok := RunAs(exe, []string{"desktop", "--bogus"}); !ok || code != 2 {
		t.Fatalf("code %d handled %v", code, ok)
	}
	if _, err := ReadMarker(exe); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a marker was written for a refused start: %v", err)
	}
}
