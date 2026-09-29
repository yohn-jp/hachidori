//go:build windows

package home

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func TestWindowsDefaultLocator(t *testing.T) {
	local := t.TempDir()
	t.Setenv("LOCALAPPDATA", local)
	l, err := DefaultLocator()
	if err != nil || l.Path != filepath.Join(local, "Hachidori", "bootstrap.json") {
		t.Fatal(l, err)
	}
	t.Setenv("LOCALAPPDATA", "")
	if _, err := DefaultLocator(); err == nil {
		t.Fatal("empty LOCALAPPDATA accepted")
	}
}

func TestWindowsDiscoverRememberForget(t *testing.T) {
	local := t.TempDir()
	t.Setenv("LOCALAPPDATA", local)
	t.Setenv("HACHIDORI_HOME", "")
	if d, err := Discover(""); err != nil || d.Source != SourceUnconfigured {
		t.Fatal(d, err)
	}
	if _, err := os.Stat(filepath.Join(local, "Hachidori")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatal("discovery created the locator directory")
	}
	root := t.TempDir()
	// Forward slashes normalize to a Windows path.
	h, err := Remember(filepath.ToSlash(root))
	if err != nil || h.Root != filepath.Clean(root) {
		t.Fatal(h, err)
	}
	if d, err := Discover(""); err != nil || d.Source != SourceLocator || d.Home.Root != h.Root {
		t.Fatal(d, err)
	}
	// The strict CLI resolver never consults the locator.
	if _, err := Resolve(""); err == nil {
		t.Fatal("Resolve used the bootstrap locator")
	}
	if err := Forget(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(root); err != nil {
		t.Fatal("forget removed the home")
	}
	if d, err := Discover(""); err != nil || d.Source != SourceUnconfigured {
		t.Fatal(d, err)
	}
}
