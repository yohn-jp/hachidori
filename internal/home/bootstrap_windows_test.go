//go:build windows

package home

import (
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"
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

func TestWindowsLocatorReadHandleAllowsAtomicReplace(t *testing.T) {
	l := testLocator(t)
	a, b := t.TempDir(), t.TempDir()
	if _, err := l.Save(a); err != nil {
		t.Fatal(err)
	}

	f, err := openBootstrapReadFile(l.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	done := make(chan error, 1)
	go func() {
		_, err := l.Save(b)
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("replace while locator reader was open: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("replace blocked by locator read handle")
	}

	// The already-open handle still observes one complete old record while the
	// pathname resolves to the complete replacement.
	raw, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}
	var old Bootstrap
	if err := json.Unmarshal(raw, &old); err != nil || old.Schema != BootstrapSchema || old.Home != a {
		t.Fatalf("old read handle: %+v %v", old, err)
	}
	got, found, err := l.Lookup()
	if err != nil || !found || got.Root != b {
		t.Fatalf("replacement lookup: %+v found=%v err=%v", got, found, err)
	}
}
