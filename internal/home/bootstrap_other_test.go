//go:build !windows

package home

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// 12. non-Windows behavior is unchanged: no locator, no hidden state, and
// desktop discovery adds no implicit resolution beyond explicit/env.
func TestNonWindowsHasNoBootstrap(t *testing.T) {
	fake := t.TempDir()
	t.Setenv("HOME", fake)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(fake, "config"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(fake, "cache"))
	t.Setenv("LOCALAPPDATA", filepath.Join(fake, "local"))
	t.Setenv("HACHIDORI_HOME", "")

	if _, err := DefaultLocator(); !errors.Is(err, ErrBootstrapUnsupported) {
		t.Fatal(err)
	}
	if _, err := Remember(t.TempDir()); !errors.Is(err, ErrBootstrapUnsupported) {
		t.Fatal(err)
	}
	if err := Forget(); !errors.Is(err, ErrBootstrapUnsupported) {
		t.Fatal(err)
	}
	d, err := Discover("")
	if err != nil || d.Source != SourceUnconfigured {
		t.Fatal(d, err)
	}
	explicit := t.TempDir()
	if d, err := Discover(explicit); err != nil || d.Source != SourceExplicit || d.Home.Root != explicit {
		t.Fatal(d, err)
	}
	t.Setenv("HACHIDORI_HOME", explicit)
	if d, err := Discover(""); err != nil || d.Source != SourceEnv || d.Home.Root != explicit {
		t.Fatal(d, err)
	}
	if entries, _ := os.ReadDir(fake); len(entries) != 0 {
		t.Fatalf("hidden state created: %v", entries)
	}
	// The strict CLI resolver is untouched.
	t.Setenv("HACHIDORI_HOME", "")
	if _, err := Resolve(""); err == nil {
		t.Fatal("Resolve gained implicit resolution")
	}
}
