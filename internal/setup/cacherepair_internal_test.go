package setup

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// The delete step re-checks every inventoried path at the moment of deletion:
// a path that is not a clean relative one, or that would leave the artifact
// root, is refused and the file it names is untouched.
func TestRemoveCacheEntryRefusesPathTraversal(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "artifact")
	if err := os.MkdirAll(filepath.Join(root, "__pycache__"), 0o755); err != nil {
		t.Fatal(err)
	}
	victim := filepath.Join(base, "victim.cpython-312.pyc")
	inside := filepath.Join(root, "__pycache__", "m.cpython-312.pyc")
	for _, p := range []string{victim, inside} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, rel := range []string{"../victim.cpython-312.pyc", "__pycache__/../../victim.cpython-312.pyc", "/victim.cpython-312.pyc", filepath.ToSlash(victim), ".", "", "__pycache__/./m.cpython-312.pyc", `__pycache__\..\..\victim.cpython-312.pyc`} {
		if err := removeCacheEntry(root, rel, false); !errors.Is(err, ErrCacheRepairRefused) {
			t.Errorf("removeCacheEntry(%q) = %v, want a refusal", rel, err)
		}
	}
	for _, p := range []string{victim, inside} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s was deleted", p)
		}
	}
	if err := removeCacheEntry(root, "__pycache__/m.cpython-312.pyc", false); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(inside); err == nil {
		t.Fatal("a clean in-root path was not deleted")
	}
}
