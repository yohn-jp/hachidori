package setup

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func pycFile(t *testing.T, p string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, append([]byte{0xcb, 0x0d, 0x0d, 0x0a}, make([]byte, 12)...), 0o644); err != nil {
		t.Fatal(err)
	}
}

func openRoot(t *testing.T, dir string) *os.Root {
	t.Helper()
	r, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func symlinkOrSkip(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symbolic links are not available here: %v", err)
	}
}

func stillThere(p string) bool { _, err := os.Lstat(p); return err == nil }

// The delete step re-checks every inventoried path at the moment of deletion:
// a path that is not a clean relative one, or that would leave the artifact
// root, is refused and the file it names is untouched.
func TestRemoveCacheEntryRefusesPathTraversal(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "artifact")
	victim := filepath.Join(base, "victim.cpython-312.pyc")
	inside := filepath.Join(root, "__pycache__", "m.cpython-312.pyc")
	pycFile(t, victim)
	pycFile(t, inside)
	r := openRoot(t, root)
	for _, rel := range []string{"../victim.cpython-312.pyc", "__pycache__/../../victim.cpython-312.pyc", "/victim.cpython-312.pyc", filepath.ToSlash(victim), ".", "", "__pycache__/./m.cpython-312.pyc", `__pycache__\..\..\victim.cpython-312.pyc`} {
		if _, err := removeCacheEntry(r, rel, false); !errors.Is(err, ErrCacheRepairRefused) {
			t.Errorf("removeCacheEntry(%q) = %v, want a refusal", rel, err)
		}
	}
	if !stillThere(victim) || !stillThere(inside) {
		t.Fatal("a file was deleted by a refused path")
	}
	if gone, err := removeCacheEntry(r, "__pycache__/m.cpython-312.pyc", false); err != nil || !gone || stillThere(inside) {
		t.Fatalf("a clean in-root path: gone=%v err=%v", gone, err)
	}
}

// A link swapped in for any directory on the way to an entry, or for the entry
// itself, after the inventory is never followed out of the artifact: nothing
// outside is deleted.
func TestRemoveCacheEntryNeverFollowsASwappedLink(t *testing.T) {
	setup := func(t *testing.T) (root, outside string) {
		base := t.TempDir()
		root = filepath.Join(base, "artifact")
		outside = filepath.Join(base, "outside")
		pycFile(t, filepath.Join(root, "a", "b", "__pycache__", "x.cpython-312.pyc"))
		pycFile(t, filepath.Join(outside, "b", "__pycache__", "x.cpython-312.pyc"))
		return root, outside
	}
	const rel = "a/b/__pycache__/x.cpython-312.pyc"

	t.Run("outer directory", func(t *testing.T) {
		root, outside := setup(t)
		r := openRoot(t, root)
		os.RemoveAll(filepath.Join(root, "a"))
		symlinkOrSkip(t, outside, filepath.Join(root, "a"))
		if _, err := removeCacheEntry(r, rel, false); !errors.Is(err, ErrCacheRepairRefused) {
			t.Fatalf("err = %v", err)
		}
		if !stillThere(filepath.Join(outside, "b", "__pycache__", "x.cpython-312.pyc")) {
			t.Fatal("a file outside the artifact was deleted")
		}
	})
	t.Run("cache directory", func(t *testing.T) {
		root, outside := setup(t)
		r := openRoot(t, root)
		cache := filepath.Join(root, "a", "b", "__pycache__")
		os.RemoveAll(cache)
		symlinkOrSkip(t, filepath.Join(outside, "b", "__pycache__"), cache)
		if _, err := removeCacheEntry(r, rel, false); !errors.Is(err, ErrCacheRepairRefused) {
			t.Fatalf("file: err = %v", err)
		}
		if _, err := removeCacheEntry(r, "a/b/__pycache__", true); !errors.Is(err, ErrCacheRepairRefused) {
			t.Fatalf("directory: err = %v", err)
		}
		if !stillThere(filepath.Join(outside, "b", "__pycache__", "x.cpython-312.pyc")) {
			t.Fatal("a file outside the artifact was deleted")
		}
	})
	t.Run("link to a directory outside the root", func(t *testing.T) {
		root, outside := setup(t)
		r := openRoot(t, root)
		os.RemoveAll(filepath.Join(root, "a", "b"))
		symlinkOrSkip(t, filepath.Join(outside, "b"), filepath.Join(root, "a", "b"))
		if _, err := removeCacheEntry(r, rel, false); err == nil {
			t.Fatal("a link out of the root was followed")
		}
		if !stillThere(filepath.Join(outside, "b", "__pycache__", "x.cpython-312.pyc")) {
			t.Fatal("a file outside the artifact was deleted")
		}
	})
	t.Run("entry", func(t *testing.T) {
		root, outside := setup(t)
		r := openRoot(t, root)
		file := filepath.Join(root, filepath.FromSlash(rel))
		target := filepath.Join(outside, "b", "__pycache__", "x.cpython-312.pyc")
		os.Remove(file)
		symlinkOrSkip(t, target, file)
		if _, err := removeCacheEntry(r, rel, false); !errors.Is(err, ErrCacheRepairRefused) {
			t.Fatalf("err = %v", err)
		}
		if !stillThere(target) {
			t.Fatal("a file outside the artifact was deleted through a link")
		}
	})
}

// The whole repair, with the tree changed between the inventory and the
// deletion: a cache directory swapped for a link to a directory outside, and the
// artifact root swapped for a link. The repair is refused and nothing outside
// the artifact is deleted.
func TestRepairRefusesATreeSwappedAfterInventory(t *testing.T) {
	t.Run("cache directory", func(t *testing.T) {
		h, m := MaterializeFakeClef(t, "cpu")
		dir := h.Path("models", filepath.FromSlash(ModelDirName(m)))
		pycFile(t, filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc"))
		outside := filepath.Join(t.TempDir(), "outside")
		victim := filepath.Join(outside, "joint_schema_model.cpython-312.pyc")
		pycFile(t, victim)
		t.Cleanup(func() { afterCacheInventory = func() {} })
		afterCacheInventory = func() {
			os.RemoveAll(filepath.Join(dir, "__pycache__"))
			symlinkOrSkip(t, outside, filepath.Join(dir, "__pycache__"))
		}
		rep, err := RepairArtifactCache(h, KindModel, ClefFlash, nil)
		if !errors.Is(err, ErrCacheRepairRefused) || rep.Verified {
			t.Fatalf("repair = %+v, %v; want a refusal", rep, err)
		}
		if !stillThere(victim) {
			t.Fatal("a file outside the artifact was deleted")
		}
	})
	t.Run("artifact root", func(t *testing.T) {
		h, m := MaterializeFakeClef(t, "cpu")
		dir := h.Path("models", filepath.FromSlash(ModelDirName(m)))
		pyc := filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc")
		pycFile(t, pyc)
		moved := dir + ".moved"
		t.Cleanup(func() { afterCacheInventory = func() {} })
		afterCacheInventory = func() {
			if err := os.Rename(dir, moved); err != nil {
				t.Fatal(err)
			}
			symlinkOrSkip(t, moved, dir)
		}
		rep, err := RepairArtifactCache(h, KindModel, ClefFlash, nil)
		if !errors.Is(err, ErrCacheRepairRefused) || rep.Verified {
			t.Fatalf("repair = %+v, %v; want a refusal", rep, err)
		}
		if !stillThere(filepath.Join(moved, "__pycache__", "joint_schema_model.cpython-312.pyc")) {
			t.Fatal("a file behind the swapped root was deleted")
		}
	})
}
