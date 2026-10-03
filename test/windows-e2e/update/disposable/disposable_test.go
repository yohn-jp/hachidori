package disposable

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func write(t *testing.T, path, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestScratchIsRemovedAndContainsOnlyItself(t *testing.T) {
	var root string
	t.Run("inner", func(t *testing.T) {
		s := NewScratch(t)
		root = s.Root
		write(t, s.Path("home", "state", "x.json"), "x")
		if !s.Contains(s.Path("home", "state")) || s.Contains(filepath.Dir(s.Root)) || s.Contains(s.Root+"-sibling") {
			t.Fatal("Contains does not bound the scratch folder")
		}
		if !strings.Contains(filepath.Base(s.Root), "hachidori-e2e-") {
			t.Fatalf("unexpected scratch name %s", s.Root)
		}
	})
	if Exists(root) {
		t.Fatalf("scratch folder %s outlived its test", root)
	}
}

func TestWithin(t *testing.T) {
	root := filepath.Join(t.TempDir(), "a")
	for path, want := range map[string]bool{
		root:                                   true,
		filepath.Join(root, "b", "c"):          true,
		filepath.Join(root, "..", "a", "b"):    true,
		filepath.Join(root, ".."):              false,
		root + "x":                             false,
		filepath.Join(filepath.Dir(root), "z"): false,
	} {
		if got := Within(root, path); got != want {
			t.Errorf("Within(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestCopyFileIsIndependentAndExclusive(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.exe")
	write(t, src, "MZ-original")
	dst := filepath.Join(dir, "app", "copy.exe")
	sum, err := CopyFile(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if want, _ := FileSHA256(src); sum != want {
		t.Fatalf("copy sum %s, source %s", sum, want)
	}
	write(t, dst, "changed")
	if b, _ := os.ReadFile(src); string(b) != "MZ-original" {
		t.Fatal("writing the copy changed the source")
	}
	if _, err := CopyFile(src, dst); err == nil {
		t.Fatal("an existing destination must not be overwritten")
	}
}

func TestSnapshotDiffDetectsEveryKindOfChange(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "models", "m", "w.bin"), "weights")
	write(t, filepath.Join(root, "state", "history", "e.json"), "{}")
	write(t, filepath.Join(root, "state", "updates", "result.json"), "r")
	os.MkdirAll(filepath.Join(root, "state", "empty"), 0o755)
	keep := func(rel string, dir bool) bool {
		return rel != "state/updates" && !strings.HasPrefix(rel, "state/updates/")
	}
	before, err := Snapshot(root, keep)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := before["state/updates"]; ok {
		t.Fatal("an excluded directory was recorded")
	}
	if before["state/empty"] != DirMarker || before["models/m/w.bin"] == "" {
		t.Fatalf("snapshot %v", before)
	}
	again, _ := Snapshot(root, keep)
	if d := Diff(before, again); len(d) != 0 {
		t.Fatalf("identical state differs: %v", d)
	}

	write(t, filepath.Join(root, "state", "updates", "result.json"), "changed but excluded")
	write(t, filepath.Join(root, "models", "m", "w.bin"), "WEIGHTS")
	os.Remove(filepath.Join(root, "state", "history", "e.json"))
	write(t, filepath.Join(root, "state", "new.json"), "n")
	os.Remove(filepath.Join(root, "state", "empty"))
	after, _ := Snapshot(root, keep)
	want := []string{"added: state/new.json", "changed: models/m/w.bin", "removed: state/empty", "removed: state/history/e.json"}
	if got := Diff(before, after); !reflect.DeepEqual(got, want) {
		t.Fatalf("diff %v, want %v", got, want)
	}
}

func TestSnapshotOfMissingRootIsEmpty(t *testing.T) {
	snap, err := Snapshot(filepath.Join(t.TempDir(), "absent"), nil)
	if err != nil || len(snap) != 0 {
		t.Fatalf("%v %v", snap, err)
	}
}

func TestRemoveAllGivesUpAtItsDeadline(t *testing.T) {
	start := time.Now()
	if err := RemoveAll(filepath.Join(t.TempDir(), "never-existed"), time.Second); err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("RemoveAll of a missing path must return at once")
	}
}
