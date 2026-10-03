// Package disposable holds the scratch-folder discipline of the update E2E:
// every home, executable copy and profile a scenario touches lives in one
// throwaway folder created for that scenario, and nothing outside it is ever
// written. It also snapshots and compares persisted state, which is how a
// scenario proves an update left the promised state unchanged.
package disposable

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// Scratch is one scenario's throwaway folder.
type Scratch struct{ Root string }

// NewScratch creates the folder under the OS temporary directory and removes it
// when the test ends. Removal is retried for a bounded time because a process
// the scenario started may still be releasing a file; a folder that cannot be
// removed is reported in the log, never as a failure of the proof itself.
func NewScratch(t testing.TB) *Scratch {
	t.Helper()
	root, err := os.MkdirTemp("", "hachidori-e2e-")
	if err != nil {
		t.Fatalf("scratch folder: %v", err)
	}
	// Resolve symlinks and short names once so every later comparison uses the
	// same spelling.
	if r, err := filepath.EvalSymlinks(root); err == nil {
		root = r
	}
	s := &Scratch{Root: root}
	t.Cleanup(func() {
		if err := RemoveAll(root, 15*time.Second); err != nil {
			t.Logf("scratch folder %s was not fully removed: %v", root, err)
		}
	})
	return s
}

// Path joins parts under the scratch root.
func (s *Scratch) Path(parts ...string) string {
	return filepath.Join(append([]string{s.Root}, parts...)...)
}

// Contains reports whether path lies inside the scratch root.
func (s *Scratch) Contains(path string) bool { return Within(s.Root, path) }

// Within reports whether path is root or lies beneath it, after cleaning.
func Within(root, path string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(path))
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel))
}

// RemoveAll removes path, retrying until deadline.
func RemoveAll(path string, within time.Duration) error {
	deadline := time.Now().Add(within)
	for {
		err := os.RemoveAll(path)
		if err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return err
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// CopyFile copies src to dst (creating its folder) and returns the copy's
// SHA-256.
func CopyFile(src, dst string) (string, error) {
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", err
	}
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o755)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(out, h), in); err != nil {
		out.Close()
		return "", err
	}
	if err := out.Close(); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// FileSHA256 hashes a file.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// DirMarker is the snapshot value of a directory.
const DirMarker = "<dir>"

// Snapshot records the content of every file and directory under root that
// keep accepts (by slash-separated path relative to root). A directory that
// keep rejects is not entered. A root that does not exist yields an empty
// snapshot.
func Snapshot(root string, keep func(rel string, dir bool) bool) (map[string]string, error) {
	snap := map[string]string{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if keep != nil && !keep(rel, d.IsDir()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			snap[rel] = DirMarker
			return nil
		}
		sum, err := FileSHA256(path)
		if err != nil {
			return fmt.Errorf("%s: %w", rel, err)
		}
		snap[rel] = sum
		return nil
	})
	return snap, err
}

// Diff describes, in a stable order, how after differs from before. An empty
// result means the two snapshots are identical.
func Diff(before, after map[string]string) []string {
	var out []string
	for rel, sum := range before {
		got, ok := after[rel]
		switch {
		case !ok:
			out = append(out, "removed: "+rel)
		case got != sum:
			out = append(out, "changed: "+rel)
		}
	}
	for rel := range after {
		if _, ok := before[rel]; !ok {
			out = append(out, "added: "+rel)
		}
	}
	slices.Sort(out)
	return out
}

// Exists reports whether path exists (any file type).
func Exists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil
}
