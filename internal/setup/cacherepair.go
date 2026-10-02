package setup

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
)

// ErrCacheRepairRefused marks a repair that found something it is not allowed
// to remove or cannot prove safe. Nothing was deleted.
var ErrCacheRepairRefused = errors.New("artifact cache repair refused")

// CacheRepair is the audit record of one RepairArtifactCache run. Paths are
// relative to the artifact root; directories end with "/".
type CacheRepair struct {
	Kind     string   `json:"kind"`
	ID       string   `json:"id"`
	Found    bool     `json:"found"`
	Removed  []string `json:"removed"`
	Verified bool     `json:"verified"`
}

// pycacheDir is the directory name Python writes bytecode caches into.
const pycacheDir = "__pycache__"

// bytecodeCacheName is the name of a CPython bytecode cache file:
// <module>.cpython-<version>[.opt-1|.opt-2].pyc.
var bytecodeCacheName = regexp.MustCompile(`^[A-Za-z0-9_]+\.cpython-[0-9]{2,3}(\.opt-[12])?\.pyc$`)

// RepairArtifactCache removes Python bytecode-cache pollution from one
// already-materialized artifact: the catalog model id (KindModel) or the
// variant id (KindVariant). It exists because artifacts are immutable and the
// verifier is read-only; an earlier launch that let Python import an artifact
// module could leave __pycache__ beside its files, which strict variant
// verification correctly refuses. Verification is never relaxed: this is the
// separate, explicit mutation, and the strict verifier is its final authority.
//
// The artifact is resolved from its Hachidori identity, never from a path, and
// must be a real directory inside HACHIDORI_HOME. The repair deletes only
// entries that are all of: unmanifested, regular files named like a CPython
// cache (<module>.cpython-<NN>.pyc) with a bytecode header, directly inside a
// __pycache__ directory; and then those __pycache__ directories once empty.
// Any other unmanifested entry, symbolic link or irregular file, manifest name
// that is not a clean relative path, or manifested file that resembles a
// candidate refuses the whole repair before anything is deleted.
//
// The result reports success only when the normal verification then passes. A
// clean artifact is left untouched. The caller must ensure no worker is
// running from the artifact.
func RepairArtifactCache(h home.Home, kind, id string, obs *Observer) (CacheRepair, error) {
	rep := CacheRepair{Kind: kind, ID: id, Removed: []string{}}
	var base, dir string
	manifest := map[string]bool{}
	switch kind {
	case KindVariant:
		m, v, err := FindVariant(h, id)
		if err != nil {
			return rep, err
		}
		base, dir = h.Path("variants"), h.VariantDir(m.ID, v.ID)
		manifest[home.VariantManifestFile] = true
		for rel := range v.Files {
			manifest[rel] = true
		}
	case KindModel:
		m, err := modelByID(id)
		if err != nil {
			return rep, err
		}
		base, dir = h.Path("models"), h.Path("models", filepath.FromSlash(ModelDirName(m)))
		if _, err := InspectSource(dir, m); err != nil {
			return rep, fmt.Errorf("model %s: %w", id, err)
		}
		manifest["hachidori-model.json"] = true
		for rel := range m.Files {
			manifest[rel] = true
		}
	default:
		return rep, fmt.Errorf("cache repair applies to a model or a variant, not %q", kind)
	}
	for rel := range manifest {
		if !cleanRel(rel) {
			return rep, refusal("the manifest names %q, which is not a relative path inside the artifact", rel)
		}
	}
	root, err := confinedDir(h, base, dir, "repair")
	if err != nil {
		return rep, fmt.Errorf("%w: %v", ErrCacheRepairRefused, err)
	}

	files, dirs, err := inventoryCachePollution(root, manifest)
	if err != nil {
		return rep, err
	}
	rep.Found = len(files)+len(dirs) > 0
	if rep.Found {
		afterCacheInventory()
		if err := removeInventoried(root, files, dirs, &rep); err != nil {
			return rep, err
		}
	}
	if err := Verify(h, kind, id, obs); err != nil {
		return rep, fmt.Errorf("%s %s: strict verification failed after cache repair: %w", kind, id, err)
	}
	rep.Verified = true
	return rep, nil
}

// cleanRel reports whether rel is a plain slash-separated relative path inside
// an artifact: no dot segments, no leading slash, and no backslash or colon,
// which name a separator, a drive or a stream on Windows.
func cleanRel(rel string) bool {
	return rel != "." && fs.ValidPath(rel) && !strings.ContainsAny(rel, `\:`)
}

func refusal(format string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrCacheRepairRefused, fmt.Sprintf(format, a...))
}

// inventoryCachePollution walks root without following links and returns the
// recognized bytecode-cache files and __pycache__ directories to remove. It
// deletes nothing; any entry it cannot classify as manifested or recognized
// cache pollution is a refusal naming every such entry.
func inventoryCachePollution(root string, manifest map[string]bool) (files, dirs []string, err error) {
	holders := map[string]bool{} // directories that hold a manifested file
	folded := map[string]string{}
	for rel := range manifest {
		for d := path.Dir(rel); d != "."; d = path.Dir(d) {
			holders[d] = true
		}
		folded[strings.ToLower(rel)] = rel
	}
	var unknown []string
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, werr error) error {
		if werr != nil {
			return werr
		}
		r, rerr := filepath.Rel(root, p)
		if rerr != nil || r == "." {
			return rerr
		}
		rel := filepath.ToSlash(r)
		typ := d.Type()
		switch {
		case typ&fs.ModeSymlink != 0:
			return refusal("%s is a symbolic link", rel)
		case d.IsDir():
			if holders[rel] {
				return nil
			}
			if path.Base(rel) == pycacheDir && path.Base(path.Dir(rel)) != pycacheDir {
				dirs = append(dirs, rel)
				return nil
			}
			unknown = append(unknown, rel+"/")
			return fs.SkipDir
		case !typ.IsRegular():
			return refusal("%s is not a regular file", rel)
		}
		if manifest[rel] {
			return nil
		}
		if other, ok := folded[strings.ToLower(rel)]; ok {
			return refusal("%s differs from the manifested %s only by letter case", rel, other)
		}
		if !isBytecodeCacheFile(p, rel) {
			unknown = append(unknown, rel)
			return nil
		}
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return nil, nil, refusal("the artifact holds entries that are neither manifested nor Python bytecode cache: %s", strings.Join(unknown, ", "))
	}
	sort.Strings(files)
	sort.Strings(dirs)
	return files, dirs, nil
}

// isBytecodeCacheFile reports whether the regular file at abs, artifact
// relative path rel, is structurally a CPython bytecode cache: named
// <module>.cpython-<NN>.pyc directly inside a __pycache__ directory, with the
// 16-byte header whose magic number ends in "\r\n".
func isBytecodeCacheFile(abs, rel string) bool {
	if path.Base(path.Dir(rel)) != pycacheDir || !bytecodeCacheName.MatchString(path.Base(rel)) {
		return false
	}
	f, err := os.Open(abs)
	if err != nil {
		return false
	}
	defer f.Close()
	var hdr [16]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return false
	}
	return bytes.Equal(hdr[2:4], []byte("\r\n"))
}

// afterCacheInventory is a test seam between the inventory and the deletion,
// where an attacker or another process could change the tree.
var afterCacheInventory = func() {}

// removeInventoried deletes the inventoried files, then the inventoried
// directories deepest first, through one os.Root opened on the artifact: the
// deletions cannot be steered outside it by a link swapped in after the
// inventory.
func removeInventoried(rootPath string, files, dirs []string, rep *CacheRepair) error {
	root, err := openArtifactRoot(rootPath)
	if err != nil {
		return err
	}
	defer root.Close()
	for _, rel := range files {
		if _, err := removeCacheEntry(root, rel, false); err != nil {
			return err
		}
		rep.Removed = append(rep.Removed, rel)
	}
	// A directory that still holds anything is left alone.
	sort.Slice(dirs, func(i, j int) bool { return len(dirs[i]) > len(dirs[j]) })
	for _, rel := range dirs {
		gone, err := removeCacheEntry(root, rel, true)
		if err != nil {
			return err
		}
		if gone {
			rep.Removed = append(rep.Removed, rel+"/")
		}
	}
	return nil
}

// openArtifactRoot opens the artifact root for deletion and proves that what
// was opened is the real directory the path names now: not a link, not a
// reparse point, and the same directory the open reached.
func openArtifactRoot(path string) (*os.Root, error) {
	fi, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !realDir(fi) {
		return nil, refusal("the artifact root is no longer a real directory")
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, err
	}
	if st, err := root.Stat("."); err != nil || !os.SameFile(fi, st) {
		root.Close()
		return nil, refusal("the artifact root changed while it was being repaired")
	}
	return root, nil
}

// realDir reports whether fi describes a directory that is neither a symbolic
// link nor any other reparse point (Go reports a Windows junction or mount
// point as an irregular, not a directory, entry).
func realDir(fi fs.FileInfo) bool {
	return fi.IsDir() && fi.Mode()&(fs.ModeSymlink|fs.ModeIrregular) == 0
}

// removeCacheEntry deletes one inventoried entry, an empty directory when dir,
// and reports whether it is gone. Every step goes through root, which refuses
// any path that would leave the artifact, and the walk to the entry holds each
// parent directory to the same standard immediately before descending: it must
// be a real directory (no link, no reparse point), and the directory opened
// must be the very one that was checked, so a parent swapped for a link between
// the check and the open is refused instead of followed. The entry itself must
// still be a real file (or directory); a link swapped in at the last moment is
// only ever unlinked, never followed.
func removeCacheEntry(root *os.Root, rel string, dir bool) (bool, error) {
	if !cleanRel(rel) {
		return false, refusal("%q is not a relative path inside the artifact", rel)
	}
	parts := strings.Split(rel, "/")
	cur := root
	for _, name := range parts[:len(parts)-1] {
		fi, err := cur.Lstat(name)
		if errors.Is(err, fs.ErrNotExist) {
			return false, closeUnless(cur, root, nil)
		}
		if err != nil {
			return false, closeUnless(cur, root, err)
		}
		if !realDir(fi) {
			return false, closeUnless(cur, root, refusal("%s changed while it was being repaired", rel))
		}
		next, err := cur.OpenRoot(name)
		if err != nil {
			return false, closeUnless(cur, root, refusal("%s cannot be opened inside the artifact: %v", rel, err))
		}
		if st, err := next.Stat("."); err != nil || !os.SameFile(fi, st) {
			next.Close()
			return false, closeUnless(cur, root, refusal("%s changed while it was being repaired", rel))
		}
		closeUnless(cur, root, nil)
		cur = next
	}
	defer closeUnless(cur, root, nil)

	name := parts[len(parts)-1]
	fi, err := cur.Lstat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if (dir && !realDir(fi)) || (!dir && !fi.Mode().IsRegular()) {
		return false, refusal("%s changed while it was being repaired", rel)
	}
	if err := cur.Remove(name); err != nil {
		if dir {
			return false, nil // not empty: left in place
		}
		return false, err
	}
	return true, nil
}

// closeUnless closes r unless it is the artifact root, and returns err.
func closeUnless(r, root *os.Root, err error) error {
	if r != root {
		r.Close()
	}
	return err
}
