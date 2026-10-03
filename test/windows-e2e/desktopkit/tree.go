package desktopkit

import (
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
)

// Tree lists the entries beneath root as slash-separated relative paths
// (directories end in "/"), sorted, at most limit of them. It is the bounded
// evidence of what a launch left on disk.
func Tree(root string, limit int) []string {
	var out []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil || rel == "." {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			rel += "/"
		}
		out = append(out, rel)
		return nil
	})
	sort.Strings(out)
	if limit > 0 && len(out) > limit {
		out = append(out[:limit], "...")
	}
	return out
}

// Residue returns the relative paths beneath root that look like Hachidori's
// own staging or temporary leftovers: staged runtimes (.staging-<id>), staged
// models (<name>.staging), resumable download partials (.part, .part.json),
// atomic-write temporaries (.<name>.<random>.tmp) and writability probes. After
// a launch or a failed operation none of them may remain. A directory whose
// name contains "webview" (a WebView2 browser profile, which WebView2 owns) is
// not scanned.
func Residue(root string) []string {
	var out []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && strings.Contains(strings.ToLower(d.Name()), "webview") {
			// WebView2's own browser profile is not Hachidori-owned staging.
			return filepath.SkipDir
		}
		if IsOwnedResidue(d.Name()) {
			rel, rerr := filepath.Rel(root, p)
			if rerr == nil {
				out = append(out, filepath.ToSlash(rel))
			}
		}
		return nil
	})
	sort.Strings(out)
	return out
}

// IsOwnedResidue reports whether a file or directory name is one of Hachidori's
// staging or temporary names.
func IsOwnedResidue(name string) bool {
	switch {
	case strings.HasPrefix(name, ".staging-"),
		strings.HasSuffix(name, ".staging"),
		strings.HasSuffix(name, ".part"),
		strings.HasSuffix(name, ".part.json"),
		strings.HasPrefix(name, ".hachidori-probe-"),
		name == ".write-probe",
		strings.HasPrefix(name, ".") && strings.HasSuffix(name, ".tmp"):
		return true
	}
	return false
}

// Has reports whether the relative slash path rel is present in tree (as a
// file, or as a directory with a trailing slash).
func Has(tree []string, rel string) bool {
	for _, t := range tree {
		if t == rel {
			return true
		}
	}
	return false
}

// HasPrefix reports whether any entry of tree starts with prefix.
func HasPrefix(tree []string, prefix string) bool {
	for _, t := range tree {
		if strings.HasPrefix(t, prefix) {
			return true
		}
	}
	return false
}

// InstallMarkers returns the relative paths beneath root of files only an
// installation or a setup run creates: the activation record, a model manifest
// and the setup and worker logs. A first run or a recovery screen must leave
// none of them.
func InstallMarkers(root string) []string {
	var out []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		switch d.Name() {
		case "active-runtime.json", "hachidori-model.json", "setup.log", "worker.log":
			if rel, rerr := filepath.Rel(root, p); rerr == nil {
				out = append(out, filepath.ToSlash(rel))
			}
		}
		return nil
	})
	sort.Strings(out)
	return out
}

// Below returns the entries of tree strictly beneath dir (a slash path with a
// trailing slash, for example "runtime/").
func Below(tree []string, dir string) []string {
	var out []string
	for _, t := range tree {
		if strings.HasPrefix(t, dir) && t != dir {
			out = append(out, t)
		}
	}
	return out
}
