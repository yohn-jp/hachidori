package desktopkit

import (
	"fmt"
	"path/filepath"
	"strings"
)

// DeepPathChars is the home path length the deep-path scenarios certify.
//
// The product documents no maximum home path length, so the certified depth is
// a deliberate, conservative choice: the home itself is about 150 characters,
// which leaves roughly 110 characters of the classic 260-character Windows
// limit for everything Hachidori and the WebView2 profile create beneath it
// (the profile lives in <home>\cache\webview2). Longer homes are not claimed.
const DeepPathChars = 150

// Shape is a named kind of storage-folder path the bootstrap shard certifies.
type Shape struct {
	Name string
	// Path returns the folder for the shape beneath base. It never creates it.
	Path func(base string) (string, error)
}

// Shapes are the path shapes of W02.2: spaces, Unicode and a supported deep
// path.
func Shapes() []Shape {
	return []Shape{
		{Name: "spaces", Path: func(base string) (string, error) {
			return filepath.Join(base, "Hachidori Data  (x86)", "My Models & Runtime"), nil
		}},
		{Name: "unicode", Path: func(base string) (string, error) {
			return filepath.Join(base, "ハチドリ データ", "Привет мир", "café ñ 日本語"), nil
		}},
		{Name: "deep", Path: func(base string) (string, error) { return DeepPath(base, DeepPathChars) }},
	}
}

// DeepPath returns base plus nested folders whose total length is within 12
// characters below total and total itself. It fails when base alone is too long.
func DeepPath(base string, total int) (string, error) {
	if len(base) > total-30 {
		return "", fmt.Errorf("base path %q (%d characters) leaves no room for a %d-character deep path", base, len(base), total)
	}
	p := base
	for i := 1; ; i++ {
		seg := fmt.Sprintf("level-%02d-%s", i, strings.Repeat("d", 8))
		next := filepath.Join(p, seg)
		if len(next) > total {
			// Finish with one shorter segment so the path lands near the target.
			room := total - len(p) - 1
			if room >= 3 {
				p = filepath.Join(p, strings.Repeat("e", room))
			}
			return p, nil
		}
		p = next
	}
}
