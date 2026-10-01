package ui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The visual system defines each role once and carries the operator
// preferences every surface must honor (docs/desktop.md).
func TestVisualSystemContract(t *testing.T) {
	css := string(CSS())
	for _, want := range []string{
		// token families
		"--fs-display:", "--fs-title:", "--fs-heading:", "--fs-body:", "--fs-small:", "--fs-label:", "--fs-code:",
		"--sp-1:", "--sp-7:", "--r:", "--r-lg:", "--ctl-h:", "--measure:", "--ws-max:",
		"--focus:", "--focus-ring:", "--dur:", "--ease:",
		".tone-ok", ".tone-warn", ".tone-bad", ".tone-idle", ".tone-active",
		// text scaling follows the operator's preference
		"html { font-size: 100%; }", "font: var(--fs-body)/1.5 var(--font);",
		// preferences
		"@media (prefers-color-scheme: light)", "@media (prefers-reduced-motion: reduce)",
		"animation: none !important; transition: none !important;", "@media (forced-colors: active)",
		":focus-visible { outline: var(--focus);",
		// pending state
		`.btn[aria-busy="true"]`,
	} {
		if !strings.Contains(css, want) {
			t.Errorf("visual system lacks %q", want)
		}
	}
	// Motion is reserved for work in progress: a warning holds still.
	if strings.Contains(css, ".tone-warn .dot") {
		t.Error("warning state animates")
	}
	for _, banned := range []string{"zoom", "backdrop-filter", "blur(", "linear-gradient", "-apple-system", "SF Mono"} {
		if strings.Contains(css, banned) {
			t.Errorf("visual system uses %q", banned)
		}
	}
}

// Accessibility is never solved by zooming the WebView: no desktop source
// sets a zoom factor or a CSS zoom.
func TestNoWebViewZoom(t *testing.T) {
	for _, dir := range []string{"../desktop", "../dashboard", "../firstrun", "."} {
		files, _ := filepath.Glob(filepath.Join(dir, "*"))
		for _, f := range files {
			if strings.HasSuffix(f, "_test.go") || !(strings.HasSuffix(f, ".go") || strings.HasSuffix(f, ".html") || strings.HasSuffix(f, ".css")) {
				continue
			}
			b, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			s := string(b)
			for _, banned := range []string{"ZoomFactor", "zoom:", "SetZoom"} {
				if strings.Contains(s, banned) {
					t.Errorf("%s contains %q", f, banned)
				}
			}
		}
	}
}
