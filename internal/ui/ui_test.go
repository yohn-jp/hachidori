package ui

import (
	"os"
	"path/filepath"
	"regexp"
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
		"--focus:", "--focus-ring:", "--dur:", "--ease:", "--col-min:", "--shadow-pop:",
		// composition primitives: sections and facts, no containers
		".section {", ".spec {", ".id {",
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

// The visual system composes with type, spacing and hairline rules, not
// containers: no state edge, glow, ordinary-content shadow, or card chrome in
// the shared primitives, and a machine identity is never given a width that
// could squeeze it into a character-wide column.
func TestVisualSystemHasNoCardChrome(t *testing.T) {
	css := string(CSS())
	for _, banned := range []string{"inset 3px", "box-shadow: inset", "border-left: 3px", "0 0 0 5px", "word-break: break-all", "radial-gradient", "conic-gradient"} {
		if strings.Contains(css, banned) {
			t.Errorf("visual system uses %q", banned)
		}
	}
	// The only elevation tokens: the focus ring and the popover shadow.
	for _, rule := range regexp.MustCompile(`[^;{}]*box-shadow:[^;}]*`).FindAllString(css, -1) {
		if !strings.Contains(rule, "var(--focus-ring)") && !strings.Contains(rule, "--shadow-pop") {
			t.Errorf("shadow outside focus and popover: %q", strings.TrimSpace(rule))
		}
	}
	for _, role := range []string{".section", ".spec", ".id"} {
		m := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(role) + ` \{([^}]*)\}`).FindStringSubmatch(css)
		if m == nil {
			t.Fatalf("no %s rule", role)
		}
		for _, banned := range []string{"background", "border-radius", "box-shadow", "box-shadow"} {
			if strings.Contains(m[1], banned) {
				t.Errorf("%s rule carries %s", role, banned)
			}
		}
		if role == ".id" && regexp.MustCompile(`(^|[^-])(width|max-width)\s*:`).MatchString(m[1]) {
			t.Error(".id sets a width")
		}
	}
	// A state fills nothing: messages color their text and rule only.
	if m := regexp.MustCompile(`\.msg\.(bad|ok|warn) \{([^}]*)\}`).FindAllStringSubmatch(css, -1); len(m) != 3 {
		t.Fatal("message state rules missing")
	} else {
		for _, r := range m {
			if strings.Contains(r[2], "background") {
				t.Errorf(".msg.%s fills the surface", r[1])
			}
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

// Short explanatory text and controls take the width of their section; only
// long-form prose (.prose) carries the reading measure.
func TestShortTextIsNotNarrowedByTheSharedSystem(t *testing.T) {
	css := string(CSS())
	for _, sel := range []string{".lede", ".note", ".empty"} {
		m := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(sel) + ` \{([^}]*)\}`).FindStringSubmatch(css)
		if m == nil {
			t.Fatalf("no %s rule", sel)
		}
		if strings.Contains(m[1], "width") {
			t.Errorf("%s sets a width: %s", sel, m[1])
		}
	}
	if !regexp.MustCompile(`(?m)^\.prose \{ max-width: var\(--measure\); \}`).MatchString(css) {
		t.Error(".prose lacks the reading measure")
	}
	if strings.Contains(css, ".intent dd select") {
		t.Error("controls in .intent are capped narrower than their column")
	}
}
