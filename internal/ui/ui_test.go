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

// rules returns the declarations of every rule of css whose selector list
// contains sel (compared as one selector of a comma-separated list).
func rules(css, sel string) []string {
	var out []string
	css = regexp.MustCompile(`(?s)/\*.*?\*/`).ReplaceAllString(css, "")
	for _, m := range regexp.MustCompile(`(?m)([^{}@]+)\{([^{}]*)\}`).FindAllStringSubmatch(css, -1) {
		for _, s := range strings.Split(m[1], ",") {
			if strings.TrimSpace(s) == sel {
				out = append(out, m[2])
			}
		}
	}
	return out
}

func rule(t *testing.T, sel string) string {
	t.Helper()
	r := rules(string(CSS()), sel)
	if len(r) == 0 {
		t.Fatalf("no rule for %q", sel)
	}
	return strings.Join(r, ";")
}

// One family of entry controls: the same box, and one rule for each state an
// operator can meet. Native elements stay native.
func TestFormControlsShareOneFamilyOfStates(t *testing.T) {
	box := rule(t, "input")
	for _, sel := range []string{"textarea", "select"} {
		if got := rule(t, sel); !strings.Contains(got, box[:40]) {
			t.Errorf("%s does not share the entry-control box", sel)
		}
	}
	for _, want := range []string{"min-height: var(--ctl-h)", "border: 1px solid var(--line-2)", "border-radius: var(--r)", "background: var(--inset)"} {
		if !strings.Contains(box, want) {
			t.Errorf("entry controls lack %q", want)
		}
	}
	for _, sel := range []string{
		"input:hover:not(:disabled):not([readonly])", // hover
		"input:focus-visible",                        // focus
		"input::placeholder",                         // placeholder
		"input[readonly]",                            // read-only
		"input:disabled",                             // disabled
		`input[aria-invalid="true"]`,                 // invalid
		`input[type="number"]`,                       // numeric
		`input[type="checkbox"]`, `input[type="radio"]`,
		"textarea",
		"select:disabled", "select:hover:not(:disabled)", "select:focus-visible",
		"textarea[readonly]", `textarea[aria-invalid="true"]`, `select[aria-invalid="true"]`,
	} {
		if len(rules(string(CSS()), sel)) == 0 {
			t.Errorf("no style for %s", sel)
		}
	}
	// A select reads as a selection in its closed state: a raised face and a
	// pointer, and the native arrow is not removed (no custom combobox).
	sel := rule(t, "select")
	for _, want := range []string{"background: var(--raised)", "cursor: pointer"} {
		if !strings.Contains(sel, want) {
			t.Errorf("select lacks %q", want)
		}
	}
	if strings.Contains(string(CSS()), "appearance: none") || strings.Contains(string(CSS()), "-webkit-appearance") {
		t.Error("a control's native appearance is replaced")
	}
	if strings.Contains(rule(t, "input"), "background: var(--raised)") {
		t.Error("a text entry looks like a selection")
	}
	// Read-only and invalid are not carried by color alone.
	if !strings.Contains(rule(t, "input[readonly]"), "border-style: dashed") {
		t.Error("read-only differs from editable by color only")
	}
	if inv := rule(t, `input[aria-invalid="true"]`); !strings.Contains(inv, "border-style: double") || !strings.Contains(inv, "border-color: var(--bad)") {
		t.Errorf("invalid: %s", inv)
	}
	if !strings.Contains(rule(t, ".field-error::before"), `content: "\2715`) {
		t.Error("an error message has no mark of its own")
	}
	if !strings.Contains(rule(t, "input:disabled"), "cursor: not-allowed") {
		t.Error("disabled has no pointer")
	}
	// A numeric entry is as wide as its digits, not a fixed box.
	if num := rule(t, `input[type="number"]`); !strings.Contains(num, "width: 12ch") || !strings.Contains(num, "text-align: right") {
		t.Errorf("numeric entry: %s", num)
	}
	// Repeatable options are one structured grid, not loose boxes.
	if r := rule(t, ".rows"); !strings.Contains(r, "display: grid") || !strings.Contains(r, "border: 1px solid var(--line)") {
		t.Errorf(".rows: %s", r)
	}
	if !strings.Contains(string(CSS()), ".optional {") {
		t.Error("no optional-field presentation")
	}
}

// Primary, secondary and destructive actions, each with hover and disabled; a
// destructive action is the same outlined control whether or not it is quiet.
func TestActionsHaveOneDestructiveTreatment(t *testing.T) {
	for _, sel := range []string{".btn", ".btn.primary", ".btn.quiet", ".btn.danger", ".btn:hover:not(:disabled)", ".btn.primary:hover:not(:disabled)", ".btn.danger:hover:not(:disabled)", ".btn:disabled"} {
		if len(rules(string(CSS()), sel)) == 0 {
			t.Errorf("no style for %s", sel)
		}
	}
	for _, sel := range []string{".btn.danger", ".btn.danger.quiet"} {
		r := rule(t, sel)
		if !strings.Contains(r, "color: var(--bad)") || !strings.Contains(r, "border-color: color-mix(in srgb, var(--bad) 45%, var(--line-2))") {
			t.Errorf("%s is not the shared destructive treatment: %s", sel, r)
		}
		if strings.Contains(r, "border-color: transparent") {
			t.Errorf("%s is a bare red word", sel)
		}
	}
}

// A state is carried by its word and by the shape of its glyph, so it still
// reads without color.
func TestStatusIsNotColorAlone(t *testing.T) {
	css := string(CSS())
	glyph := map[string]string{}
	for _, tone := range []string{"ok", "warn", "bad", "idle", "active"} {
		r := rule(t, ".tone-"+tone)
		if !strings.Contains(r, "--glyph-clip:") || !strings.Contains(r, "--glyph-fill:") {
			t.Errorf(".tone-%s sets no glyph: %s", tone, r)
		}
		clip := regexp.MustCompile(`--glyph-clip:\s*([^;]+)`).FindStringSubmatch(r)
		fill := regexp.MustCompile(`--glyph-fill:\s*([^;]+)`).FindStringSubmatch(r)
		glyph[tone] = strings.TrimSpace(clip[1]) + "/" + strings.TrimSpace(fill[1])
	}
	// ok and active share the disc (active also pulses); every other state is its own shape.
	seen := map[string]string{}
	for tone, g := range glyph {
		if tone == "active" {
			continue
		}
		if other, dup := seen[g]; dup {
			t.Errorf("%s and %s have the same glyph %s", tone, other, g)
		}
		seen[g] = tone
	}
	for _, sel := range []string{".dot", ".badge::before", ".semantic-state::before", ".status::before"} {
		if len(rules(css, sel)) == 0 {
			t.Errorf("%s has no glyph", sel)
		}
	}
	if !strings.Contains(rule(t, ".dot"), "clip-path: var(--glyph-clip") {
		t.Error("the glyph shape is not applied")
	}
	// Motion stays reserved for work in progress.
	if strings.Contains(css, ".tone-bad .dot {") && strings.Contains(css, "animation") && regexp.MustCompile(`\.tone-(bad|warn|ok|idle) \.dot[^{]*\{[^}]*animation`).MatchString(css) {
		t.Error("a settled state animates")
	}
}

// Short helper text and controls use the width of the section that holds
// them. A reading measure is for long-form prose only; no helper, field or
// control carries a width of its own at any desktop width.
func TestHelperTextAndControlsAreNotCappedBelowTheirSection(t *testing.T) {
	css := string(CSS())
	if strings.Contains(rule(t, ".note"), "max-width") {
		t.Error("short helper text is capped")
	}
	for _, prose := range []string{".lede", ".prose", ".empty"} {
		if !strings.Contains(rule(t, prose), "max-width: var(--measure)") {
			t.Errorf("%s has no reading measure", prose)
		}
	}
	for _, control := range []string{"input", "textarea", "select", ".field", ".fields", ".intent dd", ".btn-row"} {
		for _, r := range rules(css, control) {
			if regexp.MustCompile(`(^|[^-])max-width\s*:`).MatchString(r) {
				t.Errorf("%s has a max-width: %s", control, r)
			}
		}
	}
	if len(rules(css, ".intent dd select")) != 0 || len(rules(css, ".intent dd input")) != 0 {
		t.Error("a decision control is narrower than its row")
	}
	// Every width cap the shared system defines, and why it may exist: the
	// workspace (88rem) and messages within it, the reading measure for prose,
	// and a numeric entry never wider than its container. A new cap must be
	// justified here, not added to a page.
	allowed := map[string]bool{"var(--measure)": true, "var(--ws-max)": true, "100%": true}
	for _, m := range regexp.MustCompile(`[^-@(]max-width:\s*([^;}]+)`).FindAllStringSubmatch(css, -1) {
		if v := strings.TrimSpace(m[1]); !allowed[v] {
			t.Errorf("unexpected width cap %q", v)
		}
		if strings.HasSuffix(strings.TrimSpace(m[1]), "px") {
			t.Errorf("fixed pixel width cap %q", m[1])
		}
	}
}
