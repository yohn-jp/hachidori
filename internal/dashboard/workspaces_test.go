package dashboard

import (
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestResourceSelectionSharesNativePatternAndBrowserPathFallback(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	pages := []struct {
		path string
		name string
	}{
		{path: "/forge", name: "dataset"},
		{path: "/workbench", name: "load_path"},
		{path: "/experiments", name: "dataset"},
		{path: "/errors", name: "path"},
	}
	for _, page := range pages {
		body := e.get(t, page.path).Body.String()
		if !strings.Contains(body, `name="`+page.name+`"`) || strings.Contains(body, `aria-label="Choose resource"`) {
			t.Errorf("browser %s lacks its typed path fallback or exposes a native picker", page.path)
		}
	}

	withPathPicker(e, &fakePathPicker{path: "/resources/chosen"})
	for _, page := range pages {
		body := e.get(t, page.path).Body.String()
		if !strings.Contains(body, `name="`+page.name+`"`) ||
			!strings.Contains(body, `aria-label="Choose resource"`) ||
			!strings.Contains(body, `<summary>Exact path`) {
			t.Errorf("desktop %s lacks the shared picker and exact-path disclosure", page.path)
		}
	}
}

// Models and Forge are first-class workspaces beside Runtime, the others and
// Settings: they are in the navigation on every page, the current page is
// marked, and they exist only where the maintenance authority is hosted.
func TestModelsAndForgeAreWorkspacesInTheNavigation(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	withSettings(e, &fakeSettings{}, nil)
	for _, p := range []string{"/", "/models", "/forge", "/workbench", "/experiments", "/errors", "/diagnostics", "/settings"} {
		rec := e.get(t, p)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s: %d", p, rec.Code)
		}
		nav := navRe.FindString(rec.Body.String())
		order := []string{`>Runtime</a>`, `href="/models"`, `href="/forge"`, `href="/workbench"`}
		at := -1
		for _, want := range order {
			i := strings.Index(nav, want)
			if i <= at {
				t.Errorf("%s: navigation lacks %q in order:\n%s", p, want, nav)
			}
			at = i
		}
		wantCurrent := map[string]string{"/models": `<a href="/models" aria-current="page">Models</a>`, "/forge": `<a href="/forge" aria-current="page">Forge</a>`}[p]
		if wantCurrent != "" && !strings.Contains(nav, wantCurrent) {
			t.Errorf("%s is not marked as the current page:\n%s", p, nav)
		}
		if n := strings.Count(nav, `aria-current="page"`); n != 1 {
			t.Errorf("%s marks %d current pages", p, n)
		}
	}
	for p, title := range map[string]string{"/models": "<title>Models · Hachidori</title>", "/forge": "<title>Forge · Hachidori</title>"} {
		if b := e.get(t, p).Body.String(); !strings.Contains(b, title) || strings.Count(b, "<h1>") != 1 {
			t.Errorf("%s title or heading", p)
		}
	}
}

// Settings keeps general application preferences. The full Models & runtimes
// and System One Forge administration moved out; Settings only points to it.
func TestSettingsNoLongerHostsModelsOrForgeAdministration(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Forge.Diagnostics = []DiagnosticRow{{ID: "probe-20261002T000000Z-0123abcd", Kind: "probe", Created: "2026-10-02T00:00:00Z", Error: "boom"}}
	withResidency(e, fm, &fakeResidency{})
	withSettings(e, &fakeSettings{}, &fakeDesktopPrefs{})
	withUpdates(e, &fakeUpdates{})
	withConnections(e, filepath.Join(t.TempDir(), "settings.json"))
	rec := e.get(t, "/settings")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /settings: %d", rec.Code)
	}
	body := rec.Body.String()
	for _, gone := range []string{`id="models-runtimes"`, `id="runtime-inventory"`, `id="model-inventory"`, `id="variant-inventory"`, `id="resident-selection"`, `id="models-choice"`,
		`id="forge-readiness"`, `id="forge-variants"`, "id=\"optimize-", "id=\"certify-", `action="/models/`, `action="/forge/`, `formaction="/models/`, `formaction="/forge/`,
		`action="/settings/models/`, `action="/settings/variants/`, `action="/settings/residents"`, "Models &amp; runtimes", "Resident models", "System One variants", "Forge readiness", "Failure diagnostics"} {
		if strings.Contains(body, gone) {
			t.Errorf("Settings still hosts %q", gone)
		}
	}
	for _, want := range []string{`action="/desktop/prefs"`, `action="/settings/defaults"`, `action="/settings/locale"`, `href="/settings/updates"`, `id="settings-connections"`,
		`action="/settings/connections/save"`, `id="settings-moved"`, `href="/models"`, `href="/forge"`} {
		if !strings.Contains(body, want) {
			t.Errorf("Settings lacks %q", want)
		}
	}
	// Without the maintenance authority there is nothing to point to.
	bare := newEnv(t)
	withSettings(bare, &fakeSettings{}, nil)
	if b := bare.get(t, "/settings").Body.String(); strings.Contains(b, `id="settings-moved"`) {
		t.Error("Settings points to workspaces that do not exist")
	}
}

var (
	idAttrRe     = regexp.MustCompile(`\sid="([^"]+)"`)
	labelledByRe = regexp.MustCompile(`aria-labelledby="([^"]+)"`)
	controlRe    = regexp.MustCompile(`<(input|select|textarea)\b[^>]*>`)
	buttonRe     = regexp.MustCompile(`(?s)<button\b[^>]*>(.*?)</button>`)
)

// The Models and Forge templates keep the workspace accessibility baseline: one
// heading, unique ids, every labelled-by target present, every visible control
// labelled, column headers scoped, buttons with text, no inline handlers.
func TestModelsAndForgeTemplatesAreAccessible(t *testing.T) {
	inv := variantInventory()
	e, fm, _ := forgeEnv(t, inv)
	fm.state.Busy = &ModelOp{Kind: "apply", Plan: []string{"validating"}, Phase: "validating"}
	fm.state.RestartRequired = false
	withResidency(e, fm, &fakeResidency{})
	for _, p := range []string{"/models", "/forge"} {
		body := e.get(t, p).Body.String()
		if !strings.Contains(body, `<html lang="en">`) || !strings.Contains(body, `<a class="skip" href="#main">`) || !strings.Contains(body, `<main class="workspace" id="main" tabindex="-1">`) {
			t.Errorf("%s lacks the shared landmarks", p)
		}
		ids := map[string]bool{}
		for _, m := range idAttrRe.FindAllStringSubmatch(body, -1) {
			if ids[m[1]] {
				t.Errorf("%s: duplicate id %q", p, m[1])
			}
			ids[m[1]] = true
		}
		for _, m := range labelledByRe.FindAllStringSubmatch(body, -1) {
			for _, id := range strings.Fields(m[1]) {
				if !ids[id] {
					t.Errorf("%s: aria-labelledby names a missing id %q", p, id)
				}
			}
		}
		for _, m := range controlRe.FindAllStringIndex(body, -1) {
			tag := body[m[0]:m[1]]
			if strings.Contains(tag, `type="hidden"`) {
				continue
			}
			if strings.Contains(tag, "aria-label=") {
				continue
			}
			before := body[:m[0]]
			if strings.LastIndex(before, "<label") <= strings.LastIndex(before, "</label>") {
				t.Errorf("%s: unlabelled control %s", p, tag)
			}
		}
		for _, m := range buttonRe.FindAllStringSubmatch(body, -1) {
			if strings.TrimSpace(regexp.MustCompile(`<[^>]*>`).ReplaceAllString(m[1], "")) == "" {
				t.Errorf("%s: a button without text: %s", p, m[0])
			}
		}
		for _, bad := range []string{"onclick=", "onsubmit=", "onchange=", "javascript:"} {
			if strings.Contains(body, bad) {
				t.Errorf("%s uses %q", p, bad)
			}
		}
		// Tables name their columns; destructive and slow actions are announced.
		if n, th := strings.Count(body, "<table"), strings.Count(body, `<th scope="col">`); n > 0 && th == 0 {
			t.Errorf("%s: tables without column headers", p)
		}
		if strings.Contains(body, `<th>`) {
			t.Errorf("%s: unscoped table header", p)
		}
	}
	// Destructive actions confirm; the apply names what it does.
	fb := e.get(t, "/forge").Body.String()
	if !strings.Contains(fb, `data-confirm="Remove this variant from HACHIDORI_HOME?`) || !strings.Contains(fb, `data-confirm="Apply this certified variant?`) ||
		!strings.Contains(fb, `data-confirm="Activate this variant as experimental/uncertified?`) {
		t.Error("a destructive or high-impact Forge action is not confirmed")
	}
}

// The Models and Forge workspaces are not a second implementation: they carry
// no script of their own beyond the shared shell behavior and the busy refresh,
// and contain no lifecycle logic (no fetch of an activation, restart or
// verification from the browser).
func TestForgeAndModelsCarryNoBrowserSideLifecycle(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Busy = &ModelOp{Kind: "apply", Plan: []string{"validating"}, Phase: "validating"}
	for _, p := range []string{"/models", "/forge"} {
		body := e.get(t, p).Body.String()
		for _, call := range []string{`fetch("/forge`, `fetch("/models`, `fetch('/forge`, `fetch('/models`, "XMLHttpRequest", `method: "POST"`, `method:"POST"`} {
			if strings.Contains(body, call) {
				t.Errorf("%s posts from script: %q", p, call)
			}
		}
		// The only scripts are the shared shell and the read-only busy refresh of
		// this very page.
		for _, m := range regexp.MustCompile(`(?s)<script>(.*?)</script>`).FindAllStringSubmatch(body, -1) {
			if strings.Contains(m[1], "fetch(") && !strings.Contains(m[1], `fetch("/live"`) && !strings.Contains(m[1], "fetch(location.pathname") {
				t.Errorf("%s: unexpected script fetch:\n%s", p, m[1])
			}
		}
	}
}

// Hosting the Tuning workspace leaves every existing workspace's content
// unchanged apart from the one added navigation entry.
func TestExistingWorkspacesAreUnchangedByTheTuningWorkspace(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	pages := []string{"/", "/models", "/forge", "/workbench", "/experiments", "/errors", "/diagnostics"}
	entry := regexp.MustCompile(`\s*<li><a href="/tuning"[^\n]*</li>`)
	token := regexp.MustCompile(`name="token" value="[0-9a-f]+"`)
	render := func(p string) string {
		rec := e.get(t, p)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s: %d", p, rec.Code)
		}
		return token.ReplaceAllString(entry.ReplaceAllString(rec.Body.String(), ""), `name="token"`)
	}
	before := map[string]string{}
	for _, p := range pages {
		before[p] = render(p)
	}
	withTuning(e, &fakeTuning{analysis: tuningAnalysis(t)})
	for _, p := range pages {
		if render(p) != before[p] {
			t.Errorf("%s changes when Tuning is hosted", p)
		}
	}
}
