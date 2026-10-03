package dashboard

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/history"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/ui"
)

func TestDevelopmentConnectionsAutoIntentAndProjection(t *testing.T) {
	e := newEnv(t)
	store := withConnections(e, filepath.Join(t.TempDir(), "settings.json"))
	endpoint := tunnel.LocalEndpoint{Host: "127.0.0.1", Port: 9123}
	if err := store.SetLocalEndpoint(endpoint); err != nil {
		t.Fatal(err)
	}
	cfg := e.d.cfg
	cfg.APIAddr = "127.0.0.1:9123"
	e.d = New(cfg)

	form := url.Values{
		"name": {"auto-dev"}, "destination": {"dev@example"},
		"remote_bind_mode": {string(settings.ConnectionAuto)}, "remote_bind": {"0.0.0.0"},
		"remote_port_mode": {string(settings.ConnectionAuto)}, "remote_port": {"7900"},
		"local_port_mode": {string(settings.ConnectionAuto)}, "local_port": {"7901"},
	}
	e.post(t, "/settings/connections/save", form)
	if action := e.lastAction(t); !action.OK {
		t.Fatalf("save Auto connection: %+v", action)
	}
	connections, err := store.Connections()
	if err != nil || len(connections) != 1 {
		t.Fatalf("saved connections: %+v, %v", connections, err)
	}
	connection := connections[0]
	if connection.RemoteBindMode != settings.ConnectionAuto || connection.RemoteBind != "" ||
		connection.RemotePortMode != settings.ConnectionAuto || connection.RemotePort != 0 ||
		connection.LocalPortMode != settings.ConnectionAuto || connection.LocalPort != 0 {
		t.Fatalf("Auto intent retained override values: %+v", connection)
	}
	if got := connection.Spec(); got.RemoteBind != "127.0.0.1" || got.RemotePort != endpoint.Port || got.LocalPort != endpoint.Port {
		t.Fatalf("Auto resolved spec: %+v", got)
	}

	page := e.get(t, "/settings").Body.String()
	for _, want := range []string{
		`<dt>Remote bind</dt><dd>Auto → 127.0.0.1</dd>`,
		`<dt>Remote port</dt><dd>Auto → 9123</dd>`,
		`<dt>Local Hachidori port</dt><dd>Auto → 9123</dd>`,
		`name="remote_bind_mode"><option value="auto" selected>Auto → 127.0.0.1</option>`,
		`name="remote_port_mode"><option value="auto" selected>Auto → 9123</option>`,
		`name="local_port_mode"><option value="auto" selected>Auto → 9123</option>`,
	} {
		if !strings.Contains(page, want) {
			t.Errorf("Settings lacks Auto intent projection %q", want)
		}
	}

	start := strings.Index(page, `id="connection-new"`)
	if start < 0 {
		t.Fatal("Settings lacks the new connection form")
	}
	newForm := page[start:]
	if name, dest, advanced := strings.Index(newForm, `name="name"`), strings.Index(newForm, `name="destination"`), strings.Index(newForm, `<summary>Advanced</summary>`); name < 0 || dest <= name || advanced <= dest {
		t.Fatalf("connection name and destination are not primary fields before Advanced: %s", newForm[:min(len(newForm), 1200)])
	}

	pinnedForm := url.Values{
		"name": {"auto-dev"}, "destination": {"dev@example"},
		"remote_bind_mode": {string(settings.ConnectionPinned)}, "remote_bind": {"127.0.0.1"},
		"remote_port_mode": {string(settings.ConnectionPinned)}, "remote_port": {"9200"},
		"local_port_mode": {string(settings.ConnectionPinned)}, "local_port": {"9300"},
	}
	e.post(t, "/settings/connections/save", pinnedForm)
	if action := e.lastAction(t); !action.OK {
		t.Fatalf("save pinned connection: %+v", action)
	}
	connections, err = store.Connections()
	if err != nil || len(connections) != 1 {
		t.Fatalf("saved pinned connections: %+v, %v", connections, err)
	}
	connection = connections[0]
	if connection.RemoteBindMode != settings.ConnectionPinned || connection.RemoteBind != "127.0.0.1" ||
		connection.RemotePortMode != settings.ConnectionPinned || connection.RemotePort != 9200 ||
		connection.LocalPortMode != settings.ConnectionPinned || connection.LocalPort != 9300 {
		t.Fatalf("pinned intent was not retained: %+v", connection)
	}
}

// Every workstation workspace renders with the one visual system
// (docs/desktop.md): the shared token and primitive stylesheet is inlined
// once, and no workspace redefines a color, typography or spacing token.
func TestWorkspacesUseTheOneVisualSystem(t *testing.T) {
	e := newEnv(t)
	withSettings(e, &fakeSettings{}, nil)
	withModels(e, &fakeModels{state: ModelsState{Inventory: variantInventory()}})
	withVariants(e, &fakeVariants{})
	sys := string(ui.CSS())
	for _, p := range []string{"/", "/workbench", "/experiments", "/errors", "/diagnostics", "/settings", "/models", "/forge"} {
		body := e.get(t, p).Body.String()
		if strings.Count(body, sys) != 1 {
			t.Errorf("%s does not inline the visual system exactly once", p)
		}
		page := strings.Replace(body, sys, "", 1)
		for _, tok := range []string{"--bg:", "--text:", "--accent:", "--fs-", "--sp-", "--r:", "--ctl-h:", "--focus:", "--dur"} {
			if regexp.MustCompile(regexp.QuoteMeta(tok) + `[a-z0-9-]*:`).MatchString(page) {
				t.Errorf("%s redefines visual-system token %s", p, tok)
			}
		}
	}
}

func cssRule(t *testing.T, body, selector string) string {
	t.Helper()
	m := regexp.MustCompile(regexp.QuoteMeta(selector) + ` \{([^}]*)\}`).FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("no CSS rule for %s", selector)
	}
	return m[1]
}

// The workstation composes on the page canvas (docs/desktop.md): no card
// chrome, no decorative state edge, no elevation for ordinary content, and a
// navigation selection that is a rule and weight rather than a pill.
func TestWorkspacesComposeWithoutCardChrome(t *testing.T) {
	e := newEnv(t)
	withSettings(e, &fakeSettings{}, nil)
	withModels(e, &fakeModels{state: ModelsState{Inventory: variantInventory()}})
	withVariants(e, &fakeVariants{})
	for _, p := range []string{"/", "/workbench", "/experiments", "/errors", "/diagnostics", "/settings", "/models", "/forge"} {
		body := e.get(t, p).Body.String()
		for _, banned := range []string{"box-shadow: inset", "inset 3px", "border-left: 3px", "word-break: break-all", "var(--r-lg)"} {
			if strings.Contains(body, banned) {
				t.Errorf("%s uses %q", p, banned)
			}
		}
		// a container is not a rounded, filled box
		if regexp.MustCompile(`\{[^}]*border-radius: var\(--r-lg\)[^}]*\}|\{[^}]*background: var\(--surface\)[^}]*border:[^}]*\}`).MatchString(body) {
			t.Errorf("%s defines a card container", p)
		}
		// every top-level headed region is a section on the canvas
		for _, m := range regexp.MustCompile(`<section class="region[^"]*"`).FindAllString(body, -1) {
			if !strings.Contains(m, "region section") {
				t.Errorf("%s has a region that is not a section: %s", p, m)
			}
		}
		sel := cssRule(t, body, `.sidenav a[aria-current="page"]`)
		if strings.Contains(sel, "background") || strings.Contains(sel, "shadow") || strings.Contains(sel, "radius") {
			t.Errorf("%s: selected navigation is a filled pill: %s", p, sel)
		}
		if !strings.Contains(sel, "border-left-color: var(--accent)") || !strings.Contains(sel, "font-weight") {
			t.Errorf("%s: selected navigation lacks its rule and weight: %s", p, sel)
		}
		if r := cssRule(t, body, ".sidenav a"); strings.Contains(r, "border-radius") {
			t.Errorf("%s: navigation items are rounded", p)
		}
	}
}

// Runtime is the reference composition: operational state with its actions,
// then technical state as one fact grid under a rule, then telemetry. Machine
// identities use the one bounded identity treatment, and the fact grid cannot
// shrink a column below the readable minimum.
func TestRuntimeReferenceComposition(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	panel := body[strings.Index(body, `<section class="readiness-panel`):]
	panel = panel[:strings.Index(panel, "</section>")]
	order := []string{`class="state-word"`, `class="actions"`, `class="spec"`, `<dt>Model</dt><dd class="id">`, `<dt>Runtime</dt><dd class="id">`}
	at := 0
	for _, s := range order {
		i := strings.Index(panel[at:], s)
		if i < 0 {
			t.Fatalf("runtime composition lacks %q after position %d", s, at)
		}
		at += i
	}
	if strings.Contains(body, `class="identity"`) || strings.Contains(body, `class="model"`) {
		t.Error("runtime still uses a page-local identity treatment")
	}
	spec := cssRule(t, strings.Split(body, "</style>")[0], ".spec")
	if !strings.Contains(spec, "minmax(min(100%, var(--col-min)), 1fr)") {
		t.Errorf("fact grid can shrink columns below the readable minimum: %s", spec)
	}
	panelCSS := cssRule(t, body, ".readiness-panel")
	for _, banned := range []string{"background", "border:", "border-radius", "box-shadow"} {
		if strings.Contains(panelCSS, banned) {
			t.Errorf("readiness composition carries %s", banned)
		}
	}
	// the state stays the primary action's neighbor, and Stop stays visible and destructive
	if !strings.Contains(panel, `<button type="submit" class="btn danger">Stop</button>`) {
		t.Error("Stop is hidden or not destructive")
	}
}

// Irreversible deletions ask first and read as destructive; the confirmation
// goes through the shell's one submit handler, not per-form script.
func TestIrreversibleActionsConfirm(t *testing.T) {
	e, _, _ := historyEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	e.post(t, "/experiments/save", url.Values{"seq": {"1"}, "label": {"a"}})
	exp := e.get(t, "/experiments").Body.String()
	if !regexp.MustCompile(`<form method="post" action="/history/delete" class="inline" data-confirm="[^"]+">.*?class="btn danger">Delete</button>`).MatchString(exp) {
		t.Error("history delete is not a confirmed destructive action")
	}

	m := &fakeModels{state: ModelsState{Inventory: modelsInventory()}}
	withModels(e, m)
	withConnections(e, filepath.Join(t.TempDir(), "settings.json"))
	e.post(t, "/settings/connections/save", profileForm("dev", "dev@host", "7843", "7843"))
	set := e.get(t, "/settings").Body.String()
	mod := e.get(t, "/models").Body.String()
	if n := strings.Count(mod, `action="/models/remove" class="inline" data-confirm=`); n == 0 {
		t.Error("model/runtime removal is not confirmed")
	}
	if !strings.Contains(set, `action="/settings/connections/remove" class="inline" data-confirm=`) {
		t.Error("connection removal is not confirmed")
	}
	if strings.Contains(set+mod+exp, "onsubmit=") {
		t.Error("a form confirms through its own inline handler")
	}
	if !strings.Contains(set, `var ask = (b && b.dataset.confirm) || f.dataset.confirm;`) {
		t.Error("the shell does not honor data-confirm")
	}
}

// The live refresh is bounded and quiet: it skips unchanged slots and hidden
// windows, and announces only the live/stale transition, never each tick.
func TestLiveRefreshIsBoundedAndQuiet(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	for _, want := range []string{`if (document.hidden) return;`, `if (dst.innerHTML !== src.innerHTML) dst.innerHTML = src.innerHTML;`,
		`document.addEventListener("visibilitychange"`, `<span class="vh" id="sync-live" role="status"></span>`} {
		if !strings.Contains(body, want) {
			t.Errorf("runtime page lacks %q", want)
		}
	}
	if regexp.MustCompile(`id="sync-text"[^>]*(role=|aria-live)`).MatchString(body) {
		t.Error("the ticking sync timestamp is a live region")
	}
}

// The history table is a bounded page of the newest entries with keyboard
// links to older pages, while Compare still offers every entry.
func TestHistoryTableIsPaged(t *testing.T) {
	e, _, root := historyEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	e.post(t, "/experiments/save", url.Values{"seq": {"1"}, "label": {"seed"}})
	list, _, err := e.d.hist.List()
	if err != nil || len(list) != 1 {
		t.Fatal(list, err)
	}
	const n = 2*historyPage + 50
	entries := make([]history.Summary, n)
	for i := range entries {
		sm := list[0]
		if i > 0 {
			sm.ID = fmt.Sprintf("20200101T000000Z-%012x", i)
			if err := os.Mkdir(filepath.Join(root, "entries", sm.ID), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		entries[i] = sm
	}
	ix, _ := json.Marshal(map[string]any{"schema": history.IndexSchema, "entries": entries})
	if err := os.WriteFile(filepath.Join(root, "index.json"), ix, 0o644); err != nil {
		t.Fatal(err)
	}
	rows := func(b string) int { return strings.Count(b, `action="/history/open"`) }
	first := e.get(t, "/experiments").Body.String()
	if rows(first) != historyPage || !strings.Contains(first, `href="/experiments?history=100#hist-h">Older</a>`) || strings.Contains(first, ">Newer</a>") {
		t.Fatalf("first page: %d rows", rows(first))
	}
	if got := strings.Count(first, `<option value="`+entries[n-1].ID+`"`); got != 2 {
		t.Errorf("compare offers the oldest entry %d times, want 2", got)
	}
	last := e.get(t, "/experiments?history=200").Body.String()
	if rows(last) != n-2*historyPage || !strings.Contains(last, `href="/experiments?history=100#hist-h">Newer</a>`) || strings.Contains(last, ">Older</a>") ||
		!strings.Contains(last, "201–250 / 250") {
		t.Fatalf("last page: %d rows", rows(last))
	}
	if b := e.get(t, "/experiments?history=99999").Body.String(); rows(b) != n-2*historyPage {
		t.Error("an out-of-range offset is not clamped to the last page")
	}
}
