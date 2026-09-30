package dashboard

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

var workspaces = []struct{ path, nav, label string }{
	{"/", "runtime", "Runtime"},
	{"/workbench", "workbench", "Workbench"},
	{"/experiments", "experiments", "Experiments"},
	{"/errors", "evidence", "Evidence"},
	{"/diagnostics", "diagnostics", "Diagnostics"},
}

var navRe = regexp.MustCompile(`(?s)<nav class="sidenav" aria-label="Workspaces">(.*?)</nav>`)

// Every workspace renders inside the one workstation shell: the same
// navigation (with the current entry marked), the compact readiness in the
// title bar and the runtime identity in the status bar.
func TestEveryWorkspaceSharesTheShell(t *testing.T) {
	e := newEnv(t)
	for _, w := range workspaces {
		rec := e.get(t, w.path)
		body := rec.Body.String()
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s: %d", w.path, rec.Code)
		}
		nav := navRe.FindStringSubmatch(body)
		if nav == nil {
			t.Fatalf("%s: no workspace navigation", w.path)
		}
		for _, o := range workspaces {
			link := `<a href="` + o.path + `">` + o.label + `</a>`
			if o.path == w.path {
				link = `<a href="` + o.path + `" aria-current="page">` + o.label + `</a>`
			}
			if !strings.Contains(nav[1], link) {
				t.Errorf("%s: navigation lacks %s", w.path, link)
			}
		}
		if n := strings.Count(nav[1], `aria-current="page"`); n != 1 {
			t.Errorf("%s: %d current navigation entries", w.path, n)
		}
		if strings.Contains(nav[1], ">Errors<") {
			t.Errorf("%s: navigation still names Errors", w.path)
		}
		for _, want := range []string{`<title>` + w.label + ` · Hachidori</title>`, `data-live="shell"`, `data-live="statusbar"`,
			`<span class="dot"></span>READY</span>`, "NVIDIA GeForce RTX 3060", "laya 0.3.21", "laya-base", "http://127.0.0.1:7843",
			"1 needs attention", `fetch("/live"`} {
			if !strings.Contains(body, want) {
				t.Errorf("%s: shell lacks %q", w.path, want)
			}
		}
	}
}

// The shell readiness restates the worker state from /v1/status, and the
// live fragment carries it so every workspace follows state changes.
func TestShellReadinessFollowsRuntimeState(t *testing.T) {
	e := newEnv(t)
	for _, tc := range []struct {
		state string
		ready bool
		tone  string
		word  string
	}{
		{worker.StateReady, true, "ok", "READY"},
		{worker.StateStarting, false, "warn", "starting"},
		{worker.StateFailed, false, "bad", "failed"},
		{worker.StateStopped, false, "idle", "stopped"},
	} {
		e.rt.mu.Lock()
		e.rt.snap.State, e.rt.snap.Ready = tc.state, tc.ready
		e.rt.mu.Unlock()
		want := `<span class="readiness tone-` + tc.tone + `" aria-label="runtime ` + tc.word + `"><span class="dot"></span>` + tc.word + `</span>`
		for _, p := range []string{"/live", "/workbench", "/errors"} {
			if !strings.Contains(e.get(t, p).Body.String(), want) {
				t.Errorf("%s in state %s lacks %s", p, tc.state, want)
			}
		}
	}
}

// Lifecycle outcomes are shown on Runtime; doctor, tunnel and desktop
// outcomes on Diagnostics.
func TestActionsReturnToTheirWorkspace(t *testing.T) {
	e := newEnv(t)
	withDesktop(e, &fakeDesktopPrefs{})
	for p, want := range map[string]string{"/runtime/restart": "/", "/doctor": "/diagnostics", "/tunnel/disconnect": "/diagnostics",
		"/desktop/prefs": "/diagnostics"} {
		rec := e.post(t, p, url.Values{})
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != want {
			t.Errorf("POST %s: %d → %q, want %q", p, rec.Code, rec.Header().Get("Location"), want)
		}
	}
	waitDoctor(t, e)
	if b := e.get(t, "/diagnostics").Body.String(); !strings.Contains(b, "all checks passed") || !strings.Contains(b, `<div class="last tone-`) {
		t.Error("diagnostics does not show the doctor result and last action")
	}
}

// Runtime leads with readiness and the facts needed to use it; detail stays
// available behind disclosure, and Diagnostics keeps failures reachable.
func TestRuntimeLeadsWithReadiness(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	ready := strings.Index(body, `class="readiness-panel tone-ok"`)
	details := strings.Index(body, `data-keep="runtime-details"`)
	if ready < 0 || details < 0 || ready > details {
		t.Fatal("readiness does not lead the runtime workspace")
	}
	for _, want := range []string{`<a class="btn primary" href="/workbench">Open Workbench</a>`, "Recovered from worker failure: worker_crash",
		`<a href="/diagnostics">Open Diagnostics</a>`} {
		if !strings.Contains(body, want) {
			t.Errorf("runtime lacks %q", want)
		}
	}
	diag := e.get(t, "/diagnostics").Body.String()
	for _, want := range []string{`data-keep="worker-failure"`, "Traceback &amp; &lt;b&gt;", `id="transport"`, `action="/doctor"`} {
		if !strings.Contains(diag, want) {
			t.Errorf("diagnostics lacks %q", want)
		}
	}
	// Not running: Start becomes the primary action and Workbench is not offered.
	e.rt.mu.Lock()
	e.rt.run, e.rt.snap.State, e.rt.snap.Ready = false, worker.StateStopped, false
	e.rt.mu.Unlock()
	body = e.get(t, "/").Body.String()
	if !strings.Contains(body, `<button type="submit" class="btn primary">Start</button>`) || strings.Contains(body, "Open Workbench") {
		t.Error("stopped runtime does not lead with Start")
	}
}

// The workstation defines its narrow-window layout and honours reduced
// motion; it has no second design system.
func TestShellLayoutRules(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	for _, want := range []string{
		`@media (max-width: 820px)`, `grid-template-areas: "bar" "nav" "main" "status"`, `.sidenav { flex-direction: row;`,
		`@media (max-width: 1180px)`, `.split, .split.narrow-side { grid-template-columns: minmax(0, 1fr); }`,
		`@media (prefers-reduced-motion: reduce)`, `animation: none !important; transition: none !important;`,
		`@media (prefers-color-scheme: light)`, `:focus-visible { outline: 2px solid var(--accent)`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("stylesheet lacks %q", want)
		}
	}
	for _, old := range []string{`class="topbar"`, `class="hero`, `class="panel"`, `class="qcard"`, "host console", "backdrop-filter", "linear-gradient"} {
		if strings.Contains(body, old) {
			t.Errorf("obsolete presentation %q remains", old)
		}
	}
	if strings.Contains(body, `placeholder="nixos-dev"`) {
		t.Error("new connection form contains a developer-specific placeholder")
	}
	if !strings.Contains(body, `max-width: min(100%, 88rem)`) {
		t.Error("workspace does not constrain its wide-screen measure")
	}
	if n := strings.Count(body, "<style>"); n != 1 {
		t.Errorf("%d stylesheets", n)
	}
}

var submitRe = regexp.MustCompile(`<button type="submit"[^>]*>`)

// Workbench: State + Question -> Run -> Result is the dominant composition;
// the exact request, definition load and export stay reachable as secondary
// disclosure, and Run is the form's default (first) submit control.
func TestWorkbenchCompositionKeepsTheLoopPrimary(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	v := form("state", []string{"scope", "In scope?", "yes", "no"}, []string{"risk", "Risky?", "low", "high"})
	body := e.post(t, "/workbench", with(v, "op", "run")).Body.String()
	wb := body[strings.Index(body, `action="/workbench"`):]
	if first := submitRe.FindString(wb); !strings.Contains(first, `value="run"`) || !strings.Contains(first, "primary") {
		t.Fatalf("first submit control is %s, want the primary Run", first)
	}
	state, result := strings.Index(body, `id="state-h"`), strings.Index(body, `id="result-h"`)
	res0, wire := strings.Index(body, `aria-label="result for question 0"`), strings.Index(body, `data-keep="wire"`)
	if state < 0 || result < state || res0 < result || wire < res0 {
		t.Fatal("workbench is not State/Question → Result, then request detail")
	}
	for _, want := range []string{`<span class="choice">yes</span>`, `confidence <strong>0.8000</strong>`, `<tr class="chosen"><td class="label">yes</td>`,
		`<summary>Load Question Definition`, `<summary>Definition identity &amp; export</summary>`, `name="op" value="export:1"`,
		`name="op" value="preview"`, `name="op" value="add"`, `name="op" value="remove:1"`, `name="op" value="choices:0"`} {
		if !strings.Contains(body, want) {
			t.Errorf("workbench lacks %q", want)
		}
	}
	if strings.Contains(body, `data-keep="wire" open`) {
		t.Error("request JSON disclosure open after a run")
	}
	if p := e.post(t, "/workbench", with(v, "op", "preview")).Body.String(); !strings.Contains(p, `data-keep="wire" open`) || !strings.Contains(p, "Request preview") {
		t.Error("preview does not open the exact request JSON")
	}
}

// Experiments: a finished run leads with its state, progress and quality
// signals, hands off to Evidence, and keeps identity and replay guidance in
// the secondary evidence-identity disclosure.
func TestExperimentRunHierarchyAndEvidenceHandoff(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	if x := waitExp(t, e); x.State != ExpSucceeded {
		t.Fatalf("experiment %+v", x)
	}
	body := e.get(t, "/experiments").Body.String()
	order := []string{`class="run tone-ok"`, `<span class="dot"></span>succeeded</p>`, `class="bar progress" role="progressbar"`,
		`<dt>Accuracy</dt>`, `<dt>ECE <small>15 bins</small></dt>`, `<dt>Request errors</dt>`, `<dt>Cases</dt>`,
		`action="/errors/use-experiment"`, `action="/experiments/export"`, `aria-label="per-question statistics"`,
		`data-keep="evidence-identity"`, "hachidori replay -dataset", `id="setup-h"`}
	at := 0
	for _, s := range order {
		i := strings.Index(body[at:], s)
		if i < 0 {
			t.Fatalf("experiments lacks %q after position %d", s, at)
		}
		at += i
	}
	if !strings.Contains(body, `<button type="submit" class="btn primary">Investigate in Evidence</button>`) ||
		!strings.Contains(body, `<button type="submit" data-run class="btn">Run experiment</button>`) {
		t.Error("finished run does not make the Evidence handoff the primary action")
	}
	forms := postForms(t, e, body)
	for _, a := range []string{"/experiments/run", "/experiments/export", "/errors/use-experiment"} {
		if _, ok := forms[a]; !ok {
			t.Errorf("no form for %s", a)
		}
	}
	if !strings.Contains(forms["/experiments/run"], `formaction="/experiments/preflight"`) {
		t.Error("preflight control missing")
	}
	rec := e.post(t, "/errors/use-experiment", url.Values{"seq": {"1"}})
	if rec.Code != http.StatusSeeOther {
		t.Fatalf("handoff: %d", rec.Code)
	}
	if b := e.get(t, "/errors").Body.String(); !strings.Contains(b, "experiment #1") || !strings.Contains(b, `<dt>Wrong</dt><dd>1</dd>`) {
		t.Error("Evidence does not open the handed-off experiment")
	}
}

// Evidence: failures and their concentration lead; the observation table,
// filters, detail, metadata and filtered export remain.
func TestEvidenceInvestigationComposition(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	e.post(t, "/errors/use-experiment", url.Values{"seq": {"1"}})
	var idx string
	for i, o := range e.d.errs.get().Report.Results {
		if !o.Correct {
			idx = itoa(i)
		}
	}
	body := e.get(t, "/errors?th=0.8&obs="+idx).Body.String()
	order := []string{`<dt>Wrong</dt>`, `<dt>High-confidence wrong</dt><dd class="bad-t">1</dd>`, "Show high-confidence wrong",
		"Concentration by question", `aria-label="observations"`, `id="detail"`, `data-keep="evidence-source"`}
	at := 0
	for _, s := range order {
		i := strings.Index(body[at:], s)
		if i < 0 {
			t.Fatalf("evidence lacks %q after position %d", s, at)
		}
		at += i
	}
	for _, want := range []string{`<tr aria-current="true"><td class="num"><a href="/errors?`, "Observation " + idx, `<span class="badge tone-bad">high-confidence wrong</span>`,
		`<span class="tag">(expected)</span>`, `href="/errors?q=x&amp;outcome=wrong&amp;th=0.8#obs"`, `method="get" action="/errors"`,
		`name="exp"`, `name="pred"`, `name="min"`, `name="max"`, `name="eclass"`, `name="sort"`, `name="desc"`, `name="th"`, "analysis control, not a policy"} {
		if !strings.Contains(body, want) {
			t.Errorf("evidence lacks %q", want)
		}
	}
	forms := postForms(t, e, body)
	for _, a := range []string{"/errors/open", "/errors/use-experiment", "/errors/export"} {
		if _, ok := forms[a]; !ok {
			t.Errorf("no form for %s", a)
		}
	}
	// Without a report, opening evidence is the workspace's primary content.
	e2, _ := newWorkbenchEnv(t)
	if b := e2.get(t, "/errors").Body.String(); !strings.Contains(b, `id="open-h"`) || !strings.Contains(b, `action="/errors/open"`) || strings.Contains(b, `aria-label="observations"`) {
		t.Error("empty Evidence workspace does not lead with opening a report")
	}
}

type fakeSettings struct {
	def settings.Defaults
	err error
}

func (f *fakeSettings) Defaults() (settings.Defaults, error) { return f.def, nil }
func (f *fakeSettings) SetDefaults(d settings.Defaults) error {
	if f.err != nil {
		return f.err
	}
	f.def = d
	return nil
}

func withSettings(e *env, s Settings, d Desktop) {
	cfg := e.d.cfg
	cfg.Settings, cfg.Desktop = s, d
	e.d = New(cfg)
}

// The Settings workspace exists only when a settings authority is hosted, is
// part of the shell navigation and shows desktop preferences and defaults.
func TestSettingsWorkspaceInShell(t *testing.T) {
	e := newEnv(t)
	if strings.Contains(navRe.FindString(e.get(t, "/").Body.String()), "/settings") {
		t.Error("serve/dashboard navigation offers Settings")
	}
	if rec := e.get(t, "/settings"); rec.Code != http.StatusNotFound {
		t.Fatalf("settings route without authority: %d", rec.Code)
	}
	prefs := &fakeDesktopPrefs{signIn: true, minimized: true}
	withSettings(e, &fakeSettings{def: settings.Defaults{Device: "cpu", Model: "laya-base"}}, prefs)
	rec := e.get(t, "/settings")
	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /settings: %d", rec.Code)
	}
	if !strings.Contains(navRe.FindString(body), `<a href="/settings" aria-current="page">Settings</a>`) {
		t.Error("navigation lacks the current Settings entry")
	}
	for _, want := range []string{`name="start_at_sign_in" value="1" checked`, `name="start_minimized" value="1" checked`,
		`<option value="cpu" selected>`, `<option value="laya-base" selected>`, `action="/settings/defaults"`, `action="/desktop/prefs"`} {
		if !strings.Contains(body, want) {
			t.Errorf("settings lacks %q", want)
		}
	}
	// Existing workflows stay reachable: Diagnostics keeps the desktop panel.
	if !strings.Contains(e.get(t, "/diagnostics").Body.String(), `action="/desktop/prefs"`) {
		t.Error("diagnostics lost the desktop panel")
	}
}

// Saving defaults and desktop preferences returns to Settings and never
// touches the runtime lifecycle.
func TestSettingsSaveDoesNotTouchRuntime(t *testing.T) {
	e := newEnv(t)
	fs, prefs := &fakeSettings{}, &fakeDesktopPrefs{}
	withSettings(e, fs, prefs)
	rec := e.post(t, "/settings/defaults", url.Values{"device": {"cpu"}, "model": {"laya-base"}, "return": {"settings"}})
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/settings" {
		t.Fatalf("defaults save: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if fs.def != (settings.Defaults{Device: "cpu", Model: "laya-base"}) {
		t.Fatalf("%+v", fs.def)
	}
	rec = e.post(t, "/desktop/prefs", url.Values{"start_minimized": {"1"}, "return": {"settings"}})
	if rec.Header().Get("Location") != "/settings" || !prefs.minimized {
		t.Fatalf("desktop save: %s %+v", rec.Header().Get("Location"), prefs)
	}
	// Without the return marker the Diagnostics panel keeps its redirect.
	if rec = e.post(t, "/desktop/prefs", nil); rec.Header().Get("Location") != "/diagnostics" {
		t.Fatalf("diagnostics redirect: %s", rec.Header().Get("Location"))
	}
	e.rt.mu.Lock()
	calls := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(calls) != 0 {
		t.Fatalf("lifecycle touched: %v", calls)
	}
	// Refused values are shown; forged tokens are refused.
	fs.err = errors.New("model refused")
	e.post(t, "/settings/defaults", url.Values{"device": {"tpu"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "model refused") {
		t.Fatalf("%+v", a)
	}
	if rec := e.post(t, "/settings/defaults", url.Values{"token": {"forged"}, "device": {"cpu"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
}

// withConnections hosts the Development Connections profiles in the Settings
// workspace, persisted by the real settings authority at path.
func withConnections(e *env, path string) *settings.Store {
	st := &settings.Store{Path: path}
	cfg := e.d.cfg
	cfg.Connections = st
	e.d = New(cfg)
	return st
}

func profileForm(name, dest string, remote, local string) url.Values {
	return url.Values{"name": {name}, "destination": {dest}, "remote_bind": {"127.0.0.1"}, "remote_port": {remote}, "local_port": {local}, "return": {"settings"}}
}

// A profile can be saved, connected through tunnel.Manager, shown with its
// exact copy-ready endpoint, disconnected, and restored after a restart, with
// no credential anywhere in the stored record.
func TestDevelopmentConnectionLifecycle(t *testing.T) {
	e := newEnv(t)
	path := filepath.Join(t.TempDir(), "settings.json")
	withConnections(e, path)

	if !strings.Contains(navRe.FindString(e.get(t, "/").Body.String()), "/settings") {
		t.Error("navigation lacks Settings when connections are hosted")
	}
	rec := e.post(t, "/settings/connections/save", profileForm("nixos-dev", "dev@nixos", "7843", "7843"))
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/settings" {
		t.Fatalf("save: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if a := e.lastAction(t); !a.OK || e.tun.Status().State != tunnel.StateIdle {
		t.Fatalf("saving must not connect: %+v %+v", a, e.tun.Status())
	}
	page := e.get(t, "/settings").Body.String()
	for _, want := range []string{`data-connection="nixos-dev"`, "HACHIDORI_ENDPOINT=http://127.0.0.1:7843",
		`action="/settings/connections/connect"`, `action="/settings/connections/reconnect"`, `action="/settings/connections/remove"`, `action="/tunnel/disconnect"`} {
		if !strings.Contains(page, want) {
			t.Errorf("settings lacks %q", want)
		}
	}

	e.post(t, "/settings/connections/connect", url.Values{"name": {"nixos-dev"}, "return": {"settings"}})
	st := e.tun.Status()
	if a := e.lastAction(t); !a.OK || st.State != tunnel.StateRunning || st.Spec == nil || st.Spec.Destination != "dev@nixos" {
		t.Fatalf("connect: %+v %+v", a, st)
	}
	page = e.get(t, "/settings").Body.String()
	for _, want := range []string{"connected", "HACHIDORI_ENDPOINT=http://127.0.0.1:7843", fmt.Sprint(st.PID)} {
		if !strings.Contains(page, want) {
			t.Errorf("connected settings lacks %q", want)
		}
	}

	// A second profile cannot run beside the first (one managed tunnel), and
	// the connected profile cannot be removed.
	e.post(t, "/settings/connections/save", profileForm("other", "dev@other", "7900", "7843"))
	e.post(t, "/settings/connections/connect", url.Values{"name": {"other"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "disconnect it first") || e.tun.Status().PID != st.PID {
		t.Fatalf("second tunnel: %+v", a)
	}
	e.post(t, "/settings/connections/remove", url.Values{"name": {"nixos-dev"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "disconnect it first") {
		t.Fatalf("removed a connected profile: %+v", a)
	}

	// Reconnect goes through the manager: a new ssh child for the profile.
	e.post(t, "/settings/connections/reconnect", url.Values{"name": {"other"}})
	if a, st2 := e.lastAction(t), e.tun.Status(); !a.OK || st2.State != tunnel.StateRunning || st2.PID == st.PID || st2.Endpoint != "http://127.0.0.1:7900" {
		t.Fatalf("reconnect: %+v %+v", a, st2)
	}
	rec = e.post(t, "/tunnel/disconnect", url.Values{"return": {"settings"}})
	if rec.Header().Get("Location") != "/settings" || e.tun.Status().State != tunnel.StateStopped {
		t.Fatalf("disconnect: %s %+v", rec.Header().Get("Location"), e.tun.Status())
	}

	// After an application restart the profiles come back from disk.
	e2 := newEnv(t)
	withConnections(e2, path)
	page = e2.get(t, "/settings").Body.String()
	for _, want := range []string{`data-connection="nixos-dev"`, `data-connection="other"`, "HACHIDORI_ENDPOINT=http://127.0.0.1:7900"} {
		if !strings.Contains(page, want) {
			t.Errorf("restored settings lacks %q", want)
		}
	}
	b, _ := os.ReadFile(path)
	for _, s := range []string{"key", "pass", "identity", "secret", "token", "known_hosts", "BEGIN"} {
		if strings.Contains(strings.ToLower(string(b)), strings.ToLower(s)) {
			t.Fatalf("settings contain %q: %s", s, b)
		}
	}
	e.post(t, "/settings/connections/remove", url.Values{"name": {"nixos-dev"}})
	if a := e.lastAction(t); !a.OK {
		t.Fatalf("remove: %+v", a)
	}
	if cs, _ := (&settings.Store{Path: path}).Connections(); len(cs) != 1 || cs[0].Name != "other" {
		t.Fatalf("after remove: %+v", cs)
	}
}

// Profiles enforce the same loopback-only validation as the manager and store
// nothing when rejected; a start failure is shown, not fixed.
func TestDevelopmentConnectionValidationAndFailure(t *testing.T) {
	e := newEnv(t)
	path := filepath.Join(t.TempDir(), "settings.json")
	withConnections(e, path)
	for name, form := range map[string]url.Values{
		"public bind": {"name": {"n"}, "destination": {"dev@nixos"}, "remote_bind": {"0.0.0.0"}, "remote_port": {"7843"}, "local_port": {"7843"}},
		"option dest": {"name": {"n"}, "destination": {"-oProxyCommand=calc"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}},
		"bad name":    {"name": {"a b"}, "destination": {"dev@nixos"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}},
		"not a port":  {"name": {"n"}, "destination": {"dev@nixos"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"x"}, "local_port": {"7843"}},
	} {
		e.post(t, "/settings/connections/save", form)
		if a := e.lastAction(t); a.OK {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("rejected input wrote settings: %v", err)
	}
	e.post(t, "/settings/connections/connect", url.Values{"name": {"missing"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "does not exist") {
		t.Fatalf("unknown profile: %+v", a)
	}

	e.post(t, "/settings/connections/save", profileForm("nixos-dev", "dev@nixos", "7843", "7843"))
	e.tun.SSH = "hachidori-no-such-ssh-client"
	e.post(t, "/settings/connections/connect", url.Values{"name": {"nixos-dev"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "ssh client not found") {
		t.Fatalf("failure not reported: %+v", a)
	}
	if page := e.get(t, "/settings").Body.String(); !strings.Contains(page, "ssh client not found") || !strings.Contains(page, "exited") {
		t.Error("settings does not show the actionable failure")
	}
}

// The Diagnostics form is not a second transport path: it saves the same
// profile in the same settings authority and connects through the same
// manager, and it no longer writes the legacy dashboard.json.
func TestDiagnosticsTunnelFormDelegatesToProfiles(t *testing.T) {
	e := newEnv(t)
	path := filepath.Join(t.TempDir(), "settings.json")
	withConnections(e, path)
	if body := e.get(t, "/diagnostics").Body.String(); !strings.Contains(body, `name="name" value="default"`) {
		t.Fatal("diagnostics form lacks the connection name")
	}
	e.post(t, "/tunnel/connect", url.Values{"name": {"nixos-dev"}, "destination": {"dev@nixos"}, "remote_bind": {"0.0.0.0"}, "remote_port": {"7843"}, "local_port": {"7843"}})
	if a := e.lastAction(t); a.OK || e.tun.Status().State != tunnel.StateIdle {
		t.Fatalf("public bind accepted: %+v", a)
	}
	e.post(t, "/tunnel/connect", url.Values{"destination": {"dev@nixos"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}})
	cs, err := (&settings.Store{Path: path}).Connections()
	if a := e.lastAction(t); !a.OK || err != nil || len(cs) != 1 || cs[0].Name != "default" || e.tun.Status().State != tunnel.StateRunning {
		t.Fatalf("connect: %+v %+v %v", a, cs, err)
	}
	// The profile is the same one Settings connects, and the running
	// spec is the profile's spec: one authority.
	if *e.tun.Status().Spec != cs[0].Spec() {
		t.Fatalf("manager runs %+v, profile is %+v", *e.tun.Status().Spec, cs[0])
	}
	page := e.get(t, "/settings").Body.String()
	if !strings.Contains(page, `data-connection="default"`) || !strings.Contains(page, "connected") {
		t.Error("settings does not show the Diagnostics connection as a running profile")
	}
	if _, err := os.Stat(e.d.cfg.PrefsPath); !os.IsNotExist(err) {
		t.Fatalf("dashboard.json written beside the profile store: %v", err)
	}
}

// Legacy saved tunnel form values prefill the form while no profile exists and
// are never rewritten or left as competing state.
func TestLegacyTunnelPrefsArePrefillOnlyWhenProfilesAreHosted(t *testing.T) {
	e := newEnv(t)
	legacy := []byte(`{"tunnel":{"destination":"dev@legacy","remote_bind":"127.0.0.1","remote_port":7000,"local_port":7843}}`)
	if err := os.WriteFile(e.d.cfg.PrefsPath, legacy, 0o644); err != nil {
		t.Fatal(err)
	}
	withConnections(e, filepath.Join(t.TempDir(), "settings.json"))
	if f := e.d.formDefaults(); f.Destination != "dev@legacy" || f.RemotePort != 7000 {
		t.Fatalf("legacy prefill: %+v", f)
	}
	e.post(t, "/settings/connections/save", profileForm("nixos-dev", "dev@nixos", "7843", "7843"))
	if f := e.d.formDefaults(); f.Destination != "dev@nixos" {
		t.Fatalf("saved profile does not take over: %+v", f)
	}
	e.post(t, "/tunnel/connect", url.Values{"destination": {"dev@nixos"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}})
	if b, _ := os.ReadFile(e.d.cfg.PrefsPath); string(b) != string(legacy) {
		t.Fatalf("legacy prefs rewritten: %s", b)
	}
}
