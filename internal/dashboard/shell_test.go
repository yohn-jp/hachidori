package dashboard

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"

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
