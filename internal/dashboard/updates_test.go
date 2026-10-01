package dashboard

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/update"
)

// fakeUpdates records every call. Only Check and Download are network
// actions in the real subsystem; the test asserts the dashboard calls them
// only when the operator posts them.
type fakeUpdates struct {
	mu       sync.Mutex
	st       update.Status
	statuses int
	channels []update.Channel
	checks   int
	dls      []string
	installs int
	err      error
}

func (f *fakeUpdates) Status() update.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.statuses++
	return f.st
}

func (f *fakeUpdates) SetChannel(c update.Channel) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.channels = append(f.channels, c)
	if f.err == nil {
		f.st.Channel, f.st.Check = c, nil
	}
	return f.err
}

func (f *fakeUpdates) Check(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.checks++
	return f.err
}
func (f *fakeUpdates) Download(tag string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.dls = append(f.dls, tag)
	return f.err
}
func (f *fakeUpdates) Install() error { f.mu.Lock(); defer f.mu.Unlock(); f.installs++; return f.err }

func (f *fakeUpdates) actions() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.checks + len(f.dls) + f.installs
}

func withUpdates(e *env, u Updates) {
	cfg := e.d.cfg
	cfg.Updates = u
	e.d = New(cfg)
}

func v(s string) update.Version { x, _ := update.ParseVersion(s); return x }

func rel(tag, relation, problem string) update.Candidate {
	return update.Candidate{Release: update.Release{Tag: tag, Version: v(tag), Problem: problem, Published: time.Date(2026, 10, 1, 13, 0, 0, 0, time.UTC)}, Relation: relation}
}

func baseStatus() update.Status {
	return update.Status{Supported: true, Channel: update.Stable, Installed: update.InstalledView{Version: "0.2.5-dev", Known: true, SHA256: strings.Repeat("a", 64)}}
}

func TestUpdatesSurfaceExistsOnlyWhenHosted(t *testing.T) {
	e := newEnv(t)
	if rec := e.get(t, "/settings/updates"); rec.Code != http.StatusNotFound {
		t.Fatalf("updates route without a subsystem: %d", rec.Code)
	}
	withUpdates(e, &fakeUpdates{st: baseStatus()})
	rec := e.get(t, "/settings/updates")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "<h1>Updates</h1>") {
		t.Fatalf("%d", rec.Code)
	}
	if !strings.Contains(navRe.FindString(e.get(t, "/").Body.String()), "/settings") {
		t.Error("hosting Updates alone must expose Settings in the shell")
	}
	if !strings.Contains(e.get(t, "/settings").Body.String(), `href="/settings/updates"`) {
		t.Error("Settings does not link to Updates")
	}
}

// The observable invariant at the UI boundary: rendering any page, opening
// Settings and Updates, and changing the channel call no network action.
func TestOpeningSettingsAndUpdatesAndChangingChannelCallsNoNetworkAction(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	withSettings(e, &fakeSettings{}, nil)
	for _, p := range []string{"/", "/settings", "/settings/updates", "/diagnostics", "/workbench", "/experiments", "/errors", "/settings/updates"} {
		if rec := e.get(t, p); rec.Code != http.StatusOK {
			t.Fatalf("%s: %d", p, rec.Code)
		}
	}
	for _, c := range []string{"development", "stable", "development"} {
		rec := e.post(t, "/settings/updates/channel", url.Values{"channel": {c}})
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/settings/updates" {
			t.Fatalf("channel %s: %d %s", c, rec.Code, rec.Header().Get("Location"))
		}
	}
	if f.actions() != 0 {
		t.Fatalf("network actions without an explicit Check or Download: checks=%d downloads=%v installs=%d", f.checks, f.dls, f.installs)
	}
	if len(f.channels) != 3 || f.channels[2] != update.Development {
		t.Errorf("channels %v", f.channels)
	}
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, "nothing was checked") {
		t.Errorf("%+v", a)
	}
	if rec := e.post(t, "/settings/updates/channel", url.Values{"channel": {"nightly"}}); rec.Code != http.StatusSeeOther || e.lastAction(t).OK || len(f.channels) != 3 {
		t.Errorf("an unknown channel reached the subsystem: %v", f.channels)
	}
}

func TestExplicitActionsAreForwardedExactlyOnce(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	e.post(t, "/settings/updates/check", nil)
	if f.checks != 1 || f.actions() != 1 {
		t.Fatalf("check: %+v", f)
	}
	e.post(t, "/settings/updates/download", url.Values{"tag": {" 0.2.6-dev "}})
	if len(f.dls) != 1 || f.dls[0] != "0.2.6-dev" || f.actions() != 2 {
		t.Fatalf("download: %v", f.dls)
	}
	e.post(t, "/settings/updates/install", nil)
	if f.installs != 1 {
		t.Fatalf("install: %d", f.installs)
	}
	f.err = errors.New("GitHub answered 403 Forbidden")
	rec := e.post(t, "/settings/updates/check", nil)
	if rec.Header().Get("Location") != "/settings/updates" {
		t.Errorf("location %s", rec.Header().Get("Location"))
	}
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "403") {
		t.Errorf("a failed check must be shown: %+v", a)
	}
	// A cross-site or token-less post never reaches the subsystem.
	n := f.actions()
	r := httpPost(e, "/settings/updates/check", "")
	if r.Code != http.StatusForbidden || f.actions() != n {
		t.Errorf("tokenless post: %d", r.Code)
	}
	if rec := e.get(t, "/settings/updates/check"); rec.Code == http.StatusOK || f.actions() != n {
		t.Errorf("GET performed an action: %d", rec.Code)
	}
}

func TestUpdatesViewRendersOnlyWhatAnExplicitCheckFound(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	body := e.get(t, "/settings/updates").Body.String()
	for _, want := range []string{`id="updates-installed-version">0.2.5-dev`, `Check for updates`, `value="stable" selected`, `Nothing downloaded and verified yet.`, "never"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(body, `id="updates-releases"`) || strings.Contains(body, `action="/settings/updates/download"`) {
		t.Error("releases or a Download button are shown before any explicit check")
	}

	at := time.Date(2026, 10, 1, 14, 0, 0, 0, time.UTC)
	f.st.Check = &update.CheckResult{Time: at, Channel: update.Development, Installed: f.st.Installed, Releases: []update.Candidate{
		rel("0.2.7-dev", update.RelNewer, "asset not found: hachidori-windows-amd64.exe.sha256"),
		rel("0.2.6-dev", update.RelNewer, ""), rel("0.2.5-dev", update.RelCurrent, ""), rel("0.2.1-dev", update.RelOlder, ""),
	}}
	f.st.LastCheck = &update.LastCheck{Time: at, Channel: update.Development, OK: true, Latest: "0.2.7-dev"}
	body = e.get(t, "/settings/updates").Body.String()
	if n := strings.Count(body, `action="/settings/updates/download"`); n != 1 || !strings.Contains(body, `name="tag" value="0.2.6-dev"`) {
		t.Errorf("Download must be offered only for the installable newer release: %d", n)
	}
	for _, want := range []string{"0.2.7-dev", "asset not found", "Not offered: an update never goes backwards."} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q", want)
		}
	}
	// Unknown installed version: the operator is told releases cannot be compared.
	f.st.Installed = update.InstalledView{}
	f.st.Check.Releases = []update.Candidate{rel("0.2.6-dev", update.RelUnknown, "")}
	body = e.get(t, "/settings/updates").Body.String()
	if !strings.Contains(body, "not known, so releases cannot be compared") || !strings.Contains(body, "unknown (identified by an explicit check)") {
		t.Error("unknown installed version is not explained")
	}
	if f.actions() != 0 {
		t.Fatal("rendering performed a network action")
	}
}

func TestUpdatesFailureStatesAreActionable(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	f.st.LastCheck = &update.LastCheck{Time: time.Now(), Channel: update.Stable, Class: update.ClassNetwork, Message: "release metadata could not be retrieved: GitHub answered 403 Forbidden"}
	body := e.get(t, "/settings/updates").Body.String()
	if !strings.Contains(body, `id="updates-check-failure"`) || !strings.Contains(body, "403") || !strings.Contains(body, "installed Hachidori is unchanged") {
		t.Error("check failure not shown with its hint")
	}
	f.st.LastCheck = nil
	f.st.Last = &update.Operation{Kind: update.KindDownload, Tag: "0.2.6-dev", Plan: []string{"checksum", "download", "verify"}, Phases: []string{"checksum", "download", "verify"},
		Phase: "verify", Started: time.Now().Add(-3 * time.Second), Finished: time.Now(),
		Failure: &update.Failure{Class: update.ClassChecksumMismatch, Phase: "verify", Step: "verifying", Message: "SHA-256 mismatch"}}
	body = e.get(t, "/settings/updates").Body.String()
	for _, want := range []string{"FAILED", "SHA-256 mismatch", "The installed executable was not changed.", `id="updates-op-hint"`, "does not match the release&#39;s checksum"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in %s", want, body[strings.Index(body, `id="updates-op"`):])
		}
	}
	if strings.Contains(body, "active runtime and model were not changed") {
		t.Error("an update failure reports the runtime message")
	}
	f.st.Last = nil
	f.st.Result = &update.Result{Tag: "0.2.6-dev", Outcome: update.OutcomeFailed, Message: "the current executable could not be moved aside", Time: time.Now()}
	body = e.get(t, "/settings/updates").Body.String()
	if !strings.Contains(body, "NOT UPDATED") || !strings.Contains(body, "previous executable was kept or restored") {
		t.Error("replacement failure not shown")
	}
	f.st.Result = &update.Result{Tag: "0.2.6-dev", Outcome: update.OutcomeApplied, Message: "replaced", Time: time.Now()}
	if !strings.Contains(e.get(t, "/settings/updates").Body.String(), "UPDATED") {
		t.Error("applied update not shown")
	}
}

// Download progress and verification reuse the long-running operation view.
func TestUpdatesDownloadProgressUsesTheSharedOperationView(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	f.st.Busy = &update.Operation{Kind: update.KindDownload, Tag: "0.2.6-dev", Plan: []string{"checksum", "download", "verify"}, Phases: []string{"checksum", "download"},
		Phase: "download", Step: update.StepDownloading, Detail: update.ExeAsset, Done: 8 << 20, Total: 16 << 20, Started: time.Now().Add(-2 * time.Second)}
	body := e.get(t, "/settings/updates").Body.String()
	for _, want := range []string{`id="models-busy"`, `role="progressbar"`, `aria-valuenow="50"`, "8.0 MiB of 16.0 MiB", "phase 2 of 3", "Checksum file", "Verification"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q", want)
		}
	}
	// The refresh script reads the local page only; it posts nothing.
	script := body[strings.Index(body, "setInterval"):]
	script = script[:strings.Index(script, "</script>")]
	if strings.Contains(script, "POST") || strings.Contains(script, "/settings/updates/") || !strings.Contains(script, "fetch(location.pathname") {
		t.Error("the refresh script must only re-read the local page")
	}
	if !strings.Contains(body, `class="btn primary" disabled`) && !strings.Contains(body, " disabled>") {
		t.Error("Check is not disabled while a download runs")
	}
	if f.actions() != 0 {
		t.Fatal("rendering performed a network action")
	}
}

func TestUpdatesReadyStateOffersRestartOnlyWhenInstallable(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	ready := &update.Ready{Tag: "0.2.6-dev", SHA256: strings.Repeat("b", 64), Target: `C:\app\hachidori.exe`, Created: time.Now()}
	f.st.Ready = ready
	body := e.get(t, "/settings/updates").Body.String()
	if !strings.Contains(body, `action="/settings/updates/install" class="inline" id="updates-install" data-confirm=`) || !strings.Contains(body, "Restart &amp; update") {
		t.Error("a verified update does not offer a confirmed Restart & update")
	}
	f.st.ReadyProblem = "release 0.2.6-dev is not offered on the stable channel"
	body = e.get(t, "/settings/updates").Body.String()
	if strings.Contains(body, `id="updates-install"`) || !strings.Contains(body, `id="updates-ready-problem"`) {
		t.Error("an update that cannot be installed offers Restart & update")
	}
	f.st.ReadyProblem, f.st.Supported = "", false
	if strings.Contains(e.get(t, "/settings/updates").Body.String(), `id="updates-install"`) {
		t.Error("unsupported platform offers Restart & update")
	}
	f.st.Ready = nil
	if !strings.Contains(e.get(t, "/settings/updates").Body.String(), `id="updates-not-ready"`) {
		t.Error("no ready update not stated")
	}
}

func TestUpdatesPageRendersInJapanese(t *testing.T) {
	e := newEnv(t)
	f := &fakeUpdates{st: baseStatus()}
	withUpdates(e, f)
	withSettings(e, &fakeSettings{loc: "ja"}, nil)
	body := e.get(t, "/settings/updates").Body.String()
	for _, want := range []string{"<h1>アップデート</h1>", "アップデートを確認", "再起動して更新"} {
		if want == "再起動して更新" {
			continue // only shown when a verified update is ready
		}
		if !strings.Contains(body, want) {
			t.Errorf("missing %q", want)
		}
	}
}

// httpPost posts with no form token.
func httpPost(e *env, path, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "http://127.0.0.1:7844"+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", "http://127.0.0.1:7844")
	e.d.ServeHTTP(rec, req)
	return rec
}
