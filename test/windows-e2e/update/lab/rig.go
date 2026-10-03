// Package lab holds the update E2E scenarios and the rig they run on.
//
// The rig composes the real update subsystem exactly as the Windows desktop
// does, with only its existing injection points substituted:
//
//   - update.Service (the code compiled into hachidori.exe) is created with
//     Transport = the local deterministic release stub, so the fixed GitHub
//     authority URLs are served by a local TLS listener and never reach a
//     network;
//   - the Settings > Updates surface is the real internal/dashboard handler
//     bound to that service, driven over real loopback HTTP with the form token,
//     so the UI projection (phases, bytes, disabled actions, failure text) is
//     the production one;
//   - settings are stored by the real internal/settings store in a throwaway
//     profile folder, the home is a throwaway folder, and the executable the
//     update replaces is a disposable copy of the certified candidate;
//   - the replacement itself runs the candidate's own `apply-update` helper
//     (update.Service.Install copies the disposable executable into
//     state/updates/helper and starts it), so the bytes certified are the
//     bytes that replace.
//
// Scenarios that do not need a Windows executable run anywhere and are also
// ordinary portable tests of this package.
package lab

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/update"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/disposable"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/releasestub"
)

// Reporter receives the scenario's bounded evidence. *e2e.Scenario satisfies
// it; portable runs use a quiet reporter.
type Reporter interface {
	Logf(format string, args ...any)
	Attach(name string, data []byte, home string)
}

type quiet struct{ t testing.TB }

func (q quiet) Logf(format string, args ...any) { q.t.Logf(format, args...) }
func (quiet) Attach(string, []byte, string)     {}

// Quiet is the Reporter of a portable run: it logs to the test and keeps no
// evidence files.
func Quiet(t testing.TB) Reporter { return quiet{t} }

// Deadlines. Every wait is on observable state with one of these bounds; a
// scenario never sleeps as its oracle.
const (
	shortWait = 30 * time.Second
	longWait  = 120 * time.Second
	pollEvery = 10 * time.Millisecond
)

// until polls cond until it holds or the deadline passes.
func until(t testing.TB, what string, within time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s", within, what)
		}
		time.Sleep(pollEvery)
	}
}

func receive(t testing.TB, what string, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(shortWait):
		t.Fatalf("timed out after %s waiting for %s", shortWait, what)
	}
}

// Exe is a deterministic fixture Windows executable: the PE signature the
// updater requires followed by size-2 reproducible bytes derived from seed.
func Exe(seed string, size int) []byte {
	if size < 2 {
		size = 2
	}
	b := make([]byte, size)
	b[0], b[1] = 'M', 'Z'
	x := uint64(14695981039346656037)
	for _, c := range []byte(seed) {
		x = (x ^ uint64(c)) * 1099511628211
	}
	for i := 2; i < size; i++ {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		b[i] = byte(x)
	}
	return b
}

// rigOptions configure a rig.
type rigOptions struct {
	// exe is the disposable executable's content; exeFrom copies a real file
	// (the candidate) instead.
	exe     []byte
	exeFrom string
	// helperStarter replaces how the replacement helper is started; nil records
	// the start and runs nothing.
	helperStarter func(exe string, args []string) error
}

// rig is one scenario's disposable installation: a home, a profile with the
// settings file, an executable, the release stub, the update service and the
// dashboard that fronts it.
type rig struct {
	t       testing.TB
	r       Reporter
	scratch *disposable.Scratch

	home, profile, exe string
	exeSHA             string
	settingsPath       string
	store              *settings.Store
	stub               *releasestub.Server
	svc                *update.Service

	started [][]string // helper starts (executable, then arguments)
	starter func(exe string, args []string) error

	dash   *httptest.Server
	token  string
	client *http.Client
}

func newRig(t testing.TB, r Reporter, o rigOptions) *rig {
	t.Helper()
	t.Setenv("LC_ALL", "C.UTF-8") // the operator UI is English; no host locale
	t.Setenv("LC_MESSAGES", "")
	t.Setenv("LANG", "C.UTF-8")
	s := disposable.NewScratch(t)
	rg := &rig{t: t, r: r, scratch: s, starter: o.helperStarter}
	rg.home = s.Path("home")
	rg.profile = s.Path("profile")
	rg.exe = s.Path("app", "hachidori.exe")
	rg.settingsPath = filepath.Join(rg.profile, "Hachidori", "settings.json")
	for _, d := range []string{rg.home, filepath.Join(rg.home, "state"), filepath.Dir(rg.exe), filepath.Dir(rg.settingsPath)} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	switch {
	case o.exeFrom != "":
		sum, err := disposable.CopyFile(o.exeFrom, rg.exe)
		if err != nil {
			t.Fatalf("copy the candidate to the disposable location: %v", err)
		}
		rg.exeSHA = sum
	default:
		data := o.exe
		if data == nil {
			data = Exe("installed", 64<<10)
		}
		if err := os.WriteFile(rg.exe, data, 0o755); err != nil {
			t.Fatal(err)
		}
		rg.exeSHA, _ = disposable.FileSHA256(rg.exe)
	}
	if !s.Contains(rg.exe) || !s.Contains(rg.home) {
		t.Fatalf("the rig escaped its scratch folder: exe %s home %s", rg.exe, rg.home)
	}
	rg.store = &settings.Store{Path: rg.settingsPath}
	rg.stub = releasestub.New(t)
	rg.svc = rg.newService()
	rg.serveConsole(rg.svc)
	return rg
}

// newService is a fresh update.Service over the rig's files: what a newly
// started application builds. Its injection points are the production ones.
func (rg *rig) newService() *update.Service {
	return &update.Service{
		Home:      func() string { return rg.home },
		Exe:       rg.exe,
		Settings:  rg.store,
		Transport: rg.stub.Transport(),
		StartHelper: func(exe string, args []string) error {
			rg.started = append(rg.started, append([]string{exe}, args...))
			if rg.starter != nil {
				return rg.starter(exe, args)
			}
			return nil
		},
		RestartHome: rg.home,
	}
}

// updates adapts the service to the dashboard like the desktop's
// updateManager (cmd/hachidori/update.go) does: the same four explicit
// actions, none of which does anything else.
type updates struct{ svc *update.Service }

func (u updates) Status() update.Status             { return u.svc.Status() }
func (u updates) SetChannel(c update.Channel) error { return u.svc.SetChannel(c) }
func (u updates) Check() error                      { return u.svc.StartCheck() }
func (u updates) Download(tag string) error         { return u.svc.StartDownload(tag) }
func (u updates) Install() error                    { return u.svc.Install() }

type idleRuntime struct{}

func (idleRuntime) Start() bool   { return false }
func (idleRuntime) Stop()         {}
func (idleRuntime) Restart()      {}
func (idleRuntime) Running() bool { return false }

// serveConsole serves the real dashboard bound to svc on loopback, replacing a
// previous one (a reopened application).
func (rg *rig) serveConsole(svc *update.Service) {
	rg.t.Helper()
	if rg.dash != nil {
		rg.dash.Close()
	}
	rg.token = dashboard.NewToken()
	d := dashboard.New(dashboard.Config{
		APIAddr:   "127.0.0.1:0",
		Status:    func() server.Status { return server.Status{Runtime: server.Runtime{Home: rg.home}} },
		Lifecycle: idleRuntime{},
		Tunnel:    tunnel.NewManager(""),
		Updates:   updates{svc},
		Token:     rg.token,
	})
	rg.dash = httptest.NewServer(d)
	rg.t.Cleanup(func() { rg.dash.Close(); d.Close() })
	rg.client = &http.Client{
		Timeout:       shortWait,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

// get reads a page of the dashboard.
func (rg *rig) get(path string) string {
	rg.t.Helper()
	resp, err := rg.client.Get(rg.dash.URL + path)
	if err != nil {
		rg.t.Fatalf("GET %s: %v", path, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		rg.t.Fatalf("GET %s: %s", path, resp.Status)
	}
	return string(body)
}

// post submits one dashboard form with the process token, as the page's own
// buttons do, and returns the redirect target.
func (rg *rig) post(path string, form url.Values) string {
	rg.t.Helper()
	if form == nil {
		form = url.Values{}
	}
	form.Set("token", rg.token)
	req, err := http.NewRequest(http.MethodPost, rg.dash.URL+path, strings.NewReader(form.Encode()))
	if err != nil {
		rg.t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", rg.dash.URL)
	resp, err := rg.client.Do(req)
	if err != nil {
		rg.t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusSeeOther {
		rg.t.Fatalf("POST %s: %s, want a 303 redirect", path, resp.Status)
	}
	return resp.Header.Get("Location")
}

const updatesPage = "/settings/updates"

func (rg *rig) setChannel(c update.Channel) {
	rg.t.Helper()
	rg.post(updatesPage+"/channel", url.Values{"channel": {string(c)}})
}

func (rg *rig) idle() bool { return rg.svc.Status().Busy == nil }

func (rg *rig) waitIdle(what string) {
	rg.t.Helper()
	until(rg.t, what, longWait, rg.idle)
}

// check performs an explicit Check for updates through the dashboard and waits
// for it to finish.
func (rg *rig) check() {
	rg.t.Helper()
	rg.post(updatesPage+"/check", nil)
	rg.waitIdle("the check to finish")
	if lc := rg.svc.Status().LastCheck; lc == nil || !lc.OK {
		rg.t.Fatalf("the check did not succeed: %+v", lc)
	}
}

// startDownload posts the Download form for tag.
func (rg *rig) startDownload(tag string) { rg.post(updatesPage+"/download", url.Values{"tag": {tag}}) }

// download downloads tag and waits for the operation to end.
func (rg *rig) download(tag string) *update.Operation {
	rg.t.Helper()
	rg.startDownload(tag)
	rg.waitIdle("the download to finish")
	last := rg.svc.Status().Last
	if last == nil {
		rg.t.Fatal("the download left no operation to report")
	}
	return last
}

// seedInstalled records that the disposable executable is release tag, the way
// an earlier explicit check or update would have.
func (rg *rig) seedInstalled(tag string) {
	rg.t.Helper()
	err := rg.store.ModifyUpdateSettings(func(s *update.Settings) error {
		s.Installed = &update.Installed{Version: tag, SHA256: rg.exeSHA}
		return nil
	})
	if err != nil {
		rg.t.Fatal(err)
	}
}

func (rg *rig) updatesDir() string { return update.Dir(rg.home) }

// regularFiles lists every regular file beneath dir, relative and slash-separated.
func regularFiles(dir string) []string {
	var out []string
	_ = filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			rel, _ := filepath.Rel(dir, p)
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	return out
}

// stagedLeftovers lists files under the update area that are not the helper's
// own copy, the ready record or the result record.
func (rg *rig) stagedFiles() []string {
	var out []string
	for _, f := range regularFiles(rg.updatesDir()) {
		if strings.HasPrefix(f, "helper/") || f == "ready.json" || f == "result.json" {
			continue
		}
		out = append(out, f)
	}
	return out
}

func (rg *rig) readyRecord() bool {
	return disposable.Exists(filepath.Join(rg.updatesDir(), "ready.json"))
}

func contains(page string, wants ...string) []string {
	var missing []string
	for _, w := range wants {
		if !strings.Contains(page, w) {
			missing = append(missing, w)
		}
	}
	return missing
}

func requireContains(t testing.TB, what, page string, wants ...string) {
	t.Helper()
	if m := contains(page, wants...); len(m) > 0 {
		t.Errorf("%s does not show %q", what, m)
	}
}

func requireAbsent(t testing.TB, what, page string, unwanted ...string) {
	t.Helper()
	for _, u := range unwanted {
		if strings.Contains(page, u) {
			t.Errorf("%s shows %q", what, u)
		}
	}
}

// buttonDisabled reports whether the first button labelled label carries the
// disabled attribute. It fails the test when there is no such button.
func buttonDisabled(t testing.TB, page, label string) bool {
	t.Helper()
	label = html.EscapeString(label) // "Restart & update" is rendered with an escaped ampersand
	i := strings.Index(page, ">"+label+"</button>")
	if i < 0 {
		t.Fatalf("the page has no %q button", label)
	}
	open := page[strings.LastIndex(page[:i], "<button"):i]
	return strings.Contains(open, " disabled")
}

// installForm reports whether the Restart & update form is offered.
func installForm(page string) bool { return strings.Contains(page, `id="updates-install"`) }

func isRefused(err error, snippet string) bool {
	var e *update.Error
	return errors.As(err, &e) && e.Class == update.ClassRefused && strings.Contains(e.Msg, snippet)
}

func shaOf(b []byte) string {
	return strings.TrimSpace(strings.SplitN(string(releasestub.ChecksumFile(b)), " ", 2)[0])
}

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

func itoa(n int) string { return strconv.Itoa(n) }

// withFlag returns args with the value of the named flag replaced.
func withFlag(args []string, name, value string) ([]string, error) {
	out := append([]string(nil), args...)
	for i := 0; i+1 < len(out); i++ {
		if out[i] == name {
			out[i+1] = value
			return out, nil
		}
	}
	return nil, fmt.Errorf("argument %s not found in %v", name, args)
}
