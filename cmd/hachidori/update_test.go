package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/update"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// fakeAuthority is the deterministic stand-in for GitHub Releases: it serves a
// fixed release list and its two assets, and records every request.
type fakeAuthority struct {
	mu   sync.Mutex
	reqs []string
	tag  string
	exe  []byte
}

func newAuthority(tag string) *fakeAuthority {
	return &fakeAuthority{tag: tag, exe: append([]byte("MZ"), []byte("hachidori "+tag+strings.Repeat("y", 2048))...)}
}

func (g *fakeAuthority) sum() string { s := sha256.Sum256(g.exe); return hex.EncodeToString(s[:]) }

func (g *fakeAuthority) count() int { g.mu.Lock(); defer g.mu.Unlock(); return len(g.reqs) }

func (g *fakeAuthority) RoundTrip(req *http.Request) (*http.Response, error) {
	g.mu.Lock()
	g.reqs = append(g.reqs, req.URL.Host+req.URL.Path)
	g.mu.Unlock()
	resp := func(code int, body string, hdr ...string) (*http.Response, error) {
		h := http.Header{}
		for i := 0; i+1 < len(hdr); i += 2 {
			h.Set(hdr[i], hdr[i+1])
		}
		return &http.Response{StatusCode: code, Status: fmt.Sprint(code), Header: h, Body: io.NopCloser(strings.NewReader(body)), ContentLength: int64(len(body)), Request: req}, nil
	}
	sumFile := g.sum() + "  " + update.ExeAsset + "\r\n"
	switch {
	case req.URL.Host == "api.github.com":
		if req.URL.Query().Get("page") != "1" {
			return resp(200, "[]")
		}
		return resp(200, fmt.Sprintf(`[{"tag_name":%q,"prerelease":true,"draft":false,"published_at":"2026-10-01T13:44:50Z","assets":[`+
			`{"name":%q,"size":%d,"state":"uploaded"},{"name":%q,"size":%d,"state":"uploaded"}]}]`,
			g.tag, update.ExeAsset, len(g.exe), update.SumAsset, len(sumFile)))
	case req.URL.Host == "github.com" && strings.Contains(req.URL.Path, "/releases/download/"):
		return resp(302, "", "Location", "https://objects.githubusercontent.com"+req.URL.Path[strings.LastIndex(req.URL.Path, "/"):])
	case req.URL.Host == "objects.githubusercontent.com" && req.URL.Path == "/"+update.ExeAsset:
		return resp(200, string(g.exe))
	case req.URL.Host == "objects.githubusercontent.com" && req.URL.Path == "/"+update.SumAsset:
		return resp(200, sumFile)
	}
	return resp(404, "")
}

// trapNetwork makes any non-loopback request through http.DefaultTransport
// (the transport anything else in the process would use) fail the test.
func trapNetwork(t *testing.T) *atomicCount {
	t.Helper()
	orig := http.DefaultTransport
	c := &atomicCount{}
	http.DefaultTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if host, _, _ := net.SplitHostPort(req.URL.Host); host != "127.0.0.1" && host != "localhost" {
			c.add()
			t.Errorf("unexpected external request through the default transport: %s", req.URL)
			return nil, fmt.Errorf("external network is not available in tests")
		}
		return orig.RoundTrip(req)
	})
	t.Cleanup(func() { http.DefaultTransport = orig })
	return c
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type atomicCount struct {
	mu sync.Mutex
	n  int
}

func (c *atomicCount) add()     { c.mu.Lock(); c.n++; c.mu.Unlock() }
func (c *atomicCount) get() int { c.mu.Lock(); defer c.mu.Unlock(); return c.n }

// Starting Hachidori and using its window makes no update-subsystem request:
// the composition builds the subsystem but nothing calls it.
func TestStartupMakesNoUpdateRequests(t *testing.T) {
	trap := trapNetwork(t)
	auth := newAuthority("0.2.6-dev")
	for name, mk := range map[string]func(t *testing.T) *desktopApp{
		"installed home": func(t *testing.T) *desktopApp {
			a, _, _ := installedApp(t, &fakeDesktop{}, false)
			return a
		},
		"first run": func(t *testing.T) *desktopApp {
			a, _, _ := testApp(&fakePlatform{version: "130.0", open: func(context.Context, desktop.Window) error { return nil }},
				func() (home.Discovery, error) { return home.Discovery{Source: home.SourceUnconfigured}, nil })
			return a
		},
	} {
		a := mk(t)
		a.UpdateTransport = auth
		switch p := a.Platform.(type) {
		case *fakeDesktop:
			p.open = func(_ context.Context, w desktop.Window) error {
				time.Sleep(100 * time.Millisecond) // the window is open and the runtime started
				get(t, w.URL)
				return nil
			}
		}
		if err := a.run(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if n := auth.count(); n != 0 {
			t.Fatalf("%s: %d update requests during startup: %v", name, n, auth.reqs)
		}
	}
	if trap.get() != 0 {
		t.Fatal("external requests during startup")
	}
}

// updatesRig is the dashboard composed exactly as the desktop composes it for
// Updates (a.newUpdateManager), over a real settings store and a real update
// service, with a fixture authority instead of GitHub.
type updatesRig struct {
	t       *testing.T
	a       *desktopApp
	auth    *fakeAuthority
	srv     *httptest.Server
	token   string
	root    string
	exe     string
	exeData []byte
	started chan []string
	quits   chan struct{}
	ctl     *app.Controller
	um      updateManager
}

func newUpdatesRig(t *testing.T) *updatesRig {
	t.Helper()
	t.Setenv("LC_ALL", "")
	t.Setenv("LANG", "C.UTF-8")
	dir := t.TempDir()
	r := &updatesRig{t: t, auth: newAuthority("0.2.6-dev"), root: filepath.Join(dir, "home"), exe: filepath.Join(dir, "app", "hachidori.exe"),
		exeData: append([]byte("MZ"), []byte("installed build")...), started: make(chan []string, 1), quits: make(chan struct{}, 1)}
	h := home.Home{Root: r.root}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	os.MkdirAll(filepath.Dir(r.exe), 0o755)
	os.WriteFile(r.exe, r.exeData, 0o755)
	r.a = &desktopApp{Exe: r.exe, UpdateTransport: r.auth, UpdateStart: func(exe string, args []string) error {
		r.started <- append([]string{exe}, args...)
		return nil
	}}
	r.ctl = app.New(app.Config{Home: r.root})
	prefs := settingsStore(filepath.Join(dir, "desktop.json"), nil)
	r.um = r.a.newUpdateManager(prefs, func() *app.Controller { return r.ctl }, "", func() { r.quits <- struct{}{} })
	tun := tunnel.NewManager("ssh")
	t.Cleanup(tun.Disconnect)
	d := dashboard.New(dashboard.Config{
		APIAddr: "127.0.0.1:7843",
		Status: func() server.Status {
			return server.Status{Schema: "hachidori.v1", Worker: worker.Snapshot{State: worker.StateReady, Ready: true}, Runtime: server.Runtime{Home: r.root}}
		},
		Lifecycle: &fakeRuntime{}, Doctor: func(io.Writer) bool { return true }, Tunnel: tun,
		Settings: prefs, Updates: r.um,
	})
	r.srv = httptest.NewServer(d)
	t.Cleanup(r.srv.Close)
	r.token = regexp.MustCompile(`name="token" value="([0-9a-f]+)"`).FindStringSubmatch(r.get("/settings/updates"))[1]
	return r
}

func (r *updatesRig) get(path string) string {
	r.t.Helper()
	resp, err := http.Get(r.srv.URL + path)
	if err != nil {
		r.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		r.t.Fatalf("GET %s: %d", path, resp.StatusCode)
	}
	return string(b)
}

func (r *updatesRig) post(path string, kv ...string) string {
	r.t.Helper()
	form := url.Values{"token": {r.token}}
	for i := 0; i+1 < len(kv); i += 2 {
		form.Set(kv[i], kv[i+1])
	}
	c := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	req, _ := http.NewRequest("POST", r.srv.URL+path, strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", r.srv.URL)
	resp, err := c.Do(req)
	if err != nil {
		r.t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusSeeOther {
		r.t.Fatalf("POST %s: %d", path, resp.StatusCode)
	}
	return r.get(resp.Header.Get("Location"))
}

func (r *updatesRig) waitDownload() string {
	r.t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if body := r.get("/settings/updates"); !strings.Contains(body, `id="models-busy"`) {
			return body
		}
		time.Sleep(5 * time.Millisecond)
	}
	r.t.Fatal("download did not finish")
	return ""
}

// The full manual flow through the composed dashboard: nothing uses the
// network until Check; only Check reads metadata; only Download reads assets;
// verification precedes Restart & update; the helper replaces the executable
// after exit and the home is untouched.
func TestUpdatesFlowThroughTheDesktopComposition(t *testing.T) {
	trap := trapNetwork(t)
	r := newUpdatesRig(t)
	sentinel := filepath.Join(r.root, "state", "history", "h1.json")
	os.MkdirAll(filepath.Dir(sentinel), 0o755)
	os.WriteFile(sentinel, []byte("evidence"), 0o644)

	// Opening Settings and Updates, and changing the channel, use no network.
	r.get("/settings")
	r.get("/settings/updates")
	body := r.post("/settings/updates/channel", "channel", "development")
	r.get("/settings/updates")
	r.post("/settings/updates/channel", "channel", "stable")
	body = r.post("/settings/updates/channel", "channel", "development")
	if n := r.auth.count(); n != 0 || !strings.Contains(body, `value="development" selected`) || strings.Contains(body, `id="updates-releases"`) {
		t.Fatalf("%d requests before any explicit action: %v", n, r.auth.reqs)
	}

	// Check reads release metadata only.
	body = r.post("/settings/updates/check")
	for _, q := range r.auth.reqs {
		if !strings.HasPrefix(q, "api.github.com/repos/yohn-jp/hachidori/releases") {
			t.Errorf("Check requested %s", q)
		}
	}
	if !strings.Contains(body, `id="updates-releases"`) || !strings.Contains(body, "0.2.6-dev") || !strings.Contains(body, `action="/settings/updates/download"`) {
		t.Fatalf("results not shown:\n%s", body)
	}
	if strings.Contains(body, `id="updates-install"`) {
		t.Fatal("Restart & update offered before a verified download")
	}
	checked := r.auth.count()

	// Download reads the checksum and the executable, then the update is ready.
	r.post("/settings/updates/download", "tag", "0.2.6-dev")
	body = r.waitDownload()
	if got := r.auth.reqs[checked:]; len(got) != 4 || !strings.HasSuffix(got[0], "/"+update.SumAsset) || !strings.HasSuffix(got[2], "/"+update.ExeAsset) {
		t.Fatalf("download requests: %v", got)
	}
	if !strings.Contains(body, `id="updates-install"`) || !strings.Contains(body, "matches the release checksum") {
		t.Fatalf("verified update not ready:\n%s", body)
	}
	if got, _ := os.ReadFile(r.exe); string(got) != string(r.exeData) {
		t.Fatal("download touched the executable")
	}
	afterDownload := r.auth.count()

	// Restart & update hands over to the helper and ends the application; the
	// helper (run here in-process with the same arguments) replaces the file.
	r.post("/settings/updates/install")
	var args []string
	select {
	case args = <-r.started:
	case <-time.After(5 * time.Second):
		t.Fatal("helper not started")
	}
	select {
	case <-r.quits:
	case <-time.After(5 * time.Second):
		t.Fatal("the application was not asked to quit")
	}
	if args[1] != "apply-update" || filepath.Dir(args[0]) != filepath.Join(r.root, "state", "updates", "helper") {
		t.Fatalf("helper command %q", args)
	}
	if got, _ := os.ReadFile(r.exe); string(got) != string(r.exeData) {
		t.Fatal("Restart & update overwrote the running executable in-process")
	}
	var restarted []string
	env := update.DefaultApplyEnv()
	env.WaitExit = func(int, time.Duration) error { return nil }
	env.Restart = func(exe string, a []string) error { restarted = append([]string{exe}, a...); return nil }
	for i, a := range args { // the test process stands for the application; the helper must not wait for itself
		if a == "--pid" {
			args[i+1] = strconv.Itoa(os.Getpid() + 1)
		}
	}
	if err := runApplyUpdate(args[2:], env, io.Discard); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(r.exe); string(got) != string(r.auth.exe) {
		t.Fatal("the executable was not replaced by the verified update")
	}
	if len(restarted) != 1 || restarted[0] != r.exe {
		t.Fatalf("restart %v", restarted)
	}
	if b, _ := os.ReadFile(sentinel); string(b) != "evidence" {
		t.Fatal("HACHIDORI_HOME content changed")
	}
	if r.auth.count() != afterDownload || trap.get() != 0 {
		t.Fatal("install or the helper used the network")
	}
}

func TestRestartAndUpdateIsRefusedWhileTheApplicationIsBusy(t *testing.T) {
	trapNetwork(t)
	r := newUpdatesRig(t)
	r.post("/settings/updates/channel", "channel", "development")
	r.post("/settings/updates/check")
	r.post("/settings/updates/download", "tag", "0.2.6-dev")
	r.waitDownload()
	// A maintenance action in flight: the controller reports an operation.
	r.ctl = app.New(app.Config{Home: r.root, Setup: func(string, string, string, io.Writer, *setup.Observer) error {
		time.Sleep(300 * time.Millisecond)
		return nil
	}})
	if err := r.ctl.Setup(app.SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	if err := r.um.Install(); err == nil || !strings.Contains(err.Error(), "in progress") {
		t.Fatalf("install while busy: %v", err)
	}
	select {
	case a := <-r.started:
		t.Fatalf("helper started while busy: %v", a)
	default:
	}
}

func TestApplyUpdateCommandAcceptsOnlyItsFixedFlags(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing-home")
	env := update.DefaultApplyEnv()
	env.WaitExit = func(int, time.Duration) error { t.Error("waited"); return nil }
	env.Restart = func(string, []string) error { t.Error("restarted"); return nil }
	for _, args := range [][]string{
		nil,
		{"--url", "https://example.com/x.exe"},
		{"--source", "/tmp/evil.exe", "--target", "/tmp/x.exe", "--home", "/tmp", "--pid", "5"},
		{"--home", missing, "--pid", "999999", "--target", "/tmp/x.exe", "extra"},
		{"--home", missing, "--pid", "999999", "--target", "/tmp/x.exe"}, // no ready record
	} {
		if err := runApplyUpdate(args, env, io.Discard); err == nil {
			t.Errorf("accepted %q", args)
		}
	}
	if _, err := os.Stat(missing); !os.IsNotExist(err) {
		t.Errorf("a refused helper created %s", missing)
	}
	if cmds := commands(); cmds["apply-update"] == nil {
		t.Fatal("the helper command is not dispatched")
	}
}
