package dashboard

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// TestMain doubles as a fake ssh (see internal/tunnel tests): it stays up
// until terminated. No real SSH is used.
func TestMain(m *testing.M) {
	if os.Getenv("HACHIDORI_FAKE_SSH") != "" {
		time.Sleep(time.Minute)
		return
	}
	os.Exit(m.Run())
}

const xss = `<script>alert("x")</script>`

type fakeRuntime struct {
	mu    sync.Mutex
	snap  worker.Snapshot
	calls []string
	run   bool
}

func (f *fakeRuntime) Decide([]worker.Item) ([][]api.Result, float64, error) { return nil, 0, nil }
func (f *fakeRuntime) Ready() bool                                           { return f.Snapshot().Ready }
func (f *fakeRuntime) State() string                                         { return f.Snapshot().State }
func (f *fakeRuntime) Snapshot() worker.Snapshot {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.snap
}
func (f *fakeRuntime) Start() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "start")
	was := f.run
	f.run = true
	return !was
}
func (f *fakeRuntime) Stop() {
	f.mu.Lock()
	f.calls = append(f.calls, "stop")
	f.run = false
	f.mu.Unlock()
}
func (f *fakeRuntime) Restart() {
	f.mu.Lock()
	f.calls = append(f.calls, "restart")
	f.run = true
	f.mu.Unlock()
}
func (f *fakeRuntime) Running() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.run
}

type env struct {
	d       *Dashboard
	rt      *fakeRuntime
	api     http.Handler
	tun     *tunnel.Manager
	home    string
	doctorN int
}

func newEnv(t testing.TB) *env {
	t.Helper()
	exe, _ := os.Executable()
	t.Setenv("HACHIDORI_FAKE_SSH", "run")
	// The host locale must not pick the operator UI language for tests.
	t.Setenv("LC_ALL", "")
	t.Setenv("LC_MESSAGES", "")
	t.Setenv("LANG", "C.UTF-8")
	home := t.TempDir()
	os.MkdirAll(filepath.Join(home, "state"), 0o755)
	rt := &fakeRuntime{run: true, snap: worker.Snapshot{
		State: worker.StateReady, Phase: "ready", Ready: true, PID: 4242, Starts: 2, Restarts: 1, Requests: 17,
		Errors: map[string]int64{"not_ready": 1}, QueueDepth: 0, QueueLimit: 64, LatencyP50MS: 25.3, LatencyP95MS: 30.5,
		Info: worker.Info{"provider": "laya", "laya_version": "0.3.21", "provider_version": "0.3.21", "torch_version": "2.11.0+cu128", "torch_cuda": "12.8",
			"python_version": "3.12.11", "device": "cuda:0", "device_name": "NVIDIA GeForce RTX 3060", "load_ms": 812.5, "warmup_ms": 90.1,
			"python_executable": filepath.Join(home, "runtime", "python.exe"), "hf_home": filepath.Join(home, "cache", "hf"),
			"model_dir": filepath.Join(home, "models", "laya")},
		Accelerator: map[string]any{"memory_allocated": float64(600 << 20), "memory_reserved": float64(700 << 20),
			"memory_free": float64(10 << 30), "memory_total": float64(12 << 30)},
		LastFailure: &worker.FailureView{Class: worker.ClassCrash, Message: xss, Stderr: []string{"Traceback & <b>"}},
	}}
	e := &env{rt: rt, home: home}
	started := time.Now()
	rtInfo := server.Runtime{Home: home, Runtime: "0.1.0-cu128", ModelID: "laya-base", Model: "convaiinnovations--laya/55cf4c4e", Device: "cuda"}
	e.api = server.HandlerSince(rt, rtInfo, started)
	e.tun = tunnel.NewManager(exe)
	t.Cleanup(e.tun.Disconnect)
	e.d = New(Config{
		APIAddr:   "127.0.0.1:7843",
		Status:    func() server.Status { return server.StatusBody(rt, rtInfo, started) },
		Lifecycle: rt,
		Doctor: func(out io.Writer) bool {
			e.doctorN++
			fmt.Fprintln(out, "PASS home [hachidori] "+xss)
			return true
		},
		Tunnel:    e.tun,
		PrefsPath: filepath.Join(home, "state", "dashboard.json"),
	})
	return e
}

func (e *env) get(t testing.TB, path string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "http://127.0.0.1:7844"+path, nil)
	e.d.ServeHTTP(rec, req)
	return rec
}

func (e *env) post(t testing.TB, path string, form url.Values) *httptest.ResponseRecorder {
	if form == nil {
		form = url.Values{}
	}
	if !form.Has("token") {
		form.Set("token", e.d.token)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "http://127.0.0.1:7844"+path, strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", "http://127.0.0.1:7844")
	e.d.ServeHTTP(rec, req)
	return rec
}

func (e *env) lastAction(t *testing.T) *Action {
	e.d.mu.Lock()
	defer e.d.mu.Unlock()
	if e.d.last == nil {
		t.Fatal("no action recorded")
	}
	return e.d.last
}

// Accelerator numbers taken before the inference now in flight are shown, but
// labelled; the label is absent when they are current.
func TestStaleAcceleratorStatsAreLabelled(t *testing.T) {
	e := newEnv(t)
	shown := func() string { return e.get(t, "/").Body.String() + e.get(t, "/diagnostics").Body.String() }
	if strings.Contains(shown(), "last known") {
		t.Fatal("current accelerator numbers are labelled stale")
	}
	e.rt.mu.Lock()
	e.rt.snap.AcceleratorStale = true
	e.rt.mu.Unlock()
	body := shown()
	if strings.Count(body, "last known") != 2 || !strings.Contains(body, "allocated 600 MiB") {
		t.Fatalf("stale numbers must stay visible and be labelled in both places (%d labels)", strings.Count(body, "last known"))
	}
}

func TestStatusIsTheV1StatusDocument(t *testing.T) {
	e := newEnv(t)
	rec := e.get(t, "/api/status")
	api := httptest.NewRecorder()
	e.api.ServeHTTP(api, httptest.NewRequest("GET", "http://127.0.0.1:7843/v1/status", nil))
	var a, b map[string]any
	json.Unmarshal(rec.Body.Bytes(), &a)
	json.Unmarshal(api.Body.Bytes(), &b)
	delete(a, "uptime_s")
	delete(b, "uptime_s")
	if rt, _ := a["runtime"].(map[string]any); rt["model_id"] != "laya-base" {
		t.Errorf("status lacks the selected model identity: %v", a["runtime"])
	}
	if rec.Code != 200 || fmt.Sprint(a) != fmt.Sprint(b) || a["worker"] == nil {
		t.Fatalf("dashboard status differs from /v1/status:\n%v\n%v", a, b)
	}
	// Runtime and Diagnostics render values from that same document.
	body := e.get(t, "/").Body.String() + e.get(t, "/diagnostics").Body.String()
	for _, want := range []string{"READY", "4242", "2 / 1", "17", "p50 25.3 ms, p95 30.5 ms", "laya 0.3.21", "convaiinnovations--laya/55cf4c4e", "laya-base",
		"0.1.0-cu128", "3.12.11", "2.11.0&#43;cu128 / 12.8", "NVIDIA GeForce RTX 3060", "allocated 600 MiB, reserved 700 MiB, free 10240 MiB, total 12288 MiB",
		"812.5 ms / 90.1 ms", "worker_crash", "0 / 64"} {
		if !strings.Contains(body, want) {
			t.Errorf("page lacks %q", want)
		}
	}
	e.rt.mu.Lock()
	e.rt.snap.State, e.rt.snap.Ready, e.rt.snap.PID = worker.StateStopped, false, 0
	e.rt.mu.Unlock()
	if live := e.get(t, "/live").Body.String(); !strings.Contains(live, ">stopped<") || strings.Contains(live, "READY") {
		t.Fatalf("live fragment does not follow runtime state:\n%s", live)
	}
}

func TestRenderedValuesAreEscaped(t *testing.T) {
	e := newEnv(t)
	e.post(t, "/doctor", nil)
	waitDoctor(t, e)
	e.post(t, "/tunnel/connect", url.Values{"destination": {xss}, "remote_bind": {"127.0.0.1"}, "remote_port": {"1"}, "local_port": {"1"}})
	for _, p := range []string{"/", "/diagnostics", "/live"} {
		if b := e.get(t, p).Body.String(); strings.Contains(b, xss) || strings.Contains(b, "<b>") {
			t.Fatalf("unescaped runtime/error value in %s", p)
		}
	}
	body := e.get(t, "/diagnostics").Body.String()
	// worker last failure + doctor output; the rejected destination is quoted in the action message
	if n := strings.Count(body, "&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt;"); n < 2 || !strings.Contains(body, "invalid ssh destination") {
		t.Fatalf("escaped values missing (%d):\n%s", n, body)
	}
}

func TestLifecycleActionsUseLifecycleAuthority(t *testing.T) {
	e := newEnv(t)
	for _, op := range []string{"stop", "start", "start", "restart"} {
		if rec := e.post(t, "/runtime/"+op, nil); rec.Code != http.StatusSeeOther {
			t.Fatalf("%s: %d", op, rec.Code)
		}
	}
	if got := fmt.Sprint(e.rt.calls); got != "[stop start start restart]" {
		t.Fatalf("lifecycle calls %s", got)
	}
	e.post(t, "/runtime/start", nil)
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "already running") {
		t.Fatalf("duplicate start not reported: %+v", a)
	}
	if rec := e.post(t, "/runtime/explode", nil); rec.Code != 404 {
		t.Fatalf("unknown op: %d", rec.Code)
	}
}

func TestStateChangesRequirePOSTAndToken(t *testing.T) {
	e := newEnv(t)
	for _, p := range []string{"/runtime/stop", "/runtime/start", "/runtime/restart", "/doctor", "/tunnel/connect", "/tunnel/disconnect"} {
		if rec := e.get(t, p); rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("GET %s: %d", p, rec.Code)
		}
		if rec := e.post(t, p, url.Values{"token": {"forged"}}); rec.Code != http.StatusForbidden {
			t.Errorf("POST %s with bad token: %d", p, rec.Code)
		}
	}
	// Cross-site form posts are refused even with a token.
	form := url.Values{"token": {e.d.token}}
	for _, h := range []map[string]string{{"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}} {
		req := httptest.NewRequest("POST", "http://127.0.0.1:7844/runtime/stop", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		for k, v := range h {
			req.Header.Set(k, v)
		}
		rec := httptest.NewRecorder()
		e.d.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Errorf("%v: %d", h, rec.Code)
		}
	}
	if len(e.rt.calls) != 0 {
		t.Fatalf("refused requests reached the lifecycle: %v", e.rt.calls)
	}
}

func TestNonLoopbackHostRefused(t *testing.T) {
	e := newEnv(t)
	for _, host := range []string{"evil.example", "evil.example:7844", "192.168.1.5:7844", "0.0.0.0:7844"} {
		req := httptest.NewRequest("GET", "/", nil)
		req.Host = host
		rec := httptest.NewRecorder()
		e.d.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Errorf("Host %s: %d", host, rec.Code)
		}
	}
	for _, host := range []string{"127.0.0.1:7844", "localhost:7844", "[::1]:7844"} {
		req := httptest.NewRequest("GET", "/", nil)
		req.Host = host
		rec := httptest.NewRecorder()
		e.d.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Errorf("Host %s: %d", host, rec.Code)
		}
	}
}

func TestDefaultListenIsLoopback(t *testing.T) {
	if err := server.CheckLoopback(DefaultListen); err != nil {
		t.Fatal(err)
	}
}

func waitDoctor(t *testing.T, e *env) DoctorRun {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		e.d.mu.Lock()
		d := e.d.doctor
		e.d.mu.Unlock()
		if !d.Running && !d.Finished.IsZero() {
			return d
		}
		if time.Now().After(deadline) {
			t.Fatal("doctor did not finish")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestDoctor(t *testing.T) {
	e := newEnv(t)
	e.post(t, "/doctor", nil)
	d := waitDoctor(t, e)
	if !d.OK || !strings.Contains(d.Output, "PASS home") || e.doctorN != 1 {
		t.Fatalf("%+v", d)
	}
}

var tokenRe = regexp.MustCompile(`name="token" value="([0-9a-f]+)"`)

func TestTunnelConnectDisconnectAndPrefs(t *testing.T) {
	e := newEnv(t)
	if m := tokenRe.FindStringSubmatch(e.get(t, "/").Body.String()); m == nil || m[1] != e.d.token {
		t.Fatal("page does not carry the form token")
	}
	// Form defaults: loopback bind, the API port.
	body := e.get(t, "/diagnostics").Body.String()
	if !strings.Contains(body, `name="remote_bind" value="127.0.0.1"`) || !strings.Contains(body, `name="local_port" value="7843"`) {
		t.Fatal("form defaults missing")
	}
	bad := url.Values{"destination": {"-oProxyCommand=calc"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}}
	e.post(t, "/tunnel/connect", bad)
	if a := e.lastAction(t); a.OK || e.tun.Status().State != tunnel.StateIdle {
		t.Fatalf("invalid destination accepted: %+v", a)
	}
	bad = url.Values{"destination": {"dev@nixos"}, "remote_bind": {"0.0.0.0"}, "remote_port": {"7843"}, "local_port": {"7843"}}
	e.post(t, "/tunnel/connect", bad)
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "loopback") {
		t.Fatalf("public remote bind accepted: %+v", a)
	}

	good := url.Values{"destination": {"dev@nixos"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}}
	e.post(t, "/tunnel/connect", good)
	st := e.tun.Status()
	if a := e.lastAction(t); !a.OK || st.State != tunnel.StateRunning || st.PID == 0 {
		t.Fatalf("connect: %+v %+v", a, st)
	}
	e.post(t, "/tunnel/connect", good)
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, "nothing started") || e.tun.Status().PID != st.PID {
		t.Fatalf("duplicate connect: %+v", a)
	}
	page := e.get(t, "/diagnostics").Body.String()
	for _, want := range []string{"HACHIDORI_ENDPOINT=http://127.0.0.1:7843", fmt.Sprint(st.PID), "dev@nixos"} {
		if !strings.Contains(page, want) {
			t.Errorf("diagnostics lacks %q", want)
		}
	}
	// Runtime keeps the compact transport state: destination and caller endpoint.
	root := e.get(t, "/").Body.String()
	for _, want := range []string{"HACHIDORI_ENDPOINT=http://127.0.0.1:7843", "dev@nixos", `href="/diagnostics#transport"`} {
		if !strings.Contains(root, want) {
			t.Errorf("runtime lacks %q", want)
		}
	}

	e.post(t, "/tunnel/disconnect", nil)
	if a := e.lastAction(t); !a.OK || e.tun.Status().State != tunnel.StateStopped {
		t.Fatalf("disconnect: %+v %+v", a, e.tun.Status())
	}

	// Only non-secret form values are persisted, and only that file is written.
	b, err := os.ReadFile(filepath.Join(e.home, "state", "dashboard.json"))
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]map[string]any
	if err := json.Unmarshal(b, &raw); err != nil || len(raw) != 1 {
		t.Fatalf("prefs: %s", b)
	}
	keys := []string{}
	for k := range raw["tunnel"] {
		keys = append(keys, k)
	}
	if len(keys) != 4 || raw["tunnel"]["destination"] != "dev@nixos" {
		t.Fatalf("prefs keys %v", keys)
	}
	for _, s := range []string{"key", "pass", "identity", "secret", "BEGIN", "known_hosts"} {
		if strings.Contains(strings.ToLower(string(b)), strings.ToLower(s)) {
			t.Fatalf("prefs contain %q: %s", s, b)
		}
	}
	// A fresh dashboard restores the saved form values.
	e2 := New(Config{APIAddr: "127.0.0.1:7843", Status: e.d.cfg.Status, Lifecycle: e.rt, Doctor: e.d.cfg.Doctor,
		Tunnel: tunnel.NewManager("ssh"), PrefsPath: e.d.cfg.PrefsPath})
	if f := e2.formDefaults(); f.Destination != "dev@nixos" {
		t.Fatalf("prefs not restored: %+v", f)
	}
	assertOnlyPrefsWritten(t, e.home)
}

func TestCloseTerminatesManagedTunnel(t *testing.T) {
	e := newEnv(t)
	e.post(t, "/tunnel/connect", url.Values{"destination": {"dev@nixos"}, "remote_bind": {"127.0.0.1"}, "remote_port": {"7843"}, "local_port": {"7843"}})
	st := e.tun.Status()
	if st.State != tunnel.StateRunning {
		t.Fatalf("%+v", st)
	}
	e.d.Close()
	if st := e.tun.Status(); st.State != tunnel.StateStopped {
		t.Fatalf("after Close: %+v", st)
	}
	// The child is gone, not just marked: signalling its PID now fails.
	if p, err := os.FindProcess(st.PID); err == nil && p.Signal(syscall.Signal(0)) == nil {
		t.Fatalf("ssh child %d still alive after Close", st.PID)
	}
}

// The dashboard has no dataset surface: uploads are refused and nothing but
// the preferences file is ever written under HACHIDORI_HOME.
func TestNoDatasetReachesTheHost(t *testing.T) {
	e := newEnv(t)
	dataset := `{"id":"case-001","state":"s","questions":[],"expected":{"q":"yes"}}`
	for _, p := range []string{"/upload", "/dataset", "/eval", "/benchmark", "/api/eval", "/"} {
		rec := e.post(t, p, url.Values{"dataset": {dataset}})
		if rec.Code != http.StatusNotFound && rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("POST %s: %d", p, rec.Code)
		}
	}
	e.post(t, "/doctor", nil)
	waitDoctor(t, e)
	e.post(t, "/runtime/restart", nil)
	assertOnlyPrefsWritten(t, e.home)
}

func assertOnlyPrefsWritten(t *testing.T, home string) {
	t.Helper()
	filepath.Walk(home, func(p string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && p != filepath.Join(home, "state", "dashboard.json") {
			t.Errorf("unexpected file under HACHIDORI_HOME: %s", p)
		}
		return nil
	})
}

var formRe = regexp.MustCompile(`(?s)<form\b([^>]*)>(.*?)</form>`)

// postForms returns a page's state-changing forms by action, checking that
// each is a POST carrying the form token.
func postForms(t *testing.T, e *env, body string) map[string]string {
	t.Helper()
	token := `name="token" value="` + e.d.token + `"`
	got := map[string]string{}
	for _, m := range formRe.FindAllStringSubmatch(body, -1) {
		action := regexp.MustCompile(`action="([^"]+)"`).FindStringSubmatch(m[1])
		if strings.Contains(m[1], `method="get"`) {
			continue // read-only analysis controls
		}
		if action == nil || !strings.Contains(m[1], `method="post"`) {
			t.Errorf("form without POST action: %s", m[1])
			continue
		}
		if !strings.Contains(m[2], token) {
			t.Errorf("form %s lacks the form token", action[1])
		}
		got[action[1]] = m[1] + m[2]
	}
	return got
}

// Every state-changing control is a same-origin POST form carrying the form
// token. Runtime exposes the lifecycle actions; Diagnostics exposes doctor
// and the tunnel launcher.
func TestPageFormsAreWiredToActionRoutes(t *testing.T) {
	e := newEnv(t)
	root := postForms(t, e, e.get(t, "/").Body.String())
	body := e.get(t, "/diagnostics").Body.String()
	diag := postForms(t, e, body)
	got := map[string]string{}
	for k, v := range root {
		got[k] = v
	}
	for k, v := range diag {
		got[k] = v
	}
	for page, want := range map[string]struct {
		got  map[string]string
		want []string
	}{"/": {root, []string{"/runtime/stop", "/runtime/restart"}},
		"/diagnostics": {diag, []string{"/doctor", "/diagnostics/export", "/tunnel/connect", "/tunnel/disconnect"}}} {
		if len(want.got) != len(want.want) {
			t.Errorf("%s forms %v, want %v", page, want.got, want.want)
		}
		for _, a := range want.want {
			if _, ok := want.got[a]; !ok {
				t.Errorf("%s: no form for %s", page, a)
			}
		}
	}
	// The tunnel form keeps the existing field names; its submit control is
	// associated with it.
	connect := got["/tunnel/connect"]
	for _, n := range []string{"destination", "remote_bind", "remote_port", "local_port"} {
		if !strings.Contains(connect, `name="`+n+`"`) {
			t.Errorf("tunnel form lacks %s", n)
		}
	}
	id := regexp.MustCompile(`id="([^"]+)"`).FindStringSubmatch(connect)
	if !strings.Contains(connect, `type="submit"`) && (id == nil || !regexp.MustCompile(`<button type="submit"[^>]*form="`+id[1]+`"`).MatchString(body)) {
		t.Error("tunnel form has no submit control")
	}
	// Doctor cannot be started twice from the page.
	e.d.mu.Lock()
	e.d.doctor.Running = true
	e.d.mu.Unlock()
	if !regexp.MustCompile(`(?s)action="/doctor".*?<button[^>]*disabled`).MatchString(e.get(t, "/diagnostics").Body.String()) {
		t.Error("doctor button not disabled while running")
	}
}

// The live fragment refreshes every status slot of the Runtime and
// Diagnostics workspaces independently and never carries the tunnel form, so
// refreshes cannot clobber its input.
func TestLiveFragmentRefreshesPageSlots(t *testing.T) {
	e := newEnv(t)
	page, live := e.get(t, "/").Body.String()+e.get(t, "/diagnostics").Body.String(), e.get(t, "/live").Body.String()
	slotRe := regexp.MustCompile(`data-live="([a-z]+)"`)
	slots := slotRe.FindAllStringSubmatch(live, -1)
	if len(slots) < 2 {
		t.Fatalf("live fragment slots: %v", slots)
	}
	for _, s := range slots {
		if !strings.Contains(page, s[0]) {
			t.Errorf("page has no slot %s", s[1])
		}
	}
	if strings.Contains(live, `name="destination"`) || strings.Contains(live, "<html") {
		t.Error("live fragment contains the tunnel form or a full page")
	}
	if !strings.Contains(page, `fetch("/live"`) || !strings.Contains(page, "setInterval(refresh, 3000)") {
		t.Error("page does not poll /live every 3 s")
	}
	for _, want := range []string{"READY", "NVIDIA GeForce RTX 3060", "12288 MiB", "0 / 64", "25.3 ms", "idle"} {
		if !strings.Contains(live, want) {
			t.Errorf("live fragment lacks %q", want)
		}
	}
}

func TestAttentionSummarizesOperationalProblems(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/live").Body.String()
	if !strings.Contains(body, "Recovered from worker failure: worker_crash") {
		t.Errorf("recovered failure not flagged:\n%s", body)
	}
	e.post(t, "/runtime/stop", nil)
	e.rt.mu.Lock()
	e.rt.snap.State, e.rt.snap.Ready = worker.StateStopped, false
	e.rt.mu.Unlock()
	v := e.d.view("Runtime", "runtime")
	a := alerts(v)
	// A stopped runtime is a stable state, not an alert; only the worker's own
	// unrecovered failure is.
	if len(a) != 1 || a[0].Title != "Last worker failure: worker_crash" || a[0].Level != "bad" {
		t.Fatalf("alerts %+v", a)
	}
	e.rt.mu.Lock()
	e.rt.run, e.rt.snap.State, e.rt.snap.Ready, e.rt.snap.LastFailure = true, worker.StateReady, true, nil
	e.rt.mu.Unlock()
	if a := alerts(e.d.view("Runtime", "runtime")); len(a) != 0 {
		t.Fatalf("healthy runtime flagged: %+v", a)
	}
	if !strings.Contains(e.get(t, "/live").Body.String(), "No operator attention required") {
		t.Error("all-clear state not shown")
	}
}

func TestGPUMemoryBreakdown(t *testing.T) {
	m := gpuMem(map[string]any{"memory_allocated": 25.0, "memory_reserved": 50.0, "memory_free": 40.0, "memory_total": 100.0})
	if m == nil || m.Alloc != 25 || m.Cached != 25 || m.Other != 10 || m.Free != 40 || m.Used != 60 || m.Level != "ok" {
		t.Fatalf("%+v", m)
	}
	if m := gpuMem(map[string]any{"memory_free": 3.0, "memory_total": 100.0}); m == nil || m.Level != "bad" {
		t.Fatalf("pressure not flagged: %+v", m)
	}
	if gpuMem(nil) != nil || gpuMem(map[string]any{"memory_total": 0.0, "memory_free": 0.0}) != nil {
		t.Fatal("breakdown without device stats")
	}
}

// section is the part of body from the first occurrence of from up to the
// next occurrence of to after it ("" when from is absent; the rest of body
// when to is).
func section(body, from, to string) string {
	i := strings.Index(body, from)
	if i < 0 {
		return ""
	}
	if j := strings.Index(body[i+len(from):], to); j >= 0 {
		return body[i : i+len(from)+j]
	}
	return body[i:]
}

type fakeModels struct {
	mu      sync.Mutex
	state   ModelsState
	calls   []string
	err     error
	restart int
	start   int
	stop    int
}

func (f *fakeModels) rec(s string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, s)
	return f.err
}
func (f *fakeModels) State() ModelsState {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.state
}
func (f *fakeModels) Verify(kind, id string) error { return f.rec("verify " + kind + " " + id) }
func (f *fakeModels) Materialize(dev, model string) error {
	return f.rec("materialize " + dev + " " + model)
}
func (f *fakeModels) Repair(dev, model string) error   { return f.rec("repair " + dev + " " + model) }
func (f *fakeModels) Activate(dev, model string) error { return f.rec("activate " + dev + " " + model) }
func (f *fakeModels) Remove(kind, id string) error     { return f.rec("remove " + kind + " " + id) }
func (f *fakeModels) Start() error                     { f.start++; return f.rec("start") }
func (f *fakeModels) Stop() error                      { f.stop++; return f.rec("stop") }
func (f *fakeModels) Restart() error                   { f.restart++; return f.rec("restart") }

func modelsInventory() setup.Inventory {
	return setup.Inventory{
		Runtimes: []setup.RuntimeEntry{
			{ID: "cu128-aaaa", Device: "cuda", Platform: "windows/amd64", Python: "3.12.11", Provider: "laya==0.3.21", Torch: "2.11.0+cu128", Supported: true, Materialized: true, Active: true, Verified: true},
			{ID: "cpu-bbbb", Device: "cpu", Platform: "windows/amd64", Python: "3.12.11", Provider: "laya==0.3.21", Torch: "2.11.0+cpu", Supported: true, Materialized: true},
		},
		Models: []setup.ModelEntry{
			{ID: "laya-base", Provider: "laya", Repo: "convaiinnovations/laya", Revision: "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851", Files: 5, Materialized: true, Active: true},
			{ID: "laya-other", Provider: "laya", Repo: "example/other", Revision: "1234567890abcdef1234567890abcdef12345678", Files: 1, Materialized: true},
			{ID: "laya-absent", Provider: "laya", Repo: "example/absent", Revision: "fedcba0987654321fedcba0987654321fedcba09", Files: 1},
		},
	}
}

func withModels(e *env, m Models) {
	cfg := e.d.cfg
	cfg.Models = m
	e.d = New(cfg)
}

// The manager renders the immutable catalog facts and the typed state,
// offers removal only for materialized unused artifacts, and exists only
// when the maintenance authority is hosted.
func TestModelsManagerView(t *testing.T) {
	e := newEnv(t)
	if rec := e.post(t, "/models/activate", url.Values{"device": {"cpu"}}); rec.Code != http.StatusNotFound && rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("models route without authority: %d", rec.Code)
	}
	fm := &fakeModels{state: ModelsState{Inventory: modelsInventory(), RestartRequired: true,
		Busy: &ModelOp{Kind: "materialize", Device: "cpu", Model: "laya-other", Phase: "model", Plan: []string{"preparing", "runtime", "model", "publish"},
			Phases: []string{"preparing", "runtime", "model"}, Step: "downloading", Detail: "model.safetensors", Item: 3, Items: 7, Done: 412 << 20, Total: 800 << 20,
			Started: time.Now().Add(-130 * time.Second)},
		Last: &ModelOp{Kind: "verify", Target: "model laya-base", Plan: []string{"model"}, Phases: []string{"model"}, Failure: "sha256 mismatch",
			FailurePhase: "model", FailureStep: "verifying", Log: "/home/x/logs/setup.log", Started: time.Now().Add(-5 * time.Second), Finished: time.Now()}}}
	withModels(e, fm)
	body := e.get(t, "/models").Body.String()
	for _, want := range []string{`id="models-runtimes"`, "cu128-aaaa", "windows/amd64 · python 3.12.11 · laya",
		"convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851", "5 pinned file(s)", `<span class="badge tone-ok">active</span>`,
		`<span class="badge">not materialized</span>`, `id="restart-required"`, `action="/models/restart"`,
		`id="models-busy"`, "phase 3 of 4 · Downloading", `(3 of 7)`, "412.0 MiB of 800.0 MiB (52%)", `aria-valuenow="52"`,
		`id="models-last"`, "sha256 mismatch", "Failed in phase <strong>Model</strong> · Verifying", "The active runtime and model were not changed.",
		`href="/diagnostics"`, "/home/x/logs/setup.log",
		`formaction="/models/materialize"`, `formaction="/models/repair"`, `formaction="/models/activate"`,
		`<option value="cuda">`, `<option value="laya-absent">`} {
		if !strings.Contains(body, want) {
			t.Errorf("manager lacks %q", want)
		}
	}
	// Active artifacts have no Remove action; unused materialized ones do.
	if n := strings.Count(body, `action="/models/remove"`); n != 2 {
		t.Errorf("%d remove forms, want 2 (unused runtime, unused model)", n)
	}
	for _, bad := range []string{`name="id" value="cu128-aaaa"><button type="submit" class="btn danger"`, `name="id" value="laya-base"><button type="submit" class="btn danger"`, `name="id" value="laya-absent"><button type="submit" class="btn danger"`} {
		if strings.Contains(body, bad) {
			t.Errorf("Remove offered for %q", bad)
		}
	}
	if strings.Count(body, `id="models-runtimes"`) != 1 || !strings.Contains(navRe.FindString(body), "/models") || !strings.Contains(navRe.FindString(body), "/forge") {
		t.Error("navigation/section mismatch")
	}
}

// Every form action is forwarded to the authority with explicit arguments,
// returns to Settings, shows refusals, and needs the form token.
func TestModelsActionsForwarded(t *testing.T) {
	e := newEnv(t)
	fm := &fakeModels{state: ModelsState{Inventory: modelsInventory()}}
	withModels(e, fm)
	for _, c := range []struct {
		op   string
		form url.Values
		want string
	}{
		{"verify", url.Values{"kind": {"model"}, "id": {"laya-base"}}, "verify model laya-base"},
		{"materialize", url.Values{"device": {"cuda"}, "model": {"laya-base"}}, "materialize cuda laya-base"},
		{"repair", url.Values{"device": {"cpu"}, "model": {"laya-base"}}, "repair cpu laya-base"},
		{"activate", url.Values{"device": {"cuda"}, "model": {"laya-other"}}, "activate cuda laya-other"},
		{"remove", url.Values{"kind": {"runtime"}, "id": {"cpu-bbbb"}}, "remove runtime cpu-bbbb"},
		{"restart", url.Values{}, "restart"},
	} {
		c.form.Set("return", "models")
		rec := e.post(t, "/models/"+c.op, c.form)
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/models" {
			t.Fatalf("%s: %d %s", c.op, rec.Code, rec.Header().Get("Location"))
		}
		if a := e.lastAction(t); !a.OK {
			t.Fatalf("%s: %+v", c.op, a)
		}
		if got := fm.calls[len(fm.calls)-1]; got != c.want {
			t.Fatalf("%s forwarded %q, want %q", c.op, got, c.want)
		}
	}
	if rec := e.post(t, "/models/format-disk", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("unknown op: %d", rec.Code)
	}
	if rec := e.post(t, "/models/remove", url.Values{"token": {"forged"}, "kind": {"model"}, "id": {"laya-other"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
	n := len(fm.calls)
	fm.err = errors.New("the active runtime/model cannot be removed")
	e.post(t, "/models/remove", url.Values{"kind": {"model"}, "id": {"laya-base"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "cannot be removed") || len(fm.calls) != n+1 {
		t.Fatalf("refusal not shown: %+v", a)
	}
	// Activation forwarded no lifecycle call: the worker is untouched.
	e.rt.mu.Lock()
	calls := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(calls) != 0 {
		t.Fatalf("lifecycle touched: %v", calls)
	}
}

// The last explicit verification, kept by the application, is shown beside
// the artifact.
func TestModelsVerifyResultShown(t *testing.T) {
	e := newEnv(t)
	fm := &fakeModels{state: ModelsState{Inventory: modelsInventory(), Checks: map[string]ModelCheck{
		"model laya-other": {OK: false, Msg: "config.json: sha256 bad", Time: time.Now()},
		"model laya-base":  {OK: true, Time: time.Now()},
	}}}
	withModels(e, fm)
	body := e.get(t, "/models").Body.String()
	if !strings.Contains(body, "failed 20") || !strings.Contains(body, "config.json: sha256 bad") || !strings.Contains(body, "verified 20") {
		t.Error("verification results not shown")
	}
	// Starting a verification only forwards it: the page decides nothing.
	if rec := e.post(t, "/models/verify", url.Values{"kind": {"model"}, "id": {"laya-other"}}); rec.Code != http.StatusSeeOther {
		t.Fatal(rec.Code)
	}
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, "started") {
		t.Fatalf("action %+v", a)
	}
}

// With the application hosted, every Runtime page lifecycle action is the
// application's. The worker lifecycle this dashboard was created with belongs
// to the runtime it was bound to; after an activation (or a failed start) only
// the application binds the active runtime and model, so a restart through the
// stale lifecycle would start the old model again.
func TestRuntimeLifecycleRoutesThroughApplication(t *testing.T) {
	e := newEnv(t)
	fm := &fakeModels{state: ModelsState{Inventory: modelsInventory()}}
	withModels(e, fm)
	for _, op := range []string{"start", "restart", "stop"} {
		e.post(t, "/runtime/"+op, nil)
	}
	e.rt.mu.Lock()
	lc := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(lc) != 0 || fm.start != 1 || fm.restart != 1 || fm.stop != 1 {
		t.Fatalf("lifecycle=%v app start=%d restart=%d stop=%d", lc, fm.start, fm.restart, fm.stop)
	}
	// A refusal from the application is shown, not swallowed.
	fm.err = errors.New("no valid active runtime; run setup first")
	e.post(t, "/runtime/restart", nil)
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "no valid active runtime") {
		t.Fatalf("refusal not shown: %+v", a)
	}
}

// Without the application (serve, dashboard) the worker lifecycle remains the
// authority.
func TestRuntimeLifecycleWithoutApplicationUsesTheWorkerLifecycle(t *testing.T) {
	e := newEnv(t)
	e.post(t, "/runtime/restart", nil)
	e.rt.mu.Lock()
	lc := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(lc) != 1 {
		t.Fatalf("lifecycle=%v", lc)
	}
}

// bundleFiles reads the only bundle exported under the test home.
func bundleFiles(t *testing.T, home string) map[string][]byte {
	t.Helper()
	dir := filepath.Join(home, "state", "diagnostics")
	ents, err := os.ReadDir(dir)
	if err != nil || len(ents) != 1 {
		t.Fatalf("bundle files = %v, %v", ents, err)
	}
	b, _ := os.ReadFile(filepath.Join(dir, ents[0].Name()))
	zr, err := zip.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	out := map[string][]byte{}
	for _, f := range zr.File {
		rc, _ := f.Open()
		out[f.Name], _ = io.ReadAll(rc)
		rc.Close()
	}
	return out
}

func TestDiagnosticsExportIsExplicitLocalAndAllowlisted(t *testing.T) {
	e := newEnv(t)
	dir := filepath.Join(e.home, "state", "diagnostics")
	// Rendering Diagnostics never exports; only the token-guarded POST does.
	body := e.get(t, "/diagnostics").Body.String()
	if !strings.Contains(body, `action="/diagnostics/export"`) || !strings.Contains(body, "Nothing is uploaded") {
		t.Error("Diagnostics lacks the export control")
	}
	if _, err := os.Stat(dir); err == nil {
		t.Fatal("a bundle directory exists before any explicit export")
	}
	req := httptest.NewRequest("POST", "http://127.0.0.1:7844/diagnostics/export", strings.NewReader(""))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	e.d.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("export without the form token = %d", rec.Code)
	}
	if _, err := os.Stat(dir); err == nil {
		t.Fatal("a tokenless request exported a bundle")
	}

	os.MkdirAll(filepath.Join(e.home, "logs"), 0o755)
	os.WriteFile(filepath.Join(e.home, "logs", "worker.log"), []byte("[worker] READY\nsecret request body\n"), 0o600)
	if rec := e.post(t, "/diagnostics/export", nil); rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/diagnostics" {
		t.Fatalf("export = %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, "nothing was uploaded") {
		t.Fatalf("action = %+v", a)
	}
	files := bundleFiles(t, e.home)
	if len(files) != 3 {
		t.Fatalf("archive has %d entries, want 3", len(files))
	}
	for name, data := range files {
		if bytes.Contains(data, []byte("secret request body")) || bytes.Contains(data, []byte("Traceback &")) {
			t.Errorf("%s carries excluded content", name)
		}
	}
	if !bytes.Contains(files["worker-log-tail.txt"], []byte("[worker] READY")) {
		t.Error("log tail lacks the Hachidori worker line")
	}
}

func TestDiagnosticsExportRecordsWebView2Version(t *testing.T) {
	e := newEnv(t)
	cfg := e.d.cfg
	cfg.WebView2 = "154.0.4258.37"
	e.d = New(cfg)
	e.post(t, "/diagnostics/export", nil)
	facts := bundleFiles(t, e.home)["facts.json"]
	if !bytes.Contains(facts, []byte(`"version": "154.0.4258.37"`)) || !bytes.Contains(facts, []byte(`"recovery": ""`)) {
		t.Errorf("facts lack WebView2 version / recovery state:\n%s", facts)
	}
}

// fakePathPicker answers every native choice with path, or err.
type fakePathPicker struct {
	path  string
	err   error
	calls []string
}

func (p *fakePathPicker) pick(kind string) (string, error) {
	p.calls = append(p.calls, kind)
	return p.path, p.err
}
func (p *fakePathPicker) PickOpen(context.Context, string) (string, error)   { return p.pick("open") }
func (p *fakePathPicker) PickSave(context.Context, string) (string, error)   { return p.pick("save") }
func (p *fakePathPicker) PickFolder(context.Context, string) (string, error) { return p.pick("folder") }

func withPathPicker(e *env, p PathPicker) {
	cfg := e.d.cfg
	cfg.PathPicker = p
	e.d = New(cfg)
}

func TestPathPickerButtonsOnlyInDesktopComposition(t *testing.T) {
	pages := []string{"/workbench", "/experiments", "/errors"}
	e := newEnv(t)
	for _, p := range pages {
		body := e.get(t, p).Body.String()
		if strings.Contains(body, "/pick") || strings.Contains(body, "pick-load") || strings.Contains(body, `C:\`) {
			t.Errorf("browser %s shows native picker actions or a Windows example path", p)
		}
		if !strings.Contains(body, `type="text"`) || strings.Contains(body, "Choose file") {
			t.Errorf("browser %s lost typed path entry", p)
		}
	}
	for _, p := range []string{"/experiments/pick", "/errors/pick"} {
		if body := e.post(t, p, url.Values{"pick": {"dataset"}}).Body.String(); !strings.Contains(body, "native path selection is unavailable") {
			t.Errorf("%s without a picker: %s", p, body)
		}
	}
	withPathPicker(e, &fakePathPicker{})
	for _, p := range pages {
		body := e.get(t, p).Body.String()
		if !strings.Contains(body, "…</button>") || !strings.Contains(body, "formnovalidate") || strings.Contains(body, "Choose file") {
			t.Errorf("desktop %s lacks semantically named native picker actions", p)
		}
	}
	if body := e.post(t, "/experiments/pick", url.Values{"token": {"stale"}, "pick": {"dataset"}}); body.Code != http.StatusForbidden {
		t.Errorf("stale token pick = %d", body.Code)
	}
}

func TestExperimentsPickKeepsTheForm(t *testing.T) {
	e := newEnv(t)
	p := &fakePathPicker{path: "/data/chosen"}
	withPathPicker(e, p)
	typed := url.Values{"dataset": {"/data/typed.jsonl"}, "definitions": {"/defs/a.json"}, "warmup": {"2"}, "passes": {"3"}}
	posted := func(kind string) url.Values {
		v := url.Values{"pick": {kind}}
		for k, vs := range typed {
			v[k] = vs
		}
		return v
	}

	body := e.post(t, "/experiments/pick", posted("dataset")).Body.String()
	for _, s := range []string{`<option value="/data/chosen" selected>chosen</option>`, `<option value="/defs/a.json" selected>a.json</option>`, `name="warmup" value="2"`, `name="passes" value="3"`} {
		if !strings.Contains(body, s) {
			t.Errorf("dataset pick: form lacks %q", s)
		}
	}
	body = e.post(t, "/experiments/pick", posted("definition-file")).Body.String()
	if !strings.Contains(body, "<option value=\"/defs/a.json\n/data/chosen\" selected>a.json, chosen</option>") {
		t.Error("definition file pick replaced the definitions already entered")
	}
	body = e.post(t, "/experiments/pick", posted("definition-folder")).Body.String()
	if !strings.Contains(body, "<option value=\"/defs/a.json\n/data/chosen\" selected>") || p.calls[len(p.calls)-1] != "folder" {
		t.Error("definition folder pick did not append a folder")
	}

	p.err = ErrPickCancelled
	body = e.post(t, "/experiments/pick", posted("dataset")).Body.String()
	if !strings.Contains(body, `<option value="/data/typed.jsonl" selected>`) || strings.Contains(body, "choosing a path") {
		t.Error("cancellation changed the form or reported a failure")
	}
	p.err = errors.New("dialog broke")
	body = e.post(t, "/experiments/pick", posted("dataset")).Body.String()
	if !strings.Contains(body, `<option value="/data/typed.jsonl" selected>`) || !strings.Contains(body, "choosing a path: dialog broke") {
		t.Error("picker failure lost the form or its error")
	}
}

func TestErrorsPickFillsPathsWithoutOpeningOrWriting(t *testing.T) {
	e := newEnv(t)
	p := &fakePathPicker{path: "/reports/chosen.json"}
	withPathPicker(e, p)
	body := e.post(t, "/errors/pick", url.Values{"pick": {"open"}}).Body.String()
	if !strings.Contains(body, `name="path" value="/reports/chosen.json"`) || p.calls[0] != "open" {
		t.Errorf("open pick did not fill the report path")
	}
	p.err = ErrPickCancelled
	body = e.post(t, "/errors/pick", url.Values{"pick": {"open"}, "path": {"/typed.json"}}).Body.String()
	if strings.Contains(body, "choosing a path") || !strings.Contains(body, `name="path" value="/typed.json"`) {
		t.Error("cancelled open pick reported a failure or lost the typed path")
	}
	if _, err := os.Stat("/reports/chosen.json"); err == nil {
		t.Fatal("unexpected file")
	}
}

// Selection, activation and the running worker are three different facts. A
// model chosen in the form changes nothing by itself; once OpenDecider-nano is
// activated under a running Laya worker, the manager shows Laya as serving now,
// OpenDecider-nano as active but applying only on restart, and offers the
// restart explicitly. Nothing is downloaded, activated or restarted by viewing
// or by choosing.
func TestModelsManagerSeparatesSelectionActivationAndRunning(t *testing.T) {
	e := newEnv(t) // the running worker serves laya-base on cuda
	inv := modelsInventory()
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "opendecider-nano", Model: "manjunathshiva--opendecider-nano/7e42a1508d2beef44717d044831e87f2fc4db9f2", Device: "cuda"}
	inv.Models[0].Active = false
	inv.Models = append(inv.Models, setup.ModelEntry{ID: "opendecider-nano", Provider: "opendecider", Repo: "manjunathshiva/opendecider-nano",
		Revision: "7e42a1508d2beef44717d044831e87f2fc4db9f2", Files: 7, Materialized: true, Active: true})
	fm := &fakeModels{state: ModelsState{Inventory: inv, RestartRequired: true}}
	withModels(e, fm)

	body := e.get(t, "/models?model=opendecider-nano").Body.String()
	for _, want := range []string{
		`id="models-serving"`, `id="artifact-running" data-artifact="SOURCE"`, `id="artifact-next" data-artifact="SOURCE"`,
		`<span class="badge tone-ok">running</span>`, `<span class="badge tone-ok">active · applies on restart</span>`,
		`<option value="opendecider-nano" selected>`, `<option value="cuda" selected>`,
		`id="restart-required"`, "changes nothing by itself", "opendecider · manjunathshiva/opendecider-nano@7e42a1508d2beef44717d044831e87f2fc4db9f2",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("manager lacks %q", want)
		}
	}
	if strings.Count(body, `<option value="laya-base" selected>`) != 0 {
		t.Error("the form pre-selects a model that is not the active one")
	}
	run, next := section(body, `id="artifact-running"`, `id="artifact-next"`), section(body, `id="artifact-next"`, `id="artifact-differs"`)
	if !strings.Contains(run, `<dd class="mono">laya-base</dd>`) || !strings.Contains(next, `<dd class="mono">opendecider-nano</dd>`) || !strings.Contains(body, `id="artifact-differs"`) {
		t.Errorf("running and next-start artifacts are not told apart:\nrunning:%s\nnext:%s", run, next)
	}
	e.rt.mu.Lock()
	lc := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(fm.calls) != 0 || len(lc) != 0 {
		t.Fatalf("viewing/choosing acted: maintenance=%v lifecycle=%v", fm.calls, lc)
	}

	// After the restart the worker serves what is active and nothing is pending.
	fm.state.RestartRequired = false
	cfg, status := e.d.cfg, e.d.cfg.Status
	cfg.Status = func() server.Status {
		st := status()
		st.Runtime.ModelID = "opendecider-nano"
		return st
	}
	e.d = New(cfg)
	body = e.get(t, "/models").Body.String()
	if run := section(body, `id="artifact-running"`, `id="artifact-next"`); !strings.Contains(run, `<dd class="mono">opendecider-nano</dd>`) || strings.Contains(body, "applies on restart") || strings.Contains(body, `id="restart-required"`) || strings.Contains(body, `id="artifact-differs"`) {
		t.Error("after restart the manager still reports a pending change")
	}
}
