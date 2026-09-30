package dashboard

import (
	"archive/zip"
	"bytes"
	"encoding/json"
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
	"github.com/yohn-jp/hachidori/internal/server"
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

func newEnv(t *testing.T) *env {
	t.Helper()
	exe, _ := os.Executable()
	t.Setenv("HACHIDORI_FAKE_SSH", "run")
	home := t.TempDir()
	os.MkdirAll(filepath.Join(home, "state"), 0o755)
	rt := &fakeRuntime{run: true, snap: worker.Snapshot{
		State: worker.StateReady, Phase: "ready", Ready: true, PID: 4242, Starts: 2, Restarts: 1, Requests: 17,
		Errors: map[string]int64{"not_ready": 1}, QueueDepth: 0, QueueLimit: 64, LatencyP50MS: 25.3, LatencyP95MS: 30.5,
		Info: worker.Info{"provider": "laya", "laya_version": "0.3.21", "torch_version": "2.11.0+cu128", "torch_cuda": "12.8",
			"python_version": "3.12.11", "device": "cuda:0", "device_name": "NVIDIA GeForce RTX 3060", "load_ms": 812.5, "warmup_ms": 90.1},
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

func (e *env) get(t *testing.T, path string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "http://127.0.0.1:7844"+path, nil)
	e.d.ServeHTTP(rec, req)
	return rec
}

func (e *env) post(t *testing.T, path string, form url.Values) *httptest.ResponseRecorder {
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

func TestStatusIsTheV1StatusDocument(t *testing.T) {
	e := newEnv(t)
	rec := e.get(t, "/api/status")
	api := httptest.NewRecorder()
	e.api.ServeHTTP(api, httptest.NewRequest("GET", "/v1/status", nil))
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
	// The page renders values from that same document.
	body := e.get(t, "/").Body.String()
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
	}{"/": {root, []string{"/runtime/start", "/runtime/stop", "/runtime/restart"}},
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
	if len(a) < 2 || a[0].Title != "Runtime is not running" || a[0].Level != "bad" || a[1].Level != "bad" {
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
