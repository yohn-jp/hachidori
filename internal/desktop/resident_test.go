package desktop

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// fakeApp records everything the tray asks of the application controller.
type fakeApp struct {
	mu       sync.Mutex
	snap     app.Snapshot
	restarts int
	other    int // any call other than Snapshot/Restart would land here
}

func (f *fakeApp) Snapshot() app.Snapshot { f.mu.Lock(); defer f.mu.Unlock(); return f.snap }
func (f *fakeApp) Restart() error         { f.mu.Lock(); defer f.mu.Unlock(); f.restarts++; return nil }
func (f *fakeApp) set(s app.State)        { f.mu.Lock(); f.snap = app.Snapshot{State: s}; f.mu.Unlock() }

type fakeStartup struct {
	on      bool
	command string
	enables int
	writes  int // times the stored command actually changed
}

func (f *fakeStartup) Enabled() (bool, error) { return f.on, nil }
func (f *fakeStartup) Enable(c string) error {
	f.enables++
	if !f.on || f.command != c {
		f.writes++
	}
	f.on, f.command = true, c
	return nil
}
func (f *fakeStartup) Disable() error { f.on, f.command = false, ""; return nil }

func newResident(t *testing.T) (*Resident, *fakeApp, *fakeStartup) {
	t.Helper()
	fa, fs := &fakeApp{}, &fakeStartup{}
	m := &Manager{Path: filepath.Join(t.TempDir(), "desktop.json"), Startup: fs, Command: `"C:\h\hachidori.exe" desktop --background`}
	return &Resident{App: fa, Prefs: m}, fa, fs
}

// 5: tray status derives from canonical application state.
func TestSummarizeMapsEveryApplicationState(t *testing.T) {
	want := map[app.State]struct {
		label string
		level Level
	}{
		app.Ready:        {"Ready", LevelOK},
		app.Starting:     {"Starting", LevelBusy},
		app.Warming:      {"Starting", LevelBusy},
		app.Installing:   {"Setting up", LevelBusy},
		app.Stopping:     {"Stopping", LevelBusy},
		app.Installed:    {"Stopped", LevelBusy},
		app.Failed:       {"Needs attention", LevelAttention},
		app.Unconfigured: {"Needs attention", LevelAttention},
		app.NotInstalled: {"Needs attention", LevelAttention},
		app.State("x"):   {"Needs attention", LevelAttention}, // unknown is never "Ready"
	}
	for st, w := range want {
		got := Summarize(app.Snapshot{State: st})
		if got.Label != w.label || got.Level != w.level || got.State != st {
			t.Errorf("Summarize(%q) = %+v, want %+v", st, got, w)
		}
	}
}

// fakeRuntime is a controller Runtime whose worker state the test sets.
type fakeRuntime struct {
	mu      sync.Mutex
	running bool
	st      string
	phase   string
}

func (f *fakeRuntime) Start() bool   { f.mu.Lock(); defer f.mu.Unlock(); f.running = true; return true }
func (f *fakeRuntime) Stop()         { f.mu.Lock(); defer f.mu.Unlock(); f.running = false }
func (f *fakeRuntime) Restart()      {}
func (f *fakeRuntime) Running() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.running }
func (f *fakeRuntime) Status() server.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	return server.Status{Worker: worker.Snapshot{State: f.st, Phase: f.phase}}
}
func (f *fakeRuntime) set(st, phase string) { f.mu.Lock(); f.st, f.phase = st, phase; f.mu.Unlock() }

// The tray follows the real controller: worker transitions observed through
// the supervisor status surface as the tray label with no tray-side logic.
func TestTrayFollowsTheRealControllerState(t *testing.T) {
	rt := &fakeRuntime{st: worker.StateStarting, phase: "loading"}
	c := app.New(app.Config{
		Home:      t.TempDir(),
		Installed: func(string) bool { return true },
		Open:      func(string) (app.Runtime, error) { return rt, nil },
	})
	r := &Resident{App: c, Prefs: &Manager{}}
	if got := r.Summary().Label; got != "Stopped" {
		t.Fatalf("before start: %q", got)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	for _, step := range []struct{ st, phase, label string }{
		{worker.StateStarting, "loading", "Starting"},
		{worker.StateStarting, "warming", "Starting"},
		{worker.StateReady, "ready", "Ready"},
		{worker.StateFailed, "failed", "Needs attention"},
	} {
		rt.set(step.st, step.phase)
		if got := r.Summary().Label; got != step.label {
			t.Errorf("worker %s/%s: tray %q, want %q", step.st, step.phase, got, step.label)
		}
	}
	if err := c.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestMenuHasTheRequiredEntries(t *testing.T) {
	m := BuildMenu(Summarize(app.Snapshot{State: app.Ready}), Preferences{StartAtSignIn: true})
	var labels []string
	for _, it := range m {
		if !it.Separator {
			labels = append(labels, it.Label)
		}
	}
	want := []string{"Hachidori: Ready", "Open Hachidori", "Restart Runtime", "Diagnostics",
		"Start Hachidori when I sign in", "Start minimized", "Quit Hachidori"}
	if strings.Join(labels, "|") != strings.Join(want, "|") {
		t.Fatalf("menu %v, want %v", labels, want)
	}
	if !m[0].Disabled {
		t.Error("the state line must be read-only")
	}
	for _, it := range m {
		if it.ID == MenuStartAtSignIn && !it.Checked {
			t.Error("start-at-sign-in not shown checked")
		}
		if it.ID == MenuStartMinimized && it.Checked {
			t.Error("start-minimized shown checked")
		}
	}
	// Restart is not offered while there is nothing installed to restart.
	for _, st := range []app.State{app.Unconfigured, app.NotInstalled, app.Installing, app.Stopping} {
		for _, it := range BuildMenu(Summarize(app.Snapshot{State: st}), Preferences{}) {
			if it.ID == MenuRestart && !it.Disabled {
				t.Errorf("Restart Runtime enabled in state %s", st)
			}
		}
	}
}

// 1: close-to-tray leaves runtime ownership intact.
func TestCloseHidesToTrayAndNeverTouchesTheRuntime(t *testing.T) {
	r, fa, _ := newResident(t)
	fa.set(app.Ready)
	if got := r.OnClose(); got != ActionHideWithNotice {
		t.Fatalf("first close = %v, want hide with the discoverability notice", got)
	}
	for i := 0; i < 3; i++ {
		if got := r.OnClose(); got != ActionHide {
			t.Fatalf("later close #%d = %v, want plain hide", i, got)
		}
	}
	if fa.restarts != 0 || fa.other != 0 {
		t.Fatalf("closing the window acted on the runtime: %+v", fa)
	}
}

// 2: Open restores the existing window (an action on the same window, never
// a second launch). A failed state opens on diagnostics.
func TestOpenShowsExistingWindowAndFailureOpensDiagnostics(t *testing.T) {
	r, fa, _ := newResident(t)
	fa.set(app.Ready)
	if a, err := r.OnMenu(MenuOpen); err != nil || a != ActionShow {
		t.Fatalf("Open (ready) = %v, %v", a, err)
	}
	fa.set(app.Failed)
	if a, _ := r.OnMenu(MenuOpen); a != ActionShowDiagnostics {
		t.Fatalf("Open (failed) = %v, want diagnostics", a)
	}
	if a := r.OnActivate(); a != ActionShowDiagnostics {
		t.Fatalf("activation (failed) = %v, want diagnostics", a)
	}
	fa.set(app.Ready)
	if a := r.OnActivate(); a != ActionShow {
		t.Fatalf("activation (ready) = %v", a)
	}
	if a, _ := r.OnMenu(MenuDiagnostics); a != ActionShowDiagnostics {
		t.Fatalf("Diagnostics = %v", a)
	}
	if fa.restarts != 0 {
		t.Fatal("opening restarted the runtime")
	}
	if got := OpenURL("http://127.0.0.1:7844/", Summarize(app.Snapshot{State: app.Failed})); got != "http://127.0.0.1:7844/#diagnostics" {
		t.Fatalf("OpenURL(failed) = %q", got)
	}
	if got := OpenURL("http://127.0.0.1:7844/", Summarize(app.Snapshot{State: app.Ready})); got != "http://127.0.0.1:7844/" {
		t.Fatalf("OpenURL(ready) = %q", got)
	}
	// The diagnostics URL stays inside the navigation allow-list.
	p, _ := NewPolicy("http://127.0.0.1:7844/")
	if !p.AllowNavigation("http://127.0.0.1:7844/" + DiagnosticsFragment) {
		t.Fatal("diagnostics URL is outside the navigation policy")
	}
}

// 3: Quit ends the session (the shell then shuts owned resources down), and
// after Quit a close is a real close.
func TestQuitIsExplicitAndFinal(t *testing.T) {
	r, fa, _ := newResident(t)
	if a := r.OnClose(); a == ActionQuit {
		t.Fatal("closing the window must not quit")
	}
	if a, err := r.OnMenu(MenuQuit); err != nil || a != ActionQuit {
		t.Fatalf("Quit = %v, %v", a, err)
	}
	if a := r.OnClose(); a != ActionQuit {
		t.Fatalf("close after Quit = %v, want quit", a)
	}
	if fa.restarts != 0 {
		t.Fatal("Quit restarted the runtime")
	}
}

// Restart Runtime is the only runtime action the tray performs, once per
// request, and only when the user asks.
func TestRestartRuntimeIsExplicitOnly(t *testing.T) {
	r, fa, _ := newResident(t)
	fa.set(app.Failed) // a failed start is shown, never auto-retried
	r.OnActivate()
	r.Summary()
	r.Menu()
	if fa.restarts != 0 {
		t.Fatal("state observation restarted the runtime (unbounded restart loop risk)")
	}
	if a, err := r.OnMenu(MenuRestart); err != nil || a != ActionNone {
		t.Fatalf("Restart = %v, %v", a, err)
	}
	r.Wait()
	if fa.restarts != 1 {
		t.Fatalf("restarts = %d, want 1", fa.restarts)
	}
}

func TestUnknownMenuEntryIsRejected(t *testing.T) {
	r, _, _ := newResident(t)
	if _, err := r.OnMenu(MenuID(99)); err == nil {
		t.Fatal("unknown menu entry accepted")
	}
}

// 6: login-start enable/disable through the menu is idempotent.
func TestStartAtSignInToggleIsIdempotent(t *testing.T) {
	r, _, fs := newResident(t)
	if _, err := r.OnMenu(MenuStartAtSignIn); err != nil || !fs.on {
		t.Fatalf("toggle on: %v on=%v", err, fs.on)
	}
	if _, err := r.OnMenu(MenuStartAtSignIn); err != nil || fs.on {
		t.Fatalf("toggle off: %v on=%v", err, fs.on)
	}
	m := r.Prefs
	for i := 0; i < 3; i++ {
		if err := m.SetStartAtSignIn(true); err != nil {
			t.Fatal(err)
		}
	}
	if fs.writes != 2 { // once for the toggle-on, once after the disable; never on repeats
		t.Fatalf("startup entry rewritten %d times, want 2", fs.writes)
	}
	for i := 0; i < 3; i++ {
		if err := m.SetStartAtSignIn(false); err != nil {
			t.Fatal(err)
		}
	}
	if p, err := m.Get(); err != nil || p.StartAtSignIn {
		t.Fatalf("after disable: %+v %v", p, err)
	}
}

// 8: start minimized applies to a sign-in launch only; the runtime start is
// independent of window visibility.
func TestStartHidden(t *testing.T) {
	for _, c := range []struct {
		bg, min, want bool
	}{{true, true, true}, {true, false, false}, {false, true, false}, {false, false, false}} {
		if got := StartHidden(c.bg, Preferences{StartMinimized: c.min}); got != c.want {
			t.Errorf("StartHidden(bg=%v, minimized=%v) = %v", c.bg, c.min, got)
		}
	}
}

func TestPreferencesRoundTripAndAreSmallIntegrationMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "desktop.json")
	m := &Manager{Path: path, Startup: &fakeStartup{}}
	if p, err := m.Get(); err != nil || p.StartMinimized || p.StartAtSignIn {
		t.Fatalf("defaults: %+v %v", p, err)
	}
	if err := m.SetStartMinimized(true); err != nil {
		t.Fatal(err)
	}
	m2 := &Manager{Path: path, Startup: &fakeStartup{}}
	if p, _ := m2.Get(); !p.StartMinimized {
		t.Fatal("start minimized not persisted")
	}
	// The record carries no runtime/model/secret fields.
	var raw map[string]any
	if err := readJSONFile(path, &raw); err != nil {
		t.Fatal(err)
	}
	for k := range raw {
		switch k {
		case "schema", "start_minimized", "close_notice_shown":
		default:
			t.Errorf("unexpected preferences field %q", k)
		}
	}
	if m2.ShouldShowCloseNotice() != true || m2.ShouldShowCloseNotice() != false {
		t.Fatal("close notice must be shown exactly once")
	}
	m3 := &Manager{Path: path}
	if m3.ShouldShowCloseNotice() {
		t.Fatal("close notice repeated across restarts")
	}
}

func TestMalformedPreferencesAreReportedNotFatal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "desktop.json")
	writeFile(t, path, `{"schema":"other/9"}`)
	m := &Manager{Path: path, Startup: &fakeStartup{on: true}}
	p, err := m.Get()
	if err == nil {
		t.Fatal("unknown schema not reported")
	}
	if !p.StartAtSignIn || p.StartMinimized {
		t.Fatalf("defaults + real startup state expected, got %+v", p)
	}
	if m.ShouldShowCloseNotice() {
		t.Fatal("unreadable record must not trigger notices")
	}
	if err := m.SetStartMinimized(true); err == nil {
		t.Fatal("overwrote a record of unknown schema")
	}
}

func TestManagerWithoutStartupIsUnsupportedNotSilent(t *testing.T) {
	m := &Manager{}
	if err := m.SetStartAtSignIn(true); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("err = %v", err)
	}
}

// 7: login-start is per-user and unelevated by construction.
func TestStartupMechanismIsPerUserRunKey(t *testing.T) {
	if !strings.HasPrefix(RunKeyPath, `Software\Microsoft\Windows\CurrentVersion\Run`) {
		t.Fatalf("RunKeyPath %q is not the per-user Run key path", RunKeyPath)
	}
	got, err := StartupCommand(`C:\Program Files\Hachidori\hachidori.exe`, "")
	if err != nil || got != `"C:\Program Files\Hachidori\hachidori.exe" desktop --background` {
		t.Fatalf("command = %q, %v", got, err)
	}
	got, err = StartupCommand(`C:\h\hachidori.exe`, `D:\`)
	if err != nil || got != `"C:\h\hachidori.exe" desktop --background --home "D:\\"` {
		t.Fatalf("command = %q, %v", got, err)
	}
	for _, bad := range [][2]string{{"", ""}, {`C:\a"b.exe`, ""}, {`C:\a.exe`, "D:\\x\"y"}, {"C:\\a\n.exe", ""}} {
		if _, err := StartupCommand(bad[0], bad[1]); err == nil {
			t.Errorf("StartupCommand(%q,%q) accepted", bad[0], bad[1])
		}
	}
}

func TestWindowClassAndActivateMessageArePerUser(t *testing.T) {
	a, b := WindowClassName("S-1-5-21-1"), WindowClassName("S-1-5-21-2")
	if a == b || a != WindowClassName("S-1-5-21-1") {
		t.Fatalf("window class not per-user/deterministic: %q %q", a, b)
	}
	if ActivateMessageName("S-1-5-21-1") == ActivateMessageName("S-1-5-21-2") {
		t.Fatal("activate message shared between users")
	}
	if strings.Contains(a, "21-1") {
		t.Fatalf("class %q leaks the raw user id", a)
	}
}
