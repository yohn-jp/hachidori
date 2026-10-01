package main

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

type fakeDesktop struct {
	held      bool
	activated int
	opened    int
	reported  []string
	activeErr error
	open      func(context.Context, desktop.Window) error
}

func (f *fakeDesktop) RuntimeVersion() (string, error) { return "129.0.0.0", nil }
func (f *fakeDesktop) AcquireInstance() (func(), error) {
	if f.held {
		return nil, desktop.ErrAlreadyRunning
	}
	f.held = true
	return func() { f.held = false }, nil
}
func (f *fakeDesktop) Open(ctx context.Context, w desktop.Window) error {
	f.opened++
	if f.open != nil {
		return f.open(ctx, w)
	}
	return nil
}
func (f *fakeDesktop) Activate() error               { f.activated++; return f.activeErr }
func (f *fakeDesktop) ReportError(title, msg string) { f.reported = append(f.reported, msg) }

type noStartup struct{}

func (noStartup) Enabled() (bool, error) { return false, nil }
func (noStartup) Enable(string) error    { return nil }
func (noStartup) Disable() error         { return nil }

// fakeRuntime is a controllable app.Runtime that counts lifecycle calls.
type fakeRuntime struct {
	mu                      sync.Mutex
	running                 bool
	starts, stops, restarts int
}

func (r *fakeRuntime) Start() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.running {
		return false
	}
	r.running = true
	r.starts++
	return true
}
func (r *fakeRuntime) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.running = false
	r.stops++
}
func (r *fakeRuntime) Restart() { r.mu.Lock(); r.restarts++; r.mu.Unlock() }
func (r *fakeRuntime) Running() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running
}
func (r *fakeRuntime) Status() server.Status {
	return server.Status{Schema: "hachidori.v1", Worker: worker.Snapshot{State: worker.StateReady, Ready: true}}
}
func (r *fakeRuntime) counts() (int, int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.starts, r.stops, r.restarts
}

// installedApp is the unified composition on a configured, installed home whose
// runtime is the returned fake. opens counts how many runtimes were bound: more
// than one would be a second worker owner.
func installedApp(t testing.TB, p desktop.Platform, background bool) (*desktopApp, *fakeRuntime, *atomic.Int32) {
	t.Helper()
	root := t.TempDir()
	rt, opens := &fakeRuntime{}, &atomic.Int32{}
	a, _, _ := testApp(p, func() (home.Discovery, error) {
		return home.Discovery{Home: home.Home{Root: root}, Source: home.SourceLocator}, nil
	})
	a.Env.Load = func(string) (home.Active, error) { return home.Active{}, nil }
	a.Open = func(string) (app.Runtime, error) { opens.Add(1); return rt, nil }
	a.PrefsPath = filepath.Join(t.TempDir(), "desktop.json")
	a.Startup = noStartup{}
	a.Background = background
	return a, rt, opens
}

// A loopback port that is already taken ends the launch before any runtime is
// bound or started: nothing is left running when run returns.
func TestListenFailureStartsNoRuntime(t *testing.T) {
	for _, taken := range []string{"api", "dashboard"} {
		t.Run(taken, func(t *testing.T) {
			busy, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer busy.Close()
			f := &fakeDesktop{}
			a, rt, opens := installedApp(t, f, false)
			if taken == "api" {
				a.APIAddr = busy.Addr().String()
			} else {
				a.DashAddr = busy.Addr().String()
			}
			err = a.run()
			if err == nil || !strings.Contains(err.Error(), busy.Addr().String()) {
				t.Fatalf("run = %v, want the address-in-use error naming %s", err, busy.Addr())
			}
			starts, _, _ := rt.counts()
			if starts != 0 || opens.Load() != 0 || f.opened != 0 {
				t.Fatalf("runtime started %d, bound %d, windows %d; a launch that cannot serve must start nothing", starts, opens.Load(), f.opened)
			}
			if f.held {
				t.Fatal("the single-instance guard was not released")
			}
		})
	}
}

// A duplicate launch, from either entry, activates the running instance and
// starts nothing: no worker log, no controller, no window of its own.
func TestDuplicateLaunchActivatesExistingInstance(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	homeDir := t.TempDir()
	f := &fakeDesktop{held: true} // another instance of this user is running
	err := runDesktop(f, &neverPicker{}, noStartup{}, filepath.Join(t.TempDir(), "desktop.json"), []string{"--home", homeDir})
	if err != nil {
		t.Fatalf("duplicate launch returned %v; activating the running instance is a success", err)
	}
	if f.activated != 1 || f.opened != 0 || len(f.reported) != 0 {
		t.Fatalf("activated=%d opened=%d reported=%v", f.activated, f.opened, f.reported)
	}
	if _, err := os.Stat(filepath.Join(homeDir, "logs")); !os.IsNotExist(err) {
		t.Fatalf("a duplicate launch touched the home (a second runtime owner started): %v", err)
	}

	a, _, opens := installedApp(t, f, false)
	if err := a.launch(); err != nil || f.activated != 2 || f.opened != 0 || opens.Load() != 0 {
		t.Fatalf("no-argument duplicate launch: err=%v activated=%d opened=%d runtimes=%d", err, f.activated, f.opened, opens.Load())
	}
}

func TestDuplicateLaunchWhoseActivationFailsStaysAnError(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	f := &fakeDesktop{held: true, activeErr: errors.New("no window")}
	err := runDesktop(f, &neverPicker{}, noStartup{}, "", []string{"--home", t.TempDir()})
	if !errors.Is(err, desktop.ErrAlreadyRunning) || f.opened != 0 {
		t.Fatalf("err = %v opened=%d", err, f.opened)
	}
}

// Explicit `desktop` and the no-argument launch are one composition: with no
// home either enters first run inside the resident (tray) window.
func TestExplicitAndNoArgumentLaunchShareOneComposition(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	var windows []desktop.Window
	f := &fakeDesktop{open: func(_ context.Context, w desktop.Window) error { windows = append(windows, w); return nil }}
	if err := runDesktop(f, &neverPicker{}, noStartup{}, "", nil); err != nil {
		t.Fatal(err)
	}
	a, _, _ := testApp(f, func() (home.Discovery, error) { return home.Discovery{Source: home.SourceUnconfigured}, nil })
	if err := a.launch(); err != nil {
		t.Fatal(err)
	}
	if len(windows) != 2 {
		t.Fatalf("%d windows", len(windows))
	}
	for i, w := range windows {
		if w.Resident == nil || w.StartHidden {
			t.Errorf("entry %d: resident=%v hidden=%v; first run must be a visible tray window", i, w.Resident != nil, w.StartHidden)
		}
	}
}

// A configured launch is resident: close hides to the tray, Restart and a
// second activation act on the one controller, and only one runtime is bound.
func TestConfiguredLaunchIsResidentOnOneController(t *testing.T) {
	f := &fakeDesktop{}
	a, rt, opens := installedApp(t, f, false)
	f.open = func(_ context.Context, w desktop.Window) error {
		res := w.Resident
		if res == nil || w.StartHidden {
			t.Fatalf("resident=%v hidden=%v", res != nil, w.StartHidden)
		}
		if got := res.Summary().Label; got != "Ready" {
			t.Errorf("summary %q", got)
		}
		if act := res.OnClose(); act != desktop.ActionHideWithNotice {
			t.Errorf("first close = %v, want hide with notice", act)
		}
		if act := res.OnClose(); act != desktop.ActionHide {
			t.Errorf("close = %v, want hide", act)
		}
		if act := res.OnActivate(); act != desktop.ActionShow {
			t.Errorf("activate = %v", act)
		}
		if _, err := res.OnMenu(desktop.MenuRestart); err != nil {
			t.Error(err)
		}
		res.Wait()
		if act, _ := res.OnMenu(desktop.MenuQuit); act != desktop.ActionQuit {
			t.Errorf("quit = %v", act)
		}
		return nil
	}
	if err := a.launch(); err != nil {
		t.Fatal(err)
	}
	starts, stops, restarts := rt.counts()
	if opens.Load() != 1 || starts != 1 || restarts != 1 || stops != 1 {
		t.Fatalf("runtimes %d starts %d restarts %d stops %d", opens.Load(), starts, restarts, stops)
	}
}

// Start minimized applies only to a healthy configured background launch.
func TestBackgroundStartMinimizedAndVisibleRecovery(t *testing.T) {
	prefs := func(a *desktopApp) {
		if err := (&desktop.Manager{Path: a.PrefsPath, Startup: a.Startup}).SetStartMinimized(true); err != nil {
			t.Fatal(err)
		}
	}
	run := func(a *desktopApp, f *fakeDesktop) desktop.Window {
		var got desktop.Window
		f.open = func(_ context.Context, w desktop.Window) error { got = w; return nil }
		if err := a.launch(); err != nil {
			t.Fatal(err)
		}
		return got
	}

	f := &fakeDesktop{}
	a, _, _ := installedApp(t, f, true)
	prefs(a)
	if w := run(a, f); !w.StartHidden {
		t.Error("healthy background launch with Start minimized must start in the tray")
	}

	f = &fakeDesktop{}
	a, _, _ = installedApp(t, f, false)
	prefs(a)
	if w := run(a, f); w.StartHidden {
		t.Error("an explicit launch always shows the window")
	}

	// The runtime cannot be bound: the window is shown on the diagnostics.
	f = &fakeDesktop{}
	a, _, _ = installedApp(t, f, true)
	prefs(a)
	a.Open = func(string) (app.Runtime, error) { return nil, errors.New("no runtime") }
	w := run(a, f)
	if w.StartHidden || !strings.HasSuffix(w.URL, desktop.DiagnosticsFragment) {
		t.Errorf("failed background start hidden=%v url=%s; recovery must be visible", w.StartHidden, w.URL)
	}
	if len(f.reported) != 0 {
		t.Errorf("a visible recovery window must not also raise a dialog: %v", f.reported)
	}

	// First run is never hidden.
	f = &fakeDesktop{}
	a, _, _ = testApp(f, func() (home.Discovery, error) { return home.Discovery{Source: home.SourceUnconfigured}, nil })
	a.PrefsPath, a.Startup, a.Background = filepath.Join(t.TempDir(), "desktop.json"), noStartup{}, true
	prefs(a)
	if w := run(a, f); w.StartHidden {
		t.Error("first run with Start minimized must be visible")
	}
}

// A background launch that fails before a window exists still tells the user.
func TestBackgroundLaunchFailureBeforeWindowIsReported(t *testing.T) {
	f := &fakeDesktop{}
	a, _, _ := testApp(f, func() (home.Discovery, error) { return home.Discovery{}, nil })
	a.APIAddr = "0.0.0.0:1" // not loopback
	a.Background = true
	if err := a.launch(); err == nil || len(f.reported) != 1 || f.opened != 0 {
		t.Fatalf("err=%v reported=%v opened=%d", err, f.reported, f.opened)
	}
	f = &fakeDesktop{}
	a, _, _ = testApp(f, func() (home.Discovery, error) { return home.Discovery{}, nil })
	a.APIAddr = "0.0.0.0:1"
	if err := a.launch(); err == nil || len(f.reported) != 0 {
		t.Fatalf("interactive failure raised a dialog: %v", f.reported)
	}
}

// First run -> setup -> READY inside the same window and the same controller:
// the tray's Resident observes the setup that the wizard drove, exactly one
// runtime is bound and started, and no second window opens.
func TestFirstRunSetupTransitionsIntoResidentLifecycle(t *testing.T) {
	root := filepath.Join(t.TempDir(), "Hachidori")
	var installed atomic.Bool
	rt, opens := &fakeRuntime{}, &atomic.Int32{}
	var remembered atomic.Int32
	f := &fakeDesktop{}
	a, setups, _ := testApp(f, func() (home.Discovery, error) { return home.Discovery{Source: home.SourceUnconfigured}, nil })
	a.Picker = pickerFunc(func() string { return root })
	a.Env.Load = func(string) (home.Active, error) {
		if !installed.Load() {
			return home.Active{}, errors.New("not installed")
		}
		return home.Active{}, nil
	}
	a.Setup = func(string, string, string, io.Writer, func(setup.Phase)) error {
		setups.Add(1)
		installed.Store(true)
		return nil
	}
	a.Remember = func(r string) (home.Home, error) { remembered.Add(1); return home.Home{Root: r}, nil }
	a.Open = func(string) (app.Runtime, error) { opens.Add(1); return rt, nil }
	a.PrefsPath, a.Startup = filepath.Join(t.TempDir(), "desktop.json"), noStartup{}

	f.open = func(_ context.Context, w desktop.Window) error {
		res := w.Resident
		if res == nil {
			t.Fatal("first-run window is not resident")
		}
		if got := res.Summary().Label; got != "Needs attention" {
			t.Errorf("before setup: %q", got)
		}
		page, err := http.Get(w.URL)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(page.Body)
		page.Body.Close()
		m := regexp.MustCompile(`[0-9a-f]{32}`).FindString(string(b))
		if m == "" {
			t.Fatal("no wizard token in the page")
		}
		post := func(path string, kv ...string) {
			t.Helper()
			form := url.Values{"token": {m}}
			for i := 0; i < len(kv); i += 2 {
				form.Set(kv[i], kv[i+1])
			}
			resp, err := http.PostForm(w.URL+"wizard/"+path, form)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode != 200 {
				t.Fatalf("%s: %d", path, resp.StatusCode)
			}
		}
		post("browse")
		post("install", "device", "cpu")
		deadline := time.Now().Add(10 * time.Second)
		for res.Summary().Label != "Ready" {
			if time.Now().After(deadline) {
				t.Fatalf("never became Ready: %+v", res.Summary())
			}
			time.Sleep(10 * time.Millisecond)
		}
		if _, err := res.OnMenu(desktop.MenuRestart); err != nil {
			t.Error(err)
		}
		res.Wait()
		return nil
	}
	if err := a.launch(); err != nil {
		t.Fatal(err)
	}
	starts, _, restarts := rt.counts()
	if f.opened != 1 || setups.Load() != 1 || remembered.Load() != 1 || opens.Load() != 1 || starts != 1 || restarts != 1 {
		t.Fatalf("windows %d setups %d remembered %d runtimes %d starts %d restarts %d",
			f.opened, setups.Load(), remembered.Load(), opens.Load(), starts, restarts)
	}
}

type pickerFunc func() string

func (p pickerFunc) PickFolder(context.Context, string) (string, error) { return p(), nil }
