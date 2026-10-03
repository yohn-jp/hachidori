package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
)

// The settings authority composes the existing desktop preferences (an existing
// desktop.json keeps working) and stores runtime defaults beside it without
// rewriting desktop.json.
func TestSettingsStoreComposesDesktopPrefs(t *testing.T) {
	dir := t.TempDir()
	prefs := filepath.Join(dir, "desktop.json")
	legacy := `{"schema":"hachidori.desktop/1","start_minimized":true,"close_notice_shown":true}`
	if err := os.WriteFile(prefs, []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}
	st := settingsStore(prefs, &desktop.Manager{Path: prefs})
	if _, min, err := st.Prefs(); err != nil || !min {
		t.Fatalf("start minimized %v %v", min, err)
	}
	if err := st.SetDefaults(settings.Defaults{Device: "cpu", Model: setup.DefaultModel}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "settings.json")); err != nil {
		t.Fatalf("settings.json not beside desktop.json: %v", err)
	}
	if b, _ := os.ReadFile(prefs); string(b) != legacy {
		t.Fatalf("saving defaults rewrote desktop.json: %s", b)
	}
}

// The desktop hosts Development Connection profiles in the same settings
// store, beside desktop.json, and they survive a restart of the composition.
func TestSettingsStoreHostsDevelopmentConnections(t *testing.T) {
	dir := t.TempDir()
	prefs := filepath.Join(dir, "desktop.json")
	var _ dashboard.Connections = settingsStore(prefs, nil)
	c := settings.Connection{Name: "nixos-dev", Destination: "dev@nixos", RemoteBind: "127.0.0.1", RemoteBindMode: settings.ConnectionPinned, RemotePort: 7843, RemotePortMode: settings.ConnectionPinned, LocalPort: 7843, LocalPortMode: settings.ConnectionPinned}
	if err := settingsStore(prefs, nil).SaveConnection(c); err != nil {
		t.Fatal(err)
	}
	got, err := settingsStore(prefs, nil).Connections()
	if err != nil || len(got) != 1 || got[0].Name != c.Name || got[0].Destination != c.Destination || got[0].RemoteBind != c.RemoteBind || got[0].RemoteBindMode != c.RemoteBindMode || got[0].RemotePort != c.RemotePort || got[0].RemotePortMode != c.RemotePortMode || got[0].LocalPort != c.LocalPort || got[0].LocalPortMode != c.LocalPortMode {
		t.Fatalf("after restart: %+v %v", got, err)
	}
	if _, err := os.Stat(prefs); !os.IsNotExist(err) {
		t.Fatalf("saving a profile wrote desktop.json: %v", err)
	}
}

func TestSettingsStoreResolvesAutoFromBoundDesktopAPIEndpoint(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	endpoint, err := tunnel.LocalEndpointFromAddr(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	store := settingsStore(filepath.Join(t.TempDir(), "desktop.json"), nil)
	if err := store.SetLocalEndpoint(endpoint); err != nil {
		t.Fatal(err)
	}
	auto := settings.Connection{
		Name: "auto-dev", Destination: "devhost",
		RemoteBindMode: settings.ConnectionAuto, RemotePortMode: settings.ConnectionAuto, LocalPortMode: settings.ConnectionAuto,
	}
	if err := store.SaveConnection(auto); err != nil {
		t.Fatal(err)
	}
	profiles, err := store.Connections()
	if err != nil || len(profiles) != 1 {
		t.Fatalf("saved connection: %+v %v", profiles, err)
	}
	got := profiles[0].Spec()
	want := tunnel.Spec{Destination: "devhost", RemoteBind: "127.0.0.1", RemotePort: endpoint.Port, LocalPort: endpoint.Port}
	if got != want {
		t.Fatalf("managed API endpoint %q resolved to %+v, want %+v", ln.Addr(), got, want)
	}
	if profiles[0].RemotePort != 0 || profiles[0].LocalPort != 0 || profiles[0].RemotePortMode != settings.ConnectionAuto || profiles[0].LocalPortMode != settings.ConnectionAuto {
		t.Fatalf("resolved values replaced stored intent: %+v", profiles[0])
	}
}

// Proof 11 and 12: explicit commands stay CLI commands and the no-argument
// desktop is reachable only through the platform hook.
func TestRunDispatch(t *testing.T) {
	var called atomic.Int32
	noArg := func() error { called.Add(1); return nil }

	if code := run(nil, noArg); code != 0 || called.Load() != 1 {
		t.Fatalf("no args with a desktop hook: code %d, hook calls %d", code, called.Load())
	}
	called.Store(0)
	if code := run(nil, func() error { called.Add(1); return errors.New("boom") }); code != 1 || called.Load() != 1 {
		t.Fatalf("desktop failure: code %d", code)
	}
	called.Store(0)
	if code := run(nil, nil); code != 2 {
		t.Fatalf("no args and no hook must print usage (2), got %d", code)
	}
	for _, args := range [][]string{
		{"bogus"},
		{"status", "-endpoint", "http://127.0.0.1:1"}, // a real CLI command that fails fast
	} {
		if code := run(args, noArg); code == 0 {
			t.Fatalf("%v succeeded", args)
		}
	}
	if called.Load() != 0 {
		t.Fatalf("the desktop hook ran for explicit CLI commands (%d)", called.Load())
	}
}

type fakePlatform struct {
	release atomic.Int32
	open    func(ctx context.Context, w desktop.Window) error
	version string
	opened  atomic.Int32
}

func (p *fakePlatform) RuntimeVersion() (string, error) { return p.version, nil }
func (p *fakePlatform) AcquireInstance() (func(), error) {
	return func() { p.release.Add(1) }, nil
}
func (p *fakePlatform) Activate() error            { return nil }
func (p *fakePlatform) ReportError(string, string) {}
func (p *fakePlatform) Open(ctx context.Context, w desktop.Window) error {
	p.opened.Add(1)
	return p.open(ctx, w)
}

type neverPicker struct{ calls atomic.Int32 }

func (p *neverPicker) PickFolder(context.Context, string) (string, error) {
	p.calls.Add(1)
	return "", desktop.ErrPickCancelled
}

func get(t *testing.T, url string) (int, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func testApp(p desktop.Platform, d func() (home.Discovery, error)) (*desktopApp, *atomic.Int32, *atomic.Int32) {
	var setups, remembers atomic.Int32
	return &desktopApp{
		Platform: p, Picker: &neverPicker{}, Discover: d,
		Remember: func(string) (home.Home, error) { remembers.Add(1); return home.Home{}, errors.New("unexpected") },
		APIAddr:  "127.0.0.1:0", DashAddr: "127.0.0.1:0",
		Setup:  func(string, string, string, io.Writer, *setup.Observer) error { setups.Add(1); return nil },
		Stderr: &bytes.Buffer{},
	}, &setups, &remembers
}

// Proof 1: no bootstrap -> a native window with the first-run wizard, before
// any runtime, home, setup or bootstrap write exists.
func TestNoBootstrapOpensFirstRunWindow(t *testing.T) {
	var dataDir, url string
	p := &fakePlatform{version: "130.0"}
	p.open = func(ctx context.Context, w desktop.Window) error {
		dataDir, url = w.DataDir, w.URL
		if !w.Policy.AllowNavigation(w.URL) || w.Policy.AllowNavigation("https://example.com/") {
			t.Error("window policy must allow only the local origin")
		}
		if _, err := os.Stat(w.DataDir); err != nil {
			os.MkdirAll(w.DataDir, 0o755) // the real window creates its profile
		}
		code, body := get(t, w.URL)
		if code != 200 || !strings.Contains(body, "Where should Hachidori keep its models and runtime?") {
			t.Errorf("first-run page %d", code)
		}
		code, body = get(t, w.URL+"wizard/state")
		if code != 200 || !strings.Contains(body, `"mode":"first_run"`) || !strings.Contains(body, `"stage":"select"`) || !strings.Contains(body, `"state":"unconfigured"`) {
			t.Errorf("state %d %s", code, body)
		}
		return nil
	}
	a, setups, remembers := testApp(p, func() (home.Discovery, error) { return home.Discovery{Source: home.SourceUnconfigured}, nil })
	if err := a.run(); err != nil {
		t.Fatal(err)
	}
	if p.opened.Load() != 1 || p.release.Load() != 1 || setups.Load() != 0 || remembers.Load() != 0 {
		t.Fatalf("opened %d released %d setups %d remembers %d", p.opened.Load(), p.release.Load(), setups.Load(), remembers.Load())
	}
	if !strings.HasPrefix(url, "http://127.0.0.1:") {
		t.Fatalf("url %q", url)
	}
	// With no home yet, the window profile is throwaway and removed on exit.
	if _, err := os.Stat(dataDir); !os.IsNotExist(err) {
		t.Fatalf("temporary window profile %s not removed: %v", dataDir, err)
	}
	if !strings.HasPrefix(dataDir, os.TempDir()) {
		t.Fatalf("data dir %q", dataDir)
	}
}

// Proof 3: a stored home that vanished shows recovery, never a fresh install.
func TestMissingStoredHomeShowsRecovery(t *testing.T) {
	loc := home.Locator{Path: filepath.Join(t.TempDir(), "bootstrap.json")}
	gone := filepath.Join(t.TempDir(), "gone")
	if err := os.Mkdir(gone, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := loc.Save(gone); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	p := &fakePlatform{version: "130.0"}
	p.open = func(ctx context.Context, w desktop.Window) error {
		_, body := get(t, w.URL+"wizard/state")
		if !strings.Contains(body, `"mode":"recovery_missing"`) || !strings.Contains(body, `"stage":"select"`) || !strings.Contains(body, "missing or unavailable") {
			t.Errorf("recovery state %s", body)
		}
		return nil
	}
	a, setups, remembers := testApp(p, func() (home.Discovery, error) {
		h, _, err := loc.Lookup()
		return home.Discovery{Home: h, Source: home.SourceLocator}, err
	})
	if err := a.run(); err != nil {
		t.Fatal(err)
	}
	if setups.Load() != 0 || remembers.Load() != 0 {
		t.Fatal("recovery installed or rewrote the locator on its own")
	}
	if _, err := os.Stat(gone); !os.IsNotExist(err) {
		t.Fatal("the missing home was recreated")
	}
	b, found, err := loc.Load()
	if err != nil || !found || b.Home != gone {
		t.Fatalf("locator %+v %v %v", b, found, err)
	}
}

// Prerequisite failures happen before anything starts.
func TestPreflightFailureStartsNothing(t *testing.T) {
	p := &fakePlatform{version: ""} // WebView2 missing
	p.open = func(context.Context, desktop.Window) error { t.Error("window opened"); return nil }
	a, _, _ := testApp(p, func() (home.Discovery, error) { t.Error("discovery ran"); return home.Discovery{}, nil })
	if err := a.run(); !errors.Is(err, desktop.ErrWebView2Missing) {
		t.Fatalf("err %v", err)
	}
}

var _ dashboard.Models = modelManager{}

// The dashboard's manager is the application controller over the real setup
// authority: the inventory lists only catalog identities, an activation
// whose artifacts are not materialized is refused without writing anything,
// and removal cannot escape HACHIDORI_HOME or name a non-catalog artifact.
func TestModelManagerOverRealSetupAuthority(t *testing.T) {
	root := t.TempDir()
	h := home.Home{Root: filepath.Join(root, "home")}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	ctl := app.New(app.Config{Home: h.Root})
	m := modelManager{ctl: func() *app.Controller { return ctl }}

	st := m.State()
	if st.Err != "" || len(st.Inventory.Runtimes) != len(setup.Devices) || len(st.Inventory.Models) != len(setup.Models) || st.RestartRequired {
		t.Fatalf("state %+v", st)
	}
	for _, r := range st.Inventory.Runtimes {
		if r.Materialized || r.Active {
			t.Fatalf("empty home reports %+v", r)
		}
	}
	if err := m.Activate("cuda", setup.DefaultModel); err != nil {
		t.Fatalf("activation was not accepted: %v", err)
	}
	waitModelsIdle(t, m)
	if _, err := os.Stat(h.Path("state", "active-runtime.json")); !os.IsNotExist(err) {
		t.Fatal("failed activation wrote an activation record")
	}
	if got := m.State().Last; got == nil || got.Kind != app.OpActivate || got.Failure == "" {
		t.Fatalf("failure not reported: %+v", got)
	}
	for _, c := range [][2]string{{"runtime", "../../outside"}, {"runtime", outside}, {"model", "../outside"}, {"model", ""}, {"runtime", "cuda-notcatalog"}} {
		if err := m.Remove(c[0], c[1]); err != nil {
			t.Fatalf("Remove(%v) was not accepted: %v", c, err)
		}
		waitModelsIdle(t, m)
		if got := m.State().Last; got == nil || got.Kind != app.OpRemove || got.Failure == "" {
			t.Fatalf("Remove(%v) did not fail: %+v", c, got)
		}
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatal("directory outside the home removed")
	}
	// Without a home nothing is inspected.
	if st := (modelManager{ctl: func() *app.Controller { return app.New(app.Config{}) }}).State(); st.Err == "" {
		t.Fatal("no home is not reported")
	}
}

func TestModelManagerStartsAndProjectsDesiredStateOperation(t *testing.T) {
	root := t.TempDir()
	h := home.Home{Root: filepath.Join(root, "home")}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	prefs := settingsStore(filepath.Join(root, "desktop.json"), nil)
	ctl := app.New(app.Config{Home: h.Root, Residents: func() []string { ids, _ := prefs.Residents(); return ids }, SetResidents: prefs.SetResidents})
	m := modelManager{ctl: func() *app.Controller { return ctl }}
	intent := dashboard.DesiredStateRequest{Model: setup.DefaultModel, DeviceMode: "pinned", Device: "cpu", Residents: []string{"opendecider-nano"}}
	if err := m.StartDesiredState(intent); err != nil {
		t.Fatalf("desired-state operation was not accepted: %v", err)
	}
	waitModelsIdle(t, m)
	op := m.State().Last
	if op == nil || op.Kind != app.OpDesiredState || op.Model != setup.DefaultModel || op.DeviceMode != "pinned" || op.RequestedDevice != "cpu" || op.ResolvedDevice != "cpu" ||
		len(op.Residents) != 1 || op.Residents[0] != "opendecider-nano" {
		t.Fatalf("desired-state operation was not projected: %+v", op)
	}
}

// waitModelsIdle waits for the maintenance action in flight to finish: the
// actions are accepted at once and run in the background.
func waitModelsIdle(t *testing.T, m modelManager) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for m.State().Busy != nil {
		if time.Now().After(deadline) {
			t.Fatal("maintenance action did not finish")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

type cancelPathPicker struct{ neverPicker }

func (*cancelPathPicker) PickOpen(context.Context, string) (string, error) {
	return "", desktop.ErrPickCancelled
}
func (*cancelPathPicker) PickSave(context.Context, string) (string, error) {
	return "", errors.New("dialog broke")
}

func TestDashboardPathPickerIsDesktopFileCapabilityOnly(t *testing.T) {
	if p := dashboardPathPicker(&neverPicker{}); p != nil {
		t.Fatal("a folder-only picker became a dashboard path picker")
	}
	if p := dashboardPathPicker(desktop.NativePicker()); p != nil && runtime.GOOS != "windows" {
		t.Fatal("a platform without native file dialogs exposed a path picker")
	}
	p := dashboardPathPicker(&cancelPathPicker{})
	if p == nil {
		t.Fatal("a file-capable picker was not exposed")
	}
	if _, err := p.PickOpen(context.Background(), ""); !errors.Is(err, dashboard.ErrPickCancelled) {
		t.Fatalf("cancel = %v, want dashboard.ErrPickCancelled", err)
	}
	if _, err := p.PickFolder(context.Background(), ""); !errors.Is(err, dashboard.ErrPickCancelled) {
		t.Fatalf("folder cancel = %v", err)
	}
	if _, err := p.PickSave(context.Background(), ""); err == nil || errors.Is(err, dashboard.ErrPickCancelled) {
		t.Fatalf("failure = %v, want the picker error", err)
	}
}

// The operator UI locale is stored by the same settings authority beside
// desktop.json, exists before any home is selected (first run resolves it
// there too), and survives a restart of the composition.
func TestSettingsStorePersistsOperatorLocale(t *testing.T) {
	t.Setenv("LC_ALL", "")
	t.Setenv("LC_MESSAGES", "")
	t.Setenv("LANG", "C.UTF-8")
	prefs := filepath.Join(t.TempDir(), "desktop.json")
	var _ dashboard.Settings = settingsStore(prefs, nil)
	if got := settingsStore(prefs, nil).ResolvedLocale(); got != i18n.English {
		t.Fatalf("unset locale resolves to %q", got)
	}
	if err := settingsStore(prefs, nil).SetLocale("ja"); err != nil {
		t.Fatal(err)
	}
	if got := settingsStore(prefs, nil).ResolvedLocale(); got != i18n.Japanese {
		t.Fatalf("after restart: %q", got)
	}
}
