package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

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
		Setup:  func(string, string, string, io.Writer, func(setup.Phase)) error { setups.Add(1); return nil },
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
