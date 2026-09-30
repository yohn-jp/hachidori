package desktop

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestDashboardURL(t *testing.T) {
	for addr, want := range map[string]string{
		"127.0.0.1:7844": "http://127.0.0.1:7844/",
		"[::1]:9000":     "http://[::1]:9000/",
		"localhost:7844": "http://localhost:7844/",
	} {
		got, err := DashboardURL(addr)
		if err != nil || got != want {
			t.Errorf("DashboardURL(%q) = %q, %v; want %q", addr, got, err, want)
		}
	}
	for _, addr := range []string{"0.0.0.0:7844", "192.168.1.2:7844", "example.com:7844", "127.0.0.1", "127.0.0.1:0", ":7844"} {
		if got, err := DashboardURL(addr); err == nil {
			t.Errorf("DashboardURL(%q) = %q, want error", addr, got)
		}
	}
}

func TestNewPolicyRequiresLoopbackHTTP(t *testing.T) {
	for _, u := range []string{
		"https://127.0.0.1:7844/", "http://example.com:7844/", "http://10.0.0.1:7844/",
		"http://127.0.0.1/", "file:///C:/x.html", "http://user@127.0.0.1:7844/", "::",
	} {
		if _, err := NewPolicy(u); err == nil {
			t.Errorf("NewPolicy(%q) accepted", u)
		}
	}
}

func TestPolicyAllowsOnlyTheDashboardOrigin(t *testing.T) {
	p, err := NewPolicy("http://127.0.0.1:7844/")
	if err != nil {
		t.Fatal(err)
	}
	allow := []string{
		"http://127.0.0.1:7844/",
		"http://127.0.0.1:7844/live",
		"http://127.0.0.1:7844/?x=1#frag",
	}
	deny := []string{
		"http://127.0.0.1:7843/",          // the inference API: different origin
		"http://127.0.0.1:7845/",          // other local service
		"http://localhost:7844/",          // same socket, different origin
		"https://127.0.0.1:7844/",         // scheme change
		"http://evil.example/",            // external
		"https://developer.microsoft.com", // external, even Microsoft
		"http://user:pw@127.0.0.1:7844/",  // userinfo tricks
		"http://127.0.0.1:7844@evil.example/",
		"file:///C:/Windows/win.ini",
		"data:text/html,<script>alert(1)</script>",
		"javascript:alert(1)",
		"about:blank",
		"edge://settings",
		"ms-settings:",
		"",
		"%zz",
	}
	for _, u := range allow {
		if !p.AllowNavigation(u) {
			t.Errorf("AllowNavigation(%q) = false, want true", u)
		}
	}
	for _, u := range deny {
		if p.AllowNavigation(u) {
			t.Errorf("AllowNavigation(%q) = true, want false", u)
		}
	}
	for _, u := range append(allow, deny...) {
		if p.AllowNewWindow(u) {
			t.Errorf("AllowNewWindow(%q) = true; popups are never allowed", u)
		}
	}
	if (Policy{}).AllowNavigation("http://127.0.0.1:7844/") {
		t.Error("zero Policy must deny everything")
	}
}

func TestInstanceNameIsPerUserAndUnprivileged(t *testing.T) {
	a := InstanceName("S-1-5-21-1-2-3-1001")
	if a != InstanceName("S-1-5-21-1-2-3-1001") {
		t.Fatal("InstanceName is not deterministic")
	}
	if b := InstanceName("S-1-5-21-1-2-3-1002"); a == b {
		t.Fatal("two users share one instance name")
	}
	if !strings.HasPrefix(a, `Local\`) || strings.HasPrefix(a, `Global\`) {
		t.Fatalf("instance name %q must be in the unprivileged Local namespace", a)
	}
	if strings.Contains(a, "1001") {
		t.Fatalf("instance name %q leaks the raw user id", a)
	}
}

type fakePlatform struct {
	version    string
	versionErr error
	held       bool
	calls      []string
}

func (f *fakePlatform) RuntimeVersion() (string, error) {
	f.calls = append(f.calls, "detect")
	return f.version, f.versionErr
}

func (f *fakePlatform) AcquireInstance() (func(), error) {
	f.calls = append(f.calls, "acquire")
	if f.held {
		return nil, ErrAlreadyRunning
	}
	f.held = true
	return func() { f.held = false }, nil
}

func (f *fakePlatform) Open(context.Context, Window) error { return nil }
func (f *fakePlatform) Activate() error {
	f.calls = append(f.calls, "activate")
	return nil
}
func (f *fakePlatform) ReportError(string, string) {}

func TestPreflightWebView2MissingIsExplicitAndTakesNoGuard(t *testing.T) {
	f := &fakePlatform{}
	_, _, err := Preflight(f)
	if !errors.Is(err, ErrWebView2Missing) {
		t.Fatalf("err = %v, want ErrWebView2Missing", err)
	}
	if f.held || strings.Join(f.calls, ",") != "detect" {
		t.Fatalf("calls = %v held = %v: detection must fail before the guard is taken", f.calls, f.held)
	}
	msg := err.Error()
	for _, want := range []string{"WebView2 Runtime is not installed", "does not download or install",
		"https://developer.microsoft.com/microsoft-edge/webview2/", "hachidori dashboard"} {
		if !strings.Contains(msg, want) {
			t.Errorf("diagnostic %q lacks %q", msg, want)
		}
	}
}

func TestPreflightDetectionError(t *testing.T) {
	boom := errors.New("registry unreadable")
	f := &fakePlatform{versionErr: boom}
	if _, _, err := Preflight(f); !errors.Is(err, boom) || f.held {
		t.Fatalf("err = %v held = %v", err, f.held)
	}
}

func TestPreflightSecondLaunchIsAlreadyRunning(t *testing.T) {
	f := &fakePlatform{version: "129.0.2792.65"}
	v, release, err := Preflight(f)
	if err != nil || v != "129.0.2792.65" {
		t.Fatalf("first launch: %q %v", v, err)
	}
	if _, _, err := Preflight(f); !errors.Is(err, ErrAlreadyRunning) {
		t.Fatalf("second launch: err = %v, want ErrAlreadyRunning", err)
	}
	release()
	_, release, err = Preflight(f)
	if err != nil {
		t.Fatalf("launch after release: %v", err)
	}
	release()
	if want := "detect,acquire,detect,acquire,detect,acquire"; strings.Join(f.calls, ",") != want {
		t.Fatalf("calls = %v, want %s", f.calls, want)
	}
}

func TestNavigationFailureClassification(t *testing.T) {
	if _, ok := NavigationFailure(true, 0); ok {
		t.Error("successful navigation reported as failure")
	}
	// The navigation policy cancels off-origin loads; that is not a failure.
	if _, ok := NavigationFailure(false, webErrorOperationCanceled); ok {
		t.Error("policy-cancelled navigation reported as failure")
	}
	f, ok := NavigationFailure(false, 12) // cannot connect: dashboard down
	if !ok || f.Boundary != BoundaryNavigation || !f.Reloadable || !strings.Contains(f.Detail, "12") {
		t.Fatalf("failure = %+v, %v", f, ok)
	}
}

func TestProcessFailureClassification(t *testing.T) {
	for kind, want := range map[uint32]struct{ reload, fatal bool }{
		0: {false, true}, 1: {true, false}, 2: {true, false}, 3: {true, false}, 6: {true, false},
		4: {false, false}, 5: {false, false}, 9: {false, false},
	} {
		f := ProcessFailure(kind)
		if f.Boundary != BoundaryProcess || f.Reloadable != want.reload || f.Fatal != want.fatal || f.Detail == "" {
			t.Errorf("kind %d: %+v", kind, f)
		}
	}
}

func TestFailureTrackerBoundsReloadsThenNotifiesOnce(t *testing.T) {
	now := time.Unix(1000, 0)
	tr := &FailureTracker{MaxReloads: 2, StableAfter: time.Minute, Now: func() time.Time { return now }}
	render := ProcessFailure(1)
	for i := 0; i < 2; i++ {
		if r := tr.Report(render); !r.Reload || r.Notify {
			t.Fatalf("failure %d: %+v, want reload", i, r)
		}
		now = now.Add(time.Second)
	}
	// Budget exhausted: an actionable native notification, never a blank window.
	r := tr.Report(render)
	if r.Reload || !r.Notify || !strings.Contains(r.Message, "render process") || !strings.Contains(r.Message, "hachidori dashboard") {
		t.Fatalf("exhausted: %+v", r)
	}
	// Repeated failures neither reload forever nor prompt repeatedly.
	if r := tr.Report(render); r.Reload || r.Notify {
		t.Fatalf("after notify: %+v", r)
	}
	if got := len(tr.Recent()); got != 4 {
		t.Fatalf("recorded %d failures, want 4", got)
	}
	// Only a stable interval restores the budget.
	now = now.Add(time.Minute)
	if r := tr.Report(render); !r.Reload {
		t.Fatalf("after stable interval: %+v", r)
	}
}

func TestFailureTrackerFatalAndBenign(t *testing.T) {
	tr := &FailureTracker{}
	if r := tr.Report(ProcessFailure(0)); r.Reload || !r.Notify {
		t.Fatalf("browser process exit: %+v", r)
	}
	if r := tr.Report(ProcessFailure(4)); r.Reload || r.Notify {
		t.Fatalf("helper process failure: %+v", r)
	}
	tr = &FailureTracker{}
	if r := tr.Report(ControllerFailure(errors.New("error creating controller with 80070005"))); !r.Notify || !strings.Contains(r.Message, "80070005") {
		t.Fatalf("controller failure: %+v", r)
	}
	for i := 0; i < 100; i++ {
		tr.Report(ProcessFailure(4))
	}
	if len(tr.Recent()) != maxRecentFailures {
		t.Fatalf("recent failures unbounded: %d", len(tr.Recent()))
	}
}
