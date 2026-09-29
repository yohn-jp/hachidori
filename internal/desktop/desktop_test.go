package desktop

import (
	"context"
	"errors"
	"strings"
	"testing"
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
