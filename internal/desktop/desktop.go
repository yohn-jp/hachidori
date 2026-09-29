// Package desktop is the thin Windows desktop shell: a native top-level window
// hosting Microsoft WebView2 that shows the existing host-local dashboard.
//
// The shell owns only the window's lifetime. It has no runtime, lifecycle,
// status or doctor logic of its own: the caller runs the same composition as
// `hachidori dashboard` (inference API, resident worker, dashboard) and the
// window merely navigates to the dashboard's loopback URL.
//
// Everything here except the Native platform is platform neutral and tested
// on every OS. The WebView2 binding and Win32 calls live in *_windows.go
// files, so non-Windows builds never import Windows UI dependencies.
package desktop

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
)

// ErrUnsupported is returned by the Native platform on non-Windows systems.
var ErrUnsupported = errors.New("hachidori desktop is only available on Windows; " +
	"on this system run 'hachidori dashboard' and open the printed http://127.0.0.1:7844/ URL in a browser")

// ErrAlreadyRunning is returned when this user already has a desktop shell
// running. The second launch activates that shell's window (Platform.Activate)
// and exits without starting any runtime component.
var ErrAlreadyRunning = errors.New("hachidori desktop is already running for this user; " +
	"switch to its window (only one desktop shell may own the local runtime)")

// ErrWebView2Missing is returned before any window or runtime is created when
// the Microsoft Edge WebView2 Runtime is not installed. Hachidori never
// downloads or installs it.
var ErrWebView2Missing = errors.New("the Microsoft Edge WebView2 Runtime is not installed. " +
	"It is a Windows prerequisite that Hachidori does not download or install: install the " +
	"Evergreen WebView2 Runtime from https://developer.microsoft.com/microsoft-edge/webview2/ " +
	"(or via your organization's software deployment), then run 'hachidori desktop' again. " +
	"Without it, 'hachidori dashboard' serves the same UI to any local browser")

// Window describes the one window the shell opens.
type Window struct {
	Title   string
	URL     string // initial navigation: the loopback dashboard
	DataDir string // WebView2 user data folder (browser profile), under HACHIDORI_HOME
	Policy  Policy // top-level navigation allow-list

	// Resident, when set, makes the window resident: a tray icon is shown,
	// closing the window hides it to the tray, and the window ends only when
	// the user chooses Quit or ctx is done. When nil the window has no tray
	// and closing it ends the session.
	Resident *Resident
	// StartHidden starts in the tray without showing the window (only
	// meaningful with Resident).
	StartHidden bool
}

// Platform is the OS surface the shell needs. Native returns the real one;
// tests substitute fakes so no test needs WebView2, a GPU or a display.
type Platform interface {
	// RuntimeVersion reports the installed WebView2 Runtime version, or ""
	// when none is installed.
	RuntimeVersion() (string, error)
	// AcquireInstance takes the per-user single-instance guard. It returns
	// ErrAlreadyRunning when another shell of the same user holds it.
	AcquireInstance() (release func(), err error)
	// Open creates the native window, navigates it to w.URL and blocks until
	// the window is closed or ctx is done (which closes the window). All
	// window and WebView2 resources are released before it returns.
	Open(ctx context.Context, w Window) error
	// Activate asks the desktop shell already running for this user to show
	// and focus its window. It is what a second launch does instead of
	// starting another runtime owner.
	Activate() error
	// ReportError tells the user about a fatal startup failure when no
	// console may be visible (for example a sign-in launch).
	ReportError(title, message string)
}

// Preflight runs, in order, everything that must succeed before the runtime
// is started or a window is created: WebView2 Runtime detection, then the
// single-instance guard. On success the caller owns release and must call it
// when the shell exits.
func Preflight(p Platform) (version string, release func(), err error) {
	version, err = p.RuntimeVersion()
	if err != nil {
		return "", nil, fmt.Errorf("detecting the WebView2 Runtime: %w", err)
	}
	if version == "" {
		return "", nil, ErrWebView2Missing
	}
	release, err = p.AcquireInstance()
	if err != nil {
		return "", nil, err
	}
	return version, release, nil
}

// DashboardURL is the root URL of a dashboard bound to addr. addr must be a
// loopback host:port with an explicit port.
func DashboardURL(addr string) (string, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "", err
	}
	if !loopback(host) {
		return "", fmt.Errorf("dashboard address %s is not loopback", addr)
	}
	if port == "" || port == "0" {
		return "", fmt.Errorf("dashboard address %s needs an explicit port", addr)
	}
	return "http://" + net.JoinHostPort(host, port) + "/", nil
}

// Policy decides which top-level navigations the WebView may perform.
// Exactly one origin is allowed: the loopback dashboard the shell opened.
// Everything else (other hosts, other ports, other schemes such as file:,
// data:, javascript: or about:) is cancelled. New-window requests (popups,
// target=_blank) are always refused; the shell never opens a second window.
type Policy struct {
	origin string // "http://127.0.0.1:7844"
}

// NewPolicy builds the policy for the dashboard at dashboardURL, which must be
// an http URL on a loopback host with an explicit port.
func NewPolicy(dashboardURL string) (Policy, error) {
	u, err := url.Parse(dashboardURL)
	if err != nil {
		return Policy{}, err
	}
	if u.Scheme != "http" || u.User != nil || u.Port() == "" || !loopback(u.Hostname()) {
		return Policy{}, fmt.Errorf("desktop window may only show a loopback http dashboard with an explicit port, not %q", dashboardURL)
	}
	return Policy{origin: "http://" + strings.ToLower(u.Host)}, nil
}

// AllowNavigation reports whether a top-level navigation to uri stays on the
// dashboard origin.
func (p Policy) AllowNavigation(uri string) bool {
	if p.origin == "" {
		return false
	}
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "http" || u.User != nil || u.Opaque != "" {
		return false
	}
	return "http://"+strings.ToLower(u.Host) == p.origin
}

// AllowNewWindow reports whether a popup / new-window request may create a
// window. It never may: the shell owns exactly one window and does not hand
// URLs from page content to other applications.
func (p Policy) AllowNewWindow(string) bool { return false }

func loopback(host string) bool {
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
