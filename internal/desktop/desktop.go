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
	"io"
	"net"
	"net/url"
	"strings"
	"sync"
	"time"
)

// Epoch is the origin of the startup marks: the instant the executable's
// main function was entered, taken there by the caller. It is not the OS
// process creation time (Go does not provide one); loader and runtime
// initialization before main are outside the measurement.
type Epoch time.Time

// Mark logs one startup milestone to w as the time since e, so the startup
// budgets in docs/desktop.md can be read from a physical Windows launch
// without a profiler. A zero epoch (no entry time supplied) logs nothing.
func (e Epoch) Mark(w io.Writer, stage string) {
	if time.Time(e).IsZero() {
		return
	}
	fmt.Fprintf(w, "hachidori: startup %s +%dms since entry\n", stage, time.Since(time.Time(e)).Milliseconds())
}

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
	// Epoch is the executable entry time the shell's startup marks are
	// measured from.
	Epoch Epoch
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

// WebViewBoundary names where a WebView2 failure was observed after the
// environment-created boundary.
type WebViewBoundary string

const (
	// BoundaryController: environment/controller creation or configuration.
	BoundaryController WebViewBoundary = "controller"
	// BoundaryNavigation: a top-level navigation completed unsuccessfully.
	BoundaryNavigation WebViewBoundary = "navigation"
	// BoundaryProcess: a WebView2 browser, renderer or GPU process failed.
	BoundaryProcess WebViewBoundary = "process"
)

// WebViewFailure is one observed WebView2 failure. Reloadable failures may be
// retried by navigating the same loopback URL again; Fatal ones cannot be
// repaired by the shell. A failure that is neither is only recorded.
type WebViewFailure struct {
	Boundary   WebViewBoundary
	Detail     string
	Reloadable bool
	Fatal      bool
}

// webErrorOperationCanceled is COREWEBVIEW2_WEB_ERROR_STATUS_OPERATION_CANCELED:
// the navigation policy (or a superseding navigation) cancelled the load. That
// is a decision, not a failure.
const webErrorOperationCanceled = 14

// NavigationFailure classifies a NavigationCompleted event. It reports false
// for a successful load and for a deliberately cancelled one.
func NavigationFailure(success bool, webErrorStatus int32) (WebViewFailure, bool) {
	if success || webErrorStatus == webErrorOperationCanceled {
		return WebViewFailure{}, false
	}
	return WebViewFailure{Boundary: BoundaryNavigation, Reloadable: true,
		Detail: fmt.Sprintf("navigation failed (WebView2 web error status %d)", webErrorStatus)}, true
}

// ProcessFailure classifies a ProcessFailed event by its
// COREWEBVIEW2_PROCESS_FAILED_KIND. Renderer, frame-renderer and GPU process
// failures are repaired by reloading; a browser process exit needs a new
// WebView2 environment, which only restarting the shell provides; helper
// process exits are recorded without user-visible action.
func ProcessFailure(kind uint32) WebViewFailure {
	f := WebViewFailure{Boundary: BoundaryProcess}
	switch kind {
	case 0:
		f.Detail, f.Fatal = "the WebView2 browser process exited", true
	case 1:
		f.Detail, f.Reloadable = "the WebView2 render process exited", true
	case 2:
		f.Detail, f.Reloadable = "the WebView2 render process is unresponsive", true
	case 3:
		f.Detail, f.Reloadable = "a WebView2 frame render process exited", true
	case 6:
		f.Detail, f.Reloadable = "the WebView2 GPU process exited", true
	default:
		f.Detail = fmt.Sprintf("a WebView2 helper process failed (kind %d)", kind)
	}
	return f
}

// ControllerFailure is a fatal failure creating or configuring the WebView2
// controller.
func ControllerFailure(err error) WebViewFailure {
	return WebViewFailure{Boundary: BoundaryController, Fatal: true, Detail: err.Error()}
}

// FailureResponse is what the native shell must do about a reported failure.
type FailureResponse struct {
	Reload  bool   // navigate the window to its dashboard URL again
	Notify  bool   // tell the user through a native surface, not the page
	Message string // actionable text for Notify
}

// FailureTracker turns WebView2 failures into bounded, deterministic actions
// so a renderer or navigation failure is never only a blank window and never
// an endless reload loop: at most MaxReloads reloads, then one notification.
// The budget resets only after StableAfter passes without a failure.
type FailureTracker struct {
	MaxReloads  int           // 0 selects 3
	StableAfter time.Duration // 0 selects 2 minutes
	Now         func() time.Time

	mu       sync.Mutex
	reloads  int
	last     time.Time
	notified bool
	recent   []WebViewFailure
}

const maxRecentFailures = 16

// Report records f and decides the response.
func (t *FailureTracker) Report(f WebViewFailure) FailureResponse {
	t.mu.Lock()
	defer t.mu.Unlock()
	limit, stable, now := t.MaxReloads, t.StableAfter, time.Now
	if limit <= 0 {
		limit = 3
	}
	if stable <= 0 {
		stable = 2 * time.Minute
	}
	if t.Now != nil {
		now = t.Now
	}
	at := now()
	if !t.last.IsZero() && at.Sub(t.last) >= stable {
		t.reloads, t.notified = 0, false
	}
	t.last = at
	if t.recent = append(t.recent, f); len(t.recent) > maxRecentFailures {
		t.recent = t.recent[len(t.recent)-maxRecentFailures:]
	}
	switch {
	case f.Reloadable && t.reloads < limit:
		t.reloads++
		return FailureResponse{Reload: true}
	case f.Fatal || f.Reloadable:
		if t.notified {
			return FailureResponse{}
		}
		t.notified = true
		return FailureResponse{Notify: true, Message: "Hachidori cannot display its window: " + f.Detail + ". " +
			"The runtime is not affected. Quit Hachidori from the tray icon and start it again, " +
			"or run 'hachidori dashboard' and open the printed loopback URL in a browser."}
	}
	return FailureResponse{}
}

// Recent returns the most recent recorded failures, oldest first.
func (t *FailureTracker) Recent() []WebViewFailure {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]WebViewFailure(nil), t.recent...)
}
