package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
)

// errActivated is returned by the shell hook when another desktop shell of
// this user was already running and was asked to show its window instead.
var errActivated = errors.New("activated the running desktop instance")

// shellSession is what the desktop hook hands runHost after its preflight.
type shellSession struct {
	// Release drops the single-instance guard when the process ends.
	Release func()
	// Prefs backs the dashboard's desktop preferences panel.
	Prefs dashboard.Desktop
	// Attach opens the resident window on the dashboard and blocks until the
	// user quits (or ctx ends). ctrl is the application controller that owns
	// the runtime; the tray only reads its state and requests Restart.
	Attach func(ctx context.Context, dashURL string, ctrl *app.Controller) error
}

// shellHook lets the desktop command run its preflight after flags and home
// are resolved but before any runtime component starts, then attach a window
// once the dashboard is listening.
type shellHook func(h home.Home, src home.Source, background bool) (*shellSession, error)

// cmdDesktop is the Windows desktop shell. After WebView2 detection and the
// per-user single-instance guard it runs the dashboard composition (runHost)
// with the application controller as the one runtime owner, and attaches one
// resident window with a tray icon. Closing the window hides it to the tray;
// Quit ends the process the same way Ctrl+C ends `hachidori dashboard`.
func cmdDesktop(p desktop.Platform, args []string) error {
	if runtime.GOOS != "windows" {
		return desktop.ErrUnsupported
	}
	prefsPath, err := desktop.PrefsPath()
	if err != nil {
		return err
	}
	return runDesktop(p, desktop.NativeStartup(), prefsPath, args)
}

// runDesktop is cmdDesktop with its OS surfaces injected (tests use fakes).
func runDesktop(p desktop.Platform, st desktop.Startup, prefsPath string, args []string) error {
	background := false
	err := runHost("desktop", args, func(h home.Home, src home.Source, bg bool) (*shellSession, error) {
		background = bg
		return desktopSession(p, st, prefsPath, h, src, bg)
	})
	switch {
	case errors.Is(err, errActivated):
		fmt.Fprintln(os.Stderr, "hachidori: already running for this user; activated the existing window")
		return nil
	case err != nil && background:
		// A sign-in launch has no visible console: say why nothing appeared.
		p.ReportError("Hachidori could not start", err.Error())
	}
	return err
}

func desktopSession(p desktop.Platform, st desktop.Startup, prefsPath string, h home.Home, src home.Source, background bool) (*shellSession, error) {
	version, release, err := desktop.Preflight(p)
	if errors.Is(err, desktop.ErrAlreadyRunning) {
		// Duplicate launch: activate the running shell instead of starting a
		// second runtime owner. Nothing else has been started yet.
		if aerr := p.Activate(); aerr != nil {
			return nil, fmt.Errorf("%w (and it could not be activated: %v)", err, aerr)
		}
		return nil, errActivated
	}
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "hachidori: WebView2 Runtime %s\n", version)

	// The startup entry runs this same executable in background mode. A home
	// that the desktop cannot rediscover on its own is passed explicitly.
	homeArg := h.Root
	if src == home.SourceLocator {
		homeArg = ""
	}
	exe, err := os.Executable()
	if err != nil {
		release()
		return nil, fmt.Errorf("locating the executable: %w", err)
	}
	cmdline, err := desktop.StartupCommand(exe, homeArg)
	if err != nil {
		release()
		return nil, err
	}
	mgr := &desktop.Manager{Path: prefsPath, Startup: st, Command: cmdline}

	attach := func(ctx context.Context, dashURL string, ctrl *app.Controller) error {
		pol, err := desktop.NewPolicy(dashURL)
		if err != nil {
			return err
		}
		res := &desktop.Resident{App: ctrl, Prefs: mgr}
		defer res.Wait()
		return p.Open(ctx, desktop.Window{Title: "Hachidori", URL: dashURL,
			DataDir: h.Path("cache", "webview2"), Policy: pol,
			Resident: res, StartHidden: desktop.StartHidden(background, res.Preferences())})
	}
	return &shellSession{Release: release, Prefs: mgr, Attach: attach}, nil
}
