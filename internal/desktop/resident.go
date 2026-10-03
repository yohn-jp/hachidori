package desktop

import (
	"errors"
	"sync"

	"github.com/yohn-jp/hachidori/internal/app"
)

// Level is the concise tray severity of the application state.
type Level int

const (
	// LevelOK: the runtime is ready.
	LevelOK Level = iota
	// LevelBusy: something is in progress (starting, warming, setting up,
	// stopping), the runtime is deliberately stopped, or serving is paused
	// while model engineering uses the accelerator.
	LevelBusy
	// LevelAttention: the user has something to fix or read: a failure, no
	// home selected, or no installed runtime.
	LevelAttention
)

// Summary is the concise state shown by the tray. It is derived only from an
// app.Snapshot (the application controller's canonical state); the tray has
// no status calculation of its own.
type Summary struct {
	Label string // "Ready", "Starting", "Needs attention", ...
	Level Level
	State app.State
}

// Summarize maps the canonical application state to the tray summary. The
// mapping is total over app.State; an unknown future state is reported as
// needing attention rather than as ready.
func Summarize(s app.Snapshot) Summary {
	sum := Summary{State: s.State}
	switch s.State {
	case app.Ready:
		sum.Label, sum.Level = "Ready", LevelOK
	case app.Starting, app.Warming:
		sum.Label, sum.Level = "Starting", LevelBusy
	case app.Installing:
		sum.Label, sum.Level = "Setting up", LevelBusy
	case app.Stopping:
		sum.Label, sum.Level = "Stopping", LevelBusy
	case app.Installed:
		sum.Label, sum.Level = "Stopped", LevelBusy
	case app.Paused:
		// Serving is intentionally down while Forge or Tuning uses the GPU; the
		// controller brings it back by itself.
		sum.Label, sum.Level = "Paused", LevelBusy
	default: // Failed, Unconfigured, NotInstalled and any unknown state
		sum.Label, sum.Level = "Needs attention", LevelAttention
	}
	return sum
}

// Tooltip is the tray icon tooltip for a summary.
func (s Summary) Tooltip() string { return "Hachidori: " + s.Label }

// DiagnosticsFragment is the fragment of the dashboard's diagnostics section.
const DiagnosticsFragment = "#diagnostics"

// OpenURL is where the window should be when it is opened from the tray:
// the dashboard root, or its diagnostics section when the application needs
// attention (so a failed background start is never hidden from the user).
func OpenURL(dashboardURL string, s Summary) string {
	if s.Level == LevelAttention {
		return dashboardURL + DiagnosticsFragment
	}
	return dashboardURL
}

// MenuID identifies a tray menu entry.
type MenuID int

const (
	MenuStatus MenuID = iota + 1 // read-only state line
	MenuOpen
	MenuRestart
	MenuDiagnostics
	MenuStartAtSignIn
	MenuStartMinimized
	MenuQuit
)

// MenuItem is one tray menu entry. A MenuItem with Separator set is a line.
type MenuItem struct {
	ID        MenuID
	Label     string
	Disabled  bool
	Checked   bool
	Separator bool
}

// BuildMenu builds the tray menu: the concise state, Open Hachidori,
// Restart Runtime, Diagnostics, the two opt-in preferences, and Quit.
func BuildMenu(s Summary, p Preferences) []MenuItem {
	return []MenuItem{
		{ID: MenuStatus, Label: "Hachidori: " + s.Label, Disabled: true},
		{Separator: true},
		{ID: MenuOpen, Label: "Open Hachidori"},
		{ID: MenuRestart, Label: "Restart Runtime", Disabled: !canRestart(s)},
		{ID: MenuDiagnostics, Label: "Diagnostics"},
		{Separator: true},
		{ID: MenuStartAtSignIn, Label: "Start Hachidori when I sign in", Checked: p.StartAtSignIn},
		{ID: MenuStartMinimized, Label: "Start minimized", Checked: p.StartMinimized},
		{Separator: true},
		{ID: MenuQuit, Label: "Quit Hachidori"},
	}
}

// canRestart: restart needs an installed runtime and no setup/stop in flight.
func canRestart(s Summary) bool {
	switch s.State {
	case app.Unconfigured, app.NotInstalled, app.Installing, app.Stopping, app.Paused:
		return false
	}
	return true
}

// Application is the slice of the application controller the tray consumes.
// *app.Controller implements it.
type Application interface {
	Snapshot() app.Snapshot
	Restart() error
}

// Action tells the native shell what to do on the UI thread after Resident
// handled an event.
type Action int

const (
	ActionNone            Action = iota
	ActionShow                   // show/restore/focus the existing window
	ActionShowDiagnostics        // same, navigated to the diagnostics section
	ActionHide                   // hide the window to the tray
	ActionHideWithNotice         // hide, and tell the user the app keeps running
	ActionQuit                   // destroy the window and end the desktop session
)

// Resident is the platform-neutral behavior of the resident desktop: what
// closing the window, the tray menu and a second launch mean. It holds no
// runtime state. Runtime ownership stays with the application controller,
// and the only runtime action it ever performs is the user's explicit
// Restart Runtime (worker restart policy stays with the supervisor).
type Resident struct {
	App   Application
	Prefs *Manager

	mu       sync.Mutex
	quitting bool
	wg       sync.WaitGroup
}

// Summary is the current concise application state.
func (r *Resident) Summary() Summary { return Summarize(r.App.Snapshot()) }

// Preferences reads the current preferences; a read failure shows them off.
func (r *Resident) Preferences() Preferences {
	p, _ := r.Prefs.Get()
	return p
}

// Menu is the tray menu for the current state and preferences.
func (r *Resident) Menu() []MenuItem { return BuildMenu(r.Summary(), r.Preferences()) }

// OnClose handles the user closing the main window. Closing hides to the
// tray and leaves the runtime untouched; the very first time it also asks
// the shell to say so. After Quit was chosen, closing really closes.
func (r *Resident) OnClose() Action {
	r.mu.Lock()
	quitting := r.quitting
	r.mu.Unlock()
	if quitting {
		return ActionQuit
	}
	if r.Prefs.ShouldShowCloseNotice() {
		return ActionHideWithNotice
	}
	return ActionHide
}

// OnActivate handles a second launch or a tray click: show the existing
// window. It never starts another runtime owner.
func (r *Resident) OnActivate() Action { return r.showAction() }

func (r *Resident) showAction() Action {
	if r.Summary().Level == LevelAttention {
		return ActionShowDiagnostics
	}
	return ActionShow
}

// OnMenu handles a tray menu selection.
func (r *Resident) OnMenu(id MenuID) (Action, error) {
	switch id {
	case MenuOpen:
		return r.showAction(), nil
	case MenuDiagnostics:
		return ActionShowDiagnostics, nil
	case MenuRestart:
		// Off the UI thread: a restart waits for the worker to stop. The
		// controller itself serializes and de-duplicates repeated requests.
		r.wg.Add(1)
		go func() {
			defer r.wg.Done()
			_ = r.App.Restart()
		}()
		return ActionNone, nil
	case MenuStartAtSignIn:
		p, err := r.Prefs.Get()
		if err != nil {
			return ActionNone, err
		}
		return ActionNone, r.Prefs.SetStartAtSignIn(!p.StartAtSignIn)
	case MenuStartMinimized:
		p, err := r.Prefs.Get()
		if err != nil {
			return ActionNone, err
		}
		return ActionNone, r.Prefs.SetStartMinimized(!p.StartMinimized)
	case MenuQuit:
		r.mu.Lock()
		r.quitting = true
		r.mu.Unlock()
		return ActionQuit, nil
	}
	return ActionNone, errors.New("unknown tray menu entry")
}

// Wait waits for an in-flight Restart Runtime request started by OnMenu.
func (r *Resident) Wait() { r.wg.Wait() }

// StartHidden reports whether a launch begins in the tray without showing the
// window: only a background (sign-in) launch with Start minimized enabled.
// An explicit launch always shows the window.
func StartHidden(background bool, p Preferences) bool { return background && p.StartMinimized }
