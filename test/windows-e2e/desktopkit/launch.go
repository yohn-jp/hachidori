package desktopkit

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/server"
)

// Launcher starts the one candidate executable inside a disposable profile.
type Launcher struct {
	// Exe is the candidate hachidori.exe, from e2e.Scenario.Candidate().Path.
	Exe     string
	Profile Profile
	// Extra are KEY=value environment entries applied after the profile's.
	Extra []string
}

// Instance is one started candidate process and the loopback endpoints it was
// asked to bind.
type Instance struct {
	*Proc
	// API and Dash are the inference API and dashboard (window origin)
	// addresses the process binds when it becomes the owner.
	API  string
	Dash string
}

func (l Launcher) env() []string { return l.Profile.Env(os.Environ(), l.Extra...) }

// Run starts the candidate with args (a CLI command such as setup).
func (l Launcher) Run(args ...string) (*Proc, error) {
	return Start(l.Exe, args, l.env(), l.Profile.Root)
}

// Desktop starts `hachidori desktop` on two fresh loopback addresses. It is the
// same desktop composition and single-instance guard as the no-argument
// launch; the flags only choose where it would bind, which lets a scenario tell
// every would-be owner's endpoints apart.
func (l Launcher) Desktop() (*Instance, error) {
	a, err := FreeAddrs(2)
	if err != nil {
		return nil, err
	}
	return l.DesktopOn(a[0], a[1])
}

// DesktopOn is Desktop on explicit addresses.
func (l Launcher) DesktopOn(api, dash string) (*Instance, error) {
	p, err := l.Run("desktop", "--listen", api, "--addr", dash)
	if err != nil {
		return nil, err
	}
	return &Instance{Proc: p, API: api, Dash: dash}, nil
}

// NoArg starts hachidori.exe with no arguments: the product's double-click
// entry point, which binds the default loopback addresses.
func (l Launcher) NoArg() (*Instance, error) {
	p, err := l.Run()
	if err != nil {
		return nil, err
	}
	return &Instance{Proc: p, API: server.DefaultListen, Dash: dashboard.DefaultListen}, nil
}

// Endpoints are the owner endpoints of an instance.
func (i *Instance) Endpoints() []string { return []string{i.API, i.Dash} }

// WaitWizard polls the instance's first-run state until pred accepts it. It
// fails at once, with the process output, if the process exits first: a desktop
// that exits instead of showing a recovery state is exactly the failure the
// recovery scenarios exist to catch.
func (i *Instance) WaitWizard(timeout time.Duration, what string, pred func(firstrun.View) bool) (firstrun.View, error) {
	var last firstrun.View
	err := Poll(timeout, what, func() (bool, error) {
		if code, exited := i.Exited(); exited {
			return false, Fatal{fmt.Errorf("the process exited with code %d before %s; output:\n%s", code, what, i.Output())}
		}
		v, err := WizardState(i.Dash)
		if err != nil {
			return false, err
		}
		last = v
		return pred(v), nil
	})
	return last, err
}

var startMode = regexp.MustCompile(`hachidori: desktop start: (\w+)`)

// StartMode is the startup mode the process logged ("first_run", "resume",
// "recovery_missing", "recovery_invalid" or "launch"), or "" if it has not
// logged one.
func (i *Instance) StartMode() string {
	m := startMode.FindStringSubmatch(i.Output())
	if m == nil {
		return ""
	}
	return m[1]
}

// shellNavigated is the line the native shell logs when its first loopback
// navigation completed successfully (internal/desktop). It is the
// product's own, non-visual evidence that the shell composed, attached
// WebView2 and loaded the page; it says nothing about how the page looks.
const shellNavigated = "startup first navigation completed"

// WaitShell waits until the shell logged that its first navigation completed,
// and fails at once, with the output, if the process exits first. A scenario
// uses it to know the application is fully up (and stays up) before it reads
// or ends it.
func (i *Instance) WaitShell(timeout time.Duration) error {
	return Poll(timeout, "the desktop shell's first navigation", func() (bool, error) {
		if code, exited := i.Exited(); exited {
			return false, Fatal{fmt.Errorf("the process exited with code %d; output:\n%s", code, i.Output())}
		}
		return strings.Contains(i.Output(), shellNavigated), nil
	})
}
