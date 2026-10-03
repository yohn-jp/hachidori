package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Bounds. Every wait in a scenario is one of these deadlines on an observable
// condition; none is a sleep that stands in for an observation.
const (
	// BoundLaunch bounds a launch to the bound, stopped runtime (servers up,
	// controller bound), which does not start the worker.
	BoundLaunch = 90 * time.Second
	// BoundReady bounds Start/Restart/recovery to READY: interpreter start,
	// the real worker's import, load and three warmup predictions.
	BoundReady = 120 * time.Second
	// BoundStop bounds a Stop: the worker's shutdown message, with the
	// supervisor's own 10 second grace before it kills the process.
	BoundStop = 45 * time.Second
	// BoundQuit bounds Quit to process exit: the controller's 15 second
	// shutdown bound plus the worker grace.
	BoundQuit = 60 * time.Second
	// ObserveNegative is how long a scenario watches that something does NOT
	// happen (the worker is not restarted by itself after Stop or after
	// recovery gave up). It covers the supervisor's 2 second restart backoff
	// three times over, so a restart that was going to happen is seen.
	ObserveNegative = 8 * time.Second
	poll            = 100 * time.Millisecond
)

// Reporter is what a scenario offers for evidence: bounded, scrubbed log lines
// and attachments (*e2e.Scenario).
type Reporter interface {
	Logf(format string, args ...any)
	Attach(name string, data []byte, home string)
}

// Session is one disposable fixture home with a disposable profile and, once
// launched, the real hachidori.exe running on it.
type Session struct {
	T       testing.TB
	R       Reporter
	ExePath string
	Dir     string
	Fix     Fixture
	Profile Profile

	Exe    *Exe
	Client *Client

	launches int
}

// NewSession first checks that the executable it will launch is byte for byte
// the certified candidate (expectSHA256), then builds the disposable fixture
// home and profile under a fresh temporary directory and arranges that everything the scenario started is
// ended and its evidence attached when the test ends.
func NewSession(t testing.TB, r Reporter, exePath, expectSHA256, device string) (*Session, error) {
	if sum, err := FileSHA256(exePath); err != nil {
		return nil, fmt.Errorf("hash the executable under test: %w", err)
	} else if sum != expectSHA256 {
		return nil, fmt.Errorf("the executable under test has sha256 %s, the certified candidate is %s", sum, expectSHA256)
	}
	py, err := HostPython()
	if err != nil {
		return nil, err
	}
	dir := t.TempDir()
	prof, err := NewProfile(filepath.Join(dir, "profile"))
	if err != nil {
		return nil, err
	}
	fix, err := NewHome(filepath.Join(dir, "home"), device, py)
	if err != nil {
		return nil, err
	}
	s := &Session{T: t, R: r, ExePath: exePath, Dir: dir, Fix: fix, Profile: prof}
	t.Cleanup(s.cleanup)
	r.Logf("fixture home: device %s, runtime %s, model %s; host python %s", device, fix.RuntimeID, fix.ModelID, filepath.Base(py))
	return s, nil
}

func (s *Session) cleanup() {
	s.attachEvidence()
	if s.Exe != nil {
		s.Exe.Terminate()
	}
	// Whatever a failed scenario left of this home's worker is ended, so one
	// scenario's failure cannot bleed into the next.
	if all, err := ListProcesses(); err == nil {
		for _, p := range Roots(WorkerProcesses(all, s.Fix.Home.Root)) {
			_ = KillTree(p.PID)
		}
	}
}

func (s *Session) attachEvidence() {
	if s.Exe != nil {
		s.R.Attach(fmt.Sprintf("%s-hachidori-exe-launch%d.log", s.T.Name(), s.launches), s.Exe.LogTail(64<<10), s.Fix.Home.Root)
	}
	s.R.Attach(s.T.Name()+"-worker.log", TailFile(s.Fix.Home.Path("logs", "worker.log"), 64<<10), s.Fix.Home.Root)
}

// ChooseHome says how a launch finds its home.
type ChooseHome int

const (
	// ExplicitHome passes --home.
	ExplicitHome ChooseHome = iota
	// RememberedHome passes nothing: the home is the one the disposable
	// profile's bootstrap locator names, as after a first run.
	RememberedHome
)

// Launch starts the real hachidori.exe (`desktop` composition) on the home and
// waits, bounded, for the controller to be bound with the worker not started,
// which is how a normal launch of an installed home begins.
func (s *Session) Launch(how ChooseHome) error { return s.LaunchMode(how, "launch", "installed") }

// LaunchMode is Launch for a home that is expected to open in another desktop
// mode (the recovery and resume screens) or state.
func (s *Session) LaunchMode(how ChooseHome, wantMode, wantState string) error {
	if s.Exe != nil && !s.Exe.Exited() {
		return fmt.Errorf("a previous launch (pid %d) is still running", s.Exe.PID())
	}
	addrs, err := FreeAddrs(2)
	if err != nil {
		return err
	}
	opt := LaunchOptions{Exe: s.ExePath, Profile: s.Profile, APIAddr: addrs[0], Dash: addrs[1]}
	s.launches++
	opt.LogPath = filepath.Join(s.Dir, fmt.Sprintf("hachidori-exe-launch%d.log", s.launches))
	switch how {
	case ExplicitHome:
		opt.Home = s.Fix.Home.Root
	case RememberedHome:
		if _, err := s.Profile.Locator().Save(s.Fix.Home.Root); err != nil {
			return fmt.Errorf("remember the home in the disposable profile: %w", err)
		}
	}
	e, err := Launch(opt)
	if err != nil {
		return err
	}
	s.Exe = e
	s.Client = NewClient(opt.APIAddr, opt.Dash)
	s.R.Logf("launched hachidori.exe pid %d (home %s, api %s, dashboard %s)", e.PID(), map[ChooseHome]string{ExplicitHome: "--home", RememberedHome: "bootstrap locator"}[how], opt.APIAddr, opt.Dash)
	return Eventually(BoundLaunch, poll, "controller bound (stopped runtime)", func() (bool, string, error) {
		if err := s.exitedErr(); err != nil {
			return false, "", err
		}
		w, err := s.Client.Wizard()
		if err != nil {
			return false, err.Error(), nil
		}
		seen := fmt.Sprintf("desktop mode %q state %q", w.Mode, w.State)
		if w.Mode != wantMode || w.State != wantState {
			return false, seen, nil
		}
		if wantMode == "launch" {
			if _, err := s.Client.Status(); err != nil {
				return false, err.Error(), nil
			}
		}
		return true, seen, nil
	})
}

// exitedErr is non-nil when the executable has ended while a scenario still
// waits on it; the error carries its exit code and output tail.
func (s *Session) exitedErr() error {
	if s.Exe == nil || !s.Exe.Exited() {
		return nil
	}
	code, _ := s.Exe.Wait(time.Second)
	return fmt.Errorf("hachidori.exe exited with code %d before the awaited state; output tail:\n%s", code, s.Exe.LogTail(4<<10))
}

// Start is the dashboard's Start form followed by the bounded wait for READY.
func (s *Session) Start() (server.Status, error) { return s.act("start", 0) }

// Restart is the dashboard's Restart form followed by the bounded wait for a
// READY worker that is not prev (the worker that was running before it).
func (s *Session) Restart(prev int) (server.Status, error) { return s.act("restart", prev) }

// Do posts a lifecycle form ("start", "stop", "restart") without waiting for
// its outcome, for scenarios whose expected outcome is not READY.
func (s *Session) Do(op string) error { return s.Client.RuntimeAction(op) }

func (s *Session) act(op string, prev int) (server.Status, error) {
	if err := s.Client.RuntimeAction(op); err != nil {
		return server.Status{}, err
	}
	return s.WaitReady(op, prev)
}

// WaitReady waits, bounded, for READY with a worker pid other than prev (0:
// any). It stops early, with the output tail, if the executable exits.
func (s *Session) WaitReady(what string, prev int) (server.Status, error) {
	var st server.Status
	err := Eventually(BoundReady, poll, "READY after "+what, func() (bool, string, error) {
		if err := s.exitedErr(); err != nil {
			return false, "", err
		}
		code, h, err := s.Client.Health()
		if err != nil {
			return false, err.Error(), nil
		}
		x, err := s.Client.Status()
		if err != nil {
			return false, err.Error(), nil
		}
		st = x
		return code == 200 && h.Ready && x.Worker.State == worker.StateReady && x.Worker.PID > 0 && x.Worker.PID != prev,
			fmt.Sprintf("health %d %q; worker %s phase %q pid %d", code, h.State, x.Worker.State, x.Worker.Phase, x.Worker.PID), nil
	})
	return st, err
}

// Stop is the dashboard's Stop form followed by the bounded wait for the worker
// to be stopped.
func (s *Session) Stop() (server.Status, error) {
	if err := s.Client.RuntimeAction("stop"); err != nil {
		return server.Status{}, err
	}
	return s.Client.WaitWorkerState(worker.StateStopped, BoundStop)
}

// WaitGone waits, bounded, until every pid is no longer a running process.
func WaitGone(what string, timeout time.Duration, pids ...int) error {
	return Eventually(timeout, poll, what+" gone", func() (bool, string, error) {
		var alive []int
		for _, p := range pids {
			if Alive(p) {
				alive = append(alive, p)
			}
		}
		return len(alive) == 0, fmt.Sprintf("still running: %v", alive), nil
	})
}

// Workers lists the worker process roots the home currently owns.
func (s *Session) Workers() ([]Process, error) { return WorkersOf(s.Fix.Home.Root) }

// WorkersOf lists the worker process roots the home at root currently owns.
func WorkersOf(root string) ([]Process, error) {
	all, err := ListProcesses()
	if err != nil {
		return nil, err
	}
	return Roots(WorkerProcesses(all, root)), nil
}

// ExpectOneWorker checks that exactly one worker is owned by the home and that
// it is the process the status reports (no second runtime owner, no orphan).
func (s *Session) ExpectOneWorker(pid int) error {
	ws, err := s.Workers()
	if err != nil {
		return err
	}
	if len(ws) != 1 {
		return fmt.Errorf("%d worker processes own the home, want exactly 1: %v", len(ws), PIDs(ws))
	}
	if ws[0].PID != pid {
		return fmt.Errorf("the worker process is %d, status reports %d", ws[0].PID, pid)
	}
	return nil
}

// ExpectNoWorker checks that the home owns no worker process.
func (s *Session) ExpectNoWorker() error {
	ws, err := s.Workers()
	if err != nil {
		return err
	}
	if len(ws) != 0 {
		return fmt.Errorf("worker processes still own the home: %v", PIDs(ws))
	}
	return nil
}

// Quit asks the running desktop shell to quit (the Quit Hachidori path) and
// waits, bounded, for the process to exit; it returns the exit code. The window
// may not exist for a moment after the servers come up, so the request is
// retried until BoundQuit.
func (s *Session) Quit() (int, error) {
	if runtime.GOOS != "windows" {
		return 0, fmt.Errorf("Quit needs the Windows desktop shell")
	}
	err := Eventually(BoundQuit, poll, "quit request accepted by the desktop window", func() (bool, string, error) {
		if s.Exe.Exited() {
			return true, "", nil
		}
		err := RequestQuit()
		if err == nil {
			return true, "", nil
		}
		if IsNoWindow(err) {
			return false, err.Error(), nil
		}
		return false, "", err
	})
	if err != nil {
		return 0, err
	}
	return s.Exe.Wait(BoundQuit)
}

// ExpectPortsFree checks that nothing listens on the launch's API and dashboard
// addresses any more.
func (s *Session) ExpectPortsFree() error {
	for _, a := range []string{s.Client.APIAddr, s.Client.DashAddr} {
		if Listening(a) {
			return fmt.Errorf("%s is still listening after the executable exited", a)
		}
	}
	return nil
}

// ExpectPersisted checks the persisted state against an earlier digest.
func (s *Session) ExpectPersisted(before PersistedState) error {
	after, err := s.Fix.Persisted()
	if err != nil {
		return err
	}
	if d := before.Diff(after); d != "" {
		return fmt.Errorf("%s", d)
	}
	left, err := s.Fix.Leftovers()
	if err != nil {
		return err
	}
	if len(left) != 0 {
		return fmt.Errorf("staging or partial files appeared: %v", left)
	}
	return nil
}

// HomeFileExists is true when the path (relative to the home) exists.
func (s *Session) HomeFileExists(rel ...string) bool {
	_, err := os.Stat(s.Fix.Home.Path(rel...))
	return err == nil
}
