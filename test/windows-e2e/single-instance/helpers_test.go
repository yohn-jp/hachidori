package singleinstance

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
)

const (
	// startupWait bounds a cold candidate reaching its first screen on a hosted
	// runner; activateWait bounds a second launch (it retries internally for up
	// to ten seconds while the owner's window comes up); workerWait bounds the
	// stand-in worker to READY.
	startupWait  = 90 * time.Second
	activateWait = 60 * time.Second
	workerWait   = 60 * time.Second

	// activated is what a duplicate launch prints once it asked the running
	// instance to show its window (cmd/hachidori desktopApp.launch).
	activated = "already running for this user; activated the existing window"
)

// world is one scenario's installed home, remembered through the locator.
type world struct {
	*desktopkit.Run
	home string
	// exe is the image name of the candidate, as the process table shows it.
	exe string
	// baseline are the pids of python.exe processes that existed before the
	// scenario started anything: they are not workers of this run.
	baseline map[int]bool
}

// newWorld installs the fixture home (a folder with spaces and Unicode, so the
// owner's launch also crosses that path shape) and remembers it in the
// disposable profile's bootstrap locator, then warms the candidate: it starts
// once, reaches its shell, and is ended. The warm-up pays the cold-start cost
// of the first execution of a fresh executable (loader, scanning) so the race
// scenarios measure the single-instance guard, not that cost.
func newWorld(t *testing.T, r *desktopkit.Run) *world {
	t.Helper()
	w := &world{Run: r, home: r.Path("homes", "Hachidori Home ハチドリ"), exe: filepath.Base(r.L.Exe)}
	if _, err := desktopkit.InstallFixture(w.home); err != nil {
		t.Fatalf("install the fixture home: %v", err)
	}
	if _, err := desktopkit.WriteLocator(r.Profile.LocatorPath(), w.home); err != nil {
		t.Fatal(err)
	}
	procs, err := desktopkit.Processes()
	if err != nil {
		t.Fatal(err)
	}
	w.baseline = map[int]bool{}
	for _, p := range desktopkit.Named(procs, "python.exe") {
		w.baseline[p.PID] = true
	}
	warm := r.Desktop("warm-up")
	if err := warm.WaitShell(startupWait); err != nil {
		t.Fatalf("warm-up launch: %v", err)
	}
	r.Stop(warm)
	return w
}

// owner starts the one owner, waits until it serves its dashboard and its shell
// is up, and returns it.
func (w *world) owner(t *testing.T, label string) *desktopkit.Instance {
	t.Helper()
	o := w.Desktop(label)
	if _, err := o.WaitWizard(startupWait, "the owner's dashboard", func(v firstrun.View) bool { return v.Mode == firstrun.ModeLaunch && v.Dashboard }); err != nil {
		t.Fatal(err)
	}
	if err := o.WaitShell(startupWait); err != nil {
		t.Fatal(err)
	}
	if got := o.StartMode(); got != string(firstrun.ModeLaunch) {
		t.Fatalf("the owner logged startup mode %q, want launch; output:\n%s", got, o.Output())
	}
	return o
}

// startWorker presses the dashboard's Start (a same-origin form post with the
// page's token) and waits until the API reports READY, then returns the pid of
// the one worker the owner started.
func startWorker(t *testing.T, o *desktopkit.Instance) int {
	t.Helper()
	tok, err := desktopkit.DashboardToken(o.Dash)
	if err != nil {
		t.Fatal(err)
	}
	if err := desktopkit.PostRuntime(o.Dash, "start", tok); err != nil {
		t.Fatal(err)
	}
	if err := desktopkit.Poll(workerWait, "the owner's worker READY", func() (bool, error) {
		if code, exited := o.Exited(); exited {
			return false, desktopkit.Fatal{Err: fmt.Errorf("the owner exited with code %d; output:\n%s", code, o.Output())}
		}
		return desktopkit.Ready(o.API)
	}); err != nil {
		t.Fatalf("%v\nowner output:\n%s", err, o.Output())
	}
	st, err := desktopkit.Status(o.API)
	if err != nil {
		t.Fatal(err)
	}
	if st.Worker.PID == 0 || !st.Worker.Ready {
		t.Fatalf("READY without a worker pid: %+v", st.Worker)
	}
	return st.Worker.PID
}

// census is one observation of who exists: the candidate processes, the worker
// stand-in processes, and every listening TCP socket owned by a candidate.
type census struct {
	Candidates []int
	Workers    []int
	Listeners  []desktopkit.Listener
}

func (w *world) census() (census, error) {
	procs, err := desktopkit.Processes()
	if err != nil {
		return census{}, err
	}
	ls, err := desktopkit.Listeners()
	if err != nil {
		return census{}, err
	}
	c := census{Candidates: desktopkit.PIDs(desktopkit.Named(procs, w.exe))}
	for _, p := range desktopkit.Named(procs, "python.exe") {
		if !w.baseline[p.PID] {
			c.Workers = append(c.Workers, p.PID)
		}
	}
	mine := map[int]bool{}
	for _, p := range c.Candidates {
		mine[p] = true
	}
	for _, l := range ls {
		if mine[l.PID] {
			c.Listeners = append(c.Listeners, l)
		}
	}
	return c, nil
}

func (c census) String() string {
	var ls []string
	for _, l := range c.Listeners {
		ls = append(ls, fmt.Sprintf("%s:%d(pid %d)", l.Host, l.Port, l.PID))
	}
	return fmt.Sprintf("candidates %v, workers %v, candidate listeners [%s]", c.Candidates, c.Workers, strings.Join(ls, " "))
}

// wantOnly fails unless the census is exactly one owner with its two
// endpoints and, when worker is non-zero, exactly one worker (a child of the
// owner); with worker zero there is no worker at all.
func wantOnly(t *testing.T, c census, o *desktopkit.Instance, worker int) {
	t.Helper()
	if len(c.Candidates) != 1 || c.Candidates[0] != o.Pid {
		t.Errorf("candidate processes %v, want only the owner %d", c.Candidates, o.Pid)
	}
	want := map[string]bool{o.API: false, o.Dash: false}
	for _, l := range c.Listeners {
		addr := fmt.Sprintf("%s:%d", l.Host, l.Port)
		if _, ok := want[addr]; !ok || l.PID != o.Pid {
			t.Errorf("unexpected candidate listener %s (pid %d); owner %d owns %s and %s", addr, l.PID, o.Pid, o.API, o.Dash)
			continue
		}
		want[addr] = true
	}
	for addr, seen := range want {
		if !seen {
			t.Errorf("the owner does not listen on %s", addr)
		}
	}
	switch {
	case worker == 0 && len(c.Workers) != 0:
		t.Errorf("worker processes %v exist, want none", c.Workers)
	case worker != 0 && (len(c.Workers) != 1 || c.Workers[0] != worker):
		t.Errorf("worker processes %v, want exactly the owner's worker %d", c.Workers, worker)
	}
}

// launchTogether starts every launch at the same instant: each start function
// is released by one gate, so the processes are created within a few
// milliseconds of each other.
func launchTogether(t *testing.T, starts []func() (*desktopkit.Instance, error)) []*desktopkit.Instance {
	t.Helper()
	gate := make(chan struct{})
	out := make([]*desktopkit.Instance, len(starts))
	errs := make([]error, len(starts))
	var wg sync.WaitGroup
	for i, start := range starts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-gate
			out[i], errs[i] = start()
		}()
	}
	close(gate)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("start launch %d: %v", i, err)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
	return out
}

// startsOf builds n start functions: one through the no-argument (double-click)
// entry point and the rest through `desktop`, each on its own addresses (all
// chosen up front, so no two launches can be handed the same one) so any launch
// that became a second owner would be visible on its own endpoints.
func startsOf(t *testing.T, w *world, n int) []func() (*desktopkit.Instance, error) {
	t.Helper()
	addrs, err := desktopkit.FreeAddrs(2 * (n - 1))
	if err != nil {
		t.Fatal(err)
	}
	starts := []func() (*desktopkit.Instance, error){w.L.NoArg}
	for i := 0; i+1 < len(addrs); i += 2 {
		api, dash := addrs[i], addrs[i+1]
		starts = append(starts, func() (*desktopkit.Instance, error) { return w.L.DesktopOn(api, dash) })
	}
	return starts
}

// requireNotDialable fails if something accepts connections on addr.
func requireNotDialable(t *testing.T, what, addr string) {
	t.Helper()
	if desktopkit.Dialable(addr) {
		t.Errorf("%s: %s accepts connections, but no second owner may bind an endpoint", what, addr)
	}
}
