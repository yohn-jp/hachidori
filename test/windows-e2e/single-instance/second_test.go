package singleinstance

import (
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// TestSecondLaunchContract proves the non-visual second-launch contract of
// W06.2 against the packaged executable while the owner has a READY worker:
// a second launch exits 0, starts no second worker, binds no endpoint and
// leaves the owner and its worker pid unchanged. The second launch is made
// through both entry points: the no-argument (double-click) entry, whose
// would-be endpoints are the default addresses, and `desktop` on fresh
// addresses.
//
// It asserts nothing visual: that the owner's window is shown or focused
// (W06.6) stays physical evidence, and no result of this scenario is a visual
// PASS.
func TestSecondLaunchContract(t *testing.T) {
	s := e2e.Begin(t, "single-instance-second-launch")
	r := desktopkit.NewRun(t, s)
	w := newWorld(t, r)

	owner := w.owner(t, "owner")
	workerPID := startWorker(t, owner)
	before, err := w.census()
	if err != nil {
		t.Fatal(err)
	}
	s.Logf("owner pid %d, worker pid %d; %s", owner.Pid, workerPID, before)
	wantOnly(t, before, owner, workerPID)
	if owner.API == server.DefaultListen || owner.Dash == dashboard.DefaultListen {
		t.Fatalf("the owner holds a default address (%s, %s); the no-argument second launch could not be told apart", owner.API, owner.Dash)
	}

	for _, entry := range []string{"no-argument", "desktop"} {
		var second *desktopkit.Instance
		if entry == "no-argument" {
			second = w.NoArg("second-" + entry)
		} else {
			second = w.Desktop("second-" + entry)
		}
		code, err := second.WaitExit(activateWait)
		if err != nil {
			t.Fatalf("%s second launch did not exit: %v\noutput:\n%s", entry, err, second.Output())
		}
		out := second.Output()
		if code != 0 {
			t.Errorf("%s second launch exited %d, want 0; output:\n%s", entry, code, out)
		}
		if !strings.Contains(out, activated) {
			t.Errorf("%s second launch did not report activating the running instance; output:\n%s", entry, out)
		}
		if second.StartMode() != "" || strings.Contains(out, "WebView2 Runtime") {
			t.Errorf("%s second launch started a desktop composition; output:\n%s", entry, out)
		}
		s.Logf("%s second launch: exit %d after activating the owner", entry, code)

		// No endpoint of its own, for either entry point.
		requireNotDialable(t, entry+" second launch API", second.API)
		requireNotDialable(t, entry+" second launch dashboard", second.Dash)

		// The owner and its one worker are exactly as before.
		if !owner.Alive() {
			t.Fatalf("the owner exited during the %s second launch; output:\n%s", entry, owner.Output())
		}
		after, err := w.census()
		if err != nil {
			t.Fatal(err)
		}
		wantOnly(t, after, owner, workerPID)
		st, err := desktopkit.Status(owner.API)
		if err != nil {
			t.Fatalf("owner status after the %s second launch: %v", entry, err)
		}
		if st.Worker.PID != workerPID || !st.Worker.Ready || st.Worker.Starts != 1 {
			t.Errorf("after the %s second launch the worker is %+v, want pid %d READY with one start", entry, st.Worker, workerPID)
		}
	}
	// The owner's dashboard still answers: it was never replaced.
	if _, err := desktopkit.WizardState(owner.Dash); err != nil {
		t.Errorf("the owner's dashboard: %v", err)
	}
	s.Logf("owner %d and worker %d unchanged after both second launches; no visual assertion is made", owner.Pid, workerPID)
}

// TestGuardIsReleasedWhenTheOwnerEnds certifies that the single-instance guard
// does not outlive its owner: after the owner process (and its worker) is
// terminated abruptly, with no chance to release anything, the next launch
// becomes the new owner instead of activating a window that no longer exists.
func TestGuardIsReleasedWhenTheOwnerEnds(t *testing.T) {
	s := e2e.Begin(t, "single-instance-guard-released")
	r := desktopkit.NewRun(t, s)
	w := newWorld(t, r)

	first := w.owner(t, "first-owner")
	firstWorker := startWorker(t, first)
	r.Stop(first)
	if err := desktopkit.Poll(startupWait, "the first owner's worker to end with it", func() (bool, error) {
		c, err := w.census()
		if err != nil {
			return false, err
		}
		if len(c.Candidates) != 0 || len(c.Workers) != 0 {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("%v (first owner %d, worker %d)", err, first.Pid, firstWorker)
	}

	second := w.owner(t, "second-owner")
	if second.Pid == first.Pid {
		t.Fatalf("pid %d reused unexpectedly", second.Pid)
	}
	if _, exited := second.Exited(); exited {
		t.Fatalf("the launch after the owner ended exited instead of becoming the owner; output:\n%s", second.Output())
	}
	c, err := w.census()
	if err != nil {
		t.Fatal(err)
	}
	wantOnly(t, c, second, 0)
	s.Logf("owner %d ended; the next launch %d became the owner; %s", first.Pid, second.Pid, c)
}
