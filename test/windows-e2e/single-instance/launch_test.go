package singleinstance

import (
	"fmt"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// raceSize is the number of near-simultaneous launches of the first wave: one
// double-click entry and five `desktop` launches on distinct endpoints.
const raceSize = 6

// TestNearSimultaneousLaunchesHaveOneOwnerOneWorkerOneEndpoint certifies the
// single-instance guard of the packaged executable under a launch race. Six
// processes start within milliseconds of each other for the same user; each
// would bind its own endpoints if it became an owner. Exactly one survives as
// the owner; the other five exit 0 having activated it; only the owner binds
// endpoints (one API, one dashboard); and the one worker the owner is then
// asked to start is the only worker process on the machine.
func TestNearSimultaneousLaunchesHaveOneOwnerOneWorkerOneEndpoint(t *testing.T) {
	s := e2e.Begin(t, "single-instance-concurrent-launch")
	r := desktopkit.NewRun(t, s)
	w := newWorld(t, r)

	insts := launchTogether(t, startsOf(t, w, raceSize))
	for i, in := range insts {
		w.Track(fmt.Sprintf("racer-%d", i), in.Proc)
	}

	// Observable condition: all but one process has exited. The survivor is the
	// owner; no scenario code decides who it is.
	if err := desktopkit.Poll(startupWait+activateWait, "all but one launch to exit", func() (bool, error) {
		exited := 0
		for _, in := range insts {
			if _, done := in.Exited(); done {
				exited++
			}
		}
		if exited == len(insts) {
			return false, desktopkit.Fatal{Err: fmt.Errorf("every launch exited; no owner survived")}
		}
		return exited == len(insts)-1, nil
	}); err != nil {
		t.Fatalf("%v\n%s", err, raceOutputs(insts))
	}
	var owner *desktopkit.Instance
	var losers []*desktopkit.Instance
	for _, in := range insts {
		if in.Alive() {
			owner = in
		} else {
			losers = append(losers, in)
		}
	}
	if owner == nil {
		t.Fatalf("no owner is alive\n%s", raceOutputs(insts))
	}
	if t.Failed() {
		t.FailNow()
	}
	s.Logf("%d launches: one owner (pid %d), %d exited", len(insts), owner.Pid, len(losers))

	// The five others exited 0 by activating the owner and logged no startup.
	for _, l := range losers {
		code, _ := l.Exited()
		if code != 0 || !strings.Contains(l.Output(), activated) || l.StartMode() != "" {
			t.Errorf("loser pid %d: exit %d, startup mode %q, activation message present=%v; output:\n%s",
				l.Pid, code, l.StartMode(), strings.Contains(l.Output(), activated), l.Output())
		}
	}
	if m := owner.StartMode(); m != "launch" {
		t.Errorf("the owner logged startup mode %q, want launch; output:\n%s", m, owner.Output())
	}
	starters := 0
	for _, in := range insts {
		if in.StartMode() != "" {
			starters++
		}
	}
	if starters != 1 {
		t.Errorf("%d launches started a desktop composition, want exactly 1", starters)
	}

	// One owner serving its endpoints.
	if err := desktopkit.Poll(startupWait, "the owner's endpoints", func() (bool, error) {
		if !owner.Alive() {
			return false, desktopkit.Fatal{Err: fmt.Errorf("the owner exited; output:\n%s", owner.Output())}
		}
		if _, err := desktopkit.WizardState(owner.Dash); err != nil {
			return false, err
		}
		_, err := desktopkit.Status(owner.API)
		return err == nil, err
	}); err != nil {
		t.Fatal(err)
	}
	if err := owner.WaitShell(startupWait); err != nil {
		t.Fatal(err)
	}

	// No loser bound anything of its own.
	for _, l := range losers {
		if l.API != owner.API {
			requireNotDialable(t, fmt.Sprintf("loser pid %d API", l.Pid), l.API)
		}
		if l.Dash != owner.Dash {
			requireNotDialable(t, fmt.Sprintf("loser pid %d dashboard", l.Pid), l.Dash)
		}
	}
	before, err := w.census()
	if err != nil {
		t.Fatal(err)
	}
	s.Logf("before Start: %s", before)
	wantOnly(t, before, owner, 0)

	// One owner, one worker: Start is pressed once and one worker process exists.
	workerPID := startWorker(t, owner)
	after, err := w.census()
	if err != nil {
		t.Fatal(err)
	}
	s.Logf("after Start: %s", after)
	wantOnly(t, after, owner, workerPID)
	if procs, err := desktopkit.Processes(); err != nil {
		t.Fatal(err)
	} else if kids := desktopkit.ChildrenNamed(procs, owner.Pid, "python.exe"); len(kids) != 1 || kids[0].PID != workerPID {
		t.Errorf("the owner's worker children %v, want only %d", desktopkit.PIDs(kids), workerPID)
	}
	if st, err := desktopkit.Status(owner.API); err != nil || st.Worker.Starts != 1 {
		t.Errorf("worker starts %d (err %v), want exactly 1", st.Worker.Starts, err)
	}
	// The default endpoints were never bound by anyone: the owner used its own.
	if owner.API != server.DefaultListen {
		requireNotDialable(t, "the default API endpoint", server.DefaultListen)
		requireNotDialable(t, "the default dashboard endpoint", dashboard.DefaultListen)
	}
}

func raceOutputs(insts []*desktopkit.Instance) string {
	var b strings.Builder
	for i, in := range insts {
		code, exited := in.Exited()
		fmt.Fprintf(&b, "--- launch %d pid %d exited=%v code=%d\n%s\n", i, in.Pid, exited, code, in.Output())
	}
	return b.String()
}
