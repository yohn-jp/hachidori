package app

import (
	"context"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// recFake is a fakeRT whose status also reports the supervisor's restart
// count, so unexpected-exit recovery can be simulated deterministically.
type recFake struct {
	*fakeRT
	mu       sync.Mutex
	restarts int
}

func (f *recFake) Status() server.Status {
	st := f.fakeRT.Status()
	f.mu.Lock()
	st.Worker.Restarts = f.restarts
	f.mu.Unlock()
	return st
}

func (f *recFake) setRestarts(n int) { f.mu.Lock(); f.restarts = n; f.mu.Unlock() }

func newRecEnv(t *testing.T) (*Controller, *recFake, *atomic.Int32, *atomic.Int32) {
	t.Helper()
	rt := &recFake{fakeRT: newFakeRT()}
	var opens, setups atomic.Int32
	c := New(Config{
		Home:      "/h",
		Installed: func(string) bool { return true },
		Open:      func(string) (Runtime, error) { opens.Add(1); return rt, nil },
		Setup: func(string, string, string, io.Writer, *setup.Observer) error {
			setups.Add(1)
			return nil
		},
	})
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = c.Close(ctx)
	})
	return c, rt, &opens, &setups
}

var crash = &worker.FailureView{Class: worker.ClassCrash, Message: "worker exited with status 9"}

// A crash while the supervisor still has budget is a recovery in progress,
// not an operator stop and not a plain failure.
func TestUnexpectedExitIsRecoveringNotOperatorStop(t *testing.T) {
	c, rt, _, _ := newRecEnv(t)
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	rt.set(true, worker.StateReady, "ready", nil)
	if s := c.Snapshot(); s.State != Ready || s.Recovery != nil || s.OperatorStopped {
		t.Fatalf("ready snapshot %+v", s)
	}
	rt.setRestarts(1)
	rt.set(true, worker.StateRestarting, "", crash)
	s := c.Snapshot()
	if s.State != Starting || s.OperatorStopped || s.Recovery == nil ||
		s.Recovery.State != RecoveryRecovering || s.Recovery.Restarts != 1 ||
		s.Recovery.Cause == nil || s.Recovery.Cause.Class != worker.ClassCrash {
		t.Fatalf("recovering snapshot %+v recovery %+v", s, s.Recovery)
	}
}

// An operator Stop is distinguishable from a crash, and the marker clears on
// the next Start.
func TestOperatorStopIsDistinguishableFromCrash(t *testing.T) {
	c, rt, _, _ := newRecEnv(t)
	c.Start()
	rt.set(true, worker.StateReady, "ready", crash) // an earlier crash was recovered
	rt.setRestarts(1)
	if err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	s := c.Snapshot()
	if !s.OperatorStopped || s.Recovery != nil || s.State != Installed {
		t.Fatalf("after operator stop: %+v recovery %+v", s, s.Recovery)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	if s := c.Snapshot(); s.OperatorStopped {
		t.Fatalf("operator_stopped survived Start: %+v", s)
	}
}

func TestQuitIsOperatorStop(t *testing.T) {
	c, rt, _, _ := newRecEnv(t)
	c.Start()
	rt.set(true, worker.StateReady, "ready", nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if s := c.Snapshot(); !s.OperatorStopped || s.Recovery != nil {
		t.Fatalf("after quit: %+v", s)
	}
}

// Exhausted automatic recovery surfaces Needs attention (Failed) with the
// cause and an operator action; startup failures are not "recovery".
func TestSpentRestartBudgetGivesUpAndNeedsAttention(t *testing.T) {
	c, rt, opens, setups := newRecEnv(t)
	c.Start()
	rt.setRestarts(3)
	rt.set(false, worker.StateFailed, "", crash)
	s := c.Snapshot()
	if s.State != Failed || s.Failure == nil || s.Recovery == nil || s.Recovery.State != RecoveryGaveUp ||
		s.Recovery.Restarts != 3 || s.OperatorStopped {
		t.Fatalf("gave up snapshot %+v recovery %+v", s, s.Recovery)
	}
	if !strings.Contains(s.Recovery.Message, "no fallback to another device") {
		t.Fatalf("message %q", s.Recovery.Message)
	}
	// Stop is a no-op for a worker the supervisor gave up on; the evidence stays.
	if err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	if s := c.Snapshot(); s.Recovery == nil || s.OperatorStopped {
		t.Fatalf("stop erased recovery evidence: %+v", s)
	}
	// Operator action reopens the same runtime; it never re-runs setup.
	if err := c.Restart(); err != nil {
		t.Fatal(err)
	}
	if opens.Load() != 2 || setups.Load() != 0 {
		t.Fatalf("opens=%d setups=%d", opens.Load(), setups.Load())
	}
	if s := c.Snapshot(); s.Recovery != nil {
		t.Fatalf("recovery still reported after operator restart: %+v", s.Recovery)
	}
}

func TestStartupFailureIsNotRecovery(t *testing.T) {
	c, rt, _, _ := newRecEnv(t)
	c.Start()
	rt.set(false, worker.StateFailed, "loading", &worker.FailureView{Class: worker.ClassDevice, Message: "CUDA is unavailable"})
	s := c.Snapshot()
	if s.State != Failed || s.Recovery != nil || s.Failure.Class != worker.ClassDevice {
		t.Fatalf("snapshot %+v recovery %+v", s, s.Recovery)
	}
}

// Real supervisor: a worker that keeps dying after READY is restarted a
// bounded number of times, then the controller reports gave-up and stops.
func TestRealSupervisorCrashLoopIsBounded(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	var opens atomic.Int32
	open := func(root string) (Runtime, error) {
		opens.Add(1)
		cfg := worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeWorkerEnv + "=crash"},
			StartTimeout: 10 * time.Second, RequestTimeout: 5 * time.Second}
		sup := worker.NewSupervisor(cfg, worker.Policy{MaxRestarts: 2, Window: time.Minute, Backoff: 10 * time.Millisecond, QueueDepth: 4})
		return &WorkerBinding{Lifecycle: worker.NewLifecycle(ctx, sup), Supervisor: sup,
			Info: server.Runtime{Home: root}, Started: time.Now()}, nil
	}
	c := New(Config{Home: t.TempDir(), Installed: func(string) bool { return true }, Open: open})
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "bounded recovery to give up", func() bool {
		s := c.Snapshot()
		return s.Recovery != nil && s.Recovery.State == RecoveryGaveUp
	})
	s := c.Snapshot()
	if s.State != Failed || s.Status.Worker.Starts != 3 || s.Recovery.Restarts != 2 || s.Recovery.Cause.Class != worker.ClassCrash {
		t.Fatalf("state %s starts %d recovery %+v", s.State, s.Status.Worker.Starts, s.Recovery)
	}
	time.Sleep(300 * time.Millisecond) // no further restarts once given up
	if n := c.Snapshot().Status.Worker.Starts; n != 3 {
		t.Fatalf("restart loop continued: starts=%d", n)
	}
	// Explicit operator action resets the budget on a fresh supervisor.
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	if s := c.Snapshot(); opens.Load() != 2 || s.Status.Worker.Restarts != 0 || s.Status.Worker.Starts > 1 {
		t.Fatalf("opens %d worker %+v", opens.Load(), s.Status.Worker)
	}
	cctx, ccancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer ccancel()
	if err := c.Close(cctx); err != nil {
		t.Fatal(err)
	}
}
