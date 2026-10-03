package recovery

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
	"github.com/yohn-jp/hachidori/test/windows-e2e/runtime/harness"
)

func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.ShardRecovery)) }

// TestCandidateIdentity is the shard's shared precondition: it certifies the
// exact candidate bytes the workflow built once.
func TestCandidateIdentity(t *testing.T) { e2e.VerifyCandidateScenario(t) }

// Every scenario runs the real packaged hachidori.exe of the candidate in a
// disposable profile on a disposable fixture home (see the runtime shard and
// package harness for what the fixture does and does not replace: only the
// model payload behind the real controller, worker and HTTP API). A failure is
// injected from outside, then the scenario follows failure -> observable
// evidence -> bounded behavior -> consistent persisted state, waiting only on
// observable conditions with bounded deadlines (harness.Eventually/Holds).

var check = harness.Check

func newSession(t *testing.T, s *e2e.Scenario, device string) *harness.Session {
	t.Helper()
	c := s.Candidate()
	sess, err := harness.NewSession(t, s, c.Path, c.SHA256, device)
	check(t, "disposable fixture session", err)
	s.Logf("certifying candidate %s sha256=%s", c.File, c.SHA256)
	return sess
}

// startReady launches the executable on the home and starts the worker.
func startReady(t *testing.T, sess *harness.Session) server.Status {
	t.Helper()
	check(t, "launch", sess.Launch(harness.ExplicitHome))
	st, err := sess.Start()
	check(t, "Start to READY", err)
	sess.ExpectIdentity(st)
	check(t, "one worker", sess.ExpectOneWorker(st.Worker.PID))
	return st
}

// worker-kill-recovers: ending the worker process is reported as a recovery
// and the worker returns to READY with a new pid, serving typed inference
// again, with nothing persisted changed.
//
// Proves W09.1 (ending the worker reports "Recovering from an unexpected worker
// exit" and the worker returns to READY with a new pid) and, for worker
// termination, part of W09.3 (persisted state consistent after worker
// termination).
func TestWorkerKillRecovers(t *testing.T) {
	s := e2e.Begin(t, "worker-kill-recovers")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	st1 := startReady(t, sess)
	sess.Typed("alpha beta beta", "q", []string{"alpha", "beta"})

	check(t, "end the worker", sess.KillWorker(st1.Worker.PID))
	rec, err := sess.AwaitRecovering()
	check(t, "observable recovery", err)
	if lf := rec.Worker.LastFailure; lf == nil || lf.Class != worker.ClassCrash {
		t.Fatalf("the recovery does not name the worker crash: %+v", lf)
	}
	s.Logf("recovering: worker %s, restarts in window %d, last failure %s", rec.Worker.State, rec.Worker.Restarts, rec.Worker.LastFailure.Class)

	st2, err := sess.WaitReady("worker recovery", st1.Worker.PID)
	check(t, "recovered to READY", err)
	if st2.Worker.Restarts != 1 || st2.Worker.Starts != st1.Worker.Starts+1 {
		t.Fatalf("recovery was not one bounded restart: restarts %d starts %d (was %d)", st2.Worker.Restarts, st2.Worker.Starts, st1.Worker.Starts)
	}
	sess.ExpectIdentity(st2)
	check(t, "exactly one worker (the new one)", sess.ExpectOneWorker(st2.Worker.PID))
	check(t, "no recovery notice once READY", sess.ExpectNoRecoveryNotice())
	if w, err := sess.Client.Wizard(); err != nil || w.State != "ready" {
		t.Fatalf("application state %+v %v, want ready", w, err)
	}
	sess.Typed("beta alpha alpha", "q", []string{"alpha", "beta"})
	s.Logf("worker pid %d -> %d", st1.Worker.PID, st2.Worker.PID)
	check(t, "persisted state consistent after the termination", sess.ExpectPersisted(before))
}

// repeated-failure-needs-attention: three bounded automatic recoveries, then
// the fourth termination inside the window ends automatic recovery in Needs
// attention, where it stays until the operator chooses Restart Runtime.
//
// Proves W09.2 (more than 3 worker kills in 10 minutes end in Needs attention
// and "Automatic recovery stopped" until Restart Runtime).
func TestRepeatedFailureNeedsAttention(t *testing.T) {
	s := e2e.Begin(t, "repeated-failure-needs-attention")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	st := startReady(t, sess)
	pid := st.Worker.PID

	// The budget is 3 automatic restarts within 10 minutes (worker.DefaultPolicy).
	for i := 1; i <= 3; i++ {
		check(t, fmt.Sprintf("kill %d", i), sess.KillWorker(pid))
		_, err := sess.AwaitRecovering()
		check(t, fmt.Sprintf("kill %d: observable recovery", i), err)
		next, err := sess.WaitReady(fmt.Sprintf("recovery %d", i), pid)
		check(t, fmt.Sprintf("kill %d: recovered", i), err)
		if next.Worker.Restarts != i {
			t.Fatalf("kill %d: restarts in window %d, want %d", i, next.Worker.Restarts, i)
		}
		check(t, "one worker", sess.ExpectOneWorker(next.Worker.PID))
		pid = next.Worker.PID
		s.Logf("recovery %d: new worker pid %d", i, pid)
	}

	check(t, "fourth kill", sess.KillWorker(pid))
	failed, err := sess.AwaitGaveUp()
	check(t, "Needs attention / Automatic recovery stopped", err)
	if lf := failed.Worker.LastFailure; lf == nil || lf.Class != worker.ClassCrash {
		t.Fatalf("the failure that ended recovery is %+v, want %s", lf, worker.ClassCrash)
	}
	w, err := sess.Client.Wizard()
	check(t, "application state", err)
	if w.State != "failed" || w.Failure == nil || w.Failure.Class != worker.ClassCrash {
		t.Fatalf("application state %q failure %+v, want failed (Needs attention) with %s", w.State, w.Failure, worker.ClassCrash)
	}
	// Bounded: 1 start + 3 automatic restarts, never a fifth start by itself.
	starts := failed.Worker.Starts
	if starts != 4 {
		t.Fatalf("starts %d, want 4 (1 start + 3 bounded restarts)", starts)
	}
	sess.ExpectNotServing("Needs attention")
	sess.ExpectRefused("Needs attention")
	check(t, "recovery stays stopped", harness.Holds(harness.ObserveNegative, 200*time.Millisecond, "failed stays failed until Restart", func() (bool, string, error) {
		x, err := sess.Client.Status()
		if err != nil {
			return false, "", err
		}
		ws, err := sess.Workers()
		return x.Worker.State == worker.StateFailed && x.Worker.Starts == starts && len(ws) == 0,
			fmt.Sprintf("worker %s starts %d workers %v", x.Worker.State, x.Worker.Starts, harness.PIDs(ws)), err
	}))

	// Restart Runtime is the operator's act and begins a fresh budget.
	back, err := sess.Restart(pid)
	check(t, "Restart Runtime", err)
	if back.Worker.Restarts != 0 {
		t.Fatalf("the budget did not restart with the operator's Restart: %d restarts in window", back.Worker.Restarts)
	}
	check(t, "one worker", sess.ExpectOneWorker(back.Worker.PID))
	check(t, "no recovery notice after Restart", sess.ExpectNoRecoveryNotice())
	sess.Typed("alpha alpha beta", "q", []string{"alpha", "beta"})
	s.Logf("Needs attention after kill 4 (starts %d), Restart Runtime -> worker pid %d READY", starts, back.Worker.PID)
	check(t, "persisted state consistent", sess.ExpectPersisted(before))
}

// host-kill-no-orphan-worker: the executable is ended hard (no orderly Quit).
// Its worker must not outlive it (the worker exits on stdin EOF), nothing is
// left half-written, and the next launch of the same home reaches READY.
//
// Proves, for interruption, part of W09.3 (persisted state is consistent after
// supported interruption recovery).
func TestHostKillLeavesNoOrphanWorker(t *testing.T) {
	s := e2e.Begin(t, "host-kill-no-orphan-worker")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	st1 := startReady(t, sess)
	sess.Typed("beta beta alpha", "q", []string{"alpha", "beta"})
	exePID := sess.Exe.PID()

	check(t, "end hachidori.exe hard (not its worker)", harness.KillProcess(exePID))
	_, err = sess.Exe.Wait(harness.BoundStop)
	check(t, "executable exit", err)
	check(t, "executable gone", harness.WaitGone("hachidori.exe", harness.BoundStop, exePID))
	check(t, "the worker does not outlive the executable", harness.WaitGone("worker", harness.BoundQuit, st1.Worker.PID))
	check(t, "no owned worker process", sess.ExpectNoWorker())
	check(t, "no owned listening port", sess.ExpectPortsFree())
	check(t, "persisted state consistent after the interruption", sess.ExpectPersisted(before))
	s.Logf("after the hard kill of pid %d: worker pid %d gone, ports closed, persisted state unchanged", exePID, st1.Worker.PID)

	// The next launch of the same home recovers: no wizard, READY, serving.
	check(t, "relaunch", sess.Launch(harness.ExplicitHome))
	st2, err := sess.Start()
	check(t, "Start after the interruption", err)
	if st2.Worker.PID == st1.Worker.PID {
		t.Fatalf("worker pid reused")
	}
	sess.ExpectIdentity(st2)
	check(t, "one worker", sess.ExpectOneWorker(st2.Worker.PID))
	sess.Typed("alpha alpha alpha beta", "q", []string{"alpha", "beta"})
	check(t, "persisted state consistent after recovery", sess.ExpectPersisted(before))
}

// corrupt-activation-record-recovery: a damaged state/active-runtime.json is not
// guessed at or overwritten. The launch shows the resume state for that home
// (no runtime bound, no worker, nothing installed) and leaves the files as they
// are; once the record is whole again the same launch path reaches READY.
//
// Proves, for corrupt state, part of W09.3 (persisted state is consistent after
// supported corrupt-state recovery).
func TestCorruptActivationRecord(t *testing.T) {
	s := e2e.Begin(t, "corrupt-activation-record-recovery")
	sess := newSession(t, s, "cpu")
	good, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	record := sess.Fix.Home.Path("state", "active-runtime.json")
	original, err := os.ReadFile(record)
	check(t, "activation record", err)

	damaged := original[:len(original)/2]
	check(t, "damage the activation record", os.WriteFile(record, damaged, 0o644))
	damagedState, err := sess.Fix.Persisted()
	check(t, "persisted state", err)

	check(t, "launch on a damaged record", sess.LaunchMode(harness.ExplicitHome, "resume", "not_installed"))
	if _, err := sess.Client.Status(); err == nil {
		t.Fatalf("a runtime was bound from a damaged activation record")
	}
	sess.ExpectNotServing("damaged activation record")
	sess.ExpectRefused("damaged activation record")
	check(t, "no worker", sess.ExpectNoWorker())
	check(t, "the damaged record and manifests are untouched", sess.ExpectPersisted(damagedState))
	if got, err := os.ReadFile(record); err != nil || string(got) != string(damaged) {
		t.Fatalf("the activation record was rewritten or removed (%v)", err)
	}
	s.Logf("damaged record: desktop opened in resume mode, nothing bound, nothing started, files untouched")
	code, err := sess.Quit()
	check(t, "Quit", err)
	if code != 0 {
		t.Fatalf("exit code %d", code)
	}

	check(t, "restore the activation record", os.WriteFile(record, original, 0o644))
	check(t, "launch on the restored record", sess.Launch(harness.ExplicitHome))
	st, err := sess.Start()
	check(t, "Start", err)
	sess.ExpectIdentity(st)
	sess.Typed("beta beta", "q", []string{"alpha", "beta"})
	check(t, "persisted state consistent after recovery", sess.ExpectPersisted(good))
}

// damaged-worker-script-replaced: the worker script the executable delivered
// into the home is damaged while the runtime is stopped. Starting it fails
// observably and is not retried by itself; the operator's Restart Runtime makes
// the controller deliver this build's script again (a damaged copy is replaced,
// content addressed), and the worker is READY with a new pid.
//
// Proves, for corrupt delivered state, part of W09.3 (persisted state is
// consistent after supported corrupt-state recovery).
func TestDamagedWorkerScriptIsReplaced(t *testing.T) {
	s := e2e.Begin(t, "damaged-worker-script-replaced")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	st1 := startReady(t, sess)
	script := sess.WorkerScriptPath(st1)
	digest := st1.Runtime.Worker.SHA256

	_, err = sess.Stop()
	check(t, "Stop", err)
	check(t, "damage the delivered worker script", os.WriteFile(script, []byte("this is not a worker\n"), 0o644))
	if sum, err := harness.FileSHA256(script); err != nil || sum == digest {
		t.Fatalf("the script was not damaged (%v)", err)
	}

	// The damaged script cannot start: an observable startup failure that is not retried.
	check(t, "Start posts", sess.Do("start"))
	failed, err := sess.Client.WaitWorkerState(worker.StateFailed, harness.BoundReady)
	check(t, "observable startup failure", err)
	if lf := failed.Worker.LastFailure; lf == nil || lf.Class != worker.ClassStartup {
		t.Fatalf("last failure %+v, want class %s", lf, worker.ClassStartup)
	}
	sess.ExpectNotServing("damaged worker script")
	starts := failed.Worker.Starts
	check(t, "no automatic retry", harness.Holds(harness.ObserveNegative/2, 200*time.Millisecond, "failed stays failed", func() (bool, string, error) {
		x, err := sess.Client.Status()
		if err != nil {
			return false, "", err
		}
		ws, err := sess.Workers()
		return x.Worker.State == worker.StateFailed && x.Worker.Starts == starts && len(ws) == 0, fmt.Sprintf("worker %s starts %d", x.Worker.State, x.Worker.Starts), err
	}))

	// Restart Runtime: the controller opens the home again, which delivers the
	// worker script of this build (replacing the damaged copy).
	st2, err := sess.Restart(st1.Worker.PID)
	check(t, "Restart Runtime", err)
	sess.ExpectWorkerScript(st2)
	if st2.Runtime.Worker.SHA256 != digest {
		t.Fatalf("worker build identity changed: %s -> %s", digest, st2.Runtime.Worker.SHA256)
	}
	if sum, err := harness.FileSHA256(filepath.Clean(script)); err != nil || sum != digest {
		t.Fatalf("the damaged script was not replaced: digest %s (%v)", sum, err)
	}
	check(t, "one worker", sess.ExpectOneWorker(st2.Worker.PID))
	sess.Typed("alpha beta alpha", "q", []string{"alpha", "beta"})
	s.Logf("damaged script: startup failure %s observed, Restart Runtime replaced it, worker pid %d READY", worker.ClassStartup, st2.Worker.PID)
	check(t, "persisted state consistent", sess.ExpectPersisted(before))
}
