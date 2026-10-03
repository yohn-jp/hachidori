package runtime

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
	"github.com/yohn-jp/hachidori/test/windows-e2e/runtime/harness"
)

func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.ShardRuntime)) }

// TestCandidateIdentity is the shard's shared precondition: it certifies the
// exact candidate bytes the workflow built once.
func TestCandidateIdentity(t *testing.T) { e2e.VerifyCandidateScenario(t) }

// What every scenario below runs: the real packaged hachidori.exe of the
// candidate (`hachidori.exe desktop`, the composition a double-click starts),
// launched as a subprocess in a disposable profile on a disposable fixture
// home. The application controller inside it opens the home, its supervisor
// starts the real hachidori_worker.py the executable itself delivers, and the
// scenario talks to it over the real loopback HTTP API and the dashboard's own
// forms. The only stand-in is the model payload: the home's private Python
// environment carries tiny deterministic torch/laya stubs instead of PyTorch,
// Laya and weights, so nothing production-sized is downloaded or installed.
// Every wait is a bounded poll of an observable condition (harness.Eventually
// and harness.Holds), never a sleep that stands in for an observation.

var check = harness.Check

func newSession(t *testing.T, s *e2e.Scenario, device string) *harness.Session {
	t.Helper()
	c := s.Candidate()
	sess, err := harness.NewSession(t, s, c.Path, c.SHA256, device)
	check(t, "disposable fixture session", err)
	s.Logf("certifying candidate %s sha256=%s", c.File, c.SHA256)
	return sess
}

// cpu-ready-typed-inference: the real packaged exe -> controller -> worker ->
// HTTP API -> inference path on a deterministic CPU home.
//
// Proves W03.1 (the CPU runtime reaches READY with model, device and runtime
// identity; the home is a prepared fixture, so the setup materialization
// itself is not exercised here) and W12.2 (a deterministic fixture CPU home
// crosses packaged executable, controller, worker, HTTP API and typed
// inference; not a production-model result).
func TestCPUReadyTypedInference(t *testing.T) {
	s := e2e.Begin(t, "cpu-ready-typed-inference")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)

	check(t, "launch", sess.Launch(harness.ExplicitHome))
	// Opening Hachidori binds the runtime without starting the worker.
	sess.ExpectNotServing("a bound, stopped runtime")
	check(t, "no worker before Start", sess.ExpectNoWorker())

	st, err := sess.Start()
	check(t, "Start to READY", err)
	sess.ExpectIdentity(st)
	sess.ExpectWorkerScript(st)
	check(t, "one worker", sess.ExpectOneWorker(st.Worker.PID))
	s.Logf("READY: model %s device %s runtime %s worker pid %d starts %d", st.Runtime.ModelID, st.Runtime.Device, st.Runtime.Runtime, st.Worker.PID, st.Worker.Starts)

	sess.Typed("the plan says beta, then beta again; alpha once", "pick", []string{"alpha", "beta"})
	sess.Typed("yes, yes and YES; no", "answer", []string{"yes", "no", "maybe"})
	after, err := sess.Client.Status()
	check(t, "status after inference", err)
	if after.Worker.Requests < 2 || after.Worker.PID != st.Worker.PID {
		t.Fatalf("the worker did not serve the requests: requests %d pid %d (was %d)", after.Worker.Requests, after.Worker.PID, st.Worker.PID)
	}
	check(t, "persisted state unchanged", sess.ExpectPersisted(before))
}

// stop-start-restart: the lifecycle actions of the real controller through the
// dashboard's own forms, each observed on the worker process itself.
//
// Proves W08.1 (Restart gives a new worker pid with no second owner) and the
// Stop half of W08.2 (Stop does not restart the worker by itself; the Quit half
// is quit-leaves-no-worker-or-port).
func TestStopStartRestart(t *testing.T) {
	s := e2e.Begin(t, "stop-start-restart")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	check(t, "launch", sess.Launch(harness.ExplicitHome))

	st1, err := sess.Start()
	check(t, "Start to READY", err)
	check(t, "one worker", sess.ExpectOneWorker(st1.Worker.PID))
	sess.Typed("alpha alpha beta", "q", []string{"alpha", "beta"})

	// Stop: the worker ends, the API stays bound and says not ready.
	stopped, err := sess.Stop()
	check(t, "Stop", err)
	check(t, "stopped worker process gone", harness.WaitGone("stopped worker", harness.BoundStop, st1.Worker.PID))
	check(t, "no worker owned after Stop", sess.ExpectNoWorker())
	sess.ExpectNotServing("after Stop")
	sess.ExpectRefused("after Stop")
	// Stop never restarts the worker by itself: observed over a window that
	// covers the supervisor's restart backoff several times.
	startsAfterStop := stopped.Worker.Starts
	check(t, "Stop is not followed by a restart", harness.Holds(harness.ObserveNegative, 200*time.Millisecond, "stopped stays stopped", func() (bool, string, error) {
		x, err := sess.Client.Status()
		if err != nil {
			return false, "", err
		}
		ws, err := sess.Workers()
		if err != nil {
			return false, "", err
		}
		w, err := sess.Client.Wizard()
		if err != nil {
			return false, "", err
		}
		return x.Worker.State == worker.StateStopped && x.Worker.Starts == startsAfterStop && len(ws) == 0 && w.State == "installed",
			fmt.Sprintf("worker %s starts %d workers %v app %s", x.Worker.State, x.Worker.Starts, harness.PIDs(ws), w.State), nil
	}))

	// Start after Stop: a new worker.
	st2, err := sess.Start()
	check(t, "Start after Stop", err)
	if st2.Worker.PID == st1.Worker.PID {
		t.Fatalf("Start after Stop reused worker pid %d", st1.Worker.PID)
	}
	check(t, "one worker after Start", sess.ExpectOneWorker(st2.Worker.PID))
	sess.Typed("beta beta beta alpha", "q", []string{"alpha", "beta"})

	// Restart: a new worker, the old one gone, never two owners.
	st3, err := sess.Restart(st2.Worker.PID)
	check(t, "Restart to a new READY worker", err)
	check(t, "restarted-away worker gone", harness.WaitGone("old worker", harness.BoundStop, st2.Worker.PID))
	check(t, "exactly one worker after Restart", sess.ExpectOneWorker(st3.Worker.PID))
	sess.ExpectIdentity(st3)
	sess.Typed("alpha", "q", []string{"alpha", "beta"})
	s.Logf("worker pids: start %d, after Stop+Start %d, after Restart %d", st1.Worker.PID, st2.Worker.PID, st3.Worker.PID)
	check(t, "persisted state unchanged", sess.ExpectPersisted(before))
}

// quit-leaves-no-worker-or-port: Quit Hachidori (the desktop shell's own quit
// path) ends the process within its bound and leaves no owned worker and no
// owned listening port.
//
// Proves W06.4 (Quit frees the API and dashboard ports and leaves no owned
// worker process) and the Quit half of W08.2 (Quit never restarts the worker).
func TestQuitLeavesNoWorkerOrPort(t *testing.T) {
	s := e2e.Begin(t, "quit-leaves-no-worker-or-port")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	check(t, "launch", sess.Launch(harness.ExplicitHome))
	st, err := sess.Start()
	check(t, "Start to READY", err)
	sess.Typed("alpha beta beta", "q", []string{"alpha", "beta"})
	if !harness.Listening(sess.Client.APIAddr) || !harness.Listening(sess.Client.DashAddr) {
		t.Fatalf("the API and dashboard must be listening before Quit")
	}
	exePID, workerPID := sess.Exe.PID(), st.Worker.PID

	code, err := sess.Quit()
	check(t, "Quit", err)
	if code != 0 {
		t.Fatalf("hachidori.exe exited with code %d after Quit; output tail:\n%s", code, sess.Exe.LogTail(4<<10))
	}
	check(t, "executable gone", harness.WaitGone("hachidori.exe", harness.BoundStop, exePID))
	check(t, "worker gone", harness.WaitGone("worker", harness.BoundStop, workerPID))
	check(t, "no owned worker process", sess.ExpectNoWorker())
	check(t, "no owned listening port", sess.ExpectPortsFree())
	s.Logf("after Quit: exit code %d, worker pid %d gone, %s and %s closed", code, workerPID, sess.Client.APIAddr, sess.Client.DashAddr)
	// Quit does not leave a worker to be restarted later: nothing reappears.
	check(t, "nothing restarts after Quit", harness.Holds(harness.ObserveNegative/2, 250*time.Millisecond, "no worker after Quit", func() (bool, string, error) {
		ws, err := sess.Workers()
		return len(ws) == 0 && !harness.Listening(sess.Client.APIAddr), fmt.Sprintf("workers %v", harness.PIDs(ws)), err
	}))
	check(t, "persisted state unchanged", sess.ExpectPersisted(before))
}

// same-home-relaunch: after Quit, a new launch with no --home finds the same
// home through the profile's bootstrap locator, opens the dashboard (not the
// first-run wizard) and reaches READY from that home.
//
// Proves W04.1 (after Quit, a new launch reaches READY from the same home
// without the wizard).
func TestSameHomeRelaunch(t *testing.T) {
	s := e2e.Begin(t, "same-home-relaunch")
	sess := newSession(t, s, "cpu")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)

	check(t, "first launch from the remembered home", sess.Launch(harness.RememberedHome))
	st1, err := sess.Start()
	check(t, "first Start", err)
	sess.Typed("alpha beta", "q", []string{"alpha", "beta"})
	code, err := sess.Quit()
	check(t, "Quit", err)
	if code != 0 {
		t.Fatalf("first run exited with code %d", code)
	}
	check(t, "first worker gone", harness.WaitGone("worker", harness.BoundStop, st1.Worker.PID))
	check(t, "no owned worker after Quit", sess.ExpectNoWorker())

	// The locator is the only thing that remembers the home, and it holds
	// nothing else.
	raw, err := os.ReadFile(sess.Profile.Locator().Path)
	check(t, "bootstrap locator", err)
	var boot map[string]any
	check(t, "bootstrap locator JSON", json.Unmarshal(raw, &boot))
	if len(boot) != 2 || boot["schema"] == nil || boot["home"] == nil {
		t.Fatalf("bootstrap.json must hold only schema and home, has %d keys", len(boot))
	}

	check(t, "relaunch with no --home", sess.Launch(harness.RememberedHome))
	w, err := sess.Client.Wizard()
	check(t, "desktop state", err)
	if w.Mode != "launch" {
		t.Fatalf("the relaunch opened in mode %q, want a normal launch (not the first-run wizard)", w.Mode)
	}
	check(t, "no worker until Start", sess.ExpectNoWorker())
	st2, err := sess.Start()
	check(t, "Start after relaunch", err)
	if st2.Worker.PID == st1.Worker.PID {
		t.Fatalf("the relaunch reused worker pid %d", st1.Worker.PID)
	}
	if st2.Runtime.Runtime != st1.Runtime.Runtime || st2.Runtime.ModelID != st1.Runtime.ModelID || st2.Runtime.Device != st1.Runtime.Device {
		t.Fatalf("the relaunch is a different runtime: %+v vs %+v", st2.Runtime, st1.Runtime)
	}
	sess.ExpectIdentity(st2)
	check(t, "one worker", sess.ExpectOneWorker(st2.Worker.PID))
	sess.Typed("beta beta alpha", "q", []string{"alpha", "beta"})
	s.Logf("relaunch from the same home: runtime %s, worker pid %d -> %d", st2.Runtime.Runtime, st1.Worker.PID, st2.Worker.PID)
	check(t, "persisted state unchanged", sess.ExpectPersisted(before))
}

// cuda-unavailable-no-cpu-fallback: a home that requests CUDA on a host with no
// usable GPU (the fixture's torch reports none) fails with the device failure,
// stays failed, answers nothing, and never runs the model on the CPU.
//
// Proves W13.2 (requesting CUDA with no usable GPU reports the failure and
// never switches to CPU). It says nothing about real CUDA hardware.
func TestCUDAUnavailableNeverFallsBackToCPU(t *testing.T) {
	s := e2e.Begin(t, "cuda-unavailable-no-cpu-fallback")
	sess := newSession(t, s, "cuda")
	before, err := sess.Fix.Persisted()
	check(t, "persisted state", err)
	check(t, "launch", sess.Launch(harness.ExplicitHome))

	failedStatus := func(what string) server.Status {
		t.Helper()
		st, err := sess.Client.WaitWorkerState(worker.StateFailed, harness.BoundReady)
		check(t, what+": runtime reports failed", err)
		lf := st.Worker.LastFailure
		if lf == nil || lf.Class != worker.ClassDevice || !strings.Contains(lf.Message, "CUDA") {
			t.Fatalf("%s: last failure %+v, want class %s naming CUDA", what, lf, worker.ClassDevice)
		}
		if st.Runtime.Device != "cuda" || len(st.Worker.Info) != 0 {
			t.Fatalf("%s: runtime device %q worker info %v: the requested device must stay cuda and no worker may report ready", what, st.Runtime.Device, st.Worker.Info)
		}
		sess.ExpectNotServing(what)
		sess.ExpectRefused(what)
		check(t, what+": no worker process", sess.ExpectNoWorker())
		w, err := sess.Client.Wizard()
		check(t, what+": application state", err)
		if w.State != "failed" || w.Failure == nil || w.Failure.Class != worker.ClassDevice {
			t.Fatalf("%s: application state %q failure %+v, want failed with %s", what, w.State, w.Failure, worker.ClassDevice)
		}
		return st
	}

	check(t, "Start posts", sess.Do("start"))
	first := failedStatus("Start")
	s.Logf("Start on %s: failed with class %s, no worker, no answer", first.Runtime.Device, first.Worker.LastFailure.Class)
	// A deterministic startup failure is not retried by itself.
	check(t, "no automatic retry", harness.Holds(harness.ObserveNegative, 200*time.Millisecond, "failed stays failed", func() (bool, string, error) {
		x, err := sess.Client.Status()
		if err != nil {
			return false, "", err
		}
		ws, err := sess.Workers()
		return x.Worker.State == worker.StateFailed && x.Worker.Starts == first.Worker.Starts && len(ws) == 0,
			fmt.Sprintf("worker %s starts %d workers %v", x.Worker.State, x.Worker.Starts, harness.PIDs(ws)), err
	}))
	// An operator Restart tries cuda again and fails the same way: no cpu.
	refusals := func() int {
		return strings.Count(string(harness.TailFile(sess.Fix.Home.Path("logs", "worker.log"), 256<<10)), "fatal device_unavailable: CUDA requested")
	}
	if n := refusals(); n != 1 {
		t.Fatalf("the worker log shows %d CUDA refusals after Start, want 1", n)
	}
	check(t, "Restart posts", sess.Do("restart"))
	// The Restart is a new attempt on the same device: a second refusal.
	check(t, "second CUDA refusal after Restart", harness.Eventually(harness.BoundReady, 200*time.Millisecond, "second CUDA refusal in the worker log", func() (bool, string, error) {
		return refusals() >= 2, fmt.Sprintf("%d refusals", refusals()), nil
	}))
	failedStatus("Restart")
	if n := refusals(); n != 2 {
		t.Fatalf("the worker log shows %d CUDA refusals, want exactly 2 (Start, Restart) and no retry", n)
	}
	check(t, "persisted state unchanged (still cuda)", sess.ExpectPersisted(before))
}
