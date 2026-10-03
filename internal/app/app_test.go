package app

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// The production binding is the same lifecycle authority the dashboard uses.
var _ dashboard.Lifecycle = (*WorkerBinding)(nil)

// fakeRT stands in for a worker.Lifecycle + server.StatusBody binding.
type fakeRT struct {
	mu                       sync.Mutex
	running                  bool
	state, phase             string
	fail                     *worker.FailureView
	starts, stops, restarts  int
	stopGate                 chan struct{}
	startState, startedPhase string
}

func newFakeRT() *fakeRT {
	return &fakeRT{state: worker.StateStarting, startState: worker.StateStarting}
}

func (f *fakeRT) Start() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.running {
		return false
	}
	f.running, f.state, f.phase, f.fail = true, f.startState, "spawning", nil
	f.starts++
	return true
}

func (f *fakeRT) Stop() {
	f.mu.Lock()
	gate := f.stopGate
	f.mu.Unlock()
	if gate != nil {
		<-gate
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.running {
		f.running, f.state = false, worker.StateStopped
		f.stops++
	}
}

func (f *fakeRT) Restart() {
	f.Stop()
	f.Start()
	f.mu.Lock()
	f.restarts++
	f.mu.Unlock()
}

func (f *fakeRT) Running() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.running
}

func (f *fakeRT) Status() server.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	return server.Status{Schema: "hachidori.v1", Worker: worker.Snapshot{State: f.state, Phase: f.phase,
		Ready: f.state == worker.StateReady, LastFailure: f.fail}}
}

func (f *fakeRT) counts() (starts, stops, restarts int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.starts, f.stops, f.restarts
}

// set simulates what the supervisor reports.
func (f *fakeRT) set(running bool, state, phase string, fail *worker.FailureView) {
	f.mu.Lock()
	f.running, f.state, f.phase, f.fail = running, state, phase, fail
	f.mu.Unlock()
}

// fakeSetup is a controllable setup authority.
type fakeSetup struct {
	calls    atomic.Int32
	gate     chan struct{} // released to let setup finish
	phases   []setup.Phase
	err      error
	onFinish func()
}

func (s *fakeSetup) run(root, device, model string, log io.Writer, obs *setup.Observer) error {
	s.calls.Add(1)
	for i, p := range s.phases {
		obs.OnPhase(p)
		if i == 0 && s.gate != nil {
			<-s.gate
		}
	}
	if s.gate != nil && len(s.phases) == 0 {
		<-s.gate
	}
	if s.err == nil && s.onFinish != nil {
		s.onFinish()
	}
	return s.err
}

type env struct {
	c         *Controller
	rt        *fakeRT
	setup     *fakeSetup
	installed atomic.Bool
	opens     atomic.Int32
	openErr   error
	openGate  chan struct{}
}

func newEnv(t *testing.T, root string, installed bool) *env {
	t.Helper()
	e := &env{rt: newFakeRT(), setup: &fakeSetup{}}
	e.installed.Store(installed)
	e.c = New(Config{
		Home:      root,
		Setup:     e.setup.run,
		Installed: func(string) bool { return e.installed.Load() },
		Open: func(string) (Runtime, error) {
			e.opens.Add(1)
			if e.openGate != nil {
				<-e.openGate
			}
			if e.openErr != nil {
				return nil, e.openErr
			}
			return e.rt, nil
		},
	})
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = e.c.Close(ctx)
	})
	return e
}

func (e *env) state() State { return e.c.Snapshot().State }

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func waitIdle(t *testing.T, c *Controller) Snapshot {
	t.Helper()
	var s Snapshot
	waitFor(t, "idle controller", func() bool { s = c.Snapshot(); return s.Operation == nil })
	return s
}

// 1. The projection covers every state deterministically.
func TestProjectionTable(t *testing.T) {
	st := func(state, phase string, lf *worker.FailureView) *server.Status {
		return &server.Status{Worker: worker.Snapshot{State: state, Phase: phase, LastFailure: lf}}
	}
	setupFail := &Failure{Source: SourceSetup, Message: "x"}
	cases := []struct {
		name      string
		home, op  string
		running   bool
		status    *server.Status
		last      *Failure
		installed bool
		want      State
	}{
		{"no home", "", "", false, nil, nil, false, Unconfigured},
		{"no home wins over everything", "", OpSetup, true, st(worker.StateReady, "", nil), setupFail, true, Unconfigured},
		{"home without activation", "/h", "", false, nil, nil, false, NotInstalled},
		{"setup running", "/h", OpSetup, false, nil, nil, false, Installing},
		{"activation, no runtime bound", "/h", "", false, nil, nil, true, Installed},
		{"bound and stopped", "/h", "", false, st(worker.StateStopped, "", nil), nil, true, Installed},
		{"start in flight", "/h", OpStart, false, nil, nil, true, Starting},
		{"spawning", "/h", "", true, st(worker.StateStarting, "spawning", nil), nil, true, Starting},
		{"loading", "/h", "", true, st(worker.StateStarting, "loading", nil), nil, true, Starting},
		{"warming observed", "/h", "", true, st(worker.StateStarting, "warming", nil), nil, true, Warming},
		{"supervisor restarting", "/h", "", true, st(worker.StateRestarting, "warming", nil), nil, true, Starting},
		{"ready", "/h", "", true, st(worker.StateReady, "ready", nil), nil, true, Ready},
		{"stop in flight", "/h", OpStop, true, st(worker.StateReady, "ready", nil), nil, true, Stopping},
		{"worker failed, loop returning", "/h", "", true, st(worker.StateFailed, "", &worker.FailureView{Class: "model_load"}), nil, true, Failed},
		{"supervisor gave up", "/h", "", false, st(worker.StateFailed, "", &worker.FailureView{Class: "crash"}), nil, true, Failed},
		{"last action failed", "/h", "", false, nil, setupFail, false, Failed},
	}
	for _, c := range cases {
		got, f := project(c.home, c.op, c.running, c.status, c.last, c.installed)
		if got != c.want {
			t.Errorf("%s: state %s, want %s", c.name, got, c.want)
		}
		if (got == Failed) != (f != nil) {
			t.Errorf("%s: failure %v must be set iff state is failed", c.name, f)
		}
	}
}

// 1. Projections through the controller itself.
func TestControllerProjections(t *testing.T) {
	if s := newEnv(t, "", false).c.Snapshot(); s.State != Unconfigured || s.Home != "" {
		t.Fatalf("unresolved home: %+v", s)
	}
	e := newEnv(t, "/h", false)
	if e.state() != NotInstalled {
		t.Fatalf("state %s", e.state())
	}
	if err := e.c.Start(); !errors.Is(err, ErrNotInstalled) {
		t.Fatalf("start without activation: %v", err)
	}
	if e.opens.Load() != 0 || e.state() != NotInstalled {
		t.Fatal("rejected start changed something")
	}
	e.installed.Store(true)
	if e.state() != Installed {
		t.Fatalf("state %s", e.state())
	}
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if e.state() != Starting {
		t.Fatalf("state %s after start", e.state())
	}
	e.rt.set(true, worker.StateStarting, "warming", nil)
	if e.state() != Warming {
		t.Fatalf("state %s while warming", e.state())
	}
	e.rt.set(true, worker.StateReady, "ready", nil)
	s := e.c.Snapshot()
	if s.State != Ready || s.Status == nil || !s.Status.Worker.Ready || s.Failure != nil {
		t.Fatalf("ready snapshot %+v", s)
	}
	if s.Last == nil || s.Last.Kind != OpStart || s.Last.Failure != nil || s.Last.Finished.IsZero() {
		t.Fatalf("last operation %+v", s.Last)
	}
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	if e.state() != Installed {
		t.Fatalf("state %s after stop", e.state())
	}
}

// 2. Duplicate Start (sequential, concurrent, and while starting) never
// binds or starts a second runtime.
func TestDuplicateStartIsIdempotent(t *testing.T) {
	e := newEnv(t, "/h", true)
	e.openGate = make(chan struct{})
	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for range 32 {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- e.c.Start() }()
	}
	waitFor(t, "open", func() bool { return e.opens.Load() == 1 })
	if e.state() != Starting {
		t.Fatalf("state %s while open is in flight", e.state())
	}
	close(e.openGate)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if starts, _, _ := e.rt.counts(); e.opens.Load() != 1 || starts != 1 {
		t.Fatalf("opens=%d starts=%d, want 1/1", e.opens.Load(), starts)
	}
}

// 3. Duplicate setup is rejected deterministically; exactly one runs.
func TestDuplicateSetupRejected(t *testing.T) {
	e := newEnv(t, "/h", false)
	e.setup.gate = make(chan struct{})
	e.setup.phases = []setup.Phase{setup.PhasePreparing}
	var ok, dup atomic.Int32
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			switch err := e.c.Setup(SetupParams{Device: "cpu"}); {
			case err == nil:
				ok.Add(1)
			case errors.Is(err, ErrSetupRunning):
				dup.Add(1)
			default:
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if ok.Load() != 1 || dup.Load() != 15 {
		t.Fatalf("accepted=%d rejected=%d", ok.Load(), dup.Load())
	}
	if e.state() != Installing {
		t.Fatalf("state %s", e.state())
	}
	close(e.setup.gate)
	waitIdle(t, e.c)
	if n := e.setup.calls.Load(); n != 1 {
		t.Fatalf("setup ran %d times", n)
	}
	if err := e.c.Setup(SetupParams{}); err == nil {
		t.Fatal("setup without explicit device accepted")
	}
}

// 4. Setup and start never mutate the active state concurrently.
func TestSetupAndStartDoNotRace(t *testing.T) {
	e := newEnv(t, "/h", true)
	e.setup.gate = make(chan struct{})
	e.setup.phases = []setup.Phase{setup.PhasePreparing, setup.PhaseActivation}
	if err := e.c.Setup(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	for name, act := range map[string]func() error{"start": e.c.Start, "restart": e.c.Restart, "stop": e.c.Stop} {
		if err := act(); !errors.Is(err, ErrBusy) {
			t.Fatalf("%s during setup: %v, want ErrBusy", name, err)
		}
	}
	if err := e.c.SetHome("/other"); !errors.Is(err, ErrBusy) {
		t.Fatalf("SetHome during setup: %v", err)
	}
	if e.opens.Load() != 0 {
		t.Fatal("runtime bound while setup was running")
	}
	close(e.setup.gate)
	waitIdle(t, e.c)

	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if err := e.c.Setup(SetupParams{Device: "cpu"}); !errors.Is(err, ErrRuntimeBusy) {
		t.Fatalf("setup under a running worker: %v", err)
	}
	if n := e.setup.calls.Load(); n != 1 {
		t.Fatalf("setup ran %d times", n)
	}
	// After a successful setup the next Start rebinds the (possibly new)
	// activation instead of reusing a stale binding.
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	e.setup.gate = nil
	if err := e.c.Setup(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, e.c)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if n := e.opens.Load(); n != 2 {
		t.Fatalf("opens=%d, want a rebind after setup", n)
	}
}

// 5. A worker failure is a failed, not-ready state carrying the worker's
// structured cause, never an inference result.
func TestWorkerFailureMapsToFailed(t *testing.T) {
	e := newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	wf := &worker.FailureView{Class: worker.ClassModelLoad, Message: "no weights", Stderr: []string{"Traceback"}}
	e.rt.set(false, worker.StateFailed, "loading", wf)
	s := e.c.Snapshot()
	if s.State != Failed || s.Status.Worker.Ready {
		t.Fatalf("snapshot %+v", s)
	}
	want := Failure{Source: SourceWorker, Class: worker.ClassModelLoad, Message: "no weights", Stderr: []string{"Traceback"}}
	if s.Failure == nil || s.Failure.Source != want.Source || s.Failure.Class != want.Class || s.Failure.Message != want.Message || len(s.Failure.Stderr) != 1 {
		t.Fatalf("failure %+v, want %+v", s.Failure, want)
	}
	b, _ := json.Marshal(s)
	for _, k := range []string{`"results"`, `"choice"`, `"confidence"`} {
		if strings.Contains(string(b), k) {
			t.Fatalf("snapshot contains semantic output field %s: %s", k, b)
		}
	}
	// Stop does not erase the cause; only a new Start/Restart does.
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	if e.state() != Failed {
		t.Fatalf("state %s after stop of a failed worker", e.state())
	}
}

// 6. Recovery and restart transitions are deterministic.
func TestRecoveryAndRestart(t *testing.T) {
	e := newEnv(t, "/h", true)
	e.openErr = errors.New("worker script does not match runtime manifest")
	err := e.c.Start()
	var f *Failure
	if !errors.As(err, &f) || f.Source != SourceRuntime {
		t.Fatalf("open failure: %v", err)
	}
	if s := e.c.Snapshot(); s.State != Failed || s.Failure.Source != SourceRuntime {
		t.Fatalf("snapshot %+v", s)
	}
	e.openErr = nil
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	e.rt.set(true, worker.StateReady, "ready", nil)
	if e.state() != Ready {
		t.Fatalf("state %s", e.state())
	}
	// Supervisor gives up; an explicit operator Start recovers on a fresh
	// binding of the same runtime (the bounded restart budget resets only by
	// this operator action).
	e.rt.set(false, worker.StateFailed, "", &worker.FailureView{Class: worker.ClassCrash, Message: "exit 7"})
	if e.state() != Failed {
		t.Fatal("not failed")
	}
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if e.state() != Starting || e.opens.Load() != 3 {
		t.Fatalf("state %s opens %d", e.state(), e.opens.Load())
	}
	e.rt.set(true, worker.StateReady, "ready", nil)
	if err := e.c.Restart(); err != nil {
		t.Fatal(err)
	}
	if _, _, restarts := e.rt.counts(); restarts != 1 || e.state() != Starting {
		t.Fatalf("restarts=%d state=%s", restarts, e.state())
	}
	// Setup failure is failed with the setup cause, then a retry recovers.
	e.c.Stop()
	e.setup.err = errors.New("runtime cpu-x: uv sync failed")
	if err := e.c.Setup(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if s.State != Failed || s.Failure.Source != SourceSetup || !strings.Contains(s.Failure.Message, "uv sync") {
		t.Fatalf("snapshot %+v", s)
	}
	e.setup.err = nil
	if err := e.c.Setup(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	if s := waitIdle(t, e.c); s.State != Installed || s.Failure != nil {
		t.Fatalf("snapshot after retry %+v", s)
	}
}

// 7. Setup progress is exactly the phases setup reported; no percentage.
func TestSetupPhasesAreReportedNotInvented(t *testing.T) {
	e := newEnv(t, "/h", false)
	e.setup.gate = make(chan struct{})
	e.setup.phases = []setup.Phase{setup.PhasePreparing, setup.PhaseRuntime, setup.PhaseModel, setup.PhaseActivation}
	e.setup.onFinish = func() { e.installed.Store(true) }
	sub, cancel := e.c.Subscribe()
	defer cancel()
	if err := e.c.Setup(SetupParams{Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	<-sub
	waitFor(t, "first phase", func() bool {
		op := e.c.Snapshot().Operation
		return op != nil && op.Phase == string(setup.PhasePreparing)
	})
	s := e.c.Snapshot()
	if s.State != Installing || len(s.Operation.Phases) != 1 || s.Operation.Device != "cuda" || s.Operation.Cancellable {
		t.Fatalf("snapshot %+v", s.Operation)
	}
	b, _ := json.Marshal(s)
	for _, k := range []string{"percent", "progress", "eta"} {
		if strings.Contains(strings.ToLower(string(b)), k) {
			t.Fatalf("snapshot invents %s: %s", k, b)
		}
	}
	close(e.setup.gate)
	s = waitIdle(t, e.c)
	want := []string{"preparing", "runtime", "model", "activation"}
	if strings.Join(s.Last.Phases, ",") != strings.Join(want, ",") || s.State != Installed {
		t.Fatalf("last %+v state %s", s.Last, s.State)
	}
}

// 8. The controller holds no process state: the application state follows
// the authority's status without any controller call.
func TestStateComesFromAuthorities(t *testing.T) {
	e := newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		state, phase string
		want         State
	}{
		{worker.StateStarting, "loading", Starting},
		{worker.StateStarting, "warming", Warming},
		{worker.StateReady, "ready", Ready},
		{worker.StateRestarting, "", Starting},
		{worker.StateReady, "ready", Ready},
	} {
		e.rt.set(true, c.state, c.phase, nil)
		s := e.c.Snapshot()
		if s.State != c.want || s.Status.Worker.State != c.state {
			t.Fatalf("worker %s/%s: state %s", c.state, c.phase, s.State)
		}
	}
	// Installation state comes from the activation authority, not a cache.
	e2 := newEnv(t, "/h", false)
	e2.installed.Store(true)
	if e2.state() != Installed {
		t.Fatal("installation state is cached")
	}
}

// 9. Close waits for owned work, is bounded, and stops the runtime once.
func TestCloseIsBounded(t *testing.T) {
	e := newEnv(t, "/h", false)
	e.setup.gate = make(chan struct{})
	if err := e.c.Setup(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	err := e.c.Close(ctx)
	cancel()
	if err == nil || !strings.Contains(err.Error(), "setup still in progress") || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("bounded close: %v", err)
	}
	for name, act := range map[string]func() error{"start": e.c.Start, "restart": e.c.Restart, "stop": e.c.Stop,
		"setup": func() error { return e.c.Setup(SetupParams{Device: "cpu"}) }, "home": func() error { return e.c.SetHome("/x") }} {
		if err := act(); !errors.Is(err, ErrClosed) {
			t.Fatalf("%s after close: %v", name, err)
		}
	}
	close(e.setup.gate)
	if err := e.c.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	e = newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	e.rt.mu.Lock()
	e.rt.stopGate = make(chan struct{})
	e.rt.mu.Unlock()
	ctx, cancel = context.WithTimeout(context.Background(), 20*time.Millisecond)
	err = e.c.Close(ctx)
	cancel()
	if err == nil || !strings.Contains(err.Error(), "runtime still stopping") {
		t.Fatalf("bounded runtime stop: %v", err)
	}
	close(e.rt.stopGate)
	if err := e.c.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "single stop", func() bool { _, stops, _ := e.rt.counts(); return stops == 1 })
	if e.rt.Running() {
		t.Fatal("runtime still running after close")
	}
}

// Stop is observable as stopping while the worker shuts down; a duplicate
// Stop is a no-op; other actions are rejected meanwhile.
func TestStoppingIsObservable(t *testing.T) {
	e := newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	e.rt.mu.Lock()
	e.rt.stopGate = make(chan struct{})
	e.rt.mu.Unlock()
	done := make(chan error)
	go func() { done <- e.c.Stop() }()
	waitFor(t, "stopping", func() bool { return e.state() == Stopping })
	if err := e.c.Stop(); err != nil {
		t.Fatalf("duplicate stop: %v", err)
	}
	if err := e.c.Start(); !errors.Is(err, ErrBusy) {
		t.Fatalf("start while stopping: %v", err)
	}
	close(e.rt.stopGate)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if _, stops, _ := e.rt.counts(); e.state() != Installed || stops != 1 {
		t.Fatalf("state %s stops %d", e.state(), stops)
	}
}

func TestSetHome(t *testing.T) {
	e := newEnv(t, "", false)
	if err := e.c.Start(); !errors.Is(err, ErrUnconfigured) {
		t.Fatal(err)
	}
	if err := e.c.Setup(SetupParams{Device: "cpu"}); !errors.Is(err, ErrUnconfigured) {
		t.Fatal(err)
	}
	if err := e.c.SetHome("/h"); err != nil {
		t.Fatal(err)
	}
	if s := e.c.Snapshot(); s.State != NotInstalled || s.Home == "" {
		t.Fatalf("snapshot %+v", s)
	}
	e.installed.Store(true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if err := e.c.SetHome(""); !errors.Is(err, ErrRuntimeBusy) {
		t.Fatalf("SetHome under running worker: %v", err)
	}
}

// ---- real worker.Supervisor / worker.Lifecycle behind the controller ----

const fakeWorkerEnv = "HACHIDORI_APP_FAKE_WORKER"

// TestMain doubles as a fake private worker speaking the worker protocol.
func TestMain(m *testing.M) {
	if mode := os.Getenv(fakeWorkerEnv); mode != "" {
		fakeWorker(mode)
		os.Exit(0)
	}
	if spec := os.Getenv(fakeResidentEnv); spec != "" {
		fakeResidentWorker(spec)
		os.Exit(0)
	}
	if spec := os.Getenv(fakeProbeEnv); spec != "" {
		fakeProbeWorker(spec)
		os.Exit(0)
	}
	if spec := os.Getenv(fakeExecEnv); spec != "" {
		fakeExecWorker(spec)
		os.Exit(0)
	}
	if spec := os.Getenv(fakeTrialEnv); spec != "" {
		fakeTrialWorker(spec)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func fakeWorker(mode string) {
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	if mode == "fatal" {
		emit(map[string]any{"event": "phase", "phase": "loading"})
		emit(map[string]any{"event": "fatal", "class": worker.ClassModelLoad, "message": "no weights"})
		os.Exit(3)
	}
	if mode == "crash" {
		// Ready, then die unexpectedly: the supervisor's bounded recovery.
		emit(map[string]any{"event": "ready", "info": map[string]any{"provider": "fake"}})
		time.Sleep(100 * time.Millisecond)
		os.Exit(9)
	}
	emit(map[string]any{"event": "phase", "phase": "warming"})
	time.Sleep(200 * time.Millisecond)
	emit(map[string]any{"event": "ready", "info": map[string]any{"provider": "fake"}})
	dec := json.NewDecoder(os.Stdin)
	for {
		var req struct {
			ID int64  `json:"id"`
			Op string `json:"op"`
		}
		if dec.Decode(&req) != nil {
			return
		}
		emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{}})
		if req.Op == "shutdown" {
			return
		}
	}
}

func realOpen(t *testing.T, ctx context.Context, mode string, opens *atomic.Int32) OpenFunc {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return func(root string) (Runtime, error) {
		opens.Add(1)
		cfg := worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeWorkerEnv + "=" + mode},
			StartTimeout: 10 * time.Second, RequestTimeout: 5 * time.Second}
		sup := worker.NewSupervisor(cfg, worker.Policy{MaxRestarts: 0, Window: time.Minute, QueueDepth: 4})
		return &WorkerBinding{Lifecycle: worker.NewLifecycle(ctx, sup), Supervisor: sup,
			Info: server.Runtime{Home: root}, Started: time.Now()}, nil
	}
}

// 5/8/9 against the real process authority: one worker, warming observed
// from the supervisor phase, ready, clean stop; a startup failure is failed
// with the worker's class.
func TestRealLifecycleAuthority(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var opens atomic.Int32
	c := New(Config{Home: t.TempDir(), Installed: func(string) bool { return true }, Open: realOpen(t, ctx, "ok", &opens)})
	for range 5 {
		if err := c.Start(); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, "warming", func() bool { return c.Snapshot().State == Warming })
	waitFor(t, "ready", func() bool { return c.Snapshot().State == Ready })
	s := c.Snapshot()
	if s.Status.Worker.Starts != 1 || s.Status.Worker.PID == 0 || opens.Load() != 1 {
		t.Fatalf("worker starts=%d pid=%d opens=%d", s.Status.Worker.Starts, s.Status.Worker.PID, opens.Load())
	}
	if err := c.Restart(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "ready after restart", func() bool { return c.Snapshot().State == Ready })
	if n := c.Snapshot().Status.Worker.Starts; n != 2 {
		t.Fatalf("starts=%d after restart", n)
	}
	cctx, ccancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer ccancel()
	if err := c.Close(cctx); err != nil {
		t.Fatal(err)
	}
	if s := c.Snapshot(); s.State != Installed || s.Status.Worker.State != worker.StateStopped {
		t.Fatalf("after close: %s / %s", s.State, s.Status.Worker.State)
	}

	c = New(Config{Home: t.TempDir(), Installed: func(string) bool { return true }, Open: realOpen(t, ctx, "fatal", &opens)})
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "failed", func() bool { return c.Snapshot().State == Failed })
	s = c.Snapshot()
	if s.Failure.Source != SourceWorker || s.Failure.Class != worker.ClassModelLoad || s.Status.Worker.Ready {
		t.Fatalf("failure %+v", s.Failure)
	}
	if err := c.Close(cctx); err != nil {
		t.Fatal(err)
	}
}

// The production OpenFunc goes through server.WorkerConfig: a home without an
// activation record is a runtime failure, not a started worker.
func TestWorkerRuntimeUsesWorkerConfig(t *testing.T) {
	var got *WorkerBinding
	open := WorkerRuntime(context.Background(), io.Discard, worker.DefaultPolicy, func(b *WorkerBinding) { got = b })
	if _, err := open(t.TempDir()); err == nil || !strings.Contains(err.Error(), "no active runtime") {
		t.Fatalf("open of empty home: %v", err)
	}
	if got != nil {
		t.Fatal("binding created without a valid activation")
	}
	c := New(Config{Home: t.TempDir(), Open: open})
	if s := c.Snapshot(); s.State != NotInstalled {
		t.Fatalf("default installation check: %s", s.State)
	}
}
