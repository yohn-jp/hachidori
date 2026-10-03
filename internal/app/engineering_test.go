package app

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// ---- desktop lifetime != serving lifetime ----

// Binding the runtime does not start the worker. The state it leaves is the
// stable stopped state, not a failure, and the explicit Start and Stop remain
// the operator's actions on it.
func TestBindDoesNotStartServingAndStartStopStillWork(t *testing.T) {
	e := newEnv(t, t.TempDir(), true)
	if err := e.c.Bind(); err != nil {
		t.Fatal(err)
	}
	if starts, _, _ := e.rt.counts(); starts != 0 || e.rt.Running() {
		t.Fatalf("Bind started the worker: starts %d running %v", starts, e.rt.Running())
	}
	s := e.c.Snapshot()
	if s.State != Installed || s.Failure != nil || s.Recovery != nil || !s.OperatorStopped || s.Status == nil || s.Paused != nil {
		t.Fatalf("a bound, stopped runtime: %+v", s)
	}
	if err := e.c.Bind(); err != nil || e.opens.Load() != 1 {
		t.Fatalf("a second Bind must not rebind: err %v opens %d", err, e.opens.Load())
	}
	if err := e.c.Stop(); err != nil {
		t.Fatalf("Stop of a stopped runtime is a no-op: %v", err)
	}

	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	e.rt.set(true, worker.StateReady, "ready", nil)
	if s := e.c.Snapshot(); s.State != Ready || s.OperatorStopped {
		t.Fatalf("after Start: %+v", s)
	}
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	if s := e.c.Snapshot(); s.State != Installed || !s.OperatorStopped || e.rt.Running() {
		t.Fatalf("after Stop: %+v", s)
	}
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if starts, _, _ := e.rt.counts(); starts != 2 {
		t.Fatalf("explicit Start twice: %d starts", starts)
	}
}

// A runtime that cannot be bound is a failure the operator sees, exactly as a
// failed Start is, and a Start afterwards binds it.
func TestBindFailureIsRecordedAndNotInstalledIsRefused(t *testing.T) {
	e := newEnv(t, t.TempDir(), true)
	e.openErr = errors.New("activation record refused")
	if err := e.c.Bind(); err == nil {
		t.Fatal("a refused binding succeeded")
	}
	if s := e.c.Snapshot(); s.State != Failed || s.Failure == nil || s.Failure.Source != SourceRuntime {
		t.Fatalf("snapshot %+v", s)
	}
	n := newEnv(t, t.TempDir(), false)
	if err := n.c.Bind(); !errors.Is(err, ErrNotInstalled) || n.opens.Load() != 0 {
		t.Fatalf("not installed: %v opens %d", err, n.opens.Load())
	}
}

// ---- the model-engineering transaction ----

// started is how often each resident's worker has been started: the proof that
// serving was brought back exactly when it should be and not in between.
func started(cc *certController) map[string]int {
	out := map[string]int{}
	for m, r := range residentsNow(cc.set) {
		out[m] = r.Status.Worker.Starts
	}
	return out
}

func allDown(t *testing.T, point string, got map[string]ResidentStatus) {
	t.Helper()
	if len(got) == 0 {
		t.Fatalf("%s was never observed", point)
	}
	for m, r := range got {
		if r.Running || r.Status.Worker.State == worker.StateReady {
			t.Fatalf("%s: resident %s was serving during the transaction: %+v", point, m, r.Status.Worker)
		}
	}
}

// READY -> composed Forge operation -> READY. Serving is down from the start of
// the ownership through the build, the probe, the cpu reference and the
// candidate, is never brought back in between (nested phases restore nothing),
// is projected as an intentional pause, and comes back exactly once.
func TestBuildAndEvaluateOwnsTheAcceleratorForTheWholeTransaction(t *testing.T) {
	cc := newCertController(t, nil)
	cc.e.mode = map[string]string{}
	before := started(cc)
	pre := cc.c.Snapshot()
	if pre.State != Ready || pre.Paused != nil {
		t.Fatalf("serving before: %+v", pre)
	}
	startsAt := map[string]map[string]int{}
	for _, point := range []string{"build", "probe", eval.ForgeTargetSource, eval.ForgeTargetVariant} {
		point := point
		cc.at[point] = func() {
			startsAt[point] = started(cc)
			// The operator cannot bring serving back in the middle of it.
			if err := cc.c.Start(); !errors.Is(err, ErrBusy) {
				t.Errorf("%s: Start during the transaction = %v, want ErrBusy", point, err)
			}
		}
	}
	if err := cc.c.BuildAndEvaluate(buildEvaluateParams(cc.e)); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, cc.c)
	if m := s.Maintenance; m == nil || m.Kind != OpForgeBuildEvaluate || m.Failure != nil {
		t.Fatalf("maintenance %+v", m)
	}
	for _, point := range []string{"build", "probe", eval.ForgeTargetSource, eval.ForgeTargetVariant} {
		allDown(t, point, cc.during[point])
		if !equalCounts(startsAt[point], before) {
			t.Fatalf("%s: workers were started again in the middle of the transaction: %v then %v", point, before, startsAt[point])
		}
		snap := cc.snaps[point]
		if snap.State != Paused || snap.Paused == nil || snap.Paused.Owner != OpForgeBuildEvaluate || len(snap.Paused.Residents) != 2 ||
			snap.OperatorStopped || snap.Failure != nil || snap.Recovery != nil {
			t.Fatalf("%s: the pause is not projected as intentional: %+v", point, snap)
		}
	}
	bothReady(t, cc.set)
	after := started(cc)
	for m, n := range before {
		if after[m] != n+1 {
			t.Fatalf("resident %s restored %d times, want exactly once (%v then %v)", m, after[m]-n, before, after)
		}
	}
	if s.State != Ready || s.Paused != nil || s.OperatorStopped {
		t.Fatalf("after the transaction: %+v", s)
	}
	cc.assertUntouched(t)
}

func equalCounts(a, b map[string]int) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

// A transaction that fails still restores what it stopped, once.
func TestBuildAndEvaluateFailureRestoresServingThatWasRunning(t *testing.T) {
	cc := newCertController(t, func(e *certEnv) { e.mode[eval.ForgeTargetVariant] = "unquantized" })
	before := started(cc)
	if err := cc.c.BuildAndEvaluate(buildEvaluateParams(cc.e)); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, cc.c)
	if f := s.Maintenance.Failure; f == nil || f.Phase != string(CertPhaseCandidate) {
		t.Fatalf("failure %+v", s.Maintenance.Failure)
	}
	bothReady(t, cc.set)
	for m, n := range started(cc) {
		if n != before[m]+1 {
			t.Fatalf("resident %s: %d starts, want %d", m, n, before[m]+1)
		}
	}
	if s.State != Ready || s.Paused != nil {
		t.Fatalf("snapshot %+v", s)
	}
	cc.assertUntouched(t)
}

// A request that does not resolve never touches serving.
func TestBuildAndEvaluateThatDoesNotResolveLeavesServingAlone(t *testing.T) {
	cc := newCertController(t, nil)
	before := started(cc)
	p := buildEvaluateParams(cc.e)
	p.Dataset = "/missing/dataset.jsonl"
	if err := cc.c.BuildAndEvaluate(p); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, cc.c)
	if s.Maintenance.Failure == nil {
		t.Fatal("an unresolvable request succeeded")
	}
	if !equalCounts(started(cc), before) || s.State != Ready {
		t.Fatalf("serving was disturbed by a request that never reached the GPU: %v then %v, %+v", before, started(cc), s)
	}
}

// OPERATOR STOPPED -> operation -> still STOPPED, on success and on failure. The
// transaction stops nothing (nothing runs), so it restores nothing, never
// projects a pause, and the operator's intent is intact throughout.
func TestOperatorStoppedServingStaysStoppedAcrossModelEngineering(t *testing.T) {
	for _, tc := range []struct {
		name string
		mut  func(*certEnv)
		fail bool
	}{
		{"success", nil, false},
		{"failure", func(e *certEnv) { e.mode[eval.ForgeTargetVariant] = "unquantized" }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cc := newCertController(t, tc.mut)
			if err := cc.c.Stop(); err != nil {
				t.Fatal(err)
			}
			if s := cc.c.Snapshot(); !s.OperatorStopped || s.State != Installed {
				t.Fatalf("operator stop: %+v", s)
			}
			before := started(cc)
			if err := cc.c.BuildAndEvaluate(buildEvaluateParams(cc.e)); err != nil {
				t.Fatal(err)
			}
			s := waitIdle(t, cc.c)
			if (s.Maintenance.Failure != nil) != tc.fail {
				t.Fatalf("outcome %+v", s.Maintenance)
			}
			for _, point := range []string{"build", "probe", eval.ForgeTargetSource} {
				if snap := cc.snaps[point]; snap.Paused != nil || snap.State == Paused || !snap.OperatorStopped {
					t.Fatalf("%s: an operator stop was projected as something else: %+v", point, snap)
				}
			}
			if after := started(cc); !equalCounts(after, before) {
				t.Fatalf("serving was started after the operator stopped it: %v then %v", before, after)
			}
			for m, r := range residentsNow(cc.set) {
				if r.Running {
					t.Fatalf("resident %s is running", m)
				}
			}
			if s := cc.c.Snapshot(); !s.OperatorStopped || s.State != Installed || s.Paused != nil {
				t.Fatalf("after: %+v", s)
			}
		})
	}
}

// The standalone execution and the certification are transactions of the same
// kind: a certification keeps serving down across the probe and both runs and
// restores once.
func TestCertifyVariantRestoresServingOnceAfterTheWholeTransaction(t *testing.T) {
	cc := newCertController(t, nil)
	before := started(cc)
	at := map[string]map[string]int{}
	for _, point := range []string{"probe", eval.ForgeTargetSource, eval.ForgeTargetVariant} {
		point := point
		cc.at[point] = func() { at[point] = started(cc) }
	}
	if err := cc.c.CertifyVariant(cc.e.params); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, cc.c)
	for point, got := range at {
		if !equalCounts(got, before) {
			t.Fatalf("%s: serving was restored mid-transaction: %v then %v", point, before, got)
		}
		if p := cc.snaps[point].Paused; p == nil || p.Owner != OpForgeCertify {
			t.Fatalf("%s: pause %+v", point, p)
		}
	}
	bothReady(t, cc.set)
	for m, n := range started(cc) {
		if n != before[m]+1 {
			t.Fatalf("resident %s: %d starts, want %d", m, n, before[m]+1)
		}
	}
}

// A standalone session that fails to bring serving back says so beside its own
// failure, and the pause is over once the restore has been attempted.
func TestStandaloneExecuteReleasesTheOwnershipOnEveryExit(t *testing.T) {
	var e *leaseEnv
	e = newLeaseEnv(t, "cuda", func(context.Context) (ExecutionResult, error) {
		if s := e.c.Snapshot(); s.State != Paused || s.Paused == nil || s.Paused.Owner != OpExecute {
			t.Errorf("during the session: %+v", s)
		}
		return ExecutionResult{}, errors.New("the evaluation failed")
	})
	_, err := e.c.Execute(context.Background(), ExecuteParams{Target: sourceTarget("cuda", "float32")})
	if err == nil || err.Error() != "the evaluation failed" {
		t.Fatalf("err %v", err)
	}
	bothReady(t, e.set)
	if s := e.c.Snapshot(); s.Paused != nil || s.State != Ready || e.c.holding() != nil {
		t.Fatalf("the ownership outlived the session: %+v", s)
	}
}

// A probe on the accelerator is model-engineering work too.
func TestProbeOnTheAcceleratorStopsServingAndRestoresIt(t *testing.T) {
	var during Snapshot
	e := newLeaseEnv(t, "cuda", func(context.Context) (ExecutionResult, error) { return ExecutionResult{}, nil })
	e.c.cfg.Maintenance.Probe = func(context.Context, string, ProbeParams, io.Writer, *setup.Observer) (ProbeRecord, error) {
		during = e.c.Snapshot()
		return ProbeRecord{Result: ProbePassed}, nil
	}
	if err := e.c.Probe(ProbeParams{Variant: "v", Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, e.c)
	if during.State != Paused || during.Paused == nil || during.Paused.Owner != OpProbe {
		t.Fatalf("during the probe: %+v", during)
	}
	bothReady(t, e.set)
	if s := e.c.Snapshot(); s.State != Ready || s.Paused != nil {
		t.Fatalf("after: %+v", s)
	}
}

// ---- device capacity without a model ----

// With serving stopped the device is observed by the runtime's torch (no worker,
// no model load); the observation is the device's capacity and nothing else.
func TestAcceleratorIsObservedWithoutServing(t *testing.T) {
	e := newEnv(t, t.TempDir(), true)
	probes := make(chan struct{}, 4)
	e.c.cfg.Accelerator = func(context.Context, string) (setup.AcceleratorFacts, error) {
		probes <- struct{}{}
		return setup.AcceleratorFacts{CUDAAvailable: true, DeviceName: "RTX 3060", VRAMTotal: 12 << 30, VRAMFree: 11 << 30}, nil
	}
	if got := e.c.Accelerator(); !got.Pending && got.TotalBytes == 0 {
		t.Fatalf("first read %+v", got)
	}
	var got AcceleratorObservation
	waitFor(t, "the observation", func() bool { got = e.c.Accelerator(); return !got.Pending })
	if got.Name != "RTX 3060" || got.TotalBytes != 12<<30 || got.Err != "" {
		t.Fatalf("observation %+v", got)
	}
	if starts, _, _ := e.rt.counts(); starts != 0 || e.rt.Running() || e.opens.Load() != 0 {
		t.Fatal("observing the device started or bound a runtime")
	}
	e.c.Accelerator()
	if len(probes) != 1 {
		t.Fatalf("a current observation was repeated: %d probes", len(probes))
	}
}

// While a worker serves it reports the accelerator itself, and while model
// engineering owns the device nothing else touches it.
func TestAcceleratorIsNotObservedWhileServingOrEngineering(t *testing.T) {
	e := newEnv(t, t.TempDir(), true)
	calls := make(chan struct{}, 4)
	e.c.cfg.Accelerator = func(context.Context, string) (setup.AcceleratorFacts, error) {
		calls <- struct{}{}
		return setup.AcceleratorFacts{CUDAAvailable: true, VRAMTotal: 1}, nil
	}
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	e.rt.set(true, worker.StateReady, "ready", nil)
	if got := e.c.Accelerator(); !got.Pending {
		t.Fatalf("observation while serving: %+v", got)
	}
	time.Sleep(20 * time.Millisecond)
	if len(calls) != 0 {
		t.Fatal("the device was observed while a worker served")
	}
}
