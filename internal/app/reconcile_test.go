package app

import (
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/setup"
)

// reconcileEnv is a Controller over fakes whose active dependency runtime is
// stale until the fake reconciliation succeeds.
type reconcileEnv struct {
	*env
	stale      atomic.Bool
	reconciles atomic.Int32
	failWith   atomic.Pointer[error]
	order      chan string
}

func newReconcileEnv(t *testing.T) *reconcileEnv {
	t.Helper()
	r := &reconcileEnv{order: make(chan string, 16)}
	r.stale.Store(true)
	r.env = &env{rt: newFakeRT(), setup: &fakeSetup{}}
	r.installed.Store(true)
	r.c = New(Config{
		Home:      "/h",
		Installed: func(string) bool { return true },
		Open: func(string) (Runtime, error) {
			r.opens.Add(1)
			r.order <- "open"
			return r.rt, nil
		},
		Maintenance: Maintenance{
			Assess: func(string) (setup.Reconciliation, bool) {
				if r.stale.Load() {
					return setup.Reconciliation{Device: "cuda", State: setup.CompatStale, ActiveRuntime: "cu128-old", RequiredEnvironment: "cu128-new"}, true
				}
				return setup.Reconciliation{Device: "cuda", State: setup.CompatCurrent, ActiveRuntime: "cu128-new", RequiredEnvironment: "cu128-new"}, true
			},
			Reconcile: func(string, io.Writer, *setup.Observer) (setup.Reconciliation, bool, error) {
				r.reconciles.Add(1)
				r.order <- "reconcile"
				if e := r.failWith.Load(); e != nil {
					return setup.Reconciliation{}, false, *e
				}
				r.stale.Store(false)
				return setup.Reconciliation{State: setup.CompatCurrent}, true, nil
			},
		},
	})
	t.Cleanup(func() { _ = r.c.Close(t.Context()) })
	return r
}

func (r *reconcileEnv) next(t *testing.T) string {
	t.Helper()
	select {
	case s := <-r.order:
		return s
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for the next step")
		return ""
	}
}

// Opening Hachidori with a stale dependency runtime reconciles it first, with
// no operator action, and only then binds the activation the reconciliation
// produced.
func TestBindReconcilesAStaleRuntimeBeforeBinding(t *testing.T) {
	r := newReconcileEnv(t)
	if err := r.c.Bind(); err != nil {
		t.Fatalf("Bind = %v", err)
	}
	if got := r.next(t); got != "reconcile" {
		t.Fatalf("first step %q: the stale runtime must be reconciled before it is bound", got)
	}
	if got := r.next(t); got != "open" {
		t.Fatalf("second step %q", got)
	}
	waitFor(t, "bound", func() bool { return r.c.Snapshot().Status != nil && r.c.Snapshot().Operation == nil })
	if r.reconciles.Load() != 1 || r.rt.Running() {
		t.Fatalf("reconciles %d, running %v: Bind reconciles once and does not start the worker", r.reconciles.Load(), r.rt.Running())
	}
}

// Start does the same and then starts the worker on the reconciled runtime.
func TestStartReconcilesAStaleRuntimeThenStarts(t *testing.T) {
	r := newReconcileEnv(t)
	if err := r.c.Start(); err != nil {
		t.Fatalf("Start = %v", err)
	}
	if got := r.next(t); got != "reconcile" {
		t.Fatalf("first step %q", got)
	}
	waitFor(t, "worker started", func() bool { return r.rt.Running() })
	if r.reconciles.Load() != 1 || r.opens.Load() != 1 {
		t.Fatalf("reconciles %d opens %d", r.reconciles.Load(), r.opens.Load())
	}
}

// While it runs the reconciliation is an observable operation with phases, and
// a failure is explicit: the application reports it, nothing is bound, and the
// next attempt reconciles again.
func TestFailedReconciliationIsExplicitAndRetried(t *testing.T) {
	r := newReconcileEnv(t)
	boom := errors.New("uv: network is unreachable")
	r.failWith.Store(&boom)
	if err := r.c.Bind(); err != nil {
		t.Fatalf("Bind = %v", err)
	}
	r.next(t)
	waitFor(t, "failure", func() bool { s := r.c.Snapshot(); return s.Operation == nil && s.State == Failed })
	s := r.c.Snapshot()
	if s.Failure == nil || s.Failure.Source != SourceSetup || s.Failure.Message != boom.Error() || s.Last == nil || s.Last.Kind != OpReconcile {
		t.Fatalf("snapshot failure %+v last %+v", s.Failure, s.Last)
	}
	if r.opens.Load() != 0 {
		t.Fatal("a runtime that could not be reconciled was bound")
	}

	r.failWith.Store(nil)
	if err := r.c.Bind(); err != nil {
		t.Fatalf("retry Bind = %v", err)
	}
	r.next(t)
	if got := r.next(t); got != "open" {
		t.Fatalf("after the retry %q", got)
	}
	if r.reconciles.Load() != 2 {
		t.Fatalf("reconciles %d", r.reconciles.Load())
	}
}

// A current or equivalent runtime is not reconciled: starting is just starting.
func TestCurrentRuntimeIsNotReconciled(t *testing.T) {
	r := newReconcileEnv(t)
	r.stale.Store(false)
	if err := r.c.Start(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "worker started", func() bool { return r.rt.Running() })
	if r.reconciles.Load() != 0 {
		t.Fatalf("a current runtime was reconciled %d times", r.reconciles.Load())
	}
}

func TestReconcilePlanAndKind(t *testing.T) {
	if p := plan(OpReconcile, ""); len(p) == 0 || p[0] != string(setup.PhasePreparing) || p[len(p)-1] != string(setup.PhaseActivation) {
		t.Fatalf("plan %v", p)
	}
	if isMaintenance(OpReconcile) {
		t.Fatal("a failed reconciliation must be the application's failure, not a background maintenance result")
	}
	if st, _ := project("/h", OpReconcile, false, nil, nil, true); st != Installing {
		t.Fatalf("state during reconciliation %q", st)
	}
}
