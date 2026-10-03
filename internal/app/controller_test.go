package app

import (
	"errors"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
)

func TestReconcileBindHandoffFailureIsVisibleAndRetryable(t *testing.T) {
	r := newReconcileEnv(t)
	boom := errors.New("cannot open reconciled activation")
	r.c.cfg.Open = func(string) (Runtime, error) { return nil, boom }
	if err := r.c.Bind(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "Bind failure after reconciliation", func() bool { return r.c.Snapshot().State == Failed })
	s := r.c.Snapshot()
	if s.Operation != nil || s.Last == nil || s.Last.Kind != OpBind || s.Failure == nil || s.Failure.Message != boom.Error() || s.Failure.Phase != PhasePreflight {
		t.Fatalf("handoff failure: %+v", s)
	}
	r.c.cfg.Open = func(string) (Runtime, error) { return r.rt, nil }
	if err := r.c.Bind(); err != nil {
		t.Fatal(err)
	}
	if s = r.c.Snapshot(); s.Status == nil || s.Failure != nil || r.rt.Running() || r.reconciles.Load() != 1 {
		t.Fatalf("Bind retry must reuse reconciliation without starting serving: %+v", s)
	}
}

func TestReconcilePlanIncludesConditionalVariantVerification(t *testing.T) {
	found := false
	for _, phase := range plan(OpReconcile, "") {
		if phase == string(setup.PhaseVariant) {
			found = true
		}
	}
	if !found {
		t.Fatal("variant verification must be observable")
	}
}
