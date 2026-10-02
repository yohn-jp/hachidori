package app

import (
	"context"
	"errors"
	"io"
	"slices"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
)

// variantEnv is a maintEnv whose forge operations are fakes that report the
// phases the real ones enter.
type variantEnv struct {
	*maintEnv
	mu      sync.Mutex
	seen    []string
	refuse  error
	changed bool
}

func newVariantEnv(t *testing.T) *variantEnv {
	t.Helper()
	e := &variantEnv{maintEnv: newMaintEnv(t), changed: true}
	rec := func(s string) { e.mu.Lock(); e.seen = append(e.seen, s); e.mu.Unlock() }
	e.c.cfg.Maintenance.ActivateVariant = func(root, device, model, variant string, experimental bool, log io.Writer, obs *setup.Observer) (bool, error) {
		rec("activate-variant " + device + " " + model + " " + variant)
		if experimental {
			rec("experimental")
		}
		for _, p := range []setup.Phase{setup.PhaseRuntime, setup.PhaseModel, setup.PhaseVariant} {
			obs.OnPhase(p)
		}
		if e.refuse != nil {
			return false, e.refuse
		}
		obs.OnPhase(setup.PhaseActivation)
		return e.changed, nil
	}
	e.c.cfg.Maintenance.Optimize = func(ctx context.Context, root, model, recipe string, log io.Writer, obs *setup.Observer) error {
		rec("optimize " + model + " " + recipe)
		for _, p := range []setup.Phase{setup.PhaseModel, setup.PhaseStarting, setup.PhaseLoadingSource, setup.PhaseResolving, setup.PhaseQuantizing} {
			obs.OnPhase(p)
		}
		obs.OnProgress(setup.Progress{Step: setup.StepMaterialize, Detail: "25 Linear modules"}) // indeterminate: no total
		if e.refuse != nil {
			return e.refuse
		}
		obs.OnPhase(setup.PhaseSerializing)
		obs.OnPhase(setup.PhaseVerifying)
		obs.OnProgress(setup.Progress{Step: setup.StepVerify, Detail: "model.safetensors", Done: 5, Total: 10, Item: 1, Items: 2})
		obs.OnPhase(setup.PhasePublish)
		return nil
	}
	e.c.cfg.Maintenance.Certify = func(root string, p CertifyParams, log io.Writer, obs *setup.Observer) error {
		rec("certify " + p.Variant + " " + p.Reference + " " + p.Candidate + " " + p.Policy)
		obs.OnPhase(setup.PhaseLoadingRuns)
		obs.OnPhase(setup.PhaseComparing)
		if e.refuse != nil {
			return e.refuse
		}
		obs.OnPhase(setup.PhaseRecording)
		return nil
	}
	return e
}

func (e *variantEnv) calls() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.seen...)
}

// Optimization and certification are controller operations on the existing
// operation/progress model: accepted, their real phases in order, indeterminate
// steps without a total and a determinate one with it, and a failure with the
// phase it happened in. A failed one never changes the application state.
func TestOptimizeAndCertifyAreOperations(t *testing.T) {
	e := newVariantEnv(t)
	if err := e.c.Optimize(setup.ClefFlash, "recipe-1"); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	m := s.Maintenance
	if m == nil || m.Kind != OpOptimize || m.Failure != nil || m.Model != setup.ClefFlash || m.Target != "recipe recipe-1" {
		t.Fatalf("optimize outcome %+v", m)
	}
	want := []string{"model", "starting", "loading_source", "resolving", "quantizing", "serializing", "verifying", "publish"}
	if !slices.Equal(m.Phases, want) {
		t.Fatalf("phases %v, want %v", m.Phases, want)
	}
	for _, p := range m.Phases {
		if !slices.Contains(m.Plan, p) {
			t.Errorf("phase %s is not in the plan %v", p, m.Plan)
		}
	}
	if m.Cancellable {
		t.Fatal("optimization claims to be cancellable")
	}

	if err := e.c.Certify(CertifyParams{Variant: "v1", Reference: "ref.json", Candidate: "cand.json"}); err != nil {
		t.Fatal(err)
	}
	m = waitIdle(t, e.c).Maintenance
	if m.Kind != OpCertify || m.Failure != nil || !slices.Equal(m.Phases, []string{"loading_runs", "comparing", "recording"}) || m.Target != "variant v1" {
		t.Fatalf("certify outcome %+v", m)
	}

	// A failed operation reports the phase it failed in and nothing else changes.
	e.refuse = errors.New("source is not materialized")
	if err := e.c.Optimize(setup.ClefFlash, "recipe-1"); err != nil {
		t.Fatal(err)
	}
	s = waitIdle(t, e.c)
	if s.Maintenance.Failure == nil || s.Maintenance.Failure.Phase != "quantizing" || s.State == Failed {
		t.Fatalf("failure %+v, state %s", s.Maintenance.Failure, s.State)
	}
	if err := e.c.Certify(CertifyParams{Variant: "v1"}); err != nil {
		t.Fatal(err)
	}
	if f := waitIdle(t, e.c).Maintenance.Failure; f == nil || f.Phase != "comparing" {
		t.Fatalf("certify failure %+v", f)
	}
}

// Optimizing runs beside a running worker without touching it, and another
// action is refused while it runs (one action at a time).
func TestOptimizeBesideRunningWorker(t *testing.T) {
	e := newVariantEnv(t)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	rt := e.runtimes()[0]
	if err := e.c.Optimize(setup.ClefFlash, "r"); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, e.c)
	if st, sp, rs := rt.counts(); st != 1 || sp != 0 || rs != 0 || !rt.Running() || len(e.runtimes()) != 1 {
		t.Fatalf("optimization touched the worker: %d %d %d", st, sp, rs)
	}
	if s := e.c.Snapshot(); s.RestartRequired {
		t.Fatal("optimization asked for a restart")
	}
}

// Variant activation is the existing activation operation with a variant
// phase: a changed activation under a running worker is restart-required, a
// refusal leaves the application untouched and reports the phase, and the
// experimental request is passed through explicitly and never implied.
func TestActivateVariantOperation(t *testing.T) {
	e := newVariantEnv(t)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	old := e.runtimes()[0]
	if err := e.c.ActivateVariant(SetupParams{Device: "cuda", Model: setup.ClefFlash}, "", false); err == nil {
		t.Fatal("a variant activation without a variant was accepted")
	}
	if err := e.c.ActivateVariant(SetupParams{Model: setup.ClefFlash}, "v1", false); err == nil {
		t.Fatal("a variant activation without an explicit device was accepted")
	}
	if err := e.c.ActivateVariant(SetupParams{Device: "cuda", Model: setup.ClefFlash}, "v1", false); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if !s.RestartRequired || s.Maintenance == nil || s.Maintenance.Kind != OpActivate || s.Maintenance.Target != "variant v1" ||
		!slices.Equal(s.Maintenance.Phases, []string{"runtime", "model", "variant", "activation"}) {
		t.Fatalf("snapshot %+v maint %+v", s.RestartRequired, s.Maintenance)
	}
	if !slices.Equal(s.Maintenance.Plan, []string{"runtime", "model", "variant", "activation"}) {
		t.Fatalf("plan %v", s.Maintenance.Plan)
	}
	if slices.Contains(e.calls(), "experimental") {
		t.Fatal("an experimental request appeared without being asked for")
	}
	if st, sp, rs := old.counts(); st != 1 || sp != 0 || rs != 0 || !old.Running() {
		t.Fatalf("activation touched the worker: %d %d %d", st, sp, rs)
	}

	e.refuse = errors.New("variant is not certified")
	if err := e.c.ActivateVariant(SetupParams{Device: "cuda", Model: setup.ClefFlash}, "v2", true); err != nil {
		t.Fatal(err)
	}
	s = waitIdle(t, e.c)
	if !slices.Contains(e.calls(), "experimental") || s.Maintenance.Failure == nil || s.Maintenance.Failure.Phase != "variant" || s.State == Failed {
		t.Fatalf("refusal %+v state %s calls %v", s.Maintenance.Failure, s.State, e.calls())
	}
	if err := e.c.Restart(); err != nil {
		t.Fatal(err)
	}
	if got := e.c.Snapshot(); got.RestartRequired && len(e.runtimes()) != 2 {
		t.Fatal("restart did not apply the activation")
	}
}

// The phase plans of the new operations are fixed and name only phases the
// operations really enter.
func TestVariantPlans(t *testing.T) {
	want := map[string][]string{
		OpOptimize: {"model", "preflight", "preparing", "runtime", "starting", "loading_source", "resolving", "quantizing", "serializing", "verifying", "publish"},
		OpCertify:  {"loading_runs", "comparing", "recording"},
	}
	for kind, w := range want {
		if got := plan(kind, ""); !slices.Equal(got, w) {
			t.Errorf("plan(%s) = %v, want %v", kind, got, w)
		}
	}
	if got := plan(OpVerify, "variant v"); !slices.Equal(got, []string{"variant"}) {
		t.Errorf("verify variant plan %v", got)
	}
	if got := plan(OpRemove, "variant v"); !slices.Equal(got, []string{"variant"}) {
		t.Errorf("remove variant plan %v", got)
	}
	if got := plan(OpActivate, ""); !slices.Equal(got, []string{"runtime", "model", "activation"}) {
		t.Errorf("source activation plan changed: %v", got)
	}
	if !isMaintenance(OpOptimize) || !isMaintenance(OpCertify) {
		t.Error("forge operations are not maintenance operations (their failure would fail the application)")
	}
}
