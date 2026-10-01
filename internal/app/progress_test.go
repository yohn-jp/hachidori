package app

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// An operation reports the phase it is in and the progress of the work inside
// it, as the setup authority reports them; a new phase starts without the
// previous phase's step, and a total appears only when the work has one.
func TestOperationReportsPhaseAndProgress(t *testing.T) {
	e := newEnv(t, "/h", false)
	gate, step := make(chan struct{}), make(chan struct{})
	e.c.cfg.Setup = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		obs.OnPhase(setup.PhasePreparing)
		obs.OnPhase(setup.PhaseModel)
		obs.OnProgress(setup.Progress{Step: setup.StepDownload, Detail: "model.safetensors", Done: 40, Total: 100, Item: 3, Items: 7})
		<-step
		obs.OnProgress(setup.Progress{Step: setup.StepVerify, Detail: "model"})
		<-gate
		return nil
	}
	if err := e.c.Setup(SetupParams{Device: "cuda", Model: "opendecider-nano"}); err != nil {
		t.Fatal(err)
	}
	var op *Operation
	waitFor(t, "download progress", func() bool {
		op = e.c.Snapshot().Operation
		return op != nil && op.Progress != nil
	})
	if !slices.Equal(op.Plan, []string{"preparing", "runtime", "model", "activation"}) || op.Phase != "model" ||
		!slices.Equal(op.Phases, []string{"preparing", "model"}) {
		t.Fatalf("operation %+v", op)
	}
	p := op.Progress
	if p.Step != setup.StepDownload || p.Done != 40 || p.Total != 100 || p.Item != 3 || p.Items != 7 || !p.Determinate() || p.Fraction() != 0.4 {
		t.Fatalf("progress %+v", p)
	}
	close(step)
	waitFor(t, "indeterminate step", func() bool {
		op = e.c.Snapshot().Operation
		return op != nil && op.Progress != nil && op.Progress.Step == setup.StepVerify
	})
	if op.Progress.Determinate() {
		t.Fatalf("a step without a total is determinate: %+v", op.Progress)
	}
	// The snapshot is a copy: what the controller later reports is not
	// changed by a reader holding an earlier one.
	if before := e.c.Snapshot().Operation; before != nil {
		before.Progress.Done = 99
		if again := e.c.Snapshot().Operation; again.Progress.Done == 99 {
			t.Fatal("snapshot shares its progress with the controller")
		}
	}
	close(gate)
	s := waitIdle(t, e.c)
	if s.Last == nil || s.Last.Failure != nil || s.Last.Progress == nil || s.Last.Finished.IsZero() {
		t.Fatalf("finished %+v", s.Last)
	}
}

// A failure names the phase and the step it happened in, keeps both after the
// action ended, and a later action replaces them.
func TestFailureNamesPhaseAndStep(t *testing.T) {
	e := newMaintEnv(t)
	e.c.cfg.Maintenance.Materialize = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		obs.OnPhase(setup.PhasePreparing)
		obs.OnPhase(setup.PhaseRuntime)
		obs.OnProgress(setup.Progress{Step: setup.StepMaterialize, Detail: "installing the locked packages (cu128)"})
		return errors.New("uv sync: exit status 1")
	}
	if err := e.c.Materialize(SetupParams{Device: "cuda", Model: "opendecider-nano"}); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	f := s.Maintenance.Failure
	if f == nil || f.Source != SourceSetup || f.Phase != "runtime" || f.Step != "materializing" || !strings.Contains(f.Message, "uv sync") {
		t.Fatalf("failure %+v", f)
	}
	if got := f.Error(); got != "setup (runtime): uv sync: exit status 1" {
		t.Fatalf("Error() = %q", got)
	}
	if s.State != Installed || s.Failure != nil {
		t.Fatalf("a failed maintenance action changed the application: %+v", s)
	}
	// The failure stays until the next action replaces it.
	if s2 := e.c.Snapshot(); s2.Maintenance == nil || s2.Maintenance.Failure == nil {
		t.Fatal("failure not retained")
	}
	e.c.cfg.Maintenance.Materialize = func(root, device, model string, log io.Writer, obs *setup.Observer) error { return nil }
	if err := e.c.Materialize(SetupParams{Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	if s := waitIdle(t, e.c); s.Maintenance.Failure != nil {
		t.Fatalf("a successful action kept the previous failure: %+v", s.Maintenance)
	}
}

// Every maintenance action is accepted at once and runs in the background:
// the caller is never held for the duration of a hash or a deletion.
func TestMaintenanceActionsDoNotBlockTheCaller(t *testing.T) {
	e := newMaintEnv(t)
	gate := make(chan struct{})
	block := func() { <-gate }
	e.c.cfg.Maintenance.Verify = func(root, kind, id string, obs *setup.Observer) error { block(); return nil }
	e.c.cfg.Maintenance.Remove = func(root, kind, id string, obs *setup.Observer) error { block(); return nil }
	e.c.cfg.Maintenance.Activate = func(root, device, model string, log io.Writer, obs *setup.Observer) (bool, error) {
		block()
		return false, nil
	}
	for name, start := range map[string]func() error{
		OpVerify:   func() error { return e.c.Verify(setup.KindModel, "laya-base") },
		OpRemove:   func() error { return e.c.Remove(setup.KindModel, "laya-base") },
		OpActivate: func() error { return e.c.Activate(SetupParams{Device: "cpu"}) },
	} {
		if err := start(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if op := e.c.Snapshot().Operation; op == nil || op.Kind != name {
			t.Fatalf("%s is not observable while it runs: %+v", name, op)
		}
		// One action at a time, and the same action reports itself.
		if err := e.c.Verify(setup.KindRuntime, "x"); !errors.Is(err, ErrBusy) && !errors.Is(err, ErrOperationRunning) {
			t.Fatalf("%s: a second action was admitted: %v", name, err)
		}
		gate <- struct{}{}
		waitIdle(t, e.c)
	}
}

// The outcome of an explicit verification is kept per artifact by the
// controller (not by a page), survives later actions, and is dropped when the
// artifact is removed.
func TestVerifyOutcomeIsKeptPerArtifact(t *testing.T) {
	e := newMaintEnv(t)
	e.err = errors.New("config.json: sha256 bad")
	if err := e.c.Verify(setup.KindModel, "laya-base"); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, e.c)
	e.err = nil
	if err := e.c.Verify(setup.KindRuntime, "cpu-x"); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if ck := s.Checks["model laya-base"]; ck.OK || ck.Message != "config.json: sha256 bad" || ck.Time.IsZero() {
		t.Fatalf("failed check %+v", ck)
	}
	if ck := s.Checks["runtime cpu-x"]; !ck.OK || ck.Message != "" {
		t.Fatalf("passed check %+v", ck)
	}
	if err := e.c.Materialize(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	if got := waitIdle(t, e.c).Checks; len(got) != 2 {
		t.Fatalf("a later action dropped checks: %+v", got)
	}
	if err := e.c.Remove(setup.KindModel, "laya-base"); err != nil {
		t.Fatal(err)
	}
	if got := waitIdle(t, e.c).Checks; len(got) != 1 || !got["runtime cpu-x"].OK {
		t.Fatalf("checks after removal %+v", got)
	}
	// The returned map is a copy.
	s = e.c.Snapshot()
	s.Checks["runtime cpu-x"] = Check{}
	if !e.c.Snapshot().Checks["runtime cpu-x"].OK {
		t.Fatal("snapshot shares its checks with the controller")
	}
}

func TestPlanPerAction(t *testing.T) {
	for _, tc := range []struct {
		kind, target string
		want         []string
	}{
		{OpSetup, "", []string{"preparing", "runtime", "model", "activation"}},
		{OpMaterialize, "", []string{"preparing", "runtime", "model", "publish"}},
		{OpRepair, "", []string{"preparing", "runtime", "model", "publish"}},
		{OpActivate, "", []string{"runtime", "model", "activation"}},
		{OpVerify, "runtime cpu-x", []string{"runtime"}},
		{OpRemove, "model laya-base", []string{"model"}},
		{OpStart, "", nil},
	} {
		if got := plan(tc.kind, tc.target); !slices.Equal(got, tc.want) {
			t.Errorf("plan(%s, %q) = %v, want %v", tc.kind, tc.target, got, tc.want)
		}
	}
}

// An action's output goes to the home's setup log when its caller gave none,
// so a failure can be diagnosed from the file the operator is pointed to.
func TestMaintenanceOutputGoesToTheSetupLog(t *testing.T) {
	root := t.TempDir()
	e := newMaintEnv(t)
	e.c.home = root
	e.c.cfg.Maintenance.Materialize = func(r, device, model string, log io.Writer, obs *setup.Observer) error {
		io.WriteString(log, "uv: resolved 120 packages\n")
		return errors.New("boom")
	}
	if err := e.c.Materialize(SetupParams{Device: "cuda", Model: "opendecider-nano"}); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, e.c)
	b, err := os.ReadFile(SetupLogPath(root))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "materialize cuda opendecider-nano") || !strings.Contains(string(b), "uv: resolved 120 packages") {
		t.Fatalf("setup log:\n%s", b)
	}
	if SetupLogPath(root) != filepath.Join(root, "logs", "setup.log") {
		t.Fatal(SetupLogPath(root))
	}
}

// A runtime that is refused before any worker starts is a failure of the
// preflight, not of a worker phase, and keeps the cause the authority gave.
func TestRuntimeRefusedBeforeStartIsAPreflightFailure(t *testing.T) {
	e := newEnv(t, "/h", true)
	e.openErr = setup.ErrWorkerContract
	err := e.c.Start()
	var f *Failure
	if !errors.As(err, &f) || f.Source != SourceRuntime || f.Phase != PhasePreflight {
		t.Fatalf("Start = %v", err)
	}
	s := e.c.Snapshot()
	if s.State != Failed || s.Failure == nil || s.Failure.Phase != PhasePreflight || s.Failure.Message != setup.ErrWorkerContract.Error() {
		t.Fatalf("snapshot %+v failure %+v", s, s.Failure)
	}
}

// A worker failure carries the phase the worker had reported when it failed,
// so the operator can tell a model that did not load from one that did not
// warm up.
func TestWorkerFailureCarriesItsPhase(t *testing.T) {
	for _, phase := range []string{"preflight", "spawning", "importing", "loading", "warming", "ready"} {
		st := &server.Status{Worker: worker.Snapshot{State: worker.StateFailed, Phase: phase,
			LastFailure: &worker.FailureView{Class: worker.ClassModelLoad, Message: "no weights", Stderr: []string{"trace"}}}}
		state, f := project("/h", "", true, st, nil, true)
		if state != Failed || f == nil || f.Phase != phase || f.Class != worker.ClassModelLoad || len(f.Stderr) != 1 {
			t.Errorf("phase %s: %v %+v", phase, state, f)
		}
		if f.Error() != "worker ("+phase+"): model_load: no weights" {
			t.Errorf("Error() = %q", f.Error())
		}
	}
}
