package app

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Model-aware setup -> application controller: the controller passes the
// caller's catalog model ID (empty means the catalog default) straight to the
// setup authority, reports the setup phases the authority entered, and never
// runs setup beside a running worker (activation cannot change under it).
func TestControllerSetupModelSelection(t *testing.T) {
	var mu sync.Mutex
	var gotModels []string
	record := func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		mu.Lock()
		gotModels = append(gotModels, model)
		mu.Unlock()
		obs.OnPhase(setup.PhasePreparing)
		if _, err := setup.LookupModel(model); err != nil {
			return err
		}
		obs.OnPhase(setup.PhaseModel)
		obs.OnPhase(setup.PhaseActivation)
		return nil
	}

	e := newEnv(t, "/h", false)
	e.c.cfg.Setup = record
	e.installed.Store(true)

	// Default and every explicit catalog ID are accepted.
	ids := []string{""}
	for _, m := range setup.Models {
		ids = append(ids, m.ID)
	}
	for _, id := range ids {
		if err := e.c.Setup(SetupParams{Device: "cpu", Model: id}); err != nil {
			t.Fatalf("model %q rejected: %v", id, err)
		}
		s := waitIdle(t, e.c)
		if s.Last == nil || s.Last.Failure != nil || s.Last.Model != id ||
			!slices.Equal(s.Last.Phases, []string{"preparing", "model", "activation"}) {
			t.Fatalf("model %q: last %+v", id, s.Last)
		}
	}
	if !slices.Equal(gotModels, ids) {
		t.Fatalf("setup authority saw models %v, want %v", gotModels, ids)
	}
	if _, err := setup.LookupModel(""); err != nil {
		t.Fatalf("empty model must resolve to the catalog default %s: %v", setup.DefaultModel, err)
	}

	// A model outside the catalog fails as a setup failure after the real
	// phases that were entered; nothing is invented and the failure is kept.
	e.c.cfg.Setup = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		return setup.RunObserved(home.Home{Root: t.TempDir()}, device, model, log, obs)
	}
	if err := e.c.Setup(SetupParams{Device: "cpu", Model: "not-in-catalog"}); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if s.Last == nil || s.Last.Failure == nil || s.Last.Failure.Source != SourceSetup ||
		!slices.Equal(s.Last.Phases, []string{"preparing"}) {
		t.Fatalf("unsupported model: last %+v", s.Last)
	}
}

// Setup is refused while the worker runs, for default and explicit models,
// so the activation cannot change concurrently with a running worker.
func TestControllerSetupRefusedWhileWorkerRuns(t *testing.T) {
	e := newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"", setup.DefaultModel} {
		if err := e.c.Setup(SetupParams{Device: "cpu", Model: id}); !errors.Is(err, ErrRuntimeBusy) {
			t.Fatalf("model %q under a running worker: %v", id, err)
		}
	}
	if n := e.setup.calls.Load(); n != 0 {
		t.Fatalf("setup ran %d times beside a running worker", n)
	}
}

// maintEnv is a controller with fake maintenance authority and one fake
// worker binding per Open, so rebinding is observable.
type maintEnv struct {
	c       *Controller
	mu      sync.Mutex
	rts     []*fakeRT
	calls   []string
	changed bool
	err     error
	gate    chan struct{}
}

func newMaintEnv(t *testing.T) *maintEnv {
	t.Helper()
	e := &maintEnv{changed: true}
	rec := func(s string) {
		e.mu.Lock()
		e.calls = append(e.calls, s)
		e.mu.Unlock()
	}
	e.c = New(Config{
		Home:      "/h",
		Installed: func(string) bool { return true },
		Open: func(string) (Runtime, error) {
			e.mu.Lock()
			defer e.mu.Unlock()
			rt := newFakeRT()
			rt.startState = "ready"
			e.rts = append(e.rts, rt)
			return rt, nil
		},
		Maintenance: Maintenance{
			Inspect: func(root string, verify bool) setup.Inventory {
				rec("inspect " + root)
				return setup.Inventory{}
			},
			Materialize: func(root, device, model string, log io.Writer, obs *setup.Observer) error {
				rec("materialize " + device + " " + model)
				obs.OnPhase(setup.PhasePreparing)
				if e.gate != nil {
					<-e.gate
				}
				obs.OnPhase(setup.PhasePublish)
				return e.err
			},
			Repair: func(root, device, model string, log io.Writer, obs *setup.Observer) error {
				rec("repair " + device + " " + model)
				return e.err
			},
			Activate: func(root, device, model string, log io.Writer, obs *setup.Observer) (bool, error) {
				rec("activate " + device + " " + model)
				return e.changed && e.err == nil, e.err
			},
			Verify: func(root, kind, id string, obs *setup.Observer) error { rec("verify " + kind + " " + id); return e.err },
			Remove: func(root, kind, id string, obs *setup.Observer) error { rec("remove " + kind + " " + id); return e.err },
		},
	})
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = e.c.Close(ctx)
	})
	return e
}

func (e *maintEnv) runtimes() []*fakeRT {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]*fakeRT(nil), e.rts...)
}

func (e *maintEnv) called() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.calls...)
}

// Activation under a running worker reports restart-required and does not
// restart it; only the operator's Restart rebinds the new activation, and an
// artifact of the still-running worker cannot be removed in between.
func TestActivateReportsRestartRequiredWithoutRestarting(t *testing.T) {
	e := newMaintEnv(t)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	old := e.runtimes()[0]
	if err := e.c.Activate(SetupParams{Device: "cpu", Model: "laya-base"}); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if !s.RestartRequired || s.Maintenance == nil || s.Maintenance.Kind != OpActivate || s.Maintenance.Failure != nil {
		t.Fatalf("snapshot %+v", s)
	}
	if st, sp, rs := old.counts(); st != 1 || sp != 0 || rs != 0 || !old.Running() || len(e.runtimes()) != 1 {
		t.Fatalf("activation touched the worker: starts=%d stops=%d restarts=%d opens=%d", st, sp, rs, len(e.runtimes()))
	}
	// A Start beside the running worker stays a no-op; removal is refused.
	if err := e.c.Start(); err != nil || len(e.runtimes()) != 1 {
		t.Fatalf("Start: %v opens=%d", err, len(e.runtimes()))
	}
	if err := e.c.Remove(setup.KindModel, "laya-base"); !errors.Is(err, ErrRestartRequired) {
		t.Fatalf("Remove while restart required: %v", err)
	}
	if slices.Contains(e.called(), "remove model laya-base") {
		t.Fatal("removal reached the setup authority")
	}

	if err := e.c.Restart(); err != nil {
		t.Fatal(err)
	}
	rts := e.runtimes()
	if len(rts) != 2 || old.Running() || !rts[1].Running() {
		t.Fatalf("restart did not rebind: opens=%d oldRunning=%v", len(rts), old.Running())
	}
	if s := e.c.Snapshot(); s.RestartRequired {
		t.Fatal("restart-required not cleared by the restart")
	}
	if err := e.c.Remove(setup.KindModel, "laya-base"); err != nil {
		t.Fatalf("Remove after restart: %v", err)
	}
	if s := waitIdle(t, e.c); s.Maintenance == nil || s.Maintenance.Kind != OpRemove || s.Maintenance.Failure != nil {
		t.Fatalf("remove outcome %+v", s.Maintenance)
	}
}

// An unchanged activation requires no restart; with no worker running the
// next Start simply binds the new activation.
func TestActivateStoppedAndUnchanged(t *testing.T) {
	e := newMaintEnv(t)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	e.changed = false
	if err := e.c.Activate(SetupParams{Device: "cpu"}); err != nil || waitIdle(t, e.c).RestartRequired {
		t.Fatalf("unchanged activation: %v", err)
	}
	e.changed = true
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := e.c.Activate(SetupParams{Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	if waitIdle(t, e.c).RestartRequired {
		t.Fatal("restart required with no worker running")
	}
	if err := e.c.Start(); err != nil || len(e.runtimes()) != 2 {
		t.Fatalf("Start after activation: %v opens=%d", err, len(e.runtimes()))
	}
	if err := e.c.Activate(SetupParams{}); err == nil {
		t.Fatal("activation without an explicit device accepted")
	}
}

// A failed maintenance action is reported but neither fails the application
// nor requires a restart.
func TestMaintenanceFailureDoesNotFailApplication(t *testing.T) {
	e := newMaintEnv(t)
	e.err = errors.New("boom")
	if err := e.c.Activate(SetupParams{Device: "cpu"}); err != nil {
		t.Fatal(err)
	}
	if s := waitIdle(t, e.c); s.Maintenance == nil || s.Maintenance.Kind != OpActivate || s.Maintenance.Failure == nil {
		t.Fatalf("activate failure not reported: %+v", s.Maintenance)
	}
	if err := e.c.Verify(setup.KindRuntime, "cpu-x"); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if s.State != Installed || s.Failure != nil || s.Last != nil || s.RestartRequired ||
		s.Maintenance == nil || s.Maintenance.Kind != OpVerify || s.Maintenance.Failure == nil || s.Maintenance.Target != "runtime cpu-x" {
		t.Fatalf("snapshot %+v", s)
	}
}

// Materialize is explicit, async and allowed beside a running worker; it
// reports real phases, is single-flight, and never changes the activation
// (which only Activate does). Repair is refused under a running worker.
func TestMaterializeAndRepairRules(t *testing.T) {
	e := newMaintEnv(t)
	e.gate = make(chan struct{})
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if err := e.c.Materialize(SetupParams{Device: "cuda", Model: "laya-base"}); err != nil {
		t.Fatal(err)
	}
	if err := e.c.Materialize(SetupParams{Device: "cuda"}); !errors.Is(err, ErrOperationRunning) {
		t.Fatalf("second materialize: %v", err)
	}
	if err := e.c.Stop(); !errors.Is(err, ErrBusy) {
		t.Fatalf("Stop during materialize: %v", err)
	}
	if s := e.c.Snapshot(); s.State != Starting && s.State != Ready && s.State != Warming {
		t.Fatalf("worker state hidden by materialize: %s", s.State)
	}
	close(e.gate)
	s := waitIdle(t, e.c)
	if s.Maintenance == nil || s.Maintenance.Failure != nil || !slices.Equal(s.Maintenance.Phases, []string{"preparing", "publish"}) || s.RestartRequired {
		t.Fatalf("snapshot %+v", s)
	}
	if slices.ContainsFunc(e.called(), func(c string) bool { return strings.HasPrefix(c, "activate") }) {
		t.Fatal("materialize activated")
	}
	if err := e.c.Repair(SetupParams{Device: "cuda"}); !errors.Is(err, ErrRuntimeBusy) {
		t.Fatalf("Repair beside a running worker: %v", err)
	}
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := e.c.Repair(SetupParams{Device: "cuda", Model: "laya-base"}); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, e.c)
	if err := e.c.Materialize(SetupParams{}); err == nil {
		t.Fatal("materialize without an explicit device accepted")
	}
}

func TestInventoryNeedsHome(t *testing.T) {
	c := New(Config{})
	if _, err := c.Inventory(false); !errors.Is(err, ErrUnconfigured) {
		t.Fatalf("no home: %v", err)
	}
	if err := c.Remove(setup.KindModel, "laya-base"); !errors.Is(err, ErrUnconfigured) {
		t.Fatalf("remove without home: %v", err)
	}
	e := newMaintEnv(t)
	home, err := filepath.Abs("/h")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.c.Inventory(true); err != nil || !slices.Equal(e.called(), []string{"inspect " + filepath.Clean(home)}) {
		t.Fatalf("inventory: %v %v", err, e.called())
	}
}
