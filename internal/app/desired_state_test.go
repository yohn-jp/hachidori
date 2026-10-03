package app

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
)

type desiredRig struct {
	e                    *applyEnv
	store                *settings.Store
	failRuntime          bool
	repairs              int
	failRestoreResidents bool
}

func wireDesiredRig(t *testing.T, e *applyEnv) *desiredRig {
	t.Helper()
	models := slices.Clone(setup.Models)
	setup.Models = append(setup.Models, home.ModelManifest{ID: modelB, Provider: e.model.Provider,
		Repo: "fixture/" + modelB, Revision: strings.Repeat("ab", 20)})
	t.Cleanup(func() { setup.Models = models })
	r := &desiredRig{e: e, store: &settings.Store{Path: filepath.Join(t.TempDir(), "settings.json")}}
	e.c.cfg.Residents = func() []string {
		ids, _ := r.store.Residents()
		return ids
	}
	e.c.cfg.SetResidents = func(ids []string) error {
		if r.failRestoreResidents && len(ids) == 0 {
			return errors.New("resident restore refused")
		}
		if err := r.store.SetResidents(ids); err != nil {
			return err
		}
		e.extra = slices.Contains(ids, modelB)
		return nil
	}
	e.c.cfg.Maintenance.Verify = func(root, kind, id string, obs *setup.Observer) error {
		if obs != nil {
			switch kind {
			case setup.KindRuntime:
				obs.OnPhase(setup.PhaseRuntime)
			case setup.KindModel:
				obs.OnPhase(setup.PhaseModel)
			case setup.KindVariant:
				obs.OnPhase(setup.PhaseVariant)
			}
		}
		if kind == setup.KindRuntime && r.failRuntime {
			return errors.New("fixture runtime prerequisite is missing")
		}
		if kind == setup.KindVariant {
			return setup.Verify(home.Home{Root: root}, kind, id, obs)
		}
		return nil
	}
	e.c.cfg.Maintenance.Repair = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		r.repairs++
		r.failRuntime = false
		if obs != nil {
			obs.OnPhase(setup.PhaseRuntime)
			obs.OnPhase(setup.PhaseModel)
		}
		return nil
	}
	e.c.cfg.Maintenance.Activate = func(root, device, model string, log io.Writer, obs *setup.Observer) (bool, error) {
		if obs != nil {
			obs.OnPhase(setup.PhaseRuntime)
			obs.OnPhase(setup.PhaseModel)
			obs.OnPhase(setup.PhaseActivation)
		}
		spec, err := setup.Desired(device)
		if err != nil {
			return false, err
		}
		m, err := setup.LookupModel(model)
		if err != nil {
			return false, err
		}
		next := home.Active{Runtime: spec.ID(), ModelID: m.ID, Model: setup.ModelDirName(m), Device: device}
		var previous home.Active
		readErr := home.ReadJSON(filepath.Join(root, "state", "active-runtime.json"), &previous)
		changed := readErr != nil || previous != next
		return changed, home.WriteJSON(filepath.Join(root, "state", "active-runtime.json"), next)
	}
	return r
}

func desiredSource(e *applyEnv, device DeviceIntent) DesiredStateParams {
	return DesiredStateParams{Model: e.model.ID, Device: device}
}

func TestReconcileDesiredStateNoOpResolvesAutoAndProvesCurrentServingState(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	rig := wireDesiredRig(t, e)
	e.start()
	before := e.c.Snapshot()
	active := e.record()

	result, err := e.c.ReconcileDesiredState(context.Background(), desiredSource(e, DeviceIntent{Mode: DeviceModeAuto}))
	if err != nil {
		t.Fatal(err)
	}
	if result.Device.Mode != DeviceModeAuto || result.Device.Requested != "" || result.Device.Resolved != "cuda" || result.Device.Actual != "cuda" {
		t.Fatalf("device resolution %+v", result.Device)
	}
	if result.Model != e.model.ID || result.Variant != "" || result.Serving.Smoke.Question == "" {
		t.Fatalf("result %+v", result)
	}
	if e.openCount() != 1 || e.record() != active || e.c.Snapshot().Status.Worker.PID != before.Status.Worker.PID {
		t.Fatal("an exact no-op reconciliation restarted or changed the serving state")
	}
	op := e.c.Snapshot().Maintenance
	if op == nil || op.Kind != OpDesiredState || op.Failure != nil || op.DeviceMode != string(DeviceModeAuto) || op.ResolvedDevice != "cuda" || op.ActualDevice != "cuda" {
		t.Fatalf("operation %+v", op)
	}
	if !slices.Contains(op.Phases, PhaseDesiredResolve) || !slices.Contains(op.Phases, PhaseApplySmoke) || !slices.Contains(op.Phases, PhaseApplyFinal) {
		t.Fatalf("phases %v", op.Phases)
	}
	if ids, err := (&settings.Store{Path: rig.store.Path}).Residents(); err != nil || len(ids) != 0 {
		t.Fatalf("persisted residents %v, error %v", ids, err)
	}
}

func TestReconcileDesiredStateAutoRequiresValidActivation(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	if err := os.Remove(e.h.Path("state", "active-runtime.json")); err != nil {
		t.Fatal(err)
	}
	_, err := e.c.ReconcileDesiredState(context.Background(), desiredSource(e, DeviceIntent{Mode: DeviceModeAuto}))
	if err == nil || !strings.Contains(err.Error(), "active activation") {
		t.Fatalf("Auto without a valid activation: %v", err)
	}
	var ae *ApplyError
	if !errors.As(err, &ae) || ae.Phase != PhaseDesiredResolve {
		t.Fatalf("failure phase: %#v", err)
	}
	if op := e.c.Snapshot().Maintenance; op == nil || op.ResolvedDevice != "" || op.Failure == nil {
		t.Fatalf("operation %+v", op)
	}
}

func TestReconcileDesiredStateSourceTargetFromCertifiedVariant(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	e.start()
	if _, err := e.apply(context.Background()); err != nil {
		t.Fatal(err)
	}

	result, err := e.c.ReconcileDesiredState(context.Background(), desiredSource(e, DeviceIntent{Mode: DeviceModeAuto}))
	if err != nil {
		t.Fatal(err)
	}
	if result.Variant != "" || result.Serving.Variant != "" || result.Device.Resolved != "cuda" {
		t.Fatalf("source result %+v", result)
	}
	e.requireServingSource()
}

func TestReconcileDesiredStateAppliesCertifiedVariantTarget(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	e.start()
	request := DesiredStateParams{Model: e.model.ID, Variant: e.v.ID, Device: DeviceIntent{Mode: DeviceModeAuto}}
	result, err := e.c.ReconcileDesiredState(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != e.model.ID || result.Variant != e.v.ID || result.Serving.Variant != e.v.ID || result.Device.Resolved != "cuda" {
		t.Fatalf("variant result %+v", result)
	}
	e.requireServingVariant(e.v.ID)
}

func TestReconcileDesiredStatePersistsAndRebindsResidentChange(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	rig := wireDesiredRig(t, e)
	e.start()
	request := desiredSource(e, DeviceIntent{Mode: DeviceModePinned, Value: "cuda"})
	request.Residents = []string{modelB, e.model.ID, modelB}
	result, err := e.c.ReconcileDesiredState(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(result.Residents, []string{e.model.ID, modelB}) {
		t.Fatalf("normalized residents %v", result.Residents)
	}
	if got, err := (&settings.Store{Path: rig.store.Path}).Residents(); err != nil || !slices.Equal(got, result.Residents) {
		t.Fatalf("persisted residents %v, error %v", got, err)
	}
	if len(e.c.Snapshot().Residents) != 2 || e.c.Snapshot().RestartRequired {
		t.Fatalf("resident set was not reconciled: %+v", e.c.Snapshot())
	}
}

func TestReconcileDesiredStatePinnedDeviceIsHonored(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	e.start()
	request := desiredSource(e, DeviceIntent{Mode: DeviceModePinned, Value: "cpu"})
	result, err := e.c.ReconcileDesiredState(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Device.Mode != DeviceModePinned || result.Device.Requested != "cpu" || result.Device.Resolved != "cpu" || result.Device.Actual != "cpu" {
		t.Fatalf("pinned device resolution %+v", result.Device)
	}
	if active, _, _, err := e.h.LoadActive(); err != nil || active.Device != "cpu" {
		t.Fatalf("active device %+v, error %v", active, err)
	}
}

func TestReconcileDesiredStateMissingPrerequisiteRequiresAuthorization(t *testing.T) {
	t.Run("refuses without provisioning authorization", func(t *testing.T) {
		e := newApplyEnv(t, "accepted")
		rig := wireDesiredRig(t, e)
		rig.failRuntime = true
		e.start()
		_, err := e.c.ReconcileDesiredState(context.Background(), desiredSource(e, DeviceIntent{Mode: DeviceModePinned, Value: "cuda"}))
		if err == nil || !strings.Contains(err.Error(), "provisioning was not authorized") || rig.repairs != 0 {
			t.Fatalf("error %v, repairs %d", err, rig.repairs)
		}
		e.requireServingSource()
	})
	t.Run("repairs when explicitly authorized", func(t *testing.T) {
		e := newApplyEnv(t, "accepted")
		rig := wireDesiredRig(t, e)
		rig.failRuntime = true
		e.start()
		request := desiredSource(e, DeviceIntent{Mode: DeviceModePinned, Value: "cuda"})
		request.AllowProvision = true
		result, err := e.c.ReconcileDesiredState(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		if rig.repairs != 1 || !result.Serving.Materialized {
			t.Fatalf("repairs %d result %+v", rig.repairs, result)
		}
	})
}

func TestReconcileDesiredStateExactServingMismatchRollsBack(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	e.start()
	if _, err := e.apply(context.Background()); err != nil {
		t.Fatal(err)
	}
	e.setMode("source", "cpu_on_cuda")
	_, err := e.c.ReconcileDesiredState(context.Background(), desiredSource(e, DeviceIntent{Mode: DeviceModeAuto}))
	var ae *ApplyError
	if !errors.As(err, &ae) || ae.Phase != PhaseApplyProve || !ae.RolledBack || !strings.Contains(err.Error(), "no fallback to another device") {
		t.Fatalf("error %#v", err)
	}
	e.requireServingVariant(e.v.ID)
	if op := e.c.Snapshot().Maintenance; op == nil || op.DeviceMode != string(DeviceModeAuto) || op.ResolvedDevice != "cuda" || op.ActualDevice != "cpu" {
		t.Fatalf("operation did not retain requested and actual device facts: %+v", op)
	}
}

func TestReconcileDesiredStateResidentRollbackRestoresPersistedSelection(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	rig := wireDesiredRig(t, e)
	e.start()
	e.failOpenAt = 2
	request := desiredSource(e, DeviceIntent{Mode: DeviceModePinned, Value: "cuda"})
	request.Residents = []string{modelB}
	_, err := e.c.ReconcileDesiredState(context.Background(), request)
	var ae *ApplyError
	if !errors.As(err, &ae) || !ae.RolledBack || !strings.Contains(err.Error(), "cannot be bound") {
		t.Fatalf("error %#v", err)
	}
	if ids, err := (&settings.Store{Path: rig.store.Path}).Residents(); err != nil || len(ids) != 0 {
		t.Fatalf("rollback left persisted residents %v, error %v", ids, err)
	}
	e.requireServingSource()
}

func TestReconcileDesiredStatePreservesPrimaryAndResidentRollbackFailures(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	rig := wireDesiredRig(t, e)
	e.start()
	e.failOpenAt = 2
	rig.failRestoreResidents = true
	request := desiredSource(e, DeviceIntent{Mode: DeviceModePinned, Value: "cuda"})
	request.Residents = []string{modelB}
	_, err := e.c.ReconcileDesiredState(context.Background(), request)
	var ae *ApplyError
	if !errors.As(err, &ae) || ae.Primary == nil || ae.Rollback == nil || ae.RolledBack ||
		!strings.Contains(ae.Primary.Error(), "cannot be bound") || !strings.Contains(ae.Rollback.Error(), "resident restore refused") {
		t.Fatalf("primary and rollback were not preserved: %#v", err)
	}
}

// rewriteActiveRuntime turns the active runtime into another self-consistent
// runtime (its identity is the one its own spec derives) and returns its name.
func rewriteActiveRuntime(t *testing.T, e *applyEnv, mut func(*home.RuntimeSpec)) string {
	t.Helper()
	var a home.Active
	var rm home.RuntimeManifest
	if err := home.ReadJSON(e.h.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	if err := home.ReadJSON(e.h.Path("runtime", a.Runtime, "manifest.json"), &rm); err != nil {
		t.Fatal(err)
	}
	old := a.Runtime
	mut(&rm.Spec)
	rm.Identity, a.Runtime = rm.Spec.ID(), rm.Spec.ID()
	if err := os.Rename(e.h.Path("runtime", old), e.h.Path("runtime", rm.Identity)); err != nil {
		t.Fatal(err)
	}
	if err := home.WriteJSON(e.h.Path("runtime", rm.Identity, "manifest.json"), rm); err != nil {
		t.Fatal(err)
	}
	if err := home.WriteJSON(e.h.Path("state", "active-runtime.json"), a); err != nil {
		t.Fatal(err)
	}
	return rm.Identity
}

// Auto resolves the device from an activation whose runtime is an older
// dependency environment of that device: the desired-state transaction is what
// provisions the required runtime, so that activation is a valid baseline. The
// device-runtime invariant that remains is the real one: the activation's
// runtime must be a runtime of its device.
func TestReconcileDesiredStateAutoAcceptsAnOlderRuntimeOfTheSameDevice(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	older := rewriteActiveRuntime(t, e, func(s *home.RuntimeSpec) { s.Lock = strings.Repeat("0", 64) })
	tx, err := e.c.beginDesiredState(desiredSource(e, DeviceIntent{Mode: DeviceModeAuto}))
	if err != nil {
		t.Fatal(err)
	}
	a, _, err := tx.resolveAutoDevice()
	if err != nil || a.Runtime != older || a.Device != "cuda" {
		t.Fatalf("resolveAutoDevice = %+v, %v", a, err)
	}
}

func TestReconcileDesiredStateAutoRejectsARuntimeOfAnotherDevice(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	wireDesiredRig(t, e)
	rewriteActiveRuntime(t, e, func(s *home.RuntimeSpec) { s.Flavor, s.Torch, s.Lock = "cpu", "2.11.0+cpu", strings.Repeat("1", 64) })
	tx, err := e.c.beginDesiredState(desiredSource(e, DeviceIntent{Mode: DeviceModeAuto}))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := tx.resolveAutoDevice(); err == nil || !strings.Contains(err.Error(), "not a runtime of its cuda device") {
		t.Fatalf("resolveAutoDevice = %v", err)
	}
}
