package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// OpDesiredState is one Controller-owned model and residency reconciliation.
const OpDesiredState = "desired_state"

// PhaseDesiredResolve is the device-selection phase of desired-state work.
const PhaseDesiredResolve = "resolving"

// PhaseDesiredProvision is the authorized managed-prerequisite phase.
const PhaseDesiredProvision = "provisioning"

// DeviceMode records whether a device came from Auto or an operator pin.
type DeviceMode string

const (
	DeviceModeAuto   DeviceMode = "auto"
	DeviceModePinned DeviceMode = "pinned"
)

// DeviceIntent is a device choice. Value must be empty for Auto and cpu or
// cuda for a pinned choice.
type DeviceIntent struct {
	Mode  DeviceMode `json:"mode"`
	Value string     `json:"value,omitempty"`
}

// DesiredStateParams name the exact serving target, device-selection mode and
// additional resident catalog models. Variant empty means the source artifact
// of Model. AllowProvision explicitly authorizes Repair to restore the target
// runtime and source prerequisites when they are missing or fail verification.
type DesiredStateParams struct {
	Model          string       `json:"model"`
	Variant        string       `json:"variant,omitempty"`
	Device         DeviceIntent `json:"device"`
	Residents      []string     `json:"residents"`
	AllowProvision bool         `json:"allow_provision,omitempty"`
}

// DesiredDeviceResolution records the operator's selection and the device
// the transaction resolved before changing serving state.
type DesiredDeviceResolution struct {
	Mode      DeviceMode `json:"mode"`
	Requested string     `json:"requested,omitempty"`
	Resolved  string     `json:"resolved"`
	Actual    string     `json:"actual,omitempty"`
}

// DesiredStateResult is returned only after the exact requested state is
// READY, proven, smoke-tested and still bound to the proven worker.
type DesiredStateResult struct {
	Model     string                  `json:"model"`
	Variant   string                  `json:"variant,omitempty"`
	Device    DesiredDeviceResolution `json:"device"`
	Residents []string                `json:"residents"`
	Serving   ApplyResult             `json:"serving"`
}

// ReconcileDesiredState reconciles model, variant, device and resident intent
// as one Controller operation. Auto uses the device of a valid current
// activation; it never probes hardware or substitutes another device.
func (c *Controller) ReconcileDesiredState(ctx context.Context, p DesiredStateParams) (DesiredStateResult, error) {
	tx, err := c.beginDesiredState(p)
	if err != nil {
		return DesiredStateResult{}, err
	}
	serving, err := c.runDesiredState(ctx, tx)
	if err != nil {
		return DesiredStateResult{}, err
	}
	return DesiredStateResult{
		Model: tx.model.ID, Variant: tx.variantID(),
		Device:    DesiredDeviceResolution{Mode: p.Device.Mode, Requested: p.Device.Value, Resolved: tx.p.Device, Actual: serving.ReportedDevice},
		Residents: slices.Clone(tx.desiredResidents), Serving: serving,
	}, nil
}

// StartDesiredState accepts one desired-state operation and runs the same
// transaction in the background for a caller that must not block on setup or
// worker readiness. The result and failure are projected through Snapshot.
func (c *Controller) StartDesiredState(p DesiredStateParams) error {
	tx, err := c.beginDesiredState(p)
	if err != nil {
		return err
	}
	go func() { _, _ = c.runDesiredState(context.Background(), tx) }()
	return nil
}

func (c *Controller) beginDesiredState(p DesiredStateParams) (*applyTx, error) {
	if p.Model == "" && p.Variant == "" {
		return nil, errors.New("desired state: a source model or variant is required")
	}
	switch p.Device.Mode {
	case DeviceModeAuto:
		if p.Device.Value != "" {
			return nil, errors.New("desired state: Auto does not accept a pinned device value")
		}
	case DeviceModePinned:
		if p.Device.Value != "cpu" && p.Device.Value != "cuda" {
			return nil, fmt.Errorf("desired state: a pinned device must be cpu or cuda, got %q", p.Device.Value)
		}
	default:
		return nil, fmt.Errorf("desired state: device mode must be auto or pinned, got %q", p.Device.Mode)
	}
	residents, err := settings.NormalizeResidents(p.Residents)
	if err != nil {
		return nil, err
	}
	p.Residents = residents

	target := "source " + p.Model
	if p.Variant != "" {
		target = setup.KindVariant + " " + p.Variant
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.admit(); err != nil {
		return nil, err
	}
	device := p.Device.Value
	op := c.begin(OpDesiredState, device, p.Model)
	op.Target = target
	op.DeviceMode = string(p.Device.Mode)
	op.RequestedDevice = p.Device.Value
	op.Residents = slices.Clone(residents)
	op.ResolvedDevice = device
	op.Plan = plan(OpDesiredState, target)
	return &applyTx{
		c: c, op: op, root: c.home, prevRT: c.rt, stoppedBefore: c.stopped,
		p:       ApplyParams{Model: p.Model, Variant: p.Variant, Device: device, Materialize: p.AllowProvision},
		desired: &p, desiredResidents: residents,
	}, nil
}

func (c *Controller) runDesiredState(ctx context.Context, tx *applyTx) (ApplyResult, error) {
	log, closeLog := OpenSetupLog(tx.root, OpDesiredState, tx.p.Device, tx.p.Model, tx.op.Target)
	defer closeLog()
	tx.log, tx.obs = log, c.observerFor(tx.op)
	tx.scrub = redact.New(tx.root)
	res, err := tx.run(ctx)
	c.mu.Lock()
	defer c.mu.Unlock()
	var f *Failure
	if err != nil {
		phase := tx.op.Phase
		var ae *ApplyError
		if errors.As(err, &ae) {
			phase = ae.Phase
		}
		f = &Failure{Source: SourceSetup, Phase: phase, Message: err.Error()}
	}
	c.finish(tx.op, f)
	return res, err
}

func (t *applyTx) setResolvedDevice(device string) {
	t.p.Device = device
	t.c.mu.Lock()
	t.op.Device = device
	t.op.ResolvedDevice = device
	t.c.notify()
	t.c.mu.Unlock()
}

func (t *applyTx) resolveAutoDevice() (home.Active, []byte, error) {
	h := t.h()
	a, _, _, err := h.LoadActive()
	if err != nil {
		return home.Active{}, nil, fmt.Errorf("resolving Auto device from the active activation: %w", err)
	}
	model, err := setup.ActiveModel(a)
	if err != nil {
		return home.Active{}, nil, err
	}
	spec, err := setup.Desired(a.Device)
	if err != nil {
		return home.Active{}, nil, fmt.Errorf("active activation has unsupported device %q: %w", a.Device, err)
	}
	if a.Runtime != spec.ID() {
		return home.Active{}, nil, fmt.Errorf("active activation runtime %q does not match its %s device runtime %q", a.Runtime, a.Device, spec.ID())
	}
	if model.ID != a.ModelID && a.ModelID != "" {
		return home.Active{}, nil, fmt.Errorf("active activation model %q does not match catalog model %q", a.ModelID, model.ID)
	}
	if a.Variant != "" {
		if _, _, err := h.LoadVariant(a); err != nil {
			return home.Active{}, nil, fmt.Errorf("active activation variant is invalid: %w", err)
		}
	}
	raw, err := h.ReadActiveRecord()
	if err != nil {
		return home.Active{}, nil, fmt.Errorf("reading the active activation for Auto resolution: %w", err)
	}
	t.autoRecord = raw
	return a, raw, nil
}

func sameActiveTarget(a home.Active, model home.ModelManifest, variant string, device string) bool {
	return a.ModelID == model.ID && a.Model == setup.ModelDirName(model) && a.Variant == variant && a.Device == device && !a.Experimental
}

func expectedResidents(def string, desired []string) []string {
	return append([]string{def}, additionalResidents(def, desired)...)
}

func exactResidentModels(rt Runtime, want []string) error {
	views := viewOf(rt)
	got := make([]string, 0, len(views))
	for _, v := range views {
		got = append(got, v.Model)
	}
	wanted := slices.Clone(want)
	slices.Sort(got)
	slices.Sort(wanted)
	if !slices.Equal(got, wanted) {
		return fmt.Errorf("the bound resident set is %v, not the requested %v", got, wanted)
	}
	return nil
}

func (t *applyTx) validateDesiredPrerequisites(runtimeID string) error {
	verify := t.c.cfg.Maintenance.Verify
	runtimeErr := verify(t.root, setup.KindRuntime, runtimeID, t.obs)
	modelErr := verify(t.root, setup.KindModel, t.model.ID, t.obs)
	var variantErr error
	if t.hasVariant {
		variantErr = verify(t.root, setup.KindVariant, t.variant.ID, t.obs)
	}
	if variantErr != nil {
		return fmt.Errorf("verifying requested variant %s: %w", t.variant.ID, variantErr)
	}
	spec, err := setup.Desired(t.p.Device)
	if err != nil {
		return err
	}
	var causes []error
	if runtimeErr != nil {
		causes = append(causes, fmt.Errorf("runtime %s: %w", runtimeID, runtimeErr))
		t.repairModels = append(t.repairModels, t.model.ID)
	}
	if modelErr != nil {
		causes = append(causes, fmt.Errorf("source model %s: %w", t.model.ID, modelErr))
		t.repairModels = append(t.repairModels, t.model.ID)
	}
	for _, id := range additionalResidents(t.model.ID, t.desiredResidents) {
		model, err := setup.LookupModel(id)
		if err != nil {
			return err
		}
		if !spec.Provides(model.Provider) {
			return fmt.Errorf("the %s runtime %s does not carry provider %s needed by resident model %s", t.p.Device, spec.ID(), model.Provider, id)
		}
		if err := verify(t.root, setup.KindModel, id, t.obs); err != nil {
			causes = append(causes, fmt.Errorf("resident model %s: %w", id, err))
			t.repairModels = append(t.repairModels, id)
		}
	}
	if len(causes) == 0 {
		return nil
	}
	problem := errors.Join(causes...)
	if !t.desired.AllowProvision {
		return fmt.Errorf("target prerequisites are missing or invalid; provisioning was not authorized: %w", problem)
	}
	t.repairModels = slices.Compact(t.repairModels)
	t.needsRepair = true
	return nil
}

func (t *applyTx) transactDesired(ctx context.Context) (ApplyResult, string, error) {
	c, h := t.c, t.h()
	phase := PhaseApplyValidate
	if t.needsRepair {
		phase = PhaseDesiredProvision
		c.phase(t.op, phase)
		if t.prevRT != nil && t.prevRT.Running() {
			t.rebound = true
			t.prevRT.Stop()
		}
		for _, model := range t.repairModels {
			if err := c.cfg.Maintenance.Repair(t.root, t.p.Device, model, t.log, t.obs); err != nil {
				return ApplyResult{}, phase, fmt.Errorf("repairing runtime and source prerequisites for %s: %w", model, err)
			}
		}
		if err := c.cfg.Maintenance.Verify(t.root, setup.KindRuntime, mustDesiredRuntime(t.p.Device), t.obs); err != nil {
			return ApplyResult{}, phase, fmt.Errorf("runtime remains invalid after repair: %w", err)
		}
		for _, model := range append([]string{t.model.ID}, additionalResidents(t.model.ID, t.desiredResidents)...) {
			if err := c.cfg.Maintenance.Verify(t.root, setup.KindModel, model, t.obs); err != nil {
				return ApplyResult{}, phase, fmt.Errorf("source or resident model %s remains invalid after repair: %w", model, err)
			}
		}
		t.materialized = true
	}
	if err := ctx.Err(); err != nil {
		return ApplyResult{}, phase, err
	}

	residentsChanged := !slices.Equal(t.prevResidents, t.desiredResidents)
	if residentsChanged {
		if c.cfg.Residents == nil || c.cfg.SetResidents == nil {
			return ApplyResult{}, phase, errors.New("the desired resident selection cannot be persisted by this Controller")
		}
		t.residentsMutated = true
		if err := c.cfg.SetResidents(slices.Clone(t.desiredResidents)); err != nil {
			return ApplyResult{}, phase, fmt.Errorf("saving the desired resident selection: %w", err)
		}
		got, err := settings.NormalizeResidents(c.cfg.Residents())
		if err != nil {
			return ApplyResult{}, phase, fmt.Errorf("reading the saved resident selection: %w", err)
		}
		if !slices.Equal(got, t.desiredResidents) {
			return ApplyResult{}, phase, fmt.Errorf("the settings authority saved residents %v, not the requested %v", got, t.desiredResidents)
		}
	}

	variant := t.variantID()
	if !sameActiveTarget(t.prevActive, t.model, variant, t.p.Device) {
		phase = string(setup.PhaseActivation)
		if t.hasVariant {
			_, err := c.cfg.Maintenance.ActivateVariant(t.root, t.p.Device, t.model.ID, t.variant.ID, false, t.log, t.obs)
			if now, rerr := h.ReadActiveRecord(); rerr != nil || !bytes.Equal(now, t.prevRecord) {
				t.mutated = true
			}
			if err != nil {
				return ApplyResult{}, phase, err
			}
		} else {
			_, err := c.cfg.Maintenance.Activate(t.root, t.p.Device, t.model.ID, t.log, t.obs)
			if now, rerr := h.ReadActiveRecord(); rerr != nil || !bytes.Equal(now, t.prevRecord) {
				t.mutated = true
			}
			if err != nil {
				return ApplyResult{}, phase, err
			}
		}
		if err := t.recordNamesDesired(); err != nil {
			return ApplyResult{}, phase, err
		}
	}

	pending := false
	c.mu.Lock()
	pending = c.pending
	c.mu.Unlock()
	rebind := t.mutated || t.rebound || residentsChanged || pending || t.prevRT == nil || !t.prevRT.Running() ||
		c.residencyDrift(t.model.ID, t.prevRT) || !t.bindingMatchesDesired(t.prevRT)
	var rt Runtime
	if rebind {
		phase = PhaseApplyRebind
		c.phase(t.op, phase)
		if err := ctx.Err(); err != nil {
			return ApplyResult{}, phase, err
		}
		t.rebound = true
		var f *Failure
		rt, f = c.bindAndStart(OpStart, t.root, nil, t.prevRT)
		if f != nil {
			return ApplyResult{}, phase, f
		}
		c.mu.Lock()
		c.rt, c.pending, c.stopped = rt, false, false
		c.mu.Unlock()
	} else {
		rt = t.prevRT
	}

	phase = PhaseApplyReady
	c.phase(t.op, phase)
	views, err := c.awaitServing(ctx, rt, nil, c.readyTimeout(), t.scrub)
	if err != nil {
		return ApplyResult{}, phase, err
	}
	c.mu.Lock()
	t.op.ActualDevice = str(defaultView(views).Status.Worker.Info, "device")
	c.notify()
	c.mu.Unlock()
	if err := exactResidentModels(rt, expectedResidents(t.model.ID, t.desiredResidents)); err != nil {
		return ApplyResult{}, phase, err
	}

	phase = PhaseApplyProve
	c.phase(t.op, phase)
	def := defaultView(views)
	var result ApplyResult
	if t.hasVariant {
		result, err = t.prove(def)
	} else {
		result, err = t.proveDesiredSource(def)
	}
	if err != nil {
		return ApplyResult{}, phase, err
	}
	c.mu.Lock()
	t.op.ActualDevice = result.ReportedDevice
	c.notify()
	c.mu.Unlock()

	phase = PhaseApplySmoke
	c.phase(t.op, phase)
	if result.Smoke, err = t.smoke(ctx, rt, def); err != nil {
		return ApplyResult{}, phase, err
	}

	phase = PhaseApplyFinal
	c.phase(t.op, phase)
	if err := t.finalDesired(rt, def); err != nil {
		return ApplyResult{}, phase, err
	}
	result.Materialized = t.materialized
	result.Previous = ApplyPrevious{Model: t.prevActive.ModelID, Variant: t.prevActive.Variant, Device: t.prevActive.Device,
		Experimental: t.prevActive.Experimental, Running: slices.Clone(t.prevRunning)}
	return result, "", nil
}

func mustDesiredRuntime(device string) string {
	spec, _ := setup.Desired(device)
	return spec.ID()
}

func (t *applyTx) bindingMatchesDesired(rt Runtime) bool {
	if rt == nil {
		return false
	}
	status := rt.Status().Runtime
	spec, err := setup.Desired(t.p.Device)
	if err != nil || status.ModelID != t.model.ID || status.Device != t.p.Device || status.Runtime != spec.ID() {
		return false
	}
	if t.hasVariant {
		return status.Variant != nil && status.Variant.ID == t.variant.ID
	}
	return status.Variant == nil
}

func (t *applyTx) recordNamesDesired() error {
	var a home.Active
	raw, err := t.h().ReadActiveRecord()
	if err == nil {
		err = json.Unmarshal(raw, &a)
	}
	if err != nil {
		return fmt.Errorf("reading back the activation record: %w", err)
	}
	if !sameActiveTarget(a, t.model, t.variantID(), t.p.Device) {
		return fmt.Errorf("the activation record names model %q variant %q device %q experimental=%v, not the requested model %s variant %q on %s",
			a.ModelID, a.Variant, a.Device, a.Experimental, t.model.ID, t.variantID(), t.p.Device)
	}
	return nil
}

func (t *applyTx) proveDesiredSource(def memberView) (ApplyResult, error) {
	rt, w := def.Status.Runtime, def.Status.Worker
	spec, err := setup.Desired(t.p.Device)
	if err != nil {
		return ApplyResult{}, err
	}
	if rt.ModelID != t.model.ID {
		return ApplyResult{}, fmt.Errorf("the runtime serves source model %q, not %q", rt.ModelID, t.model.ID)
	}
	if rt.Device != t.p.Device {
		return ApplyResult{}, fmt.Errorf("device %q was requested but the runtime is bound to %q; there is no fallback to another device", t.p.Device, rt.Device)
	}
	if rt.Runtime != spec.ID() {
		return ApplyResult{}, fmt.Errorf("the runtime is %q, not the %s runtime %s", rt.Runtime, t.p.Device, spec.ID())
	}
	if rt.Variant != nil {
		return ApplyResult{}, fmt.Errorf("the source was requested but the runtime names variant %q", rt.Variant.ID)
	}
	target := ExecutionTarget{Kind: eval.ForgeTargetSource, Model: t.model.ID, Device: t.p.Device}
	if err := verifyExecution(target, t.model, nil, w.Info); err != nil {
		return ApplyResult{}, fmt.Errorf("serving provenance: %w", err)
	}
	return ApplyResult{Model: t.model.ID, Runtime: rt.Runtime, Device: t.p.Device, ReportedDevice: str(w.Info, "device"),
		DType: normDType(str(w.Info, "dtype")), PID: w.PID}, nil
}

func (t *applyTx) finalDesired(rt Runtime, proven memberView) error {
	now := defaultView(viewOf(rt))
	w, pw := now.Status.Worker, proven.Status.Worker
	if !now.Running || !w.Ready || w.State != worker.StateReady || w.PID != pw.PID || w.Starts != pw.Starts {
		return fmt.Errorf("after the smoke the serving worker is %s (pid %d, start %d), not the proven READY worker (pid %d, start %d)", w.State, w.PID, w.Starts, pw.PID, pw.Starts)
	}
	if err := t.recordNamesDesired(); err != nil {
		return err
	}
	if err := exactResidentModels(rt, expectedResidents(t.model.ID, t.desiredResidents)); err != nil {
		return err
	}
	after := t.c.servingState(t.h(), rt)
	if !slices.Equal(after.desired, t.desiredResidents) {
		return fmt.Errorf("the saved desired resident selection is %v, not %v", after.desired, t.desiredResidents)
	}
	if after.routing != t.before.routing {
		return errors.New("the routing policy changed during desired-state reconciliation")
	}
	return nil
}
