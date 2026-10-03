package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// OpApply is the certified-variant apply transaction (Controller.ApplyCertifiedVariant).
const OpApply = "apply"

// Phases of the apply transaction. The setup authority's own phases (runtime,
// model, variant, activation) are entered between PhaseApplySnapshot and
// PhaseApplyRebind, while it writes the activation record.
const (
	PhaseApplyValidate = "validating"
	PhaseApplySnapshot = "snapshotting"
	PhaseApplyRebind   = "rebinding"
	PhaseApplyReady    = "awaiting_ready"
	PhaseApplyProve    = "proving"
	PhaseApplySmoke    = "smoke"
	PhaseApplyFinal    = "finalizing"
	PhaseApplyRollback = "rolling_back"
)

const (
	defaultReadyTimeout = 15 * time.Minute
	defaultSmokeTimeout = 3 * time.Minute
)

// ApplyParams are the explicit inputs of ApplyCertifiedVariant: the variant,
// the device it must serve on and, optionally, its source model (which, when
// given, must be the variant's own source).
type ApplyParams struct {
	Variant string
	Device  string // "cpu" or "cuda": explicit, never defaulted, never substituted
	Model   string
	// Materialize allows the setup authority to materialize the device's
	// serving runtime and the source model when they are missing, before
	// anything is activated. Without it a missing artifact is a refusal.
	Materialize bool
}

// ApplyResult is the proof an applied variant is the serving artifact. It is
// only returned populated by a success.
type ApplyResult struct {
	Model            string `json:"model"`
	Variant          string `json:"variant"`
	Runtime          string `json:"runtime"`
	Device           string `json:"device"`          // requested
	ReportedDevice   string `json:"reported_device"` // what the worker reports
	DType            string `json:"dtype"`
	Scheme           string `json:"scheme"`
	Quantization     string `json:"quantized_execution"`
	QuantizedModules int    `json:"quantized_modules"`
	Certification    string `json:"certification"`
	PID              int    `json:"pid"`
	Materialized     bool   `json:"materialized,omitempty"`
	// Previous is the serving target this apply replaced.
	Previous ApplyPrevious `json:"previous"`
	Smoke    ApplySmoke    `json:"smoke"`
}

// ApplyPrevious is the activation and runtime state an apply replaced (and a
// failed one restores).
type ApplyPrevious struct {
	Model        string   `json:"model,omitempty"`
	Variant      string   `json:"variant,omitempty"` // empty: the source artifact
	Device       string   `json:"device,omitempty"`
	Experimental bool     `json:"experimental,omitempty"`
	Running      []string `json:"running,omitempty"` // residents that were running; empty: stopped
}

// ApplySmoke is the one typed decision asked through the serving path.
type ApplySmoke struct {
	Question    string  `json:"question"`
	Choice      string  `json:"choice"`
	Confidence  float64 `json:"confidence"`
	InferenceMS float64 `json:"inference_ms"`
}

// ApplyError is a failed apply. Primary is why it failed and is never replaced
// by anything that happened while recovering. Mutated reports that the
// activation record had been changed; RolledBack that the previous serving
// target was restored and the restored state verified; Rollback is why
// restoring failed, kept beside the primary failure as secondary evidence.
type ApplyError struct {
	Phase      string // the phase of the primary failure
	Primary    error
	Mutated    bool
	RolledBack bool
	Rollback   error
}

func (e *ApplyError) Error() string {
	switch {
	case e.Rollback != nil:
		return e.Primary.Error() + "; additionally the rollback failed: " + e.Rollback.Error()
	case e.RolledBack:
		return e.Primary.Error() + "; the previous serving target was restored and verified"
	}
	return e.Primary.Error()
}

func (e *ApplyError) Unwrap() []error {
	errs := []error{e.Primary}
	if e.Rollback != nil {
		errs = append(errs, e.Rollback)
	}
	return errs
}

func (c *Controller) readyTimeout() time.Duration {
	if c.cfg.ReadyTimeout > 0 {
		return c.cfg.ReadyTimeout
	}
	return defaultReadyTimeout
}

func (c *Controller) smokeTimeout() time.Duration {
	if c.cfg.SmokeTimeout > 0 {
		return c.cfg.SmokeTimeout
	}
	return defaultSmokeTimeout
}

func (c *Controller) restoreTimeout() time.Duration {
	if c.cfg.RestoreTimeout > 0 {
		return c.cfg.RestoreTimeout
	}
	return defaultRestoreTimeout
}

// ApplyCertifiedVariant makes an accepted, quantized variant the serving
// artifact, and proves it, as one recoverable transaction:
//
//	validate -> snapshot -> activate -> rebind -> READY -> provenance -> smoke
//
// It is one controller action (nothing else starts, stops or activates while it
// runs). Before the activation record is touched the variant must exist, its
// manifest and source link must verify and its latest certification record must
// be accepted; an uncertified, rejected, ambiguous or stale variant is refused
// and nothing changes. The activation is then written by the existing
// activation authority (Maintenance.ActivateVariant), the serving runtime is
// rebound through the controller's own Open and started, and success is withheld
// until a coherent READY observation of the default resident proves the exact
// requested variant executes on the requested device with its declared
// quantized execution, and one bounded typed-decision request through the
// normal HTTP serving handler returned a valid decision. There is no fallback to
// the source, another variant or another device.
//
// Any failure after the activation record changed restores the previous serving
// target: the exact previous activation record, the runtime bound from it, the
// residents that were running (a previously stopped runtime stays stopped), and
// verifies the restored state. The primary failure is always the one reported;
// a failed rollback is returned beside it.
func (c *Controller) ApplyCertifiedVariant(ctx context.Context, p ApplyParams) (ApplyResult, error) {
	tx, err := c.beginApply(p)
	if err != nil {
		return ApplyResult{}, err
	}
	return c.runApply(ctx, tx)
}

// StartApply accepts one apply and runs the same transaction as
// ApplyCertifiedVariant in the background, for a caller that must not block
// for the whole transaction (the dashboard). A refusal at admission (bad
// parameters, a busy controller, a restart required) is returned here; the
// transaction's own progress, its outcome and a rollback are read from the
// controller's operation state (Snapshot), exactly as for every other action.
func (c *Controller) StartApply(p ApplyParams) error {
	tx, err := c.beginApply(p)
	if err != nil {
		return err
	}
	go func() { _, _ = c.runApply(context.Background(), tx) }()
	return nil
}

// beginApply validates the explicit inputs, admits the apply as the one
// controller action and returns its transaction.
func (c *Controller) beginApply(p ApplyParams) (*applyTx, error) {
	if p.Variant == "" {
		return nil, errors.New("apply: a variant ID is required")
	}
	if p.Device != "cpu" && p.Device != "cuda" {
		return nil, fmt.Errorf("apply: the serving device must be cpu or cuda, got %q (there is no default and no fallback)", p.Device)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.admit(); err != nil {
		return nil, err
	}
	prev := c.rt
	if prev != nil && prev.Running() && (c.pending || c.residencyDrift(prev.Status().Runtime.ModelID, prev)) {
		// The running worker no longer serves the activation record, so no
		// record could restore what is running: the operator restarts first.
		return nil, ErrRestartRequired
	}
	// No model on the operation: an action naming a model is projected as an
	// action on one resident, and an apply acts on the whole runtime.
	op := c.begin(OpApply, p.Device, "")
	op.Target = setup.KindVariant + " " + p.Variant
	op.Plan = plan(OpApply, op.Target)
	return &applyTx{c: c, op: op, p: p, root: c.home, prevRT: prev, stoppedBefore: c.stopped}, nil
}

// runApply runs an admitted apply to its end, records a Forge diagnostic for a
// failure and finishes the controller operation.
func (c *Controller) runApply(ctx context.Context, tx *applyTx) (ApplyResult, error) {
	p, op := tx.p, tx.op
	log, closeLog := OpenSetupLog(tx.root, OpApply, p.Device, p.Model, op.Target)
	defer closeLog()
	tx.log, tx.obs, tx.scrub = log, c.observerFor(op), redact.New(tx.root)
	res, err := tx.run(ctx)

	diag := ""
	if err != nil {
		phase := ""
		var ae *ApplyError
		if errors.As(err, &ae) {
			phase = ae.Phase
		}
		model := p.Model
		if tx.model.ID != "" {
			model = tx.model.ID
		}
		err = WithForgeDiagnostic(tx.root, ForgeFailure{OperationID: op.ID, Kind: OpApply, Phase: phase, Started: op.Started, Finished: time.Now(),
			Model: model, Variant: p.Variant, Device: p.Device}, err)
		diag = DiagnosticID(err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var f *Failure
	if err != nil {
		f = &Failure{Source: SourceSetup, Phase: op.Phase, Message: err.Error(), Diagnostic: diag}
		var ae *ApplyError
		if errors.As(err, &ae) {
			f.Phase = ae.Phase
		}
	}
	c.finish(op, f)
	return res, err
}

// applyTx is the state of one apply transaction.
type applyTx struct {
	c       *Controller
	op      *Operation
	p       ApplyParams
	desired *DesiredStateParams
	root    string
	log     io.Writer
	obs     *setup.Observer
	scrub   redact.Scrubber

	model            home.ModelManifest
	variant          home.VariantManifest
	hasVariant       bool
	autoRecord       []byte
	desiredResidents []string
	prevResidents    []string
	repairModels     []string

	// The snapshot: everything needed to restore the previous serving target.
	prevRT        Runtime
	stoppedBefore bool
	prevRecord    []byte
	prevActive    home.Active
	prevRunning   []string // members that were running, by model
	prevDefault   string   // model of the default member of the previous binding
	before        servingState

	mutated          bool // the activation record was changed
	rebound          bool // the previous binding was stopped
	residentsMutated bool
	materialized     bool
	needsRepair      bool
}

func (t *applyTx) variantID() string {
	if t.hasVariant {
		return t.variant.ID
	}
	return ""
}

func (t *applyTx) h() home.Home { return home.Home{Root: t.root} }

func (t *applyTx) run(ctx context.Context) (ApplyResult, error) {
	if err := t.validate(ctx); err != nil {
		phase := t.op.Phase
		if phase == "" {
			phase = PhaseApplyValidate
		}
		return ApplyResult{}, &ApplyError{Phase: phase, Primary: err}
	}
	if err := t.snapshot(); err != nil {
		return ApplyResult{}, &ApplyError{Phase: PhaseApplySnapshot, Primary: err}
	}
	res, phase, err := t.transact(ctx)
	if err == nil {
		if t.desired != nil {
			fmt.Fprintf(t.log, "desired state: model %s variant %q serves on %s and answered a typed decision\n", t.model.ID, t.variantID(), t.p.Device)
		} else {
			fmt.Fprintf(t.log, "apply: variant %s serves on %s and answered a typed decision\n", t.variant.ID, t.p.Device)
		}
		return res, nil
	}
	fmt.Fprintf(t.log, "apply: failed in %s: %v\n", phase, err)
	ae := &ApplyError{Phase: phase, Primary: err, Mutated: t.mutated}
	if t.mutated || t.rebound || t.residentsMutated {
		if rerr := t.rollback(ctx); rerr != nil {
			ae.Rollback = rerr
			fmt.Fprintf(t.log, "apply: rollback failed: %v\n", rerr)
		} else {
			ae.RolledBack = true
			fmt.Fprintln(t.log, "apply: the previous serving target was restored and verified")
		}
	}
	return ApplyResult{}, ae
}

// validate refuses, before anything is changed, every variant that may not be
// applied: one that does not exist or whose manifest or source link does not
// verify, one whose certification is not accepted, and a serving target that
// is not materialized. The artifacts' digests are verified by the activation
// authority before it writes the record (a refusal there leaves it untouched).
func (t *applyTx) validate(ctx context.Context) error {
	if t.desired != nil {
		t.c.phase(t.op, PhaseDesiredResolve)
		if t.desired.Device.Mode == DeviceModeAuto {
			a, _, err := t.resolveAutoDevice()
			if err != nil {
				return err
			}
			t.setResolvedDevice(a.Device)
		}
	}
	t.c.phase(t.op, PhaseApplyValidate)
	h := t.h()
	var model home.ModelManifest
	var v home.VariantManifest
	var err error
	if t.desired == nil || t.desired.Variant != "" {
		model, v, err = setup.FindVariant(h, t.p.Variant)
		if err != nil {
			return err
		}
		if t.p.Model != "" && t.p.Model != model.ID {
			return fmt.Errorf("variant %s is a variant of %s, not %s", v.ID, model.ID, t.p.Model)
		}
		t.hasVariant = true
	} else {
		model, err = setup.LookupModel(t.desired.Model)
		if err != nil {
			return err
		}
	}
	t.model, t.variant = model, v
	if t.desired != nil {
		t.c.mu.Lock()
		t.op.Model = model.ID
		t.c.notify()
		t.c.mu.Unlock()
	}
	spec, err := setup.Desired(t.p.Device)
	if err != nil {
		return err
	}
	if t.hasVariant {
		if !spec.Provides(v.Provider) {
			return fmt.Errorf("the %s runtime %s does not carry provider %s needed by variant %s", t.p.Device, spec.ID(), v.Provider, v.ID)
		}
		if err := setup.RequireAccepted(h, v); err != nil {
			return err
		}
		st := eval.ResolveCertification(h, v)
		if st.State != eval.StateAccepted || st.Record == nil || st.Record.VariantManifestSHA256 != v.ManifestSHA256() {
			return fmt.Errorf("%w: variant %s has no accepted certification bound to its manifest", setup.ErrVariantNotCertified, v.ID)
		}
	} else if !spec.Provides(model.Provider) {
		return fmt.Errorf("the %s runtime %s does not carry provider %s needed by source model %s", t.p.Device, spec.ID(), model.Provider, model.ID)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if t.desired != nil {
		if err := t.validateDesiredPrerequisites(spec.ID()); err != nil {
			return err
		}
		return nil
	}
	have := func() (rt, src bool) {
		_, e1 := os.Stat(h.Path("runtime", spec.ID()))
		_, e2 := os.Stat(h.Path("models", filepath.FromSlash(setup.ModelDirName(model))))
		return e1 == nil, e2 == nil
	}
	if rt, src := have(); !rt || !src {
		if !t.p.Materialize {
			return fmt.Errorf("the %s serving runtime %s or the source model %s is not materialized; materialize it first, or apply with materialization allowed", t.p.Device, spec.ID(), model.ID)
		}
		if err := t.c.cfg.Maintenance.Materialize(t.root, t.p.Device, model.ID, t.log, t.obs); err != nil {
			return fmt.Errorf("materializing the serving runtime and source: %w", err)
		}
		if rt, src := have(); !rt || !src {
			return fmt.Errorf("materialization finished but the %s runtime %s or the source model %s is still missing", t.p.Device, spec.ID(), model.ID)
		}
		t.materialized = true
	}
	return nil
}

// snapshot captures the authoritative state a rollback restores: the exact
// activation record, the previous binding and which of its residents were
// running, and the desired residents and routing the transaction must not
// change.
func (t *applyTx) snapshot() error {
	t.c.phase(t.op, PhaseApplySnapshot)
	h := t.h()
	raw, err := h.ReadActiveRecord()
	if err != nil {
		return fmt.Errorf("no activation record to restore (run setup first): %w", err)
	}
	if err := json.Unmarshal(raw, &t.prevActive); err != nil {
		return fmt.Errorf("the activation record is unreadable and could not be restored: %w", err)
	}
	t.prevRecord = raw
	if t.desired != nil {
		if t.desired.Device.Mode == DeviceModeAuto && !bytes.Equal(raw, t.autoRecord) {
			return errors.New("the active activation changed while Auto device resolution was being recorded")
		}
		if t.c.cfg.Residents == nil {
			return errors.New("the current desired resident selection cannot be read by this Controller")
		}
		var err error
		t.prevResidents, err = settings.NormalizeResidents(t.c.cfg.Residents())
		if err != nil {
			return fmt.Errorf("the current desired resident selection is invalid: %w", err)
		}
	}
	if t.prevRT != nil {
		for _, m := range viewOf(t.prevRT) {
			if m.Default {
				t.prevDefault = m.Model
			}
			if m.Running {
				t.prevRunning = append(t.prevRunning, m.Model)
			}
		}
	}
	t.before = t.c.servingState(h, t.prevRT)
	return nil
}

// transact is everything that follows the snapshot. phase names where it
// stopped.
func (t *applyTx) transact(ctx context.Context) (res ApplyResult, phase string, err error) {
	if t.desired != nil {
		return t.transactDesired(ctx)
	}
	c, h, p := t.c, t.h(), t.p

	// Activate through the existing activation authority.
	phase = string(setup.PhaseActivation)
	_, aerr := c.cfg.Maintenance.ActivateVariant(t.root, p.Device, t.model.ID, t.variant.ID, false, t.log, t.obs)
	if now, rerr := h.ReadActiveRecord(); rerr != nil || !bytes.Equal(now, t.prevRecord) {
		t.mutated = true
	}
	if aerr != nil {
		return res, phase, aerr
	}
	if err := t.recordNamesTarget(); err != nil {
		return res, phase, err
	}

	// Rebind the normal serving runtime through the controller.
	phase = PhaseApplyRebind
	c.phase(t.op, phase)
	if err := ctx.Err(); err != nil {
		return res, phase, err
	}
	t.rebound = true
	rt, f := c.bindAndStart(OpStart, t.root, nil, t.prevRT)
	if f != nil {
		return res, phase, f
	}
	c.mu.Lock()
	c.rt, c.pending, c.stopped = rt, false, false
	c.mu.Unlock()

	// READY, judged from one coherent observation.
	phase = PhaseApplyReady
	c.phase(t.op, phase)
	views, err := c.awaitServing(ctx, rt, nil, c.readyTimeout(), t.scrub)
	if err != nil {
		return res, phase, err
	}

	// The exact variant is what is running.
	phase = PhaseApplyProve
	c.phase(t.op, phase)
	def := defaultView(views)
	if res, err = t.prove(def); err != nil {
		return ApplyResult{}, phase, err
	}

	// One typed decision through the normal serving path.
	phase = PhaseApplySmoke
	c.phase(t.op, phase)
	if res.Smoke, err = t.smoke(ctx, rt, def); err != nil {
		return ApplyResult{}, phase, err
	}

	// Final verification: the same worker still serves, the record still names
	// the variant, and nothing unrelated moved.
	phase = PhaseApplyFinal
	c.phase(t.op, phase)
	if err := t.final(rt, def); err != nil {
		return ApplyResult{}, phase, err
	}
	res.Materialized = t.materialized
	res.Previous = ApplyPrevious{Model: t.prevActive.ModelID, Variant: t.prevActive.Variant, Device: t.prevActive.Device,
		Experimental: t.prevActive.Experimental, Running: slices.Clone(t.prevRunning)}
	return res, "", nil
}

// recordNamesTarget reads the activation record back and requires that it names
// exactly the requested source, variant and device, certified (not experimental).
func (t *applyTx) recordNamesTarget() error {
	var a home.Active
	raw, err := t.h().ReadActiveRecord()
	if err == nil {
		err = json.Unmarshal(raw, &a)
	}
	if err != nil {
		return fmt.Errorf("reading back the activation record: %w", err)
	}
	if a.ModelID != t.model.ID || a.Variant != t.variant.ID || a.Device != t.p.Device || a.Experimental {
		return fmt.Errorf("the activation record names model %q variant %q device %q experimental=%v, not the requested variant %s on %s",
			a.ModelID, a.Variant, a.Device, a.Experimental, t.variant.ID, t.p.Device)
	}
	return nil
}

// memberView is one resident as one coherent observation saw it.
type memberView struct {
	Model   string
	Default bool
	Running bool
	Status  server.Status
}

// viewOf observes every member of rt. A resident set is asked once, so each
// member's state, PID and provenance come from a single snapshot of its own
// supervisor; a single binding is one member.
func viewOf(rt Runtime) []memberView {
	if rr, ok := rt.(ResidentRuntime); ok {
		rs := rr.ResidentStatuses()
		out := make([]memberView, len(rs))
		for i, r := range rs {
			out[i] = memberView{Model: r.Model, Default: r.Default, Running: r.Running, Status: r.Status}
		}
		return out
	}
	st := rt.Status()
	return []memberView{{Model: st.Runtime.ModelID, Default: true, Running: rt.Running(), Status: st}}
}

func defaultView(views []memberView) memberView {
	for _, v := range views {
		if v.Default {
			return v
		}
	}
	return views[0]
}

// awaitServing waits until every wanted member (all members when want is nil)
// is running and READY, judging each poll from one observation of the whole
// binding, and returns that observation. A member whose worker failed, or that
// is stopped, ends the wait with its cause: waiting longer cannot change it.
func (c *Controller) awaitServing(ctx context.Context, rt Runtime, want []string, timeout time.Duration, scrub redact.Scrubber) ([]memberView, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for {
		views := viewOf(rt)
		ready, wanted := true, 0
		for _, v := range views {
			if want != nil && !slices.Contains(want, v.Model) {
				continue
			}
			wanted++
			w := v.Status.Worker
			switch {
			case w.State == worker.StateFailed:
				msg := "its worker failed"
				if w.LastFailure != nil {
					msg += ": " + w.LastFailure.Class + ": " + scrub.Line(w.LastFailure.Message, 512)
				}
				return views, fmt.Errorf("%s did not become READY: %s", viewName(v), msg)
			case w.State == worker.StateStopped && !v.Running:
				return views, fmt.Errorf("%s was stopped while waiting for READY", viewName(v))
			case !v.Running || !w.Ready || w.State != worker.StateReady || w.Starts < 1:
				ready = false
			}
		}
		if ready && wanted > 0 {
			return views, nil
		}
		select {
		case <-ctx.Done():
			return views, fmt.Errorf("the runtime did not become READY: %w", ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func viewName(v memberView) string {
	if v.Model == "" {
		return "the runtime"
	}
	return "resident " + v.Model
}

// prove requires that what became READY is the requested variant: the status
// document's runtime identity, the quantization facts it names and the
// worker-reported execution provenance, all from the observation that said
// READY. The source, another variant, another device or an unquantized
// execution is a failure.
func (t *applyTx) prove(def memberView) (ApplyResult, error) {
	model, v := t.model, t.variant
	rt, w := def.Status.Runtime, def.Status.Worker
	spec, err := setup.Desired(t.p.Device)
	if err != nil {
		return ApplyResult{}, err
	}
	if rt.ModelID != model.ID {
		return ApplyResult{}, fmt.Errorf("the runtime serves source model %q, not %q", rt.ModelID, model.ID)
	}
	if rt.Device != t.p.Device {
		return ApplyResult{}, fmt.Errorf("device %q was requested but the runtime is bound to %q; there is no fallback to another device", t.p.Device, rt.Device)
	}
	if rt.Runtime != spec.ID() {
		return ApplyResult{}, fmt.Errorf("the runtime is %q, not the %s runtime %s", rt.Runtime, t.p.Device, spec.ID())
	}
	rv := rt.Variant
	if rv == nil {
		return ApplyResult{}, fmt.Errorf("the status names no variant: the source artifact of %s is serving, not variant %s (the source is never substituted)", model.ID, v.ID)
	}
	switch {
	case rv.ID != v.ID:
		return ApplyResult{}, fmt.Errorf("the status names variant %q, not %q", rv.ID, v.ID)
	case rv.ManifestSHA256 != v.ManifestSHA256():
		return ApplyResult{}, fmt.Errorf("variant %s serves from manifest %s, not the verified %s", v.ID, rv.ManifestSHA256, v.ManifestSHA256())
	case rv.Scheme != v.Weights.Scheme || rv.DType != v.Weights.DType:
		return ApplyResult{}, fmt.Errorf("the status names quantization %s/%s, not the variant's %s/%s", rv.Scheme, rv.DType, v.Weights.Scheme, v.Weights.DType)
	case rv.Certification != eval.StateAccepted:
		return ApplyResult{}, fmt.Errorf("the variant serves as %q, not accepted", rv.Certification)
	case rv.Source.ID != model.ID || rv.Source.Revision != model.Revision:
		return ApplyResult{}, fmt.Errorf("the variant's source is %s@%s, not %s@%s", rv.Source.ID, rv.Source.Revision, model.ID, model.Revision)
	}
	target := ExecutionTarget{Kind: eval.ForgeTargetVariant, Model: model.ID, Variant: v.ID, Device: t.p.Device}
	if err := verifyExecution(target, model, &v, w.Info); err != nil {
		return ApplyResult{}, fmt.Errorf("serving provenance: %w", err)
	}
	quant := str(w.Info, "quantized_execution")
	if quant == "" {
		return ApplyResult{}, fmt.Errorf("the worker reports no quantized execution of variant %s", v.ID)
	}
	return ApplyResult{Model: model.ID, Variant: v.ID, Runtime: rt.Runtime, Device: t.p.Device, ReportedDevice: str(w.Info, "device"),
		DType: normDType(str(w.Info, "dtype")), Scheme: v.Weights.Scheme, Quantization: quant, QuantizedModules: int(num(w.Info, "weights_quantized_modules")),
		Certification: rv.Certification, PID: w.PID}, nil
}

// smoke asks the one fixed typed decision through the serving path itself: the
// public HTTP handler built over the bound runtime, the same composition that
// serves callers. One bounded attempt, no retry.
func (t *applyTx) smoke(ctx context.Context, rt Runtime, def memberView) (ApplySmoke, error) {
	dec, ok := rt.(server.Decider)
	if !ok {
		return ApplySmoke{}, errors.New("the bound runtime cannot serve typed decisions, so the variant cannot be smoke-tested")
	}
	ctx, cancel := context.WithTimeout(ctx, t.c.smokeTimeout())
	defer cancel()
	body, err := json.Marshal(doctor.SmokeRequest)
	if err != nil {
		return ApplySmoke{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://127.0.0.1/v1/decide", bytes.NewReader(body))
	if err != nil {
		return ApplySmoke{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	rec := &smokeRecorder{header: http.Header{}, code: http.StatusOK}
	done := make(chan struct{})
	go func() {
		defer close(done)
		server.HandlerSince(dec, def.Status.Runtime, time.Now()).ServeHTTP(rec, req)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		return ApplySmoke{}, fmt.Errorf("the typed-decision smoke did not answer: %w", ctx.Err())
	}
	if rec.code != http.StatusOK {
		var eb api.ErrorBody
		if json.Unmarshal(rec.body.Bytes(), &eb) == nil && eb.Error.Class != "" {
			return ApplySmoke{}, fmt.Errorf("the typed-decision smoke failed: %s: %s", eb.Error.Class, t.scrub.Line(eb.Error.Message, 512))
		}
		return ApplySmoke{}, fmt.Errorf("the typed-decision smoke failed: HTTP %d", rec.code)
	}
	var resp api.DecideResponse
	if err := json.Unmarshal(rec.body.Bytes(), &resp); err != nil {
		return ApplySmoke{}, fmt.Errorf("the typed-decision smoke answered an unreadable response: %w", err)
	}
	qs := doctor.SmokeRequest.Questions
	if resp.Schema != api.SchemaV1 || len(resp.Results) != len(qs) {
		return ApplySmoke{}, fmt.Errorf("the typed-decision smoke answered %d results (schema %q) for %d questions", len(resp.Results), resp.Schema, len(qs))
	}
	for i, q := range qs {
		if resp.Results[i].ID != q.ID {
			return ApplySmoke{}, fmt.Errorf("the typed-decision smoke answered question %q for %q", resp.Results[i].ID, q.ID)
		}
		if err := resp.Results[i].Validate(q); err != nil {
			return ApplySmoke{}, fmt.Errorf("the typed-decision smoke answered an invalid decision: %w", err)
		}
	}
	r := resp.Results[0]
	out := ApplySmoke{Question: qs[0].ID, Choice: r.Choice, Confidence: r.Confidence}
	if resp.Timing != nil {
		out.InferenceMS = resp.Timing.InferenceMS
	}
	return out, nil
}

// smokeRecorder is the minimal http.ResponseWriter the smoke reads its answer
// from.
type smokeRecorder struct {
	header http.Header
	code   int
	body   bytes.Buffer
}

func (r *smokeRecorder) Header() http.Header         { return r.header }
func (r *smokeRecorder) WriteHeader(code int)        { r.code = code }
func (r *smokeRecorder) Write(b []byte) (int, error) { return r.body.Write(b) }

// final is the check that follows a passing smoke: the worker that was proven
// is still the one serving (the smoke did not restart it), the record still
// names the variant, and the desired residents and routing policy are what
// they were.
func (t *applyTx) final(rt Runtime, proven memberView) error {
	now := defaultView(viewOf(rt))
	w, pw := now.Status.Worker, proven.Status.Worker
	if !now.Running || !w.Ready || w.State != worker.StateReady || w.PID != pw.PID || w.Starts != pw.Starts {
		return fmt.Errorf("after the smoke the serving worker is %s (pid %d, start %d), not the proven READY worker (pid %d, start %d)", w.State, w.PID, w.Starts, pw.PID, pw.Starts)
	}
	if err := t.recordNamesTarget(); err != nil {
		return err
	}
	after := t.c.servingState(t.h(), rt)
	switch {
	case !slices.Equal(after.desired, t.before.desired):
		return errors.New("the desired resident selection changed during the apply")
	case after.routing != t.before.routing:
		return errors.New("the routing policy changed during the apply")
	}
	return nil
}

// rollback restores the previous serving target and verifies it: the exact
// previous activation record, the runtime bound from it, the residents that
// were running and nothing else. A runtime that was stopped is not started to
// prove the rollback. It reports every step that failed; it never returns nil
// for a state it could not verify.
func (t *applyTx) rollback(parent context.Context) error {
	c, h := t.c, t.h()
	c.phase(t.op, PhaseApplyRollback)
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), c.restoreTimeout())
	defer cancel()
	var errs []error
	fail := func(step string, err error) { errs = append(errs, fmt.Errorf("%s: %w", step, err)) }

	if t.rebound {
		c.mu.Lock()
		cur := c.rt
		c.mu.Unlock()
		if cur != nil && cur.Running() {
			cur.Stop()
		}
	}
	if t.mutated {
		if err := h.RestoreActiveRecord(t.prevRecord); err != nil {
			fail("restoring the activation record", err)
		} else if got, err := h.ReadActiveRecord(); err != nil || !bytes.Equal(got, t.prevRecord) {
			fail("verifying the restored activation record", errors.New("it is not byte for byte the previous record"))
		}
	}
	if t.residentsMutated {
		if c.cfg.SetResidents == nil {
			fail("restoring the desired resident selection", errors.New("the settings authority has no resident setter"))
		} else if err := c.cfg.SetResidents(slices.Clone(t.prevResidents)); err != nil {
			fail("restoring the desired resident selection", err)
		} else if c.cfg.Residents == nil {
			fail("verifying the restored desired resident selection", errors.New("the settings authority has no resident reader"))
		} else if got, err := settings.NormalizeResidents(c.cfg.Residents()); err != nil {
			fail("verifying the restored desired resident selection", err)
		} else if !slices.Equal(got, t.prevResidents) {
			fail("verifying the restored desired resident selection", fmt.Errorf("got %v, wanted %v", got, t.prevResidents))
		}
	}
	rt := t.prevRT
	if t.rebound {
		var err error
		if rt, err = c.cfg.Open(t.root); err != nil {
			fail("binding the previous runtime", err)
			return errors.Join(errs...)
		}
		started := t.restart(rt)
		c.mu.Lock()
		c.rt, c.pending = rt, false
		c.stopped = !started && t.stoppedBefore
		c.mu.Unlock()
	}
	if rt != nil && len(t.prevRunning) > 0 {
		views, err := c.awaitServing(ctx, rt, t.prevRunning, c.restoreTimeout(), t.scrub)
		if err != nil {
			fail("waiting for the previous residents", err)
		} else if dv := defaultView(views); slices.Contains(t.prevRunning, dv.Model) {
			if err := t.verifyRestored(dv); err != nil {
				fail("verifying the restored target", err)
			}
		}
	}
	if rt != nil {
		for _, v := range viewOf(rt) {
			if running := slices.Contains(t.prevRunning, v.Model); v.Running != running {
				fail("verifying the restored residents", fmt.Errorf("%s running=%v, was running=%v", viewName(v), v.Running, running))
			}
		}
	}
	if err := t.before.changed(c.servingState(h, rt)); err != nil {
		fail("verifying the restored state", err)
	}
	return errors.Join(errs...)
}

// restart starts exactly the members that were running before the apply and
// reports whether it started any.
func (t *applyTx) restart(rt Runtime) bool {
	if len(t.prevRunning) == 0 {
		return false
	}
	if rr, ok := rt.(ResidentRuntime); ok {
		for _, m := range t.prevRunning {
			_, _ = rr.StartResident(m)
		}
		return true
	}
	rt.Start()
	return true
}

// verifyRestored proves the restored default resident executes the target the
// previous activation record names: the variant it named, or the source
// artifact.
func (t *applyTx) verifyRestored(def memberView) error {
	a, info := t.prevActive, def.Status.Worker.Info
	model, err := setup.ActiveModel(a)
	if err != nil {
		return err
	}
	v, ok, err := t.h().LoadVariant(a)
	if err != nil {
		return err
	}
	if ok {
		return verifyExecution(ExecutionTarget{Kind: eval.ForgeTargetVariant, Model: model.ID, Variant: v.ID, Device: a.Device}, model, &v, info)
	}
	if id := str(info, "model_id"); id != "" && id != model.ID {
		return fmt.Errorf("the restored worker reports source model %q, not %q", id, model.ID)
	}
	if dev := str(info, "device"); dev != a.Device && !(a.Device == "cuda" && strings.HasPrefix(dev, "cuda")) {
		return fmt.Errorf("device %q was restored but the worker is on %q", a.Device, dev)
	}
	if str(info, "variant_id") != "" || str(info, "execution") == "variant" || num(info, "weights_quantized_modules") != 0 {
		return fmt.Errorf("the previous source artifact was restored but the worker runs variant %q", str(info, "variant_id"))
	}
	return nil
}
