package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Self-contained Forge certification: from one persisted variant and the
// semantic evaluation inputs, Hachidori itself produces the evidence of both
// sides and certifies. It composes the existing authorities and owns none of
// their semantics:
//
//	resolving     the exact source, variant, policy, corpus and execution targets
//	preflight     readiness of the reference and of the candidate (optimize.Preflight)
//	materializing the missing deterministic runtime, only when asked (setup.Materialize)
//	probe         the persisted variant answers one typed decision (Probe)
//	reference_run the exact source, executed as maintenance work (RunExecution, #158)
//	candidate_run the exact variant, executed as maintenance work (RunExecution, #158)
//	aligning      both runs are the same evaluation input (eval.AlignForgeRuns)
//	certifying    the existing eval.Certify, its policy and formulas unchanged
//	persisting    the certification record, bound to the two runs' evidence
//
// Certification is evidence only. Nothing here reads, writes or restarts the
// activation record, the desired residents, the routing policy or a resident:
// the accepted and the rejected verdict leave the serving configuration as it
// was.

// Phases of a Forge certification, as the operation reports them. A phase is
// reported only when it is really entered; completed and failed are the end of
// the operation (Operation.Finished and Operation.Failure).
const (
	CertPhaseResolving     setup.Phase = "resolving"
	CertPhasePreflight     setup.Phase = setup.PhasePreflight
	CertPhaseMaterializing setup.Phase = "materializing"
	CertPhaseProbe         setup.Phase = "probe"
	CertPhaseReference     setup.Phase = "reference_run"
	CertPhaseCandidate     setup.Phase = "candidate_run"
	CertPhaseAligning      setup.Phase = "aligning"
	CertPhaseCertifying    setup.Phase = "certifying"
	CertPhasePersisting    setup.Phase = "persisting"
)

// OpForgeCertify is the controller operation of a self-contained Forge
// certification (Controller.CertifyVariant). OpCertify stays the low-level
// certification of two run files.
const OpForgeCertify = "forge_certify"

// The canonical reference defaults: the pinned source at the dtype the release
// ships, on the CPU, where a high-precision reference fits. They are recorded
// as requested in the evidence and never substituted at run time.
const (
	DefaultReferenceDevice = "cpu"
	DefaultReferenceDType  = "bfloat16"
)

// ForgeCertifyParams is the semantic intent of a certification. It names no
// reference or candidate run file: the runs are produced here.
type ForgeCertifyParams struct {
	Variant string // the persisted variant to certify
	// Device is the candidate's device: explicit, never defaulted or substituted.
	Device string
	// ReferenceDevice and ReferenceDType are the source reference's. Empty
	// means the canonical defaults (DefaultReferenceDevice; DefaultReferenceDType
	// for a source whose provider has a dtype control).
	ReferenceDevice string
	ReferenceDType  string
	Dataset         string   // the evaluation corpus (eval JSONL, labelled or not)
	Questions       []string // Question Definition files or directories resolving question_refs
	Policy          string   // certification policy file; empty is the built-in profile
	Warmup, Passes  int      // as for `certify run`; zero passes is one pass
	// Materialize allows the operation to materialize a missing serving
	// runtime through the setup authority. It never activates anything.
	Materialize bool
}

// ForgeCertifyResult is what a completed certification produced.
type ForgeCertifyResult struct {
	Certification     eval.Certification
	Record            eval.CertificationRecord
	ReferenceEvidence string
	CandidateEvidence string
	Probe             ProbeRecord
	// Materialized are the devices whose runtime this operation materialized.
	Materialized []string
	// Quiesced and Restored are the Hachidori-owned residents stopped for the
	// accelerator (by the controller) and brought back.
	Quiesced, Restored []string
}

// ForgeCertifyError is a certification that failed, and where: the phase it
// was in, the cause (whose chain is preserved, an *ExecutionError's restore
// failure included) and the evidence of the runs that did complete before it.
// No certification record exists for a failed operation.
type ForgeCertifyError struct {
	Phase             string
	Err               error
	ReferenceEvidence string
	CandidateEvidence string
}

func (e *ForgeCertifyError) Error() string {
	s := fmt.Sprintf("forge certification failed in phase %s: %v", e.Phase, e.Err)
	if e.ReferenceEvidence != "" {
		s += " (reference run evidence " + e.ReferenceEvidence
		if e.CandidateEvidence != "" {
			s += ", candidate run evidence " + e.CandidateEvidence
		}
		s += ")"
	}
	return s
}

func (e *ForgeCertifyError) Unwrap() error { return e.Err }

// ForgeCertifyDeps are the replaceable parts of RunForgeCertification. Probe
// and Execute are where a controller adds its maintenance lease; the defaults
// are the isolated worker alone, which never looks at a running Hachidori's
// residents.
type ForgeCertifyDeps struct {
	Preflight   func(ctx context.Context, root string, p PreflightParams, obs *setup.Observer) (setup.PreflightReport, error)
	Materialize func(ctx context.Context, root, device, model string, log io.Writer, obs *setup.Observer) error
	Probe       func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error)
	Execute     func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error)
	Now         func() time.Time
}

func (d ForgeCertifyDeps) withDefaults() ForgeCertifyDeps {
	if d.Preflight == nil {
		d.Preflight = RunPreflight
	}
	if d.Materialize == nil {
		d.Materialize = func(ctx context.Context, root, device, model string, log io.Writer, obs *setup.Observer) error {
			return setup.MaterializeContext(ctx, home.Home{Root: root}, device, model, log, obs)
		}
	}
	if d.Probe == nil {
		d.Probe = func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
			return Probe(ctx, home.Home{Root: root}, p, ProbeDeps{}, log, obs)
		}
	}
	if d.Execute == nil {
		d.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
			return RunExecution(ctx, home.Home{Root: root}, p, ExecutionDeps{}, log)
		}
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	return d
}

// certPlan is everything resolved before any expensive work.
type certPlan struct {
	src       home.ModelManifest
	variant   home.VariantManifest
	policy    eval.CertPolicy
	input     ExecutionInput
	reference ExecutionTarget
	candidate ExecutionTarget
}

// forgeCertInputs are the policy and semantic evaluation resources resolved
// once for a composed build/evaluate operation. Keeping these values in memory
// binds certification to the inputs selected before the build began.
type forgeCertInputs struct {
	source    home.ModelManifest
	policy    eval.CertPolicy
	input     ExecutionInput
	reference ExecutionTarget
}

// resolveCertification resolves the exact source, variant, policy, corpus and
// both execution targets, refusing anything that is not exact.
func resolveCertification(h home.Home, p ForgeCertifyParams) (certPlan, error) {
	return resolveCertificationWith(h, p, nil)
}

func resolveCertificationWith(h home.Home, p ForgeCertifyParams, resolved *forgeCertInputs) (certPlan, error) {
	var c certPlan
	if p.Variant == "" {
		return c, errors.New("a certification names the variant to certify")
	}
	if p.Dataset == "" {
		return c, errors.New("a certification needs an evaluation corpus (dataset)")
	}
	if p.Device != "cpu" && p.Device != "cuda" {
		return c, fmt.Errorf("the candidate device must be cpu or cuda, got %q (there is no default and no fallback)", p.Device)
	}
	var err error
	if c.src, c.variant, err = setup.FindVariant(h, p.Variant); err != nil {
		return c, err
	}
	inputs := resolved
	if inputs == nil {
		inputs, err = resolveForgeCertInputs(c.src, p)
		if err != nil {
			return c, err
		}
	} else if home.SourceOf(inputs.source) != home.SourceOf(c.src) {
		return c, fmt.Errorf("the built variant source %s does not match the requested source %s", c.src.ID, inputs.source.ID)
	}
	c.policy, c.input, c.reference = inputs.policy, inputs.input, inputs.reference
	c.candidate = ExecutionTarget{Kind: eval.ForgeTargetVariant, Model: c.src.ID, Variant: c.variant.ID, Device: p.Device}
	for _, t := range []ExecutionTarget{c.reference, c.candidate} {
		if _, _, err := resolveTarget(h, t); err != nil {
			return c, err
		}
	}
	return c, nil
}

// resolveForgeCertInputs validates the semantic inputs and applies the
// existing canonical reference defaults before any expensive build begins.
func resolveForgeCertInputs(src home.ModelManifest, p ForgeCertifyParams) (*forgeCertInputs, error) {
	if p.Dataset == "" {
		return nil, errors.New("a certification needs an evaluation corpus (dataset)")
	}
	inputs := &forgeCertInputs{source: src, policy: eval.DefaultPolicy()}
	if p.Policy != "" {
		policy, err := eval.LoadPolicy(p.Policy)
		if err != nil {
			return nil, err
		}
		inputs.policy = policy
	}
	var defs *question.Set
	if len(p.Questions) > 0 {
		var err error
		if defs, err = question.Load(p.Questions...); err != nil {
			return nil, err
		}
	}
	cases, sum, labelled, err := eval.LoadAny(p.Dataset, defs)
	if err != nil {
		return nil, err
	}
	passes := p.Passes
	if passes == 0 {
		passes = 1
	}
	inputs.input = ExecutionInput{Dataset: p.Dataset, DatasetSHA256: sum, Cases: cases, Labelled: labelled, Warmup: p.Warmup, Passes: passes,
		HighConfidence: inputs.policy.HighConfidence}

	refDevice, refDType := p.ReferenceDevice, p.ReferenceDType
	if refDevice == "" {
		refDevice = DefaultReferenceDevice
	}
	if refDType == "" && (src.Provider == home.ProviderClef || src.Provider == setup.ProviderOpenDecider) {
		refDType = DefaultReferenceDType
	}
	if _, err := setup.Desired(refDevice); err != nil {
		return nil, fmt.Errorf("the reference device must be cpu or cuda: %w", err)
	}
	if refDType != "" && refDType != "float32" && refDType != "bfloat16" {
		return nil, fmt.Errorf("the reference dtype must be float32 or bfloat16 (high precision), got %q", refDType)
	}
	inputs.reference = ExecutionTarget{Kind: eval.ForgeTargetSource, Model: src.ID, Device: refDevice, DType: refDType}
	return inputs, nil
}

// RunForgeCertification certifies one persisted variant end to end: the
// phases listed above, in order, each reported to obs when entered. It returns
// a *ForgeCertifyError naming the phase of any failure. A rejected verdict is
// a completed certification (its evidence is recorded and returned), never an
// error. Cancellation records no certification.
func RunForgeCertification(ctx context.Context, h home.Home, p ForgeCertifyParams, deps ForgeCertifyDeps, log io.Writer, obs *setup.Observer) (res ForgeCertifyResult, err error) {
	return runForgeCertification(ctx, h, p, deps, log, obs, nil)
}

func runForgeCertification(ctx context.Context, h home.Home, p ForgeCertifyParams, deps ForgeCertifyDeps, log io.Writer, obs *setup.Observer, inputs *forgeCertInputs) (res ForgeCertifyResult, err error) {
	deps = deps.withDefaults()
	phase := ""
	enter := func(ph setup.Phase) {
		phase = string(ph)
		obs.Phase(ph)
	}
	// The sub-operations (preflight, probe, setup) report their own phases; the
	// certification's phase is the one it entered, so only their progress is
	// forwarded.
	steps := progressOnly(obs)
	fail := func(err error) (ForgeCertifyResult, error) {
		if cerr := ctx.Err(); cerr != nil && !errors.Is(err, cerr) {
			err = fmt.Errorf("%w (%v)", err, cerr)
		}
		return res, &ForgeCertifyError{Phase: phase, Err: err, ReferenceEvidence: res.ReferenceEvidence, CandidateEvidence: res.CandidateEvidence}
	}

	enter(CertPhaseResolving)
	plan, err := resolveCertificationWith(h, p, inputs)
	if err != nil {
		return fail(err)
	}

	preflight, err := certPreflight(ctx, h, p, plan, deps, &res, enter, log, steps)
	if err != nil {
		return fail(err)
	}

	enter(CertPhaseProbe)
	if res.Probe, err = deps.Probe(ctx, h.Root, ProbeParams{Variant: plan.variant.ID, Device: p.Device}, log, steps); err != nil {
		return fail(fmt.Errorf("the variant probe failed; no certification run was started: %w", err))
	}
	if res.Probe.Result != ProbePassed {
		return fail(fmt.Errorf("the variant probe did not pass (%s); no certification run was started", res.Probe.Result))
	}

	enter(CertPhaseReference)
	ref, err := certRun(ctx, h, plan, plan.reference, "reference", deps, &res, log)
	if err != nil {
		return fail(err)
	}
	enter(CertPhaseCandidate)
	cand, err := certRun(ctx, h, plan, plan.candidate, "candidate", deps, &res, log)
	if err != nil {
		return fail(err)
	}

	enter(CertPhaseAligning)
	if err := eval.AlignForgeRuns(ref, cand); err != nil {
		return fail(err)
	}
	for role, r := range map[string]eval.ForgeRun{"reference": ref, "candidate": cand} {
		if r.DatasetSHA256 != plan.input.DatasetSHA256 || r.Run.Run.Cases != len(plan.input.Cases) || r.Run.Labelled != plan.input.Labelled {
			return fail(fmt.Errorf("the %s run %s is not over the requested corpus (dataset %.12s, %d cases; requested %.12s, %d cases)",
				role, r.ID, r.DatasetSHA256, r.Run.Run.Cases, plan.input.DatasetSHA256, len(plan.input.Cases)))
		}
	}

	enter(CertPhaseCertifying)
	cert, err := eval.Certify(eval.CertifyInput{Source: plan.src, Variant: plan.variant, Reference: ref.Run, Candidate: cand.Run, Policy: plan.policy, Now: deps.Now()})
	if err != nil {
		return fail(err)
	}
	cert.Producer = eval.ProducerOf(ref, cand)
	cert.Producer.Preflight = preflight
	cert.Producer.Probe = &eval.ProducerCheck{Result: res.Probe.Result, Device: res.Probe.Device, At: res.Probe.FinishedAt}

	enter(CertPhasePersisting)
	rec, err := eval.SaveCertification(h, cert)
	if err != nil {
		return fail(fmt.Errorf("the certification could not be recorded: %w", err))
	}
	res.Certification, res.Record = cert, rec
	eval.CertificationSummary(log, cert)
	fmt.Fprintf(log, "certification recorded: %s verdict %s (reference run %s, candidate run %s); nothing was activated\n",
		rec.Report, rec.Verdict, ref.ID, cand.ID)
	return res, nil
}

// progressOnly forwards the progress of a step but not its phases.
func progressOnly(o *setup.Observer) *setup.Observer {
	if o == nil {
		return nil
	}
	return &setup.Observer{OnProgress: o.OnProgress}
}

// certPreflight is the readiness gate: the reference's and the candidate's
// preflight, and, when the only blockers are runtimes that are not
// materialized and the caller allowed it, their materialization through the
// setup authority followed by a fresh preflight. Anything else that blocks
// refuses the operation before any expensive work.
func certPreflight(ctx context.Context, h home.Home, p ForgeCertifyParams, plan certPlan, deps ForgeCertifyDeps, res *ForgeCertifyResult,
	enter func(setup.Phase), log io.Writer, obs *setup.Observer) (*eval.ProducerCheck, error) {
	gates := []PreflightParams{
		{Kind: setup.PreflightCertify, Variant: plan.variant.ID, Device: plan.reference.Device, ReferenceDType: plan.reference.DType},
		{Kind: setup.PreflightProbe, Variant: plan.variant.ID, Device: plan.candidate.Device},
	}
	for attempt := 0; ; attempt++ {
		enter(CertPhasePreflight)
		var blocked []setup.PreflightReport
		var last setup.PreflightReport
		for _, g := range gates {
			rep, err := deps.Preflight(ctx, h.Root, g, obs)
			if err != nil {
				return nil, err
			}
			if rep.Blocked() {
				blocked = append(blocked, rep)
			}
			last = rep
		}
		if len(blocked) == 0 {
			return &eval.ProducerCheck{Result: last.Outcome, Device: last.Device, At: last.CreatedAt}, nil
		}
		devices := missingRuntimes(h, blocked)
		if attempt > 0 || !p.Materialize || len(devices) == 0 {
			return nil, &setup.PreflightError{Report: blocked[0]}
		}
		for _, device := range devices {
			enter(CertPhaseMaterializing)
			if err := deps.Materialize(ctx, h.Root, device, plan.src.ID, log, obs); err != nil {
				return nil, fmt.Errorf("materializing the %s runtime: %w", device, err)
			}
			res.Materialized = append(res.Materialized, device)
		}
	}
}

// missingRuntimes are the devices of blocked reports whose serving runtime is
// simply not materialized, provided that is the only thing blocking them.
// Materialization cannot repair any other blocker, so with one present it
// returns nothing.
func missingRuntimes(h home.Home, blocked []setup.PreflightReport) []string {
	var devices []string
	for _, rep := range blocked {
		for _, f := range rep.Blockers() {
			spec, err := setup.Desired(rep.Device)
			if err != nil || f.ID != "runtime.serving" {
				return nil
			}
			if _, err := os.Stat(h.Path("runtime", setup.RuntimeDirFor(h, spec))); !errors.Is(err, os.ErrNotExist) {
				return nil
			}
		}
		if !slices.Contains(devices, rep.Device) {
			devices = append(devices, rep.Device)
		}
	}
	return devices
}

// certRun executes one side of the comparison and returns its recorded run
// after proving that it is exactly the target that was asked for. It records
// the evidence ID of a run that completed even if the run is then refused.
func certRun(ctx context.Context, h home.Home, plan certPlan, t ExecutionTarget, role string, deps ForgeCertifyDeps, res *ForgeCertifyResult, log io.Writer) (eval.ForgeRun, error) {
	er, err := deps.Execute(ctx, h.Root, ExecuteParams{Target: t, Input: plan.input}, log)
	res.Quiesced, res.Restored = append(res.Quiesced, er.Quiesced...), append(res.Restored, er.Restored...)
	// A run that was recorded is named even when the session then failed (for
	// example when the serving residents could not be restored).
	if role == "reference" {
		res.ReferenceEvidence = er.EvidenceID
	} else {
		res.CandidateEvidence = er.EvidenceID
	}
	if err != nil {
		return eval.ForgeRun{}, fmt.Errorf("the %s execution failed: %w", role, err)
	}
	run, err := eval.LoadForgeRun(h, er.EvidenceID)
	if err != nil {
		return run, fmt.Errorf("the %s run evidence cannot be read back: %w", role, err)
	}
	if err := verifyForgeEvidence(plan, t, run); err != nil {
		return run, fmt.Errorf("the %s run %s is not the requested execution: %w", role, run.ID, err)
	}
	return run, nil
}

// verifyForgeEvidence proves from the stored run that it was recorded on
// exactly the execution target t: the pinned source (and, for the candidate,
// the exact variant and its manifest, with quantized execution), the runtime of
// the requested device, and the requested and actual device and dtype. A run
// of the source is never accepted as the candidate, nor the reverse.
func verifyForgeEvidence(plan certPlan, t ExecutionTarget, run eval.ForgeRun) error {
	g, src := run.Target, home.SourceOf(plan.src)
	var why []string
	add := func(f string, a ...any) { why = append(why, fmt.Sprintf(f, a...)) }
	if g.Kind != t.Kind {
		add("it is a %s run, not a %s run", g.Kind, t.Kind)
	}
	if g.Model != src.ID || g.Provider != src.Provider || g.Repo != src.Repo || g.Revision != src.Revision || g.SourceFilesSHA256 != src.FilesSHA256 {
		add("it is not the pinned source %s@%.12s (files %.12s)", src.ID, src.Revision, src.FilesSHA256)
	}
	if spec, err := setup.Desired(t.Device); err != nil || g.Runtime != spec.ID() {
		add("it ran on runtime %q, not the %s runtime", g.Runtime, t.Device)
	}
	if g.RequestedDevice != t.Device || !(g.Device == t.Device || (t.Device == "cuda" && strings.HasPrefix(g.Device, "cuda"))) {
		add("it requested %s and ran on %s, not %s", g.RequestedDevice, g.Device, t.Device)
	}
	switch t.Kind {
	case eval.ForgeTargetSource:
		if g.Variant != "" || g.VariantManifestSHA256 != "" || g.QuantizedModules != 0 {
			add("the reference executed variant %q", g.Variant)
		}
		if g.RequestedDType != t.DType || (t.DType != "" && g.DType != t.DType) {
			add("it requested dtype %q and ran %q, not %q", g.RequestedDType, g.DType, t.DType)
		}
		if g.DType != "float32" && g.DType != "bfloat16" {
			add("the reference ran at %q, not a high-precision dtype", g.DType)
		}
	case eval.ForgeTargetVariant:
		v := plan.variant
		if g.Variant != v.ID || g.VariantManifestSHA256 != v.ManifestSHA256() {
			add("it is variant %q (manifest %.12s), not %s (manifest %.12s)", g.Variant, g.VariantManifestSHA256, v.ID, v.ManifestSHA256())
		}
		if g.Recipe != v.Recipe.Name || g.Scheme != v.Weights.Scheme {
			add("it reports recipe %q scheme %q, not %q and %q", g.Recipe, g.Scheme, v.Recipe.Name, v.Weights.Scheme)
		}
		if g.QuantizedModules == 0 || g.Quantization == "" {
			add("it does not prove quantized %s execution", v.Weights.Scheme)
		}
		if g.DType != v.Weights.DType || g.RequestedDType != "" {
			add("it ran at dtype %q, not the variant's declared %q", g.DType, v.Weights.DType)
		}
	}
	if len(why) > 0 {
		return errors.New(strings.Join(why, "; "))
	}
	return nil
}
