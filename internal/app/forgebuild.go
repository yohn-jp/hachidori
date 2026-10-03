package app

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

const (
	ForgeBuildPhaseResolve   setup.Phase = "resolve_inputs"
	ForgeBuildPhasePreflight setup.Phase = "preflight"
	ForgeBuildPhaseProvision setup.Phase = "provision"
	ForgeBuildPhaseBuild     setup.Phase = "build"

	ForgeSelectionAuto     = "auto"
	ForgeSelectionOverride = "override"
)

// ForgeResolvedValue retains both how an execution value was selected and
// what the transaction resolved it to.
type ForgeResolvedValue struct {
	Mode  string `json:"mode"`
	Value string `json:"value"`
}

// ForgeBuildEvaluateResolution is the exact execution plan used by one
// composed Forge operation.
type ForgeBuildEvaluateResolution struct {
	Source          string             `json:"source"`
	Recipe          string             `json:"recipe"`
	Variant         string             `json:"variant,omitempty"`
	CandidateDevice ForgeResolvedValue `json:"candidate_device"`
	CandidateDType  string             `json:"candidate_dtype,omitempty"`
	ReferenceDevice ForgeResolvedValue `json:"reference_device"`
	ReferenceDType  ForgeResolvedValue `json:"reference_dtype"`
}

// ForgeBuildEvaluateParams is the operator intent for one Build & evaluate
// transaction. Optimization is the canonical catalog-model/recipe request;
// Source must name the same model. Empty execution selections mean Auto.
type ForgeBuildEvaluateParams struct {
	Source       string
	Optimization optimize.Request
	// TuningProfile, when set, names the saved semantic tuning profile the
	// candidate is built from instead of Optimization's canonical recipe. The
	// profile is compiled by the tuning authority and bound into the variant's
	// provenance exactly as a Tuning build is.
	TuningProfile string

	// Device is the candidate device override; empty resolves it from the
	// validated active activation. A non-empty value is never substituted.
	Device string

	ReferenceDevice string
	ReferenceDType  string
	Dataset         string
	Questions       []string
	Policy          string
	Warmup, Passes  int

	// Materialize authorizes the operation to provision a missing source or
	// serving runtime through the existing setup authority.
	Materialize bool
}

// ForgeBuildEvaluateResult contains the immutable candidate, exact execution
// choices, and the certification evidence produced for it.
type ForgeBuildEvaluateResult struct {
	Variant       home.VariantManifest
	Resolution    ForgeBuildEvaluateResolution
	Certification ForgeCertifyResult
}

// ForgeBuildEvaluateError names the composed stage that failed. The wrapped
// cause retains optimizer, preflight, execution-session, and certification
// evidence details.
type ForgeBuildEvaluateError struct {
	Phase      string
	Err        error
	Resolution ForgeBuildEvaluateResolution
}

func (e *ForgeBuildEvaluateError) Error() string {
	msg := fmt.Sprintf("forge build and evaluate failed in phase %s: %v", e.Phase, e.Err)
	if e.Resolution.CandidateDevice.Value != "" {
		msg += " (" + forgeResolutionSummary(e.Resolution) + ")"
	}
	return msg
}

func (e *ForgeBuildEvaluateError) Unwrap() error { return e.Err }

type ForgeBuildEvaluateDeps struct {
	Build func(ctx context.Context, h home.Home, req optimize.Request, log io.Writer, obs *setup.Observer) (optimize.Result, error)
	// Certify supplies the existing preflight, materialization, probe and exact
	// execution authorities used by RunForgeCertification.
	Certify ForgeCertifyDeps
	// OnResolution updates the live Controller operation projection.
	OnResolution func(ForgeBuildEvaluateResolution)
}

func (d ForgeBuildEvaluateDeps) withDefaults() ForgeBuildEvaluateDeps {
	if d.Build == nil {
		d.Build = func(ctx context.Context, h home.Home, req optimize.Request, log io.Writer, obs *setup.Observer) (optimize.Result, error) {
			return optimize.Build(ctx, h, req, optimize.Deps{}, log, obs)
		}
	}
	d.Certify = d.Certify.withDefaults()
	return d
}

type forgeBuildPlan struct {
	source       home.ModelManifest
	optimization optimize.Request
	recipe       home.Recipe
	certify      ForgeCertifyParams
	certInputs   *forgeCertInputs
	resolution   ForgeBuildEvaluateResolution
}

// resolveForgeBuildPlan validates every caller-selected input before the
// operation provisions or builds anything.
func resolveForgeBuildPlan(h home.Home, p ForgeBuildEvaluateParams) (forgeBuildPlan, error) {
	var plan forgeBuildPlan
	if p.Source == "" {
		return plan, errors.New("a Forge build/evaluate operation names its source model")
	}
	model, err := setup.LookupModel(p.Source)
	if err != nil {
		return plan, err
	}
	if !setup.SupportsVariants(model) {
		return plan, fmt.Errorf("model %s has no variants (only System One models are optimized)", model.ID)
	}
	if p.Optimization.Model != model.ID {
		return plan, fmt.Errorf("the optimization model %q does not match requested source %q", p.Optimization.Model, model.ID)
	}
	if p.Optimization.Reproduce {
		return plan, errors.New("a Build & evaluate operation requires a publishing optimization request, not reproduction")
	}
	optimization := optimize.Request{Model: model.ID, Recipe: p.Optimization.Recipe}
	var recipe home.Recipe
	if p.TuningProfile != "" {
		if optimization, err = tunedBuildRequest(h, model, p.TuningProfile); err != nil {
			return plan, err
		}
		recipe = *optimization.CompiledRecipe
	} else if recipe, err = optimize.LookupRecipe(model.ID, p.Optimization.Recipe); err != nil {
		return plan, err
	}
	device, mode, err := resolveCandidateDevice(h, p.Device)
	if err != nil {
		return plan, err
	}
	certify := ForgeCertifyParams{
		Device: device, ReferenceDevice: p.ReferenceDevice, ReferenceDType: p.ReferenceDType,
		Dataset: p.Dataset, Questions: append([]string(nil), p.Questions...), Policy: p.Policy,
		Warmup: p.Warmup, Passes: p.Passes, Materialize: p.Materialize,
	}
	certInputs, err := resolveForgeCertInputs(model, certify)
	if err != nil {
		return plan, err
	}
	refDeviceMode := selectionMode(p.ReferenceDevice)
	refDTypeMode := selectionMode(p.ReferenceDType)
	plan = forgeBuildPlan{
		source: model, optimization: optimization, recipe: recipe,
		certify: certify, certInputs: certInputs,
		resolution: ForgeBuildEvaluateResolution{
			Source: model.ID, Recipe: recipe.Name,
			CandidateDevice: ForgeResolvedValue{Mode: mode, Value: device},
			ReferenceDevice: ForgeResolvedValue{Mode: refDeviceMode, Value: certInputs.reference.Device},
			ReferenceDType:  ForgeResolvedValue{Mode: refDTypeMode, Value: certInputs.reference.DType},
		},
	}
	return plan, nil
}

func resolveCandidateDevice(h home.Home, override string) (device, mode string, err error) {
	if override != "" {
		if _, err := setup.Desired(override); err != nil {
			return "", "", fmt.Errorf("the candidate device override is invalid: %w", err)
		}
		return override, ForgeSelectionOverride, nil
	}
	a, rm, _, err := h.LoadActive()
	if err != nil {
		return "", "", fmt.Errorf("candidate device Auto requires a valid active activation: %w", err)
	}
	model, err := setup.ActiveModel(a)
	if err != nil {
		return "", "", fmt.Errorf("candidate device Auto requires a valid active activation: %w", err)
	}
	if a.ModelID == "" {
		a.ModelID = model.ID
	}
	if _, _, err := h.LoadVariant(a); err != nil {
		return "", "", fmt.Errorf("candidate device Auto requires a valid active activation: %w", err)
	}
	if err := setup.CheckRuntimeCompatibility(a, rm); err != nil {
		return "", "", fmt.Errorf("candidate device Auto requires a valid active activation: %w", err)
	}
	spec, err := setup.Desired(a.Device)
	if err != nil || !rm.Satisfies(spec) {
		if err == nil {
			err = fmt.Errorf("active runtime %s is not the dependency runtime %s required for device %s", a.Runtime, spec.ID(), a.Device)
		}
		return "", "", fmt.Errorf("candidate device Auto requires a valid active activation: %w", err)
	}
	return a.Device, ForgeSelectionAuto, nil
}

// RunForgeBuildEvaluate builds one immutable variant and certifies that exact
// candidate against the source and semantic evaluation inputs selected before
// the build. It never activates the candidate.
func RunForgeBuildEvaluate(ctx context.Context, h home.Home, p ForgeBuildEvaluateParams, deps ForgeBuildEvaluateDeps, log io.Writer, obs *setup.Observer) (res ForgeBuildEvaluateResult, err error) {
	deps = deps.withDefaults()
	phase := string(ForgeBuildPhaseResolve)
	enter := func(next setup.Phase) {
		phase = string(next)
		obs.Phase(next)
	}
	fail := func(cause error) (ForgeBuildEvaluateResult, error) {
		if cerr := ctx.Err(); cerr != nil && !errors.Is(cause, cerr) {
			cause = fmt.Errorf("%w (%v)", cause, cerr)
		}
		return res, &ForgeBuildEvaluateError{Phase: phase, Err: cause, Resolution: res.Resolution}
	}

	enter(ForgeBuildPhaseResolve)
	plan, err := resolveForgeBuildPlan(h, p)
	if err != nil {
		return fail(err)
	}
	res.Resolution = plan.resolution
	if deps.OnResolution != nil {
		deps.OnResolution(res.Resolution)
	}
	fmt.Fprintf(log, "execution selections: candidate device %s -> %s; reference device %s -> %s; reference precision %s -> %s\n",
		res.Resolution.CandidateDevice.Mode, res.Resolution.CandidateDevice.Value,
		res.Resolution.ReferenceDevice.Mode, res.Resolution.ReferenceDevice.Value,
		res.Resolution.ReferenceDType.Mode, selectionValue(res.Resolution.ReferenceDType.Value))

	enter(ForgeBuildPhasePreflight)
	report, err := deps.Certify.Preflight(ctx, h.Root, PreflightParams{Kind: setup.PreflightOptimize, Model: plan.source.ID, Recipe: plan.recipe.Name}, progressOnly(obs))
	if err != nil {
		return fail(err)
	}
	if report.Blocked() {
		if !p.Materialize || !onlyMissingSourceBlocker(report) {
			return fail(&setup.PreflightError{Report: report})
		}
		enter(ForgeBuildPhaseProvision)
		if err := deps.Certify.Materialize(ctx, h.Root, plan.certify.Device, plan.source.ID, log, progressOnly(obs)); err != nil {
			return fail(fmt.Errorf("materializing the selected source: %w", err))
		}
		enter(ForgeBuildPhasePreflight)
		report, err = deps.Certify.Preflight(ctx, h.Root, PreflightParams{Kind: setup.PreflightOptimize, Model: plan.source.ID, Recipe: plan.recipe.Name}, progressOnly(obs))
		if err != nil {
			return fail(err)
		}
		if report.Blocked() {
			return fail(&setup.PreflightError{Report: report})
		}
	}

	enter(ForgeBuildPhaseBuild)
	built, err := deps.Build(ctx, h, plan.optimization, log, progressOnly(obs))
	if err != nil {
		return fail(err)
	}
	if built.Variant.ID == "" {
		return fail(errors.New("the optimizer returned no published variant identity"))
	}
	variantSource, variant, err := setup.FindVariant(h, built.Variant.ID)
	if err != nil {
		return fail(fmt.Errorf("the optimizer's published variant cannot be verified: %w", err))
	}
	if home.SourceOf(variantSource) != home.SourceOf(plan.source) || variant.Recipe.SHA256() != plan.recipe.SHA256() {
		return fail(errors.New("the published variant does not match the requested source and canonical recipe"))
	}
	if variant.ManifestSHA256() != built.Variant.ManifestSHA256() {
		return fail(errors.New("the optimizer result differs from the published variant manifest"))
	}
	res.Variant = variant
	res.Resolution.Variant = variant.ID
	res.Resolution.CandidateDType = variant.Weights.DType

	plan.certify.Variant = variant.ID
	cert, err := runForgeCertification(ctx, h, plan.certify, deps.Certify, log, composedCertificationObserver(obs), plan.certInputs)
	if err != nil {
		var certErr *ForgeCertifyError
		if errors.As(err, &certErr) {
			phase = certErr.Phase
		}
		return fail(err)
	}
	res.Certification = cert
	return res, nil
}

func onlyMissingSourceBlocker(report setup.PreflightReport) bool {
	blockers := report.Blockers()
	if len(blockers) != 1 || blockers[0].ID != "source.manifest" {
		return false
	}
	condition, ok := blockers[0].Facts["condition"].(string)
	return ok && condition == string(setup.SourceAbsent)
}

func composedCertificationObserver(obs *setup.Observer) *setup.Observer {
	if obs == nil {
		return nil
	}
	return &setup.Observer{
		OnProgress: obs.OnProgress,
		OnPhase: func(ph setup.Phase) {
			if ph == CertPhaseMaterializing {
				obs.Phase(ForgeBuildPhaseProvision)
				return
			}
			obs.Phase(ph)
		},
	}
}

func selectionMode(v string) string {
	if v == "" {
		return ForgeSelectionAuto
	}
	return ForgeSelectionOverride
}

func selectionValue(v string) string {
	if v == "" {
		return "provider default"
	}
	return v
}

func forgeResolutionSummary(r ForgeBuildEvaluateResolution) string {
	return fmt.Sprintf("candidate device %s -> %s; reference device %s -> %s; reference precision %s -> %s",
		r.CandidateDevice.Mode, r.CandidateDevice.Value,
		r.ReferenceDevice.Mode, r.ReferenceDevice.Value,
		r.ReferenceDType.Mode, selectionValue(r.ReferenceDType.Value))
}
