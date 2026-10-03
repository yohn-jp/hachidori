package app

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func buildEvaluateParams(e *certEnv) ForgeBuildEvaluateParams {
	return ForgeBuildEvaluateParams{
		Source:       setup.ClefFlash,
		Optimization: optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		Device:       "cuda", Dataset: e.params.Dataset, Questions: append([]string(nil), e.params.Questions...),
		Policy: e.params.Policy, Warmup: e.params.Warmup, Passes: e.params.Passes,
	}
}

func buildEvaluateDeps(t *testing.T, e *certEnv, build func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error)) ForgeBuildEvaluateDeps {
	t.Helper()
	pf := probeDeps(t, "ok", e.marker, "uncertified").Preflight
	certify := e.deps
	certify.Preflight = func(ctx context.Context, root string, p PreflightParams, obs *setup.Observer) (setup.PreflightReport, error) {
		report := optimize.Preflight(ctx, home.Home{Root: root}, optimize.PreflightRequest{
			Kind: p.Kind, Model: p.Model, Recipe: p.Recipe, Variant: p.Variant, Device: p.Device, ReferenceDType: p.ReferenceDType,
		}, pf, obs)
		return report, SavePreflight(home.Home{Root: root}, report)
	}
	return ForgeBuildEvaluateDeps{Build: build, Certify: certify}
}

func setValidActive(t *testing.T, h home.Home, device string) []byte {
	t.Helper()
	model, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := setup.Desired(device)
	if err != nil {
		t.Fatal(err)
	}
	active := home.Active{Runtime: spec.ID(), ModelID: model.ID, Model: setup.ModelDirName(model), Device: device}
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), active); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := h.LoadActive(); err != nil {
		t.Fatal(err)
	}
	raw, err := h.ReadActiveRecord()
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestForgeBuildEvaluateBindsExactBuildAndEvaluationWithoutApplying(t *testing.T) {
	e := newCertEnv(t)
	before := setValidActive(t, e.h, "cuda")
	var built int
	var progress []setup.Progress
	var phases []string
	obs := &setup.Observer{
		OnPhase:    func(p setup.Phase) { phases = append(phases, string(p)) },
		OnProgress: func(p setup.Progress) { progress = append(progress, p) },
	}
	deps := buildEvaluateDeps(t, e, func(_ context.Context, _ home.Home, req optimize.Request, _ io.Writer, op *setup.Observer) (optimize.Result, error) {
		built++
		if req.Model != setup.ClefFlash || req.Recipe != optimize.RecipeClefFlashW4A16 || req.Reproduce {
			t.Fatalf("build request %+v", req)
		}
		op.OnProgress(setup.Progress{Step: setup.StepVerify, Detail: "variant"})
		return optimize.Result{Variant: e.v}, nil
	})
	result, err := RunForgeBuildEvaluate(context.Background(), e.h, func() ForgeBuildEvaluateParams {
		p := buildEvaluateParams(e)
		p.Device = "" // Auto is resolved from the validated activation.
		return p
	}(), deps, io.Discard, obs)
	if err != nil {
		t.Fatal(err)
	}
	if built != 1 || result.Variant.ID != e.v.ID || result.Resolution.Variant != e.v.ID || result.Resolution.CandidateDType != e.v.Weights.DType {
		t.Fatalf("build result %+v, build calls %d", result, built)
	}
	if result.Resolution.CandidateDevice != (ForgeResolvedValue{Mode: ForgeSelectionAuto, Value: "cuda"}) ||
		result.Resolution.ReferenceDevice != (ForgeResolvedValue{Mode: ForgeSelectionAuto, Value: "cpu"}) ||
		result.Resolution.ReferenceDType != (ForgeResolvedValue{Mode: ForgeSelectionAuto, Value: "bfloat16"}) {
		t.Fatalf("execution resolution %+v", result.Resolution)
	}
	wantPhases := []string{"resolve_inputs", "preflight", "build", "resolving", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
	if !reflect.DeepEqual(phases, wantPhases) {
		t.Fatalf("phases %v, want %v", phases, wantPhases)
	}
	if len(progress) == 0 {
		t.Fatal("optimizer progress was not forwarded")
	}
	ref, err := eval.LoadForgeRun(e.h, result.Certification.ReferenceEvidence)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := eval.LoadForgeRun(e.h, result.Certification.CandidateEvidence)
	if err != nil {
		t.Fatal(err)
	}
	if ref.Target.Kind != eval.ForgeTargetSource || ref.Target.Model != setup.ClefFlash || ref.Target.RequestedDevice != "cpu" || ref.Target.RequestedDType != "bfloat16" {
		t.Fatalf("reference evidence %+v", ref.Target)
	}
	if candidate.Target.Kind != eval.ForgeTargetVariant || candidate.Target.Model != setup.ClefFlash || candidate.Target.Variant != e.v.ID || candidate.Target.RequestedDevice != "cuda" {
		t.Fatalf("candidate evidence %+v", candidate.Target)
	}
	after, err := e.h.ReadActiveRecord()
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("Build & evaluate changed activation: %s (%v)", after, err)
	}
}

func TestForgeBuildEvaluateRefusesUnresolvedOrMismatchedIntent(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ForgeBuildEvaluateParams, *certEnv)
		want string
	}{
		{"Auto without a valid activation", func(p *ForgeBuildEvaluateParams, _ *certEnv) { p.Device = "" }, "valid active activation"},
		{"explicit source mismatch", func(p *ForgeBuildEvaluateParams, _ *certEnv) { p.Optimization.Model = "another-source" }, "does not match requested source"},
		{"unsupported explicit device", func(p *ForgeBuildEvaluateParams, _ *certEnv) { p.Device = "gpu" }, "device override is invalid"},
		{"reproduction request", func(p *ForgeBuildEvaluateParams, _ *certEnv) { p.Optimization.Reproduce = true }, "not reproduction"},
		{"bad dataset", func(p *ForgeBuildEvaluateParams, _ *certEnv) { p.Dataset = "/missing/dataset.jsonl" }, "dataset"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := newCertEnv(t)
			p := buildEvaluateParams(e)
			tc.edit(&p, e)
			var builds int
			deps := buildEvaluateDeps(t, e, func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
				builds++
				return optimize.Result{Variant: e.v}, nil
			})
			_, err := RunForgeBuildEvaluate(context.Background(), e.h, p, deps, io.Discard, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) || builds != 0 {
				t.Fatalf("err = %v, builds = %d", err, builds)
			}
		})
	}
}

func TestForgeBuildEvaluateNamesBuildFailureAndCancellation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		ctx   func() (context.Context, context.CancelFunc)
		build func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error)
		want  string
	}{
		{
			name: "optimizer failure", ctx: func() (context.Context, context.CancelFunc) { return context.Background(), func() {} },
			build: func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
				return optimize.Result{}, errors.New("optimizer refused the build")
			},
			want: "optimizer refused the build",
		},
		{
			name: "cancelled build", ctx: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx, cancel
			},
			build: func(ctx context.Context, _ home.Home, _ optimize.Request, _ io.Writer, _ *setup.Observer) (optimize.Result, error) {
				return optimize.Result{}, ctx.Err()
			},
			want: context.Canceled.Error(),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := newCertEnv(t)
			ctx, cancel := tc.ctx()
			defer cancel()
			var phases []string
			obs := &setup.Observer{OnPhase: func(p setup.Phase) { phases = append(phases, string(p)) }}
			deps := buildEvaluateDeps(t, e, tc.build)
			_, err := RunForgeBuildEvaluate(ctx, e.h, buildEvaluateParams(e), deps, io.Discard, obs)
			var stageErr *ForgeBuildEvaluateError
			if !errors.As(err, &stageErr) || stageErr.Phase != string(ForgeBuildPhaseBuild) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %#v", err)
			}
			if !reflect.DeepEqual(phases, []string{"resolve_inputs", "preflight", "build"}) {
				t.Fatalf("phases %v", phases)
			}
		})
	}
}

func TestControllerBuildAndEvaluateRecordsResolvedPlanAndDiagnostic(t *testing.T) {
	e := newCertEnv(t)
	setValidActive(t, e.h, "cuda")
	deps := buildEvaluateDeps(t, e, func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
		return optimize.Result{}, errors.New("optimizer refused the build")
	})
	c := New(Config{Home: e.h.Root, Maintenance: Maintenance{
		Build: deps.Build, Preflight: deps.Certify.Preflight, Materialize: func(string, string, string, io.Writer, *setup.Observer) error {
			return nil
		},
	}})
	t.Cleanup(func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		_ = c.Close(ctx)
	})
	if err := c.BuildAndEvaluate(buildEvaluateParams(e)); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, c)
	if s.Maintenance == nil || s.Maintenance.Kind != OpForgeBuildEvaluate || s.Maintenance.Failure == nil || s.Maintenance.Failure.Phase != string(ForgeBuildPhaseBuild) || s.Maintenance.Failure.Diagnostic == "" {
		t.Fatalf("maintenance %+v", s.Maintenance)
	}
	resolution := s.Maintenance.ForgeResolution
	if resolution == nil || resolution.CandidateDevice != (ForgeResolvedValue{Mode: ForgeSelectionOverride, Value: "cuda"}) {
		t.Fatalf("operation resolution %+v", resolution)
	}
	diag, err := diagnostics.LoadForge(e.h.Root, s.Maintenance.Failure.Diagnostic)
	if err != nil {
		t.Fatal(err)
	}
	if diag.Operation.Kind != OpForgeBuildEvaluate || diag.Failure.Phase != string(ForgeBuildPhaseBuild) || !strings.Contains(diag.Failure.Error, "candidate device override -> cuda") {
		t.Fatalf("diagnostic operation %+v failure %+v", diag.Operation, diag.Failure)
	}
}

func TestForgeBuildEvaluateRefusesAChangedBuiltIdentity(t *testing.T) {
	e := newCertEnv(t)
	params := buildEvaluateParams(e)
	var phases []string
	deps := buildEvaluateDeps(t, e, func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
		other := e.v
		other.Recipe.Name = "other-canonical-recipe"
		return optimize.Result{Variant: other}, nil
	})
	obs := &setup.Observer{OnPhase: func(p setup.Phase) { phases = append(phases, string(p)) }}
	_, err := RunForgeBuildEvaluate(context.Background(), e.h, params, deps, io.Discard, obs)
	var stageErr *ForgeBuildEvaluateError
	if !errors.As(err, &stageErr) || stageErr.Phase != string(ForgeBuildPhaseBuild) || !strings.Contains(err.Error(), "published variant manifest") {
		t.Fatalf("error = %v", err)
	}
	if !reflect.DeepEqual(phases, []string{"resolve_inputs", "preflight", "build"}) {
		t.Fatalf("phases %v", phases)
	}
}

func TestForgeBuildEvaluateSourceProvisionRequiresAuthorization(t *testing.T) {
	e := newCertEnv(t)
	model, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(e.h.Path("models", setup.ModelDirName(model))); err != nil {
		t.Fatal(err)
	}
	p := buildEvaluateParams(e)
	var builds, materializations int
	deps := buildEvaluateDeps(t, e, func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
		builds++
		return optimize.Result{Variant: e.v}, nil
	})
	deps.Certify.Materialize = func(context.Context, string, string, string, io.Writer, *setup.Observer) error {
		materializations++
		return errors.New("test refuses real model download")
	}
	_, err = RunForgeBuildEvaluate(context.Background(), e.h, p, deps, io.Discard, nil)
	var stageErr *ForgeBuildEvaluateError
	if !errors.As(err, &stageErr) || stageErr.Phase != string(ForgeBuildPhasePreflight) || builds != 0 || materializations != 0 {
		t.Fatalf("error = %v, builds = %d, materializations = %d", err, builds, materializations)
	}
	p.Materialize = true
	_, err = RunForgeBuildEvaluate(context.Background(), e.h, p, deps, io.Discard, nil)
	if !errors.As(err, &stageErr) || stageErr.Phase != string(ForgeBuildPhaseProvision) || builds != 0 || materializations != 1 {
		t.Fatalf("authorized provisioning: error = %v, builds = %d, materializations = %d", err, builds, materializations)
	}
}

// A saved tuning profile is built by the same one operation: the plan carries
// the profile's compiled recipe and exact provenance, and an unknown profile
// is refused before anything runs.
func TestForgeBuildEvaluatePlansASavedTuningProfile(t *testing.T) {
	e := newCertEnv(t)
	setValidActive(t, e.h, "cuda")
	source, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	analysis := clefAnalysis(t, source)
	profile, err := tuning.NewDefaultProfile(analysis, "maximum-fidelity")
	if err != nil {
		t.Fatal(err)
	}
	profile.Preservation[tuning.RegionFeedForward] = tuning.PreservationChoice{Mode: tuning.PreservationPinned, Precision: tuning.PreservedPrecision}
	if err := tuning.SaveProfile(e.h, profile, analysis); err != nil {
		t.Fatal(err)
	}
	compiled, err := tuning.Compile(profile, analysis)
	if err != nil {
		t.Fatal(err)
	}
	p := buildEvaluateParams(e)
	p.Optimization.Recipe, p.TuningProfile = "", profile.ID()
	plan, err := resolveForgeBuildPlan(e.h, p)
	if err != nil {
		t.Fatal(err)
	}
	if plan.optimization.Tuning == nil || plan.optimization.Tuning.ProfileID != profile.ID() || plan.optimization.CompiledRecipe == nil ||
		plan.recipe.SHA256() != compiled.Recipe.SHA256() || plan.resolution.Recipe != compiled.Recipe.Name {
		t.Fatalf("plan does not build the exact profile: %+v", plan.optimization)
	}
	p.TuningProfile = strings.Repeat("a", 64)
	if _, err := resolveForgeBuildPlan(e.h, p); err == nil {
		t.Fatal("an unknown tuning profile was planned")
	}
}

// A reference execution whose worker dies surfaces the worker's own failure
// through the whole composed error chain, into the Forge diagnostic, and
// records no run evidence.
func TestForgeBuildEvaluateReferenceWorkerFailureReachesTheDiagnostic(t *testing.T) {
	e := newCertEnv(t)
	e.mode[eval.ForgeTargetSource] = "crash_on_decide"
	deps := buildEvaluateDeps(t, e, func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
		return optimize.Result{Variant: e.v}, nil
	})
	_, err := RunForgeBuildEvaluate(context.Background(), e.h, buildEvaluateParams(e), deps, io.Discard, nil)
	var wf *worker.Failure
	var se *ExecutionStabilityError
	if !errors.As(err, &wf) || wf.Class != worker.ClassCrash || !errors.As(err, &se) || !se.Terminal {
		t.Fatalf("error = %v", err)
	}
	if !strings.Contains(err.Error(), "worker_crash") {
		t.Fatalf("the generic stability message hides the cause: %v", err)
	}
	in, _ := collectForge(e.h.Root, ForgeFailure{Kind: OpForgeBuildEvaluate, Phase: string(CertPhaseReference), Device: "cuda", Model: setup.ClefFlash}, err)
	if in.Resources.WorkerClass != worker.ClassCrash {
		t.Fatalf("diagnostic worker class %q, want %q", in.Resources.WorkerClass, worker.ClassCrash)
	}
	if got := e.recordedRuns(); len(got) != 0 {
		t.Fatalf("an unstable reference left evidence %v", got)
	}
}

// Retrying Build & evaluate after the reference failed does not publish the
// candidate again: the deterministic build contract finds the variant that was
// already published and the optimizer is not invoked a second time.
func TestForgeBuildEvaluateRetryReusesThePublishedVariant(t *testing.T) {
	e := newCertEnv(t)
	// Start from a home whose candidate is not yet published.
	if err := os.RemoveAll(e.h.VariantDir(setup.ClefFlash, e.v.ID)); err != nil {
		t.Fatal(err)
	}
	runner := &optimizetest.Runner{}
	var existing []bool
	build := func(ctx context.Context, h home.Home, req optimize.Request, log io.Writer, obs *setup.Observer) (optimize.Result, error) {
		res, err := optimize.Build(ctx, h, req, optimize.Deps{Runner: runner, OptimizerRuntime: "optimizer-test"}, log, obs)
		existing = append(existing, res.Existing)
		return res, err
	}
	deps := buildEvaluateDeps(t, e, build)

	e.mode[eval.ForgeTargetSource] = "crash_on_decide"
	if _, err := RunForgeBuildEvaluate(context.Background(), e.h, buildEvaluateParams(e), deps, io.Discard, nil); err == nil {
		t.Fatal("the reference failure did not fail the operation")
	}
	if runner.Calls != 1 || len(existing) != 1 || existing[0] {
		t.Fatalf("first attempt: %d optimizer calls, existing %v", runner.Calls, existing)
	}
	published := setup.ListVariants(e.h, false, nil)
	if len(published) != 1 {
		t.Fatalf("published variants %+v", published)
	}

	delete(e.mode, eval.ForgeTargetSource)
	result, err := RunForgeBuildEvaluate(context.Background(), e.h, buildEvaluateParams(e), deps, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	if runner.Calls != 1 || len(existing) != 2 || !existing[1] {
		t.Fatalf("retry: %d optimizer calls, existing %v; the published variant must be reused", runner.Calls, existing)
	}
	if result.Variant.ID != published[0].ID || result.Certification.CandidateEvidence == "" || len(setup.ListVariants(e.h, false, nil)) != 1 {
		t.Fatalf("retry result %+v", result.Variant.ID)
	}
}
