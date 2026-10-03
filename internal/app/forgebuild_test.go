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
	"github.com/yohn-jp/hachidori/internal/setup"
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
