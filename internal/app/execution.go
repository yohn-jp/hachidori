package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// A Forge execution session runs one exact target as temporary maintenance
// work: it starts an isolated worker for the pinned source model or for one
// persisted variant on an explicit device, proves that the worker that became
// READY is that target, runs the existing resident evaluation (eval.RunResident)
// over the supplied normalized cases through that worker, records the
// ResidentRun as internal evidence (eval.ForgeRun) and tears the worker down.
//
// It is neither an activation nor a certification. It never reads or writes
// the activation record, the desired residents or the routing policy, never
// requires a variant to be certified, and never falls back to another artifact,
// dtype or device. Sharing the accelerator with the serving residents is the
// Controller's concern (Controller.Execute).

// Execution phases of the controller operation.
const (
	PhaseExecQuiesce = "quiescing"
	PhaseExecRun     = "executing"
	PhaseExecRestore = "restoring"
)

// ExecutionTarget names the exact thing executed.
type ExecutionTarget struct {
	Kind    string // eval.ForgeTargetSource | eval.ForgeTargetVariant
	Model   string // the catalog source model; for a variant optional, and if given it must be the variant's source
	Variant string // the persisted variant ID (variant targets only)
	Device  string // "cpu" or "cuda": explicit, never defaulted
	// DType is the explicit reference dtype of a source whose provider has a
	// dtype control (float32 or bfloat16) and empty for every other target.
	DType string
}

func (t ExecutionTarget) String() string {
	if t.Kind == eval.ForgeTargetVariant {
		return setup.KindVariant + " " + t.Variant + " on " + t.Device
	}
	return "source " + t.Model + " on " + t.Device
}

// ExecutionInput is the normalized semantic evaluation input: cases already
// resolved against their Question Definitions (eval.LoadAny), the dataset's
// digest and whether the cases carry labels.
type ExecutionInput struct {
	Dataset        string // a label for the dataset (its path or name)
	DatasetSHA256  string
	Cases          []eval.Case
	Labelled       bool
	Warmup, Passes int
	HighConfidence float64
}

// ExecuteParams are the inputs of one execution session.
type ExecuteParams struct {
	Target ExecutionTarget
	Input  ExecutionInput
}

// ExecutionResult is the stable reference to the evidence a session recorded.
// Callers load the run with eval.LoadForgeRun(h, EvidenceID); they never choose
// or see a path.
type ExecutionResult struct {
	EvidenceID   string
	Target       eval.ForgeRunTarget
	Observations int
	ErrorCount   int
	// Quiesced and Restored are the Hachidori-owned residents the session
	// stopped for the accelerator and the ones it brought back (Controller.Execute).
	Quiesced []string
	Restored []string
}

// ExecutionError is a failed session. Primary is why the execution failed (nil
// when it succeeded); Restore is a failure to put the serving configuration
// back, which is reported beside the primary failure and never instead of it.
type ExecutionError struct {
	Primary error
	Restore error
}

func (e *ExecutionError) Error() string {
	switch {
	case e.Primary == nil:
		return "the execution succeeded but the previous serving configuration was not restored: " + e.Restore.Error()
	case e.Restore == nil:
		return e.Primary.Error()
	}
	return e.Primary.Error() + "; additionally the previous serving configuration was not restored: " + e.Restore.Error()
}

func (e *ExecutionError) Unwrap() []error {
	var errs []error
	for _, err := range []error{e.Primary, e.Restore} {
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errs
}

// ExecutionDeps are the replaceable parts of RunExecution.
type ExecutionDeps struct {
	// Config resolves the launch of the isolated worker: server.SourceConfig for
	// a source, server.ProbeConfig for a variant.
	Config    func(h home.Home, t ExecutionTarget, log io.Writer) (worker.Config, server.Runtime, error)
	Preflight optimize.PreflightDeps
	Now       func() time.Time
}

func (d ExecutionDeps) withDefaults() ExecutionDeps {
	if d.Config == nil {
		d.Config = func(h home.Home, t ExecutionTarget, log io.Writer) (worker.Config, server.Runtime, error) {
			if t.Kind == eval.ForgeTargetVariant {
				return server.ProbeConfig(h, t.Device, t.Variant, log)
			}
			return server.SourceConfig(h, t.Device, t.Model, t.DType, log)
		}
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	return d
}

// resolveTarget checks the target is exact and resolves it through the
// catalog and the persisted variants: the source manifest and, for a variant,
// its manifest.
func resolveTarget(h home.Home, t ExecutionTarget) (home.ModelManifest, *home.VariantManifest, error) {
	if t.Device != "cpu" && t.Device != "cuda" {
		return home.ModelManifest{}, nil, fmt.Errorf("execution device must be cpu or cuda, got %q (there is no default and no fallback)", t.Device)
	}
	switch t.Kind {
	case eval.ForgeTargetVariant:
		if t.Variant == "" || t.DType != "" {
			return home.ModelManifest{}, nil, errors.New("a variant target names its variant and no dtype (a variant executes at the dtype its manifest declares)")
		}
		model, v, err := setup.FindVariant(h, t.Variant)
		if err != nil {
			return home.ModelManifest{}, nil, err
		}
		if t.Model != "" && t.Model != model.ID {
			return home.ModelManifest{}, nil, fmt.Errorf("variant %s is a variant of %s, not %s", v.ID, model.ID, t.Model)
		}
		return model, &v, nil
	case eval.ForgeTargetSource:
		if t.Variant != "" {
			return home.ModelManifest{}, nil, errors.New("a source target names no variant")
		}
		model, err := setup.LookupModel(t.Model)
		if err != nil {
			return home.ModelManifest{}, nil, err
		}
		if controlled := model.Provider == home.ProviderClef || model.Provider == setup.ProviderOpenDecider; controlled != (t.DType != "") {
			return home.ModelManifest{}, nil, fmt.Errorf("the dtype of a %s source target must be given explicitly (float32 or bfloat16) when its provider has a dtype control and omitted otherwise, got %q", model.Provider, t.DType)
		}
		return model, nil, nil
	}
	return home.ModelManifest{}, nil, fmt.Errorf("unknown execution target kind %q", t.Kind)
}

// normDType is a worker-reported dtype ("torch.bfloat16") as a plain name.
func normDType(s string) string { return strings.TrimPrefix(s, "torch.") }

// verifyExecution proves from what the worker reported once READY that it is
// the requested target: the source model at its pinned revision, the variant
// (never the source in its place) with its quantized execution, on the
// requested device and dtype.
func verifyExecution(t ExecutionTarget, model home.ModelManifest, v *home.VariantManifest, info map[string]any) error {
	if got := str(info, "model_id"); got != model.ID {
		return fmt.Errorf("the worker reports source model %q, not %q", got, model.ID)
	}
	if got := str(info, "model_revision"); got != model.Revision {
		return fmt.Errorf("the worker reports revision %q of %s, not the pinned %q", got, model.ID, model.Revision)
	}
	if dev := str(info, "device"); dev != t.Device && !(t.Device == "cuda" && strings.HasPrefix(dev, "cuda")) {
		return fmt.Errorf("device %q was requested but the worker is on %q; there is no fallback to another device", t.Device, dev)
	}
	dtype, quantized := normDType(str(info, "dtype")), num(info, "weights_quantized_modules")
	if v == nil {
		switch {
		case str(info, "variant_id") != "" || str(info, "execution") == "variant" || quantized != 0:
			return fmt.Errorf("a source execution was requested but the worker runs variant %q (%v quantized modules)", str(info, "variant_id"), quantized)
		case t.DType != "" && dtype != t.DType:
			return fmt.Errorf("dtype %q was requested but the worker runs %q", t.DType, dtype)
		}
		return nil
	}
	switch {
	case str(info, "variant_id") != v.ID:
		return fmt.Errorf("variant %q was requested but the worker reports %q: it did not load the persisted variant (the source is never substituted)", v.ID, str(info, "variant_id"))
	case str(info, "execution") != "variant" || str(info, "quantization_scheme") != v.Weights.Scheme || quantized == 0:
		return fmt.Errorf("the worker does not report quantized %s execution of variant %s", v.Weights.Scheme, v.ID)
	case dtype != v.Weights.DType:
		return fmt.Errorf("the worker computes in %q, not the variant's declared %q", dtype, v.Weights.DType)
	}
	return nil
}

// RunExecution is one execution session on an exact target. The evidence it
// returns is already persisted; the worker it started is gone when it returns.
// An evaluation that was cancelled, or whose worker was not provably the same
// one throughout, records nothing.
func RunExecution(ctx context.Context, h home.Home, p ExecuteParams, deps ExecutionDeps, log io.Writer) (res ExecutionResult, err error) {
	deps = deps.withDefaults()
	t := p.Target
	model, v, err := resolveTarget(h, t)
	if err != nil {
		return res, err
	}
	if v != nil {
		// Integrity of the source and the variant before anything loads.
		rep := optimize.Preflight(ctx, h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: t.Device}, deps.Preflight, nil)
		_ = SavePreflight(h, rep)
		if rep.Blocked() {
			return res, &setup.PreflightError{Report: rep}
		}
	}
	cfg, rt, err := deps.Config(h, t, log)
	if err != nil {
		return res, err
	}
	if rt.ModelID != model.ID || (v == nil) != (rt.Variant == nil) || (v != nil && (rt.Variant.ID != v.ID || rt.Variant.ManifestSHA256 != v.ManifestSHA256())) {
		return res, fmt.Errorf("the launch resolved for %s is not that target", t)
	}

	// One isolated, unsupervised-for-restart worker: if it dies mid-run the
	// run fails rather than silently continuing on another process.
	set, err := NewResidentSet(ctx, worker.Policy{Window: time.Minute, QueueDepth: 64}, model.ID,
		[]ResidentMember{{Model: model.ID, Provider: model.Provider, Info: rt, Config: cfg}})
	if err != nil {
		return res, err
	}
	set.Start()
	defer set.Stop()
	scrub := redact.New(h.Root)
	snap, err := awaitReady(ctx, set)
	if err != nil {
		if snap.LastFailure != nil {
			err = fmt.Errorf("%w: %s %s", err, snap.LastFailure.Class, scrub.Line(snap.LastFailure.Message, 512))
		}
		return res, err
	}
	// The same snapshot that said READY is what the gate judges.
	if err := verifyExecution(t, model, v, snap.Info); err != nil {
		return res, fmt.Errorf("execution provenance: %w", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return res, err
	}
	srv := &http.Server{Handler: server.HandlerSince(set, set.Runtime(), set.Started())}
	go srv.Serve(ln)
	defer srv.Close()
	in := p.Input
	run, err := eval.RunResident(client.New("http://"+ln.Addr().String()), in.Cases, in.DatasetSHA256, in.Labelled, model.ID,
		eval.ResidentOptions{Options: eval.Options{Warmup: in.Warmup, Passes: in.Passes}, HighConfidence: in.HighConfidence})
	if cerr := ctx.Err(); cerr != nil {
		return res, fmt.Errorf("execution cancelled: %w", cerr)
	}
	if err != nil {
		return res, err
	}
	run.Endpoint, run.Dataset = "forge-execution", in.Dataset
	if !run.Run.ResidentStable {
		return res, errors.New("the execution worker did not stay the same worker throughout the run; nothing is recorded")
	}
	if err := verifyExecution(t, model, v, run.Run.Identity.Provider); err != nil {
		return res, fmt.Errorf("execution provenance of the recorded run: %w", err)
	}
	ft := forgeTarget(t, model, v, rt, run.Run.Identity.Provider)
	rec, err := eval.NewForgeRun(ft, run, deps.Now())
	if err != nil {
		return res, err
	}
	if err := eval.SaveForgeRun(h, rec); err != nil {
		return res, fmt.Errorf("the run could not be recorded: %w", err)
	}
	return ExecutionResult{EvidenceID: rec.ID, Target: ft, Observations: len(run.Run.Observations), ErrorCount: run.Run.ErrorCount}, nil
}

// forgeTarget is the identity a recorded run is bound to: the resolved
// authorities plus what the READY worker reported it actually ran.
func forgeTarget(t ExecutionTarget, model home.ModelManifest, v *home.VariantManifest, rt server.Runtime, info map[string]any) eval.ForgeRunTarget {
	src := home.SourceOf(model)
	ft := eval.ForgeRunTarget{Kind: t.Kind, Model: src.ID, Provider: src.Provider, Repo: src.Repo, Revision: src.Revision, SourceFilesSHA256: src.FilesSHA256,
		Runtime: rt.Runtime, RequestedDevice: t.Device, Device: str(info, "device"), RequestedDType: t.DType, DType: normDType(str(info, "dtype"))}
	if v != nil {
		ft.Variant, ft.VariantManifestSHA256, ft.Recipe, ft.Scheme = v.ID, v.ManifestSHA256(), v.Recipe.Name, v.Weights.Scheme
		ft.Quantization, ft.QuantizedModules = str(info, "quantized_execution"), int(num(info, "weights_quantized_modules"))
	}
	return ft
}

// awaitReady waits for the worker to be READY or to fail. Every decision is
// made from a single Snapshot, which it returns.
func awaitReady(ctx context.Context, set *ResidentSet) (worker.Snapshot, error) {
	for {
		snap := set.Snapshot()
		switch {
		case snap.Ready:
			return snap, nil
		case snap.State == worker.StateFailed:
			return snap, errors.New("the execution worker failed to start")
		}
		select {
		case <-ctx.Done():
			return snap, fmt.Errorf("execution cancelled: %w", ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
}
