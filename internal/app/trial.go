package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/trial"
	"github.com/yohn-jp/hachidori/internal/tuning"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// A tuning trial session is RAM-resident, ephemeral experiment execution: one
// resident worker holds the pinned source model on the accelerator and its
// canonical weights in system RAM, and a sequence of resolved tuning plans is
// applied to it as bounded in-place deltas (internal/trial), each measured
// through the same resident evaluation Forge execution uses. Nothing here
// writes a model artifact; every measured trial is recorded as a Candidate with
// trial Evidence, never as a Variant and never as certification.
//
// It shares the accelerator with serving the way every GPU model-engineering
// operation does (Controller.RunTrials owns it for the whole session), and,
// like a Forge execution session, never reads or writes the activation record,
// the desired residents or the routing policy.

// OpTuningTrial is the controller operation of a trial session.
const OpTuningTrial = "tuning_trial"

// Phases of a trial session as the operation reports them.
const (
	TrialPhaseResolve setup.Phase = "resolve_inputs"
	TrialPhaseStart   setup.Phase = "starting"
	TrialPhaseTrials  setup.Phase = "trials"
	TrialPhaseRecord  setup.Phase = "recording"
)

// TrialParams are the semantic inputs of one trial session.
type TrialParams struct {
	// Source is the catalog model; its pinned source is what executes.
	Source string
	// Device is the accelerator the trials run on: explicit, never defaulted.
	Device string
	// Profiles are the saved layer-wise tuning profiles to trial, in order. Each
	// is resolved to its plan before the worker starts.
	Profiles []string
	// Dataset, Questions, Policy, Warmup and Passes are the evaluation intent,
	// resolved exactly as a Forge certification resolves them.
	Dataset        string
	Questions      []string
	Policy         string
	Warmup, Passes int
	// BudgetBytes is the explicit system-RAM budget of the session's canonical
	// source and transformed components. There is no default: capacity is the
	// operator's declaration, never assumed.
	BudgetBytes int64
	// AllowReconstruct permits the explicit slower reconstruction when a delta
	// cannot be applied in place; each such trial records it.
	AllowReconstruct bool
}

// TrialOutcome is what became of one profile of a session.
type TrialOutcome struct {
	Profile string
	// Result is set exactly when the trial was measured and committed.
	Result      *trial.Result
	CandidateID string
	EvidenceID  string
	// Failure says why a trial produced no measurement; Restored whether the
	// model was returned to the previous trial's state.
	Failure  string
	Restored bool
}

// TrialRun is the outcome of a session.
type TrialRun struct {
	Outcomes []TrialOutcome
	Session  trial.SessionStats
	Opened   trial.Opened
	// StartMS is the time to bring the worker READY with the source on the
	// accelerator, paid once per session; SessionMS the whole session.
	StartMS, SessionMS float64
	// Broken is set when the session could not restore a valid model state.
	Broken bool
}

// TrialDeps are the replaceable parts of RunTrials.
type TrialDeps struct {
	// Config resolves the launch of the trial worker for the pinned source.
	Config func(h home.Home, model, device string, log io.Writer) (worker.Config, server.Runtime, error)
	Now    func() time.Time
}

func (d TrialDeps) withDefaults() TrialDeps {
	if d.Config == nil {
		d.Config = func(h home.Home, model, device string, log io.Writer) (worker.Config, server.Runtime, error) {
			cfg, rt, err := server.SourceConfig(h, device, model, trialDType, log)
			if err != nil {
				return cfg, rt, err
			}
			cfg.Args = append(cfg.Args, "--trial-session")
			return cfg, rt, nil
		}
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	return d
}

// trialDType is the dtype of a trial session: the one a W4A16 variant computes
// in, so a trial and the variant built from its plan compute in the same dtype.
const trialDType = "bfloat16"

// trialRequestTimeout bounds one worker request of a trial session. Building a
// component and moving a delta can exceed a decision's two minutes.
const trialRequestTimeout = 10 * time.Minute

// verifyTrialWorker proves from what the READY worker reported that it is the
// ephemeral trial session of the requested pinned source on the requested
// device, and nothing else: not a Variant, not the plain source.
func verifyTrialWorker(model home.ModelManifest, device string, info map[string]any) error {
	switch {
	case str(info, "model_id") != model.ID:
		return fmt.Errorf("the worker reports source model %q, not %q", str(info, "model_id"), model.ID)
	case str(info, "model_revision") != model.Revision:
		return fmt.Errorf("the worker reports revision %q of %s, not the pinned %q", str(info, "model_revision"), model.ID, model.Revision)
	case str(info, "variant_id") != "":
		return fmt.Errorf("a trial session was requested but the worker runs variant %q", str(info, "variant_id"))
	case str(info, "execution") != trial.ExecutionTrial || info["trial_session"] != true:
		return fmt.Errorf("the worker reports execution %q, not a tuning trial session", str(info, "execution"))
	case str(info, "trial_transform") != trial.TransformImplementation:
		return fmt.Errorf("the worker's trial transformation is %q, this build's is %q", str(info, "trial_transform"), trial.TransformImplementation)
	case normDType(str(info, "dtype")) != trialDType:
		return fmt.Errorf("the worker computes in %q, a trial session computes in %s", normDType(str(info, "dtype")), trialDType)
	}
	if dev := str(info, "device"); dev != device && !(device == "cuda" && len(dev) >= 4 && dev[:4] == "cuda") {
		return fmt.Errorf("device %q was requested but the worker is on %q; there is no fallback to another device", device, dev)
	}
	return nil
}

type trialPlan struct {
	profile tuning.Profile
	ref     trial.ProfileRef
	plan    home.TuningPlan
}

// resolveTrialPlans checks every input of a session before anything starts: the
// source, the device, the budget, the evaluation input and every profile, each
// compiled to the exact plan it will be trialled as.
func resolveTrialPlans(h home.Home, p TrialParams) (home.ModelManifest, ExecutionInput, []trialPlan, error) {
	if p.Source == "" {
		return home.ModelManifest{}, ExecutionInput{}, nil, errors.New("a trial session names its source model")
	}
	model, err := setup.LookupModel(p.Source)
	if err != nil {
		return model, ExecutionInput{}, nil, err
	}
	if !setup.SupportsVariants(model) || model.Provider != home.ProviderClef {
		return model, ExecutionInput{}, nil, fmt.Errorf("model %s has no tuning trials (only Clef System One models are tuned)", model.ID)
	}
	if p.Device != "cpu" && p.Device != "cuda" {
		return model, ExecutionInput{}, nil, fmt.Errorf("the trial device must be cpu or cuda, got %q (there is no default and no fallback)", p.Device)
	}
	if p.BudgetBytes <= 0 {
		return model, ExecutionInput{}, nil, errors.New("a trial session needs an explicit system-RAM budget in bytes")
	}
	if len(p.Profiles) == 0 {
		return model, ExecutionInput{}, nil, errors.New("a trial session names at least one saved tuning profile")
	}
	inputs, err := resolveForgeCertInputs(model, ForgeCertifyParams{Dataset: p.Dataset, Questions: p.Questions, Policy: p.Policy, Warmup: p.Warmup, Passes: p.Passes})
	if err != nil {
		return model, ExecutionInput{}, nil, err
	}
	var plans []trialPlan
	for _, id := range p.Profiles {
		profile, analysis, err := tuning.LoadProfile(h, id)
		if err != nil {
			return model, ExecutionInput{}, nil, fmt.Errorf("profile %s: %w", id, err)
		}
		if profile.Source != home.SourceOf(model) {
			return model, ExecutionInput{}, nil, fmt.Errorf("profile %s is bound to a different source model", id)
		}
		compiled, err := tuning.Compile(profile, analysis)
		if err != nil {
			return model, ExecutionInput{}, nil, fmt.Errorf("profile %s: %w", id, err)
		}
		if compiled.Plan == nil {
			return model, ExecutionInput{}, nil, fmt.Errorf("profile %s is a legacy coarse profile with no resolved plan; upgrade it to a layer-wise profile to trial it", id)
		}
		plans = append(plans, trialPlan{profile: profile, plan: *compiled.Plan,
			ref: trial.ProfileRef{ID: profile.ID(), SHA256: profile.SHA256(), AnalysisSHA256: analysis.SHA256(), CompilerVersion: profile.CompilerVersion}})
	}
	return model, inputs.input, plans, nil
}

// residentEvaluator measures the assembled trial with the real resident
// evaluation, through the worker the session drives.
type residentEvaluator struct {
	cl       *client.Client
	set      *ResidentSet
	model    string
	input    ExecutionInput
	terminal bool
}

// cancellable stops sending requests once ctx is done, so a cancelled trial
// ends its evaluation at the next request instead of after the whole dataset.
type cancellable struct {
	*client.Client
	ctx context.Context
}

func (c cancellable) Decide(req api.DecideRequest) (api.DecideResponse, error) {
	if err := c.ctx.Err(); err != nil {
		return api.DecideResponse{}, err
	}
	return c.Client.Decide(req)
}

func (e *residentEvaluator) Evaluate(ctx context.Context, _ trial.Assembled) (trial.Measurement, error) {
	in := e.input
	e.terminal = false
	workerGone := executionTerminal(e.set, &e.terminal)
	run, err := eval.RunResident(cancellable{e.cl, ctx}, in.Cases, in.DatasetSHA256, in.Labelled, e.model,
		eval.ResidentOptions{Options: eval.Options{Warmup: in.Warmup, Passes: in.Passes}, HighConfidence: in.HighConfidence,
			Terminal: func(err error) bool { return ctx.Err() != nil || workerGone(err) }})
	if err := ctx.Err(); err != nil {
		return trial.Measurement{}, fmt.Errorf("trial evaluation cancelled: %w", err)
	}
	if err != nil {
		return trial.Measurement{}, err
	}
	if e.terminal || !run.Run.ResidentStable {
		return trial.Measurement{}, errors.New("the worker was not provably the same one throughout the trial evaluation; nothing is recorded")
	}
	return trial.MeasurementOf(run)
}

// RunTrials runs one trial session. The worker it starts is gone when it
// returns. Every profile that was measured is recorded as a Candidate with trial
// Evidence; a failed or cancelled trial leaves the model in the previous trial's
// state (the outcome says so) and records nothing. Failing to restore a valid
// state ends the session and is reported, never hidden.
func RunTrials(ctx context.Context, h home.Home, p TrialParams, deps TrialDeps, log io.Writer, obs *setup.Observer) (run TrialRun, err error) {
	deps = deps.withDefaults()
	began := deps.Now()
	if obs != nil {
		obs.Phase(TrialPhaseResolve)
	}
	model, input, plans, err := resolveTrialPlans(h, p)
	if err != nil {
		return run, err
	}
	if err := h.Ensure(); err != nil {
		return run, err
	}
	cfg, rt, err := deps.Config(h, model.ID, p.Device, log)
	if err != nil {
		return run, err
	}
	if rt.ModelID != model.ID || rt.Variant != nil {
		return run, fmt.Errorf("the launch resolved for the trial session is not the pinned source %s", model.ID)
	}
	cfg.RequestTimeout = trialRequestTimeout

	if obs != nil {
		obs.Phase(TrialPhaseStart)
	}
	// The worker outlives a cancellation of ctx: the session must still be able
	// to put the model back before it is torn down (deferred below).
	set, err := NewResidentSet(context.WithoutCancel(ctx), worker.Policy{Window: time.Minute, QueueDepth: 64}, model.ID,
		[]ResidentMember{{Model: model.ID, Provider: model.Provider, Info: rt, Config: cfg}})
	if err != nil {
		return run, err
	}
	set.Start()
	defer set.Stop()
	scrub := redact.New(h.Root)
	snap, err := awaitReady(ctx, set)
	if err != nil {
		if snap.LastFailure != nil {
			err = fmt.Errorf("%w: %s %s", err, snap.LastFailure.Class, scrub.Line(snap.LastFailure.Message, 512))
		}
		return run, err
	}
	if err := verifyTrialWorker(model, p.Device, snap.Info); err != nil {
		return run, fmt.Errorf("trial provenance: %w", err)
	}
	run.StartMS = float64(deps.Now().Sub(began)) / float64(time.Millisecond)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return run, err
	}
	srv := &http.Server{Handler: server.HandlerSince(set, set.Runtime(), set.Started())}
	go srv.Serve(ln)
	defer srv.Close()

	session, err := trial.Open(ctx, &trial.WorkerBackend{Caller: set.Default().Supervisor}, home.SourceOf(model), plans[0].plan,
		trial.Options{BudgetBytes: p.BudgetBytes, AllowReconstruct: p.AllowReconstruct, Clock: deps.Now})
	if err != nil {
		return run, fmt.Errorf("opening the trial session: %w", err)
	}
	defer func() {
		run.Session = session.Stats()
		run.Broken = run.Session.Broken
		run.SessionMS = float64(deps.Now().Sub(began)) / float64(time.Millisecond)
		_ = session.Close(context.WithoutCancel(ctx))
	}()
	run.Opened = session.Opened()

	if obs != nil {
		obs.Phase(TrialPhaseTrials)
	}
	ev := &residentEvaluator{cl: client.New("http://" + ln.Addr().String()), set: set, model: model.ID, input: input}
	for _, tp := range plans {
		if cerr := ctx.Err(); cerr != nil {
			return run, fmt.Errorf("trial session cancelled: %w", cerr)
		}
		out := TrialOutcome{Profile: tp.ref.ID}
		res, terr := session.RunTrial(ctx, tp.plan, ev)
		if terr != nil {
			out.Failure = scrub.Line(terr.Error(), 1024)
			var te *trial.Error
			out.Restored = errors.As(terr, &te) && te.Restored
			run.Outcomes = append(run.Outcomes, out)
			if errors.Is(terr, trial.ErrBroken) || (te != nil && !te.Restored) {
				return run, fmt.Errorf("the trial session cannot continue: %s", out.Failure)
			}
			if cerr := ctx.Err(); cerr != nil {
				return run, fmt.Errorf("trial session cancelled: %w", cerr)
			}
			continue
		}
		if obs != nil {
			obs.Phase(TrialPhaseRecord)
		}
		cand, err := trial.NewCandidate(home.SourceOf(model), tp.ref, res, deps.Now())
		if err == nil {
			var evi trial.Evidence
			if evi, err = trial.NewEvidence(cand, res, deps.Now()); err == nil {
				if err = trial.Save(h, cand, evi); err == nil {
					out.CandidateID, out.EvidenceID = cand.ID, evi.ID()
				}
			}
		}
		if err != nil {
			out.Failure = "the measured trial could not be recorded: " + err.Error()
		}
		r := res
		out.Result, out.Restored = &r, true
		run.Outcomes = append(run.Outcomes, out)
	}
	return run, nil
}

// linkTrialCandidates records a freshly built Variant as the materialization of
// the stored trial Candidates that name its exact source, profile and resolved
// plan. It is best effort and never changes the build's outcome: the Variant is
// complete and immutable either way, and a link that could not be written is
// only reported in the operation log.
func linkTrialCandidates(h home.Home, v home.VariantManifest, log io.Writer) {
	linked, err := trial.LinkVariant(h, v, time.Now())
	for _, id := range linked {
		fmt.Fprintf(log, "variant %s is the materialization of trial candidate %s\n", v.ID, id)
	}
	if err != nil {
		fmt.Fprintf(log, "trial candidates of variant %s could not be linked: %v\n", v.ID, err)
	}
}
