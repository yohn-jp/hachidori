package app

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

const (
	certDataset   = "../../testdata/eval/contract-example-refs.jsonl"
	certQuestions = "../../testdata/questions"
)

var activationBytes = []byte(`{"runtime":"rt","model_id":"clef-flash","device":"cuda"}`)

// certEnv is a fixture home with a built variant, an activation record the
// certification must leave byte for byte alone, and fakes for every
// authority RunForgeCertification composes: real preflight over a healthy fake
// host, the real Probe and RunExecution over a fake worker, and the real
// eval.Certify and certification store.
type certEnv struct {
	t      *testing.T
	h      home.Home
	v      home.VariantManifest
	marker string
	params ForgeCertifyParams
	deps   ForgeCertifyDeps
	phases []string
	obs    *setup.Observer

	probes   int
	executed []string // the kind of every execution, in order
	// mode bends one provenance fact of the fake worker per side.
	mode map[string]string
}

func writePolicy(t *testing.T, mutate func(*eval.CertPolicy)) string {
	t.Helper()
	p := eval.DefaultPolicy()
	p.ID = "fixture-fidelity/1"
	p.MinObservations = 1
	if mutate != nil {
		mutate(&p)
	}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func newCertEnv(t *testing.T) *certEnv {
	t.Helper()
	h, v := forgeHome(t)
	if err := os.WriteFile(h.Path("state", "active-runtime.json"), activationBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	e := &certEnv{t: t, h: h, v: v, marker: t.TempDir(), mode: map[string]string{}}
	e.params = ForgeCertifyParams{Variant: v.ID, Device: "cuda", Dataset: certDataset, Questions: []string{certQuestions},
		Policy: writePolicy(t, nil), Warmup: 1, Passes: 1}
	pf := probeDeps(t, "ok", e.marker, "uncertified").Preflight
	e.deps = ForgeCertifyDeps{
		Preflight: func(ctx context.Context, root string, p PreflightParams, obs *setup.Observer) (setup.PreflightReport, error) {
			rep := optimize.Preflight(ctx, home.Home{Root: root}, optimize.PreflightRequest{Kind: p.Kind, Variant: p.Variant, Device: p.Device, ReferenceDType: p.ReferenceDType}, pf, obs)
			return rep, SavePreflight(home.Home{Root: root}, rep)
		},
		Materialize: func(context.Context, string, string, string, io.Writer, *setup.Observer) error {
			t.Error("a runtime was materialized")
			return nil
		},
		Probe: func(ctx context.Context, _ string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
			e.probes++
			return Probe(ctx, h, p, probeDeps(t, "ok", e.marker, "uncertified"), log, obs)
		},
		Execute: e.execute,
	}
	e.obs = &setup.Observer{OnPhase: func(p setup.Phase) { e.phases = append(e.phases, string(p)) }}
	return e
}

func (e *certEnv) execute(ctx context.Context, _ string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
	e.executed = append(e.executed, p.Target.Kind)
	return RunExecution(ctx, e.h, p, execDeps(e.t, e.mode[p.Target.Kind], e.marker), log)
}

func (e *certEnv) run() (ForgeCertifyResult, error) {
	return RunForgeCertification(context.Background(), e.h, e.params, e.deps, io.Discard, e.obs)
}

// recordedRuns are the evidence IDs on disk.
func (e *certEnv) recordedRuns() []string { return runsOnDisk(e.h) }

func (e *certEnv) certificationFiles() []string {
	es, _ := os.ReadDir(h2dir(e.h, e.v))
	var out []string
	for _, f := range es {
		out = append(out, f.Name())
	}
	return out
}

func h2dir(h home.Home, v home.VariantManifest) string {
	return h.Path(filepath.FromSlash(home.CertificationDir(v.ID)))
}

// untouched asserts that nothing served or selected changed: the activation
// record is byte for byte what it was, and the variant is neither active nor
// accepted by anything but a recorded certification.
func (e *certEnv) untouched() {
	e.t.Helper()
	got, err := os.ReadFile(e.h.Path("state", "active-runtime.json"))
	if err != nil || string(got) != string(activationBytes) {
		e.t.Fatalf("activation record %q (%v), was %q", got, err, activationBytes)
	}
}

func (e *certEnv) noCertification() {
	e.t.Helper()
	if st := eval.ResolveCertification(e.h, e.v); st.State != eval.StateUncertified || len(e.certificationFiles()) != 0 {
		e.t.Fatalf("a certification exists: %s %v", st.State, e.certificationFiles())
	}
}

func asCertError(t *testing.T, err error) *ForgeCertifyError {
	t.Helper()
	var fe *ForgeCertifyError
	if !errors.As(err, &fe) {
		t.Fatalf("err = %v, want a *ForgeCertifyError", err)
	}
	return fe
}

// reseal publishes a copy of a stored run whose target or run was bent, as a
// different execution that nevertheless is a valid, self-consistent document.
func reseal(t *testing.T, h home.Home, id string, bend func(*eval.ForgeRun)) string {
	t.Helper()
	r, err := eval.LoadForgeRun(h, id)
	if err != nil {
		t.Fatal(err)
	}
	bend(&r)
	n, err := eval.NewForgeRun(r.Target, r.Run, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := eval.SaveForgeRun(h, n); err != nil {
		t.Fatal(err)
	}
	return n.ID
}

var fullPhases = []string{"resolving", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}

// 1, 7, 8, 13: both runs are produced internally, the certification is
// accepted, the producer evidence and provenance are bound, and nothing is
// activated.
func TestForgeCertificationAcceptsFromInternallyProducedRuns(t *testing.T) {
	e := newCertEnv(t)
	res, err := e.run()
	if err != nil {
		t.Fatal(err)
	}
	if res.Certification.Verdict.Status != eval.VerdictAccepted || res.Record.Verdict != eval.VerdictAccepted {
		t.Fatalf("verdict %+v", res.Certification.Verdict)
	}
	if !reflect.DeepEqual(e.phases, fullPhases) {
		t.Fatalf("phases %v, want %v (the probe and runs are not re-reported as the sub-operations' own phases)", e.phases, fullPhases)
	}
	if e.probes != 1 || !reflect.DeepEqual(e.executed, []string{eval.ForgeTargetSource, eval.ForgeTargetVariant}) {
		t.Fatalf("probes %d executions %v: the probe, then the reference, then the candidate", e.probes, e.executed)
	}
	if st := eval.ResolveCertification(e.h, e.v); st.State != eval.StateAccepted {
		t.Fatalf("resolved %s %v", st.State, st.Problems)
	}
	if got := e.recordedRuns(); len(got) != 2 {
		t.Fatalf("recorded runs %v", got)
	}

	// The record and the report name the exact producer runs and digests.
	ref, err := eval.LoadForgeRun(e.h, res.ReferenceEvidence)
	if err != nil {
		t.Fatal(err)
	}
	cand, err := eval.LoadForgeRun(e.h, res.CandidateEvidence)
	if err != nil {
		t.Fatal(err)
	}
	rp := res.Record.Producer
	if rp == nil || rp.ReferenceEvidenceID != ref.ID || rp.ReferenceRunSHA256 != ref.RunSHA256 || rp.CandidateEvidenceID != cand.ID || rp.CandidateRunSHA256 != cand.RunSHA256 {
		t.Fatalf("record producer %+v", rp)
	}
	rep, err := eval.LoadCertification(e.h, res.Record)
	if err != nil {
		t.Fatal(err)
	}
	pr := rep.Producer
	if pr == nil || pr.Schema != eval.CertificationProducerSchema || pr.Reference.EvidenceID != ref.ID || pr.Candidate.EvidenceID != cand.ID {
		t.Fatalf("report producer %+v", pr)
	}
	if pr.Probe == nil || pr.Probe.Result != ProbePassed || pr.Probe.Device != "cuda" || pr.Preflight == nil {
		t.Fatalf("gates %+v %+v", pr.Probe, pr.Preflight)
	}

	// 7: the reference is the exact pinned source, at the requested device and dtype.
	rt := pr.Reference.Target
	if rt.Kind != eval.ForgeTargetSource || rt.Model != setup.ClefFlash || rt.Revision != strings.Repeat("ef", 20) || rt.SourceFilesSHA256 == "" ||
		rt.RequestedDevice != "cpu" || rt.Device != "cpu" || rt.RequestedDType != DefaultReferenceDType || rt.DType != DefaultReferenceDType || rt.Variant != "" {
		t.Fatalf("reference target %+v", rt)
	}
	// 8: the candidate is the exact variant, with its quantized execution.
	ct := pr.Candidate.Target
	if ct.Kind != eval.ForgeTargetVariant || ct.Variant != e.v.ID || ct.VariantManifestSHA256 != e.v.ManifestSHA256() || ct.Recipe != e.v.Recipe.Name ||
		ct.Scheme != "W4A16" || ct.QuantizedModules == 0 || ct.Quantization == "" || ct.RequestedDevice != "cuda" || ct.Device != "cuda" || ct.DType != e.v.Weights.DType {
		t.Fatalf("candidate target %+v", ct)
	}
	if rt.Runtime == ct.Runtime || rt.Runtime == "" || ct.Runtime == "" {
		t.Fatalf("runtimes %q %q", rt.Runtime, ct.Runtime)
	}
	if rep.Dataset.SHA256 != ref.DatasetSHA256 || rep.Dataset.SHA256 != cand.DatasetSHA256 || len(rep.Dataset.Definitions) != 2 {
		t.Fatalf("dataset %+v", rep.Dataset)
	}
	if rep.Policy.ID != "fixture-fidelity/1" || rep.Reference.IdentitySHA256 == "" || rep.Candidate.VariantID != e.v.ID {
		t.Fatalf("report %+v / %+v", rep.Policy, rep.Candidate)
	}
	e.untouched()
}

// 2, 14: a rejection is a completed, recorded certification; nothing is activated.
func TestForgeCertificationRejectionIsRecordedAndActivatesNothing(t *testing.T) {
	e := newCertEnv(t)
	e.params.Policy = "" // the built-in profile asks for 100 observations; the fixture has 6
	res, err := e.run()
	if err != nil {
		t.Fatalf("a rejected verdict is not an error: %v", err)
	}
	if res.Certification.Verdict.Status != eval.VerdictRejected || res.Record.Verdict != eval.VerdictRejected || res.Record.Producer == nil {
		t.Fatalf("verdict %+v record %+v", res.Certification.Verdict, res.Record)
	}
	if st := eval.ResolveCertification(e.h, e.v); st.State != eval.StateRejected {
		t.Fatalf("resolved %s", st.State)
	}
	if got := e.recordedRuns(); len(got) != 2 {
		t.Fatalf("the evidence was not kept: %v", got)
	}
	e.untouched()
}

// 3: the candidate must be exactly the variant, by identity and manifest.
func TestForgeCertificationRefusesACandidateThatIsNotTheExactVariant(t *testing.T) {
	forgeSource(t) // Install the shared read-only catalog before parallel cases start.
	for name, bend := range map[string]func(*eval.ForgeRun){
		"another manifest": func(r *eval.ForgeRun) { r.Target.VariantManifestSHA256 = strings.Repeat("0", 64) },
		"another variant":  func(r *eval.ForgeRun) { r.Target.Variant = "clef-flash--w4a16--000000000000" },
		"no quantization":  func(r *eval.ForgeRun) { r.Target.QuantizedModules, r.Target.Quantization = 0, "" },
		"another runtime":  func(r *eval.ForgeRun) { r.Target.Runtime = "rt-other" },
		"another device":   func(r *eval.ForgeRun) { r.Target.Device = "cpu" },
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			e := newCertEnv(t)
			e.deps.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
				res, err := e.execute(ctx, root, p, log)
				if err == nil && p.Target.Kind == eval.ForgeTargetVariant {
					res.EvidenceID = reseal(t, e.h, res.EvidenceID, bend)
				}
				return res, err
			}
			_, err := e.run()
			fe := asCertError(t, err)
			if fe.Phase != string(CertPhaseCandidate) || !strings.Contains(err.Error(), "is not the requested execution") {
				t.Fatalf("err = %v", err)
			}
			if fe.ReferenceEvidence == "" || fe.CandidateEvidence == "" {
				t.Fatalf("the evidence of the completed runs is not named: %+v", fe)
			}
			e.noCertification()
			e.untouched()
		})
	}
}

// 4: a source run is never accepted in the candidate's place.
func TestForgeCertificationRefusesSourceSubstitutionForTheCandidate(t *testing.T) {
	e := newCertEnv(t)
	e.deps.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
		if p.Target.Kind == eval.ForgeTargetVariant {
			p.Target = ExecutionTarget{Kind: eval.ForgeTargetSource, Model: setup.ClefFlash, Device: p.Target.Device, DType: "bfloat16"}
		}
		return e.execute(ctx, root, p, log)
	}
	_, err := e.run()
	if fe := asCertError(t, err); fe.Phase != string(CertPhaseCandidate) || !strings.Contains(err.Error(), "it is a source run, not a variant run") {
		t.Fatalf("err = %v", err)
	}
	e.noCertification()
	e.untouched()
}

// The worker itself answering as the source is refused at the READY gate of #158,
// as a failed candidate execution: no source fallback.
func TestForgeCertificationRefusesAWorkerThatLoadedTheSource(t *testing.T) {
	e := newCertEnv(t)
	e.mode[eval.ForgeTargetVariant] = "source"
	_, err := e.run()
	if fe := asCertError(t, err); fe.Phase != string(CertPhaseCandidate) || !strings.Contains(err.Error(), "did not load the persisted variant") {
		t.Fatalf("err = %v", err)
	}
	e.noCertification()
}

// 5, 6: the two runs must be the same semantic evaluation input, by digest.
func TestForgeCertificationRefusesRunsOfDifferentEvaluationInputs(t *testing.T) {
	forgeSource(t) // Keep the catalog immutable while independent cases run.
	for name, tc := range map[string]struct {
		bend func(*ExecuteParams)
		want string
	}{
		"dataset digest": {func(p *ExecuteParams) { p.Input.DatasetSHA256 = strings.Repeat("9", 64) }, "normalized dataset digest differs"},
		"question definitions": {func(p *ExecuteParams) {
			cases := slices.Clone(p.Input.Cases)
			defs := slices.Clone(cases[0].Definitions)
			defs[0].Digest = strings.Repeat("7", 64)
			cases[0].Definitions = defs
			p.Input.Cases = cases
		}, "Question Definition digest differs"},
		"case set": {func(p *ExecuteParams) { p.Input.Cases = p.Input.Cases[:2] }, "case count differs"},
		"case order": {func(p *ExecuteParams) {
			cases := slices.Clone(p.Input.Cases)
			cases[0], cases[1] = cases[1], cases[0]
			p.Input.Cases = cases
		}, "observation order differs"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			e := newCertEnv(t)
			e.deps.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
				if p.Target.Kind == eval.ForgeTargetVariant {
					tc.bend(&p)
				}
				return e.execute(ctx, root, p, log)
			}
			res, err := e.run()
			fe := asCertError(t, err)
			var ae *eval.AlignmentError
			if fe.Phase != string(CertPhaseAligning) || !errors.As(err, &ae) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v", err)
			}
			if res.ReferenceEvidence == "" || res.CandidateEvidence == "" {
				t.Fatalf("both runs completed and are linked to the failure: %+v", res)
			}
			e.noCertification()
			e.untouched()
		})
	}
}

// 9: a failed probe stops before any full run, and is reported as the probe's.
func TestForgeCertificationStopsAtAFailedProbe(t *testing.T) {
	e := newCertEnv(t)
	e.deps.Probe = func(ctx context.Context, _ string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
		e.probes++
		return Probe(ctx, e.h, p, probeDeps(t, "load_fails", e.marker, "uncertified"), log, obs)
	}
	res, err := e.run()
	fe := asCertError(t, err)
	var pe *ProbeError
	if fe.Phase != string(CertPhaseProbe) || !errors.As(err, &pe) {
		t.Fatalf("err = %v", err)
	}
	if res.Probe.Result != ProbeFailed {
		t.Fatalf("the failed probe is not returned: %+v", res.Probe)
	}
	if len(e.executed) != 0 || len(e.recordedRuns()) != 0 {
		t.Fatalf("a full run started after a failed probe: %v %v", e.executed, e.recordedRuns())
	}
	if want := fullPhases[:3]; !reflect.DeepEqual(e.phases, want) {
		t.Fatalf("phases %v, want %v", e.phases, want)
	}
	e.noCertification()
	e.untouched()
}

// 10: a failed reference stops before the candidate and the certification.
func TestForgeCertificationStopsAtAFailedReference(t *testing.T) {
	e := newCertEnv(t)
	e.mode[eval.ForgeTargetSource] = "wrong_dtype"
	_, err := e.run()
	if fe := asCertError(t, err); fe.Phase != string(CertPhaseReference) || !strings.Contains(err.Error(), "reference execution failed") {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(e.executed, []string{eval.ForgeTargetSource}) || len(e.recordedRuns()) != 0 {
		t.Fatalf("executions %v runs %v", e.executed, e.recordedRuns())
	}
	e.noCertification()
	e.untouched()
}

// 11: a failed candidate leaves the reference evidence and no certification.
func TestForgeCertificationFailedCandidateLeavesSourceEvidenceOnly(t *testing.T) {
	e := newCertEnv(t)
	e.mode[eval.ForgeTargetVariant] = "unquantized"
	res, err := e.run()
	fe := asCertError(t, err)
	if fe.Phase != string(CertPhaseCandidate) || fe.ReferenceEvidence == "" || fe.CandidateEvidence != "" || fe.ReferenceEvidence != res.ReferenceEvidence {
		t.Fatalf("err = %+v", fe)
	}
	if !strings.Contains(err.Error(), fe.ReferenceEvidence) {
		t.Fatalf("the message does not link the reference evidence: %v", err)
	}
	if got := e.recordedRuns(); len(got) != 1 || got[0] != fe.ReferenceEvidence+".json" {
		t.Fatalf("runs on disk %v", got)
	}
	e.noCertification()
	e.untouched()
}

// 12: a refusal by eval.Certify records no certification at all.
func TestForgeCertificationRefusalByTheEvaluatorRecordsNothing(t *testing.T) {
	e := newCertEnv(t)
	e.deps.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
		res, err := e.execute(ctx, root, p, log)
		if err == nil && p.Target.Kind == eval.ForgeTargetVariant {
			// A self-consistent run whose resident reports another source model.
			res.EvidenceID = reseal(t, e.h, res.EvidenceID, func(r *eval.ForgeRun) {
				rt := map[string]any{}
				for k, v := range r.Run.Run.Identity.Runtime {
					rt[k] = v
				}
				rt["model_id"] = "someone-else"
				r.Run.Run.Identity.Runtime = rt
			})
		}
		return res, err
	}
	_, err := e.run()
	if fe := asCertError(t, err); fe.Phase != string(CertPhaseCertifying) || !strings.Contains(err.Error(), "runs cannot be certified") {
		t.Fatalf("err = %v", err)
	}
	e.noCertification()
	e.untouched()
}

// A record that cannot be written is a persistence failure and no certification.
func TestForgeCertificationPersistenceFailureRecordsNoAcceptance(t *testing.T) {
	e := newCertEnv(t)
	dir := h2dir(e.h, e.v)
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := e.run()
	if fe := asCertError(t, err); fe.Phase != string(CertPhasePersisting) || !strings.Contains(err.Error(), "could not be recorded") {
		t.Fatalf("err = %v", err)
	}
	if st := eval.ResolveCertification(e.h, e.v); st.State != eval.StateUncertified {
		t.Fatalf("state %s", st.State)
	}
	e.untouched()
}

// 15: cancellation fails the phase it interrupted and leaves nothing behind.
func TestForgeCertificationCancellationRecordsNothingAndActivatesNothing(t *testing.T) {
	e := newCertEnv(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e.deps.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
		cancel() // the operator cancels as the reference run starts
		return e.execute(ctx, root, p, log)
	}
	_, err := RunForgeCertification(ctx, e.h, e.params, e.deps, io.Discard, e.obs)
	if fe := asCertError(t, err); fe.Phase != string(CertPhaseReference) || !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if len(e.executed) != 1 || len(e.recordedRuns()) != 0 {
		t.Fatalf("executions %v runs %v", e.executed, e.recordedRuns())
	}
	e.noCertification()
	e.untouched()
}

// Nothing is run for an input that cannot be resolved exactly.
func TestForgeCertificationResolvesEverythingBeforeExpensiveWork(t *testing.T) {
	for name, tc := range map[string]struct {
		bend func(*ForgeCertifyParams)
		want string
	}{
		"no candidate device":     {func(p *ForgeCertifyParams) { p.Device = "" }, "no default and no fallback"},
		"unknown variant":         {func(p *ForgeCertifyParams) { p.Variant = "clef-flash--w4a16--ffffffffffff" }, "not in this home"},
		"no variant":              {func(p *ForgeCertifyParams) { p.Variant = "" }, "names the variant"},
		"no corpus":               {func(p *ForgeCertifyParams) { p.Dataset = "" }, "needs an evaluation corpus"},
		"missing corpus file":     {func(p *ForgeCertifyParams) { p.Dataset = "/nonexistent/corpus.jsonl" }, "corpus.jsonl"},
		"unresolved questions":    {func(p *ForgeCertifyParams) { p.Questions = nil }, "question"},
		"missing policy":          {func(p *ForgeCertifyParams) { p.Policy = "/nonexistent/policy.json" }, "policy.json"},
		"a low-precision dtype":   {func(p *ForgeCertifyParams) { p.ReferenceDType = "float16" }, "high precision"},
		"a reference device typo": {func(p *ForgeCertifyParams) { p.ReferenceDevice = "gpu" }, "cpu or cuda"},
	} {
		t.Run(name, func(t *testing.T) {
			e := newCertEnv(t)
			tc.bend(&e.params)
			_, err := e.run()
			if fe := asCertError(t, err); fe.Phase != string(CertPhaseResolving) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v", err)
			}
			if e.probes != 0 || len(e.executed) != 0 || !reflect.DeepEqual(e.phases, []string{"resolving"}) {
				t.Fatalf("work started: probes %d executions %v phases %v", e.probes, e.executed, e.phases)
			}
		})
	}
}

// A blocker that is not a missing runtime refuses the operation before the probe.
func TestForgeCertificationPreflightBlockerStopsBeforeTheProbe(t *testing.T) {
	e := newCertEnv(t)
	// The variant's weights no longer match the manifest.
	vdir := e.h.VariantDir(setup.ClefFlash, e.v.ID)
	files := e.v.VariantFileNames()
	if err := os.WriteFile(filepath.Join(vdir, files[0]), []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	e.params.Materialize = true // materialization cannot fix this, and must not be tried
	_, err := e.run()
	fe := asCertError(t, err)
	if _, ok := setup.AsPreflightError(err); !ok || fe.Phase != string(CertPhasePreflight) {
		t.Fatalf("err = %v", err)
	}
	if e.probes != 0 || len(e.executed) != 0 {
		t.Fatalf("work started after a blocked preflight")
	}
	e.noCertification()
}

// A missing runtime is materialized through the setup authority only when asked,
// then the preflight is repeated; nothing is activated.
func TestForgeCertificationMaterializesAMissingRuntimeOnlyWhenAsked(t *testing.T) {
	spec, err := setup.Desired("cpu")
	if err != nil {
		t.Fatal(err)
	}
	t.Run("not asked", func(t *testing.T) {
		e := newCertEnv(t)
		if err := os.RemoveAll(e.h.Path("runtime", spec.ID())); err != nil {
			t.Fatal(err)
		}
		_, err := e.run()
		if _, ok := setup.AsPreflightError(err); !ok || asCertError(t, err).Phase != string(CertPhasePreflight) {
			t.Fatalf("err = %v", err)
		}
		if e.probes != 0 {
			t.Fatal("the probe ran")
		}
		e.untouched()
	})
	t.Run("asked", func(t *testing.T) {
		e := newCertEnv(t)
		if err := os.RemoveAll(e.h.Path("runtime", spec.ID())); err != nil {
			t.Fatal(err)
		}
		e.params.Materialize = true
		var got [][2]string
		e.deps.Materialize = func(_ context.Context, root, device, model string, _ io.Writer, _ *setup.Observer) error {
			got = append(got, [2]string{device, model})
			rdir := e.h.Path("runtime", spec.ID())
			if err := os.MkdirAll(rdir, 0o755); err != nil {
				return err
			}
			return home.WriteJSON(filepath.Join(rdir, "manifest.json"), home.RuntimeManifest{Identity: spec.ID(), Spec: spec, PythonVersion: spec.Python, PythonRelPath: "env/bin/python",
				Installed: []string{"torch==" + spec.Torch, "transformers==5.17.0", "compressed-tensors==0.19.0"}})
		}
		res, err := e.run()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, [][2]string{{"cpu", setup.ClefFlash}}) || !reflect.DeepEqual(res.Materialized, []string{"cpu"}) {
			t.Fatalf("materialized %v / %v", got, res.Materialized)
		}
		want := []string{"resolving", "preflight", "materializing", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
		if !reflect.DeepEqual(e.phases, want) {
			t.Fatalf("phases %v, want %v", e.phases, want)
		}
		e.untouched() // materialization never activates
	})
	t.Run("fails", func(t *testing.T) {
		e := newCertEnv(t)
		os.RemoveAll(e.h.Path("runtime", spec.ID()))
		e.params.Materialize = true
		e.deps.Materialize = func(context.Context, string, string, string, io.Writer, *setup.Observer) error {
			return errors.New("no network")
		}
		_, err := e.run()
		if fe := asCertError(t, err); fe.Phase != string(CertPhaseMaterializing) || !strings.Contains(err.Error(), "no network") {
			t.Fatalf("err = %v", err)
		}
		if e.probes != 0 {
			t.Fatal("the probe ran")
		}
	})
}

// A materialization that does not make the preflight pass is not repeated forever.
func TestForgeCertificationMaterializesAtMostOnce(t *testing.T) {
	spec, _ := setup.Desired("cpu")
	e := newCertEnv(t)
	os.RemoveAll(e.h.Path("runtime", spec.ID()))
	e.params.Materialize = true
	n := 0
	e.deps.Materialize = func(context.Context, string, string, string, io.Writer, *setup.Observer) error { n++; return nil }
	_, err := e.run()
	if _, ok := setup.AsPreflightError(err); !ok || n != 1 {
		t.Fatalf("err = %v after %d materializations", err, n)
	}
}

// The defaults of an unspecified reference are the canonical ones, and are what is recorded.
func TestForgeCertificationExplicitReferenceSettingsAreRecordedAsRequested(t *testing.T) {
	e := newCertEnv(t)
	e.params.ReferenceDType = "float32"
	res, err := e.run()
	if err != nil {
		t.Fatal(err)
	}
	ref, err := eval.LoadForgeRun(e.h, res.ReferenceEvidence)
	if err != nil {
		t.Fatal(err)
	}
	if ref.Target.RequestedDType != "float32" || ref.Target.DType != "float32" || ref.Target.RequestedDevice != "cpu" {
		t.Fatalf("reference target %+v", ref.Target)
	}
	if res.Certification.Reference.DType != "torch.float32" {
		t.Fatalf("execution %+v", res.Certification.Reference)
	}
}

// ---- controller ----

// certController is a controller bound to a running two-resident set whose home
// holds the fixture variant, with real Probe and RunExecution over the fake
// worker, so the maintenance lease and the whole workflow run together.
type certController struct {
	*leaseEnv
	e *certEnv
	// during are the residents as one coherent reading at the probe, the
	// reference run and the candidate run (and the build, in a composed
	// Forge operation).
	during map[string]map[string]ResidentStatus
	// snaps are the controller's own projection at those same points.
	snaps map[string]Snapshot
	// at, when set, runs at a named point (probe, build, source, variant)
	// with the controller, inside the transaction.
	at map[string]func()
}

func newCertController(t *testing.T, mutate func(*certEnv)) *certController {
	t.Helper()
	e := newCertEnv(t)
	if mutate != nil {
		mutate(e)
	}
	cc := &certController{e: e, during: map[string]map[string]ResidentStatus{}, snaps: map[string]Snapshot{}, at: map[string]func(){}}
	var c *Controller
	observe := func(point string) {
		cc.during[point] = residentsNow(cc.set)
		cc.snaps[point] = c.Snapshot()
		if f := cc.at[point]; f != nil {
			f()
		}
	}
	a, b := residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok")
	set := newResidentSet(t, noRestart, a, b)
	if _, err := set.SetRouting(keepPolicy()); err != nil {
		t.Fatal(err)
	}
	desired := []string{modelB}
	c = New(Config{Home: e.h.Root, Installed: func(string) bool { return true }, Open: func(string) (Runtime, error) { return set, nil },
		Residents: func() []string { return desired }, RestoreTimeout: 10 * time.Second,
		Maintenance: Maintenance{
			Preflight: buildEvaluateDeps(t, e, nil).Certify.Preflight,
			Build: func(context.Context, home.Home, optimize.Request, io.Writer, *setup.Observer) (optimize.Result, error) {
				observe("build")
				return optimize.Result{Variant: e.v}, nil
			},
			Materialize: func(root, device, model string, log io.Writer, obs *setup.Observer) error {
				return e.deps.Materialize(context.Background(), root, device, model, log, obs)
			},
			Probe: func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (ProbeRecord, error) {
				observe("probe")
				return e.deps.Probe(ctx, root, p, log, obs)
			},
			Execute: func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
				observe(p.Target.Kind)
				return e.deps.Execute(ctx, root, p, log)
			},
		}})
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = c.Close(ctx)
	})
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	bothReady(t, set)
	cc.leaseEnv = &leaseEnv{c: c, set: set, h: e.h, activate: activationBytes, desired: desired}
	return cc
}

// 16: through the controller, the probe, the reference and the candidate run
// with the serving residents quiesced for the whole transaction, every resident
// comes back, and the activation record, the desired residents, the default and
// the routing policy are unchanged. The
// operation reports each real phase and completes.
func TestCertifyVariantThroughTheControllerLeavesServingStateUntouched(t *testing.T) {
	cc := newCertController(t, nil)
	before := residentsNow(cc.set)
	if err := cc.c.CertifyVariant(cc.e.params); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, cc.c)
	m := s.Maintenance
	if m == nil || m.Kind != OpForgeCertify || m.Failure != nil {
		t.Fatalf("maintenance %+v", m)
	}
	if !reflect.DeepEqual(m.Phases, fullPhases) || !reflect.DeepEqual(m.Plan, fullPhases) || m.Phase != "persisting" {
		t.Fatalf("phases %v plan %v phase %q", m.Phases, m.Plan, m.Phase)
	}
	// The certification is one model-engineering transaction: serving is down
	// at the probe, the cpu reference and the candidate alike (the reference
	// does not bring it back in the middle), and comes back once at the end.
	for _, step := range []string{"probe", eval.ForgeTargetSource, eval.ForgeTargetVariant} {
		if len(cc.during[step]) == 0 {
			t.Fatalf("%s was never observed", step)
		}
		for model, r := range cc.during[step] {
			if r.Running {
				t.Fatalf("%s: resident %s was running during the transaction", step, model)
			}
		}
	}
	bothReady(t, cc.set)
	after := residentsNow(cc.set)
	for model, r := range before {
		if !after[model].Running || after[model].Status.Worker.State != r.Status.Worker.State {
			t.Fatalf("resident %s: %+v then %+v", model, r.Status.Worker, after[model].Status.Worker)
		}
	}
	cc.assertUntouched(t)
	if st := eval.ResolveCertification(cc.h, cc.e.v); st.State != eval.StateAccepted {
		t.Fatalf("certification %s %v", st.State, st.Problems)
	}
	if s.RestartRequired || s.ResidencyChanged {
		t.Fatalf("the certification asked for a restart: %+v", s)
	}
}

// A failure of the operation names its phase in the operation and its diagnostic.
func TestCertifyVariantFailureNamesItsPhaseAndKeepsADiagnostic(t *testing.T) {
	cc := newCertController(t, func(e *certEnv) { e.mode[eval.ForgeTargetVariant] = "unquantized" })
	if err := cc.c.CertifyVariant(cc.e.params); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, cc.c)
	f := s.Maintenance.Failure
	if f == nil || f.Phase != string(CertPhaseCandidate) || f.Diagnostic == "" || !strings.Contains(f.Message, "candidate execution failed") {
		t.Fatalf("failure %+v", f)
	}
	if !reflect.DeepEqual(s.Maintenance.Phases, fullPhases[:5]) {
		t.Fatalf("phases %v", s.Maintenance.Phases)
	}
	bothReady(t, cc.set)
	cc.assertUntouched(t)
	cc.e.noCertification()
}

// A candidate device is required, as for every device-bound controller action.
func TestCertifyVariantRequiresADevice(t *testing.T) {
	c := New(Config{Home: t.TempDir()})
	if err := c.CertifyVariant(ForgeCertifyParams{Variant: "v"}); err == nil || !strings.Contains(err.Error(), "device is required") {
		t.Fatalf("err = %v", err)
	}
}

// The producer linkage of a record is verified against its report: a record
// that names other runs, or drops the linkage its report carries, is not trusted.
func TestForgeCertificationProducerBindingIsVerified(t *testing.T) {
	e := newCertEnv(t)
	res, err := e.run()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(h2dir(e.h, e.v), strings.TrimSuffix(res.Record.Report, ".report.json")+".record.json")
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for name, edit := range map[string]func(map[string]any){
		"another reference run": func(r map[string]any) {
			r["producer"].(map[string]any)["reference_run_sha256"] = strings.Repeat("0", 64)
		},
		"another candidate run": func(r map[string]any) {
			r["producer"].(map[string]any)["candidate_evidence_id"] = "fr-" + strings.Repeat("0", 32)
		},
		"no linkage": func(r map[string]any) { delete(r, "producer") },
	} {
		var rec map[string]any
		if err := json.Unmarshal(good, &rec); err != nil {
			t.Fatal(err)
		}
		edit(rec)
		b, _ := json.Marshal(rec)
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		if st := eval.ResolveCertification(e.h, e.v); st.State != eval.StateUncertified || len(st.Problems) != 1 || !strings.Contains(st.Problems[0], "producer") {
			t.Fatalf("%s: %s %v", name, st.State, st.Problems)
		}
	}
	if err := os.WriteFile(path, good, 0o644); err != nil {
		t.Fatal(err)
	}
	if st := eval.ResolveCertification(e.h, e.v); st.State != eval.StateAccepted {
		t.Fatalf("restored: %s %v", st.State, st.Problems)
	}
}

func TestCertifyPlanListsMaterializingOnlyWhenAllowed(t *testing.T) {
	if got := certifyPlan(false); !reflect.DeepEqual(got, fullPhases) {
		t.Fatalf("plan %v", got)
	}
	want := []string{"resolving", "preflight", "materializing", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
	if got := certifyPlan(true); !reflect.DeepEqual(got, want) {
		t.Fatalf("plan %v, want %v", got, want)
	}
}

// A failure to restore the serving residents after a run is secondary evidence
// kept in the chain; a run that did complete stays linked, and the session is
// never a success.
func TestForgeCertificationPreservesRestorationDiagnostics(t *testing.T) {
	e := newCertEnv(t)
	e.deps.Execute = func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (ExecutionResult, error) {
		res, err := e.execute(ctx, root, p, log)
		if err == nil && p.Target.Kind == eval.ForgeTargetSource {
			err = &ExecutionError{Restore: errors.New("resident clef-flash did not come back")}
		}
		return res, err
	}
	res, err := e.run()
	fe := asCertError(t, err)
	var xe *ExecutionError
	if fe.Phase != string(CertPhaseReference) || !errors.As(err, &xe) || xe.Restore == nil || xe.Primary != nil {
		t.Fatalf("err = %v", err)
	}
	if res.ReferenceEvidence == "" || fe.ReferenceEvidence != res.ReferenceEvidence || len(e.executed) != 1 {
		t.Fatalf("evidence %q executions %v", res.ReferenceEvidence, e.executed)
	}
	e.noCertification()
	e.untouched()
}
