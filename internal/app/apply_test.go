package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// certifyFixture records a real eval.Certification of v with the given verdict.
func certifyFixture(t *testing.T, h home.Home, m home.ModelManifest, v home.VariantManifest, accepted bool) eval.CertificationRecord {
	t.Helper()
	pol := eval.DefaultPolicy()
	c := eval.Certification{
		Schema: eval.CertificationSchema, CreatedAt: time.Now().UTC().Format(time.RFC3339), Source: v.Source,
		Variant: eval.CertifiedVariant{ID: v.ID, ManifestSHA256: v.ManifestSHA256(), BuildID: v.BuildID, Recipe: v.Recipe.Name,
			RecipeSHA256: v.RecipeSHA256, Scheme: v.Weights.Scheme, Engine: v.Optimizer.Engine, EngineVersion: v.Optimizer.Version},
		Reference: eval.Execution{Role: "reference", ModelID: m.ID, Provider: m.Provider, Revision: m.Revision, Device: "cpu", DType: "torch.bfloat16",
			IdentitySHA256: "ref-identity", ResidentStable: true},
		Candidate: eval.Execution{Role: "candidate", ModelID: m.ID, Provider: m.Provider, Revision: m.Revision, VariantID: v.ID, Device: "cuda",
			DType: "torch.bfloat16", IdentitySHA256: "cand-identity", ResidentStable: true},
		Dataset:  eval.CertifiedDataset{SHA256: "dataset", Cases: 100, Observations: 100, QuestionsSHA256: "questions", InputSHA256: "inputs", SentSHA256: "sent"},
		Fidelity: eval.Fidelity{Paired: 100}, Policy: pol, PolicySHA256: pol.SHA256(),
	}
	if !accepted {
		c.Fidelity.ChoiceFlips, c.Fidelity.FlipRate = 50, 0.5
	}
	c.Verdict = pol.Evaluate(c)
	if got := c.Verdict.Status == eval.VerdictAccepted; got != accepted {
		t.Fatalf("fixture verdict %s, wanted accepted=%v", c.Verdict.Status, accepted)
	}
	rec, err := eval.SaveCertification(h, c)
	if err != nil {
		t.Fatal(err)
	}
	return rec
}

// applyEnv is a Controller over a real fixture home whose serving runtimes are
// real resident sets (or worker bindings) of fake worker processes that report
// the provenance of the activation record they were opened from.
type applyEnv struct {
	t      *testing.T
	h      home.Home
	model  home.ModelManifest
	v      home.VariantManifest // the variant under test
	c      *Controller
	marker string
	ctx    context.Context

	mu          sync.Mutex
	modes       map[string]string // "source" or a variant ID -> fake worker mode
	opens       int
	failOpenAt  int // fail this Open (1-based)
	activates   []string
	activateErr error
	materials   int
	extra       bool // a second resident beside the default
	routing     bool
	noRouting   map[int]bool // Opens that bind no routing policy
	single      bool         // bind a WorkerBinding instead of a one-member set
	statusLie   func(*server.Runtime)
	sets        []*ResidentSet
	bindings    []*WorkerBinding
}

type applyOpt func(*applyEnv)

var applyPolicy = route.Policy{Schema: route.PolicySchema, ID: "apply-keep", Default: route.Rule{First: setup.ClefFlash}}

func newApplyEnv(t *testing.T, certify string, opts ...applyOpt) *applyEnv {
	t.Helper()
	h, v := forgeHome(t)
	m, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	switch certify {
	case "accepted":
		certifyFixture(t, h, m, v, true)
	case "rejected":
		certifyFixture(t, h, m, v, false)
	}
	ctx, cancel := context.WithCancel(context.Background())
	e := &applyEnv{t: t, h: h, model: m, v: v, marker: t.TempDir(), ctx: ctx, modes: map[string]string{}, noRouting: map[int]bool{}}
	for _, o := range opts {
		o(e)
	}
	e.writeActive(home.Active{})
	e.c = New(Config{Home: h.Root, Installed: func(string) bool { return true }, Open: e.open,
		Residents: func() []string {
			if e.extra {
				return []string{modelB}
			}
			return nil
		},
		RestoreTimeout: 20 * time.Second, ReadyTimeout: 20 * time.Second, SmokeTimeout: 10 * time.Second,
		Maintenance: Maintenance{ActivateVariant: e.activateVariant,
			CalibrateCapacity: func(context.Context, string, string, string, io.Writer) error { return nil },
			Materialize: func(root, device, model string, log io.Writer, obs *setup.Observer) error {
				e.mu.Lock()
				e.materials++
				e.mu.Unlock()
				spec, _ := setup.Desired(device)
				return os.MkdirAll(h.Path("runtime", spec.ID()), 0o755)
			}}})
	t.Cleanup(func() {
		sctx, scancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer scancel()
		_ = e.c.Close(sctx)
		cancel()
		for _, s := range e.sets {
			s.Stop()
		}
	})
	return e
}

// writeActive writes the activation record of the source (zero a) or a variant,
// exactly as the activation authority would.
func (e *applyEnv) writeActive(a home.Active) home.Active {
	spec, _ := setup.Desired("cuda")
	if a.Runtime == "" {
		a.Runtime, a.ModelID, a.Model, a.Device = spec.ID(), e.model.ID, setup.ModelDirName(e.model), "cuda"
	}
	if err := home.WriteJSON(e.h.Path("state", "active-runtime.json"), a); err != nil {
		e.t.Fatal(err)
	}
	return a
}

func (e *applyEnv) activateVariant(root, device, model, variant string, experimental bool, log io.Writer, obs *setup.Observer) (bool, error) {
	e.mu.Lock()
	e.activates = append(e.activates, device+" "+model+" "+variant)
	err := e.activateErr
	e.mu.Unlock()
	for _, p := range []setup.Phase{setup.PhaseRuntime, setup.PhaseModel, setup.PhaseVariant} {
		obs.OnPhase(p)
	}
	if err != nil {
		return false, err
	}
	obs.OnPhase(setup.PhaseActivation)
	spec, _ := setup.Desired(device)
	m, _ := setup.LookupModel(model)
	next := home.Active{Runtime: spec.ID(), ModelID: m.ID, Model: setup.ModelDirName(m), Device: device, Variant: variant, Experimental: experimental}
	var prev home.Active
	changed := home.ReadJSON(e.h.Path("state", "active-runtime.json"), &prev) != nil || prev != next
	return changed, home.WriteJSON(e.h.Path("state", "active-runtime.json"), next)
}

func (e *applyEnv) mode(key string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if m := e.modes[key]; m != "" {
		return m
	}
	return "ok"
}

func (e *applyEnv) setMode(key, mode string) {
	e.mu.Lock()
	e.modes[key] = mode
	e.mu.Unlock()
}

// open binds the runtime the activation record names, as the production Open does.
func (e *applyEnv) open(root string) (Runtime, error) {
	e.mu.Lock()
	e.opens++
	n := e.opens
	fail := e.failOpenAt == n
	e.mu.Unlock()
	if fail {
		return nil, errors.New("the runtime cannot be bound")
	}
	var a home.Active
	if err := home.ReadJSON(e.h.Path("state", "active-runtime.json"), &a); err != nil {
		return nil, err
	}
	model, err := setup.ActiveModel(a)
	if err != nil {
		return nil, err
	}
	exe, _ := os.Executable()
	key, variant, scheme, vdtype := "source", "", "", ""
	info := server.Runtime{Home: root, Runtime: a.Runtime, ModelID: model.ID, Model: a.Model, Device: a.Device}
	if v, ok, err := e.h.LoadVariant(a); err != nil {
		return nil, err
	} else if ok {
		key, variant, scheme, vdtype = v.ID, v.ID, v.Weights.Scheme, v.Weights.DType
		info.Variant = &server.Variant{ID: v.ID, Recipe: v.Recipe.Name, Scheme: v.Weights.Scheme, Bits: v.Weights.Bits, DType: v.Weights.DType,
			Format: v.Weights.Format, ManifestSHA256: v.ManifestSHA256(), Certification: eval.StateAccepted,
			Source: server.VariantSource{ID: v.Source.ID, Provider: v.Source.Provider, Repo: v.Source.Repo, Revision: v.Source.Revision}}
	}
	e.mu.Lock()
	lie := e.statusLie
	e.mu.Unlock()
	if lie != nil && variant == e.v.ID {
		lie(&info)
	}
	spec := strings.Join([]string{e.mode(key), model.ID, model.Revision, variant, scheme, vdtype, a.Device, "torch.bfloat16", e.marker}, "|")
	cfg := worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeExecEnv + "=" + spec}, StartTimeout: 30 * time.Second, RequestTimeout: 10 * time.Second}
	if e.single && !e.extra {
		sup := worker.NewSupervisor(cfg, noRestart)
		b := &WorkerBinding{Lifecycle: worker.NewLifecycle(e.ctx, sup), Supervisor: sup, Info: info, Started: time.Now()}
		e.mu.Lock()
		e.bindings = append(e.bindings, b)
		e.mu.Unlock()
		return b, nil
	}
	members := []ResidentMember{{Model: model.ID, Provider: model.Provider, Info: info, Config: cfg}}
	if e.extra {
		members = append(members, residentMember(e.t, modelB, "ok"))
	}
	s, err := NewResidentSet(e.ctx, noRestart, model.ID, members)
	if err != nil {
		return nil, err
	}
	if e.routing && !e.noRouting[n] {
		if _, err := s.SetRouting(applyPolicy); err != nil {
			return nil, err
		}
	}
	e.mu.Lock()
	e.sets = append(e.sets, s)
	e.mu.Unlock()
	return s, nil
}

func (e *applyEnv) record() string {
	b, _ := e.h.ReadActiveRecord()
	return string(b)
}

func (e *applyEnv) start() {
	e.t.Helper()
	if err := e.c.Start(); err != nil {
		e.t.Fatal(err)
	}
	waitFor(e.t, "the runtime READY", func() bool { return e.c.Snapshot().State == Ready })
}

func (e *applyEnv) apply(ctx context.Context) (ApplyResult, error) {
	return e.c.ApplyCertifiedVariant(ctx, ApplyParams{Variant: e.v.ID, Device: "cuda"})
}

func (e *applyEnv) activateCalls() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.activates)
}

func (e *applyEnv) openCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.opens
}

// serving is the default resident's worker provenance in one snapshot.
func (e *applyEnv) serving() (variant, execution string, snap Snapshot) {
	snap = e.c.Snapshot()
	if snap.Status == nil {
		return "", "", snap
	}
	info := snap.Status.Worker.Info
	return str(info, "variant_id"), str(info, "execution"), snap
}

func (e *applyEnv) requireServingSource() {
	e.t.Helper()
	variant, execution, snap := e.serving()
	if snap.State != Ready || variant != "" || execution != "source" || snap.Operation != nil {
		e.t.Fatalf("the source is not serving: state %s variant %q execution %q op %+v", snap.State, variant, execution, snap.Operation)
	}
}

func (e *applyEnv) requireServingVariant(id string) {
	e.t.Helper()
	variant, execution, snap := e.serving()
	if snap.State != Ready || variant != id || execution != "variant" || snap.Status.Runtime.Variant == nil || snap.Status.Runtime.Variant.ID != id || snap.Operation != nil {
		e.t.Fatalf("variant %s is not serving: state %s variant %q execution %q", id, snap.State, variant, execution)
	}
}

func asApplyError(t *testing.T, err error) *ApplyError {
	t.Helper()
	var ae *ApplyError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %v (%T), want *ApplyError", err, err)
	}
	return ae
}

// ---- success ----

func TestApplyCertifiedVariantProvesTheExactVariantAndAnswersATypedDecision(t *testing.T) {
	for name, single := range map[string]bool{"resident set": false, "single worker binding": true} {
		t.Run(name, func(t *testing.T) {
			e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.single = single })
			e.start()
			e.requireServingSource()
			before := e.record()
			oldPID := e.c.Snapshot().Status.Worker.PID

			res, err := e.apply(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if res.Model != setup.ClefFlash || res.Variant != e.v.ID || res.Device != "cuda" || res.ReportedDevice != "cuda" || res.DType != "bfloat16" ||
				res.Scheme != "W4A16" || !strings.Contains(res.Quantization, "W4A16") || res.QuantizedModules != 7 || res.Certification != eval.StateAccepted || res.PID == 0 || res.PID == oldPID {
				t.Fatalf("result %+v", res)
			}
			if res.Smoke.Question != "scope_expansion" || res.Smoke.Choice != "yes" || res.Smoke.Confidence != 0.8 {
				t.Fatalf("smoke %+v", res.Smoke)
			}
			if res.Previous.Model != setup.ClefFlash || res.Previous.Variant != "" || !slices.Equal(res.Previous.Running, []string{setup.ClefFlash}) {
				t.Fatalf("previous %+v", res.Previous)
			}
			if e.record() == before {
				t.Fatal("the activation record did not change")
			}
			e.requireServingVariant(e.v.ID)
			if got := e.activateCalls(); !slices.Equal(got, []string{"cuda " + setup.ClefFlash + " " + e.v.ID}) {
				t.Fatalf("activation calls %v", got)
			}
			s := e.c.Snapshot()
			if s.RestartRequired || s.Maintenance == nil || s.Maintenance.Kind != OpApply || s.Maintenance.Failure != nil || s.Maintenance.Target != "variant "+e.v.ID {
				t.Fatalf("snapshot %+v maintenance %+v", s.RestartRequired, s.Maintenance)
			}
			want := []string{"validating", "snapshotting", "runtime", "model", "variant", "activation", "rebinding", "awaiting_ready", "proving", "smoke", "finalizing"}
			if !slices.Equal(s.Maintenance.Phases, want) {
				t.Fatalf("phases %v, want %v", s.Maintenance.Phases, want)
			}
			for _, p := range s.Maintenance.Phases {
				if !slices.Contains(s.Maintenance.Plan, p) {
					t.Errorf("phase %s is not in the plan %v", p, s.Maintenance.Plan)
				}
			}
			// The previous source worker is gone: one runtime serves.
			if single {
				if len(e.bindings) != 2 || e.bindings[0].Running() || !e.bindings[1].Running() {
					t.Fatalf("bindings: old running=%v new running=%v", e.bindings[0].Running(), e.bindings[1].Running())
				}
			} else if len(e.sets) != 2 || e.sets[0].Running() || !e.sets[1].Running() {
				t.Fatal("the previous resident set still runs, or the new one does not")
			}
		})
	}
}

// StartApply is the same transaction as ApplyCertifiedVariant behind an
// acceptance: a refusal at admission is returned at once and changes nothing,
// and an accepted apply is observed through the controller's operation state.
func TestStartApplyAcceptsOneTransactionAndReportsThroughTheSnapshot(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	e.start()
	before := e.record()
	if err := e.c.StartApply(ApplyParams{Variant: e.v.ID}); err == nil || !strings.Contains(err.Error(), "device") {
		t.Fatalf("an apply without a device was accepted: %v", err)
	}
	if err := e.c.StartApply(ApplyParams{Device: "cuda"}); err == nil || !strings.Contains(err.Error(), "variant") {
		t.Fatalf("an apply without a variant was accepted: %v", err)
	}
	if e.record() != before || len(e.activateCalls()) != 0 {
		t.Fatal("a refused StartApply changed the activation")
	}
	if err := e.c.StartApply(ApplyParams{Variant: e.v.ID, Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	if err := e.c.StartApply(ApplyParams{Variant: e.v.ID, Device: "cuda"}); !errors.Is(err, ErrBusy) {
		t.Fatalf("a second apply while one runs = %v, want ErrBusy", err)
	}
	waitFor(t, "the apply to finish", func() bool {
		s := e.c.Snapshot()
		return s.Operation == nil && s.Maintenance != nil && s.Maintenance.Kind == OpApply
	})
	if m := e.c.Snapshot().Maintenance; m.Failure != nil || m.Target != "variant "+e.v.ID {
		t.Fatalf("maintenance %+v", m)
	}
	e.requireServingVariant(e.v.ID)
	if got := e.activateCalls(); !slices.Equal(got, []string{"cuda " + setup.ClefFlash + " " + e.v.ID}) {
		t.Fatalf("activation calls %v", got)
	}
}

// A failed background apply is the same recoverable failure: rolled back, with
// the failure and its diagnostic on the operation.
func TestStartApplyFailureRollsBackAndIsReadFromTheSnapshot(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	e.setMode("source", "ok")
	e.setMode(e.v.ID, "fatal")
	e.start()
	if err := e.c.StartApply(ApplyParams{Variant: e.v.ID, Device: "cuda"}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the apply to finish", func() bool {
		s := e.c.Snapshot()
		return s.Operation == nil && s.Maintenance != nil && s.Maintenance.Kind == OpApply
	})
	m := e.c.Snapshot().Maintenance
	if m.Failure == nil || !strings.Contains(m.Failure.Message, "the previous serving target was restored and verified") || m.Failure.Diagnostic == "" {
		t.Fatalf("maintenance failure %+v", m.Failure)
	}
	e.requireServingSource()
}

func TestApplyStartsAStoppedRuntimeAndLeavesUnrelatedStateAlone(t *testing.T) {
	e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.extra, e.routing = true, true })
	e.start()
	before := e.c.cfg.Residents()
	if _, err := e.apply(context.Background()); err != nil {
		t.Fatal(err)
	}
	e.requireServingVariant(e.v.ID)
	s := e.c.Snapshot()
	if !slices.Equal(e.c.cfg.Residents(), before) || s.ResidencyChanged || len(s.Residents) != 2 || s.Residents[0].Model != setup.ClefFlash || !s.Residents[0].Default || s.Residents[1].Model != modelB {
		t.Fatalf("residents %+v changed=%v", s.Residents, s.ResidencyChanged)
	}
	for _, r := range s.Residents {
		if !r.Running || r.Status.Worker.State != worker.StateReady {
			t.Fatalf("resident %s: %+v", r.Model, r.Status.Worker)
		}
	}
	if rs := e.sets[len(e.sets)-1].RoutingStatus(); rs == nil || rs.Policy.ID != applyPolicy.ID {
		t.Fatalf("routing policy %+v", rs)
	}

	// A runtime that was stopped is started by an apply: success means serving.
	e2 := newApplyEnv(t, "accepted")
	res, err := e2.apply(context.Background())
	if err != nil || res.Previous.Running != nil {
		t.Fatalf("res %+v err %v", res, err)
	}
	e2.requireServingVariant(e2.v.ID)
}

// ---- refusal before any mutation ----

func TestApplyRefusesWhatIsNotAcceptedBeforeAnyChange(t *testing.T) {
	tamper := func(e *applyEnv, fn func(dir, rec string)) {
		dir := e.h.Path(filepath.FromSlash(home.CertificationDir(e.v.ID)))
		es, _ := os.ReadDir(dir)
		for _, f := range es {
			if strings.HasSuffix(f.Name(), ".record.json") {
				fn(dir, f.Name())
			}
		}
	}
	for name, tc := range map[string]struct {
		certify string
		prep    func(*applyEnv)
		want    string
	}{
		"uncertified": {"none", nil, "no accepted certification record"},
		"rejected":    {"rejected", nil, "rejecting certification record"},
		"ambiguous": {"accepted", func(e *applyEnv) {
			tamper(e, func(dir, rec string) {
				b, _ := os.ReadFile(filepath.Join(dir, rec))
				os.WriteFile(filepath.Join(dir, "copy.record.json"), b, 0o644)
			})
		}, "cannot be determined"},
		"stale or mismatched": {"accepted", func(e *applyEnv) {
			tamper(e, func(dir, rec string) {
				b, _ := os.ReadFile(filepath.Join(dir, rec))
				os.WriteFile(filepath.Join(dir, rec), []byte(strings.Replace(string(b), e.v.ManifestSHA256(), strings.Repeat("0", 64), 1)), 0o644)
			})
		}, "not trusted"},
	} {
		t.Run(name, func(t *testing.T) {
			e := newApplyEnv(t, tc.certify)
			if tc.prep != nil {
				tc.prep(e)
			}
			e.start()
			before, pid := e.record(), e.c.Snapshot().Status.Worker.PID
			res, err := e.apply(context.Background())
			ae := asApplyError(t, err)
			if !errors.Is(err, setup.ErrVariantNotCertified) || !strings.Contains(err.Error(), tc.want) || ae.Phase != PhaseApplyValidate || ae.Mutated || ae.RolledBack || ae.Rollback != nil || res.Variant != "" {
				t.Fatalf("err = %v (%+v)", err, ae)
			}
			if e.record() != before || len(e.activateCalls()) != 0 || e.openCount() != 1 {
				t.Fatalf("a refused apply changed something: record changed=%v activations %v opens %d", e.record() != before, e.activateCalls(), e.openCount())
			}
			e.requireServingSource()
			if got := e.c.Snapshot().Status.Worker.PID; got != pid {
				t.Fatalf("the serving worker was restarted: pid %d -> %d", pid, got)
			}
			s := e.c.Snapshot()
			if s.Maintenance == nil || s.Maintenance.Failure == nil || s.Maintenance.Failure.Phase != PhaseApplyValidate || s.State == Failed {
				t.Fatalf("maintenance %+v state %s", s.Maintenance, s.State)
			}
		})
	}
}

func TestApplyRefusesInvalidRequestsAndStates(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	e.start()
	before := e.record()
	for name, p := range map[string]ApplyParams{
		"no variant":         {Device: "cuda"},
		"no device":          {Variant: e.v.ID},
		"an unknown device":  {Variant: e.v.ID, Device: "tpu"},
		"another source":     {Variant: e.v.ID, Device: "cuda", Model: setup.DefaultModel},
		"an unknown variant": {Variant: "clef-flash--w4a16--ffffffffffff", Device: "cuda"},
		"not a variant ID":   {Variant: "../x", Device: "cuda"},
	} {
		if _, err := e.c.ApplyCertifiedVariant(context.Background(), p); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if e.record() != before || len(e.activateCalls()) != 0 || e.openCount() != 1 {
		t.Fatal("a refused request changed something")
	}
	e.requireServingSource()

	// A serving target that is not materialized is refused unless the operator
	// allowed materialization, which then goes through the setup authority.
	spec, _ := setup.Desired("cuda")
	rdir := e.h.Path("runtime", spec.ID())
	os.RemoveAll(rdir)
	_, err := e.apply(context.Background())
	if ae := asApplyError(t, err); ae.Phase != PhaseApplyValidate || ae.Mutated || !strings.Contains(err.Error(), "not materialized") || e.materials != 0 || e.record() != before {
		t.Fatalf("err = %v materials %d", err, e.materials)
	}
	res, err := e.c.ApplyCertifiedVariant(context.Background(), ApplyParams{Variant: e.v.ID, Device: "cuda", Materialize: true})
	if err != nil || !res.Materialized || e.materials != 1 {
		t.Fatalf("res %+v err %v materials %d", res, err, e.materials)
	}
	e.requireServingVariant(e.v.ID)
}

// A running worker that no longer serves the activation record has no record
// that could restore it, so an apply asks for the restart first.
func TestApplyNeedsTheRestartTheLowLevelActivateAskedFor(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	e.start()
	if err := e.c.ActivateVariant(SetupParams{Device: "cuda", Model: setup.ClefFlash}, e.v.ID, false); err != nil {
		t.Fatal(err)
	}
	if s := waitIdle(t, e.c); !s.RestartRequired || s.Maintenance.Failure != nil {
		t.Fatalf("snapshot %+v %+v", s.RestartRequired, s.Maintenance)
	}
	before := e.record()
	if _, err := e.apply(context.Background()); !errors.Is(err, ErrRestartRequired) {
		t.Fatalf("err = %v, want ErrRestartRequired", err)
	}
	if e.record() != before || len(e.activateCalls()) != 1 {
		t.Fatal("a refused apply changed something")
	}
	// The low-level operations keep working: Restart applies the activation.
	if err := e.c.Restart(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the activated variant READY", func() bool { return e.c.Snapshot().State == Ready })
	e.requireServingVariant(e.v.ID)
	if s := e.c.Snapshot(); s.RestartRequired {
		t.Fatal("restart required after Restart")
	}
}

func TestApplyIsOneControllerAction(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	e.setMode("source", "ok")
	e.setMode(e.v.ID, "hang")
	e.c.cfg.ReadyTimeout = 60 * time.Second
	e.start()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := e.apply(ctx); done <- err }()
	waitFor(t, "the apply to wait for READY", func() bool {
		op := e.c.Snapshot().Operation
		return op != nil && op.Kind == OpApply && op.Phase == PhaseApplyReady
	})
	if err := e.c.Restart(); !errors.Is(err, ErrBusy) {
		t.Fatalf("Restart during an apply = %v, want ErrBusy", err)
	}
	if err := e.c.Stop(); !errors.Is(err, ErrBusy) {
		t.Fatalf("Stop during an apply = %v, want ErrBusy", err)
	}
	if _, err := e.apply(context.Background()); !errors.Is(err, ErrBusy) {
		t.Fatalf("a second apply = %v, want ErrBusy", err)
	}
	if s := e.c.Snapshot(); s.State != Starting {
		t.Fatalf("state during the apply = %s", s.State)
	}

	// Cancelling the apply is a failure after the activation, so it rolls back.
	cancel()
	err := <-done
	ae := asApplyError(t, err)
	if !errors.Is(err, context.Canceled) || !ae.RolledBack || ae.Phase != PhaseApplyReady {
		t.Fatalf("err = %v (%+v)", err, ae)
	}
	e.requireServingSource()
}

// ---- failure after the activation: rollback ----

func TestApplyRollsBackAnyFailureAfterTheActivation(t *testing.T) {
	forgeSource(t) // Install the shared read-only catalog before parallel cases start.
	for name, tc := range map[string]struct {
		mode  string
		phase string
		want  string
		tune  func(*applyEnv)
	}{
		"the wrong variant is running":     {"wrong_variant", PhaseApplyProve, "serving provenance", nil},
		"the source came up instead":       {"source", PhaseApplyProve, "did not load the persisted variant", nil},
		"cpu came up for the cuda request": {"cpu_on_cuda", PhaseApplyProve, "no fallback to another device", nil},
		"the wrong dtype":                  {"wrong_dtype", PhaseApplyProve, "computes in", nil},
		"no quantized execution":           {"unquantized", PhaseApplyProve, "quantized W4A16 execution", nil},
		"the worker cannot load":           {"fatal", PhaseApplyReady, "model_load", nil},
		"READY never arrives":              {"hang", PhaseApplyReady, "did not become READY", func(e *applyEnv) { e.c.cfg.ReadyTimeout = 400 * time.Millisecond }},
		"the smoke is an error":            {"decide_error", PhaseApplySmoke, "smoke failed", nil},
		"the worker dies in the smoke":     {"crash_on_decide", PhaseApplySmoke, "smoke failed", nil},
		"the smoke is not a distribution":  {"bad_probabilities", PhaseApplySmoke, "invalid decision", nil},
		"the smoke does not answer":        {"slow", PhaseApplySmoke, "did not answer", func(e *applyEnv) { e.c.cfg.SmokeTimeout = 300 * time.Millisecond }},
	} {
		for shape, single := range map[string]bool{"set": false, "binding": true} {
			if single && !slices.Contains([]string{"wrong_variant", "fatal", "decide_error"}, tc.mode) {
				continue // the single-worker binding differs only in how it is observed
			}
			t.Run(name+"/"+shape, func(t *testing.T) {
				t.Parallel()
				e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.single = single })
				e.start()
				before, oldPID := e.record(), e.c.Snapshot().Status.Worker.PID
				e.setMode(e.v.ID, tc.mode)
				if tc.tune != nil {
					tc.tune(e)
				}
				_, err := e.apply(context.Background())
				ae := asApplyError(t, err)
				if !ae.Mutated || !ae.RolledBack || ae.Rollback != nil || ae.Phase != tc.phase || !strings.Contains(ae.Primary.Error(), tc.want) {
					t.Fatalf("err = %v (%+v), want phase %s containing %q", err, ae, tc.phase, tc.want)
				}
				if !strings.Contains(err.Error(), "restored and verified") {
					t.Fatalf("the rollback is not reported: %v", err)
				}
				if e.record() != before {
					t.Fatalf("the activation record was not restored:\n%s\nwas\n%s", e.record(), before)
				}
				e.requireServingSource()
				if pid := e.c.Snapshot().Status.Worker.PID; pid == 0 || pid == oldPID {
					t.Fatalf("restored worker pid %d (before %d): want a fresh READY worker", pid, oldPID)
				}
				s := e.c.Snapshot()
				if s.RestartRequired || s.State == Failed || s.Maintenance == nil || s.Maintenance.Kind != OpApply || s.Maintenance.Failure == nil || s.Maintenance.Failure.Phase != tc.phase ||
					!strings.Contains(s.Maintenance.Failure.Message, tc.want) || s.Maintenance.Phases[len(s.Maintenance.Phases)-1] != PhaseApplyRollback {
					t.Fatalf("snapshot %+v maintenance %+v", s.State, s.Maintenance)
				}
				// Exactly one failed candidate and one restored runtime were bound.
				if e.openCount() != 3 {
					t.Fatalf("opens = %d", e.openCount())
				}
			})
		}
	}
}

func TestApplyRejectsAStatusThatDoesNotNameTheVariant(t *testing.T) {
	forgeSource(t) // Keep the catalog immutable while independent cases run.
	for name, lie := range map[string]func(*server.Runtime){
		"no variant":       func(r *server.Runtime) { r.Variant = nil },
		"another manifest": func(r *server.Runtime) { r.Variant.ManifestSHA256 = strings.Repeat("1", 64) },
		"not accepted":     func(r *server.Runtime) { r.Variant.Certification = eval.StateExperimental },
		"another device":   func(r *server.Runtime) { r.Device = "cpu" },
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.statusLie = lie })
			e.start()
			before := e.record()
			_, err := e.apply(context.Background())
			ae := asApplyError(t, err)
			if ae.Phase != PhaseApplyProve || !ae.RolledBack || e.record() != before {
				t.Fatalf("err = %v (%+v)", err, ae)
			}
			e.requireServingSource()
		})
	}
}

func TestApplyRestoresThePreviousVariantTarget(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	h := e.h
	prior := copyAnotherVariant(t, h)
	if prior.ID == e.v.ID {
		t.Fatal("the second build is the same variant")
	}
	certifyFixture(t, h, e.model, prior, true)
	e.writeActive(home.Active{Runtime: mustSpec(t, "cuda"), ModelID: e.model.ID, Model: setup.ModelDirName(e.model), Device: "cuda", Variant: prior.ID})
	e.start()
	e.requireServingVariant(prior.ID)
	before := e.record()

	e.setMode(e.v.ID, "source")
	_, err := e.apply(context.Background())
	if ae := asApplyError(t, err); !ae.RolledBack || ae.Phase != PhaseApplyProve {
		t.Fatalf("err = %v", err)
	}
	if e.record() != before {
		t.Fatal("the previous variant record was not restored")
	}
	e.requireServingVariant(prior.ID)

	// And the restored variant target is itself verified: a restored worker
	// that reports the wrong variant makes the rollback fail, loudly.
	e.setMode(e.v.ID, "wrong_dtype")
	e.setMode(prior.ID, "wrong_variant")
	_, err = e.apply(context.Background())
	ae := asApplyError(t, err)
	if ae.RolledBack || ae.Rollback == nil || !strings.Contains(ae.Rollback.Error(), "verifying the restored target") || !strings.Contains(ae.Primary.Error(), "computes in") || e.record() != before {
		t.Fatalf("err = %v (%+v)", err, ae)
	}
}

// copyAnotherVariant publishes a second variant of the same source into h: it
// is built (with other bytes, so another identity) in a scratch home that has
// the same fixture source, then copied.
func copyAnotherVariant(t *testing.T, h home.Home) home.VariantManifest {
	t.Helper()
	scratch := forgeSource(t)
	res, err := optimize.Build(context.Background(), scratch, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{Salt: "prior"}, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	src, dst := scratch.VariantDir(setup.ClefFlash, res.Variant.ID), h.VariantDir(setup.ClefFlash, res.Variant.ID)
	err = filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		if d.IsDir() {
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(dst, rel), b, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return res.Variant
}

func mustSpec(t *testing.T, device string) string {
	t.Helper()
	spec, err := setup.Desired(device)
	if err != nil {
		t.Fatal(err)
	}
	return spec.ID()
}

func TestApplyRollbackKeepsAStoppedRuntimeStopped(t *testing.T) {
	for name, operatorStopped := range map[string]bool{"never started": false, "stopped by the operator": true} {
		t.Run(name, func(t *testing.T) {
			e := newApplyEnv(t, "accepted")
			if operatorStopped {
				e.start()
				if err := e.c.Stop(); err != nil {
					t.Fatal(err)
				}
			}
			before := e.record()
			e.setMode(e.v.ID, "wrong_variant")
			_, err := e.apply(context.Background())
			ae := asApplyError(t, err)
			if !ae.RolledBack || ae.Rollback != nil {
				t.Fatalf("err = %v", err)
			}
			s := e.c.Snapshot()
			if e.record() != before || s.State != Installed || s.OperatorStopped != operatorStopped {
				t.Fatalf("record restored=%v state %s operator-stopped %v", e.record() == before, s.State, s.OperatorStopped)
			}
			for _, set := range e.sets {
				if set.Running() {
					t.Fatal("a runtime that was stopped before the apply is running after its rollback")
				}
			}
			// The restored binding is the previous activation's, still startable.
			if err := e.c.Start(); err != nil {
				t.Fatal(err)
			}
			waitFor(t, "the restored source READY", func() bool { return e.c.Snapshot().State == Ready })
			e.requireServingSource()
		})
	}
}

func TestApplyRollbackRestoresThePreviousResidentSet(t *testing.T) {
	t.Run("both running", func(t *testing.T) {
		e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.extra, e.routing = true, true })
		e.start()
		waitFor(t, "both residents READY", func() bool {
			for _, r := range e.c.Snapshot().Residents {
				if r.Status.Worker.State != worker.StateReady {
					return false
				}
			}
			return true
		})
		e.setMode(e.v.ID, "wrong_variant")
		_, err := e.apply(context.Background())
		if ae := asApplyError(t, err); !ae.RolledBack {
			t.Fatalf("err = %v", err)
		}
		e.requireServingSource()
		s := e.c.Snapshot()
		if len(s.Residents) != 2 {
			t.Fatalf("residents %+v", s.Residents)
		}
		for _, r := range s.Residents {
			if !r.Running || r.Status.Worker.State != worker.StateReady {
				t.Fatalf("resident %s was not restored: %+v", r.Model, r.Status.Worker)
			}
		}
		if rs := e.sets[len(e.sets)-1].RoutingStatus(); rs == nil || rs.Policy.ID != applyPolicy.ID {
			t.Fatalf("routing %+v", rs)
		}
	})
	t.Run("one resident was stopped", func(t *testing.T) {
		e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.extra = true })
		e.start()
		waitFor(t, "both residents READY", func() bool {
			for _, r := range e.c.Snapshot().Residents {
				if r.Status.Worker.State != worker.StateReady {
					return false
				}
			}
			return true
		})
		if err := e.c.StopResident(modelB); err != nil {
			t.Fatal(err)
		}
		e.setMode(e.v.ID, "wrong_variant")
		res, err := e.apply(context.Background())
		_ = res
		if ae := asApplyError(t, err); !ae.RolledBack {
			t.Fatalf("err = %v", err)
		}
		e.requireServingSource()
		for _, r := range e.c.Snapshot().Residents {
			want := r.Model != modelB
			if r.Running != want {
				t.Fatalf("resident %s running=%v, want %v: the previous resident set was not restored", r.Model, r.Running, want)
			}
		}
	})
}

func TestApplyRollbackFailureKeepsBothErrors(t *testing.T) {
	t.Run("the previous runtime cannot be bound", func(t *testing.T) {
		e := newApplyEnv(t, "accepted")
		e.start()
		before := e.record()
		e.failOpenAt = 3 // 1: the initial binding, 2: the apply's, 3: the rollback's
		e.setMode(e.v.ID, "wrong_variant")
		_, err := e.apply(context.Background())
		ae := asApplyError(t, err)
		if ae.RolledBack || ae.Rollback == nil || !strings.Contains(ae.Primary.Error(), "serving provenance") || !strings.Contains(ae.Rollback.Error(), "cannot be bound") {
			t.Fatalf("err = %v (%+v)", err, ae)
		}
		for _, want := range []string{"serving provenance", "additionally the rollback failed", "cannot be bound"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("%q missing from %v", want, err)
			}
		}
		// Both errors are in the chain, and the record was restored regardless.
		if !errors.Is(err, ae.Primary) || !errors.Is(err, ae.Rollback) || e.record() != before {
			t.Fatal("an error was lost")
		}
		if s := e.c.Snapshot(); s.Maintenance == nil || s.Maintenance.Failure == nil || !strings.Contains(s.Maintenance.Failure.Message, "additionally the rollback failed") {
			t.Fatalf("maintenance %+v", s.Maintenance)
		}
	})
	t.Run("the restored target is not the previous one", func(t *testing.T) {
		e := newApplyEnv(t, "accepted")
		e.start()
		e.setMode("source", "cpu_on_cuda") // the restored source worker reports cpu
		e.setMode(e.v.ID, "wrong_variant")
		_, err := e.apply(context.Background())
		ae := asApplyError(t, err)
		if ae.RolledBack || ae.Rollback == nil || !strings.Contains(ae.Rollback.Error(), "was restored but the worker is on") || !strings.Contains(ae.Primary.Error(), "serving provenance") {
			t.Fatalf("err = %v (%+v)", err, ae)
		}
	})
}

func TestApplyFailsWhenTheRoutingPolicyChanged(t *testing.T) {
	e := newApplyEnv(t, "accepted", func(e *applyEnv) { e.extra, e.routing = true, true; e.noRouting = map[int]bool{2: true} })
	e.start()
	_, err := e.apply(context.Background())
	ae := asApplyError(t, err)
	if ae.Phase != PhaseApplyFinal || !ae.RolledBack || !strings.Contains(err.Error(), "routing policy changed") {
		t.Fatalf("err = %v (%+v)", err, ae)
	}
	e.requireServingSource()
}

func TestApplyFailureLeavesABoundedDiagnostic(t *testing.T) {
	e := newApplyEnv(t, "accepted")
	e.start()
	e.setMode(e.v.ID, "source")
	_, err := e.apply(context.Background())
	id := DiagnosticID(err)
	if id == "" || asApplyError(t, err).Phase != PhaseApplyProve {
		t.Fatalf("diagnostic %q err %v", id, err)
	}
	if f := e.c.Snapshot().Maintenance.Failure; f == nil || f.Diagnostic != id {
		t.Fatalf("failure %+v", f)
	}
}

// ---- the READY gate ----

// scriptRT is a runtime whose status follows a script, one entry per read.
type scriptRT struct {
	mu     sync.Mutex
	script []worker.Snapshot
	reads  int
}

func (s *scriptRT) Start() bool   { return true }
func (s *scriptRT) Stop()         {}
func (s *scriptRT) Restart()      {}
func (s *scriptRT) Running() bool { return true }
func (s *scriptRT) Status() server.Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := min(s.reads, len(s.script)-1)
	s.reads++
	return server.Status{Worker: s.script[i]}
}

func TestAwaitServingJudgesOneCoherentObservationPerPoll(t *testing.T) {
	starting := worker.Snapshot{State: worker.StateStarting}
	ready := worker.Snapshot{State: worker.StateReady, Ready: true, Starts: 1, PID: 42}
	rt := &scriptRT{script: []worker.Snapshot{starting, {State: worker.StateReady, Ready: true, Starts: 0}, starting, ready}}
	c := New(Config{})
	views, err := c.awaitServing(context.Background(), rt, nil, 5*time.Second, redact.New(""))
	if err != nil || rt.reads != 4 || views[0].Status.Worker.PID != 42 {
		t.Fatalf("views %+v err %v reads %d: want the observation that was READY, taken in the same read", views, err, rt.reads)
	}

	failed := &scriptRT{script: []worker.Snapshot{starting, {State: worker.StateFailed, LastFailure: &worker.FailureView{Class: worker.ClassModelLoad, Message: "no weights"}}}}
	if _, err := c.awaitServing(context.Background(), failed, nil, 5*time.Second, redact.New("")); err == nil || !strings.Contains(err.Error(), "no weights") || failed.reads != 2 {
		t.Fatalf("err = %v reads %d: a failed worker ends the wait", err, failed.reads)
	}
	never := &scriptRT{script: []worker.Snapshot{starting}}
	if _, err := c.awaitServing(context.Background(), never, nil, 50*time.Millisecond, redact.New("")); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v", err)
	}
}

func TestApplyErrorKeepsPrimaryAndSecondary(t *testing.T) {
	p, r := errors.New("primary"), errors.New("secondary")
	if got := (&ApplyError{Primary: p}).Error(); got != "primary" {
		t.Fatal(got)
	}
	if got := (&ApplyError{Primary: p, RolledBack: true}).Error(); !strings.HasPrefix(got, "primary; ") || !strings.Contains(got, "restored and verified") {
		t.Fatal(got)
	}
	e := &ApplyError{Primary: p, Rollback: r}
	if got := e.Error(); !strings.HasPrefix(got, "primary; additionally") || !strings.HasSuffix(got, "secondary") || !errors.Is(e, p) || !errors.Is(e, r) {
		t.Fatal(got)
	}
	if !isMaintenance(OpApply) {
		t.Fatal("an apply failure would fail the application")
	}
	if fmt.Sprint(plan(OpApply, "")) == "" {
		t.Fatal("no plan")
	}
}
