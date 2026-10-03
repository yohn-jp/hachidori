package trial

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

const (
	mlp0 = "block.00.mlp"
	mlp1 = "block.01.mlp"
	mlp2 = "block.02.mlp"
	mlp3 = "block.03.mlp"
	mod0 = "model.language_model.layers.0.mlp.gate_proj"
	mod1 = "model.language_model.layers.1.mlp.gate_proj"
	mod2 = "model.language_model.layers.2.mlp.gate_proj"
)

func mustRun(t *testing.T, s *Session, plan home.TuningPlan, ev Evaluator) Result {
	t.Helper()
	r, err := s.RunTrial(context.Background(), plan, ev)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestSessionStartsAtSourcePrecisionAndKeepsCanonicalInRAM(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	st := s.Stats()
	if st.Cache.CanonicalBytes != int64(len(f.modules))*256*256*2 || st.Cache.TransformedBytes != 0 || st.Trials != 0 {
		t.Fatalf("stats %+v", st)
	}
	for _, p := range s.Current() {
		if p != home.PolicySourcePrecision {
			t.Fatalf("session does not start at source precision: %v", s.Current())
		}
	}
	if len(f.packedModules()) != 0 {
		t.Fatal("the resident model is not at source precision")
	}
}

func TestOpenRefusesModulesNoGroupAddresses(t *testing.T) {
	w := newWorld(t)
	f := newFake(append(moduleNames(layout()), "model.language_model.layers.9.mlp.extra"))
	_, err := Open(context.Background(), f, w.source, w.plan(t), Options{BudgetBytes: 1 << 30})
	var u *UnmappedError
	if !errors.As(err, &u) || len(u.Modules) != 1 || !f.closed {
		t.Fatalf("err = %v (closed %v)", err, f.closed)
	}
}

func TestOpenRefusesAModelThatIsNotAtSourcePrecision(t *testing.T) {
	w := newWorld(t)
	f := newFake(moduleNames(layout()))
	f.rep[mod0] = RepresentationPacked
	if _, err := Open(context.Background(), f, w.source, w.plan(t), Options{BudgetBytes: 1 << 30}); err == nil || !strings.Contains(err.Error(), "not at its source precision") {
		t.Fatalf("err = %v", err)
	}
}

func TestOpenRefusesACanonicalSourceThatExceedsTheBudget(t *testing.T) {
	w := newWorld(t)
	f := newFake(moduleNames(layout()))
	_, err := Open(context.Background(), f, w.source, w.plan(t), Options{BudgetBytes: 1 << 10})
	var b *BudgetError
	if !errors.As(err, &b) {
		t.Fatalf("err = %v", err)
	}
}

func TestFirstTrialTransformsAndRepeatedTrialReusesCachedComponents(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	ev := &evaluator{}
	a := mustRun(t, s, w.planW4(t, mlp0, mlp1), ev)
	if a.Assembly.Mode != AssemblyDifferential || a.Assembly.CacheMisses != 2 || a.Assembly.CacheHits != 0 || len(a.Assembly.Transformed) != 2 {
		t.Fatalf("first trial assembly %+v", a.Assembly)
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0, mod1}) || a.Assembly.BytesToGPU == 0 {
		t.Fatalf("packed %v, assembly %+v", f.packedModules(), a.Assembly)
	}
	// Back to source, then the same quantization again: nothing is re-transformed.
	mustRun(t, s, w.planW4(t, mlp0), ev) // mlp1 returns to source
	f.resetCounts()
	again := mustRun(t, s, w.planW4(t, mlp0, mlp1), ev)
	if again.Assembly.CacheHits != 1 || again.Assembly.CacheMisses != 0 || f.calls["transform"] != 0 || len(again.Assembly.Transformed) != 0 {
		t.Fatalf("repeated trial re-transformed: %+v (transform calls %d)", again.Assembly, f.calls["transform"])
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0, mod1}) {
		t.Fatalf("packed %v", f.packedModules())
	}
	st := s.Stats()
	if st.Cache.Hits < 1 || st.Cache.Misses != 2 || st.Trials != 3 {
		t.Fatalf("stats %+v", st)
	}
}

func TestDifferentialAssemblyTouchesOnlyTheChangedGroups(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	ev := &evaluator{}
	// A: layers 0 and 1 quantized, layer 2 source. B: layer 1 returns to source.
	mustRun(t, s, w.planW4(t, mlp0, mlp1), ev)
	f.resetCounts()
	b := mustRun(t, s, w.planW4(t, mlp0), ev)
	if len(f.applied) != 1 || !reflect.DeepEqual(f.applied[0], []string{mlp1}) || f.applyKinds[0] != "apply" {
		t.Fatalf("assembly touched %v (%v), want only %s", f.applied, f.applyKinds, mlp1)
	}
	if len(b.Assembly.Changed) != 1 || b.Assembly.Changed[0] != (Change{Group: mlp1, From: home.PolicyW4A16, To: home.PolicySourcePrecision}) {
		t.Fatalf("changed %+v", b.Assembly.Changed)
	}
	if b.Assembly.Reused != len(b.Plan.Groups)-1 {
		t.Fatalf("reused %d of %d groups", b.Assembly.Reused, len(b.Plan.Groups))
	}
	if f.calls["transform"] != 0 || b.Assembly.BytesToGPU != 256*256*2 || b.Assembly.BytesReleased != 256*256/2 {
		t.Fatalf("transform calls %d, assembly %+v", f.calls["transform"], b.Assembly)
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0}) {
		t.Fatalf("packed %v", f.packedModules())
	}
	// The resulting state is exactly plan B's.
	if cur := s.Current(); cur[mlp0] != home.PolicyW4A16 || cur[mlp1] != home.PolicySourcePrecision || cur[mlp2] != home.PolicySourcePrecision {
		t.Fatalf("current %v", cur)
	}
}

func TestAnIdenticalPlanChangesNothing(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	ev := &evaluator{}
	mustRun(t, s, w.planW4(t, mlp0), ev)
	f.resetCounts()
	r := mustRun(t, s, w.planW4(t, mlp0), ev)
	if len(r.Assembly.Changed) != 0 || r.Assembly.BytesToGPU != 0 || f.calls["transform"] != 0 {
		t.Fatalf("assembly %+v", r.Assembly)
	}
}

func TestATrialAcrossAnotherModelStructureIsRefused(t *testing.T) {
	w := newWorld(t)
	s, _ := openSession(t, w, Options{})
	plan := w.plan(t)
	plan.Groups = plan.Groups[:len(plan.Groups)-1]
	_, err := s.RunTrial(context.Background(), plan, &evaluator{})
	var te *Error
	if !errors.As(err, &te) || !errors.Is(err, ErrStructure) || !te.Restored {
		t.Fatalf("err = %v", err)
	}
}

func TestTransformFailureLeavesTheModelAndCacheUsable(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	f.failTransform = 2
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0, mlp1, mlp2), &evaluator{})
	var te *Error
	if !errors.As(err, &te) || te.Phase != "prepare" || !te.Restored || te.Recovery != "unchanged" {
		t.Fatalf("err = %v", err)
	}
	if len(f.packedModules()) != 0 || f.calls["apply"] != 0 {
		t.Fatal("the model was changed by a trial that failed to prepare")
	}
	st := s.Stats()
	if st.Cache.Pinned != 0 || st.Broken {
		t.Fatalf("stats %+v", st)
	}
	// The component transformed before the failure is valid cache, and the
	// session still works.
	f.failTransform = 0
	f.resetCounts()
	r := mustRun(t, s, w.planW4(t, mlp0, mlp1, mlp2), &evaluator{})
	if r.Assembly.CacheHits != 1 || r.Assembly.CacheMisses != 2 {
		t.Fatalf("assembly %+v", r.Assembly)
	}
}

func TestUnsupportedReplacementIsRefusedUnlessReconstructionIsAllowed(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	f.validate = &UnsupportedError{Group: mlp0, Reason: "the module is wrapped"}
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), &evaluator{})
	var u *UnsupportedError
	var te *Error
	if !errors.As(err, &u) || !errors.As(err, &te) || te.Phase != "validate" || len(f.packedModules()) != 0 || f.calls["apply"] != 0 || f.calls["reconstruct"] != 0 {
		t.Fatalf("err = %v", err)
	}

	s2, f2 := openSession(t, w, Options{AllowReconstruct: true})
	f2.validate = &UnsupportedError{Group: mlp0, Reason: "the module is wrapped"}
	r := mustRun(t, s2, w.planW4(t, mlp0), &evaluator{})
	if r.Assembly.Mode != AssemblyReconstruct || !strings.Contains(r.Assembly.Reason, "wrapped") || f2.calls["reconstruct"] != 1 || f2.calls["apply"] != 0 {
		t.Fatalf("assembly %+v", r.Assembly)
	}
	if !reflect.DeepEqual(f2.packedModules(), []string{mod0}) {
		t.Fatalf("packed %v", f2.packedModules())
	}
}

func TestIncompatibleReplacementIsNeverWorkedAround(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{AllowReconstruct: true})
	f.validate = &IncompatibleError{Group: mlp0, Reason: "shape differs"}
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), &evaluator{})
	var inc *IncompatibleError
	if !errors.As(err, &inc) || f.calls["apply"]+f.calls["reconstruct"] != 0 {
		t.Fatalf("err = %v", err)
	}
}

func TestApplyFailureThatChangedNothingIsACleanFailure(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	mustRun(t, s, w.planW4(t, mlp0), &evaluator{})
	f.failApply = f.calls["apply"] + 1
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0, mlp1), &evaluator{})
	var te *Error
	if !errors.As(err, &te) || te.Phase != "apply" || !te.Restored || te.Recovery != "unchanged" {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0}) || f.calls["reconstruct"] != 0 {
		t.Fatalf("packed %v", f.packedModules())
	}
}

func TestApplyFailureThatLostTheStateIsRebuiltFromRAM(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	mustRun(t, s, w.planW4(t, mlp0), &evaluator{})
	f.failApply, f.failApplyLost = f.calls["apply"]+1, true
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0, mlp1, mlp2), &evaluator{})
	var te *Error
	if !errors.As(err, &te) || !te.Restored || te.Recovery != "reconstruct" {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0}) {
		t.Fatalf("the known-good state was not restored: packed %v", f.packedModules())
	}
	if s.Stats().Broken {
		t.Fatal("a restored session must stay usable")
	}
	mustRun(t, s, w.planW4(t, mlp0, mlp1), &evaluator{})
}

func TestUnrestorableFailureBreaksTheSession(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	f.failApply, f.failApplyLost, f.failReconstruct = 1, true, true
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), &evaluator{})
	var te *Error
	if !errors.As(err, &te) || te.Restored || te.Lost == nil {
		t.Fatalf("err = %v", err)
	}
	if _, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), &evaluator{}); !errors.Is(err, ErrBroken) {
		t.Fatalf("a broken session ran a trial: %v", err)
	}
	if !s.Stats().Broken {
		t.Fatal("stats do not report the broken session")
	}
}

func TestApplyThatSilentlyProducesAnotherModelIsNotEvaluated(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	f.badApply = true
	ev := &evaluator{}
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), ev)
	var te *Error
	if !errors.As(err, &te) || te.Phase != "apply" || ev.calls != 0 {
		t.Fatalf("err = %v, evaluator ran %d times", err, ev.calls)
	}
}

func TestEvaluationFailureRestoresThePreviousState(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	mustRun(t, s, w.planW4(t, mlp0), &evaluator{})
	boom := errors.New("worker stopped answering")
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0, mlp1, mlp2), &evaluator{fn: func(context.Context, Assembled) (Measurement, error) { return Measurement{}, boom }})
	var te *Error
	if !errors.As(err, &te) || !errors.Is(err, boom) || te.Phase != "evaluate" || !te.Restored || te.Recovery != "reverse-delta" {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0}) {
		t.Fatalf("packed %v", f.packedModules())
	}
	if cur := s.Current(); cur[mlp1] != home.PolicySourcePrecision || cur[mlp0] != home.PolicyW4A16 {
		t.Fatalf("current %v", cur)
	}
	if got := s.Stats().Trials; got != 1 {
		t.Fatalf("a failed trial was counted: %d", got)
	}
}

func TestCancellationDuringEvaluationRestoresAndRecordsNothing(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	ctx, cancel := context.WithCancel(context.Background())
	ev := &evaluator{fn: func(ctx context.Context, _ Assembled) (Measurement, error) {
		cancel()
		return measured(), nil // even a finished measurement is dropped once cancelled
	}}
	_, err := s.RunTrial(ctx, w.planW4(t, mlp0), ev)
	var te *Error
	if !errors.As(err, &te) || !errors.Is(err, context.Canceled) || !te.Restored {
		t.Fatalf("err = %v", err)
	}
	if len(f.packedModules()) != 0 {
		t.Fatalf("a cancelled trial left %v quantized", f.packedModules())
	}
	if s.Stats().Trials != 0 {
		t.Fatal("a cancelled trial was committed")
	}
}

func TestCancellationBeforeStartTouchesNothing(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.RunTrial(ctx, w.planW4(t, mlp0), &evaluator{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if f.calls["transform"]+f.calls["apply"] != 0 {
		t.Fatal("work was done for a cancelled trial")
	}
}

func TestAMeasurementThatIsNotTrialEvidenceIsRefused(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	for name, mutate := range map[string]func(*Measurement){
		"certification mode": func(m *Measurement) { m.Mode = "certification" },
		"variant served":     func(m *Measurement) { m.Served.Execution, m.Served.VariantID = "variant", "v1" },
		"not the trial":      func(m *Measurement) { m.Served.Execution = "source" },
		"unstable worker":    func(m *Measurement) { m.ResidentStable = false },
	} {
		_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), &evaluator{fn: func(context.Context, Assembled) (Measurement, error) {
			m := measured()
			mutate(&m)
			return m, nil
		}})
		var te *Error
		if !errors.As(err, &te) || te.Phase != "evaluate" || !te.Restored {
			t.Errorf("%s: err = %v", name, err)
		}
		if len(f.packedModules()) != 0 {
			t.Errorf("%s: model not restored", name)
		}
	}
}

func TestRAMBudgetIsEnforcedWithDeterministicEviction(t *testing.T) {
	w := newWorld(t)
	// One packed component of a 256x256 module is 33808 bytes; allow the
	// canonical source plus exactly three of them. The known-good state's
	// components are pinned (a rollback needs them), so room is made only from
	// components no active or current plan uses.
	component := int64(256*32*4 + 256*2*2 + 16)
	canonical := int64(len(moduleNames(layout()))) * 256 * 256 * 2
	budget := canonical + 3*component
	s, f := openSession(t, w, Options{BudgetBytes: budget})
	ev := &evaluator{}
	mustRun(t, s, w.planW4(t, mlp0, mlp1), ev) // cache: mlp0, mlp1
	mustRun(t, s, w.planW4(t, mlp0), ev)       // mlp1 returns to source: cached, unpinned
	mustRun(t, s, w.planW4(t, mlp0, mlp2), ev) // cache: mlp0, mlp1, mlp2 (full)
	f.resetCounts()
	r := mustRun(t, s, w.planW4(t, mlp0, mlp3), ev) // needs a fourth: evicts mlp1, the only unpinned one
	if len(r.Assembly.Evicted) != 1 || f.calls["release"] != 1 || len(r.Assembly.Transformed) != 1 {
		t.Fatalf("assembly %+v (releases %d)", r.Assembly, f.calls["release"])
	}
	evicted := r.Assembly.Evicted[0]
	if st := s.Stats().Cache; st.Evictions != 1 || st.CurrentBytes > budget || st.EvictedBytes != component {
		t.Fatalf("cache %+v", st)
	}
	// Eviction costs a re-transform, never a different model: returning to mlp1
	// builds it again (evicting mlp2, now the only unpinned component) and the
	// resulting model is the plan's.
	f.resetCounts()
	back := mustRun(t, s, w.planW4(t, mlp0, mlp1), ev)
	if back.Assembly.CacheMisses != 1 || !reflect.DeepEqual(f.packedModules(), []string{mod0, mod1}) || len(back.Assembly.Evicted) != 1 || back.Assembly.Evicted[0] == evicted {
		t.Fatalf("assembly %+v packed %v", back.Assembly, f.packedModules())
	}
}

func TestATrialTooLargeForTheBudgetFailsWithoutChangingTheModel(t *testing.T) {
	w := newWorld(t)
	component := int64(256*32*4 + 256*2*2 + 16)
	canonical := int64(len(moduleNames(layout()))) * 256 * 256 * 2
	s, f := openSession(t, w, Options{BudgetBytes: canonical + component})
	mustRun(t, s, w.planW4(t, mlp0), &evaluator{})
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp0, mlp1), &evaluator{})
	var be *BudgetError
	var te *Error
	if !errors.As(err, &be) || !errors.As(err, &te) || te.Phase != "prepare" || !te.Restored {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(f.packedModules(), []string{mod0}) || f.calls["release"] != 0 {
		t.Fatalf("packed %v, releases %d: the active component must never be evicted", f.packedModules(), f.calls["release"])
	}
}

func TestReleaseFailureOfAnEvictedComponentBreaksTheAccounting(t *testing.T) {
	w := newWorld(t)
	component := int64(256*32*4 + 256*2*2 + 16)
	canonical := int64(len(moduleNames(layout()))) * 256 * 256 * 2
	s, f := openSession(t, w, Options{BudgetBytes: canonical + 2*component})
	mustRun(t, s, w.planW4(t, mlp0), &evaluator{})
	mustRun(t, s, w.planW4(t, mlp1), &evaluator{}) // mlp0 returns to source: cached, unpinned; mlp1 active
	f.failRelease = true
	_, err := s.RunTrial(context.Background(), w.planW4(t, mlp2), &evaluator{})
	if err == nil || !s.Stats().Broken {
		t.Fatalf("err = %v, broken %v", err, s.Stats().Broken)
	}
}

func TestTrialsDoNotCreateVariantArtifactsOrTouchTheHome(t *testing.T) {
	w := newWorld(t)
	// An immutable variant and the source files exist; trials must leave every
	// byte of the home as it was.
	vdir := w.h.VariantDir(w.model.ID, "variant-1")
	if err := os.MkdirAll(vdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(vdir, "hachidori-variant.json"), []byte("immutable"), 0o644); err != nil {
		t.Fatal(err)
	}
	before := snapshotTree(t, w.h.Root)
	s, _ := openSession(t, w, Options{})
	for i := 0; i < 6; i++ {
		mustRun(t, s, w.planW4(t, [][]string{{mlp0}, {mlp0, mlp1}, {mlp2}, {mlp1, mlp3}, {mlp0, mlp1, mlp2}, {mlp3}}[i]...), &evaluator{})
	}
	if after := snapshotTree(t, w.h.Root); !reflect.DeepEqual(before, after) {
		t.Fatalf("trials changed the home:\nbefore %v\nafter  %v", before, after)
	}
	if entries, _ := os.ReadDir(w.h.VariantsDir(w.model.ID)); len(entries) != 1 {
		t.Fatalf("variants dir has %d entries after trials", len(entries))
	}
}

func snapshotTree(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	_ = filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		b, _ := os.ReadFile(p)
		out[p] = info.ModTime().String() + string(b)
		return nil
	})
	return out
}

func TestTrialResultRecordsAssemblyComponentsAndResources(t *testing.T) {
	w := newWorld(t)
	s, _ := openSession(t, w, Options{})
	r := mustRun(t, s, w.planW4(t, mlp0, mlp1), &evaluator{})
	if r.PlanSHA256 != w.planW4(t, mlp0, mlp1).SHA256() || len(r.Components) != len(r.Plan.Groups) {
		t.Fatalf("result %+v", r)
	}
	if r.Assembly.AssemblyMS <= 0 || r.Resource.GPUAllocatedBytes == nil || r.Resource.HostRSSBytes == nil {
		t.Fatalf("assembly %+v resource %+v", r.Assembly, r.Resource)
	}
	packed := 0
	for _, c := range r.Components {
		if c.Policy == home.PolicyW4A16 {
			packed++
			if c.Digest == "" || c.Bytes == 0 {
				t.Errorf("packed component %+v lacks bytes or digest", c)
			}
		}
	}
	if packed != 2 || r.Cache.Misses != 2 || r.Cache.TransformedBytes == 0 {
		t.Fatalf("packed %d cache %+v", packed, r.Cache)
	}
}

func TestClosedSessionRefusesTrialsAndReleasesTheBackend(t *testing.T) {
	w := newWorld(t)
	s, f := openSession(t, w, Options{})
	if err := s.Close(context.Background()); err != nil || !f.closed {
		t.Fatalf("close: %v (closed %v)", err, f.closed)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.RunTrial(context.Background(), w.planW4(t, mlp0), &evaluator{}); err == nil {
		t.Fatal("a closed session ran a trial")
	}
}
