package trial

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

var carriedFiles = []string{"LICENSE", "chat_template.jinja", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py",
	"processor_config.json", "tokenizer.json", "tokenizer_config.json"}

// fixture materializes a source model under the real catalog ID (as the
// optimizer tests do) and returns the home and catalog entry.
func fixture(t *testing.T) (home.Home, home.ModelManifest) {
	t.Helper()
	files := map[string]string{}
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	m := home.ModelManifest{ID: setup.ClefFlash, Provider: home.ProviderClef, Repo: "test/clef", Revision: strings.Repeat("ef", 20), Files: files}
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, rel := range append([]string{"config.json", "generation_config.json", "model.safetensors"}, carriedFiles...) {
		if err := os.WriteFile(filepath.Join(dir, rel), []byte("fixture "+rel), 0o644); err != nil {
			t.Fatal(err)
		}
		d, _ := setup.FileSHA256(filepath.Join(dir, rel))
		files[rel] = d
	}
	if err := home.WriteJSON(filepath.Join(dir, "hachidori-model.json"), m); err != nil {
		t.Fatal(err)
	}
	old := setup.Models
	setup.Models = []home.ModelManifest{m}
	t.Cleanup(func() { setup.Models = old })
	return h, m
}

// layout is a four-block Clef-shaped graph: three linear-attention blocks and a
// full-attention block, each with an MLP, plus the always-preserved parts.
func layout() tuning.DeclaredLayout {
	l := tuning.DeclaredLayout{ModelType: "qwen3_5", TextModelType: "qwen3_5_text", Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes:   []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		CarriedFiles: []string{"joint_head.safetensors"}}
	l.LinearModules = []string{"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1"}
	for i := 0; i < 4; i++ {
		base := fmt.Sprintf("model.language_model.layers.%d.", i)
		l.LinearModules = append(l.LinearModules, base+"mlp.gate_proj")
		if i == 3 {
			l.LinearModules = append(l.LinearModules, base+"self_attn.q_proj")
			continue
		}
		l.LinearModules = append(l.LinearModules, base+"linear_attn.in_proj_a", base+"linear_attn.in_proj_b", base+"linear_attn.in_proj_qkv")
	}
	return l
}

type world struct {
	layout   tuning.DeclaredLayout
	h        home.Home
	model    home.ModelManifest
	source   home.VariantSource
	analysis tuning.Analysis
	base     tuning.Profile
}

func newWorld(t *testing.T) world { return newWorldWith(t, layout()) }

func newWorldWith(t *testing.T, l tuning.DeclaredLayout) world {
	t.Helper()
	h, m := fixture(t)
	a, err := tuning.Analyze(m, l)
	if err != nil {
		t.Fatal(err)
	}
	p, err := tuning.NewDefaultProfile(a, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	return world{layout: l, h: h, model: m, source: home.SourceOf(m), analysis: a, base: p}
}

// profile returns the profile with the given groups overridden to source
// precision (everything else AUTO, which is W4A16 for tunable groups).
func (w world) profile(t *testing.T, keep ...string) tuning.Profile {
	t.Helper()
	p := w.base
	if len(keep) > 0 {
		var err error
		if p, err = tuning.SetPolicy(p, w.analysis, keep, home.PolicySourcePrecision); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

func (w world) plan(t *testing.T, keep ...string) home.TuningPlan {
	t.Helper()
	c, err := tuning.Compile(w.profile(t, keep...), w.analysis)
	if err != nil {
		t.Fatal(err)
	}
	return *c.Plan
}

// profileW4 is the profile whose plan quantizes exactly the named groups.
func (w world) profileW4(t *testing.T, quantize ...string) tuning.Profile {
	t.Helper()
	tunable, err := tuning.Tunable(w.analysis)
	if err != nil {
		t.Fatal(err)
	}
	q := map[string]bool{}
	for _, id := range quantize {
		q[id] = true
	}
	var keep []string
	for _, id := range tunable {
		if !q[id] {
			keep = append(keep, id)
		}
	}
	return w.profile(t, keep...)
}

func (w world) ref(t *testing.T, p tuning.Profile) ProfileRef {
	t.Helper()
	return ProfileRef{ID: p.ID(), SHA256: p.SHA256(), AnalysisSHA256: w.analysis.SHA256(), CompilerVersion: p.CompilerVersion}
}

// fakeBackend is an in-memory resident: it keeps the representation of every
// module and the transformed components, can fail at any step, and counts what
// the session asks of it. It proves orchestration and invariants only.
type fakeBackend struct {
	mu sync.Mutex

	modules []Member
	groups  map[string][]string
	rep     map[string]string // module -> representation
	comps   map[string]Built
	backend string

	opened bool
	closed bool

	calls       map[string]int
	transformed []string
	released    []string
	applied     [][]string // groups of each Apply/Reconstruct call
	applyKinds  []string
	// failure injection: the Nth (1-based) call of an operation fails
	failTransform   int
	failApply       int
	failApplyLost   bool // the failing apply mutates one module first and reports the state lost
	failReconstruct bool
	failRelease     bool
	failState       bool
	validate        error
	badApply        bool // apply succeeds but silently leaves the model unchanged

	canonicalWrites int
	modelReplaced   int
}

func newFake(modules []string) *fakeBackend {
	f := &fakeBackend{rep: map[string]string{}, comps: map[string]Built{}, calls: map[string]int{}, backend: "fake-backend 1"}
	sort.Strings(modules)
	for _, m := range modules {
		f.modules = append(f.modules, Member{Module: m, Shape: []int64{256, 256}, DType: "bfloat16"})
		f.rep[m] = RepresentationDense
	}
	return f
}

func (f *fakeBackend) hit(op string) int {
	f.calls[op]++
	return f.calls[op]
}

func (f *fakeBackend) Open(_ context.Context, groups map[string][]string) (Opened, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hit("open")
	f.groups, f.opened = groups, true
	canonical := int64(len(f.modules)) * 256 * 256 * 2
	return Opened{Device: "cuda", DType: "bfloat16", Backend: f.backend, Modules: f.modules, CanonicalBytes: canonical}, nil
}

func (f *fakeBackend) Transform(_ context.Context, c ComponentIdentity) (Built, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if n := f.hit("transform"); f.failTransform == n {
		return Built{}, fmt.Errorf("injected transform failure %d", n)
	}
	b, err := ComponentBytes(c)
	if err != nil {
		return Built{}, err
	}
	built := Built{Bytes: b, Digest: "digest-" + c.ID()[:16]}
	f.comps[c.ID()] = built
	f.transformed = append(f.transformed, c.ID())
	return built, nil
}

func (f *fakeBackend) Release(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hit("release")
	if f.failRelease {
		return fmt.Errorf("injected release failure")
	}
	delete(f.comps, id)
	f.released = append(f.released, id)
	return nil
}

func (f *fakeBackend) Validate(_ context.Context, reps []Replacement) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hit("validate")
	return f.validate
}

func (f *fakeBackend) assemble(kind string, reps []Replacement) (Applied, error) {
	n := f.hit(kind)
	var groups []string
	for _, r := range reps {
		groups = append(groups, r.Group)
	}
	f.applied = append(f.applied, groups)
	f.applyKinds = append(f.applyKinds, kind)
	if kind == "apply" && f.failApply == n {
		if f.failApplyLost && len(reps) > 0 && len(reps[0].Modules) > 0 {
			f.rep[reps[0].Modules[0]] = representationOf(reps[0].Policy)
			return Applied{}, &StateLostError{Err: fmt.Errorf("injected: one module was replaced and the rest was not")}
		}
		return Applied{}, fmt.Errorf("injected apply failure %d", n)
	}
	if kind == "reconstruct" && f.failReconstruct {
		return Applied{}, fmt.Errorf("injected reconstruct failure")
	}
	var a Applied
	for _, r := range reps {
		if r.Component != "" {
			if _, ok := f.comps[r.Component]; !ok {
				return Applied{}, &IncompatibleError{Group: r.Group, Reason: "component " + r.Component[:12] + " is not resident"}
			}
		}
		for _, m := range r.Modules {
			want := representationOf(r.Policy)
			if f.badApply {
				continue
			}
			if f.rep[m] != want {
				f.rep[m] = want
				a.Modules++
				if want == RepresentationPacked {
					a.BytesToGPU += 256 * 256 / 2
					a.BytesReleased += 256 * 256 * 2
				} else {
					a.BytesToGPU += 256 * 256 * 2
					a.BytesReleased += 256 * 256 / 2
				}
			}
		}
	}
	return a, nil
}

func representationOf(policy string) string {
	if policy == home.PolicyW4A16 {
		return RepresentationPacked
	}
	return RepresentationDense
}

func (f *fakeBackend) Apply(_ context.Context, reps []Replacement) (Applied, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.assemble("apply", reps)
}

func (f *fakeBackend) Reconstruct(_ context.Context, reps []Replacement) (Applied, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.assemble("reconstruct", reps)
}

func (f *fakeBackend) State(context.Context) (Observed, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hit("state")
	if f.failState {
		return Observed{}, fmt.Errorf("injected state failure")
	}
	o := Observed{Policies: map[string]string{}}
	for g, mods := range f.groups {
		set := map[string]bool{}
		for _, m := range mods {
			set[f.rep[m]] = true
		}
		switch len(set) {
		case 0:
			o.Policies[g] = home.PolicySourcePrecision
		case 1:
			for r := range set {
				o.Policies[g] = PolicyOfRepresentation(r)
			}
		default:
			o.Policies[g] = "mixed"
		}
	}
	gpu, rss := int64(1<<20), int64(2<<20)
	o.GPUAllocated, o.HostRSS = &gpu, &rss
	return o, nil
}

func (f *fakeBackend) Close(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
	return nil
}

func (f *fakeBackend) packedModules() []string {
	var out []string
	for m, r := range f.rep {
		if r == RepresentationPacked {
			out = append(out, m)
		}
	}
	sort.Strings(out)
	return out
}

func (f *fakeBackend) resetCounts() {
	f.calls = map[string]int{}
	f.transformed, f.released, f.applied, f.applyKinds = nil, nil, nil, nil
}

// evaluator measures with a canned, valid trial measurement.
type evaluator struct {
	fn    func(context.Context, Assembled) (Measurement, error)
	calls int
	seen  []Assembled
}

func (e *evaluator) Evaluate(ctx context.Context, a Assembled) (Measurement, error) {
	e.calls++
	e.seen = append(e.seen, a)
	if e.fn != nil {
		return e.fn(ctx, a)
	}
	return measured(), nil
}

func measured() Measurement {
	acc := 0.9
	return Measurement{Mode: EvalModeTrial, Status: MeasurementMeasured, DatasetSHA256: strings64("d"), InputSHA256: strings64("i"), SentSHA256: strings64("s"),
		Questions: []QuestionBinding{{ID: "q", SHA256: strings64("q")}}, Labelled: true, Cases: 4, Warmup: 1, Passes: 1, Requests: 4, Succeeded: 4,
		Accuracy: &acc, ResidentStable: true, Served: ServedExecution{Model: "clef-flash", Execution: ExecutionTrial, Device: "cuda", DType: "torch.bfloat16"}}
}

func strings64(c string) string { return strings.Repeat(c[:1], 64) }

var fixedNow = time.Date(2026, 10, 3, 12, 0, 0, 0, time.UTC)

func moduleNames(l tuning.DeclaredLayout) []string { return append([]string(nil), l.LinearModules...) }

// openSession opens a session over a fresh fake backend.
func openSession(t *testing.T, w world, opt Options) (*Session, *fakeBackend) {
	t.Helper()
	f := newFake(moduleNames(w.layout))
	if opt.BudgetBytes == 0 {
		opt.BudgetBytes = 1 << 30
	}
	if opt.Clock == nil {
		ticks := 0
		opt.Clock = func() time.Time { ticks++; return fixedNow.Add(time.Duration(ticks) * 5 * time.Millisecond) }
	}
	s, err := Open(context.Background(), f, w.source, w.plan(t), opt)
	if err != nil {
		t.Fatal(err)
	}
	return s, f
}

// planW4 is the resolved plan that quantizes exactly the named tunable groups
// and keeps every other tunable group at its source precision.
func (w world) planW4(t *testing.T, quantize ...string) home.TuningPlan {
	t.Helper()
	tunable, err := tuning.Tunable(w.analysis)
	if err != nil {
		t.Fatal(err)
	}
	q := map[string]bool{}
	for _, id := range quantize {
		q[id] = true
	}
	var keep []string
	for _, id := range tunable {
		if !q[id] {
			keep = append(keep, id)
		}
	}
	return w.plan(t, keep...)
}

func tuningCompile(w world, p tuning.Profile) (home.TuningPlan, error) {
	c, err := tuning.Compile(p, w.analysis)
	if err != nil {
		return home.TuningPlan{}, err
	}
	return *c.Plan, nil
}
