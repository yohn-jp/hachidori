package optimize_test

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
)

var carried = []string{"LICENSE", "chat_template.jinja", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py",
	"processor_config.json", "tokenizer.json", "tokenizer_config.json"}

// source materializes a fixture System One model under the real catalog ID and
// returns the home and the catalog entry.
func source(t *testing.T) (home.Home, home.ModelManifest) {
	t.Helper()
	files := map[string]string{}
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	m := home.ModelManifest{ID: setup.ClefFlash, Provider: home.ProviderClef, Repo: "test/clef", Revision: strings.Repeat("ef", 20), Files: files}
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	os.MkdirAll(dir, 0o755)
	for _, rel := range append([]string{"config.json", "generation_config.json", "model.safetensors"}, carried...) {
		b := []byte("fixture " + rel)
		if err := os.WriteFile(filepath.Join(dir, rel), b, 0o644); err != nil {
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

type phases struct {
	mu       sync.Mutex
	seen     []setup.Phase
	progress []setup.Progress
}

func (p *phases) obs() *setup.Observer {
	return &setup.Observer{
		OnPhase:    func(ph setup.Phase) { p.mu.Lock(); p.seen = append(p.seen, ph); p.mu.Unlock() },
		OnProgress: func(pr setup.Progress) { p.mu.Lock(); p.progress = append(p.progress, pr); p.mu.Unlock() },
	}
}

func build(t *testing.T, h home.Home, r optimize.Runner, req optimize.Request, obs *setup.Observer) (optimize.Result, error) {
	t.Helper()
	if req.Model == "" {
		req.Model = setup.ClefFlash
	}
	if req.Recipe == "" {
		req.Recipe = optimize.RecipeClefFlashW4A16
	}
	return optimize.Build(context.Background(), h, req, optimize.Deps{Runner: r, OptimizerRuntime: "optimizer-test"}, io.Discard, obs)
}

func variantDirs(t *testing.T, h home.Home) []string {
	t.Helper()
	entries, _ := os.ReadDir(h.VariantsDir(setup.ClefFlash))
	var out []string
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

// The recipe's identity is deterministic and canonical: the same recipe always
// hashes the same and any semantic change yields a different digest.
func TestRecipeIdentityIsDeterministic(t *testing.T) {
	a, err := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if a.SHA256() != b.SHA256() || string(a.Canonical()) != string(b.Canonical()) {
		t.Fatal("recipe digest is not deterministic")
	}
	c := b
	c.Preserved = append([]home.PreservedModule(nil), b.Preserved...)
	c.Preserved[0].Reason += "."
	if c.SHA256() == a.SHA256() {
		t.Fatal("a changed preserved module did not change the recipe digest")
	}
	d := b
	d.Scheme = "W8A16"
	if d.SHA256() == a.SHA256() {
		t.Fatal("a changed scheme did not change the recipe digest")
	}
	if _, err := optimize.LookupRecipe(setup.ClefFlash, "nope"); err == nil {
		t.Fatal("an unknown recipe resolved")
	}
	if _, err := optimize.LookupRecipe(setup.DefaultModel, optimize.RecipeClefFlashW4A16); err == nil {
		t.Fatal("a recipe resolved for a model it was not written for")
	}
}

// The preserved set of the canonical recipe is derived from the pinned module
// graph: each named module group exists in the real Clef-Flash weight map.
func TestCanonicalRecipePreservesTheDecisionPath(t *testing.T) {
	r, _ := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	got := map[string]bool{}
	for _, p := range r.Preserved {
		got[p.Pattern] = true
		if p.Reason == "" || p.Precision != "bfloat16" {
			t.Errorf("preserved %q without reason/precision", p.Pattern)
		}
	}
	for _, want := range []string{"lm_head", `re:.*linear_attn\.in_proj_a$`, `re:.*linear_attn\.in_proj_b$`, `re:model\.visual.*`, "joint_head.safetensors"} {
		if !got[want] {
			t.Errorf("recipe does not preserve %s", want)
		}
	}
	if r.Engine != setup.OptimizerEngine || r.Scheme != "W4A16" || r.Algorithm != "rtn" {
		t.Errorf("recipe %+v", r)
	}
}

// A build runs through staging, verifies, publishes atomically, reports its
// real phases in order, records the engine and never changes the source.
func TestBuildPublishesVerifiedVariant(t *testing.T) {
	h, m := source(t)
	before := tree(t, h.Path("models"))
	var p phases
	res, err := build(t, h, &optimizetest.Runner{}, optimize.Request{}, p.obs())
	if err != nil {
		t.Fatal(err)
	}
	v := res.Variant
	if res.Existing || res.Dir != h.VariantDir(setup.ClefFlash, v.ID) {
		t.Fatalf("result %+v", res)
	}
	if v.Source != home.SourceOf(m) || v.Optimizer.Engine != "llmcompressor" || v.Optimizer.Version != setup.OptimizerEngineVersion ||
		v.Optimizer.Runtime != "optimizer-test" || v.Weights.Scheme != "W4A16" || v.Weights.Bits != 4 || v.Weights.GroupSize != 128 {
		t.Fatalf("manifest %+v", v)
	}
	read, err := home.ReadVariant(res.Dir)
	if err != nil || read.ID != v.ID {
		t.Fatalf("published manifest does not read back: %v", err)
	}
	if read.Files[setup.OptimizerReportFile] == "" || read.Files["config.json"] == "" || read.Files["joint_head.safetensors"] != m.Files["joint_head.safetensors"] {
		t.Fatalf("artifact digests %v", read.Files)
	}
	if err := setup.VerifyVariantArtifacts(h, m, read, nil); err != nil {
		t.Fatalf("a freshly built variant does not verify: %v", err)
	}
	if after := tree(t, h.Path("models")); !reflect.DeepEqual(before, after) {
		t.Fatal("building a variant changed the source model directory")
	}
	if names := variantDirs(t, h); len(names) != 1 || names[0] != v.ID {
		t.Fatalf("variant directory holds %v (staging must be gone)", names)
	}
	want := []setup.Phase{setup.PhaseModel, setup.PhasePreflight, setup.PhaseStarting, setup.PhaseLoadingSource, setup.PhaseResolving, setup.PhaseQuantizing,
		setup.PhaseSerializing, setup.PhaseVerifying, setup.PhasePublish}
	if !reflect.DeepEqual(p.seen, want) {
		t.Fatalf("phases %v, want %v", p.seen, want)
	}
	// Progress is only ever determinate where a total exists: byte progress of
	// hashing carries one; the optimizer's own steps are indeterminate.
	for _, pr := range p.progress {
		if pr.Total == 0 && pr.Determinate() {
			t.Errorf("fabricated progress: %+v", pr)
		}
	}
}

func tree(t *testing.T, dir string) map[string]string {
	t.Helper()
	m, err := setup.DigestTree(dir, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Failed, crashed, incomplete and cancelled builds leave no variant and no
// staging directory: nothing partial can become selectable.
func TestFailedBuildsLeaveNothingSelectable(t *testing.T) {
	for _, fail := range []string{"fatal", "crash", "incomplete", "noreport", "unpreserved", "wrong-scheme"} {
		t.Run(fail, func(t *testing.T) {
			h, _ := source(t)
			_, err := build(t, h, &optimizetest.Runner{Fail: fail}, optimize.Request{}, nil)
			if err == nil {
				t.Fatal("a failed build reported success")
			}
			if names := variantDirs(t, h); len(names) != 0 {
				t.Fatalf("a failed build left %v", names)
			}
			if entries, _ := os.ReadDir(h.Path("cache", "tmp")); len(entries) != 0 {
				t.Fatalf("recipe temp file left: %v", entries)
			}
			if got := setup.ListVariants(h, false, nil); len(got) != 0 {
				t.Fatalf("failed build is listed: %+v", got)
			}
		})
	}
}

// An interrupted build (cancelled while the optimizer holds partial output)
// is cleaned up, and so is a staging directory left by a killed process.
func TestInterruptedBuildIsCleanedUp(t *testing.T) {
	h, _ := source(t)
	ctx, cancel := context.WithCancel(context.Background())
	r := &optimizetest.Runner{OnStart: func(ctx context.Context, out string) error {
		if _, err := os.Stat(filepath.Join(out, "model.safetensors")); err != nil {
			t.Errorf("optimizer had not written partial output yet: %v", err)
		}
		cancel()
		<-ctx.Done()
		return ctx.Err()
	}}
	_, err := optimize.Build(ctx, h, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: r, OptimizerRuntime: "optimizer-test"}, io.Discard, nil)
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled build: %v", err)
	}
	if names := variantDirs(t, h); len(names) != 0 {
		t.Fatalf("an interrupted build left %v", names)
	}

	// A killed process leaves a staging directory (and, in the worst case, a
	// complete-looking manifest in it). It is never listed or selectable, and
	// the next build discards it instead of reusing it.
	stale := filepath.Join(h.VariantsDir(setup.ClefFlash), ".staging-0123456789abcdef")
	os.MkdirAll(stale, 0o755)
	os.WriteFile(filepath.Join(stale, home.VariantManifestFile), []byte("{}"), 0o644)
	if got := setup.ListVariants(h, false, nil); len(got) != 0 {
		t.Fatalf("staging directory is listed: %+v", got)
	}
	if _, _, err := setup.FindVariant(h, ".staging-0123456789abcdef"); err == nil {
		t.Fatal("a staging directory resolved as a variant")
	}
}

// The same source, recipe and engine contract yields the same variant
// identity; a second build is not run again, and a reproduction either matches
// byte for byte or surfaces the difference without publishing anything.
func TestReproductionIsSurfacedNotHidden(t *testing.T) {
	h, _ := source(t)
	r := &optimizetest.Runner{}
	first, err := build(t, h, r, optimize.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	again, err := build(t, h, r, optimize.Request{}, nil)
	if err != nil || !again.Existing || again.Variant.ID != first.Variant.ID || r.Calls != 1 {
		t.Fatalf("a repeat of the contract rebuilt (calls %d) or changed identity: %+v %v", r.Calls, again.Variant.ID, err)
	}
	same, err := build(t, h, r, optimize.Request{Reproduce: true}, nil)
	if err != nil || !same.Reproduced || same.Variant.ID != first.Variant.ID || r.Calls != 2 {
		t.Fatalf("an identical reproduction: %+v %v", same, err)
	}
	drift := &optimizetest.Runner{Salt: "nondeterministic"}
	res, err := build(t, h, drift, optimize.Request{Reproduce: true}, nil)
	var mm *optimize.MismatchError
	if !errors.As(err, &mm) || mm.Variant != first.Variant.ID || len(mm.Files) == 0 || !contains(mm.Files, "model.safetensors") {
		t.Fatalf("a differing reproduction was not surfaced: %v", err)
	}
	if res.Variant.ID != first.Variant.ID {
		t.Fatal("the mismatch result names a different variant")
	}
	if names := variantDirs(t, h); len(names) != 1 {
		t.Fatalf("a mismatching reproduction published %v", names)
	}
	// The artifact bytes are part of the identity: the drifted build is a different variant.
	ids := map[string]bool{first.Variant.ID: true}
	h2, _ := source(t)
	res2, err := build(t, h2, drift, optimize.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if ids[res2.Variant.ID] {
		t.Fatal("different artifact bytes produced the same variant identity")
	}
	if res2.Variant.BuildID != first.Variant.BuildID {
		t.Fatal("the build contract identity changed with artifact bytes")
	}
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// A build is refused, not recorded under another engine or scheme.
func TestBuildRefusesSubstitutions(t *testing.T) {
	h, _ := source(t)
	if _, err := build(t, h, &optimizetest.Runner{Version: "9.9.9"}, optimize.Request{}, nil); err == nil || !strings.Contains(err.Error(), "pinned") {
		t.Fatalf("another engine version was accepted: %v", err)
	}
	if _, err := build(t, h, &optimizetest.Runner{Engine: "bitsandbytes"}, optimize.Request{}, nil); err == nil {
		t.Fatal("another engine was accepted")
	}
	if _, err := build(t, h, &optimizetest.Runner{Fail: "wrong-scheme"}, optimize.Request{}, nil); err == nil || !strings.Contains(err.Error(), "scheme") {
		t.Fatalf("an 8-bit output was accepted as W4A16: %v", err)
	}
	if _, err := build(t, h, &optimizetest.Runner{Fail: "unpreserved"}, optimize.Request{}, nil); err == nil {
		t.Fatal("an output that does not preserve the declared modules was accepted")
	}
	if names := variantDirs(t, h); len(names) != 0 {
		t.Fatalf("refused builds left %v", names)
	}
}

// The optimizer receives only a pinned catalog identity and a named recipe:
// no other model, no free-form recipe, a materialized and intact source.
func TestBuildAcceptsOnlyPinnedSources(t *testing.T) {
	h, m := source(t)
	r := &optimizetest.Runner{}
	for _, req := range []optimize.Request{
		{Model: "some/other-model", Recipe: optimize.RecipeClefFlashW4A16},
		{Model: setup.ClefFlash, Recipe: "free-form"},
	} {
		if _, err := optimize.Build(context.Background(), h, req, optimize.Deps{Runner: r}, io.Discard, nil); err == nil {
			t.Fatalf("%+v accepted", req)
		}
	}
	// A tampered source is refused before the optimizer runs.
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("tampered"), 0o644)
	if _, err := build(t, h, r, optimize.Request{}, nil); err == nil || r.Calls != 0 {
		t.Fatalf("a tampered source was optimized (calls %d): %v", r.Calls, err)
	}
	// A model that is not materialized is refused.
	os.RemoveAll(dir)
	if _, err := build(t, h, r, optimize.Request{}, nil); err == nil || r.Calls != 0 {
		t.Fatalf("a missing source was optimized: %v", err)
	}
	// Models without variants cannot be optimized.
	setup.Models = append(setup.Models, home.ModelManifest{ID: "laya-x", Provider: "laya", Repo: "t/l", Revision: strings.Repeat("a", 40), Files: map[string]string{"a": "b"}})
	if _, err := optimize.Build(context.Background(), h, optimize.Request{Model: "laya-x", Recipe: optimize.RecipeClefFlashW4A16}, optimize.Deps{Runner: r}, io.Discard, nil); err == nil {
		t.Fatal("a model without variants was optimized")
	}
}
