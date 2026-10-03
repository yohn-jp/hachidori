package tuning

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// pinnedClefLayout reproduces the module hierarchy of the pinned Clef-Flash
// revision (Cloudflare/clef-flash at the catalog revision): 32 text blocks, a
// full-attention block after every three linear-attention blocks (24 + 8), the
// Linear modules of its safetensors index (248 language, 110 vision, lm_head =
// 359) and the carried joint head. It was read from that revision's pinned
// config.json and model.safetensors.index.json.
func pinnedClefLayout() DeclaredLayout {
	l := DeclaredLayout{ModelType: "qwen3_5", TextModelType: "qwen3_5_text", Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		CarriedFiles: []string{"joint_head.safetensors"}}
	l.LinearModules = append(l.LinearModules, "lm_head")
	for i := 0; i < 32; i++ {
		base := fmt.Sprintf("model.language_model.layers.%d.", i)
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} {
			l.LinearModules = append(l.LinearModules, base+m)
		}
		if (i+1)%4 == 0 {
			l.LayerTypes = append(l.LayerTypes, "full_attention")
			for _, m := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
				l.LinearModules = append(l.LinearModules, base+"self_attn."+m)
			}
			continue
		}
		l.LayerTypes = append(l.LayerTypes, "linear_attention")
		for _, m := range []string{"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"} {
			l.LinearModules = append(l.LinearModules, base+"linear_attn."+m)
		}
	}
	for i := 0; i < 110; i++ {
		l.LinearModules = append(l.LinearModules, fmt.Sprintf("model.visual.blocks.%d.attn.qkv", i))
	}
	return l
}

func layerwiseAnalysis(t *testing.T, layout DeclaredLayout) Analysis {
	t.Helper()
	a, err := Analyze(clefSource(t), layout)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func autoProfile(t *testing.T, a Analysis) Profile {
	t.Helper()
	p, err := NewDefaultProfile(a, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func setPolicy(t *testing.T, p Profile, a Analysis, policy string, ids ...string) Profile {
	t.Helper()
	next, err := SetPolicy(p, a, ids, policy)
	if err != nil {
		t.Fatal(err)
	}
	return next
}

func planGroup(t *testing.T, c Compilation, id string) home.TuningGroupPlan {
	t.Helper()
	for _, g := range c.Plan.Groups {
		if g.ID == id {
			return g
		}
	}
	t.Fatalf("plan has no group %s", id)
	return home.TuningGroupPlan{}
}

func TestGroupsAreStableAndDerivedFromTheRealModelStructure(t *testing.T) {
	layout := pinnedClefLayout()
	if len(layout.LinearModules) != 1+32*3+24*5+8*4+110 {
		t.Fatalf("fixture has %d modules", len(layout.LinearModules))
	}
	a := layerwiseAnalysis(t, layout)
	if a.Schema != AnalysisSchema || a.AnalyzerVersion != ClefAnalyzerVersion || len(a.Regions) != len(requiredRegions) {
		t.Fatalf("analysis %+v", a.Schema)
	}
	// 24 linear blocks x (projections, decay gate, beta gate, mlp) + 8 full
	// blocks x (attention, mlp) + output embeddings, vision tower, joint head.
	if want := 24*4 + 8*2 + 3; len(a.Groups) != want {
		t.Fatalf("%d groups, want %d", len(a.Groups), want)
	}
	for id, modules := range map[string]int{"block.00.mlp": 3, "block.00.linear-attn": 3, "block.00.linear-attn.decay-gate": 1, "block.00.linear-attn.beta-gate": 1,
		"block.03.full-attn": 4, "block.03.mlp": 3, "block.31.full-attn": 4, "output-embeddings": 1, "vision-tower": 110} {
		g, ok := a.GroupByID(id)
		if !ok || len(g.Modules) != modules {
			t.Errorf("group %s: %+v", id, g)
		}
	}
	if g, _ := a.GroupByID("block.03.full-attn"); g.LayerType != "full_attention" || g.Layer != 3 || g.Region != RegionFullAttention {
		t.Errorf("full-attention group %+v", g)
	}
	if _, ok := a.GroupByID("block.03.linear-attn"); ok {
		t.Error("a full-attention block has a linear-attention group")
	}
	if g, _ := a.GroupByID("joint-schema-head"); len(g.Files) != 1 || len(g.Modules) != 0 || g.Layer != -1 {
		t.Errorf("joint head group %+v", g)
	}

	// Identifiers, order and digest do not depend on the order the layout
	// lists its modules in.
	shuffled := pinnedClefLayout()
	for i, j := 0, len(shuffled.LinearModules)-1; i < j; i, j = i+1, j-1 {
		shuffled.LinearModules[i], shuffled.LinearModules[j] = shuffled.LinearModules[j], shuffled.LinearModules[i]
	}
	b := layerwiseAnalysis(t, shuffled)
	if !bytes.Equal(a.Canonical(), b.Canonical()) || a.SHA256() != b.SHA256() {
		t.Fatal("equivalent layouts produced different groups")
	}
	for i := 1; i < len(a.Groups); i++ {
		if a.Groups[i-1].ID >= a.Groups[i].ID {
			t.Fatalf("groups are not in canonical order at %s", a.Groups[i].ID)
		}
	}

	// A structurally different model is a different analysis.
	changed := pinnedClefLayout()
	changed.LayerTypes[3] = "linear_attention"
	if _, err := Analyze(clefSource(t), changed); err == nil {
		t.Error("a full-attention block declared as linear attention was analyzed")
	}
	extra := pinnedClefLayout()
	extra.LinearModules = append(extra.LinearModules, "model.language_model.layers.0.mlp.extra_proj")
	if _, err := Analyze(clefSource(t), extra); err == nil {
		t.Error("an unknown projection was analyzed")
	}
	fewer := pinnedClefLayout()
	for i, m := range fewer.LinearModules {
		if m == "model.language_model.layers.5.mlp.up_proj" {
			fewer.LinearModules = append(fewer.LinearModules[:i], fewer.LinearModules[i+1:]...)
			break
		}
	}
	if c := layerwiseAnalysis(t, fewer); c.SHA256() == a.SHA256() {
		t.Error("a layout missing one projection has the same analysis identity")
	}
	noHead := pinnedClefLayout()
	noHead.LinearModules = noHead.LinearModules[1:]
	if _, err := Analyze(clefSource(t), noHead); err == nil {
		t.Error("a layout without the output embeddings was analyzed")
	}
}

func TestProfileAndAnalysisRoundTripWithoutLosingPerGroupPolicy(t *testing.T) {
	a := layerwiseAnalysis(t, pinnedClefLayout())
	p := autoProfile(t, a)
	p = setPolicy(t, p, a, home.PolicySourcePrecision, "block.07.mlp", "block.31.full-attn")
	p = setPolicy(t, p, a, home.PolicyW4A16, "block.00.mlp")

	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var back Profile
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p, back) || back.ID() != p.ID() || !bytes.Equal(back.Canonical(), p.Canonical()) {
		t.Fatal("a profile did not survive serialization exactly")
	}
	rawA, _ := json.Marshal(a)
	var backA Analysis
	if err := json.Unmarshal(rawA, &backA); err != nil || backA.SHA256() != a.SHA256() || !reflect.DeepEqual(a, backA) {
		t.Fatalf("an analysis did not survive serialization: %v", err)
	}
	if got := back.Groups["block.07.mlp"]; got != (GroupChoice{Mode: GroupOverride, Policy: home.PolicySourcePrecision}) {
		t.Errorf("block.07.mlp %+v", got)
	}
	if got := back.Groups["block.00.mlp"]; got != (GroupChoice{Mode: GroupOverride, Policy: home.PolicyW4A16}) {
		t.Errorf("block.00.mlp %+v", got)
	}
	if got := back.Groups["block.01.mlp"]; got != (GroupChoice{Mode: GroupAuto}) {
		t.Errorf("an untouched group is %+v", got)
	}
	if len(back.Groups) != len(a.Groups) || back.Schema != ProfileSchema || strings.Contains(string(raw), `"preservation"`) {
		t.Errorf("a layer-wise profile is not complete or carries legacy fields: %s", raw[:200])
	}

	// persisted and reloaded from the home
	h := home.Home{Root: t.TempDir()}
	if err := SaveProfile(h, p, a); err != nil {
		t.Fatal(err)
	}
	loaded, loadedA, err := LoadProfile(h, p.ID())
	if err != nil || !reflect.DeepEqual(loaded, p) || loadedA.SHA256() != a.SHA256() {
		t.Fatalf("reload: %v", err)
	}
	tampered := loaded
	tampered.Groups = map[string]GroupChoice{}
	for id, c := range loaded.Groups {
		tampered.Groups[id] = c
	}
	tampered.Groups["block.07.mlp"] = GroupChoice{Mode: GroupAuto}
	if err := home.WriteJSON(profilePath(h, p.ID()), tampered); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadProfile(h, p.ID()); err == nil {
		t.Fatal("a stored per-group policy changed without changing the profile identity")
	}
}

func TestSchemaVersionsAreExplicitAndNeverMixed(t *testing.T) {
	a := layerwiseAnalysis(t, clefLayout())
	p := autoProfile(t, a)
	if p.Schema != "hachidori.tuning-profile/2" || a.Schema != "hachidori.tuning-analysis/2" || p.CompilerVersion != RecipeCompilerVersion {
		t.Fatalf("schemas %s %s %s", p.Schema, a.Schema, p.CompilerVersion)
	}
	cases := map[string]func(*Profile){
		"unknown schema":            func(p *Profile) { p.Schema = "hachidori.tuning-profile/3" },
		"legacy schema over groups": func(p *Profile) { p.Schema = ProfileSchemaV1 },
		"legacy analyzer":           func(p *Profile) { p.AnalyzerVersion = ClefAnalyzerVersionV1 },
		"legacy compiler":           func(p *Profile) { p.CompilerVersion = RecipeCompilerVersionV1 },
		"legacy region preservation": func(p *Profile) {
			p.Preservation = map[string]PreservationChoice{RegionVision: {Mode: PreservationAuto}}
		},
		"missing group":                func(p *Profile) { delete(p.Groups, "block.00.mlp") },
		"unknown group":                func(p *Profile) { p.Groups["block.99.mlp"] = GroupChoice{Mode: GroupAuto} },
		"unknown mode":                 func(p *Profile) { p.Groups["block.00.mlp"] = GroupChoice{Mode: "manual"} },
		"AUTO naming a policy":         func(p *Profile) { p.Groups["block.00.mlp"] = GroupChoice{Mode: GroupAuto, Policy: home.PolicyW4A16} },
		"no objective":                 func(p *Profile) { p.Objective = " " },
		"malformed migration record":   func(p *Profile) { p.MigratedFrom = &Migration{FromProfileID: "x"} },
		"bound to another recipe":      func(p *Profile) { p.RecipeSHA256 = strings.Repeat("0", 64) },
		"bound to another recipe name": func(p *Profile) { p.Recipe = "other-recipe" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			q := p
			q.Groups = map[string]GroupChoice{}
			for id, c := range p.Groups {
				q.Groups[id] = c
			}
			mutate(&q)
			if _, err := Compile(q, a); err == nil {
				t.Fatal("an invalid profile compiled")
			}
		})
	}
	// A layer-wise profile cannot be read against a legacy analysis and the
	// reverse.
	legacyA := clefAnalysis(t)
	if _, err := Compile(p, legacyA); err == nil {
		t.Error("a layer-wise profile compiled against a legacy analysis")
	}
	if _, err := Compile(defaultProfile(t, legacyA), a); err == nil {
		t.Error("a legacy profile compiled against a layer-wise analysis")
	}
	if _, err := NewDefaultProfile(legacyA, "balanced"); err == nil {
		t.Error("a layer-wise profile was created from a legacy analysis")
	}
	if _, err := NewLegacyProfile(a, "balanced"); err == nil {
		t.Error("a legacy profile was created from a layer-wise analysis")
	}
}

func TestProfileIsBoundToTheExactSourceRevisionAndModelStructure(t *testing.T) {
	a := layerwiseAnalysis(t, clefLayout())
	p := autoProfile(t, a)
	other := p
	other.Source.Revision = strings.Repeat("0", len(p.Source.Revision))
	if _, err := Compile(other, a); err == nil {
		t.Error("a profile bound to another revision compiled")
	}
	other = p
	other.Source.FilesSHA256 = strings.Repeat("0", 64)
	if _, err := Compile(other, a); err == nil {
		t.Error("a profile bound to other source files compiled")
	}
	// The same source with a structurally different model (one more block's
	// worth of modules) is another analysis: the profile does not apply.
	layout := clefLayout()
	layout.LinearModules = append(layout.LinearModules, "model.language_model.layers.1.mlp.gate_proj")
	b := layerwiseAnalysis(t, layout)
	if b.SHA256() == a.SHA256() {
		t.Fatal("structurally different models share an analysis identity")
	}
	if _, err := Compile(p, b); err == nil {
		t.Error("a saved profile silently applied to a structurally different model")
	}
	h := home.Home{Root: t.TempDir()}
	if err := SaveProfile(h, p, a); err != nil {
		t.Fatal(err)
	}
	if err := SaveProfile(h, p, b); err == nil {
		t.Error("a profile was stored against a different structure")
	}
}

func TestAutoResolvesToAnExplicitDeterministicPolicyAndKeepsTheCanonicalRecipe(t *testing.T) {
	a := layerwiseAnalysis(t, pinnedClefLayout())
	p := autoProfile(t, a)
	c, err := Compile(p, a)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(c.Recipe.Canonical(), baseline.Canonical()) {
		t.Fatal("an all-AUTO profile changed the canonical recipe")
	}
	if c.Plan == nil || c.Plan.AutoPolicy != AutoPolicyVersion || c.Plan.RecipeSHA256 != c.Recipe.SHA256() || len(c.Plan.Groups) != len(a.Groups) {
		t.Fatalf("plan %+v", c.Plan)
	}
	for _, g := range c.Plan.Groups {
		if g.Selection != home.SelectionAuto || g.Requested != home.PolicyAuto || !home.ValidPolicy(g.Effective) || g.Basis == "" {
			t.Errorf("AUTO group %+v is not an explicit recorded policy", g)
		}
	}
	// What AUTO selected: the backbone projections are quantized; the gates,
	// output embeddings, vision tower and joint head are preserved.
	for id, want := range map[string]string{"block.05.mlp": home.PolicyW4A16, "block.05.linear-attn": home.PolicyW4A16, "block.07.full-attn": home.PolicyW4A16,
		"block.05.linear-attn.decay-gate": home.PolicySourcePrecision, "block.05.linear-attn.beta-gate": home.PolicySourcePrecision,
		"output-embeddings": home.PolicySourcePrecision, "vision-tower": home.PolicySourcePrecision, "joint-schema-head": home.PolicySourcePrecision} {
		g := planGroup(t, c, id)
		if g.Effective != want || g.Selection != home.SelectionAuto {
			t.Errorf("AUTO resolved %s to %s (%s), want %s", id, g.Effective, g.Selection, want)
		}
		if g.Required != (want == home.PolicySourcePrecision) {
			t.Errorf("%s required = %v", id, g.Required)
		}
	}
	if g := planGroup(t, c, "block.05.mlp"); !strings.Contains(g.Basis, AutoPolicyVersion) {
		t.Errorf("AUTO does not record the rules that chose: %q", g.Basis)
	}
	if g := planGroup(t, c, "output-embeddings"); !strings.Contains(g.Basis, "canonical optimizer contract") {
		t.Errorf("a required group does not record why: %q", g.Basis)
	}
	again, err := Compile(autoProfile(t, a), a)
	if err != nil || again.Plan.SHA256() != c.Plan.SHA256() || !bytes.Equal(again.Recipe.Canonical(), c.Recipe.Canonical()) {
		t.Fatal("the same inputs did not resolve identically")
	}
	// AUTO is distinguishable from an override in the evidence
	o := setPolicy(t, p, a, home.PolicyW4A16, "block.05.mlp")
	oc, err := Compile(o, a)
	if err != nil {
		t.Fatal(err)
	}
	if g := planGroup(t, oc, "block.05.mlp"); g.Selection != home.SelectionOverridden || g.Requested != home.PolicyW4A16 || g.Effective != home.PolicyW4A16 {
		t.Errorf("explicit override equal to AUTO's choice: %+v", g)
	}
}

func TestOverridePrecedenceAndPreservationRules(t *testing.T) {
	a := layerwiseAnalysis(t, pinnedClefLayout())
	p := autoProfile(t, a)
	c, err := Compile(setPolicy(t, p, a, home.PolicySourcePrecision, "block.07.mlp", "block.08.mlp"), a)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"block.07.mlp", "block.08.mlp"} {
		g := planGroup(t, c, id)
		if g.Selection != home.SelectionOverridden || g.Effective != home.PolicySourcePrecision || g.Required {
			t.Errorf("%s: %+v", id, g)
		}
	}
	if g := planGroup(t, c, "block.09.mlp"); g.Effective != home.PolicyW4A16 || g.Selection != home.SelectionAuto {
		t.Errorf("an unrelated group changed: %+v", g)
	}
	// The override reaches the backend as exact modules of that group only, after
	// every canonical preserved entry.
	baseline, _ := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if len(c.Recipe.Preserved) != len(baseline.Preserved)+2 || !reflect.DeepEqual(c.Recipe.Preserved[:len(baseline.Preserved)], baseline.Preserved) {
		t.Fatalf("the canonical preservation changed: %+v", c.Recipe.Preserved)
	}
	added := c.Recipe.Preserved[len(baseline.Preserved)]
	if added.Scope != home.ScopeBackbone || added.Precision != PreservedPrecision || !strings.Contains(added.Reason, "block.07.mlp") ||
		!strings.HasPrefix(added.Pattern, "re:^(?:") || !strings.Contains(added.Pattern, `layers\.7\.mlp\.down_proj`) || strings.Contains(added.Pattern, `layers\.8\.`) {
		t.Errorf("generated entry %+v", added)
	}
	for _, m := range planGroup(t, c, "block.07.mlp").Modules {
		if !preservedBy(c.Recipe, m, home.ScopeBackbone) {
			t.Errorf("module %s of an overridden group is not preserved by the recipe", m)
		}
	}
	if preservedBy(c.Recipe, "model.language_model.layers.9.mlp.down_proj", home.ScopeBackbone) {
		t.Error("a module of another group is preserved")
	}
	// An override back to W4A16 or AUTO removes the preservation.
	back, err := Compile(setPolicy(t, setPolicy(t, p, a, home.PolicySourcePrecision, "block.07.mlp"), a, home.PolicyAuto, "block.07.mlp"), a)
	if err != nil || !bytes.Equal(back.Recipe.Canonical(), baseline.Canonical()) {
		t.Fatalf("reset to AUTO did not restore the canonical recipe: %v", err)
	}

	// Preservation rules: groups the canonical contract requires cannot be
	// transformed, whichever policy is named, and the carried head is a file.
	for _, id := range []string{"output-embeddings", "vision-tower", "joint-schema-head", "block.00.linear-attn.decay-gate", "block.00.linear-attn.beta-gate"} {
		for _, policy := range []string{home.PolicyW4A16, home.PolicySourcePrecision} {
			if _, err := SetPolicy(p, a, []string{id}, policy); err == nil || !strings.Contains(err.Error(), "cannot be overridden") {
				t.Errorf("override of required group %s to %s: %v", id, policy, err)
			}
		}
	}
	tunable, err := Tunable(a)
	if err != nil {
		t.Fatal(err)
	}
	if want := 24*2 + 8*2; len(tunable) != want {
		t.Errorf("%d tunable groups, want %d", len(tunable), want)
	}
	for _, id := range tunable {
		g, _ := a.GroupByID(id)
		if g.Region == RegionLinearAttentionDecay || g.Region == RegionLinearAttentionBeta || g.Region == RegionOutputEmbeddings || g.Region == RegionVision || g.Region == RegionJointSchemaHead {
			t.Errorf("a required group %s is listed as tunable", id)
		}
	}
}

func TestUnsupportedPoliciesAndCombinationsAreRejectedBeforeAnyBuild(t *testing.T) {
	a := layerwiseAnalysis(t, clefLayout())
	p := autoProfile(t, a)
	for _, policy := range []string{"w8a16", "w3a16", "w2a16", "prune", "int4", "", "W4A16", "source_precision"} {
		if _, err := SetPolicy(p, a, []string{"block.00.mlp"}, policy); err == nil || !strings.Contains(err.Error(), "unsupported policy") {
			t.Errorf("policy %q: %v", policy, err)
		}
	}
	if _, err := SetPolicy(p, a, []string{"model.language_model.layers.0.mlp.gate_proj"}, home.PolicyW4A16); err == nil {
		t.Error("a tensor name was accepted as a group")
	}
	if _, err := SetPolicy(Profile{Schema: ProfileSchemaV1}, a, nil, home.PolicyW4A16); err == nil {
		t.Error("a legacy profile took a group policy")
	}
	// A candidate that would quantize nothing is not the declared W4A16 variant.
	tunable, err := Tunable(a)
	if err != nil {
		t.Fatal(err)
	}
	all := setPolicy(t, p, a, home.PolicySourcePrecision, tunable...)
	if _, err := Compile(all, a); err == nil || !strings.Contains(err.Error(), "nothing would be W4A16") {
		t.Errorf("a profile that preserves everything compiled: %v", err)
	}
	if err := SaveProfile(home.Home{Root: t.TempDir()}, all, a); err == nil {
		t.Error("a profile that preserves everything was stored")
	}
	// The ordered policy set is exactly what the optimizer executes.
	var ids []string
	for i, pol := range Policies() {
		ids = append(ids, pol.ID)
		if pol.Rank != i || pol.Transformation == "" {
			t.Errorf("policy %+v is not ordered and described", pol)
		}
	}
	if !reflect.DeepEqual(ids, []string{home.PolicySourcePrecision, home.PolicyW4A16}) {
		t.Errorf("policies %v", ids)
	}
}

func TestBulkSelectionOverrideAndResetToAuto(t *testing.T) {
	a := layerwiseAnalysis(t, pinnedClefLayout())
	p := autoProfile(t, a)
	sel, err := Selection{Region: RegionFeedForward, From: 4, To: 7}.Select(a)
	if err != nil || !reflect.DeepEqual(sel, []string{"block.04.mlp", "block.05.mlp", "block.06.mlp", "block.07.mlp"}) {
		t.Fatalf("range selection %v %v", sel, err)
	}
	all, err := Selection{Region: RegionFullAttention, From: -1, To: -1}.Select(a)
	if err != nil || len(all) != 8 || all[0] != "block.03.full-attn" || all[7] != "block.31.full-attn" {
		t.Fatalf("family selection %v %v", all, err)
	}
	from, err := Selection{From: 30, To: -1}.Select(a)
	if err != nil || !reflect.DeepEqual(from, []string{"block.30.linear-attn", "block.30.mlp", "block.31.full-attn", "block.31.mlp"}) {
		t.Fatalf("open range across families %v %v", from, err)
	}
	for name, s := range map[string]Selection{
		"required family":  {Region: RegionLinearAttentionDecay, From: -1, To: -1},
		"vision":           {Region: RegionVision, From: -1, To: -1},
		"beyond the model": {Region: RegionFeedForward, From: 40, To: 50},
		"unknown family":   {Region: "unknown", From: -1, To: -1},
		"block-less range": {Region: RegionOutputEmbeddings, From: 0, To: 3},
	} {
		if ids, err := s.Select(a); err == nil {
			t.Errorf("%s selected %v: an empty selection must be an error, not a silent no-op", name, ids)
		}
	}

	bulk := setPolicy(t, p, a, home.PolicySourcePrecision, sel...)
	c, err := Compile(bulk, a)
	if err != nil {
		t.Fatal(err)
	}
	overridden := 0
	for _, g := range c.Plan.Groups {
		if g.Selection == home.SelectionOverridden {
			overridden++
			if !contains(sel, g.ID) || g.Effective != home.PolicySourcePrecision {
				t.Errorf("unexpected override %+v", g)
			}
		}
	}
	if overridden != 4 {
		t.Errorf("%d overrides, want the 4 selected", overridden)
	}
	// SetPolicy never mutates its input.
	if len(p.Groups) == 0 || p.Groups["block.04.mlp"].Mode != GroupAuto || bulk.ID() == p.ID() {
		t.Error("a bulk edit changed the profile it was derived from")
	}
	// Reset one group, then the rest: back to the exact original profile.
	partial := setPolicy(t, bulk, a, home.PolicyAuto, "block.04.mlp")
	if partial.Groups["block.04.mlp"].Mode != GroupAuto || partial.ID() == bulk.ID() || partial.ID() == p.ID() {
		t.Error("resetting one group did not produce a distinct profile")
	}
	reset := setPolicy(t, partial, a, home.PolicyAuto, sel...)
	if reset.ID() != p.ID() {
		t.Error("resetting every override did not restore the all-AUTO profile identity")
	}
}

func TestMeaningfulProfileChangesChangeIdentityAndIdenticalProfilesDoNot(t *testing.T) {
	a := layerwiseAnalysis(t, clefLayout())
	base := autoProfile(t, a)
	same := autoProfile(t, a)
	cb, _ := Compile(base, a)
	cs, _ := Compile(same, a)
	if base.ID() != same.ID() || cb.Plan.SHA256() != cs.Plan.SHA256() || cb.Recipe.SHA256() != cs.Recipe.SHA256() {
		t.Fatal("identical profiles have different identities")
	}
	identity := func(p Profile) (profile, plan, recipe string) {
		c, err := Compile(p, a)
		if err != nil {
			t.Fatal(err)
		}
		return p.ID(), c.Plan.SHA256(), c.Recipe.SHA256()
	}
	bp, bpl, br := identity(base)

	preserve := setPolicy(t, base, a, home.PolicySourcePrecision, "block.00.mlp")
	pp, ppl, pr := identity(preserve)
	if pp == bp || ppl == bpl || pr == br {
		t.Error("preserving a group did not change profile, plan and recipe identity")
	}
	// An override that restates AUTO leaves the recipe but is another effective
	// profile.
	restate := setPolicy(t, base, a, home.PolicyW4A16, "block.00.mlp")
	rp, rpl, rr := identity(restate)
	if rp == bp || rpl == bpl || rr != br {
		t.Errorf("an override equal to AUTO: profile changed %v, plan changed %v, recipe changed %v", rp != bp, rpl != bpl, rr != br)
	}
	// Different groups preserved are different identities.
	other := setPolicy(t, base, a, home.PolicySourcePrecision, "block.03.full-attn")
	op, opl, or := identity(other)
	if op == pp || opl == ppl || or == pr {
		t.Error("preserving a different group has the same identity")
	}
	// The objective is recorded intent and part of the profile identity.
	obj := base
	obj.Objective = "maximum-fidelity"
	if obj.ID() == base.ID() {
		t.Error("the objective is not part of the profile identity")
	}
	// The AUTO rules version is part of the plan: a changed resolution changes
	// the effective identity without touching the profile.
	c, _ := Compile(base, a)
	moved := *c.Plan
	moved.AutoPolicy = "clef-auto/2"
	if moved.SHA256() == c.Plan.SHA256() {
		t.Error("the AUTO policy version is not part of the effective-plan identity")
	}
	prov := Provenance(preserve, a, mustCompile(t, preserve, a))
	if prov.Schema != home.TuningProvenanceSchema || prov.PlanSHA256 != ppl || prov.ProfileID != pp || prov.CompilerVersion != RecipeCompilerVersion {
		t.Errorf("provenance %+v", prov)
	}
}

func mustCompile(t *testing.T, p Profile, a Analysis) Compilation {
	t.Helper()
	c, err := Compile(p, a)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestCompilerEvidenceReportsRegionModeForLayerwiseProfiles(t *testing.T) {
	a := layerwiseAnalysis(t, pinnedClefLayout())
	p := autoProfile(t, a)
	mode := func(p Profile, region string) PreservationMode {
		return regionMapping(t, mustCompile(t, p, a).Evidence, region).Mode
	}
	if mode(p, RegionFeedForward) != PreservationAuto {
		t.Error("an untouched region is not auto")
	}
	if got := mode(setPolicy(t, p, a, home.PolicySourcePrecision, "block.00.mlp"), RegionFeedForward); got != PreservationMixed {
		t.Errorf("one overridden group of many: %s", got)
	}
	sel, _ := Selection{Region: RegionFeedForward, From: -1, To: -1}.Select(a)
	if got := mode(setPolicy(t, p, a, home.PolicySourcePrecision, sel...), RegionFeedForward); got != PreservationOverride {
		t.Errorf("every group overridden: %s", got)
	}
	if got := regionMapping(t, mustCompile(t, setPolicy(t, p, a, home.PolicySourcePrecision, sel...), a).Evidence, RegionFeedForward); !got.Preserved || len(got.Modules) != 32*3 {
		t.Errorf("region mapping %+v", got)
	}
}

func TestLegacyProfileMigratesDeterministicallyAndEquivalently(t *testing.T) {
	layout := pinnedClefLayout()
	source := clefSource(t)
	legacyA, err := AnalyzeLegacy(source, layout)
	if err != nil {
		t.Fatal(err)
	}
	current := layerwiseAnalysis(t, layout)
	legacy, err := NewLegacyProfile(legacyA, "maximum-fidelity")
	if err != nil {
		t.Fatal(err)
	}
	legacy.Preservation[RegionFullAttention] = PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}
	legacy.Preservation[RegionOutputEmbeddings] = PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}

	// The legacy profile stays valid and keeps its exact identity and recipe.
	before := legacy.ID()
	legacyC, err := Compile(legacy, legacyA)
	if err != nil {
		t.Fatalf("the legacy profile no longer compiles: %v", err)
	}
	h := home.Home{Root: t.TempDir()}
	if err := SaveProfile(h, legacy, legacyA); err != nil {
		t.Fatal(err)
	}
	if got, _, err := LoadProfile(h, before); err != nil || got.ID() != before {
		t.Fatalf("a stored legacy profile no longer loads: %v", err)
	}

	m, err := MigrateProfile(legacy, legacyA, current)
	if err != nil {
		t.Fatal(err)
	}
	again, err := MigrateProfile(legacy, legacyA, current)
	if err != nil || again.ID() != m.ID() {
		t.Fatalf("migration is not deterministic: %v", err)
	}
	if legacy.ID() != before || m.ID() == before || m.Schema != ProfileSchema || m.Objective != legacy.Objective {
		t.Fatalf("migration changed the legacy profile or lost its intent: %+v", m)
	}
	if m.MigratedFrom == nil || m.MigratedFrom.FromProfileID != before || m.MigratedFrom.FromSchema != ProfileSchemaV1 || m.MigratedFrom.FromAnalysisSHA256 != legacyA.SHA256() ||
		!reflect.DeepEqual(m.MigratedFrom.RedundantPins, []string{RegionOutputEmbeddings}) {
		t.Fatalf("migration record %+v", m.MigratedFrom)
	}
	// Every full-attention group is an explicit override; nothing else is.
	for _, g := range current.Groups {
		want := GroupChoice{Mode: GroupAuto}
		if g.Region == RegionFullAttention {
			want = GroupChoice{Mode: GroupOverride, Policy: home.PolicySourcePrecision}
		}
		if m.Groups[g.ID] != want {
			t.Errorf("group %s migrated to %+v, want %+v", g.ID, m.Groups[g.ID], want)
		}
	}
	// The same modules are preserved: no material reinterpretation.
	mc := mustCompile(t, m, current)
	if !samePreservedModules(legacyC.Recipe, mc.Recipe, current) {
		t.Fatal("the migrated profile preserves other modules than the legacy one")
	}
	if mc.Recipe.SHA256() == legacyC.Recipe.SHA256() {
		t.Log("recipes happen to be byte-identical")
	}
	if err := SaveProfile(h, m, current); err != nil {
		t.Fatal(err)
	}

	// An all-Auto legacy profile migrates to the all-AUTO profile plus lineage.
	autoLegacy, _ := NewLegacyProfile(legacyA, "balanced")
	am, err := MigrateProfile(autoLegacy, legacyA, current)
	if err != nil {
		t.Fatal(err)
	}
	for id, c := range am.Groups {
		if c.Mode != GroupAuto {
			t.Errorf("group %s of an all-Auto legacy profile migrated to %+v", id, c)
		}
	}
	if am.MigratedFrom.RedundantPins != nil {
		t.Errorf("redundant pins %v", am.MigratedFrom.RedundantPins)
	}
	if !bytes.Equal(mustCompile(t, am, current).Recipe.Canonical(), legacyCompiled(t, autoLegacy, legacyA).Recipe.Canonical()) {
		t.Error("an all-Auto legacy profile did not migrate to the canonical recipe")
	}
}

func legacyCompiled(t *testing.T, p Profile, a Analysis) Compilation { return mustCompile(t, p, a) }

func TestMigrationRefusesAProfileOfAnotherStructureOrSource(t *testing.T) {
	source := clefSource(t)
	legacyA, err := AnalyzeLegacy(source, clefLayout())
	if err != nil {
		t.Fatal(err)
	}
	legacy, _ := NewLegacyProfile(legacyA, "balanced")
	current := layerwiseAnalysis(t, clefLayout())
	if _, err := MigrateProfile(legacy, legacyA, current); err != nil {
		t.Fatal(err)
	}
	// a structurally different current model
	layout := clefLayout()
	layout.LinearModules = append(layout.LinearModules, "model.language_model.layers.1.mlp.gate_proj")
	if _, err := MigrateProfile(legacy, legacyA, layerwiseAnalysis(t, layout)); err == nil || !strings.Contains(err.Error(), "different model structure") {
		t.Errorf("a legacy profile of another structure migrated: %v", err)
	}
	// not a legacy profile, or against a legacy "current" analysis
	if _, err := MigrateProfile(autoProfile(t, current), legacyA, current); err == nil {
		t.Error("a layer-wise profile was migrated")
	}
	if _, err := MigrateProfile(legacy, legacyA, legacyA); err == nil {
		t.Error("migration targeted a legacy analysis")
	}
	// a legacy profile whose own analysis does not match it
	bad := legacy
	bad.AnalysisSHA256 = strings.Repeat("0", 64)
	if _, err := MigrateProfile(bad, legacyA, current); err == nil {
		t.Error("a legacy profile not bound to its analysis migrated")
	}
	// legacy fields cannot ride on a layer-wise profile and vice versa
	mixed := legacy
	mixed.Groups = map[string]GroupChoice{"x": {Mode: GroupAuto}}
	if _, err := Compile(mixed, legacyA); err == nil {
		t.Error("a legacy profile carrying layer-wise fields compiled")
	}
}

func TestRecommendAndAcceptWorkOnLayerwiseProfiles(t *testing.T) {
	layout := clefLayout()
	layout.LinearModules = append(layout.LinearModules,
		"model.language_model.layers.0.linear_attn.in_proj_z", "model.language_model.layers.0.linear_attn.out_proj",
		"model.language_model.layers.1.linear_attn.in_proj_qkv")
	a := layerwiseAnalysis(t, layout)
	p := autoProfile(t, a)
	ctx := regressedContext(p)
	rec, why := Recommend(ctx, p, a)
	if rec == nil || rec.RegionID != RegionFullAttention || rec.From != PreservationAuto || rec.To != PreservationPinned {
		t.Fatalf("recommendation %+v (%s)", rec, why)
	}
	before := string(p.Canonical())
	next, err := Accept(*rec, p, a)
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Canonical()) != before || next.ID() == p.ID() || next.Schema != ProfileSchema {
		t.Fatal("accepting changed the profile or produced a non-layer-wise one")
	}
	if next.Groups["block.03.full-attn"] != (GroupChoice{Mode: GroupOverride, Policy: home.PolicySourcePrecision}) || next.Groups["block.00.mlp"].Mode != GroupAuto {
		t.Errorf("accepted profile groups %+v", next.Groups)
	}
	if _, err := Accept(*rec, next, a); err == nil {
		t.Error("recommendation applied to a profile it was not made for")
	}
	again := Recommendation{ProfileID: next.ID(), RegionID: RegionFullAttention, From: PreservationAuto, To: PreservationPinned}
	if _, err := Accept(again, next, a); err == nil {
		t.Error("an already-applied region was accepted again")
	}
	// once both supported families are overridden nothing is left to propose
	both := setPolicy(t, next, a, home.PolicySourcePrecision, "block.00.linear-attn", "block.01.linear-attn")
	if rec, why := Recommend(regressedContext(both), both, a); rec != nil || !strings.Contains(why, "already applied") {
		t.Errorf("recommendation for a profile with every supported change: %+v %q", rec, why)
	}
	// An operator's explicit W4A16 on a group keeps it out of the AUTO set.
	explicit := setPolicy(t, p, a, home.PolicyW4A16, "block.03.full-attn")
	if rec, _ := Recommend(regressedContext(explicit), explicit, a); rec == nil || rec.RegionID != RegionLinearAttention {
		t.Errorf("an explicitly quantized family was still proposed: %+v", rec)
	}
}

func TestDiffProfilesNamesGroupsAndRefusesToInventAMixedSchemaDifference(t *testing.T) {
	a := layerwiseAnalysis(t, clefLayout())
	p := autoProfile(t, a)
	q := setPolicy(t, p, a, home.PolicySourcePrecision, "block.00.mlp")
	q = setPolicy(t, q, a, home.PolicyW4A16, "block.03.full-attn")
	d, err := DiffProfiles(p, q)
	if err != nil {
		t.Fatal(err)
	}
	if d.Identical() || d.SchemaChanged() || len(d.Regions) != 0 || len(d.Groups) != 2 || d.Groups[0].GroupID != "block.00.mlp" || d.Groups[1].GroupID != "block.03.full-attn" {
		t.Fatalf("delta %+v", d)
	}
	if d.Groups[0].A != (GroupChoice{Mode: GroupAuto}) || d.Groups[0].B.Policy != home.PolicySourcePrecision {
		t.Errorf("group change %+v", d.Groups[0])
	}
	if same, _ := DiffProfiles(p, p); !same.Identical() || len(same.Groups) != 0 {
		t.Error("identical profiles differ")
	}
	legacy, _ := NewLegacyProfile(clefAnalysis(t), "balanced")
	mixed, err := DiffProfiles(legacy, p)
	if err != nil || !mixed.SchemaChanged() || len(mixed.Groups) != 0 || len(mixed.Regions) != 0 {
		t.Fatalf("mixed-schema delta %+v %v", mixed, err)
	}
}

func TestStoredLayerwiseDocumentsAreReadBackWithTheirAnalysis(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	a := layerwiseAnalysis(t, clefLayout())
	p := setPolicy(t, autoProfile(t, a), a, home.PolicySourcePrecision, "block.00.mlp")
	if err := SaveProfile(h, p, a); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{analysisPath(h, a.ID()), profilePath(h, p.ID())} {
		if _, err := os.Stat(path); err != nil {
			t.Fatal(err)
		}
	}
	// a stored analysis whose groups were edited fails its identity
	loaded, err := LoadAnalysis(h, a.ID())
	if err != nil {
		t.Fatal(err)
	}
	loaded.Groups = append([]Group(nil), loaded.Groups...)
	loaded.Groups[0].Modules = append([]string(nil), loaded.Groups[0].Modules...)
	loaded.Groups[0].Modules[0] = "model.language_model.layers.0.mlp.zzz"
	if err := home.WriteJSON(analysisPath(h, a.ID()), loaded); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAnalysis(h, a.ID()); err == nil {
		t.Error("a stored analysis changed its groups without changing its identity")
	}
	// groups that do not partition their regions are not an analysis
	broken := a
	broken.Groups = append([]Group(nil), a.Groups[1:]...)
	if err := validateAnalysis(broken); err == nil {
		t.Error("an analysis whose groups omit a member was accepted")
	}
	renamed := a
	renamed.Groups = append([]Group(nil), a.Groups...)
	renamed.Groups[0].ID = "block.00.other"
	if err := validateAnalysis(renamed); err == nil {
		t.Error("an analysis with a group identifier that is not derived from its structure was accepted")
	}
}
