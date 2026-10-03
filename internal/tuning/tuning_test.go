package tuning

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

func clefSource(t *testing.T) home.ModelManifest {
	t.Helper()
	source, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	return source
}

func clefLayout() DeclaredLayout {
	return DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text",
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes:    []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{
			"lm_head",
			"model.visual.blocks.0.attn.qkv",
			"model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a",
			"model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv",
			"model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj",
		},
		CarriedFiles: []string{"joint_head.safetensors"},
	}
}

func clefAnalysis(t *testing.T) Analysis {
	t.Helper()
	analysis, err := Analyze(clefSource(t), clefLayout())
	if err != nil {
		t.Fatal(err)
	}
	return analysis
}

func TestAnalyzeClefIsDeterministicAndSourceBound(t *testing.T) {
	source, layout := clefSource(t), clefLayout()
	first, err := Analyze(source, layout)
	if err != nil {
		t.Fatal(err)
	}
	for left, right := 0, len(layout.LinearModules)-1; left < right; left, right = left+1, right-1 {
		layout.LinearModules[left], layout.LinearModules[right] = layout.LinearModules[right], layout.LinearModules[left]
	}
	second, err := Analyze(source, layout)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Canonical(), second.Canonical()) || first.SHA256() != second.SHA256() {
		t.Fatal("equivalent declared layouts produced different analyses")
	}
	if first.Schema != AnalysisSchema || first.Source != home.SourceOf(source) || first.AnalyzerVersion != ClefAnalyzerVersion {
		t.Fatalf("analysis is not bound to the exact source and analyzer: %+v", first)
	}
	if len(first.Regions) != len(requiredRegions) {
		t.Fatalf("analysis has %d regions, want %d", len(first.Regions), len(requiredRegions))
	}

	changed := source
	changed.Revision = strings.Repeat("0", len(source.Revision))
	if _, err := Analyze(changed, clefLayout()); err == nil {
		t.Fatal("analysis accepted a different source revision")
	}
}

func TestDefaultAutoCompilesCanonicalRecipeAndExactEvidence(t *testing.T) {
	analysis := clefAnalysis(t)
	profile, err := NewDefaultProfile(analysis, "balanced typed decisions")
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := Compile(profile, analysis)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(compiled.Recipe.Canonical(), baseline.Canonical()) {
		t.Fatalf("default Auto changed canonical recipe:\n%s\n%s", compiled.Recipe.Canonical(), baseline.Canonical())
	}
	if compiled.Evidence.Source != analysis.Source || compiled.Evidence.AnalyzerVersion != ClefAnalyzerVersion || compiled.Evidence.CompilerVersion != RecipeCompilerVersion || compiled.Evidence.ProfileSHA256 != profile.SHA256() {
		t.Fatalf("compiler evidence is not bound to its inputs: %+v", compiled.Evidence)
	}
	decay := regionMapping(t, compiled.Evidence, RegionLinearAttentionDecay)
	if !decay.Preserved || decay.Precision != PreservedPrecision || !contains(decay.Modules, "model.language_model.layers.0.linear_attn.in_proj_a") {
		t.Fatalf("Auto did not report the existing decay-gate preservation: %+v", decay)
	}
	visual := regionMapping(t, compiled.Evidence, RegionVision)
	if len(visual.Modules) != 2 || !visual.Preserved {
		t.Fatalf("visual mapping does not expose exact generated modules: %+v", visual)
	}
	if !strings.Contains(string(profile.Canonical()), `"mode":"auto"`) || strings.Contains(string(profile.Canonical()), "re:") {
		t.Fatalf("profile exposes compiler patterns or omits Auto: %s", profile.Canonical())
	}
}

func TestPinnedSemanticRegionChangesCanonicalIdentity(t *testing.T) {
	analysis := clefAnalysis(t)
	profile, err := NewDefaultProfile(analysis, "preserve full attention at source precision")
	if err != nil {
		t.Fatal(err)
	}
	profile.Preservation[RegionFullAttention] = PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}
	compiled, err := Compile(profile, analysis)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.Recipe.SHA256() == baseline.SHA256() {
		t.Fatal("pinning a quantized semantic region did not change recipe identity")
	}
	optimizer := home.Optimizer{Engine: baseline.Engine, Version: "fixture"}
	baseBuild := home.DeriveBuildID(analysis.Source, analysis.Source.Provider, optimizer, baseline.SHA256(), nil)
	pinnedBuild := home.DeriveBuildID(analysis.Source, analysis.Source.Provider, optimizer, compiled.Recipe.SHA256(), nil)
	if baseBuild == pinnedBuild {
		t.Fatal("semantic preservation change did not change BuildID inputs")
	}
	fullAttention := regionMapping(t, compiled.Evidence, RegionFullAttention)
	if !fullAttention.Preserved || fullAttention.Precision != PreservedPrecision || !contains(fullAttention.RecipePatterns, "model.language_model.layers.3.self_attn.q_proj") {
		t.Fatalf("compiler evidence omits the generated exact mapping: %+v", fullAttention)
	}
}

func TestProfileValidationRejectsUnknownAndMismatchedInputs(t *testing.T) {
	analysis := clefAnalysis(t)
	valid, err := NewDefaultProfile(analysis, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*Profile){
		"unknown region": func(p *Profile) { p.Preservation["unknown"] = PreservationChoice{Mode: PreservationAuto} },
		"missing region": func(p *Profile) { delete(p.Preservation, RegionVision) },
		"source":         func(p *Profile) { p.Source.Revision = strings.Repeat("0", len(p.Source.Revision)) },
		"analyzer":       func(p *Profile) { p.AnalyzerVersion = "other/1" },
		"compiler":       func(p *Profile) { p.CompilerVersion = "other/1" },
		"analysis":       func(p *Profile) { p.AnalysisSHA256 = strings.Repeat("0", 64) },
		"pinned precision": func(p *Profile) {
			p.Preservation[RegionFullAttention] = PreservationChoice{Mode: PreservationPinned, Precision: "float16"}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			candidate := valid
			candidate.Preservation = make(map[string]PreservationChoice, len(valid.Preservation))
			for id, choice := range valid.Preservation {
				candidate.Preservation[id] = choice
			}
			mutate(&candidate)
			if _, err := Compile(candidate, analysis); err == nil {
				t.Fatal("invalid profile compiled")
			}
		})
	}
}

func TestStoredAnalysisAndProfileAreImmutableAndSourceBound(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	analysis := clefAnalysis(t)
	profile, err := NewDefaultProfile(analysis, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveProfile(h, profile, analysis); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{analysisPath(h, analysis.ID()), profilePath(h, profile.ID())} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("versioned tuning document %s was not persisted: %v", filepath.Base(path), err)
		}
	}
	loadedProfile, loadedAnalysis, err := LoadProfile(h, profile.ID())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(profile.Canonical(), loadedProfile.Canonical()) || !bytes.Equal(analysis.Canonical(), loadedAnalysis.Canonical()) {
		t.Fatal("loaded tuning documents differ from their canonical persisted inputs")
	}

	changed := profile
	changed.Objective = "another objective"
	if err := SaveProfile(h, changed, analysis); err != nil {
		t.Fatalf("a new profile identity could not be stored: %v", err)
	}
	if changed.ID() == profile.ID() {
		t.Fatal("a changed profile did not change its identity")
	}

	stored := loadedProfile
	stored.Objective = "tampered"
	if err := home.WriteJSON(profilePath(h, profile.ID()), stored); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadProfile(h, profile.ID()); err == nil {
		t.Fatal("a stored profile changed without changing its persisted identity")
	}
	if err := home.WriteJSON(profilePath(h, profile.ID()), loadedProfile); err != nil {
		t.Fatal(err)
	}
	tamperedAnalysis := loadedAnalysis
	tamperedAnalysis.LayoutSHA256 = strings.Repeat("0", 64)
	if err := home.WriteJSON(analysisPath(h, analysis.ID()), tamperedAnalysis); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAnalysis(h, analysis.ID()); err == nil {
		t.Fatal("a stored analysis changed without changing its persisted identity")
	}
}

func TestAnalyzeRejectsUnsupportedDeclaredLayout(t *testing.T) {
	bad := clefLayout()
	bad.LinearModules = append(bad.LinearModules, "model.language_model.layers.1.unknown.proj")
	if _, err := Analyze(clefSource(t), bad); err == nil {
		t.Fatal("analyzer accepted a module outside the supported Clef layout")
	}
	bad = clefLayout()
	bad.LinearModules = append(bad.LinearModules, "model.language_model.layers.1.linear_attn.unsupported")
	if _, err := Analyze(clefSource(t), bad); err == nil {
		t.Fatal("analyzer accepted an unknown linear-attention projection")
	}
	bad = clefLayout()
	bad.LayerTypes[0] = "full_attention"
	if _, err := Analyze(clefSource(t), bad); err == nil {
		t.Fatal("analyzer accepted a module that conflicts with the declared layer type")
	}
}

func regionMapping(t *testing.T, evidence CompilerEvidence, id string) RegionMapping {
	t.Helper()
	for _, mapping := range evidence.Regions {
		if mapping.RegionID == id {
			return mapping
		}
	}
	t.Fatalf("compiler evidence has no region %q", id)
	return RegionMapping{}
}
