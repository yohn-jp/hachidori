// Package tuning analyzes the supported Clef model layout and compiles
// tuning profiles into the canonical optimizer recipe.
//
// A profile (schema 2) is layer-wise: it holds one AUTO-or-override choice for
// every stable layer group of the exact source, each naming a policy the
// optimizer executes. The legacy coarse profile (schema 1, one choice per
// semantic region) remains readable and buildable and has a deterministic,
// equivalence-checked migration into schema 2 (MigrateProfile).
package tuning

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

const (
	// AnalysisSchema and ProfileSchema are the current layer-wise contracts;
	// the V1 schemas are the legacy coarse (per-region) contracts.
	AnalysisSchema   = "hachidori.tuning-analysis/2"
	ProfileSchema    = "hachidori.tuning-profile/2"
	AnalysisSchemaV1 = "hachidori.tuning-analysis/1"
	ProfileSchemaV1  = "hachidori.tuning-profile/1"

	// ClefAnalyzerVersion identifies the model-family rules used by Analyze.
	ClefAnalyzerVersion = "clef-qwen3.5/2"
	// RecipeCompilerVersion identifies the rules used by Compile.
	RecipeCompilerVersion = "home-recipe/2"
	// The legacy versions, still accepted for stored schema 1 documents.
	ClefAnalyzerVersionV1   = "clef-qwen3.5/1"
	RecipeCompilerVersionV1 = "home-recipe/1"

	PreservationAuto   PreservationMode = "auto"
	PreservationPinned PreservationMode = "pinned"
	// PreservationOverride and PreservationMixed describe a region of a
	// layer-wise profile in compiler evidence: every tunable group of the
	// region is overridden, or only some are.
	PreservationOverride PreservationMode = "override"
	PreservationMixed    PreservationMode = "mixed"

	PreservedPrecision = "bfloat16"

	RegionOutputEmbeddings     = "output-embeddings"
	RegionLinearAttentionDecay = "linear-attention-decay-gate"
	RegionLinearAttentionBeta  = "linear-attention-beta-gate"
	RegionLinearAttention      = "linear-attention-projections"
	RegionFullAttention        = "full-attention-projections"
	RegionFeedForward          = "feed-forward-projections"
	RegionVision               = "vision-tower"
	RegionJointSchemaHead      = "joint-schema-head"
)

// DeclaredLayout is the typed model metadata and Linear-module layout read
// from a digest-verified source. Modules and files are names only; analysis
// does not load weights or make physical quality or resource claims.
type DeclaredLayout struct {
	ModelType     string   `json:"model_type"`
	TextModelType string   `json:"text_model_type"`
	Architectures []string `json:"architectures"`
	LayerTypes    []string `json:"layer_types"`
	LinearModules []string `json:"linear_modules"`
	CarriedFiles  []string `json:"carried_files"`
}

// SemanticRegion is one model-family-specific group of module or file names.
// The names are exact analyzer output, not user profile inputs.
type SemanticRegion struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Modules     []string `json:"modules,omitempty"`
	Files       []string `json:"files,omitempty"`
}

// Analysis is a versioned, source-bound result of deterministic layout
// analysis. LayoutSHA256 and SHA256 bind the declared input and its result.
type Analysis struct {
	Schema          string             `json:"schema"`
	Source          home.VariantSource `json:"source"`
	AnalyzerVersion string             `json:"analyzer_version"`
	LayoutSHA256    string             `json:"layout_sha256"`
	Regions         []SemanticRegion   `json:"regions"`
	// Groups are the stable layer groups (schema 2): a partition of the
	// regions' members by transformer block and component.
	Groups []Group `json:"groups,omitempty"`
}

// PreservationMode controls how one semantic region compiles. Auto retains
// the canonical recipe behavior; pinned retains the region at source dtype.
type PreservationMode string

// PreservationChoice is the per-region setting recorded in a profile.
// Pinned profiles must state the only source precision supported by the
// current canonical optimizer contract.
type PreservationChoice struct {
	Mode      PreservationMode `json:"mode"`
	Precision string           `json:"precision,omitempty"`
}

// Profile is an immutable, versioned tuning request bound to one exact source
// and one analysis/compiler identity.
type Profile struct {
	Schema          string             `json:"schema"`
	Objective       string             `json:"objective"`
	Source          home.VariantSource `json:"source"`
	AnalyzerVersion string             `json:"analyzer_version"`
	CompilerVersion string             `json:"compiler_version"`
	AnalysisSHA256  string             `json:"analysis_sha256"`
	// Preservation is the legacy coarse choice per semantic region (schema 1).
	Preservation map[string]PreservationChoice `json:"preservation,omitempty"`
	// Recipe and RecipeSHA256 bind a layer-wise profile to the canonical
	// recipe whose AUTO policy it resolves against (schema 2).
	Recipe       string `json:"recipe,omitempty"`
	RecipeSHA256 string `json:"recipe_sha256,omitempty"`
	// Groups holds one choice for every group of the analysis (schema 2).
	Groups map[string]GroupChoice `json:"groups,omitempty"`
	// MigratedFrom records the legacy profile a schema 2 profile was derived
	// from.
	MigratedFrom *Migration `json:"migrated_from,omitempty"`
}

// RegionMapping is compiler evidence for the exact modules and files in one
// semantic region, including any recipe patterns that preserve them.
type RegionMapping struct {
	RegionID       string           `json:"region_id"`
	Mode           PreservationMode `json:"mode"`
	Modules        []string         `json:"modules,omitempty"`
	Files          []string         `json:"files,omitempty"`
	Preserved      bool             `json:"preserved"`
	Precision      string           `json:"precision,omitempty"`
	RecipePatterns []string         `json:"recipe_patterns,omitempty"`
}

// PreservedMapping explains exactly which analyzed modules or files match a
// generated home.PreservedModule entry. Patterns are compiler output only.
type PreservedMapping struct {
	Pattern   string   `json:"pattern"`
	Scope     string   `json:"scope"`
	Precision string   `json:"precision"`
	Regions   []string `json:"regions,omitempty"`
	Modules   []string `json:"modules,omitempty"`
	Files     []string `json:"files,omitempty"`
}

// CompilerEvidence exposes the concrete result of mapping semantic regions
// into the existing optimizer recipe contract.
type CompilerEvidence struct {
	Source          home.VariantSource `json:"source"`
	AnalyzerVersion string             `json:"analyzer_version"`
	CompilerVersion string             `json:"compiler_version"`
	ProfileSHA256   string             `json:"profile_sha256"`
	Regions         []RegionMapping    `json:"regions"`
	Preserved       []PreservedMapping `json:"preserved"`
}

// Compilation is the existing canonical recipe plus deterministic evidence
// for the semantic mappings that produced its preservation behavior.
type Compilation struct {
	Recipe   home.Recipe      `json:"recipe"`
	Evidence CompilerEvidence `json:"evidence"`
	// Plan is the resolved layer-wise policy of every group; nil for a legacy
	// coarse profile.
	Plan *home.TuningPlan `json:"plan,omitempty"`
}

var layerModuleRE = regexp.MustCompile(`^model\.language_model\.layers\.([0-9]+)\.([^.]+)\.(.+)$`)

var regionDescriptions = map[string]string{
	RegionOutputEmbeddings:     "output embedding rows consumed by the Clef joint head",
	RegionLinearAttentionDecay: "linear-attention recurrent decay gate projection",
	RegionLinearAttentionBeta:  "linear-attention recurrent beta gate projection",
	RegionLinearAttention:      "other linear-attention backbone projections",
	RegionFullAttention:        "full-attention backbone projections",
	RegionFeedForward:          "feed-forward backbone projections",
	RegionVision:               "vision-tower projections",
	RegionJointSchemaHead:      "separate joint-schema decision head file",
}

// Analyze produces the deterministic semantic regions and stable layer groups
// of the exact catalog Clef-Flash source. It accepts declared metadata/layout
// only and performs no model download or weight inspection.
func Analyze(source home.ModelManifest, layout DeclaredLayout) (Analysis, error) {
	a, canonical, err := analyze(source, layout)
	if err != nil {
		return Analysis{}, err
	}
	a.Schema, a.AnalyzerVersion = AnalysisSchema, ClefAnalyzerVersion
	if a.Groups, err = deriveGroups(canonical); err != nil {
		return Analysis{}, err
	}
	return a, nil
}

// AnalyzeLegacy produces the schema 1 analysis (semantic regions only) that
// legacy coarse profiles are bound to. Current tuning uses Analyze.
func AnalyzeLegacy(source home.ModelManifest, layout DeclaredLayout) (Analysis, error) {
	a, _, err := analyze(source, layout)
	if err != nil {
		return Analysis{}, err
	}
	a.Schema, a.AnalyzerVersion = AnalysisSchemaV1, ClefAnalyzerVersionV1
	return a, nil
}

func analyze(source home.ModelManifest, layout DeclaredLayout) (Analysis, DeclaredLayout, error) {
	identity, err := supportedSource(source)
	if err != nil {
		return Analysis{}, DeclaredLayout{}, err
	}
	canonical, err := canonicalLayout(layout)
	if err != nil {
		return Analysis{}, DeclaredLayout{}, err
	}
	regions, err := analyzeRegions(canonical)
	if err != nil {
		return Analysis{}, DeclaredLayout{}, err
	}
	layoutDigest, err := digestJSON(canonical)
	if err != nil {
		return Analysis{}, DeclaredLayout{}, err
	}
	return Analysis{Source: identity, LayoutSHA256: layoutDigest, Regions: regions}, canonical, nil
}

// NewDefaultProfile creates the all-AUTO layer-wise profile for an analysis.
// AUTO keeps the canonical recipe's policy for every group, so an all-AUTO
// profile compiles to the canonical recipe byte for byte.
func NewDefaultProfile(analysis Analysis, objective string) (Profile, error) {
	if err := validateAnalysis(analysis); err != nil {
		return Profile{}, err
	}
	if analysis.Schema != AnalysisSchema {
		return Profile{}, fmt.Errorf("a layer-wise profile needs a %s analysis, not %s", AnalysisSchema, analysis.Schema)
	}
	if strings.TrimSpace(objective) == "" {
		return Profile{}, errors.New("tuning profile objective is required")
	}
	canonical, err := canonicalRecipe(analysis)
	if err != nil {
		return Profile{}, err
	}
	groups := make(map[string]GroupChoice, len(analysis.Groups))
	for _, g := range analysis.Groups {
		groups[g.ID] = GroupChoice{Mode: GroupAuto}
	}
	return Profile{
		Schema: ProfileSchema, Objective: objective, Source: analysis.Source,
		AnalyzerVersion: analysis.AnalyzerVersion, CompilerVersion: RecipeCompilerVersion,
		AnalysisSHA256: analysis.SHA256(), Recipe: canonical.Name, RecipeSHA256: canonical.SHA256(), Groups: groups,
	}, nil
}

// NewLegacyProfile creates the all-Auto legacy coarse profile (schema 1) of a
// schema 1 analysis. It exists for compatibility; current tuning creates
// layer-wise profiles with NewDefaultProfile.
func NewLegacyProfile(analysis Analysis, objective string) (Profile, error) {
	if err := validateAnalysis(analysis); err != nil {
		return Profile{}, err
	}
	if analysis.Schema != AnalysisSchemaV1 {
		return Profile{}, fmt.Errorf("a legacy profile needs a %s analysis, not %s", AnalysisSchemaV1, analysis.Schema)
	}
	if strings.TrimSpace(objective) == "" {
		return Profile{}, errors.New("tuning profile objective is required")
	}
	choices := make(map[string]PreservationChoice, len(analysis.Regions))
	for _, region := range analysis.Regions {
		choices[region.ID] = PreservationChoice{Mode: PreservationAuto}
	}
	return Profile{
		Schema: ProfileSchemaV1, Objective: objective, Source: analysis.Source,
		AnalyzerVersion: analysis.AnalyzerVersion, CompilerVersion: RecipeCompilerVersionV1,
		AnalysisSHA256: analysis.SHA256(), Preservation: choices,
	}, nil
}

// Canonical is the stable JSON encoding used for profile identity.
func (p Profile) Canonical() []byte {
	b, err := json.Marshal(p)
	if err != nil {
		panic(err)
	}
	return b
}

// SHA256 is the profile identity digest.
func (p Profile) SHA256() string { return sha256Hex(p.Canonical()) }

// ID is the immutable persisted identity of a profile.
func (p Profile) ID() string { return p.SHA256() }

// Canonical is the stable JSON encoding used for analysis binding.
func (a Analysis) Canonical() []byte {
	b, err := json.Marshal(a)
	if err != nil {
		panic(err)
	}
	return b
}

// SHA256 is the analysis result digest used to bind profiles to their input.
func (a Analysis) SHA256() string { return sha256Hex(a.Canonical()) }

// ID is the immutable persisted identity of an analysis.
func (a Analysis) ID() string { return a.SHA256() }

// Compile deterministically turns a profile into the existing canonical
// home.Recipe. Auto keeps the optimizer baseline byte-for-byte; preserving a
// group (or, for a legacy profile, pinning a region) adds exactly the modules
// it addresses at the source precision. A layer-wise profile also yields the
// resolved plan of every group. It does not build a variant.
func Compile(profile Profile, analysis Analysis) (Compilation, error) {
	if err := validateAnalysis(analysis); err != nil {
		return Compilation{}, err
	}
	if err := validateProfile(profile, analysis); err != nil {
		return Compilation{}, err
	}
	if profile.Schema == ProfileSchema {
		return compileLayerwise(profile, analysis)
	}
	recipe, err := optimize.LookupRecipe(analysis.Source.ID, optimize.RecipeClefFlashW4A16)
	if err != nil {
		return Compilation{}, err
	}
	for _, region := range analysis.Regions {
		choice := profile.Preservation[region.ID]
		if choice.Mode != PreservationPinned {
			continue
		}
		for _, module := range region.Modules {
			if preservedBy(recipe, module, home.ScopeBackbone) {
				continue
			}
			recipe.Preserved = append(recipe.Preserved, home.PreservedModule{
				Pattern: module, Scope: home.ScopeBackbone, Precision: choice.Precision,
				Reason: fmt.Sprintf("tuning profile pins semantic region %s at source precision", region.ID),
			})
		}
		for _, file := range region.Files {
			if preservedBy(recipe, file, home.ScopeCarried) {
				continue
			}
			if !contains(recipe.Carry, file) {
				return Compilation{}, fmt.Errorf("pinned semantic region %q includes file %q that the canonical recipe does not carry", region.ID, file)
			}
			recipe.Preserved = append(recipe.Preserved, home.PreservedModule{
				Pattern: file, Scope: home.ScopeCarried, Precision: choice.Precision,
				Reason: fmt.Sprintf("tuning profile pins semantic region %s at source precision", region.ID),
			})
		}
	}
	if err := recipe.Validate(); err != nil {
		return Compilation{}, fmt.Errorf("compiled canonical recipe: %w", err)
	}
	evidence, err := compilerEvidence(profile, analysis, recipe)
	if err != nil {
		return Compilation{}, err
	}
	return Compilation{Recipe: recipe, Evidence: evidence}, nil
}

func validateProfile(profile Profile, analysis Analysis) error {
	switch profile.Schema {
	case ProfileSchema:
		return validateLayerwiseProfile(profile, analysis)
	case ProfileSchemaV1:
	default:
		return fmt.Errorf("tuning profile schema %q, want %q (or legacy %q)", profile.Schema, ProfileSchema, ProfileSchemaV1)
	}
	if analysis.Schema != AnalysisSchemaV1 {
		return fmt.Errorf("legacy tuning profile %q needs a %s analysis, not %s", profile.Schema, AnalysisSchemaV1, analysis.Schema)
	}
	if len(profile.Groups) != 0 || profile.Recipe != "" || profile.RecipeSHA256 != "" || profile.MigratedFrom != nil {
		return errors.New("a legacy coarse tuning profile cannot carry layer-wise fields")
	}
	if strings.TrimSpace(profile.Objective) == "" {
		return errors.New("tuning profile objective is required")
	}
	if profile.Source != analysis.Source {
		return errors.New("tuning profile source identity does not match analysis")
	}
	if profile.AnalyzerVersion != analysis.AnalyzerVersion || profile.AnalyzerVersion != ClefAnalyzerVersionV1 {
		return fmt.Errorf("tuning profile analyzer %q does not match analysis %q", profile.AnalyzerVersion, analysis.AnalyzerVersion)
	}
	if profile.CompilerVersion != RecipeCompilerVersionV1 {
		return fmt.Errorf("tuning profile compiler %q, want %q", profile.CompilerVersion, RecipeCompilerVersionV1)
	}
	if profile.AnalysisSHA256 != analysis.SHA256() {
		return errors.New("tuning profile analysis identity does not match analysis")
	}
	known := make(map[string]struct{}, len(analysis.Regions))
	for _, region := range analysis.Regions {
		known[region.ID] = struct{}{}
	}
	for id, choice := range profile.Preservation {
		if _, ok := known[id]; !ok {
			return fmt.Errorf("tuning profile names unknown semantic region %q", id)
		}
		switch choice.Mode {
		case PreservationAuto:
			if choice.Precision != "" {
				return fmt.Errorf("Auto preservation for region %q cannot name a precision", id)
			}
		case PreservationPinned:
			if choice.Precision != PreservedPrecision {
				return fmt.Errorf("pinned preservation for region %q must use source precision %q", id, PreservedPrecision)
			}
		default:
			return fmt.Errorf("tuning profile has unsupported preservation mode %q for region %q", choice.Mode, id)
		}
	}
	for _, region := range analysis.Regions {
		if _, ok := profile.Preservation[region.ID]; !ok {
			return fmt.Errorf("tuning profile has no preservation choice for semantic region %q", region.ID)
		}
	}
	return nil
}

func validateAnalysis(analysis Analysis) error {
	switch analysis.Schema {
	case AnalysisSchema:
		if analysis.AnalyzerVersion != ClefAnalyzerVersion {
			return fmt.Errorf("unsupported tuning analyzer %q", analysis.AnalyzerVersion)
		}
	case AnalysisSchemaV1:
		if analysis.AnalyzerVersion != ClefAnalyzerVersionV1 || len(analysis.Groups) != 0 {
			return fmt.Errorf("unsupported legacy tuning analyzer %q", analysis.AnalyzerVersion)
		}
	default:
		return fmt.Errorf("tuning analysis schema %q, want %q (or legacy %q)", analysis.Schema, AnalysisSchema, AnalysisSchemaV1)
	}
	if _, err := sourceForIdentity(analysis.Source); err != nil {
		return err
	}
	if !isDigest(analysis.LayoutSHA256) {
		return errors.New("tuning analysis has no valid declared-layout digest")
	}
	if len(analysis.Regions) == 0 {
		return errors.New("tuning analysis has no semantic regions")
	}
	last := ""
	seen := make(map[string]struct{}, len(analysis.Regions))
	for _, region := range analysis.Regions {
		if _, ok := regionDescriptions[region.ID]; !ok {
			return fmt.Errorf("tuning analysis has unknown semantic region %q", region.ID)
		}
		if region.ID <= last {
			return errors.New("tuning analysis regions are not in canonical order")
		}
		last = region.ID
		if _, ok := seen[region.ID]; ok {
			return fmt.Errorf("tuning analysis repeats semantic region %q", region.ID)
		}
		seen[region.ID] = struct{}{}
		if region.Description != regionDescriptions[region.ID] {
			return fmt.Errorf("tuning analysis region %q has an unexpected description", region.ID)
		}
		if !isSortedUnique(region.Modules) || !isSortedUnique(region.Files) {
			return fmt.Errorf("tuning analysis region %q names are not unique and sorted", region.ID)
		}
		if len(region.Modules)+len(region.Files) == 0 {
			return fmt.Errorf("tuning analysis region %q has no modules or files", region.ID)
		}
	}
	for _, id := range requiredRegions {
		if _, ok := seen[id]; !ok {
			return fmt.Errorf("tuning analysis is missing semantic region %q", id)
		}
	}
	if analysis.Schema == AnalysisSchema {
		return validateGroups(analysis)
	}
	return nil
}

func supportedSource(source home.ModelManifest) (home.VariantSource, error) {
	identity := home.SourceOf(source)
	expected, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		return home.VariantSource{}, err
	}
	want := home.SourceOf(expected)
	if identity != want {
		return home.VariantSource{}, fmt.Errorf("unsupported tuning source %s %s@%s; analysis requires exact catalog source %s %s@%s", identity.ID, identity.Repo, identity.Revision, want.ID, want.Repo, want.Revision)
	}
	return identity, nil
}

func sourceForIdentity(identity home.VariantSource) (home.VariantSource, error) {
	expected, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		return home.VariantSource{}, err
	}
	want := home.SourceOf(expected)
	if identity != want {
		return home.VariantSource{}, fmt.Errorf("tuning analysis is bound to unsupported source identity %s %s@%s", identity.ID, identity.Repo, identity.Revision)
	}
	return want, nil
}

func canonicalLayout(layout DeclaredLayout) (DeclaredLayout, error) {
	if layout.ModelType != "qwen3_5" || layout.TextModelType != "qwen3_5_text" {
		return DeclaredLayout{}, fmt.Errorf("unsupported Clef layout model types %q/%q", layout.ModelType, layout.TextModelType)
	}
	if len(layout.Architectures) != 1 || layout.Architectures[0] != "Qwen3_5ForConditionalGeneration" {
		return DeclaredLayout{}, fmt.Errorf("unsupported Clef architecture declaration %q", layout.Architectures)
	}
	if len(layout.LayerTypes) == 0 {
		return DeclaredLayout{}, errors.New("Clef layout has no text layer types")
	}
	for i, layerType := range layout.LayerTypes {
		if layerType != "linear_attention" && layerType != "full_attention" {
			return DeclaredLayout{}, fmt.Errorf("Clef layer %d has unsupported type %q", i, layerType)
		}
	}
	if len(layout.LinearModules) == 0 {
		return DeclaredLayout{}, errors.New("Clef layout has no declared Linear modules")
	}
	if len(layout.CarriedFiles) == 0 {
		return DeclaredLayout{}, errors.New("Clef layout has no declared carried files")
	}
	canonical := layout
	canonical.Architectures = append([]string(nil), layout.Architectures...)
	canonical.LayerTypes = append([]string(nil), layout.LayerTypes...)
	canonical.LinearModules = append([]string(nil), layout.LinearModules...)
	canonical.CarriedFiles = append([]string(nil), layout.CarriedFiles...)
	for _, module := range canonical.LinearModules {
		if module == "" || strings.TrimSpace(module) != module || strings.ContainsAny(module, "\r\n") || strings.HasPrefix(module, "re:") {
			return DeclaredLayout{}, fmt.Errorf("invalid declared Linear module name %q", module)
		}
	}
	if !isUnique(canonical.LinearModules) {
		return DeclaredLayout{}, errors.New("Clef layout repeats a Linear module name")
	}
	for _, file := range canonical.CarriedFiles {
		if file == "" || path.IsAbs(file) || path.Clean(file) != file || file == "." || strings.HasPrefix(file, "../") || strings.Contains(file, `\`) {
			return DeclaredLayout{}, fmt.Errorf("invalid declared carried file %q", file)
		}
	}
	if !isUnique(canonical.CarriedFiles) {
		return DeclaredLayout{}, errors.New("Clef layout repeats a carried file")
	}
	sort.Strings(canonical.Architectures)
	sort.Strings(canonical.LinearModules)
	sort.Strings(canonical.CarriedFiles)
	if !contains(canonical.CarriedFiles, "joint_head.safetensors") {
		return DeclaredLayout{}, errors.New("Clef layout does not declare joint_head.safetensors")
	}
	return canonical, nil
}

func analyzeRegions(layout DeclaredLayout) ([]SemanticRegion, error) {
	builders := map[string]*SemanticRegion{}
	for id, description := range regionDescriptions {
		builders[id] = &SemanticRegion{ID: id, Description: description}
	}
	for _, module := range layout.LinearModules {
		region, _, err := classifyModule(layout, module)
		if err != nil {
			return nil, err
		}
		builders[region].Modules = append(builders[region].Modules, module)
	}
	builders[RegionJointSchemaHead].Files = []string{"joint_head.safetensors"}
	regions := make([]SemanticRegion, 0, len(regionDescriptions))
	for id, builder := range builders {
		if len(builder.Modules)+len(builder.Files) == 0 {
			return nil, fmt.Errorf("Clef layout has no members for semantic region %q", id)
		}
		sort.Strings(builder.Modules)
		sort.Strings(builder.Files)
		regions = append(regions, *builder)
	}
	sort.Slice(regions, func(i, j int) bool { return regions[i].ID < regions[j].ID })
	return regions, nil
}

// classifyModule names the semantic region and transformer block (-1 for a
// module outside every block) of one declared Linear module.
func classifyModule(layout DeclaredLayout, module string) (string, int, error) {
	if module == "lm_head" {
		return RegionOutputEmbeddings, -1, nil
	}
	if strings.HasPrefix(module, "model.visual.") {
		return RegionVision, -1, nil
	}
	match := layerModuleRE.FindStringSubmatch(module)
	if match == nil {
		return "", 0, fmt.Errorf("Clef Linear module %q is outside the supported Qwen3.5 layout", module)
	}
	layer, err := strconv.Atoi(match[1])
	if err != nil || layer < 0 || layer >= len(layout.LayerTypes) {
		return "", 0, fmt.Errorf("Clef Linear module %q names an undeclared layer", module)
	}
	component, suffix := match[2], match[3]
	switch component {
	case "mlp":
		if suffix != "gate_proj" && suffix != "up_proj" && suffix != "down_proj" {
			return "", 0, fmt.Errorf("Clef Linear module %q has unsupported feed-forward projection", module)
		}
		return RegionFeedForward, layer, nil
	case "linear_attn":
		if layout.LayerTypes[layer] != "linear_attention" {
			return "", 0, fmt.Errorf("Clef module %q conflicts with declared layer type %q", module, layout.LayerTypes[layer])
		}
		switch suffix {
		case "in_proj_a":
			return RegionLinearAttentionDecay, layer, nil
		case "in_proj_b":
			return RegionLinearAttentionBeta, layer, nil
		case "in_proj_qkv", "in_proj_z", "out_proj":
			return RegionLinearAttention, layer, nil
		}
		return "", 0, fmt.Errorf("Clef Linear module %q has unsupported linear-attention projection", module)
	case "self_attn":
		if layout.LayerTypes[layer] != "full_attention" {
			return "", 0, fmt.Errorf("Clef module %q conflicts with declared layer type %q", module, layout.LayerTypes[layer])
		}
		if suffix != "q_proj" && suffix != "k_proj" && suffix != "v_proj" && suffix != "o_proj" {
			return "", 0, fmt.Errorf("Clef Linear module %q has unsupported full-attention projection", module)
		}
		return RegionFullAttention, layer, nil
	}
	return "", 0, fmt.Errorf("Clef Linear module %q has unsupported component %q", module, component)
}

func compilerEvidence(profile Profile, analysis Analysis, recipe home.Recipe) (CompilerEvidence, error) {
	byRegion := make(map[string]RegionMapping, len(analysis.Regions))
	for _, region := range analysis.Regions {
		byRegion[region.ID] = RegionMapping{
			RegionID: region.ID, Mode: regionMode(profile, analysis, region.ID),
			Modules: append([]string(nil), region.Modules...), Files: append([]string(nil), region.Files...),
		}
	}
	preserved := make([]PreservedMapping, 0, len(recipe.Preserved))
	for _, entry := range recipe.Preserved {
		mapping, err := resolvePreserved(entry, analysis)
		if err != nil {
			return CompilerEvidence{}, err
		}
		preserved = append(preserved, mapping)
		for _, regionID := range mapping.Regions {
			region := byRegion[regionID]
			region.RecipePatterns = append(region.RecipePatterns, entry.Pattern)
			byRegion[regionID] = region
		}
	}
	regions := make([]RegionMapping, 0, len(analysis.Regions))
	for _, semantic := range analysis.Regions {
		mapping := byRegion[semantic.ID]
		mapping.RecipePatterns = sortedUnique(mapping.RecipePatterns)
		allPreserved := true
		for _, module := range semantic.Modules {
			if !preservedBy(recipe, module, home.ScopeBackbone) {
				allPreserved = false
				break
			}
		}
		if allPreserved {
			for _, file := range semantic.Files {
				if !preservedBy(recipe, file, home.ScopeCarried) {
					allPreserved = false
					break
				}
			}
		}
		mapping.Preserved = allPreserved
		if allPreserved {
			mapping.Precision = PreservedPrecision
		}
		regions = append(regions, mapping)
	}
	return CompilerEvidence{
		Source: analysis.Source, AnalyzerVersion: analysis.AnalyzerVersion,
		CompilerVersion: profile.CompilerVersion, ProfileSHA256: profile.SHA256(),
		Regions: regions, Preserved: preserved,
	}, nil
}

func resolvePreserved(entry home.PreservedModule, analysis Analysis) (PreservedMapping, error) {
	result := PreservedMapping{Pattern: entry.Pattern, Scope: entry.Scope, Precision: entry.Precision}
	for _, region := range analysis.Regions {
		for _, module := range region.Modules {
			match, err := patternMatches(entry.Pattern, module)
			if err != nil {
				return PreservedMapping{}, err
			}
			if entry.Scope == home.ScopeBackbone && match {
				result.Modules = append(result.Modules, module)
				result.Regions = append(result.Regions, region.ID)
			}
		}
		for _, file := range region.Files {
			if entry.Scope == home.ScopeCarried && entry.Pattern == file {
				result.Files = append(result.Files, file)
				result.Regions = append(result.Regions, region.ID)
			}
		}
	}
	result.Modules = sortedUnique(result.Modules)
	result.Files = sortedUnique(result.Files)
	result.Regions = sortedUnique(result.Regions)
	if len(result.Modules)+len(result.Files) == 0 {
		return PreservedMapping{}, fmt.Errorf("canonical recipe preservation pattern %q matches no analyzed module or file", entry.Pattern)
	}
	return result, nil
}

func patternMatches(pattern, module string) (bool, error) {
	if !strings.HasPrefix(pattern, "re:") {
		return pattern == module, nil
	}
	re, err := regexp.Compile(pattern[3:])
	if err != nil {
		return false, fmt.Errorf("compile canonical recipe pattern %q: %w", pattern, err)
	}
	return re.MatchString(module), nil
}

func preservedBy(recipe home.Recipe, name, scope string) bool {
	for _, entry := range recipe.Preserved {
		if entry.Scope != scope {
			continue
		}
		match, err := patternMatches(entry.Pattern, name)
		if err == nil && match {
			return true
		}
	}
	return false
}

var requiredRegions = []string{
	RegionOutputEmbeddings,
	RegionLinearAttentionDecay,
	RegionLinearAttentionBeta,
	RegionLinearAttention,
	RegionFullAttention,
	RegionFeedForward,
	RegionVision,
	RegionJointSchemaHead,
}

func digestJSON(value any) (string, error) {
	b, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return sha256Hex(b), nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func isDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func isUnique(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			return false
		}
		seen[value] = struct{}{}
	}
	return true
}

func isSortedUnique(values []string) bool {
	for i := 1; i < len(values); i++ {
		if values[i-1] >= values[i] {
			return false
		}
	}
	return true
}

func sortedUnique(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	result := append([]string(nil), values...)
	sort.Strings(result)
	write := 0
	for _, value := range result {
		if write != 0 && result[write-1] == value {
			continue
		}
		result[write] = value
		write++
	}
	return result[:write]
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}
