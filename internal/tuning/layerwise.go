package tuning

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
)

// AutoPolicyVersion identifies the rules that resolve AUTO to an explicit
// policy. AUTO for a tunable group is the canonical recipe's own policy; a
// change to these rules changes the resolved plan and therefore the identity of
// every candidate built under it, without touching a saved profile.
const AutoPolicyVersion = "clef-auto/1"

// GroupMode says who decides a group's policy.
type GroupMode string

const (
	// GroupAuto leaves the policy to Hachidori; the profile records no policy.
	GroupAuto GroupMode = "auto"
	// GroupOverride is the operator's explicit, named policy.
	GroupOverride GroupMode = "override"
)

// GroupChoice is the per-group setting recorded in a layer-wise profile.
type GroupChoice struct {
	Mode   GroupMode `json:"mode"`
	Policy string    `json:"policy,omitempty"`
}

// PolicyInfo describes one transformation policy the optimizer can execute.
// Policies are ordered by Rank, rising with compression: only an ordered set
// like this may be offered on an ordered control.
type PolicyInfo struct {
	ID             string
	Rank           int
	Transformation string // exact transformation applied
}

// Policies lists, in rank order, exactly the transformations the optimizer
// executes today: keep a group at its source precision, or quantize its Linear
// modules with the canonical recipe's weight-only W4A16 scheme. No other
// precision, pruning or structural transformation exists behind this list.
//
// Backend limit, with code evidence: the optimizer writes one compressed-tensors
// quantization group (hachidori_optimizer.py asserts a single int4 group and
// weightsOf declares one scheme), setup.CheckPreserved refuses a variant whose
// saved config is not a single group matching the manifest's scheme, and the
// only per-module selector the backend addresses is its ignore list. A second
// quantized precision (for example W8A16) therefore cannot be expressed per
// group today and is not offered.
func Policies() []PolicyInfo {
	return []PolicyInfo{
		{ID: home.PolicySourcePrecision, Rank: 0, Transformation: "kept at the source precision (" + PreservedPrecision + "), not quantized"},
		{ID: home.PolicyW4A16, Rank: 1, Transformation: "weight-only int4, symmetric, group size 128, round to nearest (W4A16)"},
	}
}

// PolicyRank returns the compression rank of a policy.
func PolicyRank(id string) (int, bool) {
	for _, p := range Policies() {
		if p.ID == id {
			return p.Rank, true
		}
	}
	return 0, false
}

// Migration records the legacy coarse profile a layer-wise profile was derived
// from, and the legacy pins that changed nothing because the canonical
// contract already preserves what they named.
type Migration struct {
	FromProfileID      string   `json:"from_profile_id"`
	FromSchema         string   `json:"from_schema"`
	FromAnalysisSHA256 string   `json:"from_analysis_sha256"`
	RedundantPins      []string `json:"redundant_pins,omitempty"` // legacy regions
}

func canonicalRecipe(analysis Analysis) (home.Recipe, error) {
	return optimize.LookupRecipe(analysis.Source.ID, optimize.RecipeClefFlashW4A16)
}

// requiredGroup reports whether the canonical optimizer contract always keeps
// the group at its source precision, and why. A group the canonical recipe
// preserves only in part is a structure the contract cannot address.
func requiredGroup(canonical home.Recipe, g Group) (bool, string, error) {
	covered, total := 0, len(g.Modules)+len(g.Files)
	reason := ""
	cover := func(name, scope string) {
		for _, entry := range canonical.Preserved {
			if entry.Scope != scope {
				continue
			}
			if match, err := patternMatches(entry.Pattern, name); err == nil && match {
				covered++
				if reason == "" {
					reason = entry.Reason
				}
				return
			}
		}
	}
	for _, m := range g.Modules {
		cover(m, home.ScopeBackbone)
	}
	for _, f := range g.Files {
		cover(f, home.ScopeCarried)
	}
	switch {
	case covered == 0:
		return false, "", nil
	case covered == total:
		return true, reason, nil
	}
	return false, "", fmt.Errorf("group %q is only partly preserved by the canonical recipe, so no policy can address it", g.ID)
}

// groupPattern is the exact recipe pattern for a group's modules: an anchored
// alternation of the full module names, matched from the start by the backend
// and by the compiler's evidence mapping alike.
func groupPattern(modules []string) string {
	quoted := make([]string, len(modules))
	for i, m := range modules {
		quoted[i] = regexp.QuoteMeta(m)
	}
	return "re:^(?:" + strings.Join(quoted, "|") + ")$"
}

func validateLayerwiseProfile(profile Profile, analysis Analysis) error {
	if analysis.Schema != AnalysisSchema {
		return fmt.Errorf("layer-wise tuning profile needs a %s analysis, not %s", AnalysisSchema, analysis.Schema)
	}
	if strings.TrimSpace(profile.Objective) == "" {
		return errors.New("tuning profile objective is required")
	}
	if profile.Source != analysis.Source {
		return errors.New("tuning profile source identity does not match analysis")
	}
	if profile.AnalyzerVersion != analysis.AnalyzerVersion || profile.AnalyzerVersion != ClefAnalyzerVersion {
		return fmt.Errorf("tuning profile analyzer %q does not match analysis %q", profile.AnalyzerVersion, analysis.AnalyzerVersion)
	}
	if profile.CompilerVersion != RecipeCompilerVersion {
		return fmt.Errorf("tuning profile compiler %q, want %q", profile.CompilerVersion, RecipeCompilerVersion)
	}
	if profile.AnalysisSHA256 != analysis.SHA256() {
		return errors.New("tuning profile analysis identity does not match analysis")
	}
	if len(profile.Preservation) != 0 {
		return errors.New("a layer-wise tuning profile cannot carry legacy region preservation")
	}
	canonical, err := canonicalRecipe(analysis)
	if err != nil {
		return err
	}
	if profile.Recipe != canonical.Name || profile.RecipeSHA256 != canonical.SHA256() {
		return fmt.Errorf("tuning profile is bound to recipe %s (%s), not the current canonical recipe %s (%s): its AUTO policy is not the one it was authored against",
			profile.Recipe, short(profile.RecipeSHA256), canonical.Name, short(canonical.SHA256()))
	}
	if m := profile.MigratedFrom; m != nil && (!isDigest(m.FromProfileID) || !isDigest(m.FromAnalysisSHA256) || m.FromSchema != ProfileSchemaV1) {
		return errors.New("tuning profile migration record is not a legacy profile reference")
	}
	known := make(map[string]struct{}, len(analysis.Groups))
	for _, g := range analysis.Groups {
		known[g.ID] = struct{}{}
		choice, ok := profile.Groups[g.ID]
		if !ok {
			return fmt.Errorf("tuning profile has no choice for group %q", g.ID)
		}
		required, _, err := requiredGroup(canonical, g)
		if err != nil {
			return err
		}
		switch choice.Mode {
		case GroupAuto:
			if choice.Policy != "" {
				return fmt.Errorf("AUTO for group %q cannot name a policy", g.ID)
			}
		case GroupOverride:
			if _, ok := PolicyRank(choice.Policy); !ok {
				return fmt.Errorf("group %q overrides to unsupported policy %q (supported: %s)", g.ID, choice.Policy, policyNames())
			}
			if required {
				return fmt.Errorf("group %q is required at source precision by the canonical optimizer contract and cannot be overridden", g.ID)
			}
			if len(g.Files) != 0 {
				return fmt.Errorf("group %q is a carried file and cannot be overridden", g.ID)
			}
		default:
			return fmt.Errorf("tuning profile has unsupported mode %q for group %q", choice.Mode, g.ID)
		}
	}
	for id := range profile.Groups {
		if _, ok := known[id]; !ok {
			return fmt.Errorf("tuning profile names unknown group %q", id)
		}
	}
	return nil
}

func policyNames() string {
	var names []string
	for _, p := range Policies() {
		names = append(names, p.ID)
	}
	return strings.Join(names, ", ")
}

// resolveGroups resolves every group of the analysis to its explicit policy:
// the canonical contract's requirement, the operator's override, or what AUTO
// selects. The profile must already be validated.
func resolveGroups(profile Profile, analysis Analysis, canonical home.Recipe) ([]home.TuningGroupPlan, error) {
	plan := make([]home.TuningGroupPlan, 0, len(analysis.Groups))
	for _, g := range analysis.Groups {
		required, reason, err := requiredGroup(canonical, g)
		if err != nil {
			return nil, err
		}
		choice := profile.Groups[g.ID]
		p := home.TuningGroupPlan{ID: g.ID, Region: g.Region, Layer: g.Layer,
			Modules: append([]string(nil), g.Modules...), Files: append([]string(nil), g.Files...)}
		switch {
		case required:
			p.Selection, p.Requested, p.Effective, p.Required = home.SelectionAuto, home.PolicyAuto, home.PolicySourcePrecision, true
			p.Basis = "required by the canonical optimizer contract: " + reason
		case choice.Mode == GroupOverride:
			p.Selection, p.Requested, p.Effective = home.SelectionOverridden, choice.Policy, choice.Policy
			p.Basis = "operator override"
		default:
			p.Selection, p.Requested, p.Effective = home.SelectionAuto, home.PolicyAuto, home.PolicyW4A16
			p.Basis = "AUTO (" + AutoPolicyVersion + "): the canonical recipe " + canonical.Name + " applies " + canonical.Scheme + " to this component"
		}
		plan = append(plan, p)
	}
	return plan, nil
}

// compileLayerwise compiles a validated layer-wise profile: the canonical
// recipe plus one exact preserved-module entry for each tunable group the plan
// keeps at source precision, in group ID order, and the plan itself.
func compileLayerwise(profile Profile, analysis Analysis) (Compilation, error) {
	canonical, err := canonicalRecipe(analysis)
	if err != nil {
		return Compilation{}, err
	}
	groups, err := resolveGroups(profile, analysis, canonical)
	if err != nil {
		return Compilation{}, err
	}
	recipe := canonical
	quantized := 0
	for _, g := range groups {
		if g.Effective == home.PolicyW4A16 {
			quantized += len(g.Modules)
		}
		if g.Required || g.Effective != home.PolicySourcePrecision {
			continue
		}
		recipe.Preserved = append(recipe.Preserved, home.PreservedModule{
			Pattern: groupPattern(g.Modules), Scope: home.ScopeBackbone, Precision: PreservedPrecision,
			Reason: fmt.Sprintf("tuning profile keeps group %s at source precision (operator override)", g.ID),
		})
	}
	if quantized == 0 {
		return Compilation{}, fmt.Errorf("every group is kept at source precision, so nothing would be %s: the candidate would not be the declared %s variant", canonical.Scheme, canonical.Scheme)
	}
	if err := recipe.Validate(); err != nil {
		return Compilation{}, fmt.Errorf("compiled canonical recipe: %w", err)
	}
	plan := home.TuningPlan{Schema: home.TuningPlanSchema, AutoPolicy: AutoPolicyVersion, Recipe: recipe.Name, RecipeSHA256: recipe.SHA256(), Groups: groups}
	if err := plan.Validate(); err != nil {
		return Compilation{}, err
	}
	evidence, err := compilerEvidence(profile, analysis, recipe)
	if err != nil {
		return Compilation{}, err
	}
	return Compilation{Recipe: recipe, Evidence: evidence, Plan: &plan}, nil
}

// regionMode summarizes a region's choices for compiler evidence.
func regionMode(profile Profile, analysis Analysis, region string) PreservationMode {
	if profile.Schema != ProfileSchema {
		return profile.Preservation[region].Mode
	}
	overridden, tunable := 0, 0
	canonical, err := canonicalRecipe(analysis)
	for _, g := range analysis.Groups {
		if g.Region != region {
			continue
		}
		if required, _, rerr := requiredGroup(canonical, g); err == nil && rerr == nil && required {
			continue
		}
		tunable++
		if profile.Groups[g.ID].Mode == GroupOverride {
			overridden++
		}
	}
	switch {
	case overridden == 0:
		return PreservationAuto
	case overridden == tunable:
		return PreservationOverride
	}
	return PreservationMixed
}

// Provenance is the build provenance of a compiled profile: the exact profile
// and analysis, and, for a layer-wise profile, the digest of the resolved plan
// (the complete effective policy, AUTO resolutions included).
func Provenance(profile Profile, analysis Analysis, c Compilation) home.TuningProvenance {
	p := home.TuningProvenance{
		Schema: home.TuningProvenanceSchemaV1, Source: profile.Source,
		ProfileID: profile.ID(), ProfileSHA256: profile.SHA256(),
		AnalysisID: analysis.ID(), AnalysisSHA256: analysis.SHA256(),
		CompilerVersion: profile.CompilerVersion,
	}
	if c.Plan != nil {
		p.Schema, p.PlanSHA256 = home.TuningProvenanceSchema, c.Plan.SHA256()
	}
	return p
}

// Selection picks tunable groups: those of one region (empty for every
// region) whose block lies in [From, To] (a negative bound is open).
type Selection struct {
	Region   string
	From, To int
}

// Tunable returns the IDs of the groups of the analysis the operator may set:
// every group the canonical contract does not require at source precision.
func Tunable(analysis Analysis) ([]string, error) {
	canonical, err := canonicalRecipe(analysis)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, g := range analysis.Groups {
		required, _, err := requiredGroup(canonical, g)
		if err != nil {
			return nil, err
		}
		if !required && len(g.Files) == 0 {
			ids = append(ids, g.ID)
		}
	}
	return ids, nil
}

// Select returns the tunable groups a selection addresses, in ID order. A
// selection that addresses none is an error, never a silent no-op.
func (s Selection) Select(analysis Analysis) ([]string, error) {
	tunable, err := Tunable(analysis)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, id := range tunable {
		g, _ := analysis.GroupByID(id)
		if s.Region != "" && g.Region != s.Region {
			continue
		}
		if s.From >= 0 && (g.Layer < 0 || g.Layer < s.From) || s.To >= 0 && (g.Layer < 0 || g.Layer > s.To) {
			continue
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("the selection (region %q, blocks %d to %d) addresses no tunable group", s.Region, s.From, s.To)
	}
	return ids, nil
}

// SetPolicy returns a copy of a layer-wise profile with the named groups set
// to policy; home.PolicyAuto resets them to AUTO. The profile is never
// mutated, and the result is validated: an unsupported policy or a group that
// is not tunable is refused.
func SetPolicy(profile Profile, analysis Analysis, ids []string, policy string) (Profile, error) {
	if profile.Schema != ProfileSchema {
		return Profile{}, errors.New("only a layer-wise profile has group policies")
	}
	next := profile
	next.Groups = make(map[string]GroupChoice, len(profile.Groups))
	for id, c := range profile.Groups {
		next.Groups[id] = c
	}
	for _, id := range ids {
		if _, ok := analysis.GroupByID(id); !ok {
			return Profile{}, fmt.Errorf("unknown group %q", id)
		}
		if policy == home.PolicyAuto {
			next.Groups[id] = GroupChoice{Mode: GroupAuto}
		} else {
			next.Groups[id] = GroupChoice{Mode: GroupOverride, Policy: policy}
		}
	}
	if err := validateLayerwiseProfile(next, analysis); err != nil {
		return Profile{}, err
	}
	return next, nil
}

// MigrateProfile derives the layer-wise profile exactly equivalent to a legacy
// coarse profile. The legacy profile is untouched and stays buildable; the
// result is a new, distinct profile. Migration is refused unless the current
// analysis describes the very layout the legacy analysis was made from, and
// unless both compile to the same set of preserved modules: a profile is never
// reinterpreted into a materially different optimization.
func MigrateProfile(legacy Profile, legacyAnalysis, current Analysis) (Profile, error) {
	if legacy.Schema != ProfileSchemaV1 {
		return Profile{}, fmt.Errorf("profile schema %q is not a legacy coarse profile", legacy.Schema)
	}
	if err := validateAnalysis(legacyAnalysis); err != nil {
		return Profile{}, err
	}
	if err := validateAnalysis(current); err != nil {
		return Profile{}, err
	}
	if current.Schema != AnalysisSchema {
		return Profile{}, fmt.Errorf("migration needs a %s analysis, not %s", AnalysisSchema, current.Schema)
	}
	legacyCompiled, err := Compile(legacy, legacyAnalysis)
	if err != nil {
		return Profile{}, fmt.Errorf("legacy profile: %w", err)
	}
	if legacyAnalysis.Source != current.Source || legacyAnalysis.LayoutSHA256 != current.LayoutSHA256 {
		return Profile{}, errors.New("the legacy profile was made for a different model structure than the current analysis; it cannot be migrated")
	}
	currentRegions := make(map[string]SemanticRegion, len(current.Regions))
	for _, r := range current.Regions {
		currentRegions[r.ID] = r
	}
	for _, r := range legacyAnalysis.Regions {
		c, ok := currentRegions[r.ID]
		if !ok || !equalSorted(c.Modules, r.Modules) || !equalSorted(c.Files, r.Files) {
			return Profile{}, fmt.Errorf("region %q differs between the legacy and the current analysis; the legacy profile cannot be migrated", r.ID)
		}
	}
	migrated, err := NewDefaultProfile(current, legacy.Objective)
	if err != nil {
		return Profile{}, err
	}
	canonical, err := canonicalRecipe(current)
	if err != nil {
		return Profile{}, err
	}
	redundant := map[string]bool{}
	for _, g := range current.Groups {
		if legacy.Preservation[g.Region].Mode != PreservationPinned {
			continue
		}
		required, _, err := requiredGroup(canonical, g)
		if err != nil {
			return Profile{}, err
		}
		if required {
			redundant[g.Region] = true
			continue
		}
		migrated.Groups[g.ID] = GroupChoice{Mode: GroupOverride, Policy: home.PolicySourcePrecision}
	}
	migrated.MigratedFrom = &Migration{FromProfileID: legacy.ID(), FromSchema: legacy.Schema, FromAnalysisSHA256: legacyAnalysis.SHA256()}
	for region := range redundant {
		migrated.MigratedFrom.RedundantPins = append(migrated.MigratedFrom.RedundantPins, region)
	}
	sort.Strings(migrated.MigratedFrom.RedundantPins)
	compiled, err := Compile(migrated, current)
	if err != nil {
		return Profile{}, fmt.Errorf("migrated profile: %w", err)
	}
	if !samePreservedModules(legacyCompiled.Recipe, compiled.Recipe, current) {
		return Profile{}, errors.New("migration would change which modules are preserved; refusing to reinterpret the legacy profile")
	}
	return migrated, nil
}

// samePreservedModules reports whether two recipes preserve exactly the same
// declared modules and carried files of the analysis.
func samePreservedModules(a, b home.Recipe, analysis Analysis) bool {
	for _, r := range analysis.Regions {
		for _, m := range r.Modules {
			if preservedBy(a, m, home.ScopeBackbone) != preservedBy(b, m, home.ScopeBackbone) {
				return false
			}
		}
		for _, f := range r.Files {
			if preservedBy(a, f, home.ScopeCarried) != preservedBy(b, f, home.ScopeCarried) {
				return false
			}
		}
	}
	return true
}
