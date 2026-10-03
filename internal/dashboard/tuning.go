package dashboard

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// The Tuning workspace: model-engineering intent. It edits and persists a
// versioned layer-wise profile (an objective and, for each stable group of the
// backend-analyzed model, AUTO or one named policy the optimizer executes) and
// hands that exact saved profile to Forge. It never runs, sequences or
// inspects the optimizer: the groups, the profile store, the recipe compiler
// and the build are the typed tuning authority's. A stored legacy coarse
// profile is shown as its exact layer-wise equivalent. Generated module
// mappings and the recipe stay in Details/Evidence, and every impact figure
// states whether it was measured, estimated or not checked.

// Tuning is the typed tuning authority the dashboard edits through
// (internal/tuning over HACHIDORI_HOME, hosted by the application).
type Tuning interface {
	// Analysis is the backend's deterministic semantic-region analysis of the
	// exact catalog source. It fails when the source cannot be analyzed.
	Analysis(source string) (tuning.Analysis, error)
	// Profiles lists the saved profiles of one source, newest first.
	Profiles(source string) ([]tuning.Profile, error)
	// LoadProfile returns a saved profile and the analysis it is bound to.
	LoadProfile(id string) (tuning.Profile, tuning.Analysis, error)
	// SaveProfile persists one immutable, versioned profile with its analysis.
	SaveProfile(tuning.Profile, tuning.Analysis) error
	// Impact projects what the recorded evidence (and a documented estimator)
	// say about the exact profile. It is advisory: every value carries its
	// state, and the dashboard shows nothing it does not state.
	Impact(tuning.Profile, tuning.Analysis, tuning.Compilation) (TuningImpact, error)
	// Candidate reports the candidate built from the exact profile (its
	// applied per-group evidence and measured figures) beside the accepted
	// baseline. A figure without a record is NOT_CHECKED, never estimated.
	Candidate(tuning.Profile, tuning.Compilation) (TuningCandidate, error)
}

// ImpactValue is one projected consequence of a profile. State is Measured
// (directly observed from compatible evidence), Estimated (derived by a
// documented estimator) or NotChecked (no valid evidence). Basis names the
// evidence or the estimator.
type ImpactValue struct {
	State        SemanticState
	Value, Basis string
}

// normalized is the only way a value reaches the page: a value whose state is
// not Measured or Estimated, or that has no value or no basis, is NOT_CHECKED
// and shows no number. An unknown state is never promoted to a measurement.
func (v ImpactValue) normalized() ImpactValue {
	switch v.State {
	case Measured, Estimated:
		if strings.TrimSpace(v.Value) != "" && strings.TrimSpace(v.Basis) != "" {
			return v
		}
	}
	return ImpactValue{State: NotChecked, Basis: v.Basis}
}

// TuningImpact is the size, memory, latency and fidelity of a profile.
type TuningImpact struct {
	Size, Memory, Latency, Fidelity ImpactValue
	// MemoryUsage is the device-memory figure behind Memory: measured device
	// memory of a probe, or a lower bound (the weights) when nothing was
	// measured. Zero bytes means no figure exists.
	MemoryUsage tuning.Usage
}

// TuningObjectives is the objective vocabulary. The objective is recorded
// intent and part of the profile's identity; the per-region choices decide
// what a build preserves.
var TuningObjectives = []string{"balanced", "maximum-fidelity", "minimum-size", "lowest-latency"}

var objectiveLabels = map[string]string{
	"balanced":         "Balanced",
	"maximum-fidelity": "Maximum fidelity",
	"minimum-size":     "Smallest size",
	"lowest-latency":   "Lowest latency",
}

// regionCopy is the operator vocabulary of the semantic regions. The region
// set, membership and counts are the backend analysis'; an analysis that names
// a region without copy here is shown under its identifier.
var regionCopy = map[string]struct{ Label, Purpose string }{
	tuning.RegionOutputEmbeddings:     {"Output embeddings", "The rows the joint head reads as option vectors."},
	tuning.RegionLinearAttentionDecay: {"Linear-attention decay gate", "A small, numerically sensitive recurrent-state gate."},
	tuning.RegionLinearAttentionBeta:  {"Linear-attention beta gate", "A small, numerically sensitive recurrent-state gate."},
	tuning.RegionLinearAttention:      {"Linear-attention projections", "The remaining linear-attention backbone projections."},
	tuning.RegionFullAttention:        {"Full-attention projections", "The full-attention backbone projections."},
	tuning.RegionFeedForward:          {"Feed-forward projections", "The feed-forward backbone projections."},
	tuning.RegionVision:               {"Vision tower", "Image-input projections."},
	tuning.RegionJointSchemaHead:      {"Joint schema head", "The decision-specific scoring head."},
}

// TuningSourceOption is a catalog source a profile can be tuned for.
type TuningSourceOption struct {
	ID, Revision string
	Materialized bool
}

// TuningRegionRow is one semantic region: the saved choice and what the
// compiled profile does with it. Modules and Files are counts; the exact
// names are evidence.
type TuningRegionRow struct {
	ID, Label, Purpose string
	Modules, Files     int
	Pinned             bool
	// Recommended marks the region an advisory recommendation proposes to pin.
	Recommended bool
	// Effective is Preserved when the compiled profile keeps the whole region
	// at source precision (pinned, or already preserved by the canonical
	// policy under Auto) and Auto when Auto leaves it quantized.
	Effective SemanticState
	Note      string
}

// TuningImpactRow is one labelled, normalized impact value.
type TuningImpactRow struct {
	Key, Label string
	ImpactValue
}

// TuningProfileRow is one saved profile of the source. Overrides counts the
// groups a layer-wise profile overrides, or the regions a legacy profile pins.
type TuningProfileRow struct {
	ID, Objective string
	Overrides     int
	Legacy        bool
	Current       bool
}

// TuningEvidenceView is the Details/Evidence projection of the compiled
// profile: the generated mappings and recipe identity. It is never an input.
type TuningEvidenceView struct {
	Recipe, RecipeSHA256, CompilerVersion string
	// AutoPolicy and PlanSHA256 identify the resolved layer-wise plan: the
	// complete effective policy, AUTO resolutions included. Empty for a legacy
	// profile.
	AutoPolicy, PlanSHA256 string
	Plan                   []home.TuningGroupPlan
	Regions                []tuning.RegionMapping
	Preserved              []tuning.PreservedMapping
}

// TuningView is the Tuning workspace view model.
type TuningView struct {
	Sources    []TuningSourceOption
	Source     string
	Repo       string
	Revision   string
	Err        string // the analysis or the selected profile is unavailable
	Objectives []string
	Objective  string

	// Identity of exactly what is shown. Saved is false for the initial
	// all-Auto profile that has not been saved: ProfileID is then the
	// identity it would be saved under.
	AnalyzerVersion, AnalysisSHA256, CompilerVersion string
	ProfileSchema, ProfileID                         string
	Saved                                            bool

	Regions []TuningRegionRow
	// Layerwise is true for the layer-wise editor; the shown profile is always
	// layer-wise, a legacy one appears as Legacy's equivalent.
	Layerwise bool
	Policies  []TuningPolicyOption
	Families  []TuningFamilyRow
	Summary   TuningSummary
	Legacy    *TuningLegacyView
	// Changes lists the groups whose policy differs from the baseline's, and
	// ChangesBasis names the baseline (or says why none can be stated).
	Changes      []TuningChangeRow
	ChangesBasis string
	ChangesNote  string
	Candidate    *TuningCandidateView
	Bulk         TuningBulkForm
	Notice       string // the outcome of the last editor operation
	NoticeBad    bool
	// PreservedRegions counts the regions the compiled profile keeps at
	// source precision (pinned or preserved by the canonical policy).
	PreservedRegions int
	Impact           []TuningImpactRow
	// Envelope is the target device and memory budget; Fit places the
	// profile's memory figure against it.
	Envelope    EnvelopeView
	Fit         FitView
	memoryUsage tuning.Usage
	// AdvancedOpen opens the semantic preservation editor: only when the
	// operator has a reason to intervene (a recommendation to review or an
	// out-of-envelope plan).
	AdvancedOpen bool
	// AdvancedDisclosure holds the semantic preservation editor.
	AdvancedDisclosure DisclosureProjection
	Profiles           []TuningProfileRow
	Evidence           *TuningEvidenceView
	// EvidenceDisclosure is the shared Details/Evidence primitive that holds
	// every generated mapping.
	EvidenceDisclosure DisclosureProjection

	// Context is the Experiment/Evidence context handed to this page, nil
	// when none was.
	Context *TuningContextView
	// Compare is the candidate A/B comparison handed to this page, nil when
	// none was. It is separate from Context: it has no source baseline.
	Compare *TuningCompareView
}

// TuningContextView is an Experiment/Evidence context beside the profile it is
// bound to. Refused is the reason a context that is not bound to exactly this
// source, profile, dataset, questions and evidence is shown but informs
// nothing. Regression and Recommendation exist only for a bound, compatible
// comparison; NoRecommendation says why there is none, which is a normal state.
type TuningContextView struct {
	Refused          string
	Context          tuning.EvidenceContext
	Handoff          url.Values
	Regression       *tuning.MeasuredRegression
	Recommendation   *tuning.Recommendation
	NoRecommendation string
}

func objectiveLabel(id string) string {
	if l, ok := objectiveLabels[id]; ok {
		return l
	}
	return id
}

// overrideCount is the number of groups a layer-wise profile overrides, or of
// regions a legacy profile pins.
func overrideCount(p tuning.Profile) int {
	n := 0
	for _, c := range p.Preservation {
		if c.Mode == tuning.PreservationPinned {
			n++
		}
	}
	for _, c := range p.Groups {
		if c.Mode == tuning.GroupOverride {
			n++
		}
	}
	return n
}

// tuningSources lists the catalog models that support variants.
func tuningSources(mv *ModelsView) []TuningSourceOption {
	var out []TuningSourceOption
	for _, m := range mv.Inventory.Models {
		if setup.SupportsVariants(home.ModelManifest{Provider: m.Provider}) {
			out = append(out, TuningSourceOption{ID: m.ID, Revision: m.Revision, Materialized: m.Materialized})
		}
	}
	return out
}

func (d *Dashboard) tuningPage(w http.ResponseWriter, r *http.Request) {
	v := d.view("Tuning", "tuning")
	v.Live = false
	mv := d.modelsView(v)
	v.Models = mv
	q := r.URL.Query()
	source, profileID := q.Get("source"), q.Get("profile")
	var (
		ctx    tuning.EvidenceContext
		ctxErr error
	)
	var (
		cmp    tuning.CandidateComparison
		cmpErr error
	)
	compared := q.Get(qCmpA) != "" || q.Get(qCmpB) != ""
	handedOver := !compared && q.Get(qCand) != ""
	switch {
	case compared:
		// The profile shown is the one of candidate B, as in "Open B in Tuning".
		if cmp, cmpErr = d.resolveCandidateComparison(q); cmpErr == nil {
			source, profileID = cmp.B.ModelID, cmp.B.ProfileID
		}
	case handedOver:
		ctx, ctxErr = d.resolveTuningContext(q)
		if ctxErr == nil {
			profileID = ctx.ProfileID
		}
	}
	v.Tuning = d.tuningView(mv, source, profileID, nil)
	d.finishTuning(&v)
	defer func() {
		v.Tuning.AdvancedDisclosure = DisclosureProjection{ID: "tuning-advanced", Label: "Layer-wise policy", Open: v.Tuning.AdvancedOpen}
		d.renderView(w, "tuning", v)
	}()
	switch {
	case compared:
		v.Tuning.Compare = d.tuningCompareView(v.Tuning, cmp, cmpErr)
	case handedOver:
		v.Tuning.Context = d.tuningContextView(v.Tuning, q, ctx, ctxErr)
	}
}

// finishTuning places the shown profile against the device envelope and
// builds the editor's disclosure. Callers that edit open it themselves.
func (d *Dashboard) finishTuning(v *view) {
	v.Tuning.Envelope = d.envelopeOf(*v)
	if len(v.Tuning.Impact) > 1 {
		v.Tuning.Fit = fitOfTuning(v.Tuning)
		v.Tuning.AdvancedOpen = v.Tuning.AdvancedOpen || v.Tuning.Fit.Blocked()
	}
	v.Tuning.AdvancedDisclosure = DisclosureProjection{ID: "tuning-advanced", Label: "Layer-wise policy", Open: v.Tuning.AdvancedOpen}
}

// shows reports whether the page shows the profile with the given identity:
// the saved profile itself, or the layer-wise equivalent of that legacy one.
func (tv *TuningView) shows(profileID string) bool {
	return tv.Saved && tv.ProfileID == profileID || tv.Legacy != nil && tv.Legacy.ID == profileID
}

// tuningContextView binds a resolved context to the profile the page shows and
// derives the advisory recommendation from that exact pair.
func (d *Dashboard) tuningContextView(tv *TuningView, q url.Values, ctx tuning.EvidenceContext, err error) *TuningContextView {
	cv := &TuningContextView{}
	if err != nil {
		cv.Refused = err.Error()
		return cv
	}
	cv.Context, cv.Regression, cv.Handoff = ctx, ctx.Regression, handoffValues(q)
	if !tv.shows(ctx.ProfileID) {
		cv.Refused = "The profile named by the evidence context cannot be shown, so the context informs nothing."
		return cv
	}
	profile, analysis, err := d.cfg.Tuning.LoadProfile(ctx.ProfileID)
	if err != nil {
		cv.Refused = "The profile named by the evidence context cannot be loaded: " + err.Error()
		return cv
	}
	if err := ctx.Check(profile); err != nil {
		cv.Refused = err.Error()
		return cv
	}
	cv.Recommendation, cv.NoRecommendation = tuning.Recommend(ctx, profile, analysis)
	if cv.Recommendation != nil {
		tv.AdvancedOpen = true
		for i := range tv.Regions {
			tv.Regions[i].Recommended = tv.Regions[i].ID == cv.Recommendation.RegionID
		}
		for i := range tv.Families {
			tv.Families[i].Recommended = tv.Families[i].ID == cv.Recommendation.RegionID
		}
	}
	return cv
}

// tuningView projects the tuning authority's state for one source and,
// optionally, one saved profile or an unsaved draft. Without either it shows
// the initial all-AUTO profile, labelled as unsaved: nothing is invented or
// remembered. A stored legacy coarse profile is shown as its exact
// layer-wise equivalent (tuning.MigrateProfile), labelled as an upgrade.
func (d *Dashboard) tuningView(mv *ModelsView, source, profileID string, draft *tuning.Profile) *TuningView {
	tv := &TuningView{Sources: tuningSources(mv), Objectives: TuningObjectives, Objective: "balanced",
		EvidenceDisclosure: DisclosureProjection{ID: "tuning-evidence-details", Label: "Generated mappings and recipe"}}
	if source == "" && len(tv.Sources) > 0 {
		source = tv.Sources[0].ID
		for _, s := range tv.Sources {
			if s.Materialized {
				source = s.ID
				break
			}
		}
	}
	tv.Source = source
	for _, s := range tv.Sources {
		if s.ID == source {
			tv.Revision = s.Revision
		}
	}
	if source == "" {
		tv.Err = "No source model supports tuning."
		return tv
	}
	if !slices.ContainsFunc(tv.Sources, func(s TuningSourceOption) bool { return s.ID == source }) {
		tv.Err = fmt.Sprintf("Model %s cannot be tuned: only System One models have semantic regions.", source)
		return tv
	}
	t := d.cfg.Tuning
	analysis, err := t.Analysis(source)
	if err != nil {
		tv.Err = "Semantic analysis is unavailable: " + err.Error()
		return tv
	}
	tv.Repo = analysis.Source.Repo
	saved, err := t.Profiles(source)
	if err != nil {
		tv.Err = "Saved profiles are unavailable: " + err.Error()
	}
	isSaved := func(id string) bool {
		return slices.ContainsFunc(saved, func(p tuning.Profile) bool { return p.ID() == id })
	}
	var profile tuning.Profile
	switch {
	case draft != nil:
		profile, tv.Saved = *draft, isSaved(draft.ID())
	case profileID != "":
		p, a, err := t.LoadProfile(profileID)
		switch {
		case err != nil:
			tv.Err = "Saved profile " + short12(profileID) + " cannot be used: " + err.Error()
		case p.Schema == tuning.ProfileSchemaV1:
			migrated, merr := tuning.MigrateProfile(p, a, analysis)
			if merr != nil {
				tv.Err = "Legacy profile " + short12(profileID) + " cannot be upgraded to a layer-wise profile: " + merr.Error()
				break
			}
			profile, tv.Saved = migrated, isSaved(migrated.ID())
			tv.Legacy = &TuningLegacyView{ID: p.ID(), Objective: p.Objective, UpgradedID: migrated.ID(), UpgradedSaved: tv.Saved}
			for _, region := range analysis.Regions {
				if p.Preservation[region.ID].Mode == tuning.PreservationPinned {
					tv.Legacy.Pinned = append(tv.Legacy.Pinned, region.ID)
				}
			}
			if migrated.MigratedFrom != nil {
				tv.Legacy.Redundant = migrated.MigratedFrom.RedundantPins
			}
		case p.Source != analysis.Source || a.SHA256() != analysis.SHA256():
			tv.Err = "Saved profile " + short12(profileID) + " is not bound to the current analysis of " + source + "."
		default:
			profile, tv.Saved = p, true
		}
	default:
		for _, p := range saved {
			if p.Schema == tuning.ProfileSchema && p.Source == analysis.Source && p.AnalysisSHA256 == analysis.SHA256() {
				profile, tv.Saved = p, true
				break
			}
		}
	}
	if profile.Schema == "" {
		p, err := tuning.NewDefaultProfile(analysis, tv.Objective)
		if err != nil {
			tv.Err = "The initial profile cannot be created: " + err.Error()
			return tv
		}
		profile, tv.Saved = p, false
	}
	compiled, err := tuning.Compile(profile, analysis)
	if err != nil {
		tv.Err = "The profile cannot be compiled: " + err.Error()
		return tv
	}
	tv.Layerwise = true
	tv.Policies = tuningPolicies()
	tv.Objective = profile.Objective
	tv.AnalyzerVersion, tv.AnalysisSHA256, tv.CompilerVersion = analysis.AnalyzerVersion, analysis.SHA256(), profile.CompilerVersion
	tv.ProfileSchema, tv.ProfileID = profile.Schema, profile.ID()
	if !slices.Contains(tv.Objectives, profile.Objective) {
		tv.Objectives = append(slices.Clone(tv.Objectives), profile.Objective)
	}

	mapping := map[string]tuning.RegionMapping{}
	for _, m := range compiled.Evidence.Regions {
		mapping[m.RegionID] = m
	}
	for _, region := range analysis.Regions {
		text, ok := regionCopy[region.ID]
		if !ok {
			text.Label, text.Purpose = region.ID, region.Description
		}
		m := mapping[region.ID]
		row := TuningRegionRow{ID: region.ID, Label: text.Label, Purpose: text.Purpose, Modules: len(region.Modules), Files: len(region.Files),
			Pinned: m.Mode == tuning.PreservationPinned || m.Mode == tuning.PreservationOverride, Effective: Auto}
		switch {
		case m.Preserved && row.Pinned:
			row.Effective, row.Note = Preserved, "Every tunable group is overridden to source precision."
		case m.Preserved:
			row.Effective, row.Note = Preserved, "Required: the canonical policy preserves this region."
		case m.Mode == tuning.PreservationMixed || row.Pinned:
			row.Note = "Some groups are overridden; the rest follow AUTO."
		default:
			row.Note = "AUTO: the canonical policy quantizes this region."
		}
		if row.Effective == Preserved {
			tv.PreservedRegions++
		}
		tv.Regions = append(tv.Regions, row)
	}
	tv.Families, tv.Summary = layerwiseRows(analysis, profile, *compiled.Plan)
	tv.Bulk = TuningBulkForm{Policy: home.PolicySourcePrecision, Blocks: blockCount(analysis)}
	for _, p := range saved {
		tv.Profiles = append(tv.Profiles, TuningProfileRow{ID: p.ID(), Objective: p.Objective, Overrides: overrideCount(p),
			Legacy: p.Schema == tuning.ProfileSchemaV1, Current: tv.Saved && p.ID() == profile.ID() || tv.Legacy != nil && p.ID() == tv.Legacy.ID})
	}
	tv.Impact, tv.memoryUsage = d.tuningImpact(profile, analysis, compiled)
	tv.Evidence = &TuningEvidenceView{Recipe: compiled.Recipe.Name, RecipeSHA256: compiled.Recipe.SHA256(), CompilerVersion: compiled.Evidence.CompilerVersion,
		AutoPolicy: compiled.Plan.AutoPolicy, PlanSHA256: compiled.Plan.SHA256(), Plan: compiled.Plan.Groups,
		Regions: compiled.Evidence.Regions, Preserved: compiled.Evidence.Preserved}

	cand, err := t.Candidate(profile, compiled)
	if err != nil {
		cand = TuningCandidate{}
		tv.ChangesNote = "The accepted baseline and candidate evidence could not be read: " + err.Error()
	}
	tv.Candidate = candidateView(cand, *compiled.Plan)
	if canonical, cerr := canonicalPlan(analysis, profile.Objective); cerr == nil {
		if policies, basis := baselinePolicies(cand, canonical); policies != nil {
			tv.Changes, tv.ChangesBasis = planChanges(*compiled.Plan, policies), basis
		} else {
			tv.ChangesNote = basis + ", so the change from it is NOT_CHECKED."
		}
	}
	return tv
}

// canonicalPlan is the all-AUTO resolution: the policy of the canonical recipe.
func canonicalPlan(analysis tuning.Analysis, objective string) (home.TuningPlan, error) {
	p, err := tuning.NewDefaultProfile(analysis, objective)
	if err != nil {
		return home.TuningPlan{}, err
	}
	c, err := tuning.Compile(p, analysis)
	if err != nil {
		return home.TuningPlan{}, err
	}
	return *c.Plan, nil
}

// blockCount is the number of transformer blocks the analysis' groups name.
func blockCount(a tuning.Analysis) int {
	n := 0
	for _, g := range a.Groups {
		n = max(n, g.Layer+1)
	}
	return n
}

// tuningImpact normalizes the authority's projection. An authority that
// cannot say anything leaves every value NOT_CHECKED.
func (d *Dashboard) tuningImpact(p tuning.Profile, a tuning.Analysis, c tuning.Compilation) ([]TuningImpactRow, tuning.Usage) {
	imp, err := d.cfg.Tuning.Impact(p, a, c)
	usage := imp.MemoryUsage
	if err != nil {
		usage = tuning.Usage{}
	}
	rows := []TuningImpactRow{
		{Key: "size", Label: "Model size", ImpactValue: imp.Size},
		{Key: "memory", Label: "Memory (VRAM/RAM)", ImpactValue: imp.Memory},
		{Key: "latency", Label: "Latency", ImpactValue: imp.Latency},
		{Key: "fidelity", Label: "Fidelity", ImpactValue: imp.Fidelity},
	}
	for i := range rows {
		if err != nil {
			rows[i].ImpactValue = ImpactValue{Basis: "Evidence could not be read: " + err.Error()}
		}
		rows[i].ImpactValue = rows[i].ImpactValue.normalized()
	}
	// A memory figure is used only when its value is stated: a NOT_CHECKED
	// memory row has no figure, whatever the authority returned.
	if rows[1].State == NotChecked {
		usage = tuning.Usage{}
	}
	return rows, usage
}

// fitOfTuning places the shown profile's memory figure against the envelope.
func fitOfTuning(tv *TuningView) FitView {
	mem := tv.Impact[1]
	return fitOf(tv.Envelope.Envelope, tv.memoryUsage, mem.State, mem.Basis)
}

func short12(s string) string { return s[:min(len(s), 12)] }

// tuningProfile builds the layer-wise profile the form describes from the
// backend analysis: every group is AUTO unless its field names a policy. The
// form names groups by their stable identifiers only. A form that carries a
// legacy profile's identity records that lineage in the profile.
func (d *Dashboard) tuningProfile(r *http.Request) (string, tuning.Profile, tuning.Analysis, error) {
	source := strings.TrimSpace(r.PostFormValue("source"))
	if source == "" {
		return "", tuning.Profile{}, tuning.Analysis{}, errors.New("choose a source model")
	}
	objective := strings.TrimSpace(r.PostFormValue("objective"))
	if !slices.Contains(TuningObjectives, objective) {
		return source, tuning.Profile{}, tuning.Analysis{}, fmt.Errorf("unknown tuning objective %q", objective)
	}
	analysis, err := d.cfg.Tuning.Analysis(source)
	if err != nil {
		return source, tuning.Profile{}, tuning.Analysis{}, err
	}
	profile, err := tuning.NewDefaultProfile(analysis, objective)
	if err != nil {
		return source, profile, analysis, err
	}
	if legacyID := strings.TrimSpace(r.PostFormValue("legacy_profile")); legacyID != "" {
		legacy, legacyAnalysis, err := d.cfg.Tuning.LoadProfile(legacyID)
		if err != nil {
			return source, profile, analysis, fmt.Errorf("the legacy profile %s cannot be loaded: %w", short12(legacyID), err)
		}
		migrated, err := tuning.MigrateProfile(legacy, legacyAnalysis, analysis)
		if err != nil {
			return source, profile, analysis, fmt.Errorf("the legacy profile %s cannot be upgraded: %w", short12(legacyID), err)
		}
		profile.MigratedFrom = migrated.MigratedFrom
	}
	if profile, err = groupPolicyFields(r, profile, analysis); err != nil {
		return source, profile, analysis, err
	}
	return source, profile, analysis, nil
}

func tuningLocation(source, profileID string) string {
	q := url.Values{}
	if source != "" {
		q.Set("source", source)
	}
	if profileID != "" {
		q.Set("profile", profileID)
	}
	if len(q) == 0 {
		return "/tuning"
	}
	return "/tuning?" + q.Encode()
}

// tuningSave persists the described profile as one immutable versioned
// document and shows it. Saving builds nothing.
func (d *Dashboard) tuningSave(w http.ResponseWriter, r *http.Request) {
	if r.PostFormValue("accept_recommendation") != "" {
		d.tuningAccept(w, r)
		return
	}
	source, profile, analysis, err := d.tuningProfile(r)
	if err == nil {
		err = d.cfg.Tuning.SaveProfile(profile, analysis)
	}
	if err != nil {
		d.remember("save tuning profile", err, "")
		http.Redirect(w, r, tuningLocation(source, ""), http.StatusSeeOther)
		return
	}
	d.remember("save tuning profile "+short12(profile.ID()), nil, "saved as profile "+profile.ID()+". Nothing was built")
	http.Redirect(w, r, tuningLocation(source, profile.ID()), http.StatusSeeOther)
}

// tuningBuild saves the described profile, reads it back by its identity and
// selects exactly that saved profile in Forge, whose one build/evaluate
// operation builds, evaluates and records the candidate. Tuning starts no
// build of its own; Apply stays explicit in Forge.
func (d *Dashboard) tuningBuild(w http.ResponseWriter, r *http.Request) {
	source, profile, analysis, err := d.tuningProfile(r)
	if err == nil {
		err = d.cfg.Tuning.SaveProfile(profile, analysis)
	}
	var saved tuning.Profile
	savedID := ""
	if err == nil {
		if saved, _, err = d.cfg.Tuning.LoadProfile(profile.ID()); err == nil {
			if saved.ID() != profile.ID() {
				err = errors.New("the saved profile does not match the requested identity")
			} else {
				savedID = saved.ID()
			}
		}
	}
	if err != nil {
		d.remember("build candidate "+source, err, "")
		http.Redirect(w, r, tuningLocation(source, savedID), http.StatusSeeOther)
		return
	}
	d.remember("build candidate "+source, nil, "saved profile "+saved.ID()+" and selected it in Forge. Build candidate there builds and evaluates it in one operation; nothing was built and nothing is applied")
	q := url.Values{"source": {source}, "profile": {tuningProfilePrefix + saved.ID()}}
	http.Redirect(w, r, "/forge?"+q.Encode()+"#forge-intent-form", http.StatusSeeOther)
}

// The Experiment/Evidence context travels in the query string as identities
// only. The page never trusts them: it reopens the stored evidence, rebinds it
// and refuses the context unless every identity matches what it recomputed.
const (
	qBase, qBaseSHA = "base", "base_sha"
	qCand, qCandSHA = "cand", "cand_sha"
	qDataset        = "ds"
	qQuestions      = "qs"
	qVariant        = "variant"
)

var handoffKeys = []string{"source", "profile", qVariant, qBase, qBaseSHA, qCand, qCandSHA, qDataset, qQuestions}

func handoffValues(q url.Values) url.Values {
	out := url.Values{}
	for _, k := range handoffKeys {
		if v := q.Get(k); v != "" {
			out.Set(k, v)
		}
	}
	return out
}

// variantProfile resolves a candidate variant to the profile its build
// provenance names, through the tuning authority over HACHIDORI_HOME.
func (d *Dashboard) variantProfile(source, variant string) (string, error) {
	if d.cfg.Status == nil {
		return "", errors.New("the runtime home is unavailable")
	}
	root := d.cfg.Status().Runtime.Home
	if root == "" {
		return "", errors.New("the runtime home is unavailable")
	}
	return tuning.VariantProfile(home.Home{Root: root}, source, variant)
}

// tuningContextOf is the exact context of stored history evidence: the
// candidate entry and, for a comparison, the baseline entry. It refuses what
// cannot be compared as equivalent. The returned values are the identities a
// link carries.
func (d *Dashboard) tuningContextOf(baseID, candID string) (tuning.EvidenceContext, url.Values, error) {
	if d.hist == nil {
		return tuning.EvidenceContext{}, nil, errNoHistory
	}
	cand, candSHA, err := d.hist.Open(candID)
	if err != nil {
		return tuning.EvidenceContext{}, nil, fmt.Errorf("candidate evidence: %w", err)
	}
	ctx := tuning.EvidenceContext{DatasetSHA256: cand.DatasetSHA256, QuestionsSHA256: eval.QuestionIdentitiesSHA256(cand)}
	var ci eval.RunIdentity
	if baseID == "" {
		if ci, err = eval.RunIdentityOf(cand, candSHA); err != nil {
			return tuning.EvidenceContext{}, nil, fmt.Errorf("candidate evidence: %w", err)
		}
	} else {
		base, baseSHA, err := d.hist.Open(baseID)
		if err != nil {
			return tuning.EvidenceContext{}, nil, fmt.Errorf("baseline evidence: %w", err)
		}
		binding, err := eval.Bind(base, cand, baseSHA, candSHA)
		if err != nil {
			return tuning.EvidenceContext{}, nil, err
		}
		bi := binding.Baseline
		ci = binding.Candidate
		ctx.Baseline = &tuning.EvidenceRun{EvidenceSHA256: bi.EvidenceSHA256, ModelID: bi.ModelID, Revision: bi.Revision, VariantID: bi.VariantID}
		ctx.Regression = regressionOf(binding.Comparison)
	}
	ctx.Candidate = tuning.EvidenceRun{EvidenceSHA256: ci.EvidenceSHA256, ModelID: ci.ModelID, Revision: ci.Revision, VariantID: ci.VariantID}
	if ci.VariantID == "" {
		return tuning.EvidenceContext{}, nil, errors.New("candidate evidence does not name an executed variant")
	}
	if ctx.ProfileID, err = d.variantProfile(ci.ModelID, ci.VariantID); err != nil {
		return tuning.EvidenceContext{}, nil, err
	}
	vals := url.Values{"source": {ci.ModelID}, "profile": {ctx.ProfileID}, qVariant: {ci.VariantID}, qCand: {candID}, qCandSHA: {candSHA},
		qDataset: {ctx.DatasetSHA256}, qQuestions: {ctx.QuestionsSHA256}}
	if baseID != "" {
		vals.Set(qBase, baseID)
		vals.Set(qBaseSHA, ctx.Baseline.EvidenceSHA256)
	}
	return ctx, vals, nil
}

// regressionOf copies the measured accuracy regression out of a compatible
// comparison. It is nil when overall accuracy did not fall or no aligned
// question's accuracy fell.
func regressionOf(c eval.Comparison) *tuning.MeasuredRegression {
	if c.Status != eval.CompareCompatible || c.Aggregate == nil || !(c.Aggregate.Accuracy.Diff < 0) {
		return nil
	}
	reg := &tuning.MeasuredRegression{Cases: c.Aggregate.Cases.B, BaselineAccuracy: c.Aggregate.Accuracy.A, CandidateAccuracy: c.Aggregate.Accuracy.B,
		BaselineErrors: c.Aggregate.RequestErrors.A, CandidateErrors: c.Aggregate.RequestErrors.B}
	for _, qd := range c.Questions {
		if qd.Accuracy.Diff < 0 {
			reg.Questions = append(reg.Questions, tuning.QuestionRegression{Question: qd.ID, N: qd.N.B, Baseline: qd.Accuracy.A, Candidate: qd.Accuracy.B})
		}
	}
	if len(reg.Questions) == 0 {
		return nil
	}
	return reg
}

// resolveTuningContext rebuilds the context the query names from stored
// evidence and refuses it unless every identity it carries (source, profile,
// variant, evidence, dataset and questions) equals the recomputed one.
func (d *Dashboard) resolveTuningContext(q url.Values) (tuning.EvidenceContext, error) {
	ctx, want, err := d.tuningContextOf(q.Get(qBase), q.Get(qCand))
	if err != nil {
		return tuning.EvidenceContext{}, err
	}
	for _, k := range handoffKeys {
		if q.Get(k) != want.Get(k) {
			return tuning.EvidenceContext{}, fmt.Errorf("the evidence context does not match the stored evidence: %s is %q, not %q", k, q.Get(k), want.Get(k))
		}
	}
	return ctx, nil
}

// tuningAccept is the explicit operator action on a recommendation. It
// rebuilds the context and the recommendation, applies exactly the
// recommendation the operator was shown, and saves the result as a new,
// distinct profile. Evidence is never written and nothing is built.
func (d *Dashboard) tuningAccept(w http.ResponseWriter, r *http.Request) {
	q := url.Values{}
	for _, k := range handoffKeys {
		q.Set(k, r.PostFormValue(k))
	}
	source, region := q.Get("source"), strings.TrimSpace(r.PostFormValue("accept_recommendation"))
	fail := func(err error) {
		d.remember("accept tuning recommendation", err, "")
		http.Redirect(w, r, "/tuning?"+handoffValues(q).Encode(), http.StatusSeeOther)
	}
	ctx, err := d.resolveTuningContext(q)
	if err != nil {
		fail(err)
		return
	}
	profile, analysis, err := d.cfg.Tuning.LoadProfile(ctx.ProfileID)
	if err != nil {
		fail(err)
		return
	}
	rec, why := tuning.Recommend(ctx, profile, analysis)
	switch {
	case rec == nil:
		fail(errors.New("there is no recommendation to accept: " + why))
		return
	case rec.RegionID != region:
		fail(fmt.Errorf("the recommendation is to preserve %s, not %q", rec.RegionID, region))
		return
	}
	next, err := tuning.Accept(*rec, profile, analysis)
	if err == nil && next.Schema == tuning.ProfileSchemaV1 {
		// A legacy profile's accepted recommendation is saved as its exact
		// layer-wise equivalent: no new legacy profile is ever created.
		var current tuning.Analysis
		if current, err = d.cfg.Tuning.Analysis(source); err == nil {
			if next, err = tuning.MigrateProfile(next, analysis, current); err == nil {
				next.MigratedFrom.FromProfileID, next.MigratedFrom.FromAnalysisSHA256 = profile.ID(), analysis.SHA256()
				analysis = current
			}
		}
	}
	if err == nil {
		err = d.cfg.Tuning.SaveProfile(next, analysis)
	}
	if err != nil {
		fail(err)
		return
	}
	d.remember("accept tuning recommendation "+short12(next.ID()), nil, "saved as new profile "+next.ID()+" from profile "+profile.ID()+" with "+rec.RegionID+" preserved. The evidence is unchanged. Nothing was built")
	http.Redirect(w, r, tuningLocation(source, next.ID()), http.StatusSeeOther)
}

// The candidate A/B comparison travels in the query string as identities only,
// like the Experiment/Evidence context, and is rebuilt and refused the same
// way. Candidate B's profile is the one the page shows.
const (
	qCmpA, qCmpASHA, qCmpAVariant, qCmpAProfile = "cmp_a", "cmp_a_sha", "cmp_a_variant", "cmp_a_profile"
	qCmpB, qCmpBSHA, qCmpBVariant, qCmpBProfile = "cmp_b", "cmp_b_sha", "cmp_b_variant", "cmp_b_profile"
)

var candidateKeys = []string{"source", "profile", qCmpA, qCmpASHA, qCmpAVariant, qCmpAProfile, qCmpB, qCmpBSHA, qCmpBVariant, qCmpBProfile, qDataset, qQuestions}

// candidateComparisonOf is the exact comparison of two stored candidate
// evidence entries. It refuses what cannot be compared as equivalent. The
// returned values are the identities a link carries.
func (d *Dashboard) candidateComparisonOf(aID, bID string) (tuning.CandidateComparison, url.Values, error) {
	if d.hist == nil {
		return tuning.CandidateComparison{}, nil, errNoHistory
	}
	ra, aSHA, err := d.hist.Open(aID)
	if err != nil {
		return tuning.CandidateComparison{}, nil, fmt.Errorf("candidate A evidence: %w", err)
	}
	rb, bSHA, err := d.hist.Open(bID)
	if err != nil {
		return tuning.CandidateComparison{}, nil, fmt.Errorf("candidate B evidence: %w", err)
	}
	binding, err := eval.BindCandidates(ra, rb, aSHA, bSHA)
	if err != nil {
		return tuning.CandidateComparison{}, nil, err
	}
	run := func(id eval.RunIdentity) (tuning.CandidateRun, error) {
		profile, err := d.variantProfile(id.ModelID, id.VariantID)
		if err != nil {
			return tuning.CandidateRun{}, err
		}
		return tuning.CandidateRun{EvidenceRun: tuning.EvidenceRun{EvidenceSHA256: id.EvidenceSHA256, ModelID: id.ModelID, Revision: id.Revision, VariantID: id.VariantID},
			ProfileID: profile}, nil
	}
	cmp := measuredCandidateComparison(binding)
	if cmp.A, err = run(binding.A); err != nil {
		return tuning.CandidateComparison{}, nil, fmt.Errorf("candidate A: %w", err)
	}
	if cmp.B, err = run(binding.B); err != nil {
		return tuning.CandidateComparison{}, nil, fmt.Errorf("candidate B: %w", err)
	}
	vals := url.Values{"source": {binding.B.ModelID}, "profile": {cmp.B.ProfileID},
		qCmpA: {aID}, qCmpASHA: {aSHA}, qCmpAVariant: {binding.A.VariantID}, qCmpAProfile: {cmp.A.ProfileID},
		qCmpB: {bID}, qCmpBSHA: {bSHA}, qCmpBVariant: {binding.B.VariantID}, qCmpBProfile: {cmp.B.ProfileID},
		qDataset: {cmp.DatasetSHA256}, qQuestions: {cmp.QuestionsSHA256}}
	return cmp, vals, nil
}

// measuredCandidateComparison copies the measured quality and resource deltas
// out of a compatible binding. Resource figures the evidence does not record
// stay unavailable, never estimated. The caller binds the candidates to their
// profiles.
func measuredCandidateComparison(b eval.CandidateBinding) tuning.CandidateComparison {
	agg := b.Comparison.Aggregate
	md := func(x eval.Delta) tuning.MeasuredDelta { return tuning.MeasuredDelta{A: x.A, B: x.B, Diff: x.Diff} }
	c := tuning.CandidateComparison{DatasetSHA256: b.DatasetSHA256, QuestionsSHA256: b.QuestionsSHA256,
		Cases: agg.Cases.B, Accuracy: md(agg.Accuracy), ErrorsA: agg.RequestErrors.A, ErrorsB: agg.RequestErrors.B}
	for _, q := range b.Comparison.Questions {
		if q.Accuracy.Diff != 0 {
			c.Questions = append(c.Questions, tuning.QuestionDifference{Question: q.ID, NA: q.N.A, NB: q.N.B, Accuracy: md(q.Accuracy)})
		}
	}
	latency := func(prefix string, l eval.LatencyDelta) {
		for _, m := range []struct {
			key string
			d   eval.Delta
		}{{"p50", l.P50}, {"p95", l.P95}, {"mean", l.Mean}} {
			r := tuning.ResourceDelta{Key: prefix + "-" + m.key, Unit: "ms", Available: l.Available}
			if l.Available {
				r.MeasuredDelta = md(m.d)
			}
			c.Resources = append(c.Resources, r)
		}
	}
	latency("request", agg.RequestLatency)
	latency("inference", agg.ServerInference)
	return c
}

// resolveCandidateComparison rebuilds the comparison the query names from
// stored evidence and refuses it unless every identity it carries equals the
// recomputed one.
func (d *Dashboard) resolveCandidateComparison(q url.Values) (tuning.CandidateComparison, error) {
	cmp, want, err := d.candidateComparisonOf(q.Get(qCmpA), q.Get(qCmpB))
	if err != nil {
		return tuning.CandidateComparison{}, err
	}
	for _, k := range candidateKeys {
		if q.Get(k) != want.Get(k) {
			return tuning.CandidateComparison{}, fmt.Errorf("the candidate comparison does not match the stored evidence: %s is %q, not %q", k, q.Get(k), want.Get(k))
		}
	}
	return cmp, nil
}

// TuningDeltaRow is one compared figure of the two candidates. A, B and Diff
// are empty unless State is Measured; Diff is B minus A.
type TuningDeltaRow struct {
	Key, Label string
	State      SemanticState
	A, B, Diff string
	Basis      string
}

// TuningRegionChangeRow is one semantic region whose preservation differs
// between the two profiles.
type TuningRegionChangeRow struct {
	ID, Label, A, B string
}

// TuningCompareView is the candidate A/B comparison beside the profile page.
// Refused is the reason a comparison that is not bound to exactly these two
// candidates and profiles is shown but informs nothing. The comparison is
// descriptive: it carries no causal attribution, confidence or recommendation.
type TuningCompareView struct {
	Refused    string
	Comparison tuning.CandidateComparison
	Delta      tuning.ProfileDelta
	ProfileA   string // link to profile A in Tuning
	ProfileB   string
	Regions    []TuningRegionChangeRow
	Groups     []TuningGroupChangeRow
	Quality    []TuningDeltaRow
	Resources  []TuningDeltaRow
	Questions  []TuningDeltaRow
}

// TuningGroupChangeRow is one layer group whose choice differs between the
// two layer-wise profiles.
type TuningGroupChangeRow struct {
	ID, A, B string
}

func groupChoiceLabel(c tuning.GroupChoice) string {
	switch c.Mode {
	case tuning.GroupAuto:
		return "AUTO"
	case tuning.GroupOverride:
		return policyLabel(c.Policy)
	}
	return "Absent"
}

func preservationLabel(c tuning.PreservationChoice) string {
	switch c.Mode {
	case tuning.PreservationPinned:
		return "Preserved"
	case tuning.PreservationAuto:
		return "Auto"
	}
	return "Absent"
}

func signedCount(v int) string {
	if v > 0 {
		return "+" + strconv.Itoa(v)
	}
	return strconv.Itoa(v)
}

// tuningCompareView binds a resolved comparison to the exact profiles of both
// candidates and derives their semantic profile difference.
func (d *Dashboard) tuningCompareView(tv *TuningView, cmp tuning.CandidateComparison, err error) *TuningCompareView {
	cv := &TuningCompareView{}
	if err != nil {
		cv.Refused = err.Error()
		return cv
	}
	cv.Comparison = cmp
	if !tv.shows(cmp.B.ProfileID) {
		cv.Refused = "The profile of candidate B cannot be shown, so the comparison informs nothing."
		return cv
	}
	pa, _, err := d.cfg.Tuning.LoadProfile(cmp.A.ProfileID)
	if err != nil {
		cv.Refused = "The profile of candidate A cannot be loaded: " + err.Error()
		return cv
	}
	pb, _, err := d.cfg.Tuning.LoadProfile(cmp.B.ProfileID)
	if err != nil {
		cv.Refused = "The profile of candidate B cannot be loaded: " + err.Error()
		return cv
	}
	if err := cmp.Check(pa, pb); err != nil {
		cv.Refused = err.Error()
		return cv
	}
	if cv.Delta, err = tuning.DiffProfiles(pa, pb); err != nil {
		cv.Refused = err.Error()
		return cv
	}
	cv.ProfileA, cv.ProfileB = tuningLocation(tv.Source, pa.ID()), tuningLocation(tv.Source, pb.ID())
	for _, c := range cv.Delta.Groups {
		cv.Groups = append(cv.Groups, TuningGroupChangeRow{ID: c.GroupID, A: groupChoiceLabel(c.A), B: groupChoiceLabel(c.B)})
	}
	for _, c := range cv.Delta.Regions {
		label := c.RegionID
		if text, ok := regionCopy[c.RegionID]; ok {
			label = text.Label
		}
		cv.Regions = append(cv.Regions, TuningRegionChangeRow{ID: c.RegionID, Label: label, A: preservationLabel(c.A), B: preservationLabel(c.B)})
	}
	score := func(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) }
	count := strconv.Itoa
	cv.Quality = []TuningDeltaRow{
		{Key: "accuracy", Label: "Choice accuracy", State: Measured, A: score(cmp.Accuracy.A), B: score(cmp.Accuracy.B), Diff: signed(cmp.Accuracy.Diff, ""),
			Basis: "Candidate evidence."},
		{Key: "errors", Label: "Request errors", State: Measured, A: count(cmp.ErrorsA), B: count(cmp.ErrorsB), Diff: signedCount(cmp.ErrorsB - cmp.ErrorsA),
			Basis: "Candidate evidence."},
	}
	for _, q := range cmp.Questions {
		cv.Questions = append(cv.Questions, TuningDeltaRow{Key: q.Question, Label: q.Question, State: Measured, A: score(q.Accuracy.A), B: score(q.Accuracy.B),
			Diff: signed(q.Accuracy.Diff, ""), Basis: count(q.NA) + " / " + count(q.NB)})
	}
	labels := map[string]string{"request-p50": "Request latency p50", "request-p95": "Request latency p95", "request-mean": "Request latency mean",
		"inference-p50": "Inference latency p50", "inference-p95": "Inference latency p95", "inference-mean": "Inference latency mean"}
	cv.Resources = []TuningDeltaRow{
		{Key: "size", Label: "Model size", State: NotChecked, Basis: "Evidence reports record no model size."},
		{Key: "memory", Label: "Memory (VRAM/RAM)", State: NotChecked, Basis: "Evidence reports record no memory use."},
	}
	for _, r := range cmp.Resources {
		row := TuningDeltaRow{Key: r.Key, Label: labels[r.Key], State: NotChecked, Basis: "A report has no latency samples."}
		if r.Available {
			row.State, row.Basis = Measured, "Candidate evidence."
			row.A, row.B = strconv.FormatFloat(r.A, 'f', 1, 64)+" "+r.Unit, strconv.FormatFloat(r.B, 'f', 1, 64)+" "+r.Unit
			row.Diff = signed(r.Diff, r.Unit)
		}
		cv.Resources = append(cv.Resources, row)
	}
	return cv
}
