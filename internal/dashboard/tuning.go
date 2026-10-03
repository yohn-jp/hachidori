package dashboard

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// The Tuning workspace: model-engineering intent. It edits and persists a
// versioned semantic profile (an objective and a preservation choice for each
// backend-analyzed region) and hands that exact saved profile to Forge. It
// never runs, sequences or inspects the optimizer: the regions, the profile
// store, the recipe compiler and the build are the typed tuning authority's.
// Generated module mappings and the recipe stay in Details/Evidence, and every
// impact figure states whether it was measured, estimated or not checked.

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
	// BuildCandidate hands the exact saved profile to the Forge build
	// (app.Controller.OptimizeProfile): one accepted background operation
	// whose phases, candidate and evidence are Forge's. Nothing is applied.
	BuildCandidate(source, profileID string) error
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

// TuningProfileRow is one saved profile of the source.
type TuningProfileRow struct {
	ID, Objective string
	Pinned        int
	Current       bool
}

// TuningEvidenceView is the Details/Evidence projection of the compiled
// profile: the generated mappings and recipe identity. It is never an input.
type TuningEvidenceView struct {
	Recipe, RecipeSHA256, CompilerVersion string
	Regions                               []tuning.RegionMapping
	Preserved                             []tuning.PreservedMapping
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

	Regions  []TuningRegionRow
	Pinned   int
	Impact   []TuningImpactRow
	Profiles []TuningProfileRow
	Evidence *TuningEvidenceView
	// EvidenceDisclosure is the shared Details/Evidence primitive that holds
	// every generated mapping.
	EvidenceDisclosure DisclosureProjection

	// Context is the Experiment/Evidence context handed to this page, nil
	// when none was.
	Context *TuningContextView
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

func pinnedCount(p tuning.Profile) int {
	n := 0
	for _, c := range p.Preservation {
		if c.Mode == tuning.PreservationPinned {
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
	handedOver := q.Get(qCand) != ""
	if handedOver {
		ctx, ctxErr = d.resolveTuningContext(q)
		if ctxErr == nil {
			profileID = ctx.ProfileID
		}
	}
	v.Tuning = d.tuningView(mv, source, profileID)
	if handedOver {
		v.Tuning.Context = d.tuningContextView(v.Tuning, q, ctx, ctxErr)
	}
	d.renderView(w, "tuning", v)
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
	if !tv.Saved || tv.ProfileID != ctx.ProfileID {
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
		for i := range tv.Regions {
			tv.Regions[i].Recommended = tv.Regions[i].ID == cv.Recommendation.RegionID
		}
	}
	return cv
}

// tuningView projects the tuning authority's state for one source and,
// optionally, one saved profile. Without a saved profile it shows the initial
// all-Auto profile, labelled as unsaved: nothing is invented or remembered.
func (d *Dashboard) tuningView(mv *ModelsView, source, profileID string) *TuningView {
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
	var profile tuning.Profile
	switch {
	case profileID != "":
		p, a, err := t.LoadProfile(profileID)
		switch {
		case err != nil:
			tv.Err = "Saved profile " + short12(profileID) + " cannot be used: " + err.Error()
		case p.Source != analysis.Source || a.SHA256() != analysis.SHA256():
			tv.Err = "Saved profile " + short12(profileID) + " is not bound to the current analysis of " + source + "."
		default:
			profile, tv.Saved = p, true
		}
	default:
		for _, p := range saved {
			if p.Source == analysis.Source && p.AnalysisSHA256 == analysis.SHA256() {
				profile, tv.Saved = p, true
				break
			}
		}
	}
	if !tv.Saved {
		p, err := tuning.NewDefaultProfile(analysis, tv.Objective)
		if err != nil {
			tv.Err = "The initial profile cannot be created: " + err.Error()
			return tv
		}
		profile = p
	}
	compiled, err := tuning.Compile(profile, analysis)
	if err != nil {
		tv.Err = "The profile cannot be compiled: " + err.Error()
		return tv
	}
	tv.Objective, tv.Pinned = profile.Objective, pinnedCount(profile)
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
		row := TuningRegionRow{ID: region.ID, Label: text.Label, Purpose: text.Purpose, Modules: len(region.Modules), Files: len(region.Files),
			Pinned: profile.Preservation[region.ID].Mode == tuning.PreservationPinned, Effective: Auto}
		switch m := mapping[region.ID]; {
		case row.Pinned:
			row.Effective, row.Note = Preserved, "Pinned by this profile at source precision."
		case m.Preserved:
			row.Effective, row.Note = Preserved, "Auto: the canonical policy already preserves this region."
		default:
			row.Note = "Auto: the canonical policy quantizes this region."
		}
		tv.Regions = append(tv.Regions, row)
	}
	for _, p := range saved {
		tv.Profiles = append(tv.Profiles, TuningProfileRow{ID: p.ID(), Objective: p.Objective, Pinned: pinnedCount(p), Current: tv.Saved && p.ID() == profile.ID()})
	}
	tv.Impact = d.tuningImpact(profile, analysis, compiled)
	tv.Evidence = &TuningEvidenceView{Recipe: compiled.Recipe.Name, RecipeSHA256: compiled.Recipe.SHA256(), CompilerVersion: compiled.Evidence.CompilerVersion,
		Regions: compiled.Evidence.Regions, Preserved: compiled.Evidence.Preserved}
	return tv
}

// tuningImpact normalizes the authority's projection. An authority that
// cannot say anything leaves every value NOT_CHECKED.
func (d *Dashboard) tuningImpact(p tuning.Profile, a tuning.Analysis, c tuning.Compilation) []TuningImpactRow {
	imp, err := d.cfg.Tuning.Impact(p, a, c)
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
	return rows
}

func short12(s string) string { return s[:min(len(s), 12)] }

// tuningProfile builds the profile the form describes from the backend
// analysis: every choice is Auto unless the form pins that region. The form
// names regions by identifier only; it can express nothing else.
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
	for key, values := range r.PostForm {
		id, ok := strings.CutPrefix(key, "region.")
		if !ok {
			continue
		}
		if _, known := profile.Preservation[id]; !known {
			return source, profile, analysis, fmt.Errorf("unknown semantic region %q", id)
		}
		switch mode := values[len(values)-1]; mode {
		case string(tuning.PreservationAuto):
		case string(tuning.PreservationPinned):
			profile.Preservation[id] = tuning.PreservationChoice{Mode: tuning.PreservationPinned, Precision: tuning.PreservedPrecision}
		default:
			return source, profile, analysis, fmt.Errorf("unknown preservation %q for region %q", mode, id)
		}
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
// hands exactly that saved profile to the Forge build. The candidate, its
// progress and its evidence are then Forge's; Apply stays explicit there.
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
	if err == nil {
		err = d.cfg.Tuning.BuildCandidate(source, savedID)
	}
	if err != nil {
		d.remember("build candidate "+source, err, "")
		http.Redirect(w, r, tuningLocation(source, savedID), http.StatusSeeOther)
		return
	}
	d.remember("build candidate "+source, nil, "handed profile "+saved.ID()+" to Forge. The build, its evidence and any Apply are Forge's; nothing is applied")
	http.Redirect(w, r, "/forge", http.StatusSeeOther)
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
