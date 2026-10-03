package tuning

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
)

// EvidenceRun is the identity one evidence report records for what served it.
// VariantID is empty when the source artifact served the run.
type EvidenceRun struct {
	EvidenceSHA256 string
	ModelID        string
	Revision       string
	VariantID      string
}

// QuestionRegression is one identity-aligned question whose measured accuracy
// fell from the baseline to the candidate.
type QuestionRegression struct {
	Question            string
	N                   int
	Baseline, Candidate float64
}

// MeasuredRegression is the measured, descriptive change between compatible
// baseline and candidate evidence. It is data copied from evidence; nothing in
// it is estimated or attributed to a cause.
type MeasuredRegression struct {
	Cases                               int
	BaselineAccuracy, CandidateAccuracy float64
	BaselineErrors, CandidateErrors     int
	Questions                           []QuestionRegression
}

// EvidenceContext is the exact context an Experiment or Evidence view hands to
// Tuning: which profile and candidate, over which dataset and questions, from
// which evidence. It is advisory input and never changes evidence. Baseline and
// Regression are nil when only candidate evidence was handed over or when the
// compatible comparison measured no regression.
type EvidenceContext struct {
	ProfileID       string
	DatasetSHA256   string
	QuestionsSHA256 string
	Baseline        *EvidenceRun
	Candidate       EvidenceRun
	Regression      *MeasuredRegression
}

// Check refuses a context that is not bound to exactly this profile: the
// profile identity, its source model and revision, and the evidence runs'
// models must all agree, the candidate must be a variant and the baseline the
// source. Contexts that fail are shown separately and inform nothing.
func (c EvidenceContext) Check(profile Profile) error {
	switch {
	case c.ProfileID == "" || c.ProfileID != profile.ID():
		return fmt.Errorf("evidence context is bound to profile %s, not %s", short(c.ProfileID), short(profile.ID()))
	case c.Candidate.VariantID == "":
		return errors.New("candidate evidence does not name an executed variant")
	case c.Candidate.ModelID != profile.Source.ID || c.Candidate.Revision != profile.Source.Revision:
		return fmt.Errorf("candidate evidence serves %s@%s but the profile is for %s@%s",
			c.Candidate.ModelID, c.Candidate.Revision, profile.Source.ID, profile.Source.Revision)
	case c.DatasetSHA256 == "" || c.QuestionsSHA256 == "":
		return errors.New("evidence context does not identify its dataset and questions")
	}
	if b := c.Baseline; b != nil {
		switch {
		case b.ModelID != c.Candidate.ModelID || b.Revision != c.Candidate.Revision:
			return fmt.Errorf("baseline evidence serves %s@%s: different source models are not compared as equivalent", b.ModelID, b.Revision)
		case b.VariantID != "":
			return fmt.Errorf("baseline evidence executed variant %s: the baseline must be the source model", b.VariantID)
		}
	}
	return nil
}

// VariantProfile resolves a variant of a source to the exact profile its build
// provenance names. It refuses a variant that was not built from a profile.
func VariantProfile(h home.Home, sourceID, variantID string) (string, error) {
	if sourceID == "" || variantID == "" || strings.ContainsAny(sourceID+variantID, `/\`) || strings.Contains(sourceID+variantID, "..") {
		return "", errors.New("variant identity is not valid")
	}
	v, err := home.ReadVariant(h.VariantDir(sourceID, variantID))
	switch {
	case errors.Is(err, os.ErrNotExist):
		return "", fmt.Errorf("variant %s is not installed", variantID)
	case err != nil:
		return "", fmt.Errorf("variant %s: %w", variantID, err)
	case v.ID != variantID || v.Source.ID != sourceID:
		return "", fmt.Errorf("variant directory %s holds %s of %s", variantID, v.ID, v.Source.ID)
	case v.Tuning == nil:
		return "", fmt.Errorf("variant %s was not built from a tuning profile", variantID)
	}
	return v.Tuning.ProfileID, nil
}

// Recommendation is one proposed change to one semantic region of one saved
// profile. It is advisory: it states its evidence basis and carries no
// confidence, because the evidence measures that the candidate regressed, not
// which region caused it.
type Recommendation struct {
	ProfileID string
	RegionID  string
	From, To  PreservationMode
	// Basis lists the measured evidence the recommendation rests on.
	Basis []string
	// Rationale states the fixed model-family rule that chose the region.
	Rationale string
}

// recommendable are the Clef regions a recommendation may propose to preserve:
// the backbone projection groups the canonical recipe quantizes. The gates,
// output embeddings, vision tower and joint head are already preserved by the
// canonical policy; the feed-forward projections dominate the model size, so
// pinning them would give up the build's compression and is never suggested.
var recommendable = []string{RegionFullAttention, RegionLinearAttention}

// Recommend returns the supported preservation change that the context's
// measured regression can legitimately inform, or nil and the reason there is
// none. It is a pure, deterministic function: it chooses, among the
// recommendable regions the profile leaves quantized, the one with the fewest
// modules, and recommends nothing when the context is not bound to the
// profile, the evidence is not a confound-free measured regression, or the
// choice is not unique.
func Recommend(ctx EvidenceContext, profile Profile, analysis Analysis) (*Recommendation, string) {
	if err := ctx.Check(profile); err != nil {
		return nil, err.Error()
	}
	compiled, err := Compile(profile, analysis)
	if err != nil {
		return nil, "the profile cannot be compiled: " + err.Error()
	}
	reg := ctx.Regression
	switch {
	case ctx.Baseline == nil:
		return nil, "Only candidate evidence was handed over; a recommendation needs a compatible baseline comparison."
	case reg == nil || len(reg.Questions) == 0 || !(reg.CandidateAccuracy < reg.BaselineAccuracy):
		return nil, "The compatible comparison measured no accuracy regression."
	case reg.BaselineErrors != 0 || reg.CandidateErrors != 0:
		return nil, "Request errors in the evidence confound the measured regression."
	}
	modules := map[string]int{}
	for _, r := range analysis.Regions {
		modules[r.ID] = len(r.Modules)
	}
	preserved := map[string]bool{}
	for _, m := range compiled.Evidence.Regions {
		preserved[m.RegionID] = m.Preserved
	}
	// A region is open when the profile still leaves something of it to AUTO
	// quantization: a legacy region left Auto and not preserved by the
	// canonical policy, or a layer-wise region with an AUTO group resolved to
	// the quantizing policy.
	autoQuantized := map[string]bool{}
	if compiled.Plan != nil {
		for _, g := range compiled.Plan.Groups {
			if g.Selection == home.SelectionAuto && g.Effective == home.PolicyW4A16 {
				autoQuantized[g.Region] = true
			}
		}
	}
	var open []string
	for _, id := range recommendable {
		if _, ok := modules[id]; !ok {
			continue
		}
		if profile.Schema == ProfileSchema && autoQuantized[id] || profile.Schema == ProfileSchemaV1 && !preserved[id] && profile.Preservation[id].Mode == PreservationAuto {
			open = append(open, id)
		}
	}
	if len(open) == 0 {
		return nil, "The supported preservation changes are already applied by this profile."
	}
	sort.SliceStable(open, func(i, j int) bool { return modules[open[i]] < modules[open[j]] })
	if len(open) > 1 && modules[open[0]] == modules[open[1]] {
		return nil, fmt.Sprintf("Regions %s and %s are equally small; the evidence cannot choose between them.", open[0], open[1])
	}
	region := open[0]
	basis := []string{
		fmt.Sprintf("Evidence: baseline %s (source %s@%s) against candidate %s (variant %s), dataset %s, questions %s.",
			short(ctx.Baseline.EvidenceSHA256), ctx.Baseline.ModelID, ctx.Baseline.Revision, short(ctx.Candidate.EvidenceSHA256), ctx.Candidate.VariantID,
			short(ctx.DatasetSHA256), short(ctx.QuestionsSHA256)),
		fmt.Sprintf("Measured choice accuracy fell from %.4f to %.4f over %d cases.", reg.BaselineAccuracy, reg.CandidateAccuracy, reg.Cases),
	}
	for _, q := range reg.Questions {
		basis = append(basis, fmt.Sprintf("Question %s: accuracy %.4f to %.4f over %d observations.", q.Question, q.Baseline, q.Candidate, q.N))
	}
	rationale := fmt.Sprintf("Preserve %s (%d modules) at source precision: of the supported regions this profile still quantizes (%s) it is the smallest, so it gives up the least compression. "+
		"The evidence shows the candidate regressed; it does not show that this region caused it. Build and evaluate the new profile to learn its effect.",
		region, modules[region], strings.Join(open, ", "))
	return &Recommendation{ProfileID: profile.ID(), RegionID: region, From: PreservationAuto, To: PreservationPinned, Basis: basis, Rationale: rationale}, ""
}

// Accept is the explicit operator action: it applies the recommendation to the
// profile it names and returns the resulting distinct profile, which the caller
// persists through the profile store. The profile it was derived from and the
// evidence are untouched; nothing is built.
func Accept(rec Recommendation, profile Profile, analysis Analysis) (Profile, error) {
	if rec.ProfileID != profile.ID() {
		return Profile{}, fmt.Errorf("recommendation is for profile %s, not %s", short(rec.ProfileID), short(profile.ID()))
	}
	if err := validateProfile(profile, analysis); err != nil {
		return Profile{}, err
	}
	if !contains(recommendable, rec.RegionID) || rec.From != PreservationAuto || rec.To != PreservationPinned {
		return Profile{}, fmt.Errorf("recommendation to change region %q is not supported", rec.RegionID)
	}
	if profile.Schema == ProfileSchema {
		return acceptLayerwise(rec, profile, analysis)
	}
	if profile.Preservation[rec.RegionID].Mode != PreservationAuto {
		return Profile{}, fmt.Errorf("region %q is already pinned by the profile", rec.RegionID)
	}
	next := profile
	next.Preservation = make(map[string]PreservationChoice, len(profile.Preservation))
	for id, choice := range profile.Preservation {
		next.Preservation[id] = choice
	}
	next.Preservation[rec.RegionID] = PreservationChoice{Mode: PreservationPinned, Precision: PreservedPrecision}
	if _, err := Compile(next, analysis); err != nil {
		return Profile{}, err
	}
	return next, nil
}

// acceptLayerwise applies a recommendation to a layer-wise profile: every group
// of the region that the profile still leaves to AUTO becomes an explicit
// source-precision override. Groups the operator already set are untouched.
func acceptLayerwise(rec Recommendation, profile Profile, analysis Analysis) (Profile, error) {
	var ids []string
	for _, g := range analysis.Groups {
		if g.Region == rec.RegionID && profile.Groups[g.ID].Mode == GroupAuto {
			ids = append(ids, g.ID)
		}
	}
	if len(ids) == 0 {
		return Profile{}, fmt.Errorf("region %q is already set by the profile", rec.RegionID)
	}
	next, err := SetPolicy(profile, analysis, ids, home.PolicySourcePrecision)
	if err != nil {
		return Profile{}, err
	}
	if _, err := Compile(next, analysis); err != nil {
		return Profile{}, err
	}
	return next, nil
}

// MeasuredDelta is one measured value of both candidates and B minus A. It is
// directional and descriptive: a positive Diff means the value is larger in B,
// never that B is better.
type MeasuredDelta struct {
	A, B, Diff float64
}

// QuestionDifference is one identity-aligned question whose measured accuracy
// differs between candidate A and candidate B. N counts the observations in
// each report.
type QuestionDifference struct {
	Question string
	NA, NB   int
	Accuracy MeasuredDelta
}

// ResourceDelta is one resource figure compared between the candidates.
// Available is false when the evidence of either candidate does not record
// the figure; the values are then zero and nothing is estimated in their
// place.
type ResourceDelta struct {
	Key       string
	Unit      string
	Available bool
	MeasuredDelta
}

// CandidateRun is one candidate of a comparison: the executed variant, the
// evidence that measured it and the exact profile its build provenance names.
type CandidateRun struct {
	EvidenceRun
	ProfileID string
}

// CandidateComparison is the exact context of comparing two tuned candidates
// of one source with each other: over which dataset and questions, from which
// evidence, built from which profiles, and what the compatible evidence
// measured. It is descriptive input to Tuning. It names no baseline, infers no
// cause, carries no confidence and yields no recommendation.
type CandidateComparison struct {
	DatasetSHA256    string
	QuestionsSHA256  string
	A, B             CandidateRun
	Cases            int
	Accuracy         MeasuredDelta
	ErrorsA, ErrorsB int
	// Questions are the aligned questions whose accuracy differs, by id.
	Questions []QuestionDifference
	Resources []ResourceDelta
}

// Check refuses a comparison that is not bound to exactly these two profiles:
// each run must be a distinct executed variant of the profiles' common source
// model and revision, bound to its profile by identity, over an identified
// dataset and questions. A comparison that fails is shown separately and
// informs nothing.
func (c CandidateComparison) Check(a, b Profile) error {
	switch {
	case a.Source != b.Source:
		return fmt.Errorf("profiles are for different sources: %s@%s and %s@%s", a.Source.ID, a.Source.Revision, b.Source.ID, b.Source.Revision)
	case c.A.ProfileID == "" || c.A.ProfileID != a.ID():
		return fmt.Errorf("candidate A is bound to profile %s, not %s", short(c.A.ProfileID), short(a.ID()))
	case c.B.ProfileID == "" || c.B.ProfileID != b.ID():
		return fmt.Errorf("candidate B is bound to profile %s, not %s", short(c.B.ProfileID), short(b.ID()))
	case c.A.VariantID == "" || c.B.VariantID == "":
		return errors.New("both candidates must name an executed variant")
	case c.A.VariantID == c.B.VariantID:
		return fmt.Errorf("both evidence reports executed variant %s: two different candidates are required", c.A.VariantID)
	case c.A.ModelID != a.Source.ID || c.A.Revision != a.Source.Revision || c.B.ModelID != a.Source.ID || c.B.Revision != a.Source.Revision:
		return fmt.Errorf("candidate evidence serves %s@%s and %s@%s but the profiles are for %s@%s",
			c.A.ModelID, c.A.Revision, c.B.ModelID, c.B.Revision, a.Source.ID, a.Source.Revision)
	case c.DatasetSHA256 == "" || c.QuestionsSHA256 == "":
		return errors.New("candidate comparison does not identify its dataset and questions")
	}
	return nil
}

// RegionChange is one semantic region whose preservation choice differs
// between two profiles. A choice with an empty Mode means the profile has no
// choice for the region, because it is bound to an analysis without it.
type RegionChange struct {
	RegionID string
	A, B     PreservationChoice
}

// ProfileDelta is the deterministic semantic difference between profile A and
// profile B of one source: the objective, the identities of their analysis
// and compiler, and every region whose choice differs. It is a function of the
// two profiles only.
type ProfileDelta struct {
	A, B                   string // profile identities
	SchemaA, SchemaB       string
	ObjectiveA, ObjectiveB string
	AnalysisA, AnalysisB   string
	CompilerA, CompilerB   string
	Regions                []RegionChange // by region id (legacy coarse profiles)
	Groups                 []GroupChange  // by group id (layer-wise profiles)
}

// GroupChange is one layer group whose choice differs between two layer-wise
// profiles. A choice with an empty Mode means the profile has no choice for
// the group.
type GroupChange struct {
	GroupID string
	A, B    GroupChoice
}

// SchemaChanged reports whether the profiles use different schemas (a legacy
// coarse profile and a layer-wise one), so that no group or region difference
// is defined between them.
func (d ProfileDelta) SchemaChanged() bool { return d.SchemaA != d.SchemaB }

// Identical reports whether the profiles are the same profile.
func (d ProfileDelta) Identical() bool { return d.A == d.B }

// ObjectiveChanged reports whether the recorded objectives differ.
func (d ProfileDelta) ObjectiveChanged() bool { return d.ObjectiveA != d.ObjectiveB }

// AnalysisChanged reports whether the profiles are bound to different
// analyses of the source, so that their regions are not the same set.
func (d ProfileDelta) AnalysisChanged() bool { return d.AnalysisA != d.AnalysisB }

// CompilerChanged reports whether the profiles name different compiler
// versions.
func (d ProfileDelta) CompilerChanged() bool { return d.CompilerA != d.CompilerB }

// DiffProfiles is the semantic profile difference of two profiles of one
// exact source. It refuses profiles of different sources, which have no
// common semantic regions.
func DiffProfiles(a, b Profile) (ProfileDelta, error) {
	if a.Source != b.Source {
		return ProfileDelta{}, fmt.Errorf("profiles are for different sources: %s@%s and %s@%s", a.Source.ID, a.Source.Revision, b.Source.ID, b.Source.Revision)
	}
	d := ProfileDelta{A: a.ID(), B: b.ID(), SchemaA: a.Schema, SchemaB: b.Schema, ObjectiveA: a.Objective, ObjectiveB: b.Objective,
		AnalysisA: a.AnalysisSHA256, AnalysisB: b.AnalysisSHA256, CompilerA: a.CompilerVersion, CompilerB: b.CompilerVersion}
	if a.Schema != b.Schema {
		return d, nil
	}
	if a.Schema == ProfileSchema {
		ids := map[string]bool{}
		for id := range a.Groups {
			ids[id] = true
		}
		for id := range b.Groups {
			ids[id] = true
		}
		sorted := make([]string, 0, len(ids))
		for id := range ids {
			sorted = append(sorted, id)
		}
		sort.Strings(sorted)
		for _, id := range sorted {
			if x, y := a.Groups[id], b.Groups[id]; x != y {
				d.Groups = append(d.Groups, GroupChange{GroupID: id, A: x, B: y})
			}
		}
		return d, nil
	}
	ids := map[string]bool{}
	for id := range a.Preservation {
		ids[id] = true
	}
	for id := range b.Preservation {
		ids[id] = true
	}
	sorted := make([]string, 0, len(ids))
	for id := range ids {
		sorted = append(sorted, id)
	}
	sort.Strings(sorted)
	for _, id := range sorted {
		if x, y := a.Preservation[id], b.Preservation[id]; x != y {
			d.Regions = append(d.Regions, RegionChange{RegionID: id, A: x, B: y})
		}
	}
	return d, nil
}

func short(s string) string { return s[:min(len(s), 12)] }
