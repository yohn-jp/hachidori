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
	var open []string
	for _, id := range recommendable {
		if _, ok := modules[id]; ok && !preserved[id] && profile.Preservation[id].Mode == PreservationAuto {
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

func short(s string) string { return s[:min(len(s), 12)] }
