package dashboard

import (
	"fmt"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// The layer-wise editor of the Tuning workspace. The model is shown as a
// structured stack: component families (the backend's semantic regions) that
// expand into the stable block-level groups of the exact source. Every group
// is AUTO or carries one named policy from the ordered set the optimizer
// executes; groups the canonical optimizer contract always preserves are shown
// as required, with no control. The editor describes a profile; it neither
// runs the optimizer nor decides what the backend can do: the policy set, the
// groups, the required set and the resolved plan are the tuning authority's.

// TuningFigure is one figure of the accepted baseline and of the candidate
// built from a profile. Evaluated marks a figure that comes from an
// evaluation, comparable only between runs of the same dataset and questions.
type TuningFigure struct {
	Key, Label          string
	Baseline, Candidate ImpactValue
	Evaluated           bool
}

// TuningCandidate is what the home holds about the candidate built from one
// exact profile, and about the accepted baseline it is compared with. Variant
// is empty when no candidate was built from the profile. Baseline is the
// accepted baseline variant, empty with BaselineWhy when there is none.
// BaselineEvidence is the baseline's applied per-group evidence;
// BaselineCanonical is true when the baseline was built by the canonical
// recipe, whose per-group policy is the all-AUTO resolution. A baseline with
// neither has an unknown per-group policy.
type TuningCandidate struct {
	Variant           string
	Evidence          *home.TuningEvidence
	EvidenceErr       string
	Baseline          string
	BaselineWhy       string
	BaselineEvidence  *home.TuningEvidence
	BaselineCanonical bool
	// Comparable: the baseline's and the candidate's evaluated figures come
	// from the same dataset and questions. ComparableWhy says why not.
	Comparable    bool
	ComparableWhy string
	Figures       []TuningFigure
}

// TuningPolicyOption is one named policy of the ordered policy set.
type TuningPolicyOption struct {
	ID, Label, Transformation string
	Rank                      int
}

var policyLabels = map[string]string{
	home.PolicySourcePrecision: "Source precision (bfloat16)",
	home.PolicyW4A16:           "W4A16 (4-bit weights)",
}

func policyLabel(id string) string {
	switch id {
	case "", home.PolicyAuto:
		return "AUTO"
	}
	if l, ok := policyLabels[id]; ok {
		return l
	}
	return id
}

func tuningPolicies() []TuningPolicyOption {
	var out []TuningPolicyOption
	for _, p := range tuning.Policies() {
		out = append(out, TuningPolicyOption{ID: p.ID, Label: policyLabel(p.ID), Transformation: p.Transformation, Rank: p.Rank})
	}
	return out
}

// TuningGroupRow is one stable group: what the profile asks, what the plan
// resolves it to and why. State is Auto, Overridden, or Preserved for a group
// the canonical contract requires.
type TuningGroupRow struct {
	ID, Label, LayerType string
	Layer, Modules       int
	State                SemanticState
	Selected             string // form value: "auto" or a policy ID
	Effective            string // policy ID
	Basis                string
}

// TuningFamilyRow is one component family (a semantic region) with its
// block-level groups. Groups holds the tunable groups only; a Locked family
// has none and every group is required.
type TuningFamilyRow struct {
	ID, Label, Purpose                               string
	Groups                                           []TuningGroupRow
	Total, Required, Overridden, AtSource, Quantized int
	Locked, Recommended                              bool
	Blocks, Basis                                    string
	State                                            SemanticState // Preserved when locked, Overridden when any group is, else Auto
}

// TuningSummary counts the groups of the plan by who chose and what resulted.
type TuningSummary struct {
	Groups, Required, Tunable, Auto, Overridden, AtSource, Quantized int
}

// TuningChangeRow is a set of blocks of one family whose effective policy
// differs from the baseline's.
type TuningChangeRow struct {
	Family, Blocks, From, To string
	Groups                   int
}

// TuningFigureRow is a normalized TuningFigure.
type TuningFigureRow = TuningFigure

// TuningAppliedRow is the applied evidence of one group.
type TuningAppliedRow struct {
	ID                            string
	Selection                     SemanticState
	Requested, Effective, Applied string
	Required                      bool
}

// TuningCandidateView is the candidate built from the shown profile, beside
// the accepted baseline.
type TuningCandidateView struct {
	Variant, Baseline, BaselineWhy string
	Comparable                     bool
	ComparableWhy                  string
	Figures                        []TuningFigureRow
	Applied                        []TuningAppliedRow
	AppliedNote                    string
}

// TuningLegacyView says that the shown profile is the exact layer-wise
// equivalent of a stored legacy coarse profile, which stays valid and
// buildable as it is.
type TuningLegacyView struct {
	ID, Objective string
	Pinned        []string // regions the legacy profile pinned
	Redundant     []string // pins the canonical contract already satisfied
	UpgradedID    string   // identity of the layer-wise equivalent
	UpgradedSaved bool
}

// TuningBulkForm is the bulk-override selector's current values.
type TuningBulkForm struct {
	Region, From, To, Policy string
	Blocks                   int // number of blocks of the source, for the range bounds
}

func blockLabel(layer int) string { return fmt.Sprintf("Block %02d", layer) }

// layerwiseRows projects the resolved plan as families of groups. profile is
// the layer-wise profile whose plan it is.
func layerwiseRows(analysis tuning.Analysis, profile tuning.Profile, plan home.TuningPlan) ([]TuningFamilyRow, TuningSummary) {
	byRegion := map[string][]home.TuningGroupPlan{}
	for _, g := range plan.Groups {
		byRegion[g.Region] = append(byRegion[g.Region], g)
	}
	var (
		rows []TuningFamilyRow
		sum  TuningSummary
	)
	for _, region := range analysis.Regions {
		text, ok := regionCopy[region.ID]
		if !ok {
			text.Label, text.Purpose = region.ID, region.Description
		}
		fam := TuningFamilyRow{ID: region.ID, Label: text.Label, Purpose: text.Purpose}
		var blocks []int
		for _, g := range byRegion[region.ID] {
			fam.Total++
			sum.Groups++
			if g.Layer >= 0 {
				blocks = append(blocks, g.Layer)
			}
			if g.Effective == home.PolicySourcePrecision {
				fam.AtSource++
				sum.AtSource++
			} else {
				fam.Quantized++
				sum.Quantized++
			}
			if g.Required {
				fam.Required++
				sum.Required++
				if fam.Basis == "" {
					fam.Basis = g.Basis
				}
				continue
			}
			sum.Tunable++
			row := TuningGroupRow{ID: g.ID, Label: g.ID, Layer: g.Layer, Modules: len(g.Modules), Effective: g.Effective, Basis: g.Basis, Selected: home.PolicyAuto, State: Auto}
			if g.Layer >= 0 {
				row.Label = blockLabel(g.Layer)
				if ag, ok := analysis.GroupByID(g.ID); ok {
					row.LayerType = strings.ReplaceAll(ag.LayerType, "_", " ")
				}
			}
			if g.Selection == home.SelectionOverridden {
				row.State, row.Selected = Overridden, g.Requested
				fam.Overridden++
				sum.Overridden++
			} else {
				sum.Auto++
			}
			fam.Groups = append(fam.Groups, row)
		}
		fam.Locked = fam.Total > 0 && fam.Required == fam.Total
		fam.Blocks = compactBlocks(blocks)
		switch {
		case fam.Locked:
			fam.State = Preserved
		case fam.Overridden > 0:
			fam.State = Overridden
		default:
			fam.State = Auto
		}
		rows = append(rows, fam)
	}
	return rows, sum
}

// compactBlocks renders block indices as ranges: "0–3, 7".
func compactBlocks(blocks []int) string {
	if len(blocks) == 0 {
		return ""
	}
	b := slices.Clone(blocks)
	sort.Ints(b)
	var parts []string
	for i := 0; i < len(b); {
		j := i
		for j+1 < len(b) && b[j+1] == b[j]+1 {
			j++
		}
		if j > i {
			parts = append(parts, fmt.Sprintf("%d–%d", b[i], b[j]))
		} else {
			parts = append(parts, strconv.Itoa(b[i]))
		}
		i = j + 1
	}
	return strings.Join(parts, ", ")
}

// baselinePolicies is the per-group effective policy of the baseline and its
// basis: the accepted baseline's applied evidence, else the canonical recipe's
// all-AUTO resolution. It is nil, with the reason, when the baseline's policy
// is not known.
func baselinePolicies(c TuningCandidate, canonical home.TuningPlan) (map[string]string, string) {
	policies := map[string]string{}
	switch {
	case c.BaselineEvidence != nil:
		for _, g := range c.BaselineEvidence.Groups {
			policies[g.ID] = g.Applied
		}
		return policies, "accepted baseline " + c.Baseline + " (its applied evidence)"
	case c.Baseline != "" && !c.BaselineCanonical:
		return nil, "accepted baseline " + c.Baseline + " was not built with the canonical recipe and records no per-group evidence"
	}
	for _, g := range canonical.Groups {
		policies[g.ID] = g.Effective
	}
	if c.Baseline != "" {
		return policies, "accepted baseline " + c.Baseline + " (built with the canonical recipe)"
	}
	return policies, "the canonical recipe, the policy of every untuned variant (" + firstNonEmpty(c.BaselineWhy, "no accepted baseline variant") + ")"
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// planChanges lists, by family and effective policy, the groups whose policy
// differs from the baseline's.
func planChanges(plan home.TuningPlan, baseline map[string]string) []TuningChangeRow {
	type key struct{ region, from, to string }
	groups := map[key][]int{}
	var order []key
	for _, g := range plan.Groups {
		from, ok := baseline[g.ID]
		if !ok || from == g.Effective {
			continue
		}
		k := key{g.Region, from, g.Effective}
		if _, seen := groups[k]; !seen {
			order = append(order, k)
		}
		groups[k] = append(groups[k], g.Layer)
	}
	var out []TuningChangeRow
	for _, k := range order {
		label := k.region
		if t, ok := regionCopy[k.region]; ok {
			label = t.Label
		}
		out = append(out, TuningChangeRow{Family: label, Blocks: compactBlocks(groups[k]), From: k.from, To: k.to, Groups: len(groups[k])})
	}
	return out
}

// candidateView normalizes the authority's candidate report against the
// current plan.
func candidateView(c TuningCandidate, plan home.TuningPlan) *TuningCandidateView {
	if c.Variant == "" {
		return nil
	}
	cv := &TuningCandidateView{Variant: c.Variant, Baseline: c.Baseline, BaselineWhy: c.BaselineWhy, Comparable: c.Comparable, ComparableWhy: c.ComparableWhy}
	for _, f := range c.Figures {
		f.Baseline, f.Candidate = f.Baseline.normalized(), f.Candidate.normalized()
		cv.Figures = append(cv.Figures, f)
	}
	switch ev := c.Evidence; {
	case c.EvidenceErr != "":
		cv.AppliedNote = "The candidate's applied evidence is unreadable: " + c.EvidenceErr
	case ev == nil:
		cv.AppliedNote = "The candidate was built before per-group evidence existed (a legacy coarse profile): no per-group transformation is recorded for it."
	case ev.PlanSHA256 != plan.SHA256():
		cv.AppliedNote = "The candidate was built under another resolution of this profile (plan " + short12(ev.PlanSHA256) + "; the current plan is " + short12(plan.SHA256()) + "), so its applied evidence is not shown against the current plan."
	default:
		for _, g := range ev.Groups {
			sel := Auto
			if g.Selection == home.SelectionOverridden {
				sel = Overridden
			}
			cv.Applied = append(cv.Applied, TuningAppliedRow{ID: g.ID, Selection: sel, Requested: g.Requested, Effective: g.Effective, Applied: g.Applied, Required: g.Required})
		}
	}
	return cv
}

// --- editing ---------------------------------------------------------------

// tuningEdit applies one bulk edit to the profile the form describes and shows
// the result as an unsaved draft. It saves and builds nothing.
func (d *Dashboard) tuningEdit(w http.ResponseWriter, r *http.Request) {
	source, profile, analysis, err := d.tuningProfile(r)
	var notice string
	if err == nil {
		profile, notice, err = d.applyTuningEdit(r, profile, analysis)
	}
	v := d.view("Tuning", "tuning")
	v.Live = false
	mv := d.modelsView(v)
	v.Models = mv
	if err != nil && profile.Schema == "" {
		// The form itself could not be read: show the saved view with the reason.
		v.Tuning = d.tuningView(mv, source, "", nil)
		v.Tuning.Notice, v.Tuning.NoticeBad = err.Error(), true
	} else {
		v.Tuning = d.tuningView(mv, source, "", &profile)
		if err != nil {
			notice, v.Tuning.NoticeBad = err.Error(), true
		}
		v.Tuning.Notice = notice
	}
	v.Tuning.AdvancedOpen = true
	d.finishTuning(&v)
	d.renderView(w, "tuning", v)
}

// applyTuningEdit applies the requested operation. When the edit is refused the
// unedited profile is returned with the reason, so the operator never loses
// the editor to a refused combination.
func (d *Dashboard) applyTuningEdit(r *http.Request, profile tuning.Profile, analysis tuning.Analysis) (tuning.Profile, string, error) {
	op := strings.TrimSpace(r.PostFormValue("op"))
	var (
		sel    tuning.Selection
		policy string
		what   string
	)
	switch {
	case op == "bulk":
		var err error
		if sel, err = bulkSelection(r); err != nil {
			return profile, "", err
		}
		policy = strings.TrimSpace(r.PostFormValue("bulk_policy"))
		what = "the selected groups"
	case op == "reset_all":
		sel, policy, what = tuning.Selection{From: -1, To: -1}, home.PolicyAuto, "every tunable group"
	case strings.HasPrefix(op, "reset:"):
		sel, policy, what = tuning.Selection{Region: strings.TrimPrefix(op, "reset:"), From: -1, To: -1}, home.PolicyAuto, "this family"
	case op == "" || op == "refresh":
		return profile, "", nil
	default:
		return profile, "", fmt.Errorf("unknown editor operation %q", op)
	}
	if policy != home.PolicyAuto {
		if _, ok := tuning.PolicyRank(policy); !ok {
			return profile, "", fmt.Errorf("unsupported policy %q for the selection", policy)
		}
	}
	ids, err := sel.Select(analysis)
	if err != nil {
		return profile, "", err
	}
	next, err := tuning.SetPolicy(profile, analysis, ids, policy)
	if err == nil {
		_, err = tuning.Compile(next, analysis)
	}
	if err != nil {
		return profile, "", err
	}
	return next, fmt.Sprintf("Set %s (%d groups) to %s. This is an unsaved draft; nothing was saved or built.", what, len(ids), policyLabel(policy)), nil
}

func bulkSelection(r *http.Request) (tuning.Selection, error) {
	sel := tuning.Selection{Region: strings.TrimSpace(r.PostFormValue("bulk_region")), From: -1, To: -1}
	for _, f := range []struct {
		name string
		into *int
	}{{"bulk_from", &sel.From}, {"bulk_to", &sel.To}} {
		raw := strings.TrimSpace(r.PostFormValue(f.name))
		if raw == "" {
			continue
		}
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return sel, fmt.Errorf("the block range must be whole block numbers, got %q", raw)
		}
		*f.into = n
	}
	if sel.From >= 0 && sel.To >= 0 && sel.From > sel.To {
		return sel, fmt.Errorf("the block range %d to %d is empty", sel.From, sel.To)
	}
	return sel, nil
}

// groupPolicyFields reads the per-group selections of the editor form into a
// layer-wise profile: every group is AUTO unless its field names a policy.
func groupPolicyFields(r *http.Request, profile tuning.Profile, analysis tuning.Analysis) (tuning.Profile, error) {
	for key, values := range r.PostForm {
		id, ok := strings.CutPrefix(key, "group.")
		if !ok {
			continue
		}
		if _, known := analysis.GroupByID(id); !known {
			return profile, fmt.Errorf("unknown group %q", id)
		}
		switch value := values[len(values)-1]; value {
		case home.PolicyAuto:
		default:
			if _, ok := tuning.PolicyRank(value); !ok {
				return profile, fmt.Errorf("unsupported policy %q for group %q", value, id)
			}
			profile.Groups[id] = tuning.GroupChoice{Mode: tuning.GroupOverride, Policy: value}
		}
	}
	return profile, nil
}
