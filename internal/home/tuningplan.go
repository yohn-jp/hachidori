package home

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// Layer-wise tuning is a fine-grained, per-group choice among the
// transformations the optimizer can actually execute. The resolved plan is the
// exact, complete statement of what a build must do to every addressable group
// of the source; the evidence is what the optimizer verifiably did. Both live
// here, beside the recipe, because the optimizer builder (which cannot import
// the tuning compiler) enforces one against the other.

const (
	// TuningPlanSchema versions the resolved plan and its identity derivation.
	TuningPlanSchema = "hachidori.tuning-plan/1"
	// TuningEvidenceSchema versions the per-group build evidence.
	TuningEvidenceSchema = "hachidori.tuning-evidence/1"
	// TuningEvidenceFile is the evidence's file name inside a variant directory.
	TuningEvidenceFile = "hachidori-tuning-evidence.json"

	// The transformation policies the optimizer executes, ordered by rising
	// compression: a group is kept at the source precision, or its Linear
	// modules are weight-quantized as the canonical recipe's scheme declares.
	PolicySourcePrecision = "source-precision"
	PolicyW4A16           = "w4a16"

	// PolicyAuto is the requested policy of a group the operator left to AUTO.
	PolicyAuto = "auto"

	// SelectionAuto and SelectionOverridden say who chose a group's policy.
	SelectionAuto       = "AUTO"
	SelectionOverridden = "OVERRIDDEN"
)

// TuningGroupPlan is the resolved policy of one stable model group.
// Layer is the transformer block index, or -1 for a group that belongs to no
// block. Modules and Files are the exact Linear modules and carried files the
// group addresses. Requested is what the profile asked for (PolicyAuto when
// left to AUTO); Effective is the policy the build applies. Required marks a
// group the canonical optimizer contract always preserves: it is not tunable.
type TuningGroupPlan struct {
	ID        string   `json:"id"`
	Region    string   `json:"region"`
	Layer     int      `json:"layer"`
	Modules   []string `json:"modules,omitempty"`
	Files     []string `json:"files,omitempty"`
	Selection string   `json:"selection"`
	Requested string   `json:"requested"`
	Effective string   `json:"effective"`
	Required  bool     `json:"required,omitempty"`
	Basis     string   `json:"basis"`
}

// TuningPlan is the complete effective fine-grained policy of one build. Its
// digest is the effective-profile identity bound into the build contract: it
// covers the AUTO resolutions as well as the operator overrides.
type TuningPlan struct {
	Schema       string            `json:"schema"`
	AutoPolicy   string            `json:"auto_policy"` // version of the AUTO resolution rules
	Recipe       string            `json:"recipe"`
	RecipeSHA256 string            `json:"recipe_sha256"` // the compiled recipe the plan produced
	Groups       []TuningGroupPlan `json:"groups"`        // by ID
}

// Canonical is the plan's canonical encoding.
func (p TuningPlan) Canonical() []byte {
	b, err := json.Marshal(p)
	if err != nil {
		panic(err)
	}
	return b
}

// SHA256 is the plan digest.
func (p TuningPlan) SHA256() string { return sha256Hex(p.Canonical()) }

// ValidPolicy reports whether name is a policy the optimizer executes.
func ValidPolicy(name string) bool { return name == PolicySourcePrecision || name == PolicyW4A16 }

// Validate checks the plan is well formed and internally consistent.
func (p TuningPlan) Validate() error {
	if p.Schema != TuningPlanSchema {
		return fmt.Errorf("tuning plan schema %q, want %q", p.Schema, TuningPlanSchema)
	}
	if p.AutoPolicy == "" || p.Recipe == "" || !validSHA256(p.RecipeSHA256) {
		return errors.New("tuning plan must name its AUTO policy version, recipe and recipe digest")
	}
	if len(p.Groups) == 0 {
		return errors.New("tuning plan has no groups")
	}
	if !sort.SliceIsSorted(p.Groups, func(i, j int) bool { return p.Groups[i].ID < p.Groups[j].ID }) {
		return errors.New("tuning plan groups are not in canonical order")
	}
	for i, g := range p.Groups {
		if g.ID == "" || (i > 0 && p.Groups[i-1].ID == g.ID) {
			return fmt.Errorf("tuning plan group %q is empty or repeated", g.ID)
		}
		if len(g.Modules)+len(g.Files) == 0 || g.Region == "" || g.Basis == "" {
			return fmt.Errorf("tuning plan group %q has no members, region or basis", g.ID)
		}
		if !ValidPolicy(g.Effective) {
			return fmt.Errorf("tuning plan group %q resolves to unsupported policy %q", g.ID, g.Effective)
		}
		switch g.Selection {
		case SelectionAuto:
			if g.Requested != PolicyAuto {
				return fmt.Errorf("tuning plan group %q is AUTO but requests %q", g.ID, g.Requested)
			}
		case SelectionOverridden:
			if g.Required || g.Requested != g.Effective || !ValidPolicy(g.Requested) {
				return fmt.Errorf("tuning plan group %q has an override that is not an applicable policy", g.ID)
			}
		default:
			return fmt.Errorf("tuning plan group %q has unknown selection %q", g.ID, g.Selection)
		}
		if g.Required && g.Effective != PolicySourcePrecision {
			return fmt.Errorf("tuning plan group %q is required-preserved but resolves to %q", g.ID, g.Effective)
		}
	}
	return nil
}

// TuningGroupApplied is the evidence for one group: what was requested, what
// the plan resolved, who chose it, and the transformation the optimizer
// verifiably applied. Applied equals Effective in every published variant: a
// build whose applied transformation differs from the plan is refused.
type TuningGroupApplied struct {
	ID        string `json:"id"`
	Selection string `json:"selection"`
	Requested string `json:"requested"`
	Effective string `json:"effective"`
	Required  bool   `json:"required,omitempty"`
	Applied   string `json:"applied"`
	// Preserved is true when the group was kept at its source precision.
	Preserved    bool   `json:"preserved"`
	Modules      int    `json:"modules,omitempty"`
	Files        int    `json:"files,omitempty"`
	WrittenDType string `json:"written_dtype,omitempty"` // of the preserved Linear modules
}

// TuningEvidence is the per-group record of a layer-wise tuning build, written
// into the variant and covered by its artifact digests. UnmappedPreserved and
// UnmappedQuantized count Linear modules the optimizer handled that no group
// addresses; they are reported, never hidden.
type TuningEvidence struct {
	Schema            string               `json:"schema"`
	PlanSHA256        string               `json:"plan_sha256"`
	AutoPolicy        string               `json:"auto_policy"`
	Groups            []TuningGroupApplied `json:"groups"`
	UnmappedPreserved int                  `json:"unmapped_preserved_modules"`
	UnmappedQuantized int                  `json:"unmapped_quantized_modules"`
}

// Canonical is the evidence file's encoding.
func (e TuningEvidence) Canonical() []byte {
	b, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		panic(err)
	}
	return append(b, '\n')
}

// ReadTuningEvidence reads the evidence of a variant directory and checks that
// every group applied exactly what its plan resolved.
func ReadTuningEvidence(dir string) (TuningEvidence, error) {
	var e TuningEvidence
	if err := ReadJSON(dir+"/"+TuningEvidenceFile, &e); err != nil {
		return e, err
	}
	if e.Schema != TuningEvidenceSchema || !validSHA256(e.PlanSHA256) || len(e.Groups) == 0 {
		return e, errors.New("tuning evidence is malformed")
	}
	for _, g := range e.Groups {
		if g.Applied != g.Effective || g.Preserved != (g.Applied == PolicySourcePrecision) {
			return e, fmt.Errorf("tuning evidence group %q applied %q, resolved %q", g.ID, g.Applied, g.Effective)
		}
	}
	return e, nil
}
