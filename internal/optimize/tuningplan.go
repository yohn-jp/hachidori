package optimize

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// checkPlan binds a layer-wise tuning request to its plan before anything
// expensive runs: the provenance names exactly this plan, and the plan is the
// one the compiled recipe was produced with. A legacy provenance carries no
// plan.
func checkPlan(req Request, compiled home.Recipe) error {
	layerwise := req.Tuning.Schema == home.TuningProvenanceSchema
	switch {
	case !layerwise && req.Plan != nil:
		return errors.New("a legacy tuning provenance cannot carry a layer-wise plan")
	case layerwise && req.Plan == nil:
		return errors.New("layer-wise tuning provenance requires the resolved plan produced by its compiler")
	case !layerwise:
		return nil
	}
	if err := req.Plan.Validate(); err != nil {
		return fmt.Errorf("tuning plan: %w", err)
	}
	if req.Plan.SHA256() != req.Tuning.PlanSHA256 {
		return errors.New("the tuning plan is not the one the provenance names")
	}
	if req.Plan.Recipe != compiled.Name || req.Plan.RecipeSHA256 != compiled.SHA256() {
		return errors.New("the tuning plan was not resolved for the compiled recipe")
	}
	return nil
}

// stagedReport is the part of the optimizer's report the plan check reads.
type stagedReport struct {
	QuantizedModules int `json:"quantized_modules"`
	Preserved        map[string]struct {
		WrittenDType string `json:"written_dtype"`
	} `json:"preserved_modules"`
}

// writeTuningEvidence derives, from the optimizer's own report and the staged
// carried files, the transformation actually applied to every group, requires
// it to equal the plan's effective policy, and writes the per-group evidence
// into the stage. Nothing is inferred beyond what the report and files state.
func writeTuningEvidence(stage string, model home.ModelManifest, plan home.TuningPlan) error {
	var rep stagedReport
	if err := home.ReadJSON(filepath.Join(stage, setup.OptimizerReportFile), &rep); err != nil {
		return fmt.Errorf("the optimizer report cannot be read to verify the tuning plan: %w", err)
	}
	ev := home.TuningEvidence{Schema: home.TuningEvidenceSchema, PlanSHA256: plan.SHA256(), AutoPolicy: plan.AutoPolicy}
	mappedPreserved, mappedQuantized := 0, 0
	for _, g := range plan.Groups {
		applied, dtype, err := appliedPolicy(stage, model, g, rep)
		if err != nil {
			return err
		}
		if applied != g.Effective {
			return fmt.Errorf("the optimizer applied %s to group %s, but the plan resolved %s (%s): the build is refused, not recorded",
				applied, g.ID, g.Effective, g.Selection)
		}
		if applied == home.PolicySourcePrecision {
			mappedPreserved += len(g.Modules)
		} else {
			mappedQuantized += len(g.Modules)
		}
		ev.Groups = append(ev.Groups, home.TuningGroupApplied{ID: g.ID, Selection: g.Selection, Requested: g.Requested, Effective: g.Effective,
			Required: g.Required, Applied: applied, Preserved: applied == home.PolicySourcePrecision,
			Modules: len(g.Modules), Files: len(g.Files), WrittenDType: dtype})
	}
	if rep.QuantizedModules < mappedQuantized {
		return fmt.Errorf("the plan quantizes %d modules but the optimizer reports quantizing %d", mappedQuantized, rep.QuantizedModules)
	}
	ev.UnmappedQuantized = rep.QuantizedModules - mappedQuantized
	ev.UnmappedPreserved = len(rep.Preserved) - mappedPreserved
	return os.WriteFile(filepath.Join(stage, home.TuningEvidenceFile), ev.Canonical(), 0o644)
}

// appliedPolicy is the policy the staged artifact shows a group received.
func appliedPolicy(stage string, model home.ModelManifest, g home.TuningGroupPlan, rep stagedReport) (applied, dtype string, err error) {
	if len(g.Modules) == 0 {
		// A carried file is preserved exactly when it is byte-identical to the
		// pinned source file.
		for _, f := range g.Files {
			sum, err := fileSHA256(filepath.Join(stage, filepath.FromSlash(f)))
			if err != nil {
				return "", "", fmt.Errorf("group %s: carried file %s: %w", g.ID, f, err)
			}
			if sum != model.Files[f] {
				return "", "", fmt.Errorf("group %s: carried file %s differs from the pinned source file", g.ID, f)
			}
		}
		return home.PolicySourcePrecision, "", nil
	}
	dtypes := map[string]bool{}
	kept := 0
	for _, m := range g.Modules {
		if p, ok := rep.Preserved[m]; ok {
			switch p.WrittenDType {
			case "BF16", "F16", "F32":
			default:
				return "", "", fmt.Errorf("group %s: module %s was preserved but written as %q", g.ID, m, p.WrittenDType)
			}
			dtypes[p.WrittenDType] = true
			kept++
		}
	}
	switch kept {
	case 0:
		return home.PolicyW4A16, "", nil
	case len(g.Modules):
		names := make([]string, 0, len(dtypes))
		for d := range dtypes {
			names = append(names, d)
		}
		sort.Strings(names)
		return home.PolicySourcePrecision, strings.Join(names, "+"), nil
	}
	return "", "", fmt.Errorf("group %s: the optimizer preserved %d of its %d modules; a group is transformed as a whole", g.ID, kept, len(g.Modules))
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
