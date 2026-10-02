package dashboard

import (
	"net/http"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// The Forge readiness projection: the latest preflights, the latest probe of
// each variant and the failure diagnostics, restated from the Go authorities
// (internal/app over the records under HACHIDORI_HOME/state/forge). The
// dashboard keeps none of it, derives no state from text and creates no
// "everything is fine" state of its own: an unknown stays unknown.

// ForgeState is what the home records about Forge readiness.
type ForgeState struct {
	Preflights  []PreflightRow
	Probes      []ProbeRow
	Diagnostics []DiagnosticRow
}

// PreflightRow is one recorded preflight report. Findings holds every finding
// that is not a pass, so blockers, warnings and unknowns are what is shown.
type PreflightRow struct {
	Kind, Model, Variant, Recipe, Device string
	At, Outcome                          string // outcome: blocked | attention | ready
	Pass, Warning, Blocker, Unknown      int
	Findings                             []FindingRow
	NotMeasured                          []string
}

// FindingRow is a non-passing finding: Status is warning, blocker or unknown.
type FindingRow struct{ ID, Status, Summary string }

// ProbeRow is the latest probe of a variant on a device. Result is passed or
// failed; a pass means only that the variant loaded and answered one valid
// typed decision, and says nothing about fidelity.
type ProbeRow struct {
	Variant, Device, Result, Phase, Error  string
	StartedAt                              string
	ManifestSHA256                         string
	Provider, DType, DeviceName            string
	Choice                                 string
	Confidence                             float64
	StartupMS, LoadMS, WarmupMS, RequestMS float64
}

// DiagnosticRow is one stored failure diagnostic.
type DiagnosticRow struct {
	ID, Kind, Phase, Model, Variant, Created, Error string
}

// forgeDiagnostic serves one stored Forge diagnostic for the operator to save:
// the bounded, redacted document itself, local only. It names the diagnostic
// by its stable identity, which can never name a path.
func (d *Dashboard) forgeDiagnostic(w http.ResponseWriter, r *http.Request) {
	va := d.cfg.Variants
	id := r.PathValue("id")
	if va == nil || !diagnostics.ValidForgeID(id) {
		http.NotFound(w, r)
		return
	}
	b, err := va.Diagnostic(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="`+id+`.json"`)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}

// The Forge workspace: the System One lifecycle of one variant, projected from
// the backend records. Stage states are derived on every render from the
// inventory (build and certification state), the latest probe, the status
// document (what serves) and the activation record; nothing is remembered
// here, and a stage the records do not show is "not yet", never a failure.

// Lifecycle stage states.
const (
	StageDone    = "done"    // the record shows it happened
	StagePending = "pending" // no record yet: not done, and not an error
	StageCurrent = "current" // in effect but waiting (an activation that applies on restart)
	StageStale   = "stale"   // a record exists for another manifest of this variant ID
	StageBad     = "bad"     // the record shows a problem, a failure or a rejecting verdict
)

// LifecycleStage is one stage of a variant's lifecycle. Label and Detail are
// catalog message IDs; Fact is a recorded value shown as it is.
type LifecycleStage struct {
	Name, Label, State, Detail, Fact string
}

// Class is the stage's CSS class in the shared stage list.
func (s LifecycleStage) Class() string {
	if s.State == StageBad {
		return "failed"
	}
	return s.State
}

// ForgeVariant is one variant with its lifecycle and the Forge records that
// belong to it.
type ForgeVariant struct {
	VariantRow
	Stages []LifecycleStage
	// NextStep is a catalog message ID: what the records say comes next ("" when
	// there is nothing to say).
	NextStep string
	// CanApply: an accepted, verifying variant that is not already serving.
	CanApply    bool
	Preflights  []PreflightRow
	Diagnostics []DiagnosticRow
}

// ForgeSource is a catalog model a variant can be built from.
type ForgeSource struct {
	ID           string
	Revision     string
	Materialized bool
	Recipes      []string
	Variants     int
}

// ForgeView is the Forge workspace view model.
type ForgeView struct {
	ModelsState
	Devices      []string
	ActiveDevice string
	Controls     bool // the Forge actions are configured
	Sources      []ForgeSource
	Items        []ForgeVariant
	// Preflights and Diagnostics that belong to no variant (materialize and
	// optimize reports, failed builds); the others are listed on their variant.
	Preflights  []PreflightRow
	Diagnostics []DiagnosticRow
}

// lifecycleOf projects the stages BUILT, PROBED, CERTIFIED and ACTIVE of one
// variant from its row.
func lifecycleOf(r VariantRow) []LifecycleStage {
	built := LifecycleStage{Name: "built", Label: "BUILT", State: StageDone, Detail: "built, published and verifiable", Fact: r.Recipe}
	if r.Problem != "" {
		built.State, built.Detail, built.Fact = StageBad, "the artifact does not verify", r.Problem
	}
	probed := LifecycleStage{Name: "probed", Label: "PROBED", State: StagePending, Detail: "not yet probed (optional: a certification runs its own probe)"}
	if p := r.Probe; p != nil {
		switch {
		case r.ProbeStale:
			probed.State, probed.Detail, probed.Fact = StageStale, "probed as another manifest of this variant ID", p.Device
		case p.Result == "passed":
			probed.State, probed.Detail, probed.Fact = StageDone, "loaded and answered one typed decision (not a certification)", p.Device
		default:
			probed.State, probed.Detail, probed.Fact = StageBad, "the probe failed", p.Phase
		}
	}
	certified := LifecycleStage{Name: "certified", Label: "CERTIFIED", State: StagePending, Detail: "not yet certified", Fact: r.CertificationNote}
	switch r.Certification {
	case eval.StateAccepted:
		certified.State, certified.Detail = StageDone, "accepted by the certification policy"
	case eval.StateRejected:
		certified.State, certified.Detail = StageBad, "rejected by the certification policy (recorded evidence; the variant is never applied)"
	}
	active := LifecycleStage{Name: "active", Label: "ACTIVE", State: StagePending, Detail: "not applied"}
	switch {
	case r.Running:
		active.State, active.Detail = StageDone, "serving now"
	case r.Active && r.Pending:
		active.State, active.Detail = StageCurrent, "active · applies on restart"
	case r.Active:
		active.State, active.Detail = StageDone, "active for the next start"
	}
	return []LifecycleStage{built, probed, certified, active}
}

// nextStepOf is what the records say comes next for a variant.
func nextStepOf(r VariantRow) string {
	switch {
	case r.Problem != "":
		return "Verify the variant: its artifact reports a problem."
	case !r.SourceMaterialized:
		return "Materialize the source model in Models, or let Apply materialize it."
	case r.Certification == eval.StateRejected:
		return "Rejected: build another recipe. A rejected variant is never applied."
	case r.Certification != eval.StateAccepted:
		return "Next: certify the variant with an evaluation dataset."
	case r.Running:
		return "Serving: this certified variant is the execution artifact."
	case r.Active && r.Pending:
		return "Applies on restart: restart the runtime in Models."
	}
	return "Next: apply the certified variant."
}

// forgeOf restates the Models view as the Forge workspace.
func forgeOf(mv *ModelsView) *ForgeView {
	fv := &ForgeView{ModelsState: mv.ModelsState, Devices: mv.Devices, ActiveDevice: mv.ActiveDevice, Controls: mv.VariantControls}
	count := map[string]int{}
	for _, row := range mv.Variants {
		count[row.SourceID]++
		it := ForgeVariant{VariantRow: row, Stages: lifecycleOf(row), NextStep: nextStepOf(row)}
		it.CanApply = row.Problem == "" && row.Certification == eval.StateAccepted && !row.Running
		for _, p := range mv.Forge.Preflights {
			if p.Variant == row.ID {
				it.Preflights = append(it.Preflights, p)
			}
		}
		for _, dg := range mv.Forge.Diagnostics {
			if dg.Variant == row.ID {
				it.Diagnostics = append(it.Diagnostics, dg)
			}
		}
		fv.Items = append(fv.Items, it)
	}
	for _, m := range mv.Inventory.Models {
		if setup.SupportsVariants(home.ModelManifest{Provider: m.Provider}) {
			fv.Sources = append(fv.Sources, ForgeSource{ID: m.ID, Revision: m.Revision, Materialized: m.Materialized, Recipes: optimize.RecipeNames(m.ID), Variants: count[m.ID]})
		}
	}
	for _, p := range mv.Forge.Preflights {
		if p.Variant == "" {
			fv.Preflights = append(fv.Preflights, p)
		}
	}
	for _, dg := range mv.Forge.Diagnostics {
		if dg.Variant == "" {
			fv.Diagnostics = append(fv.Diagnostics, dg)
		}
	}
	return fv
}

// forgePage renders the Forge workspace.
func (d *Dashboard) forgePage(w http.ResponseWriter, r *http.Request) {
	v := d.view("Forge", "forge")
	v.Live = false
	mv := d.modelsView(v)
	v.Models = mv
	v.Art = artifactsOf(v, mv)
	v.Forge = forgeOf(mv)
	d.renderView(w, "forge", v)
}
