package dashboard

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
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
	Resolution  *ForgeResolution
}

// ForgeResolvedValue is a selection mode and the value the composed backend
// operation resolved it to.
type ForgeResolvedValue struct {
	Mode, Value string
}

// ForgeResolution is the operator-visible execution plan returned by one
// Build & evaluate operation.
type ForgeResolution struct {
	Source, Recipe, Variant string
	CandidateDevice         ForgeResolvedValue
	ReferenceDevice         ForgeResolvedValue
	ReferenceDType          ForgeResolvedValue
	CandidateDType          string
}

// ForgeIntentForm keeps the semantic intent visible after a native picker
// posts back to the same page.
type ForgeIntentForm struct {
	Source, Profile, Dataset, Questions, Policy string
	Device, ReferenceDevice, ReferenceDType     string
	Provisioning, Err                           string
}

// ForgeBuildEvaluateRequest is one normal Forge intent. TuningProfile names a
// saved semantic tuning profile of Source; otherwise Profile is the canonical
// optimization recipe name for Source.
type ForgeBuildEvaluateRequest struct {
	Source, Profile, Device string
	TuningProfile           string
	ReferenceDevice         string
	ReferenceDType          string
	Dataset                 string
	Questions               []string
	Policy                  string
	Materialize             bool
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
	// VRAMBytes is the device memory the probe measured (the larger of
	// allocated and reserved); 0 when it measured none.
	VRAMBytes uint64
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
	// CanApply: an accepted, verifying variant that is not already serving
	// and is not outside the resource envelope.
	CanApply bool
	// Fit places the variant's measured device memory against the envelope;
	// a blocked fit rules it out of normal Apply and activation.
	Fit FitView
	// Tuning is the Tuning page of the profile the variant was built from (or
	// of its source when no profile is recorded): the route back.
	Tuning      string
	Preflights  []PreflightRow
	Diagnostics []DiagnosticRow
}

// Status is the candidate's one operator-facing state: a catalog message ID
// and its tone. It restates the records; it decides nothing.
func (v ForgeVariant) Status() struct{ Word, Tone string } {
	type st = struct{ Word, Tone string }
	switch {
	case v.Problem != "":
		return st{"Artifact problem", "bad"}
	case v.Running && v.Fit.Blocked():
		return st{"Serving · over memory budget", "bad"}
	case v.Running:
		return st{"Serving", "ok"}
	case v.Fit.Blocked():
		return st{"Over memory budget", "bad"}
	case v.Certification == eval.StateRejected:
		return st{"Rejected by evaluation", "bad"}
	case v.Active && v.Pending:
		return st{"Applies on restart", "warn"}
	case v.CanApply:
		return st{"Ready to apply", "ok"}
	case v.Certification != eval.StateAccepted:
		return st{"Not evaluated", "idle"}
	}
	return st{"Evaluated", "idle"}
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
	PathPicker   bool
	Form         ForgeIntentForm
	Sources      []ForgeSource
	Items        []ForgeVariant
	// Preflights and Diagnostics that belong to no variant (materialize and
	// optimize reports, failed builds); the others are listed on their variant.
	Preflights  []PreflightRow
	Diagnostics []DiagnosticRow
	// Envelope is the target device and memory budget candidates are applied
	// against.
	Envelope EnvelopeView
	// Profiles are the build choices: saved tuning profiles first, then the
	// canonical recipes.
	Profiles []ForgeProfileOption

	knownDatasets, knownQuestions, knownPolicies []string
}

// DatasetInput, QuestionsInput and PolicyInput are the evaluation resources
// of a build, named by their meaning.
func (fv *ForgeView) DatasetInput() resourceSelection {
	return semanticResource(fv.knownDatasets, fv.PathPicker, "Dataset", "dataset", fv.Form.Dataset, "", false, "/forge/pick", "dataset", "Add dataset…")
}

func (fv *ForgeView) QuestionsInput() resourceSelection {
	return semanticResource(fv.knownQuestions, fv.PathPicker, "Evaluation questions", "questions", fv.Form.Questions, "Questions in the dataset", true,
		"/forge/pick", "question-file", "Add question file…", "question-folder", "Add question folder…")
}

func (fv *ForgeView) PolicyInput() resourceSelection {
	return semanticResource(fv.knownPolicies, fv.PathPicker, "Certification policy", "policy", fv.Form.Policy, "Built-in policy", false, "/forge/pick", "policy", "Add policy…")
}

// ForgeProfileOption is one build choice of a source. Value is
// "tuning:<profile id>" for a saved tuning profile, else a recipe name.
type ForgeProfileOption struct {
	Source, Value, Label string
	Tuned                bool
}

// tuningProfilePrefix marks a saved tuning profile in the build choice.
const tuningProfilePrefix = "tuning:"

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
	case r.Certification == eval.StateRejected:
		return "Rejected: build another candidate. A rejected variant is never applied."
	case r.Certification != eval.StateAccepted:
		return "Not evaluated: build a candidate to produce evaluation evidence."
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

// forgePage renders the Forge workspace. A source and build choice in the
// query (the Tuning handoff) preselect the form when they name a choice the
// page offers; anything else is ignored.
func (d *Dashboard) forgePage(w http.ResponseWriter, r *http.Request) {
	d.renderForgePage(w, r, nil)
}

// forgeHandoff is the form a Tuning handoff selects, nil without a valid one.
func forgeHandoff(r *http.Request, fv *ForgeView) *ForgeIntentForm {
	source, profile := r.URL.Query().Get("source"), r.URL.Query().Get("profile")
	for _, p := range fv.Profiles {
		if p.Source == source && p.Value == profile {
			return &ForgeIntentForm{Source: source, Profile: profile, Provisioning: "auto"}
		}
	}
	return nil
}

func (d *Dashboard) renderForgePage(w http.ResponseWriter, r *http.Request, form *ForgeIntentForm) {
	v := d.forgeView()
	v.Forge.PathPicker = d.cfg.PathPicker != nil
	if form == nil && r.Method == http.MethodGet {
		form = forgeHandoff(r, v.Forge)
	}
	if form == nil {
		form = &ForgeIntentForm{Provisioning: "auto"}
		if len(v.Forge.Sources) > 0 {
			form.Source = v.Forge.Sources[0].ID
			for _, p := range v.Forge.Profiles {
				if p.Source == form.Source {
					form.Profile = p.Value
					break
				}
			}
		}
	}
	v.Forge.Form = *form
	d.renderView(w, "forge", v)
}

// forgeView is the Forge workspace: the candidates with their fit against
// the envelope and the build choices.
func (d *Dashboard) forgeView() view {
	v := d.view("Forge", "forge")
	v.Live = false
	mv := d.modelsView(v)
	v.Models = mv
	v.Art = artifactsOf(v, mv)
	v.Forge = forgeOf(mv)
	v.Forge.Envelope = d.envelopeOf(v)
	v.Forge.knownDatasets, v.Forge.knownQuestions, v.Forge.knownPolicies = d.resourceCatalog(resDataset), d.resourceCatalog(resQuestions), d.resourceCatalog(resPolicy)
	for i := range v.Forge.Items {
		it := &v.Forge.Items[i]
		it.Fit = variantFit(v, v.Forge.Envelope, it.VariantRow)
		it.CanApply = it.CanApply && !it.Fit.Blocked()
		it.Tuning = tuningLocation(it.SourceID, "")
		if d.cfg.Tuning != nil {
			if id, err := d.variantProfile(it.SourceID, it.ID); err == nil {
				it.Tuning = tuningLocation(it.SourceID, id)
			}
		}
	}
	for _, src := range v.Forge.Sources {
		if d.cfg.Tuning != nil {
			if ps, err := d.cfg.Tuning.Profiles(src.ID); err == nil {
				for _, p := range ps {
					v.Forge.Profiles = append(v.Forge.Profiles, ForgeProfileOption{Source: src.ID, Value: tuningProfilePrefix + p.ID(), Tuned: true,
						Label: short12(p.ID()) + " · " + objectiveLabel(p.Objective) + " · " + strconv.Itoa(pinnedCount(p)) + " preserved"})
				}
			}
		}
		for _, rc := range src.Recipes {
			v.Forge.Profiles = append(v.Forge.Profiles, ForgeProfileOption{Source: src.ID, Value: rc, Label: rc})
		}
	}
	return v
}

// variantFit is the variant's device memory against the envelope: measured
// on the device while it serves, else by its latest probe of this exact
// manifest, else NOT_CHECKED. Nothing is estimated here.
func variantFit(v view, e EnvelopeView, row VariantRow) FitView {
	if row.Running {
		total, ok1 := num(v.S.Worker.Accelerator, "memory_total")
		free, ok2 := num(v.S.Worker.Accelerator, "memory_free")
		if ok1 && ok2 && total > free {
			return fitOf(e.Envelope, tuning.Usage{Bytes: uint64(total - free)}, Measured, "device memory in use while this candidate serves")
		}
	}
	if p := row.Probe; p != nil && !row.ProbeStale && p.Result == "passed" && p.VRAMBytes > 0 {
		return fitOf(e.Envelope, tuning.Usage{Bytes: p.VRAMBytes}, Measured, "probe on "+p.Device)
	}
	return fitOf(e.Envelope, tuning.Usage{}, NotChecked, "no measurement of this candidate's device memory")
}

// errOutsideEnvelope refuses a normal Apply or activation of a candidate whose
// measured memory exceeds the budget.
var errOutsideEnvelope = errors.New("the candidate exceeds the memory budget and is not applied; return to Tuning to change the plan or the budget")

// forgeBlocked reports whether the variant is outside the envelope.
func (d *Dashboard) forgeBlocked(variant string) bool {
	for _, it := range d.forgeView().Forge.Items {
		if it.ID == variant {
			return it.Fit.Blocked()
		}
	}
	return false
}

// forgePick uses the desktop's existing native picker capability for semantic
// evaluation resources. Browser dashboards keep the same absolute-path fields.
func (d *Dashboard) forgePick(w http.ResponseWriter, r *http.Request) {
	form := forgeIntentForm(r)
	if d.cfg.PathPicker == nil {
		form.Err = "native path selection is unavailable"
		d.renderForgePage(w, r, &form)
		return
	}
	var (
		path string
		err  error
	)
	switch r.PostFormValue("pick") {
	case "dataset":
		path, err = d.cfg.PathPicker.PickOpen(r.Context(), "Choose an evaluation dataset")
		if err == nil {
			form.Dataset = path
		}
	case "question-file":
		path, err = d.cfg.PathPicker.PickOpen(r.Context(), "Choose a Question Definition")
		if err == nil {
			form.Questions = appendLine(form.Questions, path)
		}
	case "question-folder":
		path, err = d.cfg.PathPicker.PickFolder(r.Context(), "Choose a Question Definition folder")
		if err == nil {
			form.Questions = appendLine(form.Questions, path)
		}
	case "policy":
		path, err = d.cfg.PathPicker.PickOpen(r.Context(), "Choose a certification policy")
		if err == nil {
			form.Policy = path
		}
	default:
		form.Err = "unknown native picker operation"
	}
	if err != nil && !pickWasCancelled(err) {
		form.Err = "choosing a path: " + err.Error()
	}
	d.renderForgePage(w, r, &form)
}

func forgeIntentForm(r *http.Request) ForgeIntentForm {
	f := func(k string) string { return strings.TrimSpace(r.PostFormValue(k)) }
	return ForgeIntentForm{
		Source: f("source"), Profile: f("profile"), Dataset: strings.TrimSpace(resourceValue(r, "dataset")), Questions: resourceValue(r, "questions"),
		Policy: strings.TrimSpace(resourceValue(r, "policy")),
		Device: f("device"), ReferenceDevice: f("reference_device"), ReferenceDType: f("reference_dtype"), Provisioning: f("provisioning"),
	}
}

// forgeBuildEvaluateRequest accepts semantic resources and selection overrides
// only. Empty device/precision fields are Auto; the backend resolves them.
func forgeBuildEvaluateRequest(r *http.Request) (ForgeBuildEvaluateRequest, error) {
	form := forgeIntentForm(r)
	req := ForgeBuildEvaluateRequest{
		Source: form.Source, Profile: form.Profile, Device: form.Device,
		ReferenceDevice: form.ReferenceDevice, ReferenceDType: form.ReferenceDType,
		Materialize: form.Provisioning == "auto",
	}
	if id, ok := strings.CutPrefix(form.Profile, tuningProfilePrefix); ok {
		req.Profile, req.TuningProfile = "", id
	}
	var err error
	if req.Dataset, err = absPath(form.Dataset, "evaluation dataset"); err != nil {
		return req, err
	}
	for _, q := range lines(form.Questions) {
		p, err := absPath(q, "Question Definition")
		if err != nil {
			return req, err
		}
		req.Questions = append(req.Questions, p)
	}
	if form.Policy != "" {
		if req.Policy, err = absPath(form.Policy, "policy"); err != nil {
			return req, err
		}
	}
	if form.Provisioning != "auto" && form.Provisioning != "never" {
		return req, errors.New("choose Auto or Do not provision for missing prerequisites")
	}
	return req, nil
}
