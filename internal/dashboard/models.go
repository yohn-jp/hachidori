package dashboard

// The Models workspace: runtime, model and residency administration, and the
// explicit statement of which execution artifact serves now and at the next
// start. Everything here restates the maintenance authority's state and the
// /v1/status document; the dashboard decides and stores nothing.

import "net/http"

// Artifact kinds: the execution artifact is always one of these, never left
// implicit.
const (
	ArtifactSource  = "SOURCE"
	ArtifactVariant = "VARIANT"
)

// ArtifactView names the execution artifact of one side (what runs now, or what
// the activation record selects for the next start). A SOURCE is the upstream
// model artifact; a VARIANT is a derived, quantized artifact with its exact
// identity. Present is false when there is nothing to name (no worker running,
// no activation record).
type ArtifactView struct {
	Present bool
	Kind    string // ArtifactSource | ArtifactVariant
	Model   string
	Device  string // requested
	// The exact variant, from the status document or the inventory.
	Variant       string
	Scheme        string
	Bits          int
	DType         string // compute dtype of what is not quantized
	Certification string // accepted | experimental/uncertified | the inventory's state
	Experimental  bool
	// Provenance the worker reported (running side only): the device and dtype
	// it actually loaded on.
	ReportedDevice, ReportedDType string
}

// ArtifactsView is what serves now and what the next start serves.
type ArtifactsView struct {
	Running, Next ArtifactView
	// Differs: the next start serves another model, device or artifact than
	// the running worker (an activation waits for a restart).
	Differs bool
}

// artifactsOf restates the running artifact from the status document and the
// next-start artifact from the activation record and the inventory.
func artifactsOf(v view, mv *ModelsView) *ArtifactsView {
	a := &ArtifactsView{}
	if v.Running {
		rt := v.S.Runtime
		r := ArtifactView{Present: rt.ModelID != "", Kind: ArtifactSource, Model: rt.ModelID, Device: rt.Device,
			ReportedDevice: opt(v.S.Worker.Info, "device"), ReportedDType: opt(v.S.Worker.Info, "dtype")}
		if rv := rt.Variant; rv != nil {
			r.Kind, r.Variant, r.Scheme, r.Bits, r.DType, r.Certification = ArtifactVariant, rv.ID, rv.Scheme, rv.Bits, rv.DType, rv.Certification
			r.Experimental = rv.Certification != "accepted"
		}
		a.Running = r
	}
	if act := mv.Inventory.Active; act != nil && (act.ModelID != "" || mv.ActiveModel != "") {
		model := act.ModelID
		if model == "" {
			model = mv.ActiveModel // a record from before model selection names no model
		}
		n := ArtifactView{Present: true, Kind: ArtifactSource, Model: model, Device: act.Device}
		if act.Variant != "" {
			n.Kind, n.Variant, n.Experimental = ArtifactVariant, act.Variant, act.Experimental
			n.Certification = "accepted"
			if act.Experimental {
				n.Certification = "experimental/uncertified"
			}
			for _, e := range mv.Inventory.Variants {
				if e.ID == act.Variant {
					n.Scheme, n.Bits, n.DType = e.Scheme, e.Bits, e.DType
					if !act.Experimental {
						n.Certification = e.Certification
					}
				}
			}
		}
		a.Next = n
	}
	if a.Running.Present && a.Next.Present {
		a.Differs = a.Running.Model != a.Next.Model || a.Running.Device != a.Next.Device || a.Running.Variant != a.Next.Variant
	}
	return a
}

// modelsPage renders the Models workspace.
func (d *Dashboard) modelsPage(w http.ResponseWriter, r *http.Request) {
	v := d.view("Models", "models")
	v.Live = false
	v.Models = d.modelsView(v)
	v.Art = artifactsOf(v, v.Models)
	d.renderView(w, "models", v)
}
