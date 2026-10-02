package dashboard

import (
	"net/http"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
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
