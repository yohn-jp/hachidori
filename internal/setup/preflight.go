package setup

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Preflight is the deterministic readiness check that runs before expensive
// Forge work (materializing a multi-GB source, building a variant, probing or
// certifying one). This file is its typed result and the host observations it
// is built from; the checks themselves live with the authority that owns what
// they inspect (internal/optimize for recipes and variants).
//
// A report never says more than was measured. Where Hachidori cannot know
// whether something will fit it says UNKNOWN: installed RAM is not a promise
// that a model fits, and the absence of an observation is not a pass.

// PreflightSchema identifies the report. A report of this schema carries its
// Binding: the identities its conclusion was drawn from.
const PreflightSchema = "hachidori.forge-preflight.v2"

// LegacyPreflightSchema is the report written before reports were bound to
// their identities. Such a report stays readable but cannot prove what it was
// about, so it is never current evidence (EvidenceLegacy).
const LegacyPreflightSchema = "hachidori.forge-preflight.v1"

// Evidence states of a recorded preflight report, judged when it is read back
// against the identities the home and the catalog hold now. Only a current
// report may be presented as the readiness of its target; a recorded outcome
// is never an authorization for anything.
const (
	// EvidenceCurrent: the report's binding equals the binding of its target
	// now.
	EvidenceCurrent = "current"
	// EvidenceStale: an identity the report was bound to has changed since.
	EvidenceStale = "stale"
	// EvidenceLegacy: the report predates bindings and proves nothing about
	// the current identities.
	EvidenceLegacy = "legacy"
)

// PreflightBinding is the exact target and identity set a preflight report is
// evidence about. Every value is derived from an existing authority (the
// catalog, the source and variant manifests, the recipe catalog, the Runtime
// Spec and the materialized runtime manifest); an empty value means the
// authority did not resolve. A report is current only while the binding
// recomputed for its target equals the recorded one.
type PreflightBinding struct {
	Kind   string `json:"kind"`
	Device string `json:"device,omitempty"`
	// Source: the catalog model and its pinned identity, and the digest of
	// the materialized source manifest ("" when it is not materialized).
	Model                string `json:"model,omitempty"`
	Provider             string `json:"provider,omitempty"`
	SourceRepo           string `json:"source_repo,omitempty"`
	SourceRevision       string `json:"source_revision,omitempty"`
	SourceFilesSHA256    string `json:"source_files_sha256,omitempty"`
	SourceManifestSHA256 string `json:"source_manifest_sha256,omitempty"`
	// Runtime: the Runtime Spec identity the operation uses (the serving
	// runtime of Device, or the optimizer runtime) and the digest of its
	// materialized manifest ("" when it is not materialized).
	Runtime               string `json:"runtime,omitempty"`
	RuntimeManifestSHA256 string `json:"runtime_manifest_sha256,omitempty"`
	// Variant and recipe: the variant manifest digest, and the recipe the
	// operation applies (optimize) or the variant was built with.
	Variant               string `json:"variant,omitempty"`
	VariantManifestSHA256 string `json:"variant_manifest_sha256,omitempty"`
	Recipe                string `json:"recipe,omitempty"`
	RecipeSHA256          string `json:"recipe_sha256,omitempty"`
	// ReferenceDType is the requested reference dtype of a certification,
	// which sizes its RAM bound.
	ReferenceDType string `json:"reference_dtype,omitempty"`
}

// Mismatch names the dimensions in which b differs from cur, in a fixed
// order; none means b still holds.
func (b PreflightBinding) Mismatch(cur PreflightBinding) []string {
	var out []string
	for _, d := range []struct {
		name     string
		was, now string
	}{
		{"kind", b.Kind, cur.Kind},
		{"device", b.Device, cur.Device},
		{"model", b.Model, cur.Model},
		{"provider", b.Provider, cur.Provider},
		{"source_repo", b.SourceRepo, cur.SourceRepo},
		{"source_revision", b.SourceRevision, cur.SourceRevision},
		{"source_files_sha256", b.SourceFilesSHA256, cur.SourceFilesSHA256},
		{"source_manifest_sha256", b.SourceManifestSHA256, cur.SourceManifestSHA256},
		{"runtime", b.Runtime, cur.Runtime},
		{"runtime_manifest_sha256", b.RuntimeManifestSHA256, cur.RuntimeManifestSHA256},
		{"variant", b.Variant, cur.Variant},
		{"variant_manifest_sha256", b.VariantManifestSHA256, cur.VariantManifestSHA256},
		{"recipe", b.Recipe, cur.Recipe},
		{"recipe_sha256", b.RecipeSHA256, cur.RecipeSHA256},
		{"reference_dtype", b.ReferenceDType, cur.ReferenceDType},
	} {
		if d.was != d.now {
			out = append(out, d.name)
		}
	}
	return out
}

// FindingStatus is the state of one finding.
type FindingStatus string

const (
	// FindingPass: measured, and nothing is wrong.
	FindingPass FindingStatus = "pass"
	// FindingWarning: measured, and something may go wrong; the work is not
	// refused.
	FindingWarning FindingStatus = "warning"
	// FindingBlocker: a known condition under which the work cannot succeed.
	FindingBlocker FindingStatus = "blocker"
	// FindingUnknown: Hachidori cannot know (not observable here, or only
	// the work itself would show it).
	FindingUnknown FindingStatus = "unknown"
)

// Preflight kinds: the operation a report gates.
const (
	PreflightMaterialize = "materialize"
	PreflightOptimize    = "optimize"
	PreflightProbe       = "probe"
	PreflightCertify     = "certify"
)

// Finding areas.
const (
	AreaIdentity    = "identity"
	AreaRuntime     = "runtime"
	AreaRecipe      = "recipe"
	AreaStorage     = "storage"
	AreaMemory      = "memory"
	AreaAccelerator = "accelerator"
)

// Outcomes of a whole report.
const (
	// OutcomeBlocked: at least one blocker; the work must not start.
	OutcomeBlocked = "blocked"
	// OutcomeAttention: no blocker, but at least one warning or unknown.
	OutcomeAttention = "attention"
	// OutcomeReady: every finding passed.
	OutcomeReady = "ready"
)

// Finding is one checked fact. ID is stable and machine readable; Summary is
// for people and is never parsed. Facts are the measured values.
type Finding struct {
	ID      string         `json:"id"`
	Area    string         `json:"area"`
	Status  FindingStatus  `json:"status"`
	Summary string         `json:"summary"`
	Facts   map[string]any `json:"facts,omitempty"`
}

// PreflightReport is the machine-readable preflight result. The CLI prints it
// and the desktop projects it; nobody derives state from Summary text.
type PreflightReport struct {
	Schema    string `json:"schema"`
	Kind      string `json:"kind"`
	Model     string `json:"model"`
	Variant   string `json:"variant,omitempty"`
	Recipe    string `json:"recipe,omitempty"`
	Device    string `json:"device,omitempty"`
	CreatedAt string `json:"created_at"`
	// Binding is what the report is evidence about. A report without one
	// (the legacy schema) is never current.
	Binding  *PreflightBinding `json:"binding,omitempty"`
	Outcome  string            `json:"outcome"`
	Findings []Finding         `json:"findings"`
	Counts   Counts            `json:"counts"`
	// NotMeasured names what this build of the preflight does not measure, so
	// a ready report is never read as a promise about it.
	NotMeasured []string `json:"not_measured,omitempty"`
}

// Counts are the findings per status.
type Counts struct {
	Pass    int `json:"pass"`
	Warning int `json:"warning"`
	Blocker int `json:"blocker"`
	Unknown int `json:"unknown"`
}

// NewPreflightReport starts a report.
func NewPreflightReport(kind, model string, now time.Time) *PreflightReport {
	return &PreflightReport{Schema: PreflightSchema, Kind: kind, Model: model, CreatedAt: now.UTC().Format(time.RFC3339Nano),
		Findings: []Finding{}, NotMeasured: []string{
			"peak RAM and VRAM use of the operation (only lower bounds from known artifact sizes are compared)",
			"throughput and duration",
		}}
}

// Add records a finding and updates the outcome.
func (r *PreflightReport) Add(f Finding) {
	r.Findings = append(r.Findings, f)
	r.finalize()
}

func (r *PreflightReport) finalize() {
	r.Counts = Counts{}
	for _, f := range r.Findings {
		switch f.Status {
		case FindingPass:
			r.Counts.Pass++
		case FindingWarning:
			r.Counts.Warning++
		case FindingBlocker:
			r.Counts.Blocker++
		case FindingUnknown:
			r.Counts.Unknown++
		}
	}
	switch {
	case r.Counts.Blocker > 0:
		r.Outcome = OutcomeBlocked
	case r.Counts.Warning > 0 || r.Counts.Unknown > 0:
		r.Outcome = OutcomeAttention
	default:
		r.Outcome = OutcomeReady
	}
}

// Blockers are the findings that refuse the work.
func (r PreflightReport) Blockers() []Finding {
	var out []Finding
	for _, f := range r.Findings {
		if f.Status == FindingBlocker {
			out = append(out, f)
		}
	}
	return out
}

// Blocked reports whether the work must not start.
func (r PreflightReport) Blocked() bool { return r.Counts.Blocker > 0 }

// PreflightError is the refusal of an operation whose preflight found a
// blocker. It carries the whole report.
type PreflightError struct{ Report PreflightReport }

func (e *PreflightError) Error() string {
	var parts []string
	for _, f := range e.Report.Blockers() {
		parts = append(parts, f.ID+": "+f.Summary)
	}
	return fmt.Sprintf("preflight for %s refused the operation before any expensive work: %s", e.Report.Kind, strings.Join(parts, "; "))
}

// AsPreflightError unwraps a PreflightError.
func AsPreflightError(err error) (*PreflightError, bool) {
	var pe *PreflightError
	return pe, errors.As(err, &pe)
}

// Memory is what the host reports about its RAM. A value is meaningful only
// when its Known flag is set.
type Memory struct {
	Total          uint64 `json:"total_bytes"`
	TotalKnown     bool   `json:"total_known"`
	Available      uint64 `json:"available_bytes"`
	AvailableKnown bool   `json:"available_known"`
}

// Host is the observation surface of the machine the preflight runs on. The
// zero value observes nothing; DefaultHost observes the real machine and tests
// substitute fixed values.
type Host struct {
	// FreeDisk reports the bytes available to the current user on the volume
	// that holds path.
	FreeDisk func(path string) (free uint64, ok bool)
	// Memory reports host RAM.
	Memory func() Memory
}

// DefaultHost observes the real machine: free disk and RAM where the platform
// exposes them, and nothing otherwise.
func DefaultHost() Host { return Host{FreeDisk: freeDisk, Memory: hostMemory} }

func (h Host) freeDisk(path string) (uint64, bool) {
	if h.FreeDisk == nil {
		return 0, false
	}
	return h.FreeDisk(nearestExisting(path))
}

func (h Host) memory() Memory {
	if h.Memory == nil {
		return Memory{}
	}
	return h.Memory()
}

// FreeDiskAt is the free space of the volume holding path, measured on its
// nearest existing ancestor (a directory that is about to be created).
func (h Host) FreeDiskAt(path string) (uint64, bool) { return h.freeDisk(path) }

// Mem is the host's RAM report.
func (h Host) Mem() Memory { return h.memory() }

func nearestExisting(path string) string {
	p := filepath.Clean(path)
	for {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		parent := filepath.Dir(p)
		if parent == p {
			return p
		}
		p = parent
	}
}

// WritableDir reports whether files can be created under dir (a missing
// directory is created). It writes and removes one probe file.
func WritableDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".hachidori-writable-*")
	if err != nil {
		return err
	}
	name := f.Name()
	f.Close()
	return os.Remove(name)
}

// Bytes renders a byte count in binary units for finding summaries.
func Bytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	f, i := float64(n)/unit, 0
	for ; f >= unit && i < 4; i++ {
		f /= unit
	}
	return fmt.Sprintf("%.1f %ciB", f, "KMGT"[i])
}

// SortedFindings orders findings by area then ID, for stable output.
func SortedFindings(fs []Finding) []Finding {
	out := append([]Finding(nil), fs...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Area != out[j].Area {
			return out[i].Area < out[j].Area
		}
		return out[i].ID < out[j].ID
	})
	return out
}
