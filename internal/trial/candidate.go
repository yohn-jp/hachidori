package trial

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

const (
	// CandidateSchema versions the Candidate document and its identity.
	CandidateSchema = "hachidori.tuning-candidate/1"
	// EvidenceSchema versions trial Evidence. It is a different schema from
	// Decision Evidence, Forge runs and certification on purpose: trial Evidence
	// can never be read as any of them.
	EvidenceSchema = "hachidori.trial-evidence/1"
	// MaterializationSchema versions the record that a Candidate became a Variant.
	MaterializationSchema = "hachidori.tuning-materialization/1"

	// CertificationNone is what trial Evidence says about certification: it is
	// ephemeral measurement of the source model with a plan applied, not
	// certification of any artifact.
	CertificationNone = "NOT_CERTIFIED"
)

// ProfileRef names the saved tuning profile a candidate's plan was resolved
// from. It is provenance; the plan is the authority a build reproduces.
type ProfileRef struct {
	ID              string `json:"id"`
	SHA256          string `json:"sha256"`
	AnalysisSHA256  string `json:"analysis_sha256"`
	CompilerVersion string `json:"compiler_version"`
}

// Candidate is a reproducible tuning result: the exact source, the profile and
// the resolved plan, and the exact set of transformed components a trial was
// composed from. It exists without any model artifact. Its identity covers
// those inputs only; measurements are Evidence owned by it.
type Candidate struct {
	Schema           string             `json:"schema"`
	ID               string             `json:"id"`
	Source           home.VariantSource `json:"source"`
	Profile          ProfileRef         `json:"profile"`
	Plan             home.TuningPlan    `json:"plan"`
	PlanSHA256       string             `json:"plan_sha256"`
	Components       []ComponentRef     `json:"components"`
	ComponentsSHA256 string             `json:"components_sha256"`
	CreatedAt        string             `json:"created_at"`
}

type candidateBinding struct {
	Schema     string             `json:"schema"`
	Source     home.VariantSource `json:"source"`
	Profile    ProfileRef         `json:"profile"`
	PlanSHA256 string             `json:"plan_sha256"`
	Components string             `json:"components_sha256"`
}

func componentsDigest(refs []ComponentRef) string {
	sorted := append([]ComponentRef(nil), refs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Group < sorted[j].Group })
	b, err := json.Marshal(sorted)
	if err != nil {
		panic(err)
	}
	return sha256Hex(b)
}

// NewCandidate derives a candidate from a measured trial and the profile its
// plan was resolved from. The candidate's identity is a function of the source,
// profile, plan and component set.
func NewCandidate(source home.VariantSource, profile ProfileRef, r Result, now time.Time) (Candidate, error) {
	c := Candidate{Schema: CandidateSchema, Source: source, Profile: profile, Plan: r.Plan, PlanSHA256: r.Plan.SHA256(),
		Components: append([]ComponentRef(nil), r.Components...), CreatedAt: now.UTC().Format(time.RFC3339)}
	c.ComponentsSHA256 = componentsDigest(c.Components)
	b, err := json.Marshal(candidateBinding{Schema: CandidateSchema, Source: source, Profile: profile, PlanSHA256: c.PlanSHA256, Components: c.ComponentsSHA256})
	if err != nil {
		return Candidate{}, err
	}
	c.ID = sha256Hex(b)
	return c, c.Validate()
}

// Validate checks the candidate is well formed and that its identity is the one
// its contents derive.
func (c Candidate) Validate() error {
	if c.Schema != CandidateSchema {
		return fmt.Errorf("candidate schema %q, want %q", c.Schema, CandidateSchema)
	}
	if c.Source.ID == "" || c.Source.Revision == "" || c.Source.FilesSHA256 == "" {
		return errors.New("candidate does not name its exact source")
	}
	if len(c.Profile.ID) != 64 || len(c.Profile.SHA256) != 64 || len(c.Profile.AnalysisSHA256) != 64 {
		return errors.New("candidate does not name its exact tuning profile")
	}
	if err := c.Plan.Validate(); err != nil {
		return fmt.Errorf("candidate plan: %w", err)
	}
	if c.Plan.SHA256() != c.PlanSHA256 {
		return errors.New("candidate plan does not match its digest")
	}
	if len(c.Components) != len(c.Plan.Groups) {
		return fmt.Errorf("candidate names %d components for %d plan groups", len(c.Components), len(c.Plan.Groups))
	}
	byGroup := map[string]ComponentRef{}
	for _, r := range c.Components {
		if r.Component == "" || r.Group == "" {
			return errors.New("candidate component reference is incomplete")
		}
		byGroup[r.Group] = r
	}
	for _, g := range c.Plan.Groups {
		r, ok := byGroup[g.ID]
		if !ok || r.Policy != g.Effective {
			return fmt.Errorf("candidate component of group %s is not the plan's %s", g.ID, g.Effective)
		}
		if g.Effective == home.PolicyW4A16 && r.Digest == "" {
			return fmt.Errorf("candidate component of group %s carries no content digest", g.ID)
		}
	}
	if c.ComponentsSHA256 != componentsDigest(c.Components) {
		return errors.New("candidate components do not match their digest")
	}
	want, err := json.Marshal(candidateBinding{Schema: CandidateSchema, Source: c.Source, Profile: c.Profile, PlanSHA256: c.PlanSHA256, Components: c.ComponentsSHA256})
	if err != nil {
		return err
	}
	if c.ID != sha256Hex(want) {
		return errors.New("candidate identity does not match its contents")
	}
	return nil
}

// Evidence is one measurement of a candidate: what was measured, on which
// dataset, in which evaluation mode, and how the trial that produced it was
// assembled. It is immutable and states in its own fields that it is not
// certification.
type Evidence struct {
	Schema           string             `json:"schema"`
	CandidateID      string             `json:"candidate_id"`
	Source           home.VariantSource `json:"source"`
	ProfileID        string             `json:"profile_id"`
	PlanSHA256       string             `json:"plan_sha256"`
	ComponentsSHA256 string             `json:"components_sha256"`
	// Ephemeral is always true: the measured object was an in-RAM trial of the
	// source, never an artifact. Certification is always CertificationNone.
	Ephemeral     bool        `json:"ephemeral"`
	Certification string      `json:"certification"`
	Execution     Execution   `json:"execution"`
	Measurement   Measurement `json:"measurement"`
	RecordedAt    string      `json:"recorded_at"`
}

// Execution is the trial execution and cache evidence sufficient to reproduce
// the measurement's setup.
type Execution struct {
	Assembly     Assembly `json:"assembly"`
	Cache        Stats    `json:"cache"`
	Resource     Resource `json:"resource"`
	EvaluationMS float64  `json:"evaluation_ms"`
	TotalMS      float64  `json:"total_ms"`
}

// NewEvidence binds a measured trial to its candidate.
func NewEvidence(c Candidate, r Result, now time.Time) (Evidence, error) {
	if r.Plan.SHA256() != c.PlanSHA256 || componentsDigest(r.Components) != c.ComponentsSHA256 {
		return Evidence{}, errors.New("the trial result is not the one the candidate was derived from")
	}
	e := Evidence{Schema: EvidenceSchema, CandidateID: c.ID, Source: c.Source, ProfileID: c.Profile.ID, PlanSHA256: c.PlanSHA256,
		ComponentsSHA256: c.ComponentsSHA256, Ephemeral: true, Certification: CertificationNone,
		Execution:   Execution{Assembly: r.Assembly, Cache: r.Cache, Resource: r.Resource, EvaluationMS: r.EvaluationMS, TotalMS: r.TotalMS},
		Measurement: r.Measurement, RecordedAt: now.UTC().Format(time.RFC3339)}
	return e, e.Validate()
}

// Canonical is the evidence's stored encoding.
func (e Evidence) Canonical() []byte {
	b, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		panic(err)
	}
	return append(b, '\n')
}

// ID is the evidence digest.
func (e Evidence) ID() string { return sha256Hex(e.Canonical()) }

// Validate refuses evidence that could be mistaken for certification or for a
// measurement of an artifact.
func (e Evidence) Validate() error {
	switch {
	case e.Schema != EvidenceSchema:
		return fmt.Errorf("trial evidence schema %q, want %q", e.Schema, EvidenceSchema)
	case len(e.CandidateID) != 64 || len(e.PlanSHA256) != 64 || len(e.ComponentsSHA256) != 64:
		return errors.New("trial evidence does not bind its candidate, plan and components")
	case !e.Ephemeral:
		return errors.New("trial evidence must state that it measured an ephemeral trial")
	case e.Certification != CertificationNone:
		return fmt.Errorf("trial evidence certification is %q, not %q", e.Certification, CertificationNone)
	}
	return e.Measurement.Validate()
}

// Materialization records that a Candidate was built by Forge into an
// immutable Variant, bound to that Variant's own manifest.
type Materialization struct {
	Schema          string `json:"schema"`
	CandidateID     string `json:"candidate_id"`
	PlanSHA256      string `json:"plan_sha256"`
	VariantID       string `json:"variant_id"`
	VariantManifest string `json:"variant_manifest_sha256"`
	RecordedAt      string `json:"recorded_at"`
}

// Status is a candidate's lifecycle position. A candidate is never certified;
// certification belongs to the clean-loaded Variant.
type Status struct {
	Measurements int      `json:"measurements"`
	Variants     []string `json:"variants,omitempty"`
}

// Stage names the stage of a candidate for display.
func (s Status) Stage() string {
	switch {
	case len(s.Variants) > 0:
		return "materialized"
	case s.Measurements > 0:
		return "measured"
	}
	return "saved"
}

// ----------------------------------------------------------------------------
// Store: HACHIDORI_HOME/state/tuning-trials/<candidate id>/

// Dir is the candidate's directory in the home.
func Dir(h home.Home, id string) string { return h.Path("state", "tuning-trials", id) }

func validID(id string) bool {
	if len(id) != 64 {
		return false
	}
	for _, r := range id {
		if !(r >= '0' && r <= '9' || r >= 'a' && r <= 'f') {
			return false
		}
	}
	return true
}

// Save stores the candidate and one measurement of it. Both documents are
// immutable: saving the same candidate again adds another Evidence document and
// never rewrites the candidate.
func Save(h home.Home, c Candidate, e Evidence) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if err := e.Validate(); err != nil {
		return err
	}
	if e.CandidateID != c.ID || e.PlanSHA256 != c.PlanSHA256 || e.ComponentsSHA256 != c.ComponentsSHA256 {
		return errors.New("the evidence is not bound to this candidate")
	}
	dir := Dir(h, c.ID)
	if err := os.MkdirAll(filepath.Join(dir, "evidence"), 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, "candidate.json")
	if existing, err := LoadCandidate(h, c.ID); err == nil {
		if existing.ID != c.ID || existing.PlanSHA256 != c.PlanSHA256 || existing.ComponentsSHA256 != c.ComponentsSHA256 {
			return fmt.Errorf("stored candidate %s does not match its identity", c.ID)
		}
	} else if errors.Is(err, os.ErrNotExist) {
		if err := home.WriteJSON(path, c); err != nil {
			return err
		}
	} else {
		return err
	}
	return home.WriteFileAtomic(filepath.Join(dir, "evidence", e.ID()+".json"), e.Canonical(), 0o644)
}

// LoadCandidate reads and verifies one candidate.
func LoadCandidate(h home.Home, id string) (Candidate, error) {
	if !validID(id) {
		return Candidate{}, errors.New("candidate ID must be a SHA-256 digest")
	}
	var c Candidate
	if err := home.ReadJSON(filepath.Join(Dir(h, id), "candidate.json"), &c); err != nil {
		return Candidate{}, err
	}
	if err := c.Validate(); err != nil {
		return Candidate{}, fmt.Errorf("stored candidate %s is invalid: %w", id, err)
	}
	if c.ID != id {
		return Candidate{}, fmt.Errorf("stored candidate does not match ID %s", id)
	}
	return c, nil
}

// LoadEvidence reads and verifies every measurement of a candidate, oldest
// recording first.
func LoadEvidence(h home.Home, id string) ([]Evidence, error) {
	if !validID(id) {
		return nil, errors.New("candidate ID must be a SHA-256 digest")
	}
	entries, err := os.ReadDir(filepath.Join(Dir(h, id), "evidence"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []Evidence
	for _, f := range entries {
		var e Evidence
		raw, err := os.ReadFile(filepath.Join(Dir(h, id), "evidence", f.Name()))
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("trial evidence %s: %w", f.Name(), err)
		}
		if err := e.Validate(); err != nil {
			return nil, fmt.Errorf("trial evidence %s: %w", f.Name(), err)
		}
		if e.CandidateID != id || f.Name() != e.ID()+".json" || !bytes.Equal(raw, e.Canonical()) {
			return nil, fmt.Errorf("trial evidence %s does not match its identity", f.Name())
		}
		out = append(out, e)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].RecordedAt != out[j].RecordedAt {
			return out[i].RecordedAt < out[j].RecordedAt
		}
		return out[i].ID() < out[j].ID()
	})
	return out, nil
}

// List returns the candidates of a source (every source when empty), newest
// first. A directory that is not a valid candidate is skipped, never repaired.
func List(h home.Home, source string) ([]Candidate, error) {
	entries, err := os.ReadDir(h.Path("state", "tuning-trials"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []Candidate
	for _, e := range entries {
		if !e.IsDir() || !validID(e.Name()) {
			continue
		}
		c, err := LoadCandidate(h, e.Name())
		if err != nil || (source != "" && c.Source.ID != source) {
			continue
		}
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt != out[j].CreatedAt {
			return out[i].CreatedAt > out[j].CreatedAt
		}
		return out[i].ID < out[j].ID
	})
	return out, nil
}

// StatusOf reports how far a candidate has progressed.
func StatusOf(h home.Home, id string) (Status, error) {
	ev, err := LoadEvidence(h, id)
	if err != nil {
		return Status{}, err
	}
	st := Status{Measurements: len(ev)}
	entries, err := os.ReadDir(filepath.Join(Dir(h, id), "materialized"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return Status{}, err
	}
	for _, f := range entries {
		var m Materialization
		if home.ReadJSON(filepath.Join(Dir(h, id), "materialized", f.Name()), &m) == nil && m.CandidateID == id && f.Name() == m.VariantID+".json" {
			st.Variants = append(st.Variants, m.VariantID)
		}
	}
	sort.Strings(st.Variants)
	return st, nil
}
