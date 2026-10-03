package home

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// A Variant is a derived, immutable execution artifact of exactly one source
// model: the same System One model at another weight precision. It is not a
// model of its own. The source model keeps its catalog identity, its directory
// under models/ and its files; a variant lives beside them under
// variants/<source model ID>/<variant ID>/ and links back to the source by
// repository, revision and the digest of the source's pinned files.
//
// The variant manifest is the variant's identity. It is written once, last,
// when the staged artifact is complete and verified; its presence marks a
// complete variant. Certification evidence is never embedded in it: the
// manifest only links to where records about it are kept.

// VariantSchema versions the variant manifest contract and its identity
// derivation.
const VariantSchema = "hachidori.variant/1"

// RecipeSchema versions the canonical recipe encoding.
const RecipeSchema = "hachidori.recipe/1"

// VariantManifestFile is the manifest's file name inside a variant directory.
const VariantManifestFile = "hachidori-variant.json"

// Providers, precision facts and states the contract names.
const (
	ProviderClef = "clef" // the System One provider kind of Clef models

	// ScopeBackbone preserves modules of the quantized backbone (by name or
	// "re:" pattern); ScopeCarried preserves a carried file byte for byte.
	ScopeBackbone = "backbone"
	ScopeCarried  = "carried"

	// CertificationPending is the only state a manifest ever records: the
	// variant was built and has not been certified *by this manifest*. The
	// effective certification state is resolved from the records at
	// CertificationLink.Records, never from the manifest.
	CertificationPending = "pending"
)

// VariantSource is the immutable source identity a variant derives from.
type VariantSource struct {
	ID       string `json:"id"`       // catalog model ID
	Provider string `json:"provider"` // provider kind
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	// FilesSHA256 is the digest of the canonical encoding of the source's
	// pinned file digests: any change to any pinned source file changes it.
	FilesSHA256 string `json:"files_sha256"`
}

// PreservedModule is a part of the model the recipe deliberately keeps at a
// higher precision instead of quantizing it.
type PreservedModule struct {
	Pattern   string `json:"pattern"`   // module name, "re:" regular expression, or a carried file
	Scope     string `json:"scope"`     // backbone | carried
	Precision string `json:"precision"` // what it is kept at
	Reason    string `json:"reason"`    // why it is preserved
}

// Recipe is the canonical Hachidori recipe: everything that decides what the
// optimizer does, nothing that describes where it runs. Its canonical JSON
// encoding (field order as declared) is hashed into the variant identity, so
// any semantic change yields a different variant.
type Recipe struct {
	Schema    string            `json:"schema"`
	Name      string            `json:"name"`
	Engine    string            `json:"engine"`    // optimizer backend
	Scheme    string            `json:"scheme"`    // weight scheme, e.g. W4A16
	Algorithm string            `json:"algorithm"` // rtn: data-free round-to-nearest
	Targets   []string          `json:"targets"`   // module classes the scheme applies to
	Preserved []PreservedModule `json:"preserved"`
	// Carry lists the source files copied unchanged into the variant, so that
	// it is complete and loadable without the source.
	Carry []string `json:"carry"`
}

// Canonical is the recipe's canonical encoding.
func (r Recipe) Canonical() []byte {
	b, err := json.Marshal(r)
	if err != nil {
		panic(err)
	}
	return b
}

// SHA256 is the recipe digest.
func (r Recipe) SHA256() string { return sha256Hex(r.Canonical()) }

// Validate checks the recipe is well formed.
func (r Recipe) Validate() error {
	if r.Schema != RecipeSchema {
		return fmt.Errorf("recipe schema %q, want %q", r.Schema, RecipeSchema)
	}
	if !nameRe.MatchString(r.Name) {
		return fmt.Errorf("recipe name %q is not a plain identifier", r.Name)
	}
	if r.Engine == "" || r.Scheme == "" || r.Algorithm == "" || len(r.Targets) == 0 {
		return errors.New("recipe must name engine, scheme, algorithm and targets")
	}
	for _, p := range r.Preserved {
		if p.Pattern == "" || p.Precision == "" || p.Reason == "" || (p.Scope != ScopeBackbone && p.Scope != ScopeCarried) {
			return fmt.Errorf("recipe preserves %+v without pattern, scope, precision and reason", p)
		}
		if p.Scope == ScopeCarried && !contains(r.Carry, p.Pattern) {
			return fmt.Errorf("recipe preserves carried file %q that it does not carry", p.Pattern)
		}
		if strings.HasPrefix(p.Pattern, "re:") {
			if _, err := regexp.Compile(p.Pattern[3:]); err != nil {
				return fmt.Errorf("recipe preserved pattern %q: %w", p.Pattern, err)
			}
		}
	}
	for _, f := range r.Carry {
		if f == "" || strings.HasPrefix(f, "/") || strings.Contains(f, "..") || strings.Contains(f, `\`) {
			return fmt.Errorf("recipe carries invalid path %q", f)
		}
	}
	return nil
}

var nameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{0,63}$`)

// Optimizer is the engine facts of the build: which backend, which version,
// in which optimizer runtime.
type Optimizer struct {
	Engine  string `json:"engine"`
	Version string `json:"version"`
	Runtime string `json:"runtime"` // optimizer runtime identity (directory under runtime/)
	Device  string `json:"device"`  // device the transformation ran on
	// Versions are the versions of every library the engine reported.
	Versions map[string]string `json:"versions"`
}

// Calibration identifies the calibration corpus a recipe used. It is nil for
// a data-free recipe.
type Calibration struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

// TuningProvenance binds a variant build to the exact persisted semantic
// profile and source analysis that compiled its recipe. IDs are their full
// SHA-256 digests so a provenance reference is directly verifiable.
type TuningProvenance struct {
	Schema          string        `json:"schema"`
	Source          VariantSource `json:"source"`
	ProfileID       string        `json:"profile_id"`
	ProfileSHA256   string        `json:"profile_sha256"`
	AnalysisID      string        `json:"analysis_id"`
	AnalysisSHA256  string        `json:"analysis_sha256"`
	CompilerVersion string        `json:"compiler_version"`
	// PlanSHA256 is the digest of the resolved layer-wise plan: the complete
	// effective policy, AUTO resolutions included. It is set exactly when the
	// profile is layer-wise (schema 2) and is part of the build contract.
	PlanSHA256 string `json:"plan_sha256,omitempty"`
}

// TuningProvenanceSchema versions the tuning linkage stored in a variant
// manifest of a layer-wise profile; TuningProvenanceSchemaV1 is the linkage of
// a legacy coarse (per-region) profile and stays valid.
const (
	TuningProvenanceSchema   = "hachidori.tuning-provenance/2"
	TuningProvenanceSchemaV1 = "hachidori.tuning-provenance/1"
)

// Validate checks the provenance's self-contained identities and source.
func (p TuningProvenance) Validate(source VariantSource) error {
	switch p.Schema {
	case TuningProvenanceSchemaV1:
		if p.PlanSHA256 != "" {
			return errors.New("legacy tuning provenance cannot carry a layer-wise plan")
		}
	case TuningProvenanceSchema:
		if !validSHA256(p.PlanSHA256) {
			return errors.New("tuning provenance has no valid layer-wise plan digest")
		}
	default:
		return fmt.Errorf("tuning provenance schema %q, want %q or legacy %q", p.Schema, TuningProvenanceSchema, TuningProvenanceSchemaV1)
	}
	if p.Source != source {
		return errors.New("tuning provenance source identity does not match variant source")
	}
	if !validSHA256(p.ProfileID) || p.ProfileID != p.ProfileSHA256 {
		return errors.New("tuning provenance profile identity does not match its digest")
	}
	if !validSHA256(p.AnalysisID) || p.AnalysisID != p.AnalysisSHA256 {
		return errors.New("tuning provenance analysis identity does not match its digest")
	}
	if strings.TrimSpace(p.CompilerVersion) == "" {
		return errors.New("tuning provenance has no compiler version")
	}
	return nil
}

// WeightPrecision declares how the variant's weights are stored and executed.
// DType is the compute/activation dtype of everything that is not quantized.
type WeightPrecision struct {
	Scheme    string `json:"scheme"` // e.g. W4A16
	Bits      int    `json:"bits"`
	GroupSize int    `json:"group_size"`
	Symmetric bool   `json:"symmetric"`
	Format    string `json:"format"` // serialization format
	DType     string `json:"dtype"`
}

// Creation is the provenance sufficient to reproduce the build.
type Creation struct {
	CreatedAt string `json:"created_at"` // RFC 3339, UTC
	Platform  string `json:"platform"`   // GOOS/GOARCH of the building executable
	Command   string `json:"command"`    // the Hachidori invocation that reproduces it
}

// CertificationLink says where records about the variant are kept. Status is
// the creation-time state only (CertificationPending); evaluation results are
// never embedded in the manifest.
type CertificationLink struct {
	Status  string `json:"status"`
	Records string `json:"records"` // relative to HACHIDORI_HOME, slash separated
}

// VariantManifest is variants/<source>/<id>/hachidori-variant.json.
type VariantManifest struct {
	Schema string `json:"schema"`
	// ID is the stable variant identity: derived from BuildID and the
	// artifact digests, so it changes whenever the source, engine, engine
	// version, recipe, calibration, tuning profile or any artifact byte changes.
	ID string `json:"id"`
	// BuildID is the identity of the build contract alone (source, engine,
	// engine version, recipe, calibration and optional tuning profile), before
	// any artifact exists. Two builds of one contract that produce different
	// bytes share a BuildID and have different IDs: the difference is visible,
	// never merged.
	BuildID       string            `json:"build_id"`
	Source        VariantSource     `json:"source"`
	Provider      string            `json:"provider"`
	Optimizer     Optimizer         `json:"optimizer"`
	Recipe        Recipe            `json:"recipe"`
	RecipeSHA256  string            `json:"recipe_sha256"`
	Calibration   *Calibration      `json:"calibration,omitempty"`
	Tuning        *TuningProvenance `json:"tuning,omitempty"`
	Weights       WeightPrecision   `json:"weights"`
	Preserved     []PreservedModule `json:"preserved"`
	Files         map[string]string `json:"files"` // artifact relpath -> sha256
	Creation      Creation          `json:"creation"`
	Certification CertificationLink `json:"certification"`
}

// SourceFilesSHA256 is the digest of the canonical encoding of a source
// model's pinned file digests.
func SourceFilesSHA256(files map[string]string) string {
	b, err := json.Marshal(files) // map keys are sorted: canonical
	if err != nil {
		panic(err)
	}
	return sha256Hex(b)
}

// SourceOf is the variant source identity of a catalog model manifest.
func SourceOf(m ModelManifest) VariantSource {
	return VariantSource{ID: m.ID, Provider: m.Provider, Repo: m.Repo, Revision: m.Revision, FilesSHA256: SourceFilesSHA256(m.Files)}
}

// DeriveBuildID is the identity of a build contract.
func DeriveBuildID(src VariantSource, provider string, opt Optimizer, recipeSHA string, cal *Calibration) string {
	return DeriveBuildIDWithTuning(src, provider, opt, recipeSHA, cal, nil)
}

// DeriveBuildIDWithTuning is the identity of a build contract, including its
// optional exact tuning provenance. A nil tuning record preserves the
// recipe-only identity of existing variants.
func DeriveBuildIDWithTuning(src VariantSource, provider string, opt Optimizer, recipeSHA string, cal *Calibration, tuning *TuningProvenance) string {
	b, err := json.Marshal(struct {
		Schema      string            `json:"schema"`
		Source      VariantSource     `json:"source"`
		Provider    string            `json:"provider"`
		Engine      string            `json:"engine"`
		Version     string            `json:"version"`
		Recipe      string            `json:"recipe_sha256"`
		Calibration *Calibration      `json:"calibration,omitempty"`
		Tuning      *TuningProvenance `json:"tuning,omitempty"`
	}{VariantSchema, src, provider, opt.Engine, opt.Version, recipeSHA, cal, tuning})
	if err != nil {
		panic(err)
	}
	return sha256Hex(b)
}

// DeriveVariantID is the variant identity: the source model, the recipe name
// and a digest of the build contract and every artifact digest.
func DeriveVariantID(sourceID, recipeName, buildID string, files map[string]string) string {
	b, err := json.Marshal(struct {
		Build string            `json:"build_id"`
		Files map[string]string `json:"files"`
	}{buildID, files})
	if err != nil {
		panic(err)
	}
	return sourceID + "--" + recipeName + "--" + sha256Hex(b)[:12]
}

// VariantDirName is the activation name (directory under variants/, slash
// separated) of a variant.
func VariantDirName(sourceID, id string) string { return sourceID + "/" + id }

// VariantsDir is variants/<source model ID>.
func (h Home) VariantsDir(sourceID string) string { return h.Path("variants", sourceID) }

// VariantDir is the directory of one variant.
func (h Home) VariantDir(sourceID, id string) string { return h.Path("variants", sourceID, id) }

// CertificationDir is where certification records about a variant are kept.
func CertificationDir(id string) string { return "state/certifications/" + id }

// Seal completes the manifest's derived fields (digests, identity and the
// certification link) from its authoritative inputs. It is called by the
// builder once the artifacts are digested.
func (m *VariantManifest) Seal() {
	m.Schema = VariantSchema
	m.RecipeSHA256 = m.Recipe.SHA256()
	m.Preserved = append([]PreservedModule(nil), m.Recipe.Preserved...)
	m.BuildID = DeriveBuildIDWithTuning(m.Source, m.Provider, m.Optimizer, m.RecipeSHA256, m.Calibration, m.Tuning)
	m.ID = DeriveVariantID(m.Source.ID, m.Recipe.Name, m.BuildID, m.Files)
	m.Certification = CertificationLink{Status: CertificationPending, Records: CertificationDir(m.ID)}
}

// Canonical is the manifest's canonical encoding: the bytes whose digest
// (ManifestSHA256) is the variant's independent content address.
func (m VariantManifest) Canonical() []byte {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		panic(err)
	}
	return append(b, '\n')
}

// ManifestSHA256 is the digest of the manifest file as written.
func (m VariantManifest) ManifestSHA256() string { return sha256Hex(m.Canonical()) }

// Validate is the structural and identity check of a manifest: every derived
// field must be the one its authoritative inputs derive, so a manifest that
// was edited, truncated or assembled by hand is rejected. It does not touch
// artifact bytes (VerifyVariantFiles in internal/setup does).
func (m VariantManifest) Validate() error {
	if m.Schema != VariantSchema {
		return fmt.Errorf("variant manifest schema %q, want %q", m.Schema, VariantSchema)
	}
	if err := m.Recipe.Validate(); err != nil {
		return err
	}
	s := m.Source
	if s.ID == "" || s.Provider == "" || s.Repo == "" || s.Revision == "" || s.FilesSHA256 == "" {
		return errors.New("variant has no complete source identity (a variant cannot exist without an immutable source)")
	}
	if m.Provider != s.Provider {
		return fmt.Errorf("variant provider %q is not its source's provider %q", m.Provider, s.Provider)
	}
	if m.Optimizer.Engine != m.Recipe.Engine || m.Optimizer.Version == "" {
		return fmt.Errorf("variant optimizer %s/%s does not match recipe engine %q", m.Optimizer.Engine, m.Optimizer.Version, m.Recipe.Engine)
	}
	if m.Weights.Scheme != m.Recipe.Scheme || m.Weights.Bits == 0 || m.Weights.Format == "" || m.Weights.DType == "" {
		return fmt.Errorf("variant weights %+v do not declare the recipe scheme %q", m.Weights, m.Recipe.Scheme)
	}
	if m.RecipeSHA256 != m.Recipe.SHA256() {
		return errors.New("variant recipe digest does not match its recipe")
	}
	if !samePreserved(m.Preserved, m.Recipe.Preserved) {
		return errors.New("variant preserved modules differ from its recipe's")
	}
	if len(m.Files) == 0 {
		return errors.New("variant lists no artifact files")
	}
	for rel, d := range m.Files {
		if rel == "" || strings.HasPrefix(rel, "/") || strings.Contains(rel, "..") || strings.Contains(rel, `\`) || rel == VariantManifestFile {
			return fmt.Errorf("variant artifact path %q is not valid", rel)
		}
		if len(d) != 64 {
			return fmt.Errorf("variant artifact %s has no sha256 digest", rel)
		}
	}
	if m.Tuning != nil {
		if err := m.Tuning.Validate(m.Source); err != nil {
			return err
		}
	}
	if want := DeriveBuildIDWithTuning(m.Source, m.Provider, m.Optimizer, m.RecipeSHA256, m.Calibration, m.Tuning); m.BuildID != want {
		return errors.New("variant build id does not match its source, engine, recipe, calibration and tuning provenance")
	}
	if want := DeriveVariantID(m.Source.ID, m.Recipe.Name, m.BuildID, m.Files); m.ID != want {
		return fmt.Errorf("variant id %q does not match its build and artifact digests (%q)", m.ID, want)
	}
	if m.Certification.Records != CertificationDir(m.ID) || m.Certification.Status != CertificationPending {
		return errors.New("variant certification link is not the one its identity derives")
	}
	return nil
}

// CheckSource reports whether the manifest derives from exactly the catalog
// model src: same ID, provider, repository, revision and pinned files.
func (m VariantManifest) CheckSource(src ModelManifest) error {
	if want := SourceOf(src); m.Source != want {
		return fmt.Errorf("variant %s derives from %s %s@%s (files %s), not from catalog model %s %s@%s (files %s)",
			m.ID, m.Source.ID, m.Source.Repo, m.Source.Revision, short(m.Source.FilesSHA256),
			want.ID, want.Repo, want.Revision, short(want.FilesSHA256))
	}
	return nil
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	return d
}

func validSHA256(d string) bool {
	if len(d) != 64 {
		return false
	}
	b, err := hex.DecodeString(d)
	return err == nil && hex.EncodeToString(b) == d
}

func samePreserved(a, b []PreservedModule) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// ReadVariant reads and validates a variant manifest from a variant
// directory.
func ReadVariant(dir string) (VariantManifest, error) {
	var m VariantManifest
	if err := ReadJSON(filepath.Join(dir, VariantManifestFile), &m); err != nil {
		return m, err
	}
	if err := m.Validate(); err != nil {
		return m, err
	}
	return m, nil
}

// VariantFileNames lists a manifest's artifact files in a fixed order.
func (m VariantManifest) VariantFileNames() []string {
	names := make([]string, 0, len(m.Files))
	for rel := range m.Files {
		names = append(names, rel)
	}
	sort.Strings(names)
	return names
}

// LoadVariant reads the variant an activation record selects. A record
// without a variant selects the source artifact and has none.
func (h Home) LoadVariant(a Active) (VariantManifest, bool, error) {
	if a.Variant == "" {
		return VariantManifest{}, false, nil
	}
	m, err := ReadVariant(h.VariantDir(a.ModelID, a.Variant))
	if err != nil {
		return m, true, fmt.Errorf("variant %s: %w", a.Variant, err)
	}
	if m.ID != a.Variant || m.Source.ID != a.ModelID {
		return m, true, fmt.Errorf("variant directory %s holds %s of %s", a.Variant, m.ID, m.Source.ID)
	}
	return m, true, nil
}
