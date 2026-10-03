package eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

// ForgeRunSchema identifies a resident run produced by a Forge execution
// session and kept as internal evidence under the home. The ResidentRun inside
// is exactly the one internal/eval defines; this document only binds it to
// what was executed so that nobody needs to be told, by a path or a claim,
// what a stored run represents.
const ForgeRunSchema = "hachidori.forge-run.v1"

// Execution target kinds.
const (
	ForgeTargetSource  = "source"  // the pinned source model
	ForgeTargetVariant = "variant" // one persisted variant of a source model
)

// ForgeRunTarget is the exact execution a run was recorded on, from the
// authorities that resolved it (the catalog source identity, the variant
// manifest, the runtime) and from what the worker that became READY reported.
// Requested* are what was asked; the unprefixed fields are what ran.
type ForgeRunTarget struct {
	Kind              string `json:"kind"`
	Model             string `json:"model"`
	Provider          string `json:"provider"`
	Repo              string `json:"source_repo"`
	Revision          string `json:"source_revision"`
	SourceFilesSHA256 string `json:"source_files_sha256"`

	Variant               string `json:"variant,omitempty"`
	VariantManifestSHA256 string `json:"variant_manifest_sha256,omitempty"`
	Recipe                string `json:"recipe,omitempty"`
	Scheme                string `json:"scheme,omitempty"`
	Quantization          string `json:"quantized_execution,omitempty"`
	QuantizedModules      int    `json:"quantized_modules,omitempty"`

	// Runtime is the dependency runtime identity (the environment); WorkerSHA256
	// is the worker implementation that ran in it. A worker-only update changes
	// the second and not the first.
	Runtime         string `json:"runtime"`
	WorkerSHA256    string `json:"worker_sha256,omitempty"`
	RequestedDevice string `json:"requested_device"`
	Device          string `json:"device"`
	RequestedDType  string `json:"requested_dtype,omitempty"`
	DType           string `json:"dtype"`
}

// ForgeRun is the persisted evidence of one execution session. ID is derived
// from the content, so the same bytes always have the same ID and a stored run
// cannot be edited without its ID no longer matching.
type ForgeRun struct {
	Schema                    string         `json:"schema"`
	ID                        string         `json:"id"`
	RecordedAt                string         `json:"recorded_at"`
	Target                    ForgeRunTarget `json:"target"`
	DatasetSHA256             string         `json:"dataset_sha256"`
	QuestionDefinitionsSHA256 string         `json:"question_definitions_sha256,omitempty"`
	RunSHA256                 string         `json:"run_sha256"`
	Run                       ResidentRun    `json:"run"`
}

var forgeRunIDRe = regexp.MustCompile(`^fr-[0-9a-f]{32}$`)

func sha256JSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// seal fills the content digests and the ID from the target and the run.
func (r *ForgeRun) seal() error {
	var err error
	r.DatasetSHA256 = r.Run.DatasetSHA256
	r.QuestionDefinitionsSHA256 = ""
	if len(r.Run.Definitions) > 0 {
		if r.QuestionDefinitionsSHA256, err = sha256JSON(r.Run.Definitions); err != nil {
			return err
		}
	}
	if r.RunSHA256, err = sha256JSON(r.Run); err != nil {
		return err
	}
	id, err := sha256JSON(struct {
		Target                    ForgeRunTarget `json:"target"`
		DatasetSHA256             string         `json:"dataset_sha256"`
		QuestionDefinitionsSHA256 string         `json:"question_definitions_sha256"`
		RunSHA256                 string         `json:"run_sha256"`
	}{r.Target, r.DatasetSHA256, r.QuestionDefinitionsSHA256, r.RunSHA256})
	r.ID = "fr-" + id[:32]
	return err
}

// NewForgeRun binds a valid resident run to the target it was executed on.
func NewForgeRun(t ForgeRunTarget, run ResidentRun, now time.Time) (ForgeRun, error) {
	r := ForgeRun{Schema: ForgeRunSchema, RecordedAt: now.UTC().Format(time.RFC3339), Target: t, Run: run}
	if err := r.seal(); err != nil {
		return ForgeRun{}, err
	}
	return r, r.Verify()
}

// Verify checks that the document is internally consistent: a valid resident
// run of the target's model, the digests and the ID recomputed from the
// content, and a target that is coherent for its kind.
func (r ForgeRun) Verify() error {
	if r.Schema != ForgeRunSchema {
		return fmt.Errorf("forge run schema %q, want %q", r.Schema, ForgeRunSchema)
	}
	if err := ValidateResidentRun(r.Run); err != nil {
		return err
	}
	t := r.Target
	switch {
	case t.Kind == ForgeTargetSource && (t.Variant != "" || t.VariantManifestSHA256 != ""):
		return errors.New("a source run names a variant")
	case t.Kind == ForgeTargetVariant && (t.Variant == "" || t.VariantManifestSHA256 == ""):
		return errors.New("a variant run lacks its variant identity")
	case t.Kind != ForgeTargetSource && t.Kind != ForgeTargetVariant:
		return fmt.Errorf("unknown execution target kind %q", t.Kind)
	case t.Model == "" || t.Revision == "" || t.SourceFilesSHA256 == "" || t.Runtime == "" || t.Device == "" || t.DType == "":
		return errors.New("the execution target identity is incomplete")
	case t.Model != r.Run.Run.Model:
		return fmt.Errorf("the run is of resident %q, not the target's model %q", r.Run.Run.Model, t.Model)
	}
	want := r
	if err := want.seal(); err != nil {
		return err
	}
	if want.ID != r.ID || want.RunSHA256 != r.RunSHA256 || want.DatasetSHA256 != r.DatasetSHA256 || want.QuestionDefinitionsSHA256 != r.QuestionDefinitionsSHA256 {
		return errors.New("the recorded identity or digests do not match the content")
	}
	return nil
}

// ForgeRunsDir is where execution sessions keep their evidence.
func ForgeRunsDir(h home.Home) string { return h.Path("state", "forge", "runs") }

// SaveForgeRun publishes r atomically under its ID. A write that is
// interrupted leaves no document under the ID (only a temporary file that no
// reader looks at), so a stored run is always a complete one.
func SaveForgeRun(h home.Home, r ForgeRun) error {
	if err := r.Verify(); err != nil {
		return err
	}
	dir := ForgeRunsDir(h)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return home.WriteFileAtomic(filepath.Join(dir, r.ID+".json"), append(b, '\n'), 0o644)
}

// LoadForgeRun reads the run with this evidence ID, strictly: a document that
// is not complete, was edited, or is stored under another ID is refused.
func LoadForgeRun(h home.Home, id string) (ForgeRun, error) {
	if !forgeRunIDRe.MatchString(id) {
		return ForgeRun{}, fmt.Errorf("%q is not a forge run ID", id)
	}
	b, err := os.ReadFile(filepath.Join(ForgeRunsDir(h), id+".json"))
	if err != nil {
		return ForgeRun{}, fmt.Errorf("forge run %s: %w", id, err)
	}
	var r ForgeRun
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return ForgeRun{}, fmt.Errorf("forge run %s is malformed: %w", id, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return ForgeRun{}, fmt.Errorf("forge run %s is malformed: trailing data", id)
	}
	if r.ID != id {
		return ForgeRun{}, fmt.Errorf("forge run %s: the document is %s", id, r.ID)
	}
	if err := r.Verify(); err != nil {
		return ForgeRun{}, fmt.Errorf("forge run %s: %w", id, err)
	}
	return r, nil
}
