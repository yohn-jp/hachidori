package diagnostics

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/redact"
)

// The Forge diagnostic is the bounded, redacted record an expensive Forge
// operation (materialize, optimize, probe, certify) leaves behind when it
// fails, so the operator can hand it back for analysis without rerunning a
// multi-hour operation.
//
// It is allowlist-based by construction: every value comes from a typed field
// of ForgeInput, filled by the owning authorities, never from a free-form map,
// the process environment, a directory listing or a log archive. Every string
// is passed through the shared redaction policy (internal/redact) and bounded;
// the document as a whole is bounded. It contains identities and digests, never
// model weights, calibration or dataset payloads, question/state bodies,
// credentials or tokens. It is stored under HACHIDORI_HOME/state/forge and is
// only written locally; it is exported only by an explicit operator action.

// ForgeSchema identifies the document. Its ID is the content identity of the
// stored document (ForgeContentID): recomputing it from the stored document
// reproduces it.
const ForgeSchema = "hachidori.forge-diagnostic/v2"

// LegacyForgeSchema is the document written before the ID was derived from the
// final, bounded content. Such a document stays readable; its ID was taken
// before truncation, so a truncated one does not verify.
const LegacyForgeSchema = "hachidori.forge-diagnostic/v1"

// Bounds of a Forge diagnostic.
const (
	// MaxForgeBytes bounds the serialized document.
	MaxForgeBytes = 96 << 10
	// MaxForgeStderrLines and MaxForgeLogLines bound the evidence tails; each
	// line is at most MaxForgeLineBytes.
	MaxForgeStderrLines = 60
	MaxForgeLogLines    = 100
	MaxForgeLineBytes   = 512
	// MaxForgeChain bounds the error chain; each entry is at most
	// maxForgeChainBytes.
	MaxForgeChain      = 16
	maxForgeChainBytes = 1024
	maxForgeFindings   = 40
	// KeepForge is how many diagnostics a home keeps; older ones are removed.
	KeepForge = 20
)

// ForgeExcluded declares in every document what it never contains.
var ForgeExcluded = []string{
	"model weight bytes",
	"calibration and dataset contents (only their identities and digests)",
	"question and state payloads",
	"credentials, tokens, authorization headers and cookies (replaced by a marker)",
	"environment variables",
	"arbitrary files and directories of the home or the user profile",
	"the HACHIDORI_HOME and user profile paths (replaced by a placeholder)",
}

// ForgeOperation identifies the failed operation.
type ForgeOperation struct {
	ID         string `json:"id"`
	Kind       string `json:"kind"`
	Phase      string `json:"phase,omitempty"` // the phase it failed in
	Step       string `json:"step,omitempty"`
	Started    string `json:"started,omitempty"`
	Finished   string `json:"finished,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`
}

// ForgeIdentity is the provenance the operation worked on.
type ForgeIdentity struct {
	Model                 string `json:"model,omitempty"`
	Provider              string `json:"provider,omitempty"`
	SourceRepo            string `json:"source_repo,omitempty"`
	SourceRevision        string `json:"source_revision,omitempty"`
	SourceFilesSHA256     string `json:"source_files_sha256,omitempty"` // digest of the pinned file digests: the source manifest identity
	Variant               string `json:"variant,omitempty"`
	VariantManifestSHA256 string `json:"variant_manifest_sha256,omitempty"`
	VariantBuildID        string `json:"variant_build_id,omitempty"`
	Runtime               string `json:"runtime,omitempty"`
}

// ForgeOptimization is the recipe and optimizer of a build or of the variant
// under test.
type ForgeOptimization struct {
	Recipe         string   `json:"recipe,omitempty"`
	RecipeSHA256   string   `json:"recipe_sha256,omitempty"`
	Backend        string   `json:"backend,omitempty"`
	BackendVersion string   `json:"backend_version,omitempty"`
	Scheme         string   `json:"scheme,omitempty"`
	Algorithm      string   `json:"algorithm,omitempty"`
	Preserved      []string `json:"preserved,omitempty"` // preserved-module selectors
}

// ForgeCertification is the identity of a certification: digests only; the
// dataset itself is never copied.
type ForgeCertification struct {
	PolicyID                string `json:"policy_id,omitempty"`
	PolicySHA256            string `json:"policy_sha256,omitempty"`
	DatasetSHA256           string `json:"dataset_sha256,omitempty"`
	QuestionsSHA256         string `json:"questions_sha256,omitempty"`
	InputSHA256             string `json:"input_sha256,omitempty"`
	ReferenceIdentitySHA256 string `json:"reference_identity_sha256,omitempty"`
	CandidateIdentitySHA256 string `json:"candidate_identity_sha256,omitempty"`
}

// ForgeRuntime is the execution environment: device, dtype, versions.
type ForgeRuntime struct {
	Device             string `json:"device,omitempty"`
	DType              string `json:"dtype,omitempty"`
	Quantization       string `json:"quantization,omitempty"`
	Python             string `json:"python,omitempty"`
	Torch              string `json:"torch,omitempty"`
	TorchCUDA          string `json:"torch_cuda,omitempty"`
	Transformers       string `json:"transformers,omitempty"`
	CompressionBackend string `json:"compression_backend,omitempty"`
	CompressionVersion string `json:"compression_version,omitempty"`
	CompressedTensors  string `json:"compressed_tensors,omitempty"`
	DeviceName         string `json:"device_name,omitempty"`
	ProviderVersion    string `json:"provider_version,omitempty"`
	OS                 string `json:"os,omitempty"`
	Arch               string `json:"arch,omitempty"`
	Hachidori          string `json:"hachidori,omitempty"`
}

// ForgeResources are the observations that exist; zero means not observed.
type ForgeResources struct {
	RAMTotal      uint64  `json:"ram_total_bytes,omitempty"`
	RAMAvailable  uint64  `json:"ram_available_bytes,omitempty"`
	VRAMTotal     uint64  `json:"vram_total_bytes,omitempty"`
	VRAMFree      uint64  `json:"vram_free_bytes,omitempty"`
	VRAMObserved  uint64  `json:"vram_allocated_bytes,omitempty"`
	WorkerRSS     uint64  `json:"worker_host_rss_bytes,omitempty"`
	DiskFree      uint64  `json:"disk_free_bytes,omitempty"`
	LoadMS        float64 `json:"load_ms,omitempty"`
	WarmupMS      float64 `json:"warmup_ms,omitempty"`
	StartupMS     float64 `json:"startup_ms,omitempty"`
	RequestMS     float64 `json:"request_ms,omitempty"`
	WorkerPID     int     `json:"worker_pid,omitempty"`
	WorkerClass   string  `json:"worker_failure_class,omitempty"`
	PartialBytes  int64   `json:"partial_download_bytes,omitempty"`
	WeightBytes   uint64  `json:"weight_bytes,omitempty"`
	WorkerStopped bool    `json:"worker_stopped,omitempty"`
}

// ForgeFailure is the failure itself.
type ForgeFailure struct {
	Error      string           `json:"error"`
	Chain      []ForgeChainLink `json:"chain"`
	Phase      string           `json:"phase,omitempty"`
	StderrTail []string         `json:"stderr_tail,omitempty"`
	LogTail    []string         `json:"log_tail,omitempty"`
}

// ForgeChainLink is one error of the chain, outermost first.
type ForgeChainLink struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// ForgeFinding is a non-passing preflight finding. Facts are not copied.
type ForgeFinding struct {
	ID      string `json:"id"`
	Status  string `json:"status"`
	Summary string `json:"summary"`
}

// ForgePreflight is the latest current preflight of exactly the failed
// operation's target, if there was one: the target it was bound to and its
// evidence state are recorded with it.
type ForgePreflight struct {
	Kind     string         `json:"kind"`
	Outcome  string         `json:"outcome"`
	At       string         `json:"at"`
	Evidence string         `json:"evidence,omitempty"`
	Model    string         `json:"model,omitempty"`
	Variant  string         `json:"variant,omitempty"`
	Recipe   string         `json:"recipe,omitempty"`
	Device   string         `json:"device,omitempty"`
	Findings []ForgeFinding `json:"findings"` // everything that is not a pass
}

// ForgeDiagnostic is the document.
type ForgeDiagnostic struct {
	Schema        string              `json:"schema"`
	ID            string              `json:"id"`
	CreatedUTC    string              `json:"created_utc"`
	Operation     ForgeOperation      `json:"operation"`
	Identity      ForgeIdentity       `json:"identity"`
	Optimization  *ForgeOptimization  `json:"optimization,omitempty"`
	Certification *ForgeCertification `json:"certification,omitempty"`
	Runtime       ForgeRuntime        `json:"runtime"`
	Resources     ForgeResources      `json:"resources"`
	Preflight     *ForgePreflight     `json:"preflight,omitempty"`
	Failure       ForgeFailure        `json:"failure"`
	// Secondary is evidence about the collection itself: a part that could not
	// be gathered. It never replaces the failure.
	Secondary []string `json:"secondary,omitempty"`
	LocalOnly bool     `json:"local_only"`
	Excluded  []string `json:"excluded"`
	Truncated bool     `json:"truncated,omitempty"`
}

// ForgeInput is everything a diagnostic is built from. The owning authorities
// fill it; BuildForge reads nothing else.
type ForgeInput struct {
	Operation     ForgeOperation
	Identity      ForgeIdentity
	Optimization  *ForgeOptimization
	Certification *ForgeCertification
	Runtime       ForgeRuntime
	Resources     ForgeResources
	Preflight     *ForgePreflight
	Err           error
	Phase         string
	StderrTail    []string
	LogTail       []string
	Secondary     []string
}

// BuildForge assembles the diagnostic: every string redacted and bounded, the
// whole document within MaxForgeBytes. root is the HACHIDORI_HOME whose path
// is replaced by a placeholder.
func BuildForge(in ForgeInput, root string, now time.Time) ForgeDiagnostic {
	s := redact.New(root)
	line := func(v string, n int) string { return s.Line(v, n) }
	d := ForgeDiagnostic{Schema: ForgeSchema, CreatedUTC: now.UTC().Format(time.RFC3339), LocalOnly: true, Excluded: ForgeExcluded}

	d.Operation = in.Operation
	d.Operation.ID, d.Operation.Kind = line(in.Operation.ID, 128), line(in.Operation.Kind, 64)
	d.Operation.Phase, d.Operation.Step = line(in.Operation.Phase, 64), line(in.Operation.Step, 64)
	d.Operation.Started, d.Operation.Finished = line(in.Operation.Started, 64), line(in.Operation.Finished, 64)

	d.Identity = scrubStruct(in.Identity, s).(ForgeIdentity)
	d.Runtime = scrubStruct(in.Runtime, s).(ForgeRuntime)
	d.Resources = in.Resources
	d.Resources.WorkerClass = line(in.Resources.WorkerClass, 64)
	if in.Optimization != nil {
		o := scrubStruct(*in.Optimization, s).(ForgeOptimization)
		o.Preserved = scrubList(in.Optimization.Preserved, s, 16, 200)
		d.Optimization = &o
	}
	if in.Certification != nil {
		c := scrubStruct(*in.Certification, s).(ForgeCertification)
		d.Certification = &c
	}
	if in.Preflight != nil {
		p := ForgePreflight{Kind: line(in.Preflight.Kind, 32), Outcome: line(in.Preflight.Outcome, 32), At: line(in.Preflight.At, 64),
			Evidence: line(in.Preflight.Evidence, 32), Model: line(in.Preflight.Model, 128), Variant: line(in.Preflight.Variant, 128),
			Recipe: line(in.Preflight.Recipe, 128), Device: line(in.Preflight.Device, 32), Findings: []ForgeFinding{}}
		for _, f := range in.Preflight.Findings {
			if len(p.Findings) == maxForgeFindings {
				break
			}
			p.Findings = append(p.Findings, ForgeFinding{ID: line(f.ID, 64), Status: line(f.Status, 16), Summary: line(f.Summary, 400)})
		}
		d.Preflight = &p
	}
	d.Failure = ForgeFailure{Phase: line(in.Phase, 64), Chain: []ForgeChainLink{}}
	if in.Err != nil {
		d.Failure.Error = line(in.Err.Error(), 2048)
		d.Failure.Chain = chain(in.Err, s)
	}
	d.Failure.StderrTail = tail(in.StderrTail, s, MaxForgeStderrLines)
	d.Failure.LogTail = tail(in.LogTail, s, MaxForgeLogLines)
	d.Secondary = scrubList(in.Secondary, s, 16, 400)

	// Hold the whole document to its bound: the evidence tails give way first,
	// oldest lines first. A provisional ID of the final shape (its length
	// does not depend on the content) keeps the measured size the stored one.
	d.ID = ForgeContentID(d)
	for size(d) > MaxForgeBytes && (len(d.Failure.LogTail) > 0 || len(d.Failure.StderrTail) > 0) {
		d.Truncated = true
		switch {
		case len(d.Failure.LogTail) > 0 && len(d.Failure.LogTail) >= len(d.Failure.StderrTail):
			d.Failure.LogTail = d.Failure.LogTail[len(d.Failure.LogTail)/4+1:]
		default:
			d.Failure.StderrTail = d.Failure.StderrTail[len(d.Failure.StderrTail)/4+1:]
		}
	}
	if size(d) > MaxForgeBytes {
		d.Truncated = true
		d.Preflight, d.Secondary = nil, d.Secondary[:min(len(d.Secondary), 2)]
	}
	// The identity is taken last, from the content as it is stored: nothing
	// changes the document after this.
	d.ID = ForgeContentID(d)
	return d
}

// size is the length of the document as SaveForge stores it.
func size(d ForgeDiagnostic) int {
	b, _ := marshalForge(d)
	return len(b)
}

func tail(lines []string, s redact.Scrubber, max int) []string {
	if len(lines) > max {
		lines = lines[len(lines)-max:]
	}
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		out = append(out, s.Line(l, MaxForgeLineBytes))
	}
	return out
}

func scrubList(in []string, s redact.Scrubber, maxN, maxLen int) []string {
	if len(in) > maxN {
		in = in[:maxN]
	}
	out := make([]string, 0, len(in))
	for _, v := range in {
		out = append(out, s.Line(v, maxLen))
	}
	return out
}

// chain walks the error tree outermost first (an error that wraps several,
// as errors.Join and a multiple-%w Errorf do, is walked depth first).
func chain(err error, s redact.Scrubber) []ForgeChainLink {
	var out []ForgeChainLink
	var walk func(error)
	walk = func(e error) {
		if e == nil || len(out) >= MaxForgeChain {
			return
		}
		out = append(out, ForgeChainLink{Type: fmt.Sprintf("%T", e), Message: s.Line(e.Error(), maxForgeChainBytes)})
		switch u := e.(type) {
		case interface{ Unwrap() []error }:
			for _, c := range u.Unwrap() {
				walk(c)
			}
		case interface{ Unwrap() error }:
			walk(u.Unwrap())
		}
	}
	walk(err)
	return out
}

// ForgeContentID is the content identity of a diagnostic, the one procedure
// that both assigns and verifies a diagnostic's ID:
//
//  1. take the final document (redacted, bounded and truncated: as stored);
//  2. clear its ID field;
//  3. serialize it canonically: encoding/json's compact Marshal of the
//     ForgeDiagnostic struct (fields in declaration order, omitempty as
//     declared, strings HTML-escaped), which a stored document decodes back
//     to exactly;
//  4. take the SHA-256 of those bytes;
//  5. the ID is "<kind>-<created, UTC, seconds>-<first 4 bytes of the
//     digest, hex>".
//
// A document's ID is valid when ForgeContentID of the document reproduces it
// (VerifyForge).
func ForgeContentID(d ForgeDiagnostic) string {
	d.ID = ""
	b, _ := json.Marshal(d)
	sum := sha256.Sum256(b)
	kind := safeID(d.Operation.Kind)
	if kind == "" {
		kind = "forge"
	}
	t, _ := time.Parse(time.RFC3339, d.CreatedUTC)
	return kind + "-" + t.UTC().Format("20060102T150405Z") + "-" + hex.EncodeToString(sum[:4])
}

// VerifyForge reports whether d's ID is the content identity of d as it is:
// the document is the one the ID was assigned to, after every bound and
// truncation.
func VerifyForge(d ForgeDiagnostic) error {
	if want := ForgeContentID(d); d.ID != want {
		return fmt.Errorf("diagnostic %s: its content identity is %s; the document is not the one its ID was assigned to", d.ID, want)
	}
	return nil
}

var idRe = regexp.MustCompile(`^[a-z0-9]+-\d{8}T\d{6}Z-[0-9a-f]{8}$`)

// ValidForgeID reports whether id has the shape of a diagnostic identity, so
// it can never name a path.
func ValidForgeID(id string) bool { return idRe.MatchString(id) }

func safeID(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, strings.ToLower(s))
}

func forgeDir(root string) string { return filepath.Join(root, "state", "forge", "diagnostics") }

// SaveForge stores d under the home atomically and prunes the oldest
// diagnostics beyond KeepForge. It returns the path.
func SaveForge(root string, d ForgeDiagnostic) (string, error) {
	if !ValidForgeID(d.ID) {
		return "", fmt.Errorf("%q is not a diagnostic identity", d.ID)
	}
	dir := forgeDir(root)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	b, err := marshalForge(d)
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, d.ID+".json")
	if err := home.WriteFileAtomic(path, b, 0o600); err != nil {
		return "", err
	}
	pruneForge(dir)
	return path, nil
}

func pruneForge(dir string) {
	ids := forgeIDs(dir)
	for len(ids) > KeepForge {
		_ = os.Remove(filepath.Join(dir, ids[0]+".json"))
		ids = ids[1:]
	}
}

// forgeIDs lists the diagnostic identities of dir, oldest first.
func forgeIDs(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	type item struct{ id, key string }
	var items []item
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".json")
		if !ok || !e.Type().IsRegular() || !ValidForgeID(id) {
			continue
		}
		// kind-<time>-<hash>: order by time, then name.
		parts := strings.SplitN(id, "-", 3)
		items = append(items, item{id, parts[1] + parts[2]})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].key < items[j].key })
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.id
	}
	return out
}

// ForgeSummary is one stored diagnostic as the listings show it.
type ForgeSummary struct {
	ID      string `json:"id"`
	Kind    string `json:"kind"`
	Phase   string `json:"phase,omitempty"`
	Model   string `json:"model,omitempty"`
	Variant string `json:"variant,omitempty"`
	Created string `json:"created_utc"`
	Error   string `json:"error"`
	Path    string `json:"path"`
}

// ForgeFilter narrows the diagnostics a lookup considers. Empty fields match
// everything.
type ForgeFilter struct{ Kind, Model, Variant string }

func (f ForgeFilter) match(d ForgeDiagnostic) bool {
	return (f.Kind == "" || d.Operation.Kind == f.Kind) && (f.Model == "" || d.Identity.Model == f.Model) && (f.Variant == "" || d.Identity.Variant == f.Variant)
}

// ListForge summarizes the stored diagnostics matching f, newest first.
func ListForge(root string, f ForgeFilter) []ForgeSummary {
	dir := forgeDir(root)
	ids := forgeIDs(dir)
	var out []ForgeSummary
	for i := len(ids) - 1; i >= 0; i-- {
		d, err := LoadForge(root, ids[i])
		if err != nil || !f.match(d) {
			continue
		}
		msg := d.Failure.Error
		if len(msg) > 200 {
			msg = msg[:200] + "..."
		}
		out = append(out, ForgeSummary{ID: d.ID, Kind: d.Operation.Kind, Phase: d.Failure.Phase, Model: d.Identity.Model, Variant: d.Identity.Variant,
			Created: d.CreatedUTC, Error: msg, Path: filepath.Join(dir, d.ID+".json")})
	}
	return out
}

// LoadForge reads one stored diagnostic by its stable identity.
func LoadForge(root, id string) (ForgeDiagnostic, error) {
	var d ForgeDiagnostic
	if !ValidForgeID(id) {
		return d, fmt.Errorf("%q is not a diagnostic identity (see `hachidori forge diagnostics list`)", id)
	}
	b, err := os.ReadFile(filepath.Join(forgeDir(root), id+".json"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return d, fmt.Errorf("no diagnostic %s in this home", id)
		}
		return d, err
	}
	if err := json.Unmarshal(b, &d); err != nil {
		return d, fmt.Errorf("diagnostic %s: %w", id, err)
	}
	switch {
	case d.ID != id || (d.Schema != ForgeSchema && d.Schema != LegacyForgeSchema):
		return d, fmt.Errorf("diagnostic %s is not a %s document", id, ForgeSchema)
	case d.Schema == ForgeSchema:
		// A current document must be the one its ID names.
		if err := VerifyForge(d); err != nil {
			return d, err
		}
	}
	return d, nil
}

// LatestForge is the most recent stored diagnostic matching f.
func LatestForge(root string, f ForgeFilter) (ForgeDiagnostic, bool) {
	for _, s := range ListForge(root, f) {
		if d, err := LoadForge(root, s.ID); err == nil {
			return d, true
		}
	}
	return ForgeDiagnostic{}, false
}

// ExportForge writes the diagnostic to dest: a file path, or, when dest is an
// existing directory, "<dir>/<id>.json". It only writes locally.
func ExportForge(d ForgeDiagnostic, dest string) (string, error) {
	if fi, err := os.Stat(dest); err == nil && fi.IsDir() {
		dest = filepath.Join(dest, d.ID+".json")
	}
	b, err := marshalForge(d)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(dest, b, 0o600); err != nil {
		return "", err
	}
	return dest, nil
}

// scrubStruct returns a copy of the struct v whose string fields went through
// the redaction policy and the line bound. Only flat structs of strings and
// numbers are scrubbed here; that is all the allowlisted sections contain.
func scrubStruct(v any, s redact.Scrubber) any {
	in := reflect.ValueOf(v)
	out := reflect.New(in.Type()).Elem()
	out.Set(in)
	for i := 0; i < out.NumField(); i++ {
		if f := out.Field(i); f.Kind() == reflect.String {
			f.SetString(s.Line(f.String(), 512))
		}
	}
	return out.Interface()
}

// marshalForge is the document as an operator reads it: indented, with the
// redaction markers' angle brackets left as they are.
func marshalForge(d ForgeDiagnostic) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(d); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// FormatForge is marshalForge for callers outside the package (the CLI).
func FormatForge(d ForgeDiagnostic) ([]byte, error) { return marshalForge(d) }
