package e2e

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/redact"
)

// The proof boundaries of the certification. Each is one shard: an
// independent job, an independent Go package under test/windows-e2e, and an
// independent evidence directory.
const (
	ShardBootstrap      = "bootstrap"
	ShardRuntime        = "runtime"
	ShardRecovery       = "recovery"
	ShardSingleInstance = "single-instance"
	ShardUpdate         = "update"
	ShardDiagnostics    = "diagnostics"
)

// Shards lists every required shard in a stable order. A required shard that
// reports no result is a certification failure.
var Shards = []string{
	ShardBootstrap,
	ShardRuntime,
	ShardRecovery,
	ShardSingleInstance,
	ShardUpdate,
	ShardDiagnostics,
}

// Outcomes of a scenario.
const (
	OutcomePass    = "PASS"
	OutcomeFail    = "FAIL"
	OutcomeBlocked = "BLOCKED"
	OutcomeSkip    = "SKIP"
	OutcomeMissing = "MISSING"
)

const (
	BlockReasonPrerequisiteNotPassed = "prerequisite_not_passed"
	BlockReasonShardPrerequisite     = "shard_prerequisite_failed"

	DependencyScenario = "scenario"
	DependencyShard    = "shard"

	MaxScenarioDependencies = 16
)

// Evidence classification. Hosted-runner evidence is CI evidence only.
const EvidenceClass = "CI_HOSTED"

// Evidence bounds.
const (
	MaxAttachmentBytes = 256 << 10
	MaxShardBytes      = 4 << 20
	maxLogLine         = 512
	maxScenarioLog     = 200
)

const requiredFile = "required.json"

var scenarioID = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

// ScenarioDependency identifies the failed prerequisite that blocked a
// scenario. Scenario dependencies name another required scenario; shard
// dependencies name a bounded shard-wide precondition.
type ScenarioDependency struct {
	Kind    string `json:"kind"`
	ID      string `json:"id"`
	Outcome string `json:"outcome"`
}

// BlockedInfo is bounded machine-readable evidence for a BLOCKED outcome.
type BlockedInfo struct {
	Reason       string               `json:"reason"`
	Dependencies []ScenarioDependency `json:"dependencies"`
}

// ScenarioResult is one scenario's outcome inside a shard result.
type ScenarioResult struct {
	ID         string       `json:"id"`
	Outcome    string       `json:"outcome"`
	DurationMS int64        `json:"duration_ms"`
	Log        []string     `json:"log,omitempty"`
	Blocked    *BlockedInfo `json:"blocked,omitempty"`
}

// CandidateRef is the candidate identity recorded in every result.
type CandidateRef struct {
	SourceCommit string `json:"source_commit"`
	File         string `json:"file"`
	SHA256       string `json:"sha256"`
	Size         int64  `json:"size"`
}

// Result is result.json: the single structured outcome of one shard.
type Result struct {
	Schema        string           `json:"schema"`
	Shard         string           `json:"shard"`
	Status        string           `json:"status"`
	EvidenceClass string           `json:"evidence_class"`
	PhysicalPass  bool             `json:"physical_pass"`
	Candidate     CandidateRef     `json:"candidate"`
	Runner        Runner           `json:"runner"`
	Run           Run              `json:"run"`
	Required      []string         `json:"required"`
	Scenarios     []ScenarioResult `json:"scenarios"`
	Problems      []string         `json:"problems,omitempty"`
	Attachments   []string         `json:"attachments,omitempty"`
	StartedAt     string           `json:"started_at"`
	FinishedAt    string           `json:"finished_at"`
}

// Enabled reports whether the process runs in certification mode.
func Enabled() bool { return os.Getenv(EnvEnable) == "1" }

// recorder is the process-wide evidence state of one shard test binary.
type recorder struct {
	mu        sync.Mutex
	scenarios map[string]*ScenarioResult
	names     map[string]int
	files     []string
	bytes     int
	started   time.Time
}

var rec = newRecorder()

func newRecorder() *recorder {
	return &recorder{scenarios: map[string]*ScenarioResult{}, names: map[string]int{}, started: time.Now().UTC()}
}

func (r *recorder) record(s ScenarioResult) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.scenarios[s.ID] = &s
}

func (r *recorder) outcome(id string) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.scenarios[id]
	if !ok {
		return "", false
	}
	return s.Outcome, true
}

func (r *recorder) results() []ScenarioResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]ScenarioResult, 0, len(r.scenarios))
	for _, scenario := range r.scenarios {
		out = append(out, *scenario)
	}
	return out
}

// Assemble builds the shard result from recorded scenarios, the required
// scenario list and any process-level problem. A required scenario that did not
// pass (absent, skipped, failed or blocked) fails the shard.
func Assemble(shard string, c Candidate, required []string, scenarios []ScenarioResult, attachments, problems []string, started, finished time.Time) Result {
	res := Result{
		Schema:        ResultSchema,
		Shard:         shard,
		EvidenceClass: EvidenceClass,
		PhysicalPass:  false,
		Candidate:     CandidateRef{SourceCommit: c.SourceCommit, File: c.File, SHA256: c.SHA256, Size: c.Size},
		Runner:        RunnerFromEnv(),
		Run:           RunFromEnv(),
		Required:      append([]string(nil), required...),
		Attachments:   attachments,
		Problems:      append([]string(nil), problems...),
		StartedAt:     started.UTC().Format(time.RFC3339),
		FinishedAt:    finished.UTC().Format(time.RFC3339),
	}
	seen := map[string]bool{}
	for _, s := range scenarios {
		seen[s.ID] = true
		res.Scenarios = append(res.Scenarios, s)
		if s.Outcome != OutcomePass && isRequired(required, s.ID) {
			res.Problems = append(res.Problems, fmt.Sprintf("required scenario %q: %s", s.ID, s.Outcome))
		}
	}
	for _, id := range required {
		if !seen[id] {
			res.Scenarios = append(res.Scenarios, ScenarioResult{ID: id, Outcome: OutcomeMissing})
			res.Problems = append(res.Problems, fmt.Sprintf("required scenario %q did not run", id))
		}
	}
	sort.Slice(res.Scenarios, func(i, j int) bool { return res.Scenarios[i].ID < res.Scenarios[j].ID })
	res.Problems = dedupe(res.Problems)
	sort.Strings(res.Problems)
	res.Status = OutcomePass
	if len(res.Problems) > 0 || len(required) == 0 {
		res.Status = OutcomeFail
		if len(required) == 0 {
			res.Problems = append(res.Problems, "shard declares no required scenarios")
		}
	}
	return res
}

func isRequired(required []string, id string) bool {
	for _, r := range required {
		if r == id {
			return true
		}
	}
	return false
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// RequiredPlan is the authoritative required scenario list plus optional
// dependency metadata. Dependencies never define additional scenarios.
type RequiredPlan struct {
	Scenarios    []string            `json:"scenarios"`
	Dependencies map[string][]string `json:"dependencies,omitempty"`
}

// LoadRequiredPlan reads required.json and validates its required scenarios
// and optional scenario-to-prerequisite mapping.
func LoadRequiredPlan(dir string) (RequiredPlan, error) {
	data, err := os.ReadFile(filepath.Join(dir, requiredFile))
	if err != nil {
		return RequiredPlan{}, err
	}
	var plan RequiredPlan
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&plan); err != nil {
		return RequiredPlan{}, fmt.Errorf("parse %s: %w", requiredFile, err)
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return RequiredPlan{}, fmt.Errorf("parse %s: trailing JSON content", requiredFile)
	}
	seen := map[string]bool{}
	for _, id := range plan.Scenarios {
		if !scenarioID.MatchString(id) {
			return RequiredPlan{}, fmt.Errorf("%s: scenario id %q is not lower-case kebab-case", requiredFile, id)
		}
		if seen[id] {
			return RequiredPlan{}, fmt.Errorf("%s: duplicate scenario id %q", requiredFile, id)
		}
		seen[id] = true
	}
	for id, dependencies := range plan.Dependencies {
		if !seen[id] {
			return RequiredPlan{}, fmt.Errorf("%s: dependency target %q is not a required scenario", requiredFile, id)
		}
		if len(dependencies) == 0 || len(dependencies) > MaxScenarioDependencies {
			return RequiredPlan{}, fmt.Errorf("%s: scenario %q must declare 1..%d dependencies", requiredFile, id, MaxScenarioDependencies)
		}
		dependencySeen := map[string]bool{}
		for _, dependency := range dependencies {
			if !scenarioID.MatchString(dependency) || !seen[dependency] {
				return RequiredPlan{}, fmt.Errorf("%s: scenario %q has invalid or non-required dependency %q", requiredFile, id, dependency)
			}
			if dependency == id || dependencySeen[dependency] {
				return RequiredPlan{}, fmt.Errorf("%s: scenario %q has a self or duplicate dependency %q", requiredFile, id, dependency)
			}
			dependencySeen[dependency] = true
		}
	}
	if err := validateDependencyCycles(plan.Dependencies); err != nil {
		return RequiredPlan{}, fmt.Errorf("%s: %w", requiredFile, err)
	}
	return plan, nil
}

// LoadRequired reads only the authoritative required scenario IDs from dir.
func LoadRequired(dir string) ([]string, error) {
	plan, err := LoadRequiredPlan(dir)
	if err != nil {
		return nil, err
	}
	return plan.Scenarios, nil
}

func validateDependencyCycles(dependencies map[string][]string) error {
	const (
		visiting = iota + 1
		visited
	)
	states := map[string]int{}
	var visit func(string) error
	visit = func(id string) error {
		if states[id] == visiting {
			return fmt.Errorf("scenario dependency cycle includes %q", id)
		}
		if states[id] == visited {
			return nil
		}
		states[id] = visiting
		for _, dependency := range dependencies[id] {
			if err := visit(dependency); err != nil {
				return err
			}
		}
		states[id] = visited
		return nil
	}
	for id := range dependencies {
		if err := visit(id); err != nil {
			return err
		}
	}
	return nil
}

// WriteResult writes <evidenceDir>/<shard>/result.json.
func WriteResult(evidenceDir string, res Result) error {
	dir := filepath.Join(evidenceDir, res.Shard)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "result.json"), append(data, '\n'), 0o644)
}

// safeName reduces an attachment name to a bare, conservative file name.
var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

func safeName(name string) string {
	name = unsafeName.ReplaceAllString(filepath.Base(name), "_")
	name = strings.Trim(name, "._")
	if name == "" {
		name = "attachment"
	}
	if len(name) > 80 {
		name = name[:80]
	}
	return name
}

// attach writes a bounded, scrubbed text attachment under the shard's
// evidence directory and returns its relative path.
func (r *recorder) attach(evidenceDir, shard, name string, data []byte, home string) (string, error) {
	text := string(data)
	truncated := false
	if len(text) > MaxAttachmentBytes {
		text, truncated = text[:MaxAttachmentBytes], true
	}
	var b strings.Builder
	sc := redact.New(home)
	for _, line := range strings.Split(text, "\n") {
		b.WriteString(sc.Line(line, maxLogLine))
		b.WriteByte('\n')
	}
	if truncated {
		b.WriteString("[truncated]\n")
	}
	out := b.String()

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.bytes+len(out) > MaxShardBytes {
		return "", errors.New("shard evidence budget exhausted")
	}
	base := safeName(name)
	r.names[base]++
	if n := r.names[base]; n > 1 {
		base = fmt.Sprintf("%d-%s", n, base)
	}
	dir := filepath.Join(evidenceDir, shard, "logs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, base), []byte(out), 0o644); err != nil {
		return "", err
	}
	r.bytes += len(out)
	rel := "logs/" + base
	r.files = append(r.files, rel)
	return rel, nil
}

func (r *recorder) attachmentList() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := append([]string(nil), r.files...)
	sort.Strings(out)
	return out
}
