package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// CertificationSchema identifies certification.json, the aggregate verdict of
// one certification run.
const CertificationSchema = "hachidori.windows-e2e.certification/v2"

// EvidenceDirPrefix is the directory prefix under which the evidence artifact
// of each shard is downloaded: <evidence>/<prefix><shard>/result.json.
const EvidenceDirPrefix = "windows-e2e-evidence-"

// ScenarioVerdict is the bounded aggregate view of one required scenario.
type ScenarioVerdict struct {
	ID      string       `json:"id"`
	Outcome string       `json:"outcome"`
	Blocked *BlockedInfo `json:"blocked,omitempty"`
}

// OutcomeCounts are stable counts over required scenarios across all shards.
type OutcomeCounts struct {
	Pass    int `json:"PASS"`
	Fail    int `json:"FAIL"`
	Blocked int `json:"BLOCKED"`
	Skip    int `json:"SKIP"`
	Missing int `json:"MISSING"`
}

func (c *OutcomeCounts) add(outcome string) {
	switch outcome {
	case OutcomePass:
		c.Pass++
	case OutcomeFail:
		c.Fail++
	case OutcomeBlocked:
		c.Blocked++
	case OutcomeSkip:
		c.Skip++
	case OutcomeMissing:
		c.Missing++
	}
}

// ShardVerdict is the aggregate's view of one required shard.
type ShardVerdict struct {
	Shard     string            `json:"shard"`
	Status    string            `json:"status"`
	Scenarios int               `json:"scenarios"`
	Outcomes  []ScenarioVerdict `json:"outcomes"`
	Problems  []string          `json:"problems,omitempty"`
}

// Certification is certification.json. Status is PASS only when the candidate
// and every required shard job succeeded and every required scenario passed
// for exactly this candidate. Anything else - including BLOCKED, SKIP and
// MISSING - is FAIL.
type Certification struct {
	Schema         string         `json:"schema"`
	Status         string         `json:"status"`
	EvidenceClass  string         `json:"evidence_class"`
	PhysicalPass   bool           `json:"physical_pass"`
	PhysicalChecks string         `json:"physical_checks"`
	Candidate      CandidateRef   `json:"candidate"`
	Run            Run            `json:"run"`
	Counts         OutcomeCounts  `json:"counts"`
	Shards         []ShardVerdict `json:"shards"`
	Problems       []string       `json:"problems,omitempty"`
}

// PhysicalChecksNotChecked states the standing physical-certification outcome:
// hosted evidence never changes it.
const PhysicalChecksNotChecked = "NOT_CHECKED"

// AggregateInput is everything Aggregate decides from.
type AggregateInput struct {
	// EvidenceDir holds <EvidenceDirPrefix><shard>/result.json per shard.
	EvidenceDir string
	// TestsDir is test/windows-e2e, holding <shard>/required.json.
	TestsDir string
	// Candidate is the candidate verified against the out-of-band identity.
	Candidate Candidate
	// RunID is the workflow run the results must belong to; "" skips the check.
	RunID string
	// JobResults maps "candidate" and "shard" to the needs.<job>.result values.
	JobResults map[string]string
}

// Aggregate fails closed. It never reads a shard's own verdict as sufficient:
// it recomputes the required scenario set from the repository, compares the
// candidate identity and run identity, and requires every shard.
func Aggregate(in AggregateInput) Certification {
	cert := Certification{
		Schema:         CertificationSchema,
		EvidenceClass:  EvidenceClass,
		PhysicalPass:   false,
		PhysicalChecks: PhysicalChecksNotChecked,
		Candidate: CandidateRef{
			SourceCommit: in.Candidate.SourceCommit,
			File:         in.Candidate.File,
			SHA256:       in.Candidate.SHA256,
			Size:         in.Candidate.Size,
		},
		Run: RunFromEnv(),
	}
	for _, job := range []string{"candidate", "shard"} {
		if got := in.JobResults[job]; got != "success" {
			cert.Problems = append(cert.Problems, fmt.Sprintf("job %s result is %q, want success", job, got))
		}
	}
	for _, shard := range Shards {
		v := verifyShard(in, shard)
		if len(v.Problems) > 0 {
			v.Status = OutcomeFail
		} else {
			v.Status = OutcomePass
		}
		cert.Shards = append(cert.Shards, v)
		for _, outcome := range v.Outcomes {
			cert.Counts.add(outcome.Outcome)
		}
		for _, p := range v.Problems {
			cert.Problems = append(cert.Problems, shard+": "+p)
		}
	}
	cert.Status = OutcomePass
	if len(cert.Problems) > 0 {
		cert.Status = OutcomeFail
	}
	return cert
}

func verifyShard(in AggregateInput, shard string) ShardVerdict {
	v := ShardVerdict{Shard: shard}
	fail := func(format string, args ...any) { v.Problems = append(v.Problems, fmt.Sprintf(format, args...)) }

	plan, err := LoadRequiredPlan(filepath.Join(in.TestsDir, shard))
	if err != nil {
		fail("required scenarios: %v", err)
	} else if !isRequired(plan.Scenarios, "candidate-identity") {
		fail("required.json must list candidate-identity")
	}

	path := filepath.Join(in.EvidenceDir, EvidenceDirPrefix+shard, "result.json")
	data, err := os.ReadFile(path)
	if err != nil {
		fail("no result evidence: %v", err)
		addMissingOutcomes(&v, plan.Scenarios, "no result evidence", fail)
		return stableShardVerdict(v)
	}
	var res Result
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&res); err != nil {
		fail("result.json does not parse: %v", err)
		addMissingOutcomes(&v, plan.Scenarios, "result cannot be parsed", fail)
		return stableShardVerdict(v)
	}

	if res.Schema != ResultSchema {
		fail("result schema %q, want %q", res.Schema, ResultSchema)
	}
	if res.Shard != shard {
		fail("result names shard %q", res.Shard)
	}
	if res.Status != OutcomePass {
		fail("shard status is %q", res.Status)
	}
	if res.EvidenceClass != EvidenceClass || res.PhysicalPass {
		fail("evidence class %q physical_pass=%v; hosted evidence is never a physical PASS", res.EvidenceClass, res.PhysicalPass)
	}
	c := in.Candidate
	if res.Candidate.SourceCommit != c.SourceCommit || res.Candidate.SHA256 != c.SHA256 ||
		res.Candidate.File != c.File || res.Candidate.Size != c.Size {
		fail("result certifies commit %s sha256 %s, not the candidate commit %s sha256 %s",
			res.Candidate.SourceCommit, res.Candidate.SHA256, c.SourceCommit, c.SHA256)
	}
	if in.RunID != "" && res.Run.ID != in.RunID {
		fail("result belongs to run %q, want %q", res.Run.ID, in.RunID)
	}

	want := append([]string(nil), plan.Scenarios...)
	got := append([]string(nil), res.Required...)
	sort.Strings(want)
	sort.Strings(got)
	if strings.Join(want, ",") != strings.Join(got, ",") {
		fail("result required scenarios %v differ from required.json %v", got, want)
	}

	byID := make(map[string]ScenarioResult, len(res.Scenarios))
	duplicates := map[string]bool{}
	for _, scenario := range res.Scenarios {
		if !isRequired(plan.Scenarios, scenario.ID) {
			fail("result contains an unrequired scenario")
		}
		if _, exists := byID[scenario.ID]; exists {
			duplicates[scenario.ID] = true
			fail("result contains a duplicate scenario")
			continue
		}
		byID[scenario.ID] = scenario
	}
	for _, problem := range res.Problems {
		fail("shard reported problem: %s", problem)
	}

	v.Outcomes = make([]ScenarioVerdict, 0, len(plan.Scenarios))
	for _, id := range plan.Scenarios {
		scenario, exists := byID[id]
		if !exists || duplicates[id] {
			v.Outcomes = append(v.Outcomes, ScenarioVerdict{ID: id, Outcome: OutcomeMissing})
			fail("required scenario %q outcome %s", id, OutcomeMissing)
			continue
		}
		if err := validateScenarioResult(scenario); err != nil {
			v.Outcomes = append(v.Outcomes, ScenarioVerdict{ID: id, Outcome: OutcomeMissing})
			fail("required scenario %q has invalid outcome evidence: %v", id, err)
			continue
		}
		if err := validateScenarioDependencies(scenario, plan, byID, duplicates); err != nil {
			v.Outcomes = append(v.Outcomes, ScenarioVerdict{ID: id, Outcome: OutcomeMissing})
			fail("required scenario %q has invalid dependency outcome: %v", id, err)
			continue
		}
		verdict := ScenarioVerdict{ID: id, Outcome: scenario.Outcome, Blocked: scenario.Blocked}
		v.Outcomes = append(v.Outcomes, verdict)
		if verdict.Outcome != OutcomePass {
			fail("required scenario %q outcome %s%s", id, verdict.Outcome, blockedDescription(verdict.Blocked))
		}
	}
	v.Scenarios = len(v.Outcomes)
	return stableShardVerdict(v)
}

func stableShardVerdict(v ShardVerdict) ShardVerdict {
	sort.Strings(v.Problems)
	v.Problems = dedupe(v.Problems)
	return v
}

func addMissingOutcomes(v *ShardVerdict, required []string, reason string, fail func(string, ...any)) {
	v.Outcomes = make([]ScenarioVerdict, 0, len(required))
	for _, id := range required {
		v.Outcomes = append(v.Outcomes, ScenarioVerdict{ID: id, Outcome: OutcomeMissing})
		fail("required scenario %q outcome %s (%s)", id, OutcomeMissing, reason)
	}
	v.Scenarios = len(v.Outcomes)
}

func validateScenarioResult(s ScenarioResult) error {
	if !scenarioID.MatchString(s.ID) {
		return fmt.Errorf("invalid scenario id")
	}
	switch s.Outcome {
	case OutcomePass, OutcomeFail, OutcomeBlocked, OutcomeSkip, OutcomeMissing:
	default:
		return fmt.Errorf("unknown outcome")
	}
	if s.Outcome != OutcomeBlocked {
		if s.Blocked != nil {
			return fmt.Errorf("%s outcome must not have blocked metadata", s.Outcome)
		}
		return nil
	}
	if s.Blocked == nil {
		return fmt.Errorf("BLOCKED outcome has no dependency evidence")
	}
	if len(s.Blocked.Dependencies) == 0 || len(s.Blocked.Dependencies) > MaxScenarioDependencies {
		return fmt.Errorf("BLOCKED outcome must name 1..%d dependencies", MaxScenarioDependencies)
	}
	seen := map[string]bool{}
	kind := ""
	for _, dependency := range s.Blocked.Dependencies {
		if !scenarioID.MatchString(dependency.ID) {
			return fmt.Errorf("invalid prerequisite id")
		}
		if dependency.Outcome == OutcomePass || !validOutcome(dependency.Outcome) {
			return fmt.Errorf("prerequisite has a non-blocking outcome")
		}
		if dependency.Kind == DependencyShard && dependency.Outcome != OutcomeFail {
			return fmt.Errorf("shard prerequisite %q must have FAIL outcome", dependency.ID)
		}
		if dependency.Kind != DependencyScenario && dependency.Kind != DependencyShard {
			return fmt.Errorf("prerequisite has an unknown kind")
		}
		if kind != "" && kind != dependency.Kind {
			return fmt.Errorf("BLOCKED outcome mixes scenario and shard prerequisites")
		}
		kind = dependency.Kind
		key := dependency.Kind + ":" + dependency.ID
		if seen[key] {
			return fmt.Errorf("duplicate prerequisite")
		}
		seen[key] = true
		if dependency.Kind == DependencyShard && dependency.ID != "windows-runner" {
			return fmt.Errorf("unknown shard prerequisite")
		}
	}
	wantReason := BlockReasonPrerequisiteNotPassed
	if kind == DependencyShard {
		wantReason = BlockReasonShardPrerequisite
	}
	if s.Blocked.Reason != wantReason {
		return fmt.Errorf("BLOCKED reason does not match prerequisite kind")
	}
	return nil
}

func validOutcome(outcome string) bool {
	switch outcome {
	case OutcomePass, OutcomeFail, OutcomeBlocked, OutcomeSkip, OutcomeMissing:
		return true
	default:
		return false
	}
}

func validateScenarioDependencies(s ScenarioResult, plan RequiredPlan, results map[string]ScenarioResult, duplicates map[string]bool) error {
	if s.Blocked != nil && s.Blocked.Dependencies[0].Kind == DependencyShard {
		if s.Outcome != OutcomeBlocked || len(s.Blocked.Dependencies) != 1 || s.Blocked.Dependencies[0].ID != "windows-runner" {
			return fmt.Errorf("shard prerequisite must block a scenario on the Windows runner")
		}
		return nil
	}
	declared := plan.Dependencies[s.ID]
	observed := map[string]string{}
	for _, id := range declared {
		outcome := OutcomeMissing
		if dependency, ok := results[id]; ok && !duplicates[id] && validateScenarioResult(dependency) == nil {
			outcome = dependency.Outcome
		}
		if outcome != OutcomePass {
			observed[id] = outcome
		}
	}
	if len(observed) == 0 {
		if s.Outcome == OutcomeBlocked {
			return fmt.Errorf("BLOCKED scenario has no failed declared prerequisite")
		}
		return nil
	}
	if s.Outcome != OutcomeBlocked || s.Blocked == nil {
		return fmt.Errorf("scenario must be BLOCKED because %d declared prerequisite(s) did not PASS", len(observed))
	}
	if len(observed) != len(s.Blocked.Dependencies) {
		return fmt.Errorf("BLOCKED dependencies differ from declared non-PASS prerequisites")
	}
	for _, dependency := range s.Blocked.Dependencies {
		if dependency.Kind != DependencyScenario || observed[dependency.ID] != dependency.Outcome {
			return fmt.Errorf("prerequisite outcome does not match result or dependency declaration")
		}
	}
	return nil
}

func blockedDescription(blocked *BlockedInfo) string {
	if blocked == nil {
		return ""
	}
	var dependencies []string
	for _, dependency := range blocked.Dependencies {
		dependencies = append(dependencies, fmt.Sprintf("%s=%s", dependency.ID, dependency.Outcome))
	}
	return " (" + blocked.Reason + ": " + strings.Join(dependencies, ", ") + ")"
}

// ValidateResult validates result/v2 scenario outcomes and BLOCKED metadata.
func ValidateResult(res Result) error {
	if res.Schema != ResultSchema {
		return fmt.Errorf("result schema %q, want %q", res.Schema, ResultSchema)
	}
	if res.Status != OutcomePass && res.Status != OutcomeFail {
		return fmt.Errorf("result status %q is invalid", res.Status)
	}
	if !isShard(res.Shard) {
		return fmt.Errorf("unknown shard %q", res.Shard)
	}
	required := make(map[string]bool, len(res.Required))
	for _, id := range res.Required {
		if !scenarioID.MatchString(id) || required[id] {
			return fmt.Errorf("invalid or duplicate required scenario %q", id)
		}
		required[id] = true
	}
	if len(required) == 0 {
		return fmt.Errorf("result declares no required scenarios")
	}
	seen := map[string]bool{}
	allPass := len(res.Problems) == 0
	for _, scenario := range res.Scenarios {
		if !required[scenario.ID] {
			return fmt.Errorf("unrequired scenario %q", scenario.ID)
		}
		if seen[scenario.ID] {
			return fmt.Errorf("duplicate scenario %q", scenario.ID)
		}
		seen[scenario.ID] = true
		if err := validateScenarioResult(scenario); err != nil {
			return fmt.Errorf("scenario %q: %w", scenario.ID, err)
		}
		if scenario.Outcome != OutcomePass {
			allPass = false
		}
	}
	for id := range required {
		if !seen[id] {
			return fmt.Errorf("required scenario %q has no outcome", id)
		}
	}
	if res.Status == OutcomePass && !allPass {
		return fmt.Errorf("PASS result contains a non-PASS or problem")
	}
	return nil
}

func isShard(shard string) bool {
	for _, required := range Shards {
		if shard == required {
			return true
		}
	}
	return false
}

// WriteCertification writes certification.json and certification.md into dir.
func WriteCertification(dir string, cert Certification) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cert, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "certification.json"), append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "certification.md"), []byte(cert.Markdown()), 0o644)
}

// Markdown renders the verdict and stable outcome counts for the GitHub job
// summary. It contains scenario identities and bounded dependency metadata,
// never scenario logs or semantic payloads.
func (c Certification) Markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Windows E2E certification: %s\n\n", c.Status)
	fmt.Fprintf(&b, "- candidate: `%s` sha256 `%s` (%d bytes)\n", c.Candidate.File, c.Candidate.SHA256, c.Candidate.Size)
	fmt.Fprintf(&b, "- source commit: `%s`\n", c.Candidate.SourceCommit)
	fmt.Fprintf(&b, "- evidence class: `%s`; physical checks: `%s` (hosted evidence is never a physical PASS)\n\n", c.EvidenceClass, c.PhysicalChecks)
	b.WriteString("### Scenario outcome counts\n\n")
	b.WriteString("| PASS | FAIL | BLOCKED | SKIP | MISSING |\n|---:|---:|---:|---:|---:|\n")
	fmt.Fprintf(&b, "| %d | %d | %d | %d | %d |\n\n", c.Counts.Pass, c.Counts.Fail, c.Counts.Blocked, c.Counts.Skip, c.Counts.Missing)
	b.WriteString("### Shards\n\n| shard | status | scenarios |\n|---|---|---:|\n")
	for _, s := range c.Shards {
		fmt.Fprintf(&b, "| %s | %s | %d |\n", s.Shard, s.Status, s.Scenarios)
	}
	for _, shard := range c.Shards {
		var points []ScenarioVerdict
		for _, outcome := range shard.Outcomes {
			if outcome.Outcome != OutcomePass {
				points = append(points, outcome)
			}
		}
		if len(points) == 0 && len(shard.Problems) == 0 {
			continue
		}
		fmt.Fprintf(&b, "\n#### %s\n\n", shard.Shard)
		for _, point := range points {
			fmt.Fprintf(&b, "- scenario `%s`: **%s**%s\n", point.ID, point.Outcome, blockedDescription(point.Blocked))
		}
		for _, problem := range shard.Problems {
			if !isScenarioOutcomeProblem(problem) {
				fmt.Fprintf(&b, "- %s\n", strings.ReplaceAll(problem, "\n", " "))
			}
		}
	}
	if len(c.Problems) > 0 {
		var runProblems []string
		for _, problem := range c.Problems {
			if strings.HasPrefix(problem, "job ") {
				runProblems = append(runProblems, problem)
			}
		}
		if len(runProblems) > 0 {
			b.WriteString("\n### Run problems\n\n")
			for _, p := range runProblems {
				fmt.Fprintf(&b, "- %s\n", strings.ReplaceAll(p, "\n", " "))
			}
		}
	}
	return b.String()
}

func isScenarioOutcomeProblem(problem string) bool {
	for _, outcome := range []string{OutcomePass, OutcomeFail, OutcomeBlocked, OutcomeSkip, OutcomeMissing} {
		if strings.Contains(problem, `" outcome `+outcome) || strings.Contains(problem, `": `+outcome) {
			return true
		}
	}
	return strings.Contains(problem, " did not run")
}

// CheckRelease is the release gate's decision over a downloaded
// certification.json: it must be a PASS for exactly the expected candidate
// and cover every required shard, with only passing scenario outcomes.
func CheckRelease(data []byte, expect Expect, file string) error {
	var c Certification
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return fmt.Errorf("parse certification: %w", err)
	}
	if c.Schema != CertificationSchema {
		return fmt.Errorf("certification schema %q, want %q", c.Schema, CertificationSchema)
	}
	if c.Status != OutcomePass || len(c.Problems) > 0 {
		return fmt.Errorf("certification status %q with %d problems", c.Status, len(c.Problems))
	}
	if c.PhysicalPass || c.EvidenceClass != EvidenceClass {
		return fmt.Errorf("certification claims physical_pass=%v class %q", c.PhysicalPass, c.EvidenceClass)
	}
	if expect.SHA256 == "" || expect.SourceCommit == "" {
		return fmt.Errorf("the expected candidate SHA-256 and source commit are required")
	}
	if !strings.EqualFold(c.Candidate.SHA256, expect.SHA256) || !strings.EqualFold(c.Candidate.SourceCommit, expect.SourceCommit) {
		return fmt.Errorf("certification covers commit %s sha256 %s, not the release candidate commit %s sha256 %s",
			c.Candidate.SourceCommit, c.Candidate.SHA256, expect.SourceCommit, expect.SHA256)
	}
	if file != "" && c.Candidate.File != file {
		return fmt.Errorf("certification covers file %q, not %q", c.Candidate.File, file)
	}
	seen := map[string]bool{}
	var counts OutcomeCounts
	for _, shard := range c.Shards {
		if !isShard(shard.Shard) || seen[shard.Shard] {
			return fmt.Errorf("unknown or duplicate shard %q", shard.Shard)
		}
		if shard.Status != OutcomePass || len(shard.Problems) > 0 {
			return fmt.Errorf("shard %s is %s", shard.Shard, shard.Status)
		}
		if shard.Scenarios == 0 || shard.Scenarios != len(shard.Outcomes) {
			return fmt.Errorf("shard %s scenario count does not match outcomes", shard.Shard)
		}
		seenScenarios := map[string]bool{}
		for _, outcome := range shard.Outcomes {
			if !scenarioID.MatchString(outcome.ID) || seenScenarios[outcome.ID] || outcome.Outcome != OutcomePass || outcome.Blocked != nil {
				return fmt.Errorf("shard %s scenario %s is invalid or not PASS", shard.Shard, outcome.ID)
			}
			seenScenarios[outcome.ID] = true
			counts.add(outcome.Outcome)
		}
		seen[shard.Shard] = true
	}
	if counts != c.Counts {
		return fmt.Errorf("certification outcome counts do not match shard results")
	}
	for _, shard := range Shards {
		if !seen[shard] {
			return fmt.Errorf("certification has no result for required shard %s", shard)
		}
	}
	return nil
}
