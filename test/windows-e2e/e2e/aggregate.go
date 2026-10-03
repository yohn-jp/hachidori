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
const CertificationSchema = "hachidori.windows-e2e.certification/v1"

// EvidenceDirPrefix is the directory prefix under which the evidence artifact
// of each shard is downloaded: <evidence>/<prefix><shard>/result.json.
const EvidenceDirPrefix = "windows-e2e-evidence-"

// ShardVerdict is the aggregate's view of one required shard.
type ShardVerdict struct {
	Shard     string   `json:"shard"`
	Status    string   `json:"status"`
	Scenarios int      `json:"scenarios"`
	Problems  []string `json:"problems,omitempty"`
}

// Certification is certification.json. Status is PASS only when the candidate
// and every required shard job succeeded and every required shard produced a
// passing result for exactly this candidate. Anything else - a failed job, a
// missing result, a mismatched identity - is FAIL.
type Certification struct {
	Schema         string         `json:"schema"`
	Status         string         `json:"status"`
	EvidenceClass  string         `json:"evidence_class"`
	PhysicalPass   bool           `json:"physical_pass"`
	PhysicalChecks string         `json:"physical_checks"`
	Candidate      CandidateRef   `json:"candidate"`
	Run            Run            `json:"run"`
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

	required, err := LoadRequired(filepath.Join(in.TestsDir, shard))
	if err != nil {
		fail("required scenarios: %v", err)
	} else if !isRequired(required, "candidate-identity") {
		fail("required.json must list candidate-identity")
	}

	path := filepath.Join(in.EvidenceDir, EvidenceDirPrefix+shard, "result.json")
	data, err := os.ReadFile(path)
	if err != nil {
		fail("no result evidence: %v", err)
		return v
	}
	var res Result
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&res); err != nil {
		fail("result.json does not parse: %v", err)
		return v
	}
	v.Scenarios = len(res.Scenarios)

	if res.Schema != ResultSchema {
		fail("result schema %q, want %q", res.Schema, ResultSchema)
	}
	if res.Shard != shard {
		fail("result names shard %q", res.Shard)
	}
	if res.Status != OutcomePass {
		fail("shard status is %q", res.Status)
	}
	if len(res.Problems) > 0 {
		fail("shard reported problems: %s", strings.Join(res.Problems, "; "))
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

	want := append([]string(nil), required...)
	got := append([]string(nil), res.Required...)
	sort.Strings(want)
	sort.Strings(got)
	if strings.Join(want, ",") != strings.Join(got, ",") {
		fail("result required scenarios %v differ from required.json %v", got, want)
	}
	outcome := map[string]string{}
	for _, s := range res.Scenarios {
		outcome[s.ID] = s.Outcome
	}
	for _, id := range required {
		if outcome[id] != OutcomePass {
			fail("required scenario %q outcome %q", id, outcome[id])
		}
	}
	return v
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

// Markdown renders the verdict for the job summary. It contains identity and
// outcomes only.
func (c Certification) Markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Windows E2E certification: %s\n\n", c.Status)
	fmt.Fprintf(&b, "- candidate: `%s` sha256 `%s` (%d bytes)\n", c.Candidate.File, c.Candidate.SHA256, c.Candidate.Size)
	fmt.Fprintf(&b, "- source commit: `%s`\n", c.Candidate.SourceCommit)
	fmt.Fprintf(&b, "- evidence class: `%s`; physical checks: `%s` (hosted evidence is never a physical PASS)\n\n", c.EvidenceClass, c.PhysicalChecks)
	b.WriteString("| shard | status | scenarios |\n|---|---|---|\n")
	for _, s := range c.Shards {
		fmt.Fprintf(&b, "| %s | %s | %d |\n", s.Shard, s.Status, s.Scenarios)
	}
	if len(c.Problems) > 0 {
		b.WriteString("\n### Problems\n\n")
		for _, p := range c.Problems {
			fmt.Fprintf(&b, "- %s\n", strings.ReplaceAll(p, "\n", " "))
		}
	}
	return b.String()
}

// CheckRelease is the release gate's decision over a downloaded
// certification.json: it must be a PASS for exactly the expected candidate
// and cover every required shard. It never trusts the status field alone.
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
	for _, s := range c.Shards {
		if s.Status != OutcomePass || len(s.Problems) > 0 {
			return fmt.Errorf("shard %s is %s", s.Shard, s.Status)
		}
		seen[s.Shard] = true
	}
	for _, s := range Shards {
		if !seen[s] {
			return fmt.Errorf("certification has no result for required shard %s", s)
		}
	}
	return nil
}
