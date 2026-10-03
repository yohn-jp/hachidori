package e2e

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/redact"
)

// state is the certification-mode context established by Main.
var state struct {
	shard       string
	candidate   Candidate
	evidenceDir string
}

// Main is the TestMain body of every shard package:
//
//	func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.ShardBootstrap)) }
//
// In portable mode (EnvEnable unset) it only runs the tests, whose scenarios
// skip. In certification mode it requires Windows, loads and verifies the
// candidate, runs the tests, and always writes the shard's result.json - also
// when verification fails - then returns a non-zero code unless every scenario
// listed in the package's required.json passed.
func Main(m *testing.M, shard string) int {
	if !Enabled() {
		return m.Run()
	}
	started := time.Now()
	var problems []string
	var c Candidate

	if runtime.GOOS != "windows" {
		problems = append(problems, "certification mode requires GOOS=windows, got "+runtime.GOOS)
	}
	evidenceDir := os.Getenv(EnvEvidenceDir)
	if evidenceDir == "" {
		fmt.Fprintf(os.Stderr, "e2e: %s is not set\n", EnvEvidenceDir)
		return 2
	}
	required, err := LoadRequired(".")
	if err != nil {
		problems = append(problems, "required scenarios: "+err.Error())
	}
	if dir := os.Getenv(EnvCandidateDir); dir == "" {
		problems = append(problems, EnvCandidateDir+" is not set")
	} else if exp := ExpectFromEnv(); exp.SHA256 == "" || exp.SourceCommit == "" {
		problems = append(problems, EnvExpectedSHA256+" and "+EnvExpectedCommit+" must both be set; the candidate identity is verified against values passed outside the artifact")
	} else if c, err = LoadCandidate(dir, exp); err != nil {
		problems = append(problems, "candidate identity: "+err.Error())
	}

	code := 0
	if len(problems) == 0 {
		state.shard, state.candidate, state.evidenceDir = shard, c, evidenceDir
		code = m.Run()
	}

	rec.mu.Lock()
	var scenarios []ScenarioResult
	for _, s := range rec.scenarios {
		scenarios = append(scenarios, *s)
	}
	rec.mu.Unlock()
	res := Assemble(shard, c, required, scenarios, rec.attachmentList(), problems, started, time.Now())
	if code != 0 && res.Status == OutcomePass {
		res.Status = OutcomeFail
		res.Problems = append(res.Problems, "test binary exited non-zero")
	}
	if err := WriteResult(evidenceDir, res); err != nil {
		fmt.Fprintln(os.Stderr, "e2e: write result:", err)
		return 2
	}
	for _, p := range res.Problems {
		fmt.Fprintln(os.Stderr, "e2e:", p)
	}
	if res.Status != OutcomePass {
		return 1
	}
	return 0
}

// Scenario is one named certification scenario of a shard.
type Scenario struct {
	t     *testing.T
	id    string
	log   []string
	start time.Time
}

// Begin starts the scenario id (lower-case kebab-case, listed in the shard's
// required.json). In portable mode it skips the test. Its outcome is recorded
// when the test ends: a failed test is FAIL, a skipped test is SKIP (which does
// not satisfy a required scenario) and anything else is PASS.
func Begin(t *testing.T, id string) *Scenario {
	t.Helper()
	if !Enabled() {
		t.Skip("Windows E2E certification is disabled; set " + EnvEnable + "=1 on a Windows runner")
	}
	if !scenarioID.MatchString(id) {
		t.Fatalf("scenario id %q is not lower-case kebab-case", id)
	}
	s := &Scenario{t: t, id: id, start: time.Now()}
	t.Cleanup(s.finish)
	return s
}

func (s *Scenario) finish() {
	outcome := OutcomePass
	switch {
	case s.t.Failed():
		outcome = OutcomeFail
	case s.t.Skipped():
		outcome = OutcomeSkip
	}
	rec.record(ScenarioResult{
		ID:         s.id,
		Outcome:    outcome,
		DurationMS: time.Since(s.start).Milliseconds(),
		Log:        s.log,
	})
}

// Candidate returns the verified candidate this run certifies.
func (s *Scenario) Candidate() Candidate { return state.candidate }

// Logf records one scrubbed, bounded evidence line for the scenario. It is not
// a place for secrets or request content.
func (s *Scenario) Logf(format string, args ...any) {
	if len(s.log) >= maxScenarioLog {
		return
	}
	line := redact.New("").Line(fmt.Sprintf(format, args...), maxLogLine)
	s.log = append(s.log, line)
	s.t.Log(line)
}

// Attach stores a bounded, scrubbed text attachment (for example a process log
// tail) in the shard's evidence directory. home is the disposable Hachidori
// home whose path is replaced by a placeholder; "" for none.
func (s *Scenario) Attach(name string, data []byte, home string) {
	s.t.Helper()
	rel, err := rec.attach(state.evidenceDir, state.shard, name, data, home)
	if err != nil {
		s.t.Errorf("attach %s: %v", name, err)
		return
	}
	s.Logf("attachment %s", rel)
}

// VerifyCandidateScenario is the shared "candidate-identity" scenario every
// shard runs: it re-hashes the executable at the moment the shard certifies it
// and checks that it is a Windows amd64 PE image, so a shard can never certify
// bytes that differ from the single candidate the workflow built.
func VerifyCandidateScenario(t *testing.T) {
	t.Helper()
	s := Begin(t, "candidate-identity")
	c := s.Candidate()
	sum, size, err := HashFile(c.Path)
	if err != nil {
		t.Fatalf("hash candidate: %v", err)
	}
	if sum != c.SHA256 || size != c.Size {
		t.Fatalf("candidate changed after verification: sha256 %s size %d, manifest sha256 %s size %d", sum, size, c.SHA256, c.Size)
	}
	if err := checkPE(c.Path); err != nil {
		t.Fatalf("candidate is not a Windows amd64 executable: %v", err)
	}
	s.Logf("candidate %s sha256=%s commit=%s", c.File, c.SHA256, c.SourceCommit)
}
