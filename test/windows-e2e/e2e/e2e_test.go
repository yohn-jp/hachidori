package e2e

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const commitA = "0123456789abcdef0123456789abcdef01234567"

func writeExe(t *testing.T, dir, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "hachidori.exe"), []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestCandidateRoundTripAndExpectations(t *testing.T) {
	dir := t.TempDir()
	writeExe(t, dir, "candidate-bytes")
	m, err := WriteCandidate(dir, "hachidori.exe", commitA, "windows", "amd64", "go1.24")
	if err != nil {
		t.Fatal(err)
	}
	c, err := LoadCandidate(dir, Expect{SourceCommit: commitA, SHA256: m.SHA256})
	if err != nil {
		t.Fatalf("LoadCandidate: %v", err)
	}
	if c.SHA256 != m.SHA256 || c.Path != filepath.Join(dir, "hachidori.exe") || c.Size != int64(len("candidate-bytes")) {
		t.Fatalf("unexpected candidate %+v", c)
	}
	if _, err := LoadCandidate(dir, Expect{SHA256: strings.Repeat("0", 64)}); err == nil {
		t.Fatal("a different expected SHA-256 must be rejected")
	}
	if _, err := LoadCandidate(dir, Expect{SourceCommit: strings.Repeat("a", 40)}); err == nil {
		t.Fatal("a different expected commit must be rejected")
	}
}

func TestLoadCandidateRejectsTamperingAndMalformedManifests(t *testing.T) {
	dir := t.TempDir()
	writeExe(t, dir, "candidate-bytes")
	if _, err := WriteCandidate(dir, "hachidori.exe", commitA, "windows", "amd64", ""); err != nil {
		t.Fatal(err)
	}
	writeExe(t, dir, "candidate-bytez")
	if _, err := LoadCandidate(dir, Expect{}); err == nil {
		t.Fatal("a modified executable must fail verification")
	}

	empty := t.TempDir()
	if _, err := LoadCandidate(empty, Expect{}); err == nil {
		t.Fatal("a missing manifest must fail")
	}

	for name, mutate := range map[string]func(m map[string]any){
		"schema":    func(m map[string]any) { m["schema"] = "other/v1" },
		"traversal": func(m map[string]any) { m["file"] = "../hachidori.exe" },
		"separator": func(m map[string]any) { m["file"] = `sub\hachidori.exe` },
		"sha":       func(m map[string]any) { m["sha256"] = "XYZ" },
		"commit":    func(m map[string]any) { m["source_commit"] = "main" },
		"unknown":   func(m map[string]any) { m["extra"] = true },
	} {
		t.Run(name, func(t *testing.T) {
			d := t.TempDir()
			writeExe(t, d, "x")
			if _, err := WriteCandidate(d, "hachidori.exe", commitA, "windows", "amd64", ""); err != nil {
				t.Fatal(err)
			}
			data, _ := os.ReadFile(filepath.Join(d, ManifestName))
			var m map[string]any
			if err := json.Unmarshal(data, &m); err != nil {
				t.Fatal(err)
			}
			mutate(m)
			out, _ := json.Marshal(m)
			if err := os.WriteFile(filepath.Join(d, ManifestName), out, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadCandidate(d, Expect{}); err == nil {
				t.Fatalf("%s manifest must be rejected", name)
			}
		})
	}
}

func TestWriteCandidateRejectsBadInput(t *testing.T) {
	dir := t.TempDir()
	if _, err := WriteCandidate(dir, "hachidori.exe", commitA, "windows", "amd64", ""); err == nil {
		t.Fatal("a missing executable must fail")
	}
	writeExe(t, dir, "x")
	if _, err := WriteCandidate(dir, "hachidori.exe", "abc", "windows", "amd64", ""); err == nil {
		t.Fatal("a short commit must fail")
	}
	if _, err := WriteCandidate(dir, "../hachidori.exe", commitA, "windows", "amd64", ""); err == nil {
		t.Fatal("a path file name must fail")
	}
	writeExe(t, dir, "")
	if _, err := WriteCandidate(dir, "hachidori.exe", commitA, "windows", "amd64", ""); err == nil {
		t.Fatal("an empty executable must fail")
	}
}

func TestAssembleFailsClosed(t *testing.T) {
	c := Candidate{Manifest: Manifest{SourceCommit: commitA, File: "hachidori.exe", SHA256: strings.Repeat("a", 64), Size: 1}}
	now := time.Now()
	pass := ScenarioResult{ID: "candidate-identity", Outcome: OutcomePass}

	res := Assemble("update", c, []string{"candidate-identity"}, []ScenarioResult{pass}, nil, nil, now, now)
	if res.Status != OutcomePass || res.PhysicalPass || res.EvidenceClass != EvidenceClass {
		t.Fatalf("all required passing must be PASS and never physical: %+v", res)
	}

	cases := map[string]Result{
		"missing": Assemble("update", c, []string{"candidate-identity", "other"}, []ScenarioResult{pass}, nil, nil, now, now),
		"skipped": Assemble("update", c, []string{"candidate-identity"}, []ScenarioResult{{ID: "candidate-identity", Outcome: OutcomeSkip}}, nil, nil, now, now),
		"blocked": Assemble("update", c, []string{"candidate-identity"}, []ScenarioResult{{ID: "candidate-identity", Outcome: OutcomeBlocked, Blocked: &BlockedInfo{Reason: BlockReasonPrerequisiteNotPassed, Dependencies: []ScenarioDependency{{Kind: DependencyScenario, ID: "other", Outcome: OutcomeFail}}}}}, nil, nil, now, now),
		"failed":  Assemble("update", c, []string{"candidate-identity"}, []ScenarioResult{{ID: "candidate-identity", Outcome: OutcomeFail}}, nil, nil, now, now),
		"problem": Assemble("update", c, []string{"candidate-identity"}, []ScenarioResult{pass}, nil, []string{"candidate identity: boom"}, now, now),
		"none":    Assemble("update", c, nil, []ScenarioResult{pass}, nil, nil, now, now),
	}
	for name, r := range cases {
		if r.Status != OutcomeFail || len(r.Problems) == 0 {
			t.Errorf("%s: want FAIL with problems, got %+v", name, r)
		}
	}
	if got := cases["missing"].Scenarios; got[len(got)-1].Outcome != OutcomeMissing && got[0].Outcome != OutcomeMissing {
		t.Errorf("a required scenario that never ran must be recorded MISSING: %+v", got)
	}
}

func TestLoadRequired(t *testing.T) {
	dir := t.TempDir()
	write := func(s string) {
		if err := os.WriteFile(filepath.Join(dir, requiredFile), []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(`{"scenarios":["candidate-identity","runtime-ready"],"dependencies":{"runtime-ready":["candidate-identity"]}}`)
	plan, err := LoadRequiredPlan(dir)
	if err != nil || len(plan.Scenarios) != 2 || len(plan.Dependencies["runtime-ready"]) != 1 {
		t.Fatalf("LoadRequiredPlan = %+v, %v", plan, err)
	}
	got, err := LoadRequired(dir)
	if err != nil || len(got) != 2 {
		t.Fatalf("LoadRequired = %v, %v", got, err)
	}
	for name, doc := range map[string]string{
		"bad id":                `{"scenarios":["Bad_ID"]}`,
		"duplicate id":          `{"scenarios":["a","a"]}`,
		"unlisted dependency":   `{"scenarios":["a"],"dependencies":{"a":["missing"]}}`,
		"dependency cycle":      `{"scenarios":["a","b"],"dependencies":{"a":["b"],"b":["a"]}}`,
		"unknown field":         `{"scenarios":["a"],"dependecies":{"a":["a"]}}`,
		"too many dependencies": `{"scenarios":["a","b","c"],"dependencies":{"a":["b","c","b","c","b","c","b","c","b","c","b","c","b","c","b","c","b"]}}`,
		"trailing JSON":         `{"scenarios":["a"]} {}`,
	} {
		t.Run(name, func(t *testing.T) {
			write(doc)
			if _, err := LoadRequiredPlan(dir); err == nil {
				t.Fatalf("invalid plan %q must be rejected", name)
			}
		})
	}
}

func TestAttachIsBoundedAndScrubbed(t *testing.T) {
	dir := t.TempDir()
	r := &recorder{names: map[string]int{}}
	big := strings.Repeat("x", MaxAttachmentBytes+10)
	rel, err := r.attach(dir, "update", "../evil name.log", []byte(big), "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(rel, "..") || !strings.HasPrefix(rel, "logs/") {
		t.Fatalf("attachment path %q escapes the evidence directory", rel)
	}
	data, err := os.ReadFile(filepath.Join(dir, "update", filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(data), "[truncated]\n") {
		t.Fatal("an oversized attachment must be marked truncated")
	}

	home := filepath.Join(dir, "home")
	rel, err = r.attach(dir, "update", "secret.log", []byte("home="+home+" url=https://user:pw@example.test/x\n"), home)
	if err != nil {
		t.Fatal(err)
	}
	data, _ = os.ReadFile(filepath.Join(dir, "update", filepath.FromSlash(rel)))
	if strings.Contains(string(data), "pw@") || strings.Contains(string(data), home) {
		t.Fatalf("attachment leaks a secret or the home path: %q", data)
	}

	r.bytes = MaxShardBytes
	if _, err := r.attach(dir, "update", "more.log", []byte("x"), ""); err == nil {
		t.Fatal("the per-shard evidence budget must be enforced")
	}
}

func TestWriteResultLayout(t *testing.T) {
	dir := t.TempDir()
	if err := WriteResult(dir, Result{Schema: ResultSchema, Shard: ShardRuntime, Status: OutcomePass}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "runtime", "result.json")); err != nil {
		t.Fatal(err)
	}
}

func TestBlockedOutcomeSerializationAndValidation(t *testing.T) {
	blocked := ScenarioResult{
		ID:      "scenario-c",
		Outcome: OutcomeBlocked,
		Blocked: &BlockedInfo{
			Reason: BlockReasonPrerequisiteNotPassed,
			Dependencies: []ScenarioDependency{{
				Kind: DependencyScenario, ID: "scenario-a", Outcome: OutcomeFail,
			}},
		},
	}
	result := Result{
		Schema: ResultSchema, Shard: ShardRuntime, Status: OutcomeFail,
		Required:  []string{"scenario-a", "scenario-c"},
		Scenarios: []ScenarioResult{{ID: "scenario-a", Outcome: OutcomeFail}, blocked},
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Result
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := ValidateResult(decoded); err != nil {
		t.Fatalf("valid BLOCKED result did not validate: %v", err)
	}
	if !strings.Contains(string(encoded), `"reason":"prerequisite_not_passed"`) || !strings.Contains(string(encoded), `"outcome":"FAIL"`) {
		t.Fatalf("serialized BLOCKED evidence is incomplete: %s", encoded)
	}

	cases := map[string]func(*Result){
		"missing dependencies": func(r *Result) { r.Scenarios[1].Blocked = nil },
		"PASS prerequisite": func(r *Result) {
			r.Scenarios[1].Blocked.Dependencies[0].Outcome = OutcomePass
		},
		"unknown reason": func(r *Result) { r.Scenarios[1].Blocked.Reason = "payload-is-not-a-reason" },
		"non-FAIL shard prerequisite": func(r *Result) {
			r.Scenarios[1].Blocked = &BlockedInfo{Reason: BlockReasonShardPrerequisite, Dependencies: []ScenarioDependency{{Kind: DependencyShard, ID: "windows-runner", Outcome: OutcomeMissing}}}
		},
		"metadata on PASS": func(r *Result) {
			r.Scenarios[0].Outcome = OutcomePass
			r.Scenarios[0].Blocked = &BlockedInfo{Reason: BlockReasonPrerequisiteNotPassed, Dependencies: []ScenarioDependency{{Kind: DependencyScenario, ID: "scenario-c", Outcome: OutcomeFail}}}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			var doc Result
			if err := json.Unmarshal(encoded, &doc); err != nil {
				t.Fatal(err)
			}
			mutate(&doc)
			if err := ValidateResult(doc); err == nil {
				t.Fatal("invalid BLOCKED evidence must be rejected")
			}
		})
	}
}

func TestScenarioHarnessProcess(t *testing.T) {
	resultDir := os.Getenv("HACHIDORI_SCENARIO_HARNESS_DIR")
	if resultDir == "" {
		return
	}
	t.Setenv(EnvEnable, "1")
	rec = newRecorder()
	state.plan = RequiredPlan{
		Scenarios:    []string{"candidate-identity", "scenario-a", "scenario-b", "scenario-c"},
		Dependencies: map[string][]string{"scenario-c": {"scenario-a"}},
	}
	rec.record(ScenarioResult{ID: "candidate-identity", Outcome: OutcomePass})

	t.Run("scenario-a", func(t *testing.T) {
		Begin(t, "scenario-a")
		t.Error("injected scenario A failure")
	})
	independentExecuted := false
	t.Run("scenario-b", func(t *testing.T) {
		Begin(t, "scenario-b")
		independentExecuted = true
	})
	dependentExecuted := false
	t.Run("scenario-c", func(t *testing.T) {
		Begin(t, "scenario-c")
		dependentExecuted = true
	})
	if !independentExecuted {
		t.Error("independent scenario B did not execute")
	}
	if dependentExecuted {
		t.Error("dependent scenario C executed after prerequisite A failed")
	}

	now := time.Now()
	result := Assemble(ShardRuntime, Candidate{}, state.plan.Scenarios, rec.results(), nil, nil, now, now)
	if err := WriteResult(resultDir, result); err != nil {
		t.Fatal(err)
	}
}

func TestScenarioRunnerContinuesIndependentWorkAndAggregatesBlockedDependency(t *testing.T) {
	harnessDir := t.TempDir()
	cmd := exec.Command(os.Args[0], "-test.run=^TestScenarioHarnessProcess$", "-test.count=1")
	for _, entry := range os.Environ() {
		if strings.HasPrefix(entry, EnvEnable+"=") || strings.HasPrefix(entry, "HACHIDORI_SCENARIO_HARNESS_DIR=") {
			continue
		}
		cmd.Env = append(cmd.Env, entry)
	}
	cmd.Env = append(cmd.Env, EnvEnable+"=1", "HACHIDORI_SCENARIO_HARNESS_DIR="+harnessDir)
	output, runErr := cmd.CombinedOutput()
	if runErr == nil {
		t.Fatalf("the isolated shard harness must exit non-zero for scenario A's failure")
	}

	data, err := os.ReadFile(filepath.Join(harnessDir, ShardRuntime, "result.json"))
	if err != nil {
		t.Fatalf("scenario harness did not retain its result: %v\n%s", err, output)
	}
	var result Result
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	fixture := newFixture(t)
	fixture.writeRequired(t, ShardRuntime, `{"scenarios":["candidate-identity","scenario-a","scenario-b","scenario-c"],"dependencies":{"scenario-c":["scenario-a"]}}`)
	result.Candidate = CandidateRef{SourceCommit: fixture.cand.SourceCommit, File: fixture.cand.File, SHA256: fixture.cand.SHA256, Size: fixture.cand.Size}
	result.Run.ID = "42"
	if result.Status != OutcomeFail || !hasOutcomeValue(result.Scenarios, "scenario-a", OutcomeFail) || !hasOutcomeValue(result.Scenarios, "scenario-b", OutcomePass) || !hasOutcomeValue(result.Scenarios, "scenario-c", OutcomeBlocked) {
		t.Fatalf("shard result did not preserve FAIL + independent PASS + dependent BLOCKED: %+v", result)
	}
	if err := ValidateResult(result); err != nil {
		t.Fatalf("shard result failed schema validation: %v", err)
	}
	artifactDir := filepath.Join(fixture.evidence, EvidenceDirPrefix+ShardRuntime)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "result.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	cert := Aggregate(fixture.input())
	if cert.Status != OutcomeFail {
		t.Fatalf("A FAIL and C BLOCKED must fail aggregate certification: %+v", cert)
	}
	if cert.Counts.Fail != 1 || cert.Counts.Blocked != 1 || cert.Counts.Pass != 12 || cert.Counts.Skip != 0 || cert.Counts.Missing != 0 {
		t.Fatalf("unexpected aggregate outcome counts: %+v", cert.Counts)
	}
	markdown := cert.Markdown()
	for _, point := range []string{"| 12 | 1 | 1 | 0 | 0 |", "#### runtime", "scenario `scenario-a`: **FAIL**", "scenario `scenario-c`: **BLOCKED**"} {
		if !strings.Contains(markdown, point) {
			t.Errorf("aggregate summary omitted %q:\n%s", point, markdown)
		}
	}
	if strings.Contains(markdown, "scenario `scenario-b`:") {
		t.Errorf("passing independent scenario B must not be reported as a problem:\n%s", markdown)
	}
}

func TestPreflightFailuresProduceExplicitOutcomes(t *testing.T) {
	plan := RequiredPlan{
		Scenarios:    []string{"candidate-identity", "independent"},
		Dependencies: map[string][]string{"independent": {"candidate-identity"}},
	}
	candidateFailure := Assemble(ShardRuntime, Candidate{}, plan.Scenarios, preflightOutcomes(plan, true, false), nil, []string{"candidate identity failed"}, time.Now(), time.Now())
	if candidateFailure.Status != OutcomeFail || !hasOutcomeValue(candidateFailure.Scenarios, "candidate-identity", OutcomeFail) || !hasOutcomeValue(candidateFailure.Scenarios, "independent", OutcomeBlocked) {
		t.Fatalf("candidate failure must be visible and block its dependent: %+v", candidateFailure)
	}
	if err := ValidateResult(candidateFailure); err != nil {
		t.Fatalf("candidate preflight result is invalid: %v", err)
	}

	windowsFailure := Assemble(ShardRuntime, Candidate{}, plan.Scenarios, preflightOutcomes(plan, false, true), nil, []string{"wrong runner"}, time.Now(), time.Now())
	if windowsFailure.Status != OutcomeFail || !hasOutcomeValue(windowsFailure.Scenarios, "candidate-identity", OutcomeBlocked) || !hasOutcomeValue(windowsFailure.Scenarios, "independent", OutcomeBlocked) {
		t.Fatalf("shard-wide runner failure must make all required loss visible: %+v", windowsFailure)
	}
	if err := ValidateResult(windowsFailure); err != nil {
		t.Fatalf("runner preflight result is invalid: %v", err)
	}
}

func hasOutcomeValue(scenarios []ScenarioResult, id, outcome string) bool {
	for _, scenario := range scenarios {
		if scenario.ID == id {
			return scenario.Outcome == outcome
		}
	}
	return false
}

func hasOutcome(scenarios []ScenarioResult, id string) bool {
	for _, scenario := range scenarios {
		if scenario.ID == id {
			return scenario.Outcome != ""
		}
	}
	return false
}

func TestShardsAreTheSixProofBoundaries(t *testing.T) {
	want := map[string]bool{"bootstrap": true, "runtime": true, "recovery": true, "single-instance": true, "update": true, "diagnostics": true}
	if len(Shards) != len(want) {
		t.Fatalf("Shards = %v", Shards)
	}
	for _, s := range Shards {
		if !want[s] {
			t.Errorf("unexpected shard %q", s)
		}
	}
}
