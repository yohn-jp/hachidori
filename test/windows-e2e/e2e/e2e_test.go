package e2e

import (
	"encoding/json"
	"os"
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
	write(`{"scenarios":["candidate-identity","runtime-ready"]}`)
	got, err := LoadRequired(dir)
	if err != nil || len(got) != 2 {
		t.Fatalf("LoadRequired = %v, %v", got, err)
	}
	write(`{"scenarios":["Bad_ID"]}`)
	if _, err := LoadRequired(dir); err == nil {
		t.Fatal("a non kebab-case id must be rejected")
	}
	write(`{"scenarios":["a","a"]}`)
	if _, err := LoadRequired(dir); err == nil {
		t.Fatal("a duplicate id must be rejected")
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
