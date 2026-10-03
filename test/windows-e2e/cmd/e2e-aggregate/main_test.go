package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

const commit = "0123456789abcdef0123456789abcdef01234567"

// setup builds a complete, passing certification input in a temp dir.
func setup(t *testing.T) (root, sha string) {
	t.Helper()
	root = t.TempDir()
	cand := filepath.Join(root, "candidate")
	if err := os.MkdirAll(cand, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cand, "hachidori.exe"), []byte("exe-bytes"), 0o755); err != nil {
		t.Fatal(err)
	}
	m, err := e2e.WriteCandidate(cand, "hachidori.exe", commit, "windows", "amd64", "go")
	if err != nil {
		t.Fatal(err)
	}
	for _, shard := range e2e.Shards {
		dir := filepath.Join(root, "tests", shard)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "required.json"), []byte(`{"scenarios":["candidate-identity"]}`), 0o644); err != nil {
			t.Fatal(err)
		}
		writeResult(t, root, shard, m, e2e.OutcomePass)
	}
	return root, m.SHA256
}

func writeResult(t *testing.T, root, shard string, m e2e.Manifest, outcome string) {
	t.Helper()
	res := e2e.Result{
		Schema: e2e.ResultSchema, Shard: shard, Status: outcome, EvidenceClass: e2e.EvidenceClass,
		Candidate: e2e.CandidateRef{SourceCommit: m.SourceCommit, File: m.File, SHA256: m.SHA256, Size: m.Size},
		Required:  []string{"candidate-identity"},
		Scenarios: []e2e.ScenarioResult{{ID: "candidate-identity", Outcome: outcome}},
	}
	dir := filepath.Join(root, "evidence", e2e.EvidenceDirPrefix+shard)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(res)
	if err := os.WriteFile(filepath.Join(dir, "result.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func args(root, sha string) []string {
	return []string{
		"-evidence", filepath.Join(root, "evidence"), "-tests", filepath.Join(root, "tests"),
		"-candidate", filepath.Join(root, "candidate"), "-commit", commit, "-sha256", sha,
		"-jobs", "candidate=success,shard=success", "-run-id", "", "-out", filepath.Join(root, "out"),
	}
}

func TestAggregateThenReleaseCheck(t *testing.T) {
	root, sha := setup(t)
	if code := aggregate(args(root, sha)); code != 0 {
		t.Fatalf("complete evidence: exit %d", code)
	}
	cert := filepath.Join(root, "out", "certification.json")
	for _, f := range []string{cert, filepath.Join(root, "out", "certification.md")} {
		if _, err := os.Stat(f); err != nil {
			t.Fatal(err)
		}
	}
	if code := releaseCheck([]string{"-certification", cert, "-commit", commit, "-sha256", sha, "-file", "hachidori.exe"}); code != 0 {
		t.Fatalf("release gate must open for the certified candidate, exit %d", code)
	}
	if code := releaseCheck([]string{"-certification", cert, "-commit", commit, "-sha256", "00" + sha[2:]}); code == 0 {
		t.Fatal("release gate must stay closed for another executable")
	}
}

func TestFailedShardWritesFailedCertificationAndClosesGate(t *testing.T) {
	root, sha := setup(t)
	data, _ := os.ReadFile(filepath.Join(root, "candidate", e2e.ManifestName))
	var m e2e.Manifest
	_ = json.Unmarshal(data, &m)
	writeResult(t, root, e2e.ShardUpdate, m, e2e.OutcomeFail)

	if code := aggregate(args(root, sha)); code == 0 {
		t.Fatal("a failed shard must fail the aggregate")
	}
	cert := filepath.Join(root, "out", "certification.json")
	if _, err := os.Stat(cert); err != nil {
		t.Fatalf("a failed run must still retain certification evidence: %v", err)
	}
	if code := releaseCheck([]string{"-certification", cert, "-commit", commit, "-sha256", sha}); code == 0 {
		t.Fatal("a FAIL certification must keep the release gate closed")
	}
}

func TestMissingShardOrIdentityFailsClosed(t *testing.T) {
	root, sha := setup(t)
	if err := os.RemoveAll(filepath.Join(root, "evidence", e2e.EvidenceDirPrefix+e2e.ShardRecovery)); err != nil {
		t.Fatal(err)
	}
	if code := aggregate(args(root, sha)); code == 0 {
		t.Fatal("a missing shard result must fail the aggregate")
	}

	root, _ = setup(t)
	a := args(root, "")
	if code := aggregate(a); code == 0 {
		t.Fatal("a missing expected SHA-256 must fail the aggregate")
	}
	if _, err := os.Stat(filepath.Join(root, "out", "certification.json")); err != nil {
		t.Fatalf("the failure must still be recorded: %v", err)
	}
}
