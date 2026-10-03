// Package workflow checks the structural contract of the Windows E2E
// certification workflow without a YAML dependency: its triggers, that the
// candidate is built exactly once, and that every shard consumes and verifies
// that candidate instead of rebuilding it.
package workflow

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

const workflowPath = "../../../.github/workflows/windows-e2e.yml"

func load(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// section returns the lines of the top-level (zero-indent) key.
func section(text, key string) string {
	lines := strings.Split(text, "\n")
	var out []string
	in := false
	for _, l := range lines {
		if !strings.HasPrefix(l, " ") && strings.TrimSpace(l) != "" && !strings.HasPrefix(l, "#") {
			in = strings.HasPrefix(l, key+":")
		}
		if in {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

// job returns the lines of one job under `jobs:` (two-space indent).
func job(text, name string) string {
	lines := strings.Split(section(text, "jobs"), "\n")
	var out []string
	in := false
	for _, l := range lines {
		if strings.HasPrefix(l, "  ") && !strings.HasPrefix(l, "   ") && strings.TrimSpace(l) != "" && !strings.HasPrefix(strings.TrimSpace(l), "#") {
			in = strings.HasPrefix(l, "  "+name+":")
		}
		if in {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

func TestTriggersArePushMainAndDispatchOnly(t *testing.T) {
	on := section(load(t), "on")
	if on == "" {
		t.Fatal("workflow has no `on:` block")
	}
	for _, forbidden := range []string{"pull_request", "pull_request_target", "schedule", "workflow_run", "merge_group"} {
		if strings.Contains(on, forbidden) {
			t.Errorf("full Windows E2E must not trigger on %s", forbidden)
		}
	}
	if !regexp.MustCompile(`(?m)^  push:\s*\n\s+branches: \[main\]`).MatchString(on) {
		t.Error("workflow must trigger on push to main only")
	}
	if !regexp.MustCompile(`(?m)^  workflow_dispatch:`).MatchString(on) {
		t.Error("workflow must offer workflow_dispatch")
	}
}

func TestCandidateIsBuiltExactlyOnce(t *testing.T) {
	text := load(t)
	builds := regexp.MustCompile(`go build[^\n]*\./cmd/hachidori`).FindAllString(text, -1)
	if len(builds) != 1 {
		t.Fatalf("the workflow must build ./cmd/hachidori exactly once, found %d", len(builds))
	}
	if !strings.Contains(job(text, "candidate"), builds[0]) {
		t.Error("the single build must be in the candidate job")
	}
	if got := strings.Count(text, "actions/upload-artifact@"); got < 2 {
		t.Errorf("expected candidate and evidence uploads, found %d upload steps", got)
	}
}

func TestArtifactUploadsAreRerunnable(t *testing.T) {
	text := load(t)
	uploads := strings.Count(text, "uses: actions/upload-artifact@")
	overwrites := strings.Count(text, "overwrite: true")
	if uploads == 0 || overwrites != uploads {
		t.Fatalf("every artifact upload must be replaceable on a workflow rerun: uploads=%d overwrite=true=%d", uploads, overwrites)
	}
}

func TestShardsConsumeAndVerifyTheSharedCandidate(t *testing.T) {
	text := load(t)
	shard := job(text, "shard")
	if shard == "" {
		t.Fatal("no shard job")
	}
	for _, want := range []string{
		"needs: candidate",
		"fail-fast: false",
		"name: windows-candidate",
		"actions/download-artifact@",
		"e2e-candidate verify",
		"HACHIDORI_E2E_CANDIDATE_SHA256: ${{ needs.candidate.outputs.sha256 }}",
		"HACHIDORI_E2E_SOURCE_SHA: ${{ github.sha }}",
		"HACHIDORI_WINDOWS_E2E: \"1\"",
	} {
		if !strings.Contains(shard, want) {
			t.Errorf("shard job lacks %q", want)
		}
	}
	if strings.Contains(shard, "go build") {
		t.Error("a shard must never build the candidate")
	}

	var got []string
	for _, m := range regexp.MustCompile(`(?m)^          - ([a-z-]+)\s*$`).FindAllStringSubmatch(shard, -1) {
		got = append(got, m[1])
	}
	want := append([]string(nil), e2e.Shards...)
	sort.Strings(got)
	sort.Strings(want)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("matrix shards = %v, want the six proof boundaries %v", got, want)
	}
}

func TestShardPackagesExist(t *testing.T) {
	for _, s := range e2e.Shards {
		for _, f := range []string{"e2e_test.go", "required.json"} {
			if _, err := os.Stat("../" + s + "/" + f); err != nil {
				t.Errorf("shard %s: %v", s, err)
			}
		}
	}
}

func TestActionsArePinnedAndPermissionsMinimal(t *testing.T) {
	text := load(t)
	pin := regexp.MustCompile(`uses: [^@\s]+@([0-9a-f]{40})(\s|$)`)
	for _, l := range strings.Split(text, "\n") {
		if strings.Contains(l, "uses:") && !pin.MatchString(l) {
			t.Errorf("action not pinned to a full commit SHA: %s", strings.TrimSpace(l))
		}
	}
	if !regexp.MustCompile(`(?m)^permissions: \{\}\s*$`).MatchString(text) {
		t.Error("top-level permissions must be empty")
	}
	if strings.Contains(text, "contents: write") {
		t.Error("certification jobs must not have write permission")
	}
}
