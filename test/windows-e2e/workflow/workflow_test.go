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
	for _, name := range []string{"candidate", "shard", "aggregate"} {
		if strings.Contains(job(text, name), "contents: write") {
			t.Errorf("job %s must not have write permission", name)
		}
	}
	if got := strings.Count(text, "contents: write"); got != 1 || !strings.Contains(job(text, "release"), "contents: write") {
		t.Errorf("only the release job may write contents, found %d grants", got)
	}
}

func TestNoJobMasksFailure(t *testing.T) {
	if strings.Contains(load(t), "continue-on-error") {
		t.Error("no step or job may mask a failure with continue-on-error")
	}
}

func TestAggregateFailsClosedOverEveryShard(t *testing.T) {
	text := load(t)
	agg := job(text, "aggregate")
	for _, want := range []string{
		"needs: [candidate, shard]",
		"if: ${{ always() }}",
		"pattern: windows-e2e-evidence-*",
		"e2e-aggregate aggregate",
		"candidate=${{ needs.candidate.result }},shard=${{ needs.shard.result }}",
		"CANDIDATE_SHA256: ${{ needs.candidate.outputs.sha256 }}",
		"name: windows-e2e-certification",
	} {
		if !strings.Contains(agg, want) {
			t.Errorf("aggregate job lacks %q", want)
		}
	}
	// The certification is uploaded also when the run failed.
	if !regexp.MustCompile(`(?s)Upload certification\s+if: \$\{\{ always\(\) \}\}`).MatchString(agg) {
		t.Error("certification evidence must be uploaded for failed runs too")
	}
}

func TestReleaseCannotPrecedeCertification(t *testing.T) {
	text := load(t)
	rel := job(text, "release")
	if rel == "" {
		t.Fatal("no release job")
	}
	for _, want := range []string{
		"needs: [candidate, aggregate]",
		"success() && github.event_name == 'push' && github.ref == 'refs/heads/main'",
		"name: windows-candidate",
		"name: windows-e2e-certification",
		"e2e-aggregate release-check",
		"e2e-candidate verify",
		"CANDIDATE_SHA256: ${{ needs.candidate.outputs.sha256 }}",
		"gh release create",
		"certified candidate",
	} {
		if !strings.Contains(rel, want) {
			t.Errorf("release job lacks %q", want)
		}
	}
	if strings.Contains(rel, "go build") {
		t.Error("the release must publish the certified candidate, never a rebuild")
	}
	if strings.Contains(rel, "always()") || strings.Contains(rel, "failure()") || strings.Contains(rel, "cancelled()") {
		t.Error("the release gate must not override the implicit success of its needs")
	}
	// gate order inside the job: verify, stage, publish, verify published.
	order := []string{"release-check", "Stage the certified bytes", "gh release create", "Verify the published asset"}
	last := -1
	for _, m := range order {
		i := strings.Index(rel, m)
		if i < 0 || i < last {
			t.Fatalf("release steps out of order around %q", m)
		}
		last = i
	}
	// Nothing but release may publish.
	for _, name := range []string{"candidate", "shard", "aggregate"} {
		if strings.Contains(job(text, name), "gh release") {
			t.Errorf("job %s must not publish a release", name)
		}
	}
}

func TestReleaseWorkflowNoLongerPublishesOnPush(t *testing.T) {
	data, err := os.ReadFile("../../../.github/workflows/release.yml")
	if err != nil {
		t.Fatal(err)
	}
	on := section(string(data), "on")
	if strings.Contains(on, "push") || strings.Contains(on, "workflow_dispatch") || !strings.Contains(on, "pull_request") {
		t.Errorf("release.yml must be pull_request only so publication happens only after certification; on: %s", on)
	}
}
