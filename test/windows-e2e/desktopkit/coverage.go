package desktopkit

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

var beginCall = regexp.MustCompile(`e2e\.Begin\(\s*t\s*,\s*"([a-z0-9-]+)"\s*\)`)

// BeginIDs returns the scenario IDs the shard's test files begin (e2e.Begin),
// plus candidate-identity when a file runs e2e.VerifyCandidateScenario.
func BeginIDs(dir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*_test.go"))
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		src := string(b)
		for _, m := range beginCall.FindAllStringSubmatch(src, -1) {
			seen[m[1]] = true
		}
		if strings.Contains(src, "e2e.VerifyCandidateScenario(t)") {
			seen["candidate-identity"] = true
		}
	}
	var ids []string
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids, nil
}

// CoverageProblem describes a drift between the shard's required.json and the
// scenarios its tests begin; it is empty when they are in step.
func CoverageProblem(dir string) (string, error) {
	required, err := e2e.LoadRequired(dir)
	if err != nil {
		return "", err
	}
	begun, err := BeginIDs(dir)
	if err != nil {
		return "", err
	}
	sort.Strings(required)
	if strings.Join(required, ",") != strings.Join(begun, ",") {
		return fmt.Sprintf("required.json lists %v but the tests begin %v", required, begun), nil
	}
	return "", nil
}

// CheckScenarioCoverage fails unless the scenarios the shard's tests begin are
// exactly the scenarios its required.json lists. A scenario that exists but is
// not required would never gate the shard; a required one that no test begins
// could never pass. It runs in portable mode, so a drift between a shard's
// tests and its contract is caught on every pull request, not only on main.
func CheckScenarioCoverage(t *testing.T, dir string) {
	t.Helper()
	problem, err := CoverageProblem(dir)
	if err != nil {
		t.Fatal(err)
	}
	if problem != "" {
		t.Error(problem)
	}
}
