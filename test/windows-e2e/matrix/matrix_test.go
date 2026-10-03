// Package matrix validates the Windows certification documentation contract:
// stable unique IDs and exactly one classification per assertion.
package matrix

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

const checklist = "../../../docs/windows-certification-checklist.md"

var (
	itemRow      = regexp.MustCompile(`^W(\d{2})$`)
	assertionRow = regexp.MustCompile(`^W(\d{2})\.(\d+)$`)
	rowPrefix    = regexp.MustCompile(`^\| W\d`)
	goTestProof  = regexp.MustCompile(`^go test \./[A-Za-z0-9_./-]+$`)
)

var classes = map[string]bool{"CI_AUTOMATED": true, "PHYSICAL_REQUIRED": true, "OPTIONAL_HARDWARE": true}

type row struct {
	cells []string
	line  int
}

func tableRows(t *testing.T) []row {
	t.Helper()
	data, err := os.ReadFile(checklist)
	if err != nil {
		t.Fatal(err)
	}
	var rows []row
	for i, line := range strings.Split(string(data), "\n") {
		if !rowPrefix.MatchString(line) {
			continue
		}
		cells := strings.Split(strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(line), "|"), "|"), "|")
		for j := range cells {
			cells[j] = strings.TrimSpace(cells[j])
		}
		rows = append(rows, row{cells: cells, line: i + 1})
	}
	return rows
}

func TestIdentifiersAreUniqueAndEveryItemIsClassified(t *testing.T) {
	seen := map[string]int{}
	items := map[int]bool{}
	assertions := map[int]int{}
	for _, r := range tableRows(t) {
		id := r.cells[0]
		if first, dup := seen[id]; dup {
			t.Errorf("line %d: duplicate certification ID %s (first on line %d)", r.line, id, first)
		}
		seen[id] = r.line

		switch {
		case itemRow.MatchString(id):
			n, _ := strconv.Atoi(itemRow.FindStringSubmatch(id)[1])
			items[n] = true
			if len(r.cells) != 5 {
				t.Errorf("line %d: item %s has %d columns, want 5", r.line, id, len(r.cells))
			} else if r.cells[3] != "NOT_CHECKED" && r.cells[3] != "PASS" && r.cells[3] != "FAIL" {
				t.Errorf("line %d: item %s has outcome %q", r.line, id, r.cells[3])
			}
		case assertionRow.MatchString(id):
			n, _ := strconv.Atoi(assertionRow.FindStringSubmatch(id)[1])
			assertions[n]++
			checkAssertion(t, r)
		default:
			t.Errorf("line %d: unrecognised certification ID %q", r.line, id)
		}
	}
	if len(items) == 0 {
		t.Fatal("no checklist items found")
	}
	for n := 1; n <= len(items); n++ {
		if !items[n] {
			t.Errorf("item IDs are not contiguous: W%02d is missing", n)
		}
		if assertions[n] == 0 {
			t.Errorf("item W%02d has no classified assertion", n)
		}
	}
	for n := range assertions {
		if !items[n] {
			t.Errorf("assertions exist for W%02d, which is not a checklist item", n)
		}
	}
}

func checkAssertion(t *testing.T, r row) {
	t.Helper()
	if len(r.cells) != 4 {
		t.Errorf("line %d: assertion %s has %d columns, want 4", r.line, r.cells[0], len(r.cells))
		return
	}
	class, proof := r.cells[2], r.cells[3]
	if !classes[class] {
		t.Errorf("line %d: %s has class %q, want one of CI_AUTOMATED, PHYSICAL_REQUIRED, OPTIONAL_HARDWARE", r.line, r.cells[0], class)
		return
	}
	if class != "CI_AUTOMATED" {
		if proof != "-" {
			t.Errorf("line %d: %s is %s and must not name an automated proof, got %q", r.line, r.cells[0], class, proof)
		}
		return
	}
	if strings.HasPrefix(proof, "shard:") {
		for _, ref := range strings.Split(proof, "+") {
			shard, scenario, ok := strings.Cut(strings.TrimPrefix(ref, "shard:"), "/")
			if !strings.HasPrefix(ref, "shard:") || !ok || scenario == "" {
				t.Errorf("line %d: %s proof %q is not shard:<name>/<scenario>", r.line, r.cells[0], ref)
				continue
			}
			if !requiredScenarios(t, shard)[scenario] {
				t.Errorf("line %d: %s cites %s/%s, which is not in that shard's required.json", r.line, r.cells[0], shard, scenario)
				continue
			}
			cited[shard+"/"+scenario] = true
		}
		return
	}
	if !goTestProof.MatchString(proof) {
		t.Errorf("line %d: CI_AUTOMATED %s needs proof shard:<name>/<scenario> or go test <package>, got %q", r.line, r.cells[0], proof)
	}
}

// cited collects every shard/scenario some assertion names as its proof.
var cited = map[string]bool{}

func requiredScenarios(t *testing.T, shard string) map[string]bool {
	t.Helper()
	known := false
	for _, s := range e2e.Shards {
		known = known || s == shard
	}
	if !known {
		return nil
	}
	ids, err := e2e.LoadRequired("../" + shard)
	if err != nil {
		t.Fatalf("shard %s: %v", shard, err)
	}
	out := map[string]bool{}
	for _, id := range ids {
		out[id] = true
	}
	return out
}

func TestEveryShardProvesAtLeastOneAssertion(t *testing.T) {
	data, err := os.ReadFile(checklist)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range e2e.Shards {
		if !strings.Contains(string(data), "| CI_AUTOMATED | shard:"+s+"/") {
			t.Errorf("shard %s has no classified CI_AUTOMATED assertion", s)
		}
	}
}

func TestEveryRequiredScenarioIsCitedByAnAssertion(t *testing.T) {
	for k := range cited {
		delete(cited, k)
	}
	for _, r := range tableRows(t) {
		if assertionRow.MatchString(r.cells[0]) {
			checkAssertion(t, r)
		}
	}
	for _, shard := range e2e.Shards {
		for id := range requiredScenarios(t, shard) {
			if id != "candidate-identity" && !cited[shard+"/"+id] {
				t.Errorf("scenario %s/%s proves no classified assertion; map it in the checklist or remove it", shard, id)
			}
		}
	}
}
