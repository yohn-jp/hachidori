package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const routingPolicyJSON = `{"schema":"hachidori.routing-policy.v1","id":"cli","default":{"first":"laya-base",
	"handoff":{"to":"opendecider-nano","when":{"choice_in":["unknown"]}}}}`

func writeTemp(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadRoutingReadsPolicyAndRefusesBadInput(t *testing.T) {
	if p, c, err := loadRouting("", ""); p != nil || c != nil || err != nil {
		t.Fatalf("no policy is no routing: %v %v %v", p, c, err)
	}
	p, c, err := loadRouting(writeTemp(t, "p.json", routingPolicyJSON), "")
	if err != nil || p == nil || p.ID != "cli" || c != nil {
		t.Fatalf("%v %v %v", p, c, err)
	}
	for name, path := range map[string]string{
		"missing":         filepath.Join(t.TempDir(), "none.json"),
		"unknown field":   writeTemp(t, "u.json", strings.Replace(routingPolicyJSON, `"id":"cli"`, `"id":"cli","x":1`, 1)),
		"not a policy":    writeTemp(t, "n.json", `{"schema":"other"}`),
		"model reference": writeTemp(t, "m.json", strings.Replace(routingPolicyJSON, `"laya-base"`, `"org/repo@rev"`, 1)),
	} {
		if p, _, err := loadRouting(path, ""); err == nil || p != nil {
			t.Errorf("%s: accepted (%v)", name, err)
		}
	}
	// Evidence that is not a resident comparison cannot back a policy.
	if _, _, err := loadRouting(writeTemp(t, "p2.json", routingPolicyJSON), writeTemp(t, "e.json", `{"schema":"x"}`)); err == nil {
		t.Error("a non-comparison was accepted as evidence")
	}
}

// Routing is a property of a resident set: a policy without --resident, and
// evidence without a policy, are refused before anything starts.
func TestServeRefusesRoutingFlagsThatCannotApply(t *testing.T) {
	policy := writeTemp(t, "p.json", routingPolicyJSON)
	if err := runHost("serve", []string{"--routing-policy", policy}); err == nil || !strings.Contains(err.Error(), "--resident") {
		t.Fatalf("%v", err)
	}
	if err := runHost("serve", []string{"--routing-evidence", policy}); err == nil || !strings.Contains(err.Error(), "--routing-policy") {
		t.Fatalf("%v", err)
	}
}
