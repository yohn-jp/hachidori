package home_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
)

func TestInariSampleEvaluationProvisioningAndParsing(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	s := h.InariSampleEvaluation()
	root := h.Path("resources", "evaluation", "inari")
	if filepath.Dir(s.Dataset) != root || filepath.Dir(s.Policy) != root || filepath.Dir(s.Questions) != root {
		t.Fatalf("sample is not install-root relative: %+v", s)
	}

	defs, err := question.Load(s.Questions)
	if err != nil {
		t.Fatal(err)
	}
	if defs.Len() != 5 {
		t.Fatalf("question definitions = %d, want 5", defs.Len())
	}
	cases, _, err := eval.Load(s.Dataset, defs)
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) != 12 || len(cases[0].Questions) != 5 {
		t.Fatalf("sample shape = %d cases x %d questions, want 12 x 5", len(cases), len(cases[0].Questions))
	}
	policy, err := eval.LoadPolicy(s.Policy)
	if err != nil {
		t.Fatal(err)
	}
	if policy.ID != "inari-sample-fidelity/1" || policy.MinObservations > len(cases)*len(cases[0].Questions) {
		t.Fatalf("sample policy %+v does not fit the sample observations", policy)
	}

	custom := h.Path("resources", "evaluation", "custom", "user.json")
	if err := os.MkdirAll(filepath.Dir(custom), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(custom, []byte("user-owned\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(s.Dataset, []byte("corrupt\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := h.EnsureEvaluationResources(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := eval.Load(s.Dataset, defs); err != nil {
		t.Fatalf("owned sample was not refreshed: %v", err)
	}
	if got, err := os.ReadFile(custom); err != nil || string(got) != "user-owned\n" {
		t.Fatalf("custom resource changed: %q %v", got, err)
	}
}
