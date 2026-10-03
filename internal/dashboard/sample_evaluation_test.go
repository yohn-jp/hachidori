package dashboard

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

func installInariSample(t *testing.T, e *env) home.EvaluationSample {
	t.Helper()
	h := home.Home{Root: e.home}
	if err := h.EnsureEvaluationResources(); err != nil {
		t.Fatal(err)
	}
	s := h.InariSampleEvaluation()
	cfg := e.d.cfg
	cfg.EvaluationSample = s
	e.d = New(cfg)
	return s
}

func TestInariSampleIsTheDefaultForgeAndExperimentResource(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	s := installInariSample(t, e)

	for path, label := range map[string]string{
		s.Dataset:   "Inari sample · dataset",
		s.Questions: "Inari sample · evaluation questions",
		s.Policy:    "Inari sample · certification policy",
	} {
		body := e.get(t, "/forge").Body.String()
		if !strings.Contains(body, path) || !strings.Contains(body, label) {
			t.Errorf("Forge does not expose default %q at %s", label, path)
		}
	}
	exp := e.get(t, "/experiments").Body.String()
	for _, want := range []string{s.Dataset, s.Questions, "Inari sample · dataset", "Inari sample · evaluation questions"} {
		if !strings.Contains(exp, want) {
			t.Errorf("Experiments lacks %q", want)
		}
	}
}

func TestKnownResourceKeepsDirectPathOverride(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	s := installInariSample(t, e)
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{`name="dataset_path"`, `name="questions_path"`, `name="policy_path"`} {
		if !strings.Contains(body, want) {
			t.Errorf("known resource has no direct path override %s", want)
		}
	}

	custom := hostPath("custom", "eval.jsonl")
	form := url.Values{"dataset": {s.Dataset}, "dataset_path": {custom}}
	req := httptest.NewRequest("POST", "/", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if got := resourceValue(req, "dataset"); got != custom {
		t.Fatalf("direct path override = %q, want %q", got, custom)
	}
}
