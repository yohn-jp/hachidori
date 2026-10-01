package route

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
)

// evidenceEndpoint is a fake endpoint with both residents loaded, enough for
// eval.RunResidents to produce real comparison evidence without any model.
type evidenceEndpoint struct{}

func (evidenceEndpoint) Decide(req api.DecideRequest) (api.DecideResponse, error) {
	served := api.Served{Model: api.ModelRef(req.Model), Provider: map[string]string{laya: "laya", nano: "opendecider"}[api.ModelRef(req.Model)]}
	var rs []api.Result
	for _, q := range req.Questions {
		rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.9,
			Probabilities: map[string]float64{q.Choices[0]: 0.9, q.Choices[1]: 0.1}})
	}
	return api.DecideResponse{Schema: api.SchemaV1, Results: rs, Timing: &api.Timing{InferenceMS: 2}, Served: &served}, nil
}

func (evidenceEndpoint) Status() (json.RawMessage, error) {
	st := func(id, provider string, pid int) map[string]any {
		return map[string]any{
			"schema":   api.SchemaV1,
			"runtime":  map[string]any{"model_id": id, "model": "org/" + id + "/rev1", "device": "cuda", "runtime": "rt1", "home": "h"},
			"uptime_s": 10,
			"worker": map[string]any{"ready": true, "pid": pid, "starts": 1,
				"provider":    map[string]any{"provider": provider, "model_revision": "rev1", "load_ms": 1000.0, "warmup_ms": 100.0, "device": "cuda"},
				"accelerator": map[string]any{"memory_allocated": 1.0, "memory_reserved": 2.0, "memory_free": 3.0, "memory_total": 4.0}},
		}
	}
	doc := st(laya, "laya", 100)
	var rs []map[string]any
	for i, r := range []struct {
		id, p string
		pid   int
	}{{laya, "laya", 100}, {nano, "opendecider", 200}} {
		rs = append(rs, map[string]any{"model": r.id, "provider": r.p, "default": i == 0, "running": true, "status": st(r.id, r.p, r.pid)})
	}
	doc["residents"] = rs
	return json.Marshal(doc)
}

func comparisonEvidence(t *testing.T, families map[string]string) []byte {
	t.Helper()
	q := func(id string) api.Question {
		return api.Question{ID: id, Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}
	}
	cases := []eval.Case{
		{ID: "c1", State: "one", Questions: []api.Question{q("r1"), q("s1")}, Expected: map[string]string{"r1": "yes", "s1": "no"}},
		{ID: "c2", State: "two", Questions: []api.Question{q("r1")}, Expected: map[string]string{"r1": "yes"}},
	}
	rep, err := eval.RunResidents(evidenceEndpoint{}, cases, "dataset-digest", eval.ResidentOptions{
		Options: eval.Options{Passes: 1}, Models: []string{laya, nano}, Families: families})
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func thresholdPolicy() Policy {
	return Policy{Schema: PolicySchema, ID: "cal", Families: map[string]string{"r1": "readiness"},
		Rules:   []Rule{{Family: "readiness", First: laya, Handoff: &Handoff{To: nano, When: When{ConfidenceBelow: ptr(0.8)}}}},
		Default: Rule{First: laya}}
}

func TestVerifyBacksThresholdsWithResidentComparisonEvidence(t *testing.T) {
	ev := comparisonEvidence(t, map[string]string{"r1": "readiness"})
	c, err := Verify(thresholdPolicy(), ev)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(c.EvidenceSHA256, "sha256:") || c.DatasetSHA256 != "dataset-digest" ||
		len(c.Rules) != 1 || c.Rules[0] != (CalibratedRule{Family: "readiness", Model: laya, Observations: 2}) {
		t.Fatalf("%+v", c)
	}
	// The evidence digest is of the exact document.
	if c2, _ := Verify(thresholdPolicy(), append(append([]byte{}, ev...), '\n')); c2.EvidenceSHA256 == c.EvidenceSHA256 {
		t.Fatal("digest does not cover the document")
	}
	// It is surfaced in status.
	r := newRouter(t, thresholdPolicy(), &fakeBackend{})
	r.SetCalibration(c)
	if got := r.Status().Calibration; got == nil || got.EvidenceSHA256 != c.EvidenceSHA256 {
		t.Fatalf("%+v", got)
	}
}

func TestVerifyRefusesThresholdsWithoutEvidenceBehindThem(t *testing.T) {
	ev := comparisonEvidence(t, map[string]string{"r1": "readiness"})

	// A rule whose family the evidence never measured.
	p := thresholdPolicy()
	p.Families = map[string]string{"r1": "other"}
	p.Rules[0].Family = "other"
	if _, err := Verify(p, ev); err == nil || !strings.Contains(err.Error(), "family other") {
		t.Fatalf("%v", err)
	}
	// A rule whose first-path model the evidence does not contain.
	p = thresholdPolicy()
	p.Rules[0].First, p.Rules[0].Handoff.To = "openjev", nano
	if _, err := Verify(p, ev); err == nil || !strings.Contains(err.Error(), "openjev") {
		t.Fatalf("%v", err)
	}
	// Not a comparison at all, and a comparison that is not aligned.
	if _, err := Verify(thresholdPolicy(), []byte(`{"schema":"x"}`)); err == nil {
		t.Fatal("accepted a non-comparison")
	}
	// A policy with no confidence or margin rule needs no calibration.
	p = Policy{Schema: PolicySchema, ID: "n", Default: Rule{First: laya, Handoff: &Handoff{To: nano, When: When{ChoiceIn: []string{"no"}}}}}
	if c, err := Verify(p, ev); err != nil || len(c.Rules) != 0 {
		t.Fatalf("%v %+v", err, c)
	}
}
