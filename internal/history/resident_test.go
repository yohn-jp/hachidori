package history

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
)

// pair is a two-resident endpoint: decide answers as the named model and
// status lists both residents. It has no lifecycle surface.
type pair struct{}

func (pair) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	var rs []api.Result
	for _, q := range r.Questions {
		rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: "yes", Confidence: 0.8,
			Probabilities: map[string]float64{"yes": 0.8, "no": 0.2}})
	}
	return api.DecideResponse{Schema: api.SchemaV1, Results: rs, Timing: &api.Timing{InferenceMS: 3},
		Served: &api.Served{Model: r.Model, Provider: "p-" + r.Model}}, nil
}

func (pair) Status() (json.RawMessage, error) {
	res := func(id string) map[string]any {
		return map[string]any{"model": id, "provider": "p-" + id, "running": true, "status": map[string]any{
			"schema": api.SchemaV1, "uptime_s": 5, "runtime": map[string]any{"model_id": id, "model": "org/" + id + "@r"},
			"worker": map[string]any{"ready": true, "pid": 7, "starts": 1, "provider": map[string]any{"provider": "p-" + id, "load_ms": 10.0}}}}
	}
	return json.Marshal(map[string]any{"schema": api.SchemaV1, "residents": []any{res("m-a"), res("m-b")}})
}

// The evidence of each model of a resident comparison is ordinary v1 Decision
// Evidence: history stores it byte for byte, strict decoding accepts it, and
// the existing comparison aligns the two like for like. The comparison
// document itself is a separate schema that history refuses as evidence.
func TestResidentComparisonEvidenceStaysCompatibleWithHistory(t *testing.T) {
	q := api.Question{ID: "x", Type: "choice", Instructions: "x?", Choices: []string{"yes", "no"}}
	cases := []eval.Case{
		{ID: "c1", State: "a", Questions: []api.Question{q}, Expected: map[string]string{"x": "yes"}},
		{ID: "c2", State: "b", Questions: []api.Question{q}, Expected: map[string]string{"x": "no"}},
	}
	cmp, err := eval.RunResidents(pair{}, cases, strings.Repeat("ab", 32), eval.ResidentOptions{Options: eval.Options{Passes: 2}, Models: []string{"m-a", "m-b"}})
	if err != nil {
		t.Fatal(err)
	}
	s, _ := newStore(t)
	var ids []string
	for _, run := range cmp.Runs {
		ev := run.Evidence("http://127.0.0.1:1", "d.jsonl", cmp.Definitions)
		b, err := json.MarshalIndent(ev, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		sm, err := s.Save(append(b, '\n'), run.Model, "")
		if err != nil {
			t.Fatalf("%s: %v", run.Model, err)
		}
		if sm.ServedModel != "org/"+run.Model+"@r" || sm.Observations != 2 || !sm.ServedConsistent {
			t.Fatalf("%+v", sm)
		}
		ids = append(ids, sm.ID)
	}
	a, _, err := s.Open(ids[0])
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := s.Open(ids[1])
	if err != nil {
		t.Fatal(err)
	}
	if c := eval.Compare(a, b); c.Status != eval.CompareCompatible || len(c.Incompatibility) != 0 || len(c.Questions) != 1 {
		t.Fatalf("%+v", c)
	}
	doc, _ := json.Marshal(cmp)
	if _, err := s.Save(doc, "cmp", ""); err == nil {
		t.Fatal("a comparison document must not be accepted as v1 evidence")
	}
}
