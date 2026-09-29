package explore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
)

type stub struct{}

// Decide answers from the state: "a" -> yes@0.95, "b" -> yes@0.6,
// "c" -> no@0.9, "fail" -> error, "partial" -> no result for q "y".
func (stub) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	if r.State == "fail" {
		return api.DecideResponse{}, os.ErrDeadlineExceeded
	}
	conf := map[string]float64{"a": 0.95, "b": 0.6, "c": 0.9, "partial": 0.7}[r.State]
	choice := "yes"
	if r.State == "c" {
		choice = "no"
	}
	var rs []api.Result
	for _, q := range r.Questions {
		if r.State == "partial" && q.ID == "y" {
			continue
		}
		p := map[string]float64{"yes": 1 - conf, "no": 1 - conf}
		p[choice] = conf
		rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: choice, Confidence: conf, Probabilities: p})
	}
	var inf float64 = float64(len(r.State)) * 3
	return api.DecideResponse{Schema: api.SchemaV1, Results: rs, Timing: &api.Timing{InferenceMS: inf}}, nil
}

func q(id string) api.Question {
	return api.Question{ID: id, Type: "choice", Instructions: id + "?", Choices: []string{"yes", "no"}}
}

// report is a real eval report: 7 observations, 2 request errors.
func report(t *testing.T) eval.Report {
	t.Helper()
	cases := []eval.Case{
		{ID: "c1", State: "a", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "no"}},
		{ID: "c2", State: "b", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "no"}},
		{ID: "c3", State: "c", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "no"}},
		{ID: "c4", State: "fail", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "yes"}},
		{ID: "c5", State: "partial", Questions: []api.Question{q("x"), q("y")}, Expected: map[string]string{"x": "yes", "y": "yes"}},
		{ID: "c6", State: "a", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "yes"}},
	}
	r := eval.Run(stub{}, cases, eval.Options{Passes: 1})
	r.DatasetSHA256 = strings.Repeat("ab", 32)
	return r
}

func encode(t *testing.T, r eval.Report) []byte {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(b, '\n')
}

func TestAnalyzeCountsAreDeterministicAndReconcile(t *testing.T) {
	r := report(t)
	before := encode(t, r)
	a, err := Analyze(r, 0.9, All())
	if err != nil {
		t.Fatal(err)
	}
	// c1 x yes/yes ok, c1 y yes/no wrong .95, c2 x yes/no wrong .6, c3 x no/yes wrong .9,
	// c3 y no/no ok, c5 x yes/yes ok, c6 x yes/yes ok.
	if a.Correct != 4 || a.Wrong != 3 || a.RequestErrors != 2 || a.HighConfWrong != 2 || len(a.Items) != 7 {
		t.Fatalf("counts %+v", a)
	}
	if !reflect.DeepEqual(a.ErrorClasses, []ClassCount{{eval.ErrClassMissingResult, 1}, {eval.ErrClassTransport, 1}}) {
		t.Fatalf("classes %+v", a.ErrorClasses)
	}
	for _, row := range a.PerQuestion {
		st := r.PerQuestion[row.ID]
		if row.Stats != st || row.Correct+row.Wrong != st.N {
			t.Errorf("%s: row %+v does not reconcile with %+v", row.ID, row, st)
		}
		if got := float64(row.Correct) / float64(st.N); got != st.Accuracy {
			t.Errorf("%s: accuracy %v != %v", row.ID, got, st.Accuracy)
		}
	}
	if y := a.PerQuestion[1]; y.ID != "y" || y.MissingResults != 1 || y.Wrong != 1 || y.HighConfWrong != 1 {
		t.Errorf("y row %+v", y)
	}
	again, _ := Analyze(r, 0.9, All())
	if !reflect.DeepEqual(a, again) {
		t.Fatal("analysis is not deterministic")
	}
	// The threshold changes the analysis only.
	low, _ := Analyze(r, 0.5, All())
	if low.HighConfWrong != 3 || low.Correct != 4 || low.Wrong != 3 {
		t.Fatalf("threshold 0.5: %+v", low)
	}
	if string(encode(t, r)) != string(before) {
		t.Fatal("analysis modified the report")
	}
}

func TestAnalyzeFiltersAndSorts(t *testing.T) {
	r := report(t)
	ids := func(f Filter) []string {
		a, err := Analyze(r, 0.9, f)
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, it := range a.Items {
			out = append(out, it.CaseID+"/"+it.QuestionID)
		}
		return out
	}
	f := All()
	for name, tc := range map[string]struct {
		mod  func(*Filter)
		want []string
	}{
		"question":   {func(f *Filter) { f.Question = "y" }, []string{"c1/y", "c3/y"}},
		"expected":   {func(f *Filter) { f.Expected = "no" }, []string{"c1/y", "c2/x", "c3/y"}},
		"predicted":  {func(f *Filter) { f.Predicted = "no" }, []string{"c3/x", "c3/y"}},
		"correct":    {func(f *Filter) { f.Outcome = OutcomeCorrect }, []string{"c1/x", "c3/y", "c5/x", "c6/x"}},
		"wrong":      {func(f *Filter) { f.Outcome = OutcomeWrong }, []string{"c1/y", "c2/x", "c3/x"}},
		"hcw":        {func(f *Filter) { f.Outcome = OutcomeHighConfWrong }, []string{"c1/y", "c3/x"}},
		"conf range": {func(f *Filter) { f.MinConf, f.MaxConf = 0.6, 0.7 }, []string{"c2/x", "c5/x"}},
		"conf desc": {func(f *Filter) { f.Sort, f.Desc = SortConfidence, true },
			[]string{"c1/x", "c1/y", "c6/x", "c3/x", "c3/y", "c5/x", "c2/x"}},
		"conf asc": {func(f *Filter) { f.Sort = SortConfidence },
			[]string{"c2/x", "c5/x", "c3/x", "c3/y", "c1/x", "c1/y", "c6/x"}},
		"inference desc": {func(f *Filter) { f.Sort, f.Desc = SortInferenceMS, true },
			[]string{"c5/x", "c1/x", "c1/y", "c2/x", "c3/x", "c3/y", "c6/x"}},
		"report desc": {func(f *Filter) { f.Desc = true },
			[]string{"c6/x", "c5/x", "c3/y", "c3/x", "c2/x", "c1/y", "c1/x"}},
	} {
		g := f
		tc.mod(&g)
		if got := ids(g); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: %v, want %v", name, got, tc.want)
		}
	}
	// Request latency sorting is a total, stable order on the report.
	g := All()
	g.Sort = SortRequestMS
	a, _ := Analyze(r, 0.9, g)
	for i := 1; i < len(a.Items); i++ {
		if a.Items[i-1].RequestMS > a.Items[i].RequestMS {
			t.Fatal("request_ms not ascending")
		}
	}
	// Error class filter.
	g = All()
	g.ErrorClass = eval.ErrClassTransport
	if a, _ := Analyze(r, 0.9, g); len(a.Errors) != 1 || a.Errors[0].CaseID != "c4" || a.RequestErrors != 2 {
		t.Fatalf("error class filter %+v", a.Errors)
	}
	// Missing inference latency sorts last in both directions.
	r.Results[0].InferenceMS = nil
	for _, desc := range []bool{false, true} {
		g := All()
		g.Sort, g.Desc = SortInferenceMS, desc
		a, _ := Analyze(r, 0.9, g)
		if last := a.Items[len(a.Items)-1]; last.Index != 0 {
			t.Errorf("desc=%v: nil inference_ms not last: %+v", desc, last)
		}
	}
}

func TestAnalyzeRejectsBadControls(t *testing.T) {
	r := report(t)
	for _, tc := range []struct {
		th float64
		f  Filter
	}{
		{1.5, All()}, {-0.1, All()},
		{0.9, Filter{MinConf: 0.8, MaxConf: 0.2}},
		{0.9, Filter{MinConf: 0, MaxConf: 2}},
		{0.9, Filter{MaxConf: 1, Outcome: "unsafe"}},
		{0.9, Filter{MaxConf: 1, Sort: "risk"}},
	} {
		if _, err := Analyze(r, tc.th, tc.f); err == nil {
			t.Errorf("accepted threshold %v filter %+v", tc.th, tc.f)
		}
	}
}

func TestDecodeAcceptsCanonicalEvidence(t *testing.T) {
	r := report(t)
	got, err := Decode(encode(t, r))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(encode(t, got), encode(t, r)) {
		t.Fatal("decode changed the report")
	}
	// A report written by eval (the export path) opens from disk, with the
	// file digest as source identity.
	p := filepath.Join(t.TempDir(), "r.json")
	os.WriteFile(p, encode(t, r), 0o644)
	if _, sum, err := Open(p); err != nil || len(sum) != 64 {
		t.Fatalf("open: %v %q", err, sum)
	}
	if _, err := eval.LoadReport(p); err != nil {
		t.Fatal(err)
	}
}

func TestDecodeRejectsIncompatibleAndMalformedEvidence(t *testing.T) {
	good := string(encode(t, report(t)))
	mut := func(f func(*eval.Report)) string {
		r := report(t)
		f(&r)
		return string(encode(t, r))
	}
	for name, tc := range map[string]struct{ doc, want string }{
		"other schema":   {strings.Replace(good, eval.EvidenceSchema, "hachidori.evidence.v2", 1), `evidence schema "hachidori.evidence.v2"`},
		"replay doc":     {`{"schema":"hachidori.replay.v1","results":[]}`, `evidence schema "hachidori.replay.v1"`},
		"not json":       {"not json", "not a Decision Evidence report"},
		"unknown field":  {strings.Replace(good, `"cases":`, `"verdict":"safe","cases":`, 1), `unknown field "verdict"`},
		"trailing":       {good + "{}", "after top-level value"},
		"wrong type":     {strings.Replace(good, `"cases": 6`, `"cases": "6"`, 1), "malformed"},
		"count mismatch": {mut(func(r *eval.Report) { r.Observations++ }), "observations is 8"},
		"rescored":       {mut(func(r *eval.Report) { r.Results[1].Correct = true }), "disagrees"},
		"confidence":     {mut(func(r *eval.Report) { r.Results[0].Confidence = 1.5 }), "outside [0, 1]"},
		"no probs":       {mut(func(r *eval.Report) { r.Results[0].Probabilities = nil }), "probabilities are missing"},
		"no dataset sha": {mut(func(r *eval.Report) { r.DatasetSHA256 = "" }), "dataset_sha256"},
		"per question":   {mut(func(r *eval.Report) { s := r.PerQuestion["x"]; s.N = 1; r.PerQuestion["x"] = s }), "per_question"},
		"error class":    {mut(func(r *eval.Report) { r.Errors[0].Class = "" }), "class is required"},
	} {
		if _, err := Decode([]byte(tc.doc)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", name, err, tc.want)
		}
	}
	if _, _, err := Open(t.TempDir()); err == nil {
		t.Error("opened a directory")
	}
}

func TestExportCarriesSourceIdentityAndSelection(t *testing.T) {
	r := report(t)
	r.Served = &eval.Served{Digest: "d1", Runtime: map[string]any{"model": "org/m@1"}}
	f := All()
	f.Outcome = OutcomeHighConfWrong
	a, _ := Analyze(r, 0.9, f)
	x := NewExport(r, "filesha", a, f)
	if x.Schema != AnalysisSchema || x.Source.Schema != eval.EvidenceSchema || x.Source.ReportSHA256 != "filesha" ||
		x.Source.DatasetSHA256 != r.DatasetSHA256 || x.Source.ServedIdentity != "d1" || x.Source.ServedModel != "org/m@1" ||
		x.Threshold != 0.9 || x.Counts != (Counts{4, 3, 2, 2, 2}) || len(x.Observations) != 2 || len(x.RequestErrors) != 2 {
		t.Fatalf("export %+v", x)
	}
	if !reflect.DeepEqual(x.Observations[0], r.Results[1]) {
		t.Fatal("exported observation differs from the report")
	}
	// An export is never mistaken for evidence.
	b, _ := json.Marshal(x)
	if _, err := Decode(b); err == nil {
		t.Fatal("analysis export decoded as evidence")
	}
}
