package dashboard

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"html"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/eval/explore"
)

// evidenceFile runs a real experiment against the fake API (answers are
// the first choice at 0.8: c1 correct, c2 wrong, c3 correct), exports the
// canonical report and returns its path.
func evidenceFile(t *testing.T, e *env) string {
	t.Helper()
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	if x := waitExp(t, e); x.State != ExpSucceeded {
		t.Fatalf("experiment %+v", x)
	}
	out := filepath.Join(t.TempDir(), "report.json")
	if body := e.post(t, "/experiments/export", url.Values{"seq": {"1"}, "export_path": {out}}).Body.String(); !strings.Contains(body, "wrote") {
		t.Fatal("export failed")
	}
	return out
}

func TestErrorsOpenReportWithoutInference(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	path := evidenceFile(t, e)
	calls := decideCalls(f)
	src, _ := os.ReadFile(path)

	e2, f2 := newWorkbenchEnv(t) // a fresh dashboard: no experiment, file only
	rec := e2.post(t, "/errors/open", url.Values{"path": {path}})
	if rec.Code != http.StatusSeeOther {
		t.Fatalf("open: %d %s", rec.Code, rec.Body)
	}
	body := html.UnescapeString(e2.get(t, "/errors").Body.String())
	sum := sha256.Sum256(src)
	for _, s := range []string{"file " + path, hex.EncodeToString(sum[:]), "org/laya@rev1",
		`<dt>Correct</dt><dd>2</dd>`, `<dt>Wrong</dt><dd>1</dd>`, `<dt>Request errors</dt><dd>0</dd>`, `<dt>High-confidence wrong</dt><dd class="">0</dd>`} {
		if !strings.Contains(body, s) {
			t.Errorf("page lacks %q", s)
		}
	}
	if decideCalls(f2) != 0 || decideCalls(f) != calls {
		t.Fatal("opening evidence reached an endpoint")
	}
	if got, _ := os.ReadFile(path); !bytes.Equal(got, src) {
		t.Fatal("source report changed")
	}
}

func TestErrorsRejectsIncompatibleEvidence(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	good := evidenceFile(t, e)
	src, _ := os.ReadFile(good)
	dir := t.TempDir()
	cases := map[string]string{
		"relative":     "report.json",
		"missing":      filepath.Join(dir, "none.json"),
		"directory":    dir,
		"other schema": writeDef(t, dir, "v2.json", strings.Replace(string(src), eval.EvidenceSchema, "hachidori.evidence.v2", 1)),
		"replay":       writeDef(t, dir, "replay.json", `{"schema":"hachidori.replay.v1","results":[]}`),
		"malformed":    writeDef(t, dir, "bad.json", strings.Replace(string(src), `"observations": 3`, `"observations": 4`, 1)),
		"unknown":      writeDef(t, dir, "unk.json", strings.Replace(string(src), `"cases": 3`, `"verdict":"safe","cases": 3`, 1)),
	}
	e.post(t, "/errors/open", url.Values{"path": {good}})
	for name, p := range cases {
		rec := e.post(t, "/errors/open", url.Values{"path": {p}})
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "cannot open report") {
			t.Errorf("%s: %d accepted", name, rec.Code)
		}
		if src := e.d.errs.get(); src == nil || src.Origin != "file "+good {
			t.Errorf("%s: a rejected file replaced the open report", name)
		}
	}
}

func TestErrorsFiltersThresholdDetailAndExport(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	path := evidenceFile(t, e)
	src, _ := os.ReadFile(path)
	// From the experiment directly (no file).
	if rec := e.post(t, "/errors/use-experiment", url.Values{"seq": {"1"}}); rec.Code != http.StatusSeeOther {
		t.Fatalf("use experiment: %d %s", rec.Code, rec.Body)
	}
	s := e.d.errs.get()
	if s.Origin != "experiment #1" || s.SHA256 == "" {
		t.Fatalf("source %+v", s)
	}
	canon := sha256.Sum256(src)
	if s.SHA256 != hex.EncodeToString(canon[:]) {
		t.Error("experiment source digest differs from the exported file's")
	}
	before, _ := json.Marshal(s.Report)

	// Threshold 0.8 marks the wrong 0.8 observation high-confidence.
	body := e.get(t, "/errors?th=0.8&outcome=high_confidence_wrong").Body.String()
	if !strings.Contains(body, `<dt>High-confidence wrong</dt><dd class="bad-t">1</dd>`) || !strings.Contains(body, "1 selected") ||
		!strings.Contains(body, `<td class="mono">c2</td>`) || strings.Contains(body, `<td class="mono">c1</td>`) {
		t.Fatalf("threshold/filter not applied")
	}
	body = e.get(t, "/errors?th=0.9&outcome=high_confidence_wrong").Body.String()
	if !strings.Contains(body, `<dt>High-confidence wrong</dt><dd class="">0</dd>`) || !strings.Contains(body, "0 selected") {
		t.Fatal("threshold 0.9")
	}
	if b, _ := json.Marshal(e.d.errs.get().Report); !bytes.Equal(b, before) {
		t.Fatal("analysis changed the report")
	}
	for q, want := range map[string]string{"q=scope": "1 selected", "exp=no": "1 selected", "pred=yes": "3 selected",
		"outcome=correct": "2 selected", "min=0.9&max=1": "0 selected", "sort=confidence&desc=1": "3 selected"} {
		if !strings.Contains(e.get(t, "/errors?"+q).Body.String(), want) {
			t.Errorf("%s: want %q", q, want)
		}
	}
	for _, q := range []string{"th=2", "th=x", "min=0.9&max=0.1", "outcome=safe", "sort=risk"} {
		if b := e.get(t, "/errors?"+q).Body.String(); !strings.Contains(b, `class="msg bad"`) {
			t.Errorf("%s accepted", q)
		}
	}
	// Detail shows the evidence needed to diagnose the decision.
	var idx int
	for i, o := range s.Report.Results {
		if o.CaseID == "c2" {
			idx = i
		}
	}
	o := s.Report.Results[idx]
	det := html.UnescapeString(e.get(t, "/errors?obs="+itoa(idx)).Body.String())
	for _, want := range []string{"Observation " + itoa(idx), o.QuestionSHA256, "<dt>expected</dt><dd>no</dd>", "<dt>predicted</dt><dd>yes</dd>",
		"0.8000", s.Report.Served.Digest, "(expected)"} {
		if !strings.Contains(det, want) {
			t.Errorf("detail lacks %q", want)
		}
	}

	// Export of the filtered selection.
	out := filepath.Join(t.TempDir(), "wrong.json")
	form := url.Values{"th": {"0.8"}, "outcome": {"wrong"}, "min": {"0"}, "max": {"1"}, "export_path": {out}}
	if b := e.post(t, "/errors/export", form).Body.String(); !strings.Contains(b, "wrote 1 observations as "+explore.AnalysisSchema) {
		t.Fatalf("export: %s", b)
	}
	var x explore.Export
	data, _ := os.ReadFile(out)
	if err := json.Unmarshal(data, &x); err != nil || x.Schema != explore.AnalysisSchema || x.Source.ReportSHA256 != s.SHA256 ||
		x.Threshold != 0.8 || x.Filter.Outcome != "wrong" || len(x.Observations) != 1 || x.Observations[0].CaseID != "c2" ||
		x.Counts.HighConfWrong != 1 || x.Counts.Selected != 1 {
		t.Fatalf("export %v %+v", err, x)
	}
	// Never over the source or any existing file.
	form.Set("export_path", path)
	if b := e.post(t, "/errors/export", form).Body.String(); !strings.Contains(b, "already exists") {
		t.Error("export replaced the source report")
	}
	if got, _ := os.ReadFile(path); !bytes.Equal(got, src) {
		t.Fatal("source report changed")
	}
	if _, err := eval.LoadReport(path); err != nil {
		t.Fatal(err)
	}
}

func TestErrorsRoutesKeepDashboardBoundaries(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	for _, p := range []string{"/errors/open", "/errors/use-experiment", "/errors/export"} {
		if rec := e.post(t, p, url.Values{"token": {"stale"}}); rec.Code != http.StatusForbidden {
			t.Errorf("%s: %d", p, rec.Code)
		}
	}
	if b := e.post(t, "/errors/use-experiment", url.Values{"seq": {"1"}}).Body.String(); !strings.Contains(b, "no finished experiment report") {
		t.Error("explored a missing experiment")
	}
	if b := e.post(t, "/errors/export", url.Values{"export_path": {filepath.Join(t.TempDir(), "x.json")}}).Body.String(); !strings.Contains(b, "no report is open") {
		t.Error("exported without a report")
	}
	if root := e.get(t, "/").Body.String(); !strings.Contains(root, `<a href="/errors">Evidence</a>`) || strings.Contains(root, ">Errors</a>") {
		t.Error("navigation lacks Evidence")
	}
	// Rendered evidence values are escaped.
	r := eval.Report{Schema: eval.EvidenceSchema, DatasetSHA256: "x", Dataset: xss, Observations: 1, Errors: []eval.RequestError{},
		PerQuestion: map[string]eval.QuestionStats{xss: {N: 1}},
		Results: []eval.Observation{{CaseID: xss, QuestionID: xss, Expected: xss, Choice: xss, Confidence: 1, Correct: true,
			Probabilities: map[string]float64{xss: 1}}}}
	b, _ := json.Marshal(r)
	p := writeDef(t, t.TempDir(), "x.json", string(b))
	e.post(t, "/errors/open", url.Values{"path": {p}})
	if body := e.get(t, "/errors?obs=0").Body.String(); strings.Contains(body, xss) || !strings.Contains(body, "Observation 0") {
		t.Fatal("unescaped evidence value")
	}
}
