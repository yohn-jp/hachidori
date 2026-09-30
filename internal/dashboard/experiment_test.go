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
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/question"
)

const datasetJSONL = `{"id":"c1","state":"a","questions":[{"id":"x","type":"choice","instructions":"X?","choices":["yes","no"]}],"expected":{"x":"yes"}}
{"id":"c2","state":"b","questions":[{"id":"x","type":"choice","instructions":"X?","choices":["yes","no"]}],"expected":{"x":"no"}}
{"id":"c3","state":"c","question_refs":[{"id":"scope","version":3}],"expected":{"scope":"yes"}}
`

// expFiles writes the dataset and a definitions directory.
func expFiles(t *testing.T) (dataset, defs string) {
	t.Helper()
	dir := t.TempDir()
	defs = filepath.Join(dir, "defs")
	os.Mkdir(defs, 0o755)
	writeDef(t, defs, "scope.json", defJSON)
	dataset = writeDef(t, dir, "dataset.jsonl", datasetJSONL)
	return dataset, defs
}

func runForm(dataset, defs string) url.Values {
	return url.Values{"dataset": {dataset}, "definitions": {defs}, "warmup": {"1"}, "passes": {"2"}}
}

func waitExp(t *testing.T, e *env) *Experiment {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if x := e.d.exp.snapshot(); x != nil && x.State != ExpRunning {
			return x
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("experiment did not finish")
	return nil
}

func decideCalls(f *fakeAPI) int {
	paths, _ := f.calls()
	return len(paths)
}

func TestExperimentPreflightFailsBeforeInference(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	dataset, defs := expFiles(t)
	badLabel := writeDef(t, t.TempDir(), "bad.jsonl",
		`{"id":"c1","state":"a","questions":[{"id":"x","type":"choice","instructions":"X?","choices":["yes","no"]}],"expected":{"x":"maybe"}}`+"\n")
	for name, tc := range map[string]struct {
		form url.Values
		want string
	}{
		"relative dataset":   {runForm("dataset.jsonl", defs), "dataset path must be absolute"},
		"relative defs":      {runForm(dataset, "defs"), "Question Definition path must be absolute"},
		"missing dataset":    {runForm(filepath.Join(filepath.Dir(dataset), "none.jsonl"), defs), "none.jsonl"},
		"unresolved refs":    {runForm(dataset, ""), "unresolved question definition scope@3"},
		"bad expected label": {runForm(badLabel, ""), `expected "maybe" is not a choice`},
		"passes":             {with(runForm(dataset, defs), "passes", "0"), "passes must be 1..100"},
		"warmup":             {with(runForm(dataset, defs), "warmup", "x"), "warmup: not a number"},
	} {
		for _, route := range []string{"/experiments/preflight", "/experiments/run"} {
			rec := e.post(t, route, tc.form)
			body := html.UnescapeString(rec.Body.String())
			if rec.Code != http.StatusOK || !strings.Contains(body, "preflight failed") || !strings.Contains(body, tc.want) {
				t.Errorf("%s %s: %d, want %q", name, route, rec.Code, tc.want)
			}
		}
	}
	if e.d.exp.snapshot() != nil || decideCalls(f) != 0 {
		t.Fatal("invalid input started an experiment or reached the endpoint")
	}
	// A valid preflight shows the resolved input and sends nothing.
	body := e.post(t, "/experiments/preflight", runForm(dataset, defs)).Body.String()
	sum := sha256.Sum256([]byte(datasetJSONL))
	def, _ := question.Parse([]byte(defJSON))
	for _, s := range []string{"preflight passed", hex.EncodeToString(sum[:]), "scope@3 · " + def.Digest(), "7 (1 warmup + 2 × 3)"} {
		if !strings.Contains(html.UnescapeString(body), s) {
			t.Errorf("preflight lacks %q", s)
		}
	}
	if decideCalls(f) != 0 {
		t.Fatal("preflight reached the endpoint")
	}
}

func TestExperimentRunsEvalRunEvidenceAndExports(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	dataset, defs := expFiles(t)
	rec := e.post(t, "/experiments/run", runForm(dataset, defs))
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/experiments" {
		t.Fatalf("run: %d %s", rec.Code, rec.Body)
	}
	x := waitExp(t, e)
	if x.State != ExpSucceeded || x.Report == nil || x.Done != 7 || x.Percent() != 100 {
		t.Fatalf("experiment %+v", x)
	}
	r := x.Report
	// Only Case.Request projections crossed the API: 1 warmup + 2 passes x 3.
	paths, bodies := f.calls()
	if len(paths) != 7 {
		t.Fatalf("decide calls %v", paths)
	}
	for _, b := range bodies {
		var req api.DecideRequest
		dec := json.NewDecoder(bytes.NewReader(b))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil || bytes.Contains(b, []byte("expected")) || bytes.Contains(b, []byte("question_refs")) {
			t.Fatalf("request is not a plain decide request (%v): %s", err, b)
		}
	}
	// The report is eval's, with identities preserved.
	sum := sha256.Sum256([]byte(datasetJSONL))
	def, _ := question.Parse([]byte(defJSON))
	if r.Schema != eval.EvidenceSchema || r.DatasetSHA256 != hex.EncodeToString(sum[:]) || r.Dataset != dataset ||
		r.Endpoint != "http://"+e.d.cfg.APIAddr || len(r.Definitions) != 1 || r.Definitions[0] != def.Identity() ||
		r.Served == nil || r.Served.Model() != "org/laya@rev1" || !r.ServedConsistent || r.Cases != 3 || r.Observations != 3 ||
		r.WarmupRequests != 1 || r.Passes != 2 || r.RequestLatency.N != 6 || r.ServerInference.N != 6 {
		t.Fatalf("report %+v", r)
	}
	// The page shows exactly the report's numbers.
	body := html.UnescapeString(e.get(t, "/experiments").Body.String())
	for _, s := range []string{"succeeded", "7 / 7", ">3</dd>", f4(r.ChoiceAccuracy), f4(r.MeanConfidence), f4(r.ECE),
		f4(r.PerQuestion["x"].Accuracy), f4(r.PerQuestion["scope"].ECE), r.DatasetSHA256, r.Served.Digest, def.Digest()} {
		if !strings.Contains(body, s) {
			t.Errorf("page lacks %q", s)
		}
	}

	// Export writes canonical evidence that the replay surface accepts.
	out := filepath.Join(t.TempDir(), "report.json")
	seq := url.Values{"seq": {"1"}, "export_path": {out}}
	if body := e.post(t, "/experiments/export", seq).Body.String(); !strings.Contains(body, "wrote hachidori.evidence.v1 report to") {
		t.Fatalf("export: %s", body)
	}
	got, err := os.ReadFile(out)
	want, _ := json.MarshalIndent(r, "", "  ")
	if err != nil || !bytes.Equal(got, append(want, '\n')) {
		t.Fatal("exported bytes are not the canonical report encoding")
	}
	loaded, err := eval.LoadReport(out)
	if err != nil {
		t.Fatal(err)
	}
	set, _ := question.Load(defs)
	cases, dsum, _ := eval.Load(loaded.Dataset, set)
	if items, err := eval.ReplayRequests(loaded, cases, dsum, eval.Selection{}); err != nil || len(items) != 3 {
		t.Fatalf("replay reconstruction: %v %d", err, len(items))
	}
	// Exports never overwrite, and need an absolute path.
	if body := e.post(t, "/experiments/export", seq).Body.String(); !strings.Contains(body, "already exists") {
		t.Error("export overwrote a file")
	}
	if body := e.post(t, "/experiments/export", url.Values{"seq": {"1"}, "export_path": {"r.json"}}).Body.String(); !strings.Contains(body, "must be absolute") {
		t.Error("relative export accepted")
	}
	if body := e.post(t, "/experiments/export", url.Values{"seq": {"9"}, "export_path": {out + "2"}}).Body.String(); !strings.Contains(body, "no finished experiment report") {
		t.Error("export of an unknown run accepted")
	}
}

func TestExperimentRequestFailureIsReported(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	f.reply = func(req api.DecideRequest) (int, any) {
		if req.State == "b" {
			return http.StatusInternalServerError, api.ErrorBody{Schema: api.SchemaV1, Error: api.ErrorInfo{Class: api.ErrInferenceFailed, Message: "boom"}}
		}
		return http.StatusOK, echo(req)
	}
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", with(runForm(dataset, defs), "warmup", "0", "passes", "1"))
	x := waitExp(t, e)
	if x.State != ExpFailed || x.Report == nil || len(x.Report.Errors) != 1 || x.Report.Errors[0].Class != api.ErrInferenceFailed || x.Err != "1 request errors" {
		t.Fatalf("experiment %+v", x)
	}
	body := e.get(t, "/experiments").Body.String()
	if !strings.Contains(body, "1 request errors") || !strings.Contains(body, "inference_failed") || !strings.Contains(body, `action="/experiments/export"`) {
		t.Error("request failure not shown / report not exportable")
	}
}

func TestExperimentServedIdentityInconsistency(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	f.status = func(n int) string {
		if n == 0 {
			return statusDoc("org/laya@rev1", 10)
		}
		return statusDoc("org/laya@rev2", 20)
	}
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	x := waitExp(t, e)
	if x.State != ExpFailed || x.Report == nil || x.Report.ServedConsistent || !strings.Contains(x.Err, "served_consistent=false") {
		t.Fatalf("experiment %+v", x)
	}
	// No served identity at all: no report.
	f.mu.Lock()
	f.status = func(int) string { return "" }
	f.mu.Unlock()
	e.post(t, "/experiments/run", runForm(dataset, defs))
	x = waitExp(t, e)
	if x.Seq != 2 || x.State != ExpFailed || x.Report != nil || !strings.Contains(x.Err, "cannot identify served runtime") {
		t.Fatalf("experiment %+v", x)
	}
}

func TestExperimentSecondRunRefusedAndLifecycleCleanup(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	f.block = make(chan struct{})
	dataset, defs := expFiles(t)
	if rec := e.post(t, "/experiments/run", runForm(dataset, defs)); rec.Code != http.StatusSeeOther {
		t.Fatalf("run: %d", rec.Code)
	}
	deadline := time.Now().Add(5 * time.Second)
	for decideCalls(f) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	// Progress is visible while the run is blocked in its first request.
	live := e.get(t, "/experiments/live").Body.String()
	if !strings.Contains(live, "data-running") || !strings.Contains(live, "0 / 7") {
		t.Fatalf("live fragment: %s", live)
	}
	if page := e.get(t, "/experiments").Body.String(); !strings.Contains(page, `class="btn primary" disabled`) {
		t.Error("run button enabled while running")
	}
	body := e.post(t, "/experiments/run", runForm(dataset, defs)).Body.String()
	if !strings.Contains(body, errBusy.Error()) || e.d.exp.snapshot().Seq != 1 {
		t.Fatal("second run not refused")
	}
	// Close aborts the run, ends the in-flight request and keeps no report.
	done := make(chan struct{})
	go func() { e.d.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not stop the experiment")
	}
	x := e.d.exp.snapshot()
	if x.State != ExpAborted || x.Report != nil || decideCalls(f) != 1 {
		t.Fatalf("experiment %+v, calls %d", x, decideCalls(f))
	}
	if body := e.post(t, "/experiments/export", url.Values{"seq": {"1"}, "export_path": {filepath.Join(t.TempDir(), "r.json")}}).Body.String(); !strings.Contains(body, "no finished experiment report") {
		t.Error("aborted run exported")
	}
	e.d.StopExperiment() // idempotent
	close(f.block)
}

func TestExperimentRoutesKeepDashboardBoundaries(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	dataset, defs := expFiles(t)
	v := runForm(dataset, defs)
	v.Set("token", "stale")
	for _, p := range []string{"/experiments/run", "/experiments/preflight", "/experiments/export"} {
		if rec := e.post(t, p, v); rec.Code != http.StatusForbidden {
			t.Errorf("%s without token: %d", p, rec.Code)
		}
	}
	if e.d.exp.snapshot() != nil || decideCalls(f) != 0 {
		t.Fatal("refused request ran")
	}
	if !strings.Contains(e.get(t, "/").Body.String(), `<a href="/experiments">Experiments</a>`) {
		t.Error("navigation lacks Experiments")
	}
	// The endpoint must be ready before a run starts.
	f.mu.Lock()
	f.status = func(int) string { return "" }
	f.mu.Unlock()
	e.post(t, "/experiments/run", runForm(dataset, defs))
	if x := waitExp(t, e); x.Report != nil {
		t.Fatal("report without served identity")
	}
}

// historyEnv is a workbench env whose dashboard also has a history root under
// its home, like the desktop composition.
func historyEnv(t *testing.T) (*env, *fakeAPI, string) {
	t.Helper()
	e, f := newWorkbenchEnv(t)
	cfg := e.d.cfg
	cfg.HistoryDir = filepath.Join(e.home, "state", "history")
	e.d = New(cfg)
	return e, f, cfg.HistoryDir
}

func TestExperimentHistorySaveListOpenDeleteLifecycle(t *testing.T) {
	e, f, root := historyEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	x := waitExp(t, e)
	if x.State != ExpSucceeded {
		t.Fatalf("experiment %+v", x)
	}
	// Memory-only until explicitly saved.
	if _, err := os.Stat(root); err == nil {
		t.Fatal("history root exists before an explicit save")
	}
	page := html.UnescapeString(e.get(t, "/experiments").Body.String())
	if !strings.Contains(page, "No experiment has been saved yet") || !strings.Contains(page, "/experiments/save") {
		t.Fatal("history region or save action missing")
	}
	exported := filepath.Join(t.TempDir(), "report.json")
	e.post(t, "/experiments/export", url.Values{"seq": {"1"}, "export_path": {exported}})
	exportedBefore, _ := os.ReadFile(exported)
	calls := decideCalls(f)

	body := html.UnescapeString(e.post(t, "/experiments/save", url.Values{"seq": {"1"}, "label": {"baseline <1>"}, "note": {"first"}}).Body.String())
	if !strings.Contains(body, "saved to history as ") {
		t.Fatalf("save: %s", body)
	}
	// The stored evidence is byte-identical to the canonical export, and the
	// label lives outside it.
	dirs, _ := os.ReadDir(filepath.Join(root, "entries"))
	if len(dirs) != 1 {
		t.Fatalf("entries %v", dirs)
	}
	id := dirs[0].Name()
	stored, err := os.ReadFile(filepath.Join(root, "entries", id, "evidence.json"))
	if err != nil || !bytes.Equal(stored, exportedBefore) {
		t.Fatalf("stored evidence differs from the canonical export: %v", err)
	}
	if bytes.Contains(stored, []byte("baseline")) {
		t.Fatal("label leaked into canonical evidence")
	}
	r := x.Report
	list := html.UnescapeString(e.get(t, "/experiments").Body.String())
	for _, s := range []string{id, "baseline <1>", "org/laya@rev1", f4(r.ChoiceAccuracy), r.DatasetSHA256[:12], dataset} {
		if !strings.Contains(list, s) {
			t.Errorf("history list lacks %q", s)
		}
	}

	// A new composition over the same home lists it again, and opens it in
	// the Evidence workspace.
	e.d = New(e.d.cfg)
	if !strings.Contains(e.get(t, "/experiments").Body.String(), id) {
		t.Fatal("history did not survive a new composition")
	}
	rec := e.post(t, "/history/open", url.Values{"id": {id}})
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/errors" {
		t.Fatalf("open: %d %s", rec.Code, rec.Body)
	}
	ev := html.UnescapeString(e.get(t, "/errors").Body.String())
	if !strings.Contains(ev, "history "+id) || !strings.Contains(ev, r.DatasetSHA256) {
		t.Fatal("Evidence workspace does not show the opened history entry")
	}
	// Listing, opening and saving never called inference.
	if decideCalls(f) != calls {
		t.Fatal("history reached the inference endpoint")
	}

	// Delete removes only the entry.
	if body := e.post(t, "/history/delete", url.Values{"id": {id}}).Body.String(); !strings.Contains(body, "deleted history entry") {
		t.Fatalf("delete: %s", body)
	}
	if left, _ := os.ReadDir(filepath.Join(root, "entries")); len(left) != 0 {
		t.Fatalf("entry not deleted: %v", left)
	}
	if after, err := os.ReadFile(exported); err != nil || !bytes.Equal(after, exportedBefore) {
		t.Fatal("exported evidence was touched")
	}
	if _, err := os.Stat(dataset); err != nil {
		t.Fatal("source dataset was touched")
	}
	if !strings.Contains(e.get(t, "/experiments").Body.String(), "No experiment has been saved yet") {
		t.Fatal("deleted entry still listed")
	}
}

func TestExperimentHistoryRefusesBadInputAndTraversal(t *testing.T) {
	e, _, root := historyEnv(t)
	victim := filepath.Join(e.home, "state", "dashboard.json")
	os.WriteFile(victim, []byte("{}"), 0o644)
	// Nothing to save yet.
	if body := e.post(t, "/experiments/save", url.Values{"seq": {"1"}}).Body.String(); !strings.Contains(body, "no finished experiment report to save") {
		t.Fatalf("save without a report: %s", body)
	}
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	e.post(t, "/experiments/save", url.Values{"seq": {"1"}})
	dirs, _ := os.ReadDir(filepath.Join(root, "entries"))
	if len(dirs) != 1 {
		t.Fatalf("entries %v", dirs)
	}
	for _, id := range []string{"../../dashboard.json", "..", "../entries/" + dirs[0].Name(), filepath.Join(root, "entries", dirs[0].Name()), ""} {
		for _, route := range []string{"/history/open", "/history/delete"} {
			rec := e.post(t, route, url.Values{"id": {id}})
			if rec.Code == http.StatusSeeOther || !strings.Contains(rec.Body.String(), "invalid history entry id") {
				t.Errorf("%s %q accepted", route, id)
			}
		}
	}
	if _, err := os.Stat(victim); err != nil {
		t.Fatal("delete escaped the history root")
	}
	if left, _ := os.ReadDir(filepath.Join(root, "entries")); len(left) != 1 {
		t.Fatal("a refused delete removed the entry")
	}
	// A malformed stored entry is reported and cannot be opened.
	bad := filepath.Join(root, "entries", "20260101T000000Z-bbbbbbbbbbbb")
	os.MkdirAll(bad, 0o755)
	os.WriteFile(filepath.Join(bad, "evidence.json"), []byte(`{"schema":"hachidori.evidence.v1","x":1}`), 0o644)
	if body := e.get(t, "/experiments").Body.String(); !strings.Contains(body, "20260101T000000Z-bbbbbbbbbbbb cannot be read") {
		t.Error("malformed entry not reported")
	}
	if rec := e.post(t, "/history/open", url.Values{"id": {"20260101T000000Z-bbbbbbbbbbbb"}}); rec.Code == http.StatusSeeOther {
		t.Error("malformed entry opened")
	}
}

func TestExperimentHistoryDisabledWithoutRoot(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	page := e.get(t, "/experiments").Body.String()
	if strings.Contains(page, "/experiments/save") || strings.Contains(page, `id="hist-h"`) {
		t.Fatal("history UI shown without a history root")
	}
	if body := e.post(t, "/experiments/save", url.Values{"seq": {"1"}}).Body.String(); !strings.Contains(body, "not enabled") {
		t.Fatalf("save without history: %s", body)
	}
	if rec := e.post(t, "/history/open", url.Values{"id": {"x"}}); rec.Code == http.StatusSeeOther {
		t.Fatal("open without history redirected")
	}
}

func f4(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) }
