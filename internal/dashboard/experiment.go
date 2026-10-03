package dashboard

// Experiment Runner: a thin operator adapter around the caller-side
// evaluation authority (internal/eval). The runner resolves Question
// Definitions with question.Load, loads and validates the whole dataset with
// eval.Load before any inference, and runs eval.RunEvidence against the
// resident runtime's existing inference API, exactly as `hachidori
// benchmark` does. Scoring, calibration, served identity and the
// hachidori.evidence.v1 report are eval's; the dashboard only counts
// requests for progress, keeps the one current report in memory and writes
// it to a file when the operator asks.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/history"
	"github.com/yohn-jp/hachidori/internal/question"
)

// Experiment run states.
const (
	ExpRunning   = "running"
	ExpSucceeded = "succeeded" // report produced, no request errors, served identity consistent
	ExpFailed    = "failed"    // no report, or a report with request errors / inconsistent served identity
	ExpAborted   = "aborted"   // stopped before completion; no report is kept
)

// Bounds on the run options accepted from the page.
const (
	maxWarmup = 10000
	maxPasses = 100
)

// ExperimentInput is what the operator selected: explicit absolute local
// paths and the eval.Options.
type ExperimentInput struct {
	Dataset     string
	Definitions []string
	Warmup      int
	Passes      int
}

// Preflight is the fully resolved, validated input of a run: the result of
// question.Load and eval.Load, before any inference.
type Preflight struct {
	Input         ExperimentInput
	DatasetSHA256 string
	Cases         int
	Questions     []string            // distinct question ids, sorted
	Definitions   []question.Identity // resolved definitions, as eval records them
	Requests      int                 // decide requests the run will send
	cases         []eval.Case
}

// Experiment is the state of the current (or last) run.
type Experiment struct {
	Seq      int
	State    string
	Pre      Preflight
	Done     int // decide requests completed (warmup included)
	Started  time.Time
	Finished time.Time
	Err      string
	Report   *eval.Report // nil while running, after an abort, or when no report was produced
	Exported []string     // files the report was written to
}

// Percent is completed requests as a percentage of the planned total.
func (e Experiment) Percent() float64 {
	if e.Pre.Requests == 0 {
		return 0
	}
	return max(0, min(100, 100*float64(e.Done)/float64(e.Pre.Requests)))
}

// experiments owns the dashboard's single experiment slot.
type experiments struct {
	mu     sync.Mutex
	seq    int
	cur    *Experiment
	cancel context.CancelFunc
	done   chan struct{}
}

func (x *experiments) snapshot() *Experiment {
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.cur == nil {
		return nil
	}
	e := *x.cur
	e.Exported = append([]string(nil), x.cur.Exported...)
	return &e
}

// errBusy is returned when a run is requested while one is running.
var errBusy = errors.New("an experiment is already running; wait for it to finish")

// preflight resolves definitions and loads the dataset with the eval
// authority. Nothing is sent to the endpoint.
func preflight(in ExperimentInput) (Preflight, error) {
	pre := Preflight{Input: in}
	dataset, err := absPath(in.Dataset, "dataset")
	if err != nil {
		return pre, err
	}
	if in.Warmup < 0 || in.Warmup > maxWarmup {
		return pre, fmt.Errorf("warmup must be 0..%d", maxWarmup)
	}
	if in.Passes < 1 || in.Passes > maxPasses {
		return pre, fmt.Errorf("passes must be 1..%d", maxPasses)
	}
	// Only the checked paths are used from here on.
	var defPaths []string
	for _, d := range in.Definitions {
		p, err := absPath(d, "Question Definition")
		if err != nil {
			return pre, err
		}
		defPaths = append(defPaths, p)
	}
	var defs *question.Set
	if len(defPaths) > 0 {
		if defs, err = question.Load(defPaths...); err != nil {
			return pre, err
		}
	}
	pre.Input = ExperimentInput{Dataset: dataset, Definitions: defPaths, Warmup: in.Warmup, Passes: in.Passes}
	cases, sum, err := eval.Load(dataset, defs)
	if err != nil {
		return pre, err
	}
	ids := map[string]bool{}
	for _, c := range cases {
		for _, q := range c.Questions {
			ids[q.ID] = true
		}
	}
	for id := range ids {
		pre.Questions = append(pre.Questions, id)
	}
	sort.Strings(pre.Questions)
	pre.DatasetSHA256, pre.Cases, pre.cases = sum, len(cases), cases
	// Shown before inference only; the report records eval's own list.
	seen := map[question.Identity]bool{}
	for _, c := range cases {
		for _, id := range c.Definitions {
			if !seen[id] {
				seen[id] = true
				pre.Definitions = append(pre.Definitions, id)
			}
		}
	}
	sort.Slice(pre.Definitions, func(i, j int) bool { return pre.Definitions[i].String() < pre.Definitions[j].String() })
	pre.Requests = in.Warmup + in.Passes*len(cases)
	return pre, nil
}

// counted is the evidence endpoint eval.RunEvidence talks to: the existing
// HTTP client, plus a request counter and an abort context.
type counted struct {
	c    *client.Client
	ctx  context.Context
	tick func()
}

func (e counted) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	if err := e.ctx.Err(); err != nil {
		return api.DecideResponse{}, err
	}
	resp, err := e.c.Decide(r)
	e.tick()
	return resp, err
}

func (e counted) Status() (json.RawMessage, error) { return e.c.Status() }

// ctxTransport ties every request to the run's context so an abort also
// ends an in-flight request.
type ctxTransport struct {
	ctx context.Context
	rt  http.RoundTripper
}

func (t ctxTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return t.rt.RoundTrip(r.Clone(t.ctx))
}

// start runs pre in the background. It refuses when a run is in progress.
func (d *Dashboard) startExperiment(pre Preflight) (*Experiment, error) {
	x := &d.exp
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.cur != nil && x.cur.State == ExpRunning {
		return nil, errBusy
	}
	ctx, cancel := context.WithCancel(context.Background())
	x.seq++
	x.cur = &Experiment{Seq: x.seq, State: ExpRunning, Pre: pre, Started: time.Now()}
	x.cancel, x.done = cancel, make(chan struct{})
	cur, done := x.cur, x.done
	c := d.endpoint()
	c.HTTP.Transport = ctxTransport{ctx: ctx, rt: http.DefaultTransport}
	ep := counted{c: c, ctx: ctx, tick: func() {
		x.mu.Lock()
		cur.Done++
		x.mu.Unlock()
	}}
	go func() {
		defer close(done)
		defer cancel()
		r, err := eval.RunEvidence(ep, pre.cases, eval.Options{Warmup: pre.Input.Warmup, Passes: pre.Input.Passes})
		x.mu.Lock()
		defer x.mu.Unlock()
		cur.Finished = time.Now()
		switch {
		case ctx.Err() != nil:
			cur.State, cur.Err = ExpAborted, "experiment stopped before completion; no report was kept"
		case err != nil:
			cur.State, cur.Err = ExpFailed, err.Error()
		default:
			r.Endpoint, r.Dataset, r.DatasetSHA256 = c.Endpoint, pre.Input.Dataset, pre.DatasetSHA256
			cur.Report = &r
			cur.State = ExpSucceeded
			switch {
			case len(r.Errors) > 0:
				cur.State, cur.Err = ExpFailed, fmt.Sprintf("%d request errors", len(r.Errors))
			case !r.ServedConsistent:
				cur.State, cur.Err = ExpFailed, "served runtime identity changed during the run; evidence is marked served_consistent=false"
			}
		}
	}()
	e := *cur
	return &e, nil
}

// StopExperiment aborts a running experiment and waits for it to end. An
// aborted run keeps no report. It is safe to call at any time.
func (d *Dashboard) StopExperiment() {
	x := &d.exp
	x.mu.Lock()
	cancel, done := x.cancel, x.done
	x.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
}

// exportReport writes the current report as canonical hachidori.evidence.v1
// JSON (the same encoding `hachidori eval -out` writes) to a new file.
func (d *Dashboard) exportReport(seq int, path string) (string, error) {
	x := &d.exp
	x.mu.Lock()
	var r *eval.Report
	if x.cur != nil && x.cur.Seq == seq && x.cur.State != ExpRunning {
		r = x.cur.Report
	}
	x.mu.Unlock()
	if r == nil {
		return "", errors.New("no finished experiment report to export")
	}
	p, err := absPath(path, "export")
	if err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", err
	}
	if err := writeNew(p, append(b, '\n')); err != nil {
		return "", err
	}
	x.mu.Lock()
	if x.cur != nil && x.cur.Seq == seq {
		x.cur.Exported = append(x.cur.Exported, p)
	}
	x.mu.Unlock()
	return p, nil
}

// expView is the Experiments page's view model.
type expView struct {
	Chrome
	Token    string
	Endpoint string
	Form     expForm
	Pre      *Preflight
	Msg      string
	Err      string
	Exp      *Experiment

	// Experiment history; filled only on full-page renders.
	HistoryOn       bool
	HistoryRoot     string
	History         []history.Summary // every entry, newest first (Compare lists all)
	HistoryProblems []history.Problem
	HistoryErr      string
	// HistoryRows is the page of History the table renders: at most
	// historyPage entries from HistoryFrom; HistoryPrev and HistoryNext are
	// the neighbouring page offsets, -1 when there is none.
	HistoryRows              []history.Summary
	HistoryFrom              int
	HistoryPrev, HistoryNext int

	// Comparison of two stored entries (POST /history/compare); nil otherwise.
	CompareA, CompareB string
	Compare            *compareView
	PathPicker         bool
}

// expForm is the run form as typed (kept on errors).
type expForm struct {
	Dataset     string
	Definitions string
	Warmup      string
	Passes      string
	ExportPath  string
}

func (f expForm) input() (ExperimentInput, error) {
	in := ExperimentInput{Dataset: strings.TrimSpace(f.Dataset)}
	for _, l := range strings.Split(f.Definitions, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			in.Definitions = append(in.Definitions, l)
		}
	}
	var err error
	if in.Warmup, err = strconv.Atoi(strings.TrimSpace(f.Warmup)); err != nil {
		return in, fmt.Errorf("warmup: not a number")
	}
	if in.Passes, err = strconv.Atoi(strings.TrimSpace(f.Passes)); err != nil {
		return in, fmt.Errorf("passes: not a number")
	}
	return in, nil
}

func formOf(in ExperimentInput) expForm {
	return expForm{Dataset: in.Dataset, Definitions: strings.Join(in.Definitions, "\n"),
		Warmup: strconv.Itoa(in.Warmup), Passes: strconv.Itoa(in.Passes)}
}

func (d *Dashboard) expView() expView {
	v := expView{Chrome: d.chrome("Experiments", "experiments"),
		Token: d.token, Endpoint: "http://" + d.cfg.APIAddr, Exp: d.exp.snapshot(), PathPicker: d.cfg.PathPicker != nil,
		Form: expForm{Warmup: "0", Passes: "1"}, HistoryOn: d.hist != nil}
	if v.Exp != nil {
		v.Form = formOf(v.Exp.Pre.Input)
	}
	return v
}

func (d *Dashboard) experimentsPick(w http.ResponseWriter, r *http.Request) {
	v := d.expPageView()
	pick := r.PostFormValue("pick")
	if pick == "export" {
		// The export form carries only the destination, like
		// experimentsExport; the run form state stays as shown.
		v.Form.ExportPath = strings.TrimSpace(r.PostFormValue("export_path"))
	} else {
		v.Form = postedForm(r)
	}
	if d.cfg.PathPicker == nil {
		v.Err = "native path selection is unavailable"
		d.renderView(w, "experiments", v)
		return
	}
	var (
		path string
		err  error
	)
	switch pick {
	case "dataset":
		if path, err = d.cfg.PathPicker.PickOpen(r.Context(), "Choose a dataset"); err == nil {
			v.Form.Dataset = path
		}
	case "definition-file":
		if path, err = d.cfg.PathPicker.PickOpen(r.Context(), "Choose a Question Definition"); err == nil {
			v.Form.Definitions = appendLine(v.Form.Definitions, path)
		}
	case "definition-folder":
		if path, err = d.cfg.PathPicker.PickFolder(r.Context(), "Choose a Question Definition folder"); err == nil {
			v.Form.Definitions = appendLine(v.Form.Definitions, path)
		}
	case "export":
		if path, err = d.cfg.PathPicker.PickSave(r.Context(), "Choose where to save the experiment report"); err == nil {
			v.Form.ExportPath = path
		}
	default:
		v.Err = "unknown native picker operation"
	}
	if err != nil && !pickWasCancelled(err) {
		v.Err = "choosing a path: " + err.Error()
	}
	d.renderView(w, "experiments", v)
}

// appendLine adds one chosen path to a one-per-line list, keeping the paths
// already entered.
func appendLine(list, line string) string {
	if strings.TrimSpace(list) == "" {
		return line
	}
	return strings.TrimRight(list, "\n") + "\n" + line
}

// expPageView is expView plus the saved-history listing. Listing reads the
// history root only and never contacts the endpoint; the live fragment does
// not list history.
func (d *Dashboard) expPageView() expView { return d.expPageAt(0) }

// historyPage bounds the saved-history rows rendered at once, so the
// Experiments workspace stays one bounded document however much history
// accumulates; older entries stay reachable page by page and in Compare.
const historyPage = 100

// expPageAt is expPageView showing the history table page at offset from.
func (d *Dashboard) expPageAt(from int) expView {
	v := d.expView()
	v.HistoryPrev, v.HistoryNext = -1, -1
	if d.hist == nil {
		v.HistoryErr = d.histErr
		return v
	}
	v.HistoryRoot = d.hist.Root()
	var err error
	if v.History, v.HistoryProblems, err = d.hist.List(); err != nil {
		v.HistoryErr = "cannot list history: " + err.Error()
	}
	from = min(max(from, 0), max(len(v.History)-1, 0)) / historyPage * historyPage
	v.HistoryFrom, v.HistoryRows = from, v.History[from:min(from+historyPage, len(v.History))]
	if from > 0 {
		v.HistoryPrev = from - historyPage
	}
	if from+historyPage < len(v.History) {
		v.HistoryNext = from + historyPage
	}
	return v
}

func (d *Dashboard) experimentsPage(w http.ResponseWriter, r *http.Request) {
	from, _ := strconv.Atoi(r.URL.Query().Get("history"))
	d.renderView(w, "experiments", d.expPageAt(from))
}

func (d *Dashboard) experimentsLive(w http.ResponseWriter, r *http.Request) {
	d.renderView(w, "experiment-live", d.expView())
}

func postedForm(r *http.Request) expForm {
	return expForm{Dataset: r.PostFormValue("dataset"), Definitions: nl(r.PostFormValue("definitions")),
		Warmup: r.PostFormValue("warmup"), Passes: r.PostFormValue("passes"), ExportPath: strings.TrimSpace(r.PostFormValue("export_path"))}
}

// experimentsPreflight validates the selection without inference.
func (d *Dashboard) experimentsPreflight(w http.ResponseWriter, r *http.Request) {
	v := d.expPageView()
	v.Form = postedForm(r)
	in, err := v.Form.input()
	if err == nil {
		var pre Preflight
		if pre, err = preflight(in); err == nil {
			v.Pre, v.Msg = &pre, "preflight passed; nothing was sent to the endpoint"
		}
	}
	if err != nil {
		v.Err = "preflight failed: " + err.Error()
	}
	d.renderView(w, "experiments", v)
}

// experimentsRun preflights and, when the whole dataset resolves and the
// endpoint is ready, starts the run in the background.
func (d *Dashboard) experimentsRun(w http.ResponseWriter, r *http.Request) {
	v := d.expPageView()
	v.Form = postedForm(r)
	fail := func(err error) {
		v.Err = err.Error()
		d.renderView(w, "experiments", v)
	}
	if e := d.exp.snapshot(); e != nil && e.State == ExpRunning {
		fail(errBusy)
		return
	}
	in, err := v.Form.input()
	if err != nil {
		fail(fmt.Errorf("preflight failed: %w", err))
		return
	}
	pre, err := preflight(in)
	if err != nil {
		fail(fmt.Errorf("preflight failed: %w", err))
		return
	}
	c := d.endpoint()
	if h, err := c.Health(); err != nil || !h.Ready {
		fail(fmt.Errorf("endpoint %s not ready (state %q): %v", c.Endpoint, h.State, err))
		return
	}
	if _, err := d.startExperiment(pre); err != nil {
		fail(err)
		return
	}
	http.Redirect(w, r, "/experiments", http.StatusSeeOther)
}

func (d *Dashboard) experimentsExport(w http.ResponseWriter, r *http.Request) {
	v := d.expPageView()
	v.Form.ExportPath = strings.TrimSpace(r.PostFormValue("export_path"))
	seq, _ := strconv.Atoi(r.PostFormValue("seq"))
	if p, err := d.exportReport(seq, v.Form.ExportPath); err != nil {
		v.Err = "export failed: " + err.Error()
	} else {
		v.Msg = "wrote " + eval.EvidenceSchema + " report to " + p
	}
	v.Exp = d.exp.snapshot()
	d.renderView(w, "experiments", v)
}

// experimentsSave stores the current finished experiment's canonical
// hachidori.evidence.v1 report in the history root. Nothing is saved unless
// the operator asks; the label and note are kept beside the evidence.
func (d *Dashboard) experimentsSave(w http.ResponseWriter, r *http.Request) {
	seq, _ := strconv.Atoi(r.PostFormValue("seq"))
	msg, err := d.saveExperiment(seq, strings.TrimSpace(r.PostFormValue("label")), nl(strings.TrimSpace(r.PostFormValue("note"))))
	v := d.expPageView()
	if err != nil {
		v.Err = "save to history failed: " + err.Error()
	} else {
		v.Msg = msg
	}
	d.renderView(w, "experiments", v)
}

var errNoHistory = errors.New("experiment history is not enabled in this workstation")

func (d *Dashboard) saveExperiment(seq int, label, note string) (string, error) {
	if d.hist == nil {
		return "", errNoHistory
	}
	x := &d.exp
	x.mu.Lock()
	var rep *eval.Report
	if x.cur != nil && x.cur.Seq == seq && x.cur.State != ExpRunning {
		rep = x.cur.Report
	}
	x.mu.Unlock()
	if rep == nil {
		return "", errors.New("no finished experiment report to save")
	}
	// The same canonical encoding as the manual export.
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return "", err
	}
	sm, err := d.hist.Save(append(b, '\n'), label, note)
	if err != nil {
		return "", err
	}
	return "saved to history as " + sm.ID, nil
}

// historyOpen opens one saved entry in the Evidence workspace through the
// same strict decoding as a report file. It never contacts the endpoint.
func (d *Dashboard) historyOpen(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PostFormValue("id"))
	err := errNoHistory
	if d.hist != nil {
		var rep eval.Report
		var sum string
		if rep, sum, err = d.hist.Open(id); err == nil {
			d.errs.set(&evidenceSource{Report: rep, SHA256: sum, Origin: "history " + id, HistoryID: id})
			http.Redirect(w, r, "/errors", http.StatusSeeOther)
			return
		}
	}
	v := d.expPageView()
	v.Err = "cannot open history entry: " + err.Error()
	d.renderView(w, "experiments", v)
}

// compareView is a descriptive comparison of two stored history entries. It
// carries the entries' identities so either source can be opened in Evidence.
type compareView struct {
	Token  string
	EntryA history.Summary
	EntryB history.Summary
	eval.Comparison
	// Tuning opens the candidate (B) in Tuning with this exact comparison as
	// its context; TuningWhy says why the pair cannot be handed over. Both are
	// empty when Tuning is unavailable.
	Tuning, TuningWhy string
}

// compareHistory opens two stored entries read-only and compares them with
// eval.Compare. It never contacts the endpoint and changes nothing stored.
func (d *Dashboard) compareHistory(a, b string) (*compareView, error) {
	if d.hist == nil {
		return nil, errNoHistory
	}
	if a == "" || b == "" {
		return nil, errors.New("select two history entries to compare")
	}
	if a == b {
		return nil, errors.New("select two different history entries to compare")
	}
	ra, sa, err := d.hist.Open(a)
	if err != nil {
		return nil, fmt.Errorf("entry A: %w", err)
	}
	rb, sb, err := d.hist.Open(b)
	if err != nil {
		return nil, fmt.Errorf("entry B: %w", err)
	}
	cv := &compareView{Token: d.token, Comparison: eval.Compare(ra, rb),
		EntryA: history.Summary{ID: a, EvidenceSHA256: sa}, EntryB: history.Summary{ID: b, EvidenceSHA256: sb}}
	if d.cfg.hasTuning() {
		if _, vals, err := d.tuningContextOf(a, b); err != nil {
			cv.TuningWhy = err.Error()
		} else {
			cv.Tuning = "/tuning?" + vals.Encode()
		}
	}
	// Operator metadata is display only; it never takes part in the comparison.
	if list, _, err := d.hist.List(); err == nil {
		for _, s := range list {
			switch s.ID {
			case a:
				cv.EntryA = s
			case b:
				cv.EntryB = s
			}
		}
	}
	return cv, nil
}

// historyCompare renders the comparison of two stored entries on the
// Experiments page. Comparison reads the history root only.
func (d *Dashboard) historyCompare(w http.ResponseWriter, r *http.Request) {
	v := d.expPageView()
	v.CompareA, v.CompareB = strings.TrimSpace(r.PostFormValue("a")), strings.TrimSpace(r.PostFormValue("b"))
	cv, err := d.compareHistory(v.CompareA, v.CompareB)
	if err != nil {
		v.Err = "cannot compare history entries: " + err.Error()
	} else {
		v.Compare = cv
	}
	d.renderView(w, "experiments", v)
}

// signed formats a delta with an explicit sign and no judgement.
func signed(v float64, unit string) string {
	s := strconv.FormatFloat(v, 'f', 4, 64)
	if unit == "ms" {
		s = strconv.FormatFloat(v, 'f', 1, 64) + " ms"
	}
	if v > 0 {
		return "+" + s
	}
	return s
}

// historyDelete removes one history entry. An already-opened copy stays in
// memory in the Evidence workspace; exported and source files are never
// touched.
func (d *Dashboard) historyDelete(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PostFormValue("id"))
	err := errNoHistory
	if d.hist != nil {
		err = d.hist.Delete(id)
	}
	v := d.expPageView()
	if err != nil {
		v.Err = "cannot delete history entry: " + err.Error()
	} else {
		v.Msg = "deleted history entry " + id + "; exported reports and datasets are untouched"
	}
	d.renderView(w, "experiments", v)
}

// statRow is one per-question row of the report, in question id order.
type statRow struct {
	ID string
	eval.QuestionStats
}

func perQuestion(m map[string]eval.QuestionStats) []statRow {
	rows := make([]statRow, 0, len(m))
	for id, s := range m {
		rows = append(rows, statRow{id, s})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].ID < rows[j].ID })
	return rows
}
