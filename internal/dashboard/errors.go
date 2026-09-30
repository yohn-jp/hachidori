package dashboard

// Error Explorer: a read-only view over one Decision Evidence report, either
// the current experiment's report or a hachidori.evidence.v1 file the
// operator opens by absolute path. All analysis is internal/eval/explore
// (pure functions of the report); opening a report never contacts the
// endpoint, and filtered exports are separate analysis documents written to
// new files, never over the source.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/eval/explore"
)

// DefaultThreshold is the initial high-confidence analysis control. It is
// only a starting value for the page, not a policy.
const DefaultThreshold = 0.9

// maxRows bounds the observation rows rendered at once; exports are not
// bounded.
const maxRows = 1000

// evidenceSource is the report the explorer is looking at.
type evidenceSource struct {
	Report eval.Report
	SHA256 string // file bytes, or the canonical encoding of an experiment report
	Origin string // "file <path>" or "experiment #N"
}

type explorer struct {
	mu  sync.Mutex
	src *evidenceSource
}

func (x *explorer) get() *evidenceSource {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.src
}

func (x *explorer) set(s *evidenceSource) {
	x.mu.Lock()
	x.src = s
	x.mu.Unlock()
}

// controls are the analysis controls as typed, carried in the query string.
type controls struct {
	Threshold string
	Question  string
	Expected  string
	Predicted string
	Outcome   string
	Min       string
	Max       string
	ErrClass  string
	Sort      string
	Desc      bool
}

func controlsOf(v url.Values) controls {
	c := controls{Threshold: v.Get("th"), Question: v.Get("q"), Expected: v.Get("exp"), Predicted: v.Get("pred"),
		Outcome: v.Get("outcome"), Min: v.Get("min"), Max: v.Get("max"), ErrClass: v.Get("eclass"), Sort: v.Get("sort"),
		Desc: v.Get("desc") == "1"}
	if c.Threshold == "" {
		c.Threshold = strconv.FormatFloat(DefaultThreshold, 'f', -1, 64)
	}
	if c.Min == "" {
		c.Min = "0"
	}
	if c.Max == "" {
		c.Max = "1"
	}
	return c
}

// Query encodes the controls (for links and hidden export fields).
func (c controls) Values() url.Values {
	v := url.Values{"th": {c.Threshold}, "min": {c.Min}, "max": {c.Max}}
	for k, s := range map[string]string{"q": c.Question, "exp": c.Expected, "pred": c.Predicted, "outcome": c.Outcome,
		"eclass": c.ErrClass, "sort": c.Sort} {
		if s != "" {
			v.Set(k, s)
		}
	}
	if c.Desc {
		v.Set("desc", "1")
	}
	return v
}

func (c controls) Query() string { return c.Values().Encode() }

func (c controls) parse() (float64, explore.Filter, error) {
	num := func(s, what string) (float64, error) {
		f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			return 0, fmt.Errorf("%s: not a number", what)
		}
		return f, nil
	}
	th, err := num(c.Threshold, "high-confidence threshold")
	if err != nil {
		return 0, explore.Filter{}, err
	}
	f := explore.Filter{Question: c.Question, Expected: c.Expected, Predicted: c.Predicted, Outcome: c.Outcome,
		ErrorClass: c.ErrClass, Sort: c.Sort, Desc: c.Desc}
	if f.MinConf, err = num(c.Min, "minimum confidence"); err != nil {
		return 0, f, err
	}
	if f.MaxConf, err = num(c.Max, "maximum confidence"); err != nil {
		return 0, f, err
	}
	return th, f, nil
}

// errView is the Evidence workspace's view model (the Error Explorer).
type errView struct {
	Chrome
	Token      string
	Src        *evidenceSource
	C          controls
	A          *explore.Analysis
	Rows       []explore.Item // at most maxRows of A.Items
	Detail     *explore.Item
	OpenPath   string
	Export     string
	Msg        string
	Err        string
	ExpSeq     int // current experiment with a report, 0 if none
	PathPicker bool
}

func (d *Dashboard) errView(q url.Values) errView {
	v := errView{Chrome: d.chrome("Evidence", "evidence"), Token: d.token,
		Src: d.errs.get(), C: controlsOf(q), PathPicker: d.cfg.PathPicker != nil}
	if e := d.exp.snapshot(); e != nil && e.Report != nil && e.State != ExpRunning {
		v.ExpSeq = e.Seq
	}
	if v.Src == nil {
		return v
	}
	th, f, err := v.C.parse()
	if err == nil {
		var a explore.Analysis
		if a, err = explore.Analyze(v.Src.Report, th, f); err == nil {
			v.A = &a
			v.Rows = a.Items[:min(len(a.Items), maxRows)]
		}
	}
	if err != nil {
		v.Err = err.Error()
	}
	if s := q.Get("obs"); s != "" && v.A != nil {
		if i, err := strconv.Atoi(s); err == nil && i >= 0 && i < len(v.Src.Report.Results) {
			o := v.Src.Report.Results[i]
			v.Detail = &explore.Item{Index: i, HighConfWrong: !o.Correct && o.Confidence >= v.A.Threshold, Observation: o}
		}
	}
	return v
}

func (d *Dashboard) errorsPick(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "invalid form", http.StatusBadRequest)
		return
	}
	v := d.errView(r.PostForm)
	if d.cfg.PathPicker == nil {
		v.Err = "native path selection is unavailable"
		d.renderView(w, "errors", v)
		return
	}
	var path string
	var err error
	switch r.PostFormValue("pick") {
	case "open":
		path, err = d.cfg.PathPicker.PickOpen(r.Context(), "Choose an evidence report")
		if err == nil {
			v.OpenPath = path
		}
	case "export":
		path, err = d.cfg.PathPicker.PickSave(r.Context(), "Choose where to save the evidence analysis")
		if err == nil {
			v.Export = path
		}
	default:
		v.Err = "unknown native picker operation"
	}
	if err != nil && !pickWasCancelled(err) {
		v.Err = "choosing a path: " + err.Error()
	}
	d.renderView(w, "errors", v)
}

func (d *Dashboard) errorsPage(w http.ResponseWriter, r *http.Request) {
	d.renderView(w, "errors", d.errView(r.URL.Query()))
}

// errorsOpen opens a report file. A rejected file leaves the current source
// unchanged.
func (d *Dashboard) errorsOpen(w http.ResponseWriter, r *http.Request) {
	v := d.errView(nil)
	v.OpenPath = strings.TrimSpace(r.PostFormValue("path"))
	p, err := absPath(v.OpenPath, "evidence report")
	if err == nil {
		var rep eval.Report
		var sum string
		if rep, sum, err = explore.Open(p); err == nil {
			d.errs.set(&evidenceSource{Report: rep, SHA256: sum, Origin: "file " + p})
			http.Redirect(w, r, "/errors", http.StatusSeeOther)
			return
		}
	}
	v.Err = "cannot open report: " + err.Error()
	d.renderView(w, "errors", v)
}

// errorsUseExperiment explores the current experiment's report. It is
// validated exactly like a file.
func (d *Dashboard) errorsUseExperiment(w http.ResponseWriter, r *http.Request) {
	v := d.errView(nil)
	seq, _ := strconv.Atoi(r.PostFormValue("seq"))
	e := d.exp.snapshot()
	if e == nil || e.Seq != seq || e.Report == nil || e.State == ExpRunning {
		v.Err = "no finished experiment report to explore"
		d.renderView(w, "errors", v)
		return
	}
	b, err := json.MarshalIndent(e.Report, "", "  ")
	if err == nil {
		b = append(b, '\n')
		var rep eval.Report
		if rep, err = explore.Decode(b); err == nil {
			sum := sha256.Sum256(b)
			d.errs.set(&evidenceSource{Report: rep, SHA256: hex.EncodeToString(sum[:]), Origin: fmt.Sprintf("experiment #%d", e.Seq)})
			http.Redirect(w, r, "/errors", http.StatusSeeOther)
			return
		}
	}
	v.Err = "cannot explore the experiment report: " + err.Error()
	d.renderView(w, "errors", v)
}

// errorsExport writes the current filtered selection as a
// hachidori.evidence-analysis.v1 document to a new file.
func (d *Dashboard) errorsExport(w http.ResponseWriter, r *http.Request) {
	v := d.errView(r.PostForm)
	v.Export = strings.TrimSpace(r.PostFormValue("export_path"))
	err := func() error {
		if v.Src == nil || v.A == nil {
			return errors.New("no report is open")
		}
		p, err := absPath(v.Export, "export")
		if err != nil {
			return err
		}
		_, f, _ := v.C.parse()
		b, err := json.MarshalIndent(explore.NewExport(v.Src.Report, v.Src.SHA256, *v.A, f), "", "  ")
		if err != nil {
			return err
		}
		if err := writeNew(p, append(b, '\n')); err != nil {
			return err
		}
		v.Msg = fmt.Sprintf("wrote %d observations as %s to %s; the source report is unchanged", len(v.A.Items), explore.AnalysisSchema, p)
		return nil
	}()
	if err != nil {
		v.Err = "export failed: " + err.Error()
	}
	d.renderView(w, "errors", v)
}
