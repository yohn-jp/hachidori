// Package explore is a read-only analysis consumer of Decision Evidence
// (hachidori.evidence.v1, eval.Report).
//
// Everything here is a pure, deterministic function of a report: it never
// changes evidence, rescoring rules, expected labels or model output, and it
// never contacts an endpoint. The high-confidence threshold is an analysis
// control only; nothing here judges whether a model or question is fit for
// any use.
package explore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/yohn-jp/hachidori/internal/eval"
)

// AnalysisSchema identifies an exported, filtered observation set. It is
// analysis data derived from a report, never a replacement for it.
const AnalysisSchema = "hachidori.evidence-analysis.v1"

// MaxReportBytes bounds a report file read by Open.
const MaxReportBytes = 256 << 20

// Decode parses one hachidori.evidence.v1 report strictly: unknown fields,
// trailing data (rejected by the schema probe), another schema or an internally inconsistent report are
// rejected rather than guessed at.
func Decode(data []byte) (eval.Report, error) {
	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return eval.Report{}, fmt.Errorf("not a Decision Evidence report: %w", err)
	}
	if probe.Schema != eval.EvidenceSchema {
		return eval.Report{}, fmt.Errorf("evidence schema %q, want %q", probe.Schema, eval.EvidenceSchema)
	}
	var r eval.Report
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return eval.Report{}, fmt.Errorf("malformed %s report: %w", eval.EvidenceSchema, err)
	}
	if err := Validate(r); err != nil {
		return eval.Report{}, fmt.Errorf("malformed %s report: %w", eval.EvidenceSchema, err)
	}
	return r, nil
}

// Open reads and decodes a report file. It also returns the SHA-256 of the
// file's bytes, which identifies the exact source analysed.
func Open(path string) (eval.Report, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return eval.Report{}, "", err
	}
	defer f.Close()
	if fi, err := f.Stat(); err != nil {
		return eval.Report{}, "", err
	} else if !fi.Mode().IsRegular() {
		return eval.Report{}, "", fmt.Errorf("%s is not a regular file", path)
	} else if fi.Size() > MaxReportBytes {
		return eval.Report{}, "", fmt.Errorf("%s: larger than %d bytes", path, MaxReportBytes)
	}
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(f); err != nil {
		return eval.Report{}, "", err
	}
	r, err := Decode(buf.Bytes())
	if err != nil {
		return eval.Report{}, "", fmt.Errorf("%s: %w", path, err)
	}
	sum := sha256.Sum256(buf.Bytes())
	return r, hex.EncodeToString(sum[:]), nil
}

// Validate checks the internal consistency of a report that eval's writer
// guarantees: counts match the observations, every observation is complete
// and scored by exact choice match, and per-question counts reconcile.
func Validate(r eval.Report) error {
	if r.Schema != eval.EvidenceSchema {
		return fmt.Errorf("schema %q, want %q", r.Schema, eval.EvidenceSchema)
	}
	if r.DatasetSHA256 == "" {
		return errors.New("dataset_sha256 is missing")
	}
	if r.Observations != len(r.Results) {
		return fmt.Errorf("observations is %d but results has %d entries", r.Observations, len(r.Results))
	}
	n := map[string]int{}
	for i, o := range r.Results {
		at := fmt.Sprintf("results[%d]", i)
		switch {
		case o.CaseID == "" || o.QuestionID == "":
			return fmt.Errorf("%s: case_id and question_id are required", at)
		case o.Expected == "" || o.Choice == "":
			return fmt.Errorf("%s: expected and choice are required", at)
		case math.IsNaN(o.Confidence) || o.Confidence < 0 || o.Confidence > 1:
			return fmt.Errorf("%s: confidence %v is outside [0, 1]", at, o.Confidence)
		case len(o.Probabilities) == 0:
			return fmt.Errorf("%s: probabilities are missing", at)
		case o.Correct != (o.Choice == o.Expected):
			return fmt.Errorf("%s: correct=%v disagrees with choice %q and expected %q", at, o.Correct, o.Choice, o.Expected)
		case o.RequestMS < 0 || (o.InferenceMS != nil && *o.InferenceMS < 0):
			return fmt.Errorf("%s: negative latency", at)
		}
		if _, ok := o.Probabilities[o.Choice]; !ok {
			return fmt.Errorf("%s: choice %q has no probability", at, o.Choice)
		}
		n[o.QuestionID]++
	}
	if len(n) != len(r.PerQuestion) {
		return fmt.Errorf("per_question has %d entries but results cover %d questions", len(r.PerQuestion), len(n))
	}
	for id, c := range n {
		if s, ok := r.PerQuestion[id]; !ok || s.N != c {
			return fmt.Errorf("per_question[%q].n does not match its %d results", id, c)
		}
	}
	for i, e := range r.Errors {
		if e.Class == "" {
			return fmt.Errorf("errors[%d]: class is required", i)
		}
	}
	return nil
}

// Outcome filter values.
const (
	OutcomeAny           = ""
	OutcomeCorrect       = "correct"
	OutcomeWrong         = "wrong"
	OutcomeHighConfWrong = "high_confidence_wrong"
)

// Sort keys.
const (
	SortReport      = "" // report order
	SortConfidence  = "confidence"
	SortRequestMS   = "request_ms"
	SortInferenceMS = "inference_ms"
)

// Filter selects observations and request errors. Empty strings match
// everything; the confidence range is inclusive.
type Filter struct {
	Question   string  `json:"question,omitempty"`
	Expected   string  `json:"expected,omitempty"`
	Predicted  string  `json:"predicted,omitempty"`
	Outcome    string  `json:"outcome,omitempty"`
	MinConf    float64 `json:"min_confidence"`
	MaxConf    float64 `json:"max_confidence"`
	ErrorClass string  `json:"error_class,omitempty"`
	Sort       string  `json:"sort,omitempty"`
	Desc       bool    `json:"descending,omitempty"`
}

// All is the filter that selects everything in report order.
func All() Filter { return Filter{MinConf: 0, MaxConf: 1} }

// Item is one selected observation with its index in Report.Results.
type Item struct {
	Index         int
	HighConfWrong bool
	eval.Observation
}

// QuestionRow is per-question error analysis beside the report's own
// statistics (Stats is copied from Report.PerQuestion, never recomputed).
type QuestionRow struct {
	ID             string
	Correct        int
	Wrong          int
	HighConfWrong  int
	MissingResults int // request errors of class missing_result for this question
	Stats          eval.QuestionStats
}

// ClassCount is the number of request errors of one class.
type ClassCount struct {
	Class string
	N     int
}

// Analysis is the result of Analyze. Counts cover the whole report; Items and
// Errors are the filtered selection.
type Analysis struct {
	Threshold     float64
	Correct       int
	Wrong         int
	RequestErrors int
	HighConfWrong int
	PerQuestion   []QuestionRow
	ErrorClasses  []ClassCount
	Questions     []string // distinct values, sorted, for filter controls
	Expected      []string
	Predicted     []string
	Items         []Item
	Errors        []eval.RequestError
}

// Analyze derives counts and the filtered, sorted selection from r. The
// threshold (in [0, 1]) marks a wrong observation high-confidence when its
// confidence is >= threshold. r is not modified.
func Analyze(r eval.Report, threshold float64, f Filter) (Analysis, error) {
	if math.IsNaN(threshold) || threshold < 0 || threshold > 1 {
		return Analysis{}, fmt.Errorf("threshold must be in [0, 1], got %v", threshold)
	}
	if math.IsNaN(f.MinConf) || math.IsNaN(f.MaxConf) || f.MinConf < 0 || f.MaxConf > 1 || f.MinConf > f.MaxConf {
		return Analysis{}, fmt.Errorf("confidence range must satisfy 0 <= min <= max <= 1, got [%v, %v]", f.MinConf, f.MaxConf)
	}
	switch f.Outcome {
	case OutcomeAny, OutcomeCorrect, OutcomeWrong, OutcomeHighConfWrong:
	default:
		return Analysis{}, fmt.Errorf("unknown outcome %q", f.Outcome)
	}
	switch f.Sort {
	case SortReport, SortConfidence, SortRequestMS, SortInferenceMS:
	default:
		return Analysis{}, fmt.Errorf("unknown sort key %q", f.Sort)
	}
	a := Analysis{Threshold: threshold, RequestErrors: len(r.Errors)}
	rows := map[string]*QuestionRow{}
	qs, exp, pred := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for i, o := range r.Results {
		hcw := !o.Correct && o.Confidence >= threshold
		row := rows[o.QuestionID]
		if row == nil {
			row = &QuestionRow{ID: o.QuestionID, Stats: r.PerQuestion[o.QuestionID]}
			rows[o.QuestionID] = row
		}
		if o.Correct {
			a.Correct++
			row.Correct++
		} else {
			a.Wrong++
			row.Wrong++
		}
		if hcw {
			a.HighConfWrong++
			row.HighConfWrong++
		}
		qs[o.QuestionID], exp[o.Expected], pred[o.Choice] = true, true, true
		if f.match(o, hcw) {
			a.Items = append(a.Items, Item{Index: i, HighConfWrong: hcw, Observation: o})
		}
	}
	classes := map[string]int{}
	for _, e := range r.Errors {
		classes[e.Class]++
		if e.Class == eval.ErrClassMissingResult && e.QuestionID != "" {
			if row := rows[e.QuestionID]; row != nil {
				row.MissingResults++
			} else {
				rows[e.QuestionID] = &QuestionRow{ID: e.QuestionID, MissingResults: 1}
			}
		}
		if f.ErrorClass == "" || f.ErrorClass == e.Class {
			a.Errors = append(a.Errors, e)
		}
	}
	for _, row := range rows {
		a.PerQuestion = append(a.PerQuestion, *row)
	}
	sort.Slice(a.PerQuestion, func(i, j int) bool { return a.PerQuestion[i].ID < a.PerQuestion[j].ID })
	for c, n := range classes {
		a.ErrorClasses = append(a.ErrorClasses, ClassCount{c, n})
	}
	sort.Slice(a.ErrorClasses, func(i, j int) bool { return a.ErrorClasses[i].Class < a.ErrorClasses[j].Class })
	a.Questions, a.Expected, a.Predicted = keys(qs), keys(exp), keys(pred)
	sortItems(a.Items, f.Sort, f.Desc)
	return a, nil
}

func (f Filter) match(o eval.Observation, hcw bool) bool {
	switch {
	case f.Question != "" && o.QuestionID != f.Question,
		f.Expected != "" && o.Expected != f.Expected,
		f.Predicted != "" && o.Choice != f.Predicted,
		o.Confidence < f.MinConf || o.Confidence > f.MaxConf:
		return false
	}
	switch f.Outcome {
	case OutcomeCorrect:
		return o.Correct
	case OutcomeWrong:
		return !o.Correct
	case OutcomeHighConfWrong:
		return hcw
	}
	return true
}

// sortItems orders by the key, ties (and SortReport) by report order.
// Observations without an inference latency sort last in either direction.
func sortItems(items []Item, key string, desc bool) {
	if key == SortReport {
		if desc {
			sort.SliceStable(items, func(i, j int) bool { return items[i].Index > items[j].Index })
		}
		return
	}
	val := func(it Item) (float64, bool) {
		switch key {
		case SortConfidence:
			return it.Confidence, true
		case SortRequestMS:
			return it.RequestMS, true
		}
		if it.InferenceMS == nil {
			return 0, false
		}
		return *it.InferenceMS, true
	}
	sort.SliceStable(items, func(i, j int) bool {
		a, aok := val(items[i])
		b, bok := val(items[j])
		switch {
		case aok != bok:
			return aok
		case !aok || a == b:
			return items[i].Index < items[j].Index
		case desc:
			return a > b
		}
		return a < b
	})
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Source identifies the report an analysis was derived from.
type Source struct {
	Schema         string `json:"schema"`
	ReportSHA256   string `json:"report_sha256"`
	Dataset        string `json:"dataset"`
	DatasetSHA256  string `json:"dataset_sha256"`
	StartedAt      string `json:"started_at"`
	ServedIdentity string `json:"served_identity_sha256,omitempty"`
	ServedModel    string `json:"served_model,omitempty"`
}

// Counts are the whole-report counts at the export's threshold, plus the
// size of the exported selection.
type Counts struct {
	Correct       int `json:"correct"`
	Wrong         int `json:"wrong"`
	RequestErrors int `json:"request_errors"`
	HighConfWrong int `json:"high_confidence_wrong"`
	Selected      int `json:"selected_observations"`
}

// Export is the exported analysis document: the source identity, the
// analysis controls and the selected observations and request errors,
// copied verbatim from the report.
type Export struct {
	Schema        string              `json:"schema"`
	Source        Source              `json:"source"`
	Threshold     float64             `json:"high_confidence_threshold"`
	Filter        Filter              `json:"filter"`
	Counts        Counts              `json:"counts"`
	Observations  []eval.Observation  `json:"observations"`
	RequestErrors []eval.RequestError `json:"request_errors"`
}

// NewExport builds the export document for a (report, analysis) pair.
func NewExport(r eval.Report, reportSHA256 string, a Analysis, f Filter) Export {
	x := Export{Schema: AnalysisSchema, Threshold: a.Threshold, Filter: f,
		Source: Source{Schema: r.Schema, ReportSHA256: reportSHA256, Dataset: r.Dataset, DatasetSHA256: r.DatasetSHA256,
			StartedAt: r.StartedAt},
		Counts: Counts{Correct: a.Correct, Wrong: a.Wrong, RequestErrors: a.RequestErrors, HighConfWrong: a.HighConfWrong,
			Selected: len(a.Items)},
		Observations: make([]eval.Observation, 0, len(a.Items)), RequestErrors: append([]eval.RequestError{}, a.Errors...)}
	if r.Served != nil {
		x.Source.ServedIdentity, x.Source.ServedModel = r.Served.Digest, r.Served.Model()
	}
	for _, it := range a.Items {
		x.Observations = append(x.Observations, it.Observation)
	}
	return x
}
