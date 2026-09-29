package dashboard

// Question Workbench: an interactive caller/operator surface that sends one
// state and one or more editable v1 choice questions through the existing
// POST /v1/decide contract of the resident runtime.
//
// The workbench keeps no state between requests: the whole editor travels in
// the form and every POST renders the page again from it. Questions are
// validated and compiled with internal/question and internal/api; only the
// compiled api.Question values are sent. There are no expected labels here.
// Question Definition files are read or written only at local paths the
// operator typed, one file per action; nothing is scanned or stored.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/question"
)

// workbenchMaxBody bounds a workbench POST: a maximal v1 request (64 KiB of
// state, 32 questions of up to 4 KiB instructions and 64 choices) form-encoded,
// with room for the editor's own fields.
const workbenchMaxBody = 4 << 20

// maxDefinitionFile bounds a Question Definition file read by the workbench.
const maxDefinitionFile = 1 << 20

// blankChoiceRows is how many empty choice rows the editor offers per
// question; "Add choice rows" adds this many more.
const blankChoiceRows = 2

// wbChoice is one editable choice row. A row with an empty label and an empty
// description is unused and ignored.
type wbChoice struct {
	Label string
	Desc  string
}

// wbQuestion is one editable question.
type wbQuestion struct {
	ID           string
	Version      string // definition version, used only for export
	Instructions string
	Choices      []wbChoice
	ExportPath   string
	// Loaded is the identity of the definition file projected into this
	// slot, when the question came from one.
	Loaded *question.Identity
}

// workbench is the complete editor state.
type workbench struct {
	State     string
	Questions []wbQuestion
	LoadPath  string
}

// definition is the Question Definition the editor row describes. Choice
// order is preserved; empty descriptions are omitted.
func (q wbQuestion) definition() (question.Definition, error) {
	d := question.Definition{Schema: question.Schema, ID: q.ID, Type: "choice", Instructions: q.Instructions}
	for _, c := range q.Choices {
		if c.Label == "" && c.Desc == "" {
			continue
		}
		d.Choices = append(d.Choices, c.Label)
		if c.Desc != "" {
			if d.Descriptions == nil {
				d.Descriptions = map[string]string{}
			}
			d.Descriptions[c.Label] = c.Desc
		}
	}
	v, err := strconv.Atoi(strings.TrimSpace(q.Version))
	if err != nil {
		return d, fmt.Errorf("question %q: version must be an integer >= 1", q.ID)
	}
	d.Version = v
	return d, nil
}

// request compiles the editor into the v1 decide request exactly as it is
// sent: only compiled api.Question values, no definition identity or label.
func (w workbench) request() api.DecideRequest {
	req := api.DecideRequest{Schema: api.SchemaV1, State: w.State, Questions: []api.Question{}}
	for _, q := range w.Questions {
		d, _ := q.definition() // the version is not part of the request
		req.Questions = append(req.Questions, d.Compile())
	}
	return req
}

// project turns a Question Definition into an editor row, keeping its
// identity so the page can show whether the row still matches it.
func project(d question.Definition) wbQuestion {
	q := wbQuestion{ID: d.ID, Version: strconv.Itoa(d.Version), Instructions: d.Instructions}
	for _, c := range d.Choices {
		q.Choices = append(q.Choices, wbChoice{Label: c, Desc: d.Descriptions[c]})
	}
	id := d.Identity()
	q.Loaded = &id
	return q
}

func blankQuestion() wbQuestion {
	return wbQuestion{Version: "1", Choices: make([]wbChoice, blankChoiceRows)}
}

// withBlankRows returns q's rows plus the unused rows the editor shows.
func (q wbQuestion) withBlankRows() wbQuestion {
	used := q.Choices[:0:0]
	for _, c := range q.Choices {
		if c.Label != "" || c.Desc != "" {
			used = append(used, c)
		}
	}
	q.Choices = append(used, make([]wbChoice, blankChoiceRows)...)
	return q
}

// nl undoes the CRLF line breaks browsers submit for textarea values, so the
// request carries the text the operator typed on every platform.
func nl(s string) string { return strings.ReplaceAll(s, "\r\n", "\n") }

// parseWorkbench reads the editor state from a posted form. Counts are
// bounded by the v1 limits so a crafted form cannot allocate unboundedly.
func parseWorkbench(r *http.Request) workbench {
	w := workbench{State: nl(r.PostFormValue("state")), LoadPath: strings.TrimSpace(r.PostFormValue("load_path"))}
	nq := bounded(r.PostFormValue("nq"), api.MaxQuestions)
	for i := 0; i < nq; i++ {
		p := "q" + strconv.Itoa(i) + "."
		q := wbQuestion{ID: r.PostFormValue(p + "id"), Version: r.PostFormValue(p + "version"),
			Instructions: nl(r.PostFormValue(p + "instructions")), ExportPath: strings.TrimSpace(r.PostFormValue(p + "export_path"))}
		nc := bounded(r.PostFormValue(p+"nc"), api.MaxChoices+blankChoiceRows)
		for j := 0; j < nc; j++ {
			c := p + "c" + strconv.Itoa(j) + "."
			q.Choices = append(q.Choices, wbChoice{Label: r.PostFormValue(c + "label"), Desc: nl(r.PostFormValue(c + "desc"))})
		}
		if id := r.PostFormValue(p + "loaded_id"); id != "" {
			v, _ := strconv.Atoi(r.PostFormValue(p + "loaded_version"))
			q.Loaded = &question.Identity{ID: id, Version: v, Digest: r.PostFormValue(p + "loaded_digest")}
		}
		w.Questions = append(w.Questions, q)
	}
	return w
}

func bounded(s string, hi int) int {
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return 0
	}
	return min(n, hi)
}

// wbRun is the outcome of one inference run from the workbench.
type wbRun struct {
	Wire     string // the exact JSON body sent to POST /v1/decide
	Response *api.DecideResponse
	Err      string
	Invalid  bool // the request failed v1 validation and was not sent
	Sent     bool
	TookMS   float64
}

// wbExport is the outcome of exporting one question.
type wbExport struct {
	Index    int
	Path     string
	Identity question.Identity
	JSON     string
	Err      string
}

// wbView is the workbench page's view model.
type wbView struct {
	Chrome
	Token     string
	Endpoint  string
	W         workbench
	Run       *wbRun
	Preview   bool
	Export    *wbExport
	LoadMsg   string
	LoadErr   string
	FormError string
}

// wbQuestionView is what the page shows for one editor row besides its
// fields: the definition identity it would export as, and the result of the
// last run for it.
type wbQuestionView struct {
	Identity *question.Identity
	DefErr   string
	Modified bool // differs from the loaded definition
	Result   *api.Result
}

func (v wbView) Question(i int) wbQuestionView {
	q := v.W.Questions[i]
	var qv wbQuestionView
	d, err := q.definition()
	if err == nil {
		err = d.Validate()
	}
	if err != nil {
		qv.DefErr = err.Error()
	} else {
		id := d.Identity()
		qv.Identity = &id
	}
	qv.Modified = q.Loaded != nil && (qv.Identity == nil || *qv.Identity != *q.Loaded)
	if v.Run != nil && v.Run.Response != nil && i < len(v.Run.Response.Results) {
		res := v.Run.Response.Results[i]
		if res.ID == q.ID {
			qv.Result = &res
		}
	}
	return qv
}

func (d *Dashboard) endpoint() *client.Client {
	return &client.Client{Endpoint: "http://" + d.cfg.APIAddr, HTTP: &http.Client{Timeout: 5 * time.Minute}}
}

func (d *Dashboard) workbenchView() wbView {
	return wbView{Chrome: Chrome{Title: "Question Workbench", Nav: "workbench", APIAddr: d.cfg.APIAddr},
		Token: d.token, Endpoint: "http://" + d.cfg.APIAddr + "/v1/decide"}
}

func (d *Dashboard) workbenchPage(w http.ResponseWriter, r *http.Request) {
	v := d.workbenchView()
	v.W = workbench{Questions: []wbQuestion{blankQuestion()}}
	d.renderView(w, "workbench", v)
}

// workbenchPost applies one editor operation and renders the result. None of
// the operations change runtime state; "run" is one POST /v1/decide.
func (d *Dashboard) workbenchPost(w http.ResponseWriter, r *http.Request) {
	v := d.workbenchView()
	v.W = parseWorkbench(r)
	for i, q := range v.W.Questions {
		v.W.Questions[i] = q.withBlankRows()
	}
	op := r.PostFormValue("op")
	switch {
	case op == "run", op == "preview":
		v.Preview = op == "preview"
		v.Run = d.workbenchRun(v.W, op == "run")
	case op == "add":
		if len(v.W.Questions) >= api.MaxQuestions {
			v.FormError = fmt.Sprintf("a request carries at most %d questions", api.MaxQuestions)
		} else {
			v.W.Questions = append(v.W.Questions, blankQuestion())
		}
	case strings.HasPrefix(op, "remove:"):
		if i, ok := index(op, "remove:", len(v.W.Questions)); ok {
			v.W.Questions = append(v.W.Questions[:i], v.W.Questions[i+1:]...)
		}
	case strings.HasPrefix(op, "choices:"):
		if i, ok := index(op, "choices:", len(v.W.Questions)); ok {
			q := &v.W.Questions[i]
			q.Choices = append(q.Choices, make([]wbChoice, blankChoiceRows)...)
		}
	case op == "load":
		d.workbenchLoad(&v)
	case strings.HasPrefix(op, "export:"):
		if i, ok := index(op, "export:", len(v.W.Questions)); ok {
			v.Export = exportDefinition(i, v.W.Questions[i])
		}
	default:
		v.FormError = "unknown workbench operation"
	}
	if len(v.W.Questions) == 0 {
		v.W.Questions = []wbQuestion{blankQuestion()}
	}
	d.renderView(w, "workbench", v)
}

func index(op, prefix string, n int) (int, bool) {
	i, err := strconv.Atoi(strings.TrimPrefix(op, prefix))
	return i, err == nil && i >= 0 && i < n
}

// workbenchRun validates the compiled request with the v1 contract and, when
// send is set and it is valid, sends it once through the existing endpoint.
func (d *Dashboard) workbenchRun(wb workbench, send bool) *wbRun {
	req := wb.request()
	run := &wbRun{}
	b, err := json.Marshal(req) // byte-for-byte what client.Decide sends
	if err != nil {
		run.Err = err.Error()
		return run
	}
	run.Wire = string(b)
	if err := req.Validate(); err != nil {
		run.Err, run.Invalid = err.Error(), true
		return run
	}
	if !send {
		return run
	}
	t0 := time.Now()
	resp, err := d.endpoint().Decide(req)
	run.Sent, run.TookMS = true, float64(time.Since(t0).Microseconds())/1000
	if err != nil {
		run.Err = err.Error()
		return run
	}
	run.Response = &resp
	return run
}

// absPath requires an explicit absolute local path: the dashboard never
// resolves paths against its own working directory.
func absPath(p, what string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("enter the %s path", what)
	}
	if !filepath.IsAbs(p) {
		return "", fmt.Errorf("%s path must be absolute, got %q", what, p)
	}
	return filepath.Clean(p), nil
}

// workbenchLoad reads one Question Definition file named by the operator,
// validates it with question.Parse and projects it into the editor: it
// replaces the only question when that one is still empty, else it is added.
func (d *Dashboard) workbenchLoad(v *wbView) {
	def, err := readDefinition(v.W.LoadPath)
	if err != nil {
		v.LoadErr = err.Error()
		return
	}
	q := project(def).withBlankRows()
	switch {
	case len(v.W.Questions) == 1 && emptyQuestion(v.W.Questions[0]):
		v.W.Questions[0] = q
	case len(v.W.Questions) >= api.MaxQuestions:
		v.LoadErr = fmt.Sprintf("a request carries at most %d questions; remove one first", api.MaxQuestions)
		return
	default:
		v.W.Questions = append(v.W.Questions, q)
	}
	v.LoadMsg = fmt.Sprintf("loaded %s (%s) from %s", q.Loaded, q.Loaded.Digest, v.W.LoadPath)
}

func readDefinition(path string) (question.Definition, error) {
	p, err := absPath(path, "Question Definition file")
	if err != nil {
		return question.Definition{}, err
	}
	f, err := os.Open(p)
	if err != nil {
		return question.Definition{}, err
	}
	defer f.Close()
	if fi, err := f.Stat(); err != nil {
		return question.Definition{}, err
	} else if !fi.Mode().IsRegular() {
		return question.Definition{}, fmt.Errorf("%s is not a regular file; choose one Question Definition file", p)
	}
	data, err := io.ReadAll(io.LimitReader(f, maxDefinitionFile+1))
	if err != nil {
		return question.Definition{}, err
	}
	if len(data) > maxDefinitionFile {
		return question.Definition{}, fmt.Errorf("%s: larger than %d bytes", p, maxDefinitionFile)
	}
	def, err := question.Parse(data)
	if err != nil {
		return question.Definition{}, fmt.Errorf("%s: %w", p, err)
	}
	return def, nil
}

func emptyQuestion(q wbQuestion) bool {
	if q.ID != "" || q.Instructions != "" || q.Loaded != nil {
		return false
	}
	for _, c := range q.Choices {
		if c.Label != "" || c.Desc != "" {
			return false
		}
	}
	return true
}

// exportDefinition writes one edited question as a hachidori.question.v1
// document to a new file. The written bytes are parsed back so the reported
// identity is the one any consumer (question.Load) will compute.
func exportDefinition(i int, q wbQuestion) *wbExport {
	ex := &wbExport{Index: i, Path: q.ExportPath}
	fail := func(err error) *wbExport { ex.Err = err.Error(); return ex }
	def, err := q.definition()
	if err != nil {
		return fail(err)
	}
	if err := def.Validate(); err != nil {
		return fail(err)
	}
	data, err := definitionJSON(def)
	if err != nil {
		return fail(err)
	}
	p, err := absPath(q.ExportPath, "export")
	if err != nil {
		return fail(err)
	}
	if err := writeNew(p, data); err != nil {
		return fail(err)
	}
	ex.Path, ex.Identity, ex.JSON = p, def.Identity(), string(data)
	return ex
}

// definitionJSON is the exported file form: the definition's fields in their
// fixed order, indented. Its identity is verified to survive a round trip
// through question.Parse.
func definitionJSON(def question.Definition) ([]byte, error) {
	b, err := json.MarshalIndent(def, "", "  ")
	if err != nil {
		return nil, err
	}
	b = append(b, '\n')
	back, err := question.Parse(b)
	if err != nil {
		return nil, err
	}
	if back.Identity() != def.Identity() {
		return nil, errors.New("exported definition does not round-trip to the same identity")
	}
	return b, nil
}

// writeNew creates path and writes data. It never replaces an existing file.
func writeNew(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("%s already exists; exports never overwrite a file", path)
		}
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(path)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(path)
		return err
	}
	return nil
}
