package dashboard

// Question Workbench: an interactive caller/operator surface that sends one
// state and one or more editable v1 choice questions through the existing
// POST /v1/decide contract of the resident runtime.
//
// The question editor keeps no state between requests: its whole editor
// travels in the form and every POST renders the page again from it. Questions
// are validated and compiled with internal/question and internal/api; only
// compiled api.Question values are sent. There are no expected labels here.
// The Clef requested capacity control edits the exact operator-owned capacity
// profile under HACHIDORI_HOME. Question Definition files are read or written
// only at local paths the operator typed, one file per action.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/worker"
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
	State                   string
	StateRef                string
	Questions               []wbQuestion
	LoadPath                string
	RequestedMaxInputTokens string
	Concurrency             string
	BatchWindowMS           string
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
	w := workbench{State: nl(r.PostFormValue("state")), LoadPath: strings.TrimSpace(r.PostFormValue("load_path")),
		RequestedMaxInputTokens: strings.TrimSpace(r.PostFormValue("requested_max_input_tokens")),
		StateRef:                strings.TrimSpace(r.PostFormValue("state_ref")),
		Concurrency:             strings.TrimSpace(r.PostFormValue("concurrency")),
		BatchWindowMS:           strings.TrimSpace(r.PostFormValue("batch_window_ms"))}
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
	Wire          string // the exact JSON body sent to POST /v1/decide
	Response      *api.DecideResponse
	Err           string
	Invalid       bool // the request failed v1 validation and was not sent
	Sent          bool
	TookMS        float64
	StateRef      string
	Concurrency   int
	BatchWindowMS *int
	Calls         []wbCall
	Results       map[string]api.Result
	Evidence      string
}

type wbCall struct {
	Wire     string
	Response *api.DecideResponse
	Err      string
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
	Token           string
	Endpoint        string
	W               workbench
	Capacity        *wbCapacity
	BatchWindow     *wbBatchWindow
	CapacityMessage string
	Run             *wbRun
	Preview         bool
	Export          *wbExport
	LoadMsg         string
	LoadErr         string
	FormError       string
	StateMessage    string
	PathPicker      bool
}

type wbBatchWindow struct {
	EffectiveMS    int
	MaxWindowMS    int
	MaxConcurrency int
}

type wbCapacity struct {
	SafeLimit       int
	EffectiveLimit  int
	Requested       string
	Editable        bool
	RestartRequired bool
	Notice          string
	homeRoot        string
	runningProfile  home.CapacityProfile
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
	if v.Run != nil {
		if res, ok := v.Run.Results[q.ID]; ok {
			qv.Result = &res
		} else if v.Run.Response != nil && i < len(v.Run.Response.Results) {
			res := v.Run.Response.Results[i]
			if res.ID == q.ID {
				qv.Result = &res
			}
		}
	}
	return qv
}

func (v wbView) StateReferenceMatches() bool {
	return v.W.StateRef == "" || home.StateRef(v.W.State) == v.W.StateRef
}

func (v wbView) MaxConcurrency() int {
	if v.BatchWindow == nil {
		return 1
	}
	return max(1, min(v.BatchWindow.MaxConcurrency, len(v.W.Questions)))
}

func (d *Dashboard) endpoint() *client.Client {
	return &client.Client{Endpoint: "http://" + d.cfg.APIAddr, HTTP: &http.Client{Timeout: 5 * time.Minute}}
}

func (d *Dashboard) workbenchView() wbView {
	v := wbView{Chrome: d.chrome("Workbench", "workbench"),
		Token: d.token, Endpoint: "http://" + d.cfg.APIAddr + "/v1/decide", PathPicker: d.cfg.PathPicker != nil,
		Capacity: d.workbenchCapacity()}
	v.BatchWindow = d.workbenchBatchWindow()
	return v
}

func (d *Dashboard) workbenchBatchWindow() *wbBatchWindow {
	if d.cfg.BatchWindow == nil || d.cfg.Status == nil {
		return nil
	}
	status := d.cfg.Status()
	if !status.Worker.Ready || status.Worker.Info["provider"] != "clef" || status.Worker.QueueLimit < 1 {
		return nil
	}
	return &wbBatchWindow{EffectiveMS: int(d.cfg.BatchWindow.BatchWindow() / time.Millisecond),
		MaxWindowMS: int(worker.MaxBatchWindow.Milliseconds()), MaxConcurrency: status.Worker.QueueLimit}
}

func (d *Dashboard) workbenchPage(w http.ResponseWriter, r *http.Request) {
	v := d.workbenchView()
	v.W = workbench{Questions: []wbQuestion{blankQuestion()}, Concurrency: "1"}
	if v.BatchWindow != nil {
		v.W.BatchWindowMS = strconv.Itoa(v.BatchWindow.EffectiveMS)
	}
	if v.Capacity != nil && v.Capacity.Editable {
		v.W.RequestedMaxInputTokens = v.Capacity.Requested
	}
	d.renderView(w, "workbench", v)
}

// workbenchPost applies one editor operation and renders the result. Capacity
// edits use their existing HACHIDORI_HOME profile; scheduler experiments use
// the live worker Policy and independent v1 decide requests.
func (d *Dashboard) workbenchPost(w http.ResponseWriter, r *http.Request) {
	v := d.workbenchView()
	v.W = parseWorkbench(r)
	if v.W.Concurrency == "" {
		v.W.Concurrency = "1"
	}
	if !r.PostForm.Has("batch_window_ms") && v.BatchWindow != nil {
		v.W.BatchWindowMS = strconv.Itoa(v.BatchWindow.EffectiveMS)
	}
	for i, q := range v.W.Questions {
		v.W.Questions[i] = q.withBlankRows()
	}
	op := r.PostFormValue("op")
	switch {
	case op == "run", op == "preview":
		v.Preview = op == "preview"
		v.Run = d.workbenchRun(r.Context(), &v.W, op == "run")
		if v.Run.Invalid {
			v.FormError = v.Run.Err
		}
	case op == "register-state":
		d.workbenchRegisterState(r.Context(), &v)
	case op == "save-capacity":
		if err := d.saveWorkbenchCapacity(v.W.RequestedMaxInputTokens); err != nil {
			v.FormError = err.Error()
		} else {
			v.CapacityMessage = "Requested input-token limit saved. Restart the runtime to apply it."
		}
		v.Capacity = d.workbenchCapacity()
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
	case op == "pick-load":
		d.workbenchPickLoad(r, &v)
	case strings.HasPrefix(op, "export:"):
		if i, ok := index(op, "export:", len(v.W.Questions)); ok {
			v.Export = exportDefinition(i, v.W.Questions[i])
		}
	case strings.HasPrefix(op, "pick-export:"):
		if i, ok := index(op, "pick-export:", len(v.W.Questions)); ok {
			d.workbenchPickExport(r, &v, i)
		}
	default:
		v.FormError = "unknown workbench operation"
	}
	v.BatchWindow = d.workbenchBatchWindow()
	if len(v.W.Questions) == 0 {
		v.W.Questions = []wbQuestion{blankQuestion()}
	}
	d.renderView(w, "workbench", v)
}

func (d *Dashboard) workbenchCapacity() *wbCapacity {
	if d.cfg.Status == nil {
		return nil
	}
	status := d.cfg.Status()
	if !status.Worker.Ready || status.Worker.Info["provider"] != "clef" {
		return nil
	}
	capacity, ok := status.Worker.Info["capacity"].(map[string]any)
	if !ok {
		return nil
	}
	profileValue := capacity["profile"]
	if profileValue == nil {
		return &wbCapacity{SafeLimit: statusInt(capacity["model_context_tokens"]),
			EffectiveLimit: statusInt(capacity["effective_max_input_tokens"])}
	}
	encoded, err := json.Marshal(profileValue)
	if err != nil {
		return nil
	}
	var running home.CapacityProfile
	if err := json.Unmarshal(encoded, &running); err != nil || running.Validate() != nil {
		return nil
	}
	view := &wbCapacity{SafeLimit: running.MaxInputTokens,
		EffectiveLimit: statusInt(capacity["effective_max_input_tokens"]),
		homeRoot:       status.Runtime.Home, runningProfile: running}
	if view.EffectiveLimit == 0 {
		view.EffectiveLimit = min(running.MaxInputTokens, statusInt(capacity["model_context_tokens"]))
		if running.RequestedMaxInputTokens > 0 {
			view.EffectiveLimit = min(view.EffectiveLimit, running.RequestedMaxInputTokens)
		}
	}
	h := home.Home{Root: status.Runtime.Home}
	profiles, err := h.LoadCapacityProfiles()
	if err != nil {
		view.Notice = "The running capacity profile cannot be edited because its saved configuration is unavailable."
		return view
	}
	saved, ok := profiles.Find(running.CapacityTarget)
	if !ok {
		view.Notice = "The running capacity profile cannot be edited because its saved target no longer matches."
		return view
	}
	view.Requested = requestedInputValue(saved.RequestedMaxInputTokens)
	withoutSavedRequest := saved
	withoutSavedRequest.RequestedMaxInputTokens = running.RequestedMaxInputTokens
	if withoutSavedRequest != running {
		view.Notice = "The saved safe capacity differs from the running worker. Restart the runtime before editing its requested limit."
		return view
	}
	view.Editable = true
	view.RestartRequired = saved.RequestedMaxInputTokens != running.RequestedMaxInputTokens
	return view
}

func (d *Dashboard) saveWorkbenchCapacity(value string) error {
	requested := 0
	if value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			return errors.New("requested input-token limit must be a positive integer or blank")
		}
		requested = parsed
	}
	view := d.workbenchCapacity()
	if view == nil || !view.Editable {
		return errors.New("a matching ready Clef capacity profile is required to edit the requested limit")
	}
	profile := view.runningProfile
	profile.RequestedMaxInputTokens = requested
	if err := (home.Home{Root: view.homeRoot}).SaveCapacityProfile(profile); err != nil {
		return fmt.Errorf("save requested input-token limit: %w", err)
	}
	return nil
}

func requestedInputValue(value int) string {
	if value <= 0 {
		return ""
	}
	return strconv.Itoa(value)
}

func statusInt(value any) int {
	switch n := value.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

func (d *Dashboard) workbenchPickLoad(r *http.Request, v *wbView) {
	if d.cfg.PathPicker == nil {
		v.FormError = "native path selection is unavailable"
		return
	}
	p, err := d.cfg.PathPicker.PickOpen(r.Context(), "Choose a Question Definition")
	if err == nil {
		v.W.LoadPath = p
	} else if !pickWasCancelled(err) {
		v.FormError = "choosing a Question Definition: " + err.Error()
	}
}

func (d *Dashboard) workbenchPickExport(r *http.Request, v *wbView, i int) {
	if d.cfg.PathPicker == nil {
		v.FormError = "native path selection is unavailable"
		return
	}
	p, err := d.cfg.PathPicker.PickSave(r.Context(), "Choose where to save the Question Definition")
	if err == nil {
		v.W.Questions[i].ExportPath = p
	} else if !pickWasCancelled(err) {
		v.FormError = "choosing an export destination: " + err.Error()
	}
}

func index(op, prefix string, n int) (int, bool) {
	i, err := strconv.Atoi(strings.TrimPrefix(op, prefix))
	return i, err == nil && i >= 0 && i < n
}

// workbenchRun validates the editor and experiment controls before changing
// the live window or sending any inference request. Multiple requests are
// independently sent to /v1/decide with balanced, contiguous groups of the
// Workbench's distinct questions so the worker can coalesce them without
// duplicate question IDs.
func (d *Dashboard) workbenchRun(ctx context.Context, wb *workbench, send bool) *wbRun {
	run := &wbRun{Results: map[string]api.Result{}}
	if send {
		// Keep the selected live window stable for the entire Workbench run.
		d.workbenchRunMu.Lock()
		defer d.workbenchRunMu.Unlock()
	}
	if wb.Concurrency == "" {
		wb.Concurrency = "1"
	}
	window := d.workbenchBatchWindow()
	concurrency, err := workbenchConcurrency(wb.Concurrency, len(wb.Questions), window)
	if err != nil {
		return invalidWorkbenchRun(run, err)
	}
	run.Concurrency = concurrency
	if concurrency > 1 && window == nil {
		return invalidWorkbenchRun(run, errors.New("same-State concurrency requires a ready Clef worker and its batch-window control"))
	}
	var requestedWindowMS *int
	if window != nil {
		ms, err := parseBatchWindowMS(wb.BatchWindowMS)
		if err != nil {
			return invalidWorkbenchRun(run, err)
		}
		requestedWindowMS = &ms
		run.BatchWindowMS = &ms
	} else if wb.BatchWindowMS != "" {
		return invalidWorkbenchRun(run, errors.New("batch-window control requires a ready Clef worker"))
	}

	base := wb.request()
	if wb.StateRef != "" {
		if home.StateRef(wb.State) != wb.StateRef {
			return invalidWorkbenchRun(run, errors.New("registered State reference does not match the current State; register it again"))
		}
		base.State, base.StateRef = "", wb.StateRef
	}
	run.StateRef = base.StateRef
	if err := base.Validate(); err != nil {
		return invalidWorkbenchRun(run, err)
	}
	stateRef := wb.StateRef
	if concurrency > 1 && stateRef == "" {
		stateRef = home.StateRef(wb.State)
	}
	if concurrency > 1 {
		run.StateRef = stateRef
	}
	requests := workbenchRequests(base, concurrency, stateRef)
	if len(requests) == 1 {
		wire, err := json.Marshal(requests[0])
		if err != nil {
			return invalidWorkbenchRun(run, err)
		}
		run.Wire = string(wire)
	} else {
		for _, req := range requests {
			wire, err := json.Marshal(req)
			if err != nil {
				return invalidWorkbenchRun(run, err)
			}
			run.Calls = append(run.Calls, wbCall{Wire: string(wire)})
		}
	}
	if !send {
		return run
	}

	apiClient := d.endpoint()
	if concurrency > 1 && wb.StateRef == "" {
		registered, err := apiClient.RegisterStateContext(ctx, api.RegisterState{Schema: api.SchemaV1, State: wb.State})
		if err != nil {
			run.Err = "register State: " + err.Error()
			return run
		}
		if registered.StateRef != stateRef {
			run.Err = "runtime returned a State reference that does not match the submitted State"
			return run
		}
		wb.StateRef = registered.StateRef
	}
	if requestedWindowMS != nil {
		if err := d.cfg.BatchWindow.SetBatchWindow(time.Duration(*requestedWindowMS) * time.Millisecond); err != nil {
			run.Err = err.Error()
			return run
		}
		if effective := int(d.cfg.BatchWindow.BatchWindow() / time.Millisecond); effective != *requestedWindowMS {
			run.Err = fmt.Sprintf("runtime applied %dms batch window, requested %dms", effective, *requestedWindowMS)
			return run
		}
		run.BatchWindowMS = requestedWindowMS
	}

	started := time.Now()
	run.Sent = true
	if len(requests) == 1 {
		call := workbenchCall(ctx, apiClient, requests[0], run.Wire)
		run.TookMS = call.TookMS
		run.Err = call.Err
		run.Response = call.Response
		if call.Response != nil {
			for _, result := range call.Response.Results {
				run.Results[result.ID] = result
			}
		}
	} else {
		run.Calls = make([]wbCall, len(requests))
		var wg sync.WaitGroup
		for i, req := range requests {
			wire, _ := json.Marshal(req) // the same typed request was marshaled above
			run.Calls[i].Wire = string(wire)
			wg.Add(1)
			go func(i int, req api.DecideRequest, wire string) {
				defer wg.Done()
				run.Calls[i] = workbenchCall(ctx, apiClient, req, wire)
			}(i, req, string(wire))
		}
		wg.Wait()
		run.TookMS = float64(time.Since(started).Microseconds()) / 1000
		failed := 0
		for _, call := range run.Calls {
			if call.Err != "" {
				failed++
				continue
			}
			if call.Response != nil {
				for _, result := range call.Response.Results {
					run.Results[result.ID] = result
				}
			}
		}
		if failed > 0 {
			run.Err = fmt.Sprintf("%d of %d independent decide requests failed", failed, len(run.Calls))
		}
	}
	run.Evidence = workbenchEvidence(run, requests, started, time.Now())
	return run
}

func workbenchConcurrency(raw string, questionCount int, window *wbBatchWindow) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n < 1 {
		return 0, errors.New("concurrency must be a whole number greater than zero")
	}
	if n > 1 && window == nil {
		return 0, errors.New("same-State concurrency requires a ready Clef worker and its batch-window control")
	}
	if window != nil && n > window.MaxConcurrency {
		return 0, fmt.Errorf("concurrency must be between 1 and the runtime queue limit of %d", window.MaxConcurrency)
	}
	if n > questionCount {
		return 0, fmt.Errorf("concurrency cannot exceed the %d distinct questions; add a unique question for each independent request", questionCount)
	}
	return n, nil
}

func parseBatchWindowMS(raw string) (int, error) {
	ms, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || ms < 0 || int64(ms) > worker.MaxBatchWindow.Milliseconds() {
		return 0, fmt.Errorf("batch window must be a whole number of milliseconds from 0 to %d", worker.MaxBatchWindow.Milliseconds())
	}
	return ms, nil
}

func invalidWorkbenchRun(run *wbRun, err error) *wbRun {
	run.Err, run.Invalid = err.Error(), true
	return run
}

func workbenchRequests(base api.DecideRequest, concurrency int, stateRef string) []api.DecideRequest {
	if concurrency == 1 {
		return []api.DecideRequest{base}
	}
	requests := make([]api.DecideRequest, concurrency)
	chunkSize, remainder := len(base.Questions)/concurrency, len(base.Questions)%concurrency
	start := 0
	for i := range requests {
		size := chunkSize
		if i < remainder {
			size++
		}
		end := start + size
		requests[i] = api.DecideRequest{Schema: api.SchemaV1, StateRef: stateRef,
			Questions: append([]api.Question(nil), base.Questions[start:end]...)}
		start = end
	}
	return requests
}

func workbenchCall(ctx context.Context, c *client.Client, req api.DecideRequest, wire string) wbCall {
	started := time.Now()
	resp, err := c.DecideContext(ctx, req)
	call := wbCall{Wire: wire, TookMS: float64(time.Since(started).Microseconds()) / 1000}
	if err != nil {
		call.Err = err.Error()
		return call
	}
	call.Response = &resp
	return call
}

func workbenchEvidence(run *wbRun, requests []api.DecideRequest, started, finished time.Time) string {
	type requestEvidence struct {
		Request  json.RawMessage     `json:"request"`
		Response *api.DecideResponse `json:"response,omitempty"`
		Error    string              `json:"error,omitempty"`
		TookMS   float64             `json:"round_trip_ms"`
	}
	evidence := struct {
		Concurrency   int               `json:"concurrency"`
		BatchWindowMS *int              `json:"batch_window_ms,omitempty"`
		StateRef      string            `json:"state_ref,omitempty"`
		StartedAt     time.Time         `json:"started_at"`
		FinishedAt    time.Time         `json:"finished_at"`
		Requests      []requestEvidence `json:"requests"`
	}{Concurrency: run.Concurrency, BatchWindowMS: run.BatchWindowMS, StateRef: run.StateRef,
		StartedAt: started.UTC(), FinishedAt: finished.UTC(), Requests: make([]requestEvidence, len(requests))}
	for i, req := range requests {
		wire, _ := json.Marshal(req)
		evidence.Requests[i].Request = wire
		if i < len(run.Calls) {
			evidence.Requests[i].Response = run.Calls[i].Response
			evidence.Requests[i].Error = run.Calls[i].Err
			evidence.Requests[i].TookMS = run.Calls[i].TookMS
		} else if len(requests) == 1 {
			evidence.Requests[i].Response = run.Response
			evidence.Requests[i].Error = run.Err
			evidence.Requests[i].TookMS = run.TookMS
		}
	}
	data, err := json.MarshalIndent(evidence, "", "  ")
	if err != nil {
		return ""
	}
	return string(data)
}

func (d *Dashboard) workbenchRegisterState(ctx context.Context, view *wbView) {
	if strings.TrimSpace(view.W.State) == "" {
		view.FormError = "State must not be blank"
		return
	}
	if len(view.W.State) > api.MaxStateBytes {
		view.FormError = fmt.Sprintf("state exceeds %d bytes", api.MaxStateBytes)
		return
	}
	registered, err := d.endpoint().RegisterStateContext(ctx, api.RegisterState{Schema: api.SchemaV1, State: view.W.State})
	if err != nil {
		view.FormError = "register State: " + err.Error()
		return
	}
	view.W.StateRef = registered.StateRef
	view.StateMessage = "State registered as " + registered.StateRef
}

// absPath requires an explicit absolute local path: the dashboard never
// resolves paths against its own working directory. The cleaned path may not
// contain "..", so a path always names exactly the location that was typed.
func absPath(p, what string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("enter the %s path", what)
	}
	if !filepath.IsAbs(p) {
		return "", fmt.Errorf("%s path must be absolute, got %q", what, p)
	}
	p = filepath.Clean(p)
	if strings.Contains(p, "..") {
		return "", fmt.Errorf("%s path must not contain \"..\", got %q", what, p)
	}
	return p, nil
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
