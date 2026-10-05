package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// fakeAPI stands in for the resident runtime's inference API. It records
// every request it receives.
type fakeAPI struct {
	mu     sync.Mutex
	bodies [][]byte
	paths  []string
	reply  func(req api.DecideRequest) (int, any)
	// status serves GET /v1/status (n counts status calls); "" fails.
	status func(n int) string
	nstat  int
	// block, when set, holds every decide request until it is closed or
	// the request is abandoned.
	block            chan struct{}
	activeDecides    int
	maxActiveDecides int
}

func statusDoc(model string, uptime int) string {
	return fmt.Sprintf(`{"schema":"hachidori.v1","runtime":{"home":"/h","runtime":"cpu-abc","model_id":"laya-base","model":%q,"device":"cpu"},`+
		`"uptime_s":%d,"worker":{"state":"ready","ready":true,"pid":42,"starts":1,`+
		`"provider":{"provider":"laya","laya_version":"0.3.21","device":"cpu","load_ms":100,"warmup_ms":3}}}`, model, uptime)
}

func (f *fakeAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	switch r.URL.Path {
	case "/health":
		f.mu.Unlock()
		w.Write([]byte(`{"ready":true,"state":"ready"}`))
		return
	case "/v1/status":
		n := f.nstat
		f.nstat++
		st := f.status
		f.mu.Unlock()
		doc := statusDoc("org/laya@rev1", 10+n)
		if st != nil {
			doc = st(n)
		}
		if doc == "" {
			http.Error(w, "down", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(doc))
		return
	case "/v1/states":
		f.bodies = append(f.bodies, b)
		f.paths = append(f.paths, r.Method+" "+r.URL.Path)
		f.mu.Unlock()
		var input api.RegisterState
		_ = json.Unmarshal(b, &input)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.StateReference{Schema: api.SchemaV1, StateRef: home.StateRef(input.State)})
		return
	}
	f.bodies = append(f.bodies, b)
	f.paths = append(f.paths, r.Method+" "+r.URL.Path)
	reply, block := f.reply, f.block
	if r.URL.Path == "/v1/decide" {
		f.activeDecides++
		f.maxActiveDecides = max(f.maxActiveDecides, f.activeDecides)
	}
	f.mu.Unlock()
	if r.URL.Path == "/v1/decide" {
		defer func() {
			f.mu.Lock()
			f.activeDecides--
			f.mu.Unlock()
		}()
	}
	if block != nil {
		select {
		case <-block:
		case <-r.Context().Done():
			return
		}
	}
	var req api.DecideRequest
	_ = json.Unmarshal(b, &req)
	code, body := http.StatusOK, any(nil)
	if reply != nil {
		code, body = reply(req)
	} else {
		body = echo(req)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}

func (f *fakeAPI) calls() ([]string, [][]byte) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.paths...), append([][]byte(nil), f.bodies...)
}

// echo answers every question with its first choice at 0.8.
func echo(req api.DecideRequest) api.DecideResponse {
	resp := api.DecideResponse{Schema: api.SchemaV1, Timing: &api.Timing{InferenceMS: 12.5, TotalMS: 14}}
	for _, q := range req.Questions {
		p := map[string]float64{}
		for i, c := range q.Choices {
			p[c] = 0.2 / float64(len(q.Choices)-1)
			if i == 0 {
				p[c] = 0.8
			}
		}
		resp.Results = append(resp.Results, api.Result{ID: q.ID, Type: q.Type, Choice: q.Choices[0], Confidence: 0.8, Probabilities: p})
	}
	return resp
}

// newWorkbenchEnv is newEnv with the dashboard's inference API address
// pointing at a fake API server.
func newWorkbenchEnv(t testing.TB) (*env, *fakeAPI) {
	t.Helper()
	e := newEnv(t)
	f := &fakeAPI{}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	cfg := e.d.cfg
	cfg.APIAddr = srv.Listener.Addr().String()
	e.d = New(cfg)
	return e, f
}

type fakeBatchWindowControl struct {
	mu     sync.Mutex
	window time.Duration
	sets   []time.Duration
}

func (c *fakeBatchWindowControl) BatchWindow() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.window
}

func (c *fakeBatchWindowControl) SetBatchWindow(window time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if window < 0 || window > worker.MaxBatchWindow || window%time.Millisecond != 0 {
		return fmt.Errorf("batch window out of range")
	}
	c.window = window
	c.sets = append(c.sets, window)
	return nil
}

func enableWorkbenchScheduler(t testing.TB, e *env, window time.Duration, queueLimit int) *fakeBatchWindowControl {
	t.Helper()
	control := &fakeBatchWindowControl{window: window}
	cfg := e.d.cfg
	previous := cfg.Status
	cfg.Status = func() server.Status {
		status := previous()
		status.Worker.State = worker.StateReady
		status.Worker.Ready = true
		status.Worker.QueueLimit = queueLimit
		status.Worker.Info = worker.Info{"provider": "clef"}
		return status
	}
	cfg.BatchWindow = control
	e.d = New(cfg)
	return control
}

func TestWorkbenchCanRegisterStateReference(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	state := "shared experiment State"
	body := e.post(t, "/workbench", with(form(state, []string{"q", "Pick", "a", "b"}), "op", "register-state")).Body.String()
	if !strings.Contains(body, home.StateRef(state)) || !strings.Contains(body, "State registered as") {
		t.Fatalf("registered reference not shown: %s", body)
	}
	paths, _ := f.calls()
	if !reflect.DeepEqual(paths, []string{"POST /v1/states"}) {
		t.Fatalf("registration calls %v", paths)
	}
}

func TestWorkbenchSameStateConcurrencyUsesIndependentDecideRequests(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	control := enableWorkbenchScheduler(t, e, 10*time.Millisecond, 4)
	barrier := make(chan struct{})
	f.block = barrier
	questions := make([][]string, 6)
	for i := range questions {
		questions[i] = []string{fmt.Sprintf("q%d", i+1), fmt.Sprintf("Question %d?", i+1), "yes", "no"}
	}
	values := with(form("shared State", questions...), "op", "run", "concurrency", "3", "batch_window_ms", "25")
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- e.post(t, "/workbench", values) }()

	deadline := time.Now().Add(3 * time.Second)
	for {
		f.mu.Lock()
		active := f.activeDecides
		f.mu.Unlock()
		if active == 3 {
			break
		}
		if time.Now().After(deadline) {
			close(barrier)
			t.Fatalf("independent decide calls did not overlap; active=%d", active)
		}
		time.Sleep(time.Millisecond)
	}
	close(barrier)
	rec := <-done
	if rec.Code != http.StatusOK {
		t.Fatalf("run: %d %s", rec.Code, rec.Body)
	}
	paths, bodies := f.calls()
	if len(paths) != 4 || paths[0] != "POST /v1/states" || paths[1] != "POST /v1/decide" || paths[2] != "POST /v1/decide" || paths[3] != "POST /v1/decide" {
		t.Fatalf("experiment endpoints %v", paths)
	}
	f.mu.Lock()
	maxActive := f.maxActiveDecides
	f.mu.Unlock()
	if maxActive != 3 {
		t.Fatalf("peak independent requests=%d, want 3", maxActive)
	}
	wantGroups := map[string][]string{"q1": {"q1", "q2"}, "q3": {"q3", "q4"}, "q5": {"q5", "q6"}}
	gotGroups := make(map[string][]string)
	for i, body := range bodies[1:] {
		var req api.DecideRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatal(err)
		}
		if req.Schema != api.SchemaV1 || req.State != "" || req.StateRef != home.StateRef("shared State") || len(req.Questions) != 2 {
			t.Fatalf("request %d was not an independent same-ref request with a two-question chunk: %+v", i, req)
		}
		ids := []string{req.Questions[0].ID, req.Questions[1].ID}
		if _, exists := gotGroups[ids[0]]; exists {
			t.Fatalf("duplicate independent request group beginning with %q", ids[0])
		}
		gotGroups[ids[0]] = ids
	}
	if !reflect.DeepEqual(gotGroups, wantGroups) {
		t.Fatalf("request question groups = %v, want contiguous input-order groups %v", gotGroups, wantGroups)
	}
	if control.BatchWindow() != 25*time.Millisecond {
		t.Fatalf("effective window=%s", control.BatchWindow())
	}
	page := html.UnescapeString(rec.Body.String())
	for _, want := range []string{"3 independent requests", "effective batch window 25 ms", "Experiment evidence (JSON)", `"concurrency": 3`, `"batch_window_ms": 25`, home.StateRef("shared State")} {
		if !strings.Contains(page, want) {
			t.Errorf("experiment evidence lacks %q", want)
		}
	}
	last := -1
	for i := range questions {
		marker := fmt.Sprintf(`aria-label="Result · Question %d"`, i)
		at := strings.Index(page, marker)
		if at < 0 || at <= last {
			t.Fatalf("rendered results do not follow input order at %q", marker)
		}
		last = at
	}
}

func TestWorkbenchRequestsUseContiguousBalancedQuestionChunks(t *testing.T) {
	for _, tc := range []struct {
		name  string
		count int
		want  [][]string
	}{
		{name: "even", count: 6, want: [][]string{{"q1", "q2"}, {"q3", "q4"}, {"q5", "q6"}}},
		{name: "remainder goes to first groups", count: 5, want: [][]string{{"q1", "q2"}, {"q3", "q4"}, {"q5"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := api.DecideRequest{Schema: api.SchemaV1, State: "state"}
			for i := 1; i <= tc.count; i++ {
				base.Questions = append(base.Questions, api.Question{ID: fmt.Sprintf("q%d", i), Instructions: fmt.Sprintf("question %d", i)})
			}
			requests := workbenchRequests(base, 3, "registered-state-ref")
			if len(requests) != len(tc.want) {
				t.Fatalf("request count=%d, want %d", len(requests), len(tc.want))
			}
			for i, req := range requests {
				if req.Schema != api.SchemaV1 || req.State != "" || req.StateRef != "registered-state-ref" {
					t.Errorf("request %d has wrong shared State identity: %+v", i, req)
				}
				var ids []string
				for _, q := range req.Questions {
					ids = append(ids, q.ID)
				}
				if !reflect.DeepEqual(ids, tc.want[i]) {
					t.Errorf("request %d question IDs=%v, want %v", i, ids, tc.want[i])
				}
			}
		})
	}
}

func TestWorkbenchRejectsExperimentControlsBeforeExecution(t *testing.T) {
	for _, tc := range []struct {
		name        string
		concurrency string
		window      string
		questions   [][]string
		want        string
	}{
		{name: "invalid concurrency", concurrency: "two", window: "10", questions: [][]string{{"q", "Pick", "a", "b"}}, want: "concurrency must be a whole number"},
		{name: "too many independent requests", concurrency: "2", window: "10", questions: [][]string{{"q", "Pick", "a", "b"}}, want: "cannot exceed the 1 distinct questions"},
		{name: "no questions", concurrency: "1", window: "10", want: "cannot exceed the 0 distinct questions"},
		{name: "window over maximum", concurrency: "1", window: "1001", questions: [][]string{{"q", "Pick", "a", "b"}}, want: "from 0 to 1000"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, f := newWorkbenchEnv(t)
			control := enableWorkbenchScheduler(t, e, 10*time.Millisecond, 4)
			v := form("state", tc.questions...)
			v.Set("op", "run")
			v.Set("concurrency", tc.concurrency)
			v.Set("batch_window_ms", tc.window)
			body := e.post(t, "/workbench", v).Body.String()
			if !strings.Contains(body, html.EscapeString(tc.want)) || !strings.Contains(body, "invalid · not sent") {
				t.Fatalf("validation message missing %q: %s", tc.want, body)
			}
			if paths, _ := f.calls(); len(paths) != 0 {
				t.Fatalf("invalid controls reached the API: %v", paths)
			}
			if control.BatchWindow() != 10*time.Millisecond || len(control.sets) != 0 {
				t.Fatalf("invalid controls changed effective window: %s", control.BatchWindow())
			}
		})
	}
}

func TestWorkbenchConcurrentRequestErrorsRemainIsolated(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	enableWorkbenchScheduler(t, e, 20*time.Millisecond, 4)
	f.reply = func(req api.DecideRequest) (int, any) {
		if req.Questions[0].ID == "bad" {
			return http.StatusFailedDependency, api.ErrorBody{Schema: api.SchemaV1,
				Error: api.ErrorInfo{Class: api.ErrInferenceFailed, Message: "one request failed"}}
		}
		return http.StatusOK, echo(req)
	}
	v := with(form("state", []string{"good", "Pick", "yes", "no"}, []string{"bad", "Pick", "yes", "no"}),
		"op", "run", "concurrency", "2", "batch_window_ms", "20")
	body := e.post(t, "/workbench", v).Body.String()
	for _, want := range []string{"1 of 2 independent decide requests failed", "one request failed", "Question 0", "yes", `"error": "inference_failed`} {
		if !strings.Contains(html.UnescapeString(body), want) {
			t.Errorf("independent result evidence lacks %q", want)
		}
	}
	paths, _ := f.calls()
	if len(paths) != 3 || paths[0] != "POST /v1/states" || paths[1] != "POST /v1/decide" || paths[2] != "POST /v1/decide" {
		t.Fatalf("failed request changed the independent request path: %v", paths)
	}
}

func TestWorkbenchSameStateConcurrentRequestsPropagateCancellation(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	enableWorkbenchScheduler(t, e, 20*time.Millisecond, 4)
	f.block = make(chan struct{})
	v := with(form("state", []string{"q1", "Pick one", "yes", "no"}, []string{"q2", "Pick two", "yes", "no"}),
		"op", "run", "concurrency", "2", "batch_window_ms", "20")
	v.Set("token", e.d.token)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest("POST", "http://127.0.0.1:7844/workbench", strings.NewReader(v.Encode())).WithContext(ctx)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", "http://127.0.0.1:7844")
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		e.d.ServeHTTP(rec, req)
		close(done)
	}()

	deadline := time.Now().Add(3 * time.Second)
	for {
		f.mu.Lock()
		active := f.activeDecides
		f.mu.Unlock()
		if active == 2 {
			break
		}
		if time.Now().After(deadline) {
			cancel()
			<-done
			t.Fatalf("concurrent decide requests did not start before cancellation; active=%d", active)
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Workbench did not return after its request context was canceled")
	}
	body := html.UnescapeString(rec.Body.String())
	if !strings.Contains(body, "2 of 2 independent decide requests failed") {
		t.Fatalf("cancellation was not isolated and reported for both requests: %s", body)
	}
	paths, _ := f.calls()
	if len(paths) != 3 || paths[0] != "POST /v1/states" || paths[1] != "POST /v1/decide" || paths[2] != "POST /v1/decide" {
		t.Fatalf("canceled experiment path %v", paths)
	}
}

func TestWorkbenchCanceledRunWaitingForGateHasNoEffects(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	control := enableWorkbenchScheduler(t, e, 10*time.Millisecond, 4)
	barrier := make(chan struct{})
	f.block = barrier
	var release sync.Once
	releaseBarrier := func() { release.Do(func() { close(barrier) }) }
	t.Cleanup(releaseBarrier)
	values := with(form("first state", []string{"q1", "Pick one", "yes", "no"}, []string{"q2", "Pick two", "yes", "no"}),
		"op", "run", "concurrency", "2", "batch_window_ms", "20")
	values.Set("token", e.d.token)
	secondValues := url.Values{}
	for key, value := range values {
		secondValues[key] = append([]string(nil), value...)
	}
	secondValues.Set("state", "second state")
	firstDone := make(chan *httptest.ResponseRecorder, 1)
	go func() { firstDone <- e.post(t, "/workbench", values) }()

	deadline := time.Now().Add(3 * time.Second)
	for {
		f.mu.Lock()
		active := f.activeDecides
		f.mu.Unlock()
		if active == 2 {
			break
		}
		if time.Now().After(deadline) {
			releaseBarrier()
			<-firstDone
			t.Fatalf("first run did not hold the gate with both decide calls active; active=%d", active)
		}
		time.Sleep(time.Millisecond)
	}

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "http://127.0.0.1:7844/workbench", strings.NewReader(secondValues.Encode())).WithContext(ctx)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", "http://127.0.0.1:7844")
	rec := httptest.NewRecorder()
	secondStarted, secondDone := make(chan struct{}), make(chan struct{})
	go func() {
		close(secondStarted)
		e.d.ServeHTTP(rec, req)
		close(secondDone)
	}()
	<-secondStarted
	cancel()
	select {
	case <-secondDone:
	case <-time.After(500 * time.Millisecond):
		releaseBarrier()
		<-firstDone
		t.Fatal("canceled Workbench request remained blocked behind the active run")
	}
	if !strings.Contains(html.UnescapeString(rec.Body.String()), "context canceled") {
		t.Fatalf("canceled run did not report its cancellation: %s", rec.Body)
	}
	paths, _ := f.calls()
	if !reflect.DeepEqual(paths, []string{"POST /v1/states", "POST /v1/decide", "POST /v1/decide"}) {
		t.Fatalf("canceled waiter caused API effects: %v", paths)
	}
	control.mu.Lock()
	sets := append([]time.Duration(nil), control.sets...)
	control.mu.Unlock()
	if !reflect.DeepEqual(sets, []time.Duration{20 * time.Millisecond}) {
		t.Fatalf("canceled waiter changed the effective window: %v", sets)
	}

	releaseBarrier()
	select {
	case <-firstDone:
	case <-time.After(3 * time.Second):
		t.Fatal("active Workbench run did not finish after release")
	}
}

func enableWorkbenchCapacity(t testing.TB, e *env, requested int) home.CapacityProfile {
	t.Helper()
	status := e.d.cfg.Status()
	profile := home.CapacityProfile{CapacityTarget: home.CapacityTarget{
		Runtime: status.Runtime.Runtime, ModelID: status.Runtime.ModelID, Provider: "clef", Repo: "repo", Revision: "rev",
		SourceFilesSHA256: "sha256", Device: "cuda", DType: "bfloat16"},
		MaxStateTokens: 80, MaxInputTokens: 100, RequestedMaxInputTokens: requested,
		MaxBatchItems: 2, MaxBatchPaddedTokens: 200, RequiredGPUHeadroomBytes: 1}
	if err := (home.Home{Root: e.home}).SaveCapacityProfile(profile); err != nil {
		t.Fatal(err)
	}
	running := profile
	effective := profile.MaxInputTokens
	if requested > 0 {
		effective = min(effective, requested)
	}
	e.rt.mu.Lock()
	e.rt.snap.Info = worker.Info{"provider": "clef", "capacity": map[string]any{
		"profile": running, "model_context_tokens": float64(16384),
		"requested_max_input_tokens": requested, "effective_max_input_tokens": effective,
	}}
	e.rt.mu.Unlock()
	return profile
}

func TestWorkbenchShowsAndSavesRequestedCapacity(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	profile := enableWorkbenchCapacity(t, e, 40)

	page := e.get(t, "/workbench").Body.String()
	for _, want := range []string{`name="requested_max_input_tokens" value="40"`, "authoritative safe input limit is 100", "effective for the running worker"} {
		if !strings.Contains(page, want) {
			t.Errorf("capacity view lacks %q", want)
		}
	}

	form := with(form("state", []string{"q", "Pick", "a", "b"}), "op", "save-capacity", "requested_max_input_tokens", "25")
	page = e.post(t, "/workbench", form).Body.String()
	profiles, err := (home.Home{Root: e.home}).LoadCapacityProfiles()
	if err != nil {
		t.Fatal(err)
	}
	saved, ok := profiles.Find(profile.CapacityTarget)
	if !ok || saved.RequestedMaxInputTokens != 25 {
		t.Fatalf("saved profile = %+v, found %v", saved, ok)
	}
	for _, want := range []string{`name="requested_max_input_tokens" value="25"`, "Requested input-token limit saved", "Restart the runtime to apply it", "effective for the running worker"} {
		if !strings.Contains(page, want) {
			t.Errorf("saved capacity view lacks %q", want)
		}
	}
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("saving capacity sent inference: %v", paths)
	}
}

// form builds a workbench form: state plus questions given as
// id, instructions, then "label" or "label=description" choices.
func form(state string, qs ...[]string) url.Values {
	v := url.Values{"state": {state}, "nq": {itoa(len(qs))}}
	for i, q := range qs {
		p := "q" + itoa(i) + "."
		v.Set(p+"id", q[0])
		v.Set(p+"version", "1")
		v.Set(p+"instructions", q[1])
		v.Set(p+"nc", itoa(len(q)-2))
		for j, c := range q[2:] {
			label, desc, _ := strings.Cut(c, "=")
			v.Set(p+"c"+itoa(j)+".label", label)
			v.Set(p+"c"+itoa(j)+".desc", desc)
		}
	}
	return v
}

func itoa(i int) string { return strconv.Itoa(i) }

func with(v url.Values, kv ...string) url.Values {
	for i := 0; i+1 < len(kv); i += 2 {
		v.Set(kv[i], kv[i+1])
	}
	return v
}

var wireRe = regexp.MustCompile(`(?s)<textarea readonly rows="8" aria-label="Exact request JSON" spellcheck="false">(.*?)</textarea>`)

func TestWorkbenchRunSendsOneExistingDecideRequest(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	v := form("diff --git a/x b/x\r\n+added line",
		[]string{"scope", "Is the change in scope?", "yes=inside the Issue", "no", "unclear"},
		[]string{"risk", "How risky is it?", "low", "high"})
	rec := e.post(t, "/workbench", with(v, "op", "run"))
	if rec.Code != http.StatusOK {
		t.Fatalf("run: %d %s", rec.Code, rec.Body)
	}
	paths, bodies := f.calls()
	if len(paths) != 1 || paths[0] != "POST /v1/decide" {
		t.Fatalf("endpoint calls %v, want exactly one POST /v1/decide", paths)
	}
	var got api.DecideRequest
	dec := json.NewDecoder(bytes.NewReader(bodies[0]))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&got); err != nil {
		t.Fatalf("request is not a plain v1 decide request: %v\n%s", err, bodies[0])
	}
	want := api.DecideRequest{Schema: api.SchemaV1, State: "diff --git a/x b/x\n+added line", Questions: []api.Question{
		{ID: "scope", Type: "choice", Instructions: "Is the change in scope?", Choices: []string{"yes", "no", "unclear"},
			Descriptions: map[string]string{"yes": "inside the Issue"}},
		{ID: "risk", Type: "choice", Instructions: "How risky is it?", Choices: []string{"low", "high"}},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sent %+v\nwant %+v", got, want)
	}
	for _, leak := range []string{"expected", "version", "digest", question.Schema, "export_path", "token"} {
		if bytes.Contains(bodies[0], []byte(leak)) {
			t.Errorf("request carries %q: %s", leak, bodies[0])
		}
	}
	body := rec.Body.String()
	m := wireRe.FindStringSubmatch(body)
	if m == nil || html.UnescapeString(m[1]) != string(bodies[0]) {
		t.Fatalf("page does not show the exact request JSON that was sent:\n%v\nsent %s", m, bodies[0])
	}
	for _, s := range []string{"2 results", "inference 12.5 ms", "0.8000", "0.1000", "0.2000", `aria-label="Result · Question 0"`, `aria-label="Result · Question 1"`} {
		if !strings.Contains(body, s) {
			t.Errorf("page lacks %q", s)
		}
	}
}

func TestWorkbenchInvalidQuestionsFailValidationWithoutInference(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	for name, tc := range map[string]struct {
		v    url.Values
		want error
	}{
		"one choice":    {form("s", []string{"q", "Pick", "only"}), (&api.DecideRequest{Schema: api.SchemaV1, State: "s", Questions: []api.Question{{ID: "q", Type: "choice", Instructions: "Pick", Choices: []string{"only"}}}}).Validate()},
		"duplicate ids": {form("s", []string{"q", "A", "a", "b"}, []string{"q", "B", "a", "b"}), nil},
		"empty state":   {form("  ", []string{"q", "A", "a", "b"}), nil},
		"no id":         {form("s", []string{"", "A", "a", "b"}), nil},
		"dup labels":    {form("s", []string{"q", "A", "a", "a"}), nil},
	} {
		req := httptest.NewRequest("POST", "/", strings.NewReader(tc.v.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		built := parseWorkbench(req).request()
		wantErr := built.Validate()
		if wantErr == nil {
			t.Fatalf("%s: fixture is valid", name)
		}
		if tc.want != nil && tc.want.Error() != wantErr.Error() {
			t.Fatalf("%s: %v != %v", name, tc.want, wantErr)
		}
		body := e.post(t, "/workbench", with(tc.v, "op", "run")).Body.String()
		if !strings.Contains(body, html.EscapeString(wantErr.Error())) || !strings.Contains(body, "invalid · not sent") {
			t.Errorf("%s: page does not show the v1 validation error %q", name, wantErr)
		}
	}
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("invalid requests reached the endpoint: %v", paths)
	}
}

func TestWorkbenchPreviewShowsRequestWithoutInference(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	v := form("state", []string{"q", "Pick", "a", "b"})
	body := e.post(t, "/workbench", with(v, "op", "preview")).Body.String()
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("preview reached the endpoint: %v", paths)
	}
	m := wireRe.FindStringSubmatch(body)
	want, _ := json.Marshal(api.DecideRequest{Schema: api.SchemaV1, State: "state",
		Questions: []api.Question{{ID: "q", Type: "choice", Instructions: "Pick", Choices: []string{"a", "b"}}}})
	if m == nil || html.UnescapeString(m[1]) != string(want) || !strings.Contains(body, "valid · not sent") {
		t.Fatalf("preview does not show the request: %v", m)
	}
}

func TestWorkbenchShowsEndpointErrors(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	f.reply = func(api.DecideRequest) (int, any) {
		return http.StatusServiceUnavailable, api.ErrorBody{Schema: api.SchemaV1,
			Error: api.ErrorInfo{Class: api.ErrNotReady, Message: "worker is starting " + xss}}
	}
	body := e.post(t, "/workbench", with(form("s", []string{"q", "Pick", "a", "b"}), "op", "run")).Body.String()
	if !strings.Contains(body, "not_ready (HTTP 503): worker is starting") || strings.Contains(body, xss) {
		t.Fatalf("endpoint error not shown (escaped): %s", body)
	}
	if paths, _ := f.calls(); len(paths) != 1 {
		t.Fatalf("calls %v", paths)
	}
}

func writeDef(t testing.TB, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const defJSON = `{
  "descriptions": {"yes": "the change stays inside the Issue"},
  "schema": "hachidori.question.v1", "id": "scope", "version": 3, "type": "choice",
  "instructions": "Is the change in scope?",
  "choices": ["yes", "no", "unclear"]
}`

func TestWorkbenchLoadsDefinitionIntoEditor(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	dir := t.TempDir()
	path := writeDef(t, dir, "scope.json", defJSON)
	def, err := question.Parse([]byte(defJSON))
	if err != nil {
		t.Fatal(err)
	}
	v := with(form("state", []string{"", "", "", ""}), "op", "load", "load_path", path)
	body := e.post(t, "/workbench", v).Body.String()
	for _, s := range []string{`name="q0.id" value="scope"`, `name="q0.version" value="3"`, `Is the change in scope?</textarea>`,
		`name="q0.c0.label" value="yes"`, `name="q0.c0.desc" value="the change stays inside the Issue"`,
		`name="q0.c2.label" value="unclear"`, `name="q0.loaded_digest" value="` + def.Digest() + `"`,
		"matches loaded definition", "loaded scope@3 (" + def.Digest() + ")", `name="nq" value="1"`} {
		if !strings.Contains(body, s) {
			t.Errorf("projected editor lacks %q", s)
		}
	}
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("loading reached the endpoint: %v", paths)
	}

	// The projected editor runs exactly the compiled definition.
	rv := with(form("state", []string{"scope", "Is the change in scope?", "yes=the change stays inside the Issue", "no", "unclear"}),
		"op", "run", "q0.version", "3", "q0.loaded_id", "scope", "q0.loaded_version", "3", "q0.loaded_digest", def.Digest())
	body = e.post(t, "/workbench", rv).Body.String()
	_, bodies := f.calls()
	var got api.DecideRequest
	json.Unmarshal(bodies[0], &got)
	if !reflect.DeepEqual(got.Questions, []api.Question{def.Compile()}) {
		t.Fatalf("sent %+v, want the compiled definition %+v", got.Questions, def.Compile())
	}
	if !strings.Contains(body, "matches loaded definition") {
		t.Error("unchanged loaded question reported as modified")
	}
	rv.Set("q0.instructions", "Edited?")
	if body := e.post(t, "/workbench", with(rv, "op", "preview")).Body.String(); !strings.Contains(body, "modified since load") {
		t.Error("edited loaded question not reported as modified")
	}

	// A second load appends instead of replacing a non-empty question.
	body = e.post(t, "/workbench", with(rv, "op", "load", "load_path", path)).Body.String()
	if !strings.Contains(body, `name="nq" value="2"`) || !strings.Contains(body, `name="q0.instructions" rows="3">Edited?`) {
		t.Error("load replaced an edited question")
	}
}

func TestWorkbenchLoadRejectsBadInput(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	dir := t.TempDir()
	bad := writeDef(t, dir, "bad.json", `{"schema":"hachidori.question.v1","id":"q","version":1,"type":"choice","instructions":"x","choices":["a","b"],"extra":1}`)
	old := writeDef(t, dir, "old.json", `{"schema":"hachidori.question.v0","id":"q","version":1,"type":"choice","instructions":"x","choices":["a","b"]}`)
	for p, want := range map[string]string{
		"relative.json":                    "path must be absolute",
		filepath.Join(dir, "a..b.json"):    `must not contain ".."`,
		"":                                 "enter the Question Definition file path",
		dir:                                "is not a regular file",
		bad:                                `unknown field "extra"`,
		old:                                `schema must be "hachidori.question.v1"`,
		filepath.Join(dir, "missing.json"): "missing.json",
	} {
		body := e.post(t, "/workbench", with(form("s", []string{"", "", "", ""}), "op", "load", "load_path", p)).Body.String()
		if !strings.Contains(html.UnescapeString(body), want) {
			t.Errorf("load %q: page lacks %q", p, want)
		}
		if strings.Contains(body, "loaded_id") {
			t.Errorf("load %q projected something", p)
		}
	}
}

func TestWorkbenchExportsValidDefinition(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	dir := t.TempDir()
	v := form("state", []string{"scope", "Is the change in scope?", "yes=inside", "no", "unclear"}, []string{"other", "x", "a", "b"})
	v.Set("q0.version", "2")
	out1, out2 := filepath.Join(dir, "a.json"), filepath.Join(dir, "b.json")
	body := e.post(t, "/workbench", with(v, "op", "export:0", "q0.export_path", out1)).Body.String()
	set, err := question.Load(out1)
	if err != nil {
		t.Fatalf("exported file is not a valid definition: %v\n%s", err, body)
	}
	defs := set.Definitions()
	want := question.Definition{Schema: question.Schema, ID: "scope", Version: 2, Type: "choice", Instructions: "Is the change in scope?",
		Choices: []string{"yes", "no", "unclear"}, Descriptions: map[string]string{"yes": "inside"}}
	if len(defs) != 1 || defs[0].Identity() != want.Identity() {
		t.Fatalf("exported %+v, want identity %+v", defs, want.Identity())
	}
	if !strings.Contains(body, "wrote scope@2 ("+want.Digest()+") to "+html.EscapeString(out1)) {
		t.Errorf("export result not shown: %s", body)
	}
	// Deterministic: the same question exports to identical bytes.
	e.post(t, "/workbench", with(v, "op", "export:0", "q0.export_path", out2))
	a, _ := os.ReadFile(out1)
	b, _ := os.ReadFile(out2)
	if !bytes.Equal(a, b) {
		t.Fatalf("exports differ:\n%s\n%s", a, b)
	}
	// Never overwrites.
	os.WriteFile(out2, []byte("keep"), 0o644)
	body = e.post(t, "/workbench", with(v, "op", "export:0", "q0.export_path", out2)).Body.String()
	if got, _ := os.ReadFile(out2); string(got) != "keep" || !strings.Contains(body, "already exists") {
		t.Fatalf("export replaced an existing file: %q", got)
	}
	// Invalid questions and relative paths are not exported.
	v.Set("q0.version", "0")
	body = e.post(t, "/workbench", with(v, "op", "export:0", "q0.export_path", filepath.Join(dir, "c.json"))).Body.String()
	if _, err := os.Stat(filepath.Join(dir, "c.json")); err == nil || !strings.Contains(body, "version must be an integer &gt;= 1") {
		t.Fatal("invalid definition exported")
	}
	v.Set("q0.version", "1")
	e.post(t, "/workbench", with(v, "op", "export:0", "q0.export_path", "c.json"))
	if _, err := os.Stat("c.json"); err == nil {
		os.Remove("c.json")
		t.Fatal("relative export path written")
	}
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("export reached the endpoint: %v", paths)
	}
}

func TestWorkbenchProjectionRoundTrips(t *testing.T) {
	def, err := question.Parse([]byte(defJSON))
	if err != nil {
		t.Fatal(err)
	}
	q := project(def).withBlankRows()
	back, err := q.definition()
	if err != nil {
		t.Fatal(err)
	}
	if back.Digest() != def.Digest() || !bytes.Equal(back.Canonical(), def.Canonical()) {
		t.Fatalf("projection changed the definition:\n%s\n%s", back.Canonical(), def.Canonical())
	}
	data, err := definitionJSON(back)
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := question.Parse(data); again.Identity() != def.Identity() {
		t.Fatal("exported form does not round-trip")
	}
}

func TestWorkbenchEditorOperations(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	v := form("s", []string{"a", "A", "x", "y"}, []string{"b", "B", "x", "y"})
	body := e.post(t, "/workbench", with(v, "op", "add")).Body.String()
	if !strings.Contains(body, `name="nq" value="3"`) {
		t.Error("add question")
	}
	body = e.post(t, "/workbench", with(v, "op", "remove:0")).Body.String()
	if !strings.Contains(body, `name="nq" value="1"`) || !strings.Contains(body, `name="q0.id" value="b"`) {
		t.Error("remove question")
	}
	body = e.post(t, "/workbench", with(v, "op", "choices:1")).Body.String()
	// 2 used rows + 2 blank rows + 2 added rows
	if !strings.Contains(body, `name="q1.nc" value="6"`) || !strings.Contains(body, `name="q0.nc" value="4"`) {
		t.Error("add choice rows")
	}
	one := form("s", []string{"a", "A", "x", "y"})
	if body := e.post(t, "/workbench", with(one, "op", "remove:0")).Body.String(); !strings.Contains(body, `name="nq" value="1"`) {
		t.Error("removing the last question leaves no editor")
	}
	if body := e.post(t, "/workbench", with(one, "op", "bogus")).Body.String(); !strings.Contains(body, "unknown workbench operation") {
		t.Error("unknown op accepted")
	}
	// Question count is bounded by the v1 limit.
	big := url.Values{"nq": {"100000"}, "op": {"preview"}}
	req := httptest.NewRequest("POST", "/", strings.NewReader(big.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if n := len(parseWorkbench(req).Questions); n != api.MaxQuestions {
		t.Errorf("parsed %d questions", n)
	}
}

func TestWorkbenchSecurityBoundaries(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	v := with(form("s", []string{"q", "Pick", "a", "b"}), "op", "run")

	// GET renders the editor and never calls the endpoint.
	if rec := e.get(t, "/workbench"); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `action="/workbench"`) {
		t.Fatalf("GET /workbench: %d", rec.Code)
	}
	// Missing token, cross-origin and cross-site POSTs are refused.
	nt := url.Values{}
	for k, vs := range v {
		nt[k] = vs
	}
	nt.Set("token", "stale")
	if rec := e.post(t, "/workbench", nt); rec.Code != http.StatusForbidden {
		t.Errorf("stale token: %d", rec.Code)
	}
	for _, h := range [][2]string{{"Origin", "http://evil.example"}, {"Sec-Fetch-Site", "cross-site"}} {
		v.Set("token", e.d.token)
		req := httptest.NewRequest("POST", "http://127.0.0.1:7844/workbench", strings.NewReader(v.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set(h[0], h[1])
		rec := httptest.NewRecorder()
		e.d.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Errorf("%s %s: %d", h[0], h[1], rec.Code)
		}
	}
	req := httptest.NewRequest("POST", "http://evil.example/workbench", strings.NewReader(v.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	e.d.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Errorf("non-loopback host: %d", rec.Code)
	}
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("refused requests reached the endpoint: %v", paths)
	}

	// A maximal v1 state fits the workbench body limit; other routes keep
	// the small limit.
	big := with(form(strings.Repeat("x", api.MaxStateBytes), []string{"q", "Pick", "a", "b"}), "op", "preview")
	if rec := e.post(t, "/workbench", big); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "valid · not sent") {
		t.Errorf("maximal state: %d", rec.Code)
	}
	if rec := e.post(t, "/doctor", url.Values{"pad": {strings.Repeat("x", 8<<10)}}); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized doctor POST: %d", rec.Code)
	}
	if rec := e.post(t, "/workbench", url.Values{"pad": {strings.Repeat("x", workbenchMaxBody)}}); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized workbench POST: %d", rec.Code)
	}
	// Rendered values are escaped.
	body := e.post(t, "/workbench", with(form(xss, []string{xss, xss, xss, "b=" + xss}), "op", "preview")).Body.String()
	if strings.Contains(body, xss) {
		t.Fatal("unescaped value in workbench page")
	}
}

func TestNavigationLinksWorkbench(t *testing.T) {
	e, _ := newWorkbenchEnv(t)
	root := e.get(t, "/").Body.String()
	if !strings.Contains(root, `<a href="/workbench">Workbench</a>`) || !strings.Contains(root, `<a href="/" aria-current="page">Runtime</a>`) {
		t.Error("runtime page lacks the navigation")
	}
	wb := e.get(t, "/workbench").Body.String()
	if !strings.Contains(wb, `<a href="/workbench" aria-current="page">Workbench</a>`) || !strings.Contains(wb, `<a href="/diagnostics">Diagnostics</a>`) {
		t.Error("workbench page lacks the navigation")
	}
	if strings.Contains(wb, `id="refresh"`) {
		t.Error("workbench page polls live status")
	}
}

func TestWorkbenchPickKeepsEditorState(t *testing.T) {
	e, f := newWorkbenchEnv(t)
	p := &fakePathPicker{path: "/defs/chosen.json"}
	withPathPicker(e, p)
	v := form("state", []string{"scope", "Is it in scope?", "yes", "no"})
	body := e.post(t, "/workbench", with(v, "op", "pick-load")).Body.String()
	for _, s := range []string{`name="load_path" value="/defs/chosen.json"`, `name="q0.id" value="scope"`, "Is it in scope?</textarea>"} {
		if !strings.Contains(body, s) {
			t.Errorf("load pick: page lacks %q", s)
		}
	}
	dir := t.TempDir()
	p.path = filepath.Join(dir, "out.json")
	body = e.post(t, "/workbench", with(v, "op", "pick-export:0")).Body.String()
	if !strings.Contains(body, `name="q0.export_path" value="`+p.path+`"`) || p.calls[1] != "save" {
		t.Error("export pick did not fill the destination")
	}
	if _, err := os.Stat(p.path); !os.IsNotExist(err) {
		t.Fatal("choosing a destination wrote the file")
	}
	p.err = errors.New("dialog broke")
	body = e.post(t, "/workbench", with(v, "op", "pick-load")).Body.String()
	if !strings.Contains(body, "choosing a Question Definition: dialog broke") || !strings.Contains(body, `name="q0.id" value="scope"`) {
		t.Error("picker failure lost the editor or its error")
	}
	if paths, _ := f.calls(); len(paths) != 0 {
		t.Fatalf("picking reached the endpoint: %v", paths)
	}
}
