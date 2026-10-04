package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/requesthistory"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/worker"
)

type fake struct {
	ready bool
	err   error
	calls int
}

type identityFake struct {
	*fake
	identities map[string]requesthistory.Identity
	failModel  string
	policy     *route.Router
}

func (f *identityFake) Identity(model string) (api.Served, bool) {
	identity, ok := f.identities[model]
	if !ok {
		return api.Served{}, false
	}
	return api.Served{Model: identity.Model}, true
}

func (f *identityFake) RequestIdentity(model string) (requesthistory.Identity, bool) {
	identity, ok := f.identities[model]
	return identity, ok
}

func (f *identityFake) DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error) {
	if model == f.failModel {
		return nil, 0, &worker.RequestError{Class: api.ErrInferenceFailed, Message: "selected resident failed"}
	}
	return f.fake.Decide(items)
}

func (f *identityFake) AutoRouter() *route.Router { return f.policy }

func (f *fake) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	f.calls++
	if f.err != nil {
		return nil, 0, f.err
	}
	var out [][]api.Result
	for _, it := range items {
		var rs []api.Result
		for _, q := range it.Questions {
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[1], Confidence: 0.8})
		}
		out = append(out, rs)
	}
	return out, 12.5, nil
}
func (f *fake) Ready() bool { return f.ready }
func (f *fake) State() string {
	if f.ready {
		return worker.StateReady
	}
	return worker.StateStarting
}
func (f *fake) Snapshot() worker.Snapshot { return worker.Snapshot{State: f.State(), Ready: f.ready} }

const body = `{"schema":"hachidori.v1","state":"s","questions":[{"id":"q","type":"choice","instructions":"i","choices":["yes","no"]}]}`

func do(h http.Handler, method, path, b string) (*httptest.ResponseRecorder, map[string]any) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, strings.NewReader(b))
	req.Host = DefaultListen
	h.ServeHTTP(rec, req)
	var m map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &m)
	return rec, m
}

func errClass(m map[string]any) string {
	e, _ := m["error"].(map[string]any)
	s, _ := e["class"].(string)
	return s
}

func TestHealthReflectsReadiness(t *testing.T) {
	f := &fake{}
	h := Handler(f, Runtime{})
	if rec, _ := do(h, "GET", "/health", ""); rec.Code != 503 {
		t.Fatalf("not ready: %d", rec.Code)
	}
	f.ready = true
	if rec, m := do(h, "GET", "/health", ""); rec.Code != 200 || m["ready"] != true {
		t.Fatalf("ready: %d %v", rec.Code, m)
	}
}

func TestDecide(t *testing.T) {
	f := &fake{ready: true}
	h := Handler(f, Runtime{Device: "cpu"})
	rec, m := do(h, "POST", "/v1/decide", body)
	if rec.Code != 200 || m["schema"] != api.SchemaV1 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	res := m["results"].([]any)[0].(map[string]any)
	if res["id"] != "q" || res["choice"] != "no" {
		t.Fatalf("%v", res)
	}
	batch := `{"schema":"hachidori.v1","requests":[` + body + `,` + body + `]}`
	rec, m = do(h, "POST", "/v1/decide/batch", batch)
	if rec.Code != 200 || len(m["responses"].([]any)) != 2 || f.calls != 2 {
		t.Fatalf("batch: %d %s", rec.Code, rec.Body)
	}
	if rec, m := do(h, "GET", "/v1/status", ""); rec.Code != 200 || m["runtime"].(map[string]any)["device"] != "cpu" {
		t.Fatalf("status: %d %v", rec.Code, m)
	}
}

func TestRequestHistoryProjectsSingleBatchAndFailure(t *testing.T) {
	requests := requesthistory.New()
	rt := Runtime{Runtime: "rt-1", ModelID: "m-1", Model: "org/model@rev", Device: "cuda",
		Variant: &Variant{ID: "variant-1"}}
	h := HandlerSinceWithHistory(&fake{ready: true}, rt, time.Now(), requests)
	if rec, _ := do(h, "POST", "/v1/decide", body); rec.Code != http.StatusOK {
		t.Fatalf("single decision: %d %s", rec.Code, rec.Body)
	}
	batch := `{"schema":"hachidori.v1","requests":[` + body + `,` + body + `]}`
	if rec, _ := do(h, "POST", "/v1/decide/batch", batch); rec.Code != http.StatusOK {
		t.Fatalf("batch decision: %d %s", rec.Code, rec.Body)
	}
	view := requests.List()
	if view.QueueDepth != 0 || view.InFlight != 0 || view.ActiveItems != 0 || len(view.Entries) != 2 {
		t.Fatalf("history view %+v", view)
	}
	if got := view.Entries[0]; got.Endpoint != "/v1/decide/batch" || got.ItemCount != 2 || got.State != "completed" || got.Runtime.Model != "m-1" || got.Runtime.Variant != "variant-1" {
		t.Fatalf("batch projection %+v", got)
	}
	if got := view.Entries[1]; got.Endpoint != "/v1/decide" || got.ItemCount != 1 || got.QuestionCount != 1 || got.StateBytes != 1 || got.State != "completed" {
		t.Fatalf("single projection %+v", got)
	}
	detail, ok := requests.Get(view.Entries[0].ID)
	if !ok || !strings.Contains(string(detail.Input), "requests") || !strings.Contains(string(detail.Output), "responses") || !strings.Contains(string(detail.Output), "confidence") {
		t.Fatalf("batch detail is not inspectable: found=%v input=%s output=%s", ok, detail.Input, detail.Output)
	}

	if rec, _ := do(h, "POST", "/v1/decide", `{"schema":"hachidori.v1","state":"bad","questions":[]}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("rejected request status: %d %s", rec.Code, rec.Body)
	}
	view = requests.List()
	if got := view.Entries[0]; got.State != "rejected" || got.ErrorClass != api.ErrRequestInvalid || got.HTTPStatus != http.StatusBadRequest {
		t.Fatalf("rejected request projection %+v", got)
	}

	failedStore := requesthistory.New()
	failed := HandlerSinceWithHistory(&fake{ready: true, err: &worker.RequestError{Class: api.ErrInferenceFailed, Message: "typed failure"}}, rt, time.Now(), failedStore)
	if rec, _ := do(failed, "POST", "/v1/decide", body); rec.Code != http.StatusInternalServerError {
		t.Fatalf("failed decision: %d %s", rec.Code, rec.Body)
	}
	entry := failedStore.List().Entries[0]
	if entry.State != "failed" || entry.ErrorClass != api.ErrInferenceFailed || entry.ErrorMessage != "typed failure" {
		t.Fatalf("execution failure projection %+v", entry)
	}

	timeoutStore := requesthistory.New()
	timeout := HandlerSinceWithHistory(&fake{ready: true, err: &worker.Failure{Class: worker.ClassUnresponsive, Message: "no response"}}, rt, time.Now(), timeoutStore)
	if rec, _ := do(timeout, "POST", "/v1/decide", body); rec.Code != http.StatusBadGateway {
		t.Fatalf("timed out decision: %d %s", rec.Code, rec.Body)
	}
	entry = timeoutStore.List().Entries[0]
	if entry.State != "timed_out" || entry.ErrorClass != api.ErrWorkerFailure || entry.ErrorMessage == "" {
		t.Fatalf("timed out request projection %+v", entry)
	}
}

func TestRequestHistoryUsesSelectedResidentIdentityIncludingAutoFailure(t *testing.T) {
	defaultIdentity := requesthistory.Identity{Runtime: "runtime-default", Model: "default-model", Device: "cuda:0", Variant: "default-variant"}
	selectedIdentity := requesthistory.Identity{Runtime: "runtime-selected", Model: "selected-model", Device: "cuda:0", Variant: "selected-variant"}
	identities := map[string]requesthistory.Identity{"selected-model": selectedIdentity}
	rt := Runtime{Runtime: defaultIdentity.Runtime, ModelID: defaultIdentity.Model, Device: defaultIdentity.Device,
		Variant: &Variant{ID: defaultIdentity.Variant}}

	targetedStore := requesthistory.New()
	targeted := &identityFake{fake: &fake{ready: true}, identities: identities}
	h := HandlerSinceWithHistory(targeted, rt, time.Now(), targetedStore)
	selectedBody := strings.Replace(body, `"state":"s"`, `"state":"s","model":"selected-model"`, 1)
	if rec, _ := do(h, "POST", "/v1/decide", selectedBody); rec.Code != http.StatusOK {
		t.Fatalf("targeted decision: %d %s", rec.Code, rec.Body)
	}
	if got := targetedStore.List().Entries[0].Runtime; got != selectedIdentity {
		t.Fatalf("targeted execution identity %+v, want %+v", got, selectedIdentity)
	}

	failedStore := requesthistory.New()
	backend := &identityFake{fake: &fake{ready: true}, identities: identities, failModel: "selected-model"}
	policy, err := route.New(route.Policy{Schema: route.PolicySchema, ID: "history-test", Default: route.Rule{First: "selected-model"}}, backend)
	if err != nil {
		t.Fatal(err)
	}
	backend.policy = policy
	h = HandlerSinceWithHistory(backend, rt, time.Now(), failedStore)
	autoBody := strings.Replace(body, `"state":"s"`, `"state":"s","route":"auto"`, 1)
	if rec, _ := do(h, "POST", "/v1/decide", autoBody); rec.Code != http.StatusFailedDependency {
		t.Fatalf("auto decision: %d %s", rec.Code, rec.Body)
	}
	entry := failedStore.List().Entries[0]
	if entry.State != "failed" || entry.Runtime != selectedIdentity || entry.ErrorClass != api.ErrRoutingFailed {
		t.Fatalf("auto failure lost selected execution identity: %+v", entry)
	}
}

func TestErrorsAreStructured(t *testing.T) {
	cases := []struct {
		err   error
		body  string
		code  int
		class string
	}{
		{nil, `{"schema":"hachidori.v1","state":"s","questions":[]}`, 400, api.ErrRequestInvalid},
		{nil, `{"schema":"hachidori.v1","bogus":1}`, 400, api.ErrRequestInvalid},
		{nil, `not json`, 400, api.ErrRequestInvalid},
		{&worker.RequestError{Class: api.ErrNotReady, Message: "starting"}, body, 503, api.ErrNotReady},
		{&worker.RequestError{Class: api.ErrCapacity, Message: "full"}, body, 429, api.ErrCapacity},
		{&worker.RequestError{Class: api.ErrInferenceFailed, Message: "x"}, body, 500, api.ErrInferenceFailed},
		// A crashed worker must never look like a semantic answer.
		{&worker.Failure{Class: worker.ClassCrash, Message: "exited"}, body, 502, api.ErrWorkerFailure},
	}
	for _, c := range cases {
		h := Handler(&fake{ready: true, err: c.err}, Runtime{})
		rec, m := do(h, "POST", "/v1/decide", c.body)
		if rec.Code != c.code || errClass(m) != c.class || m["results"] != nil {
			t.Errorf("%v %q: got %d %s", c.err, c.body, rec.Code, rec.Body)
		}
	}
}

func TestCheckLoopback(t *testing.T) {
	for _, a := range []string{"127.0.0.1:7843", "[::1]:7843", "localhost:1"} {
		if err := CheckLoopback(a); err != nil {
			t.Errorf("%s: %v", a, err)
		}
	}
	for _, a := range []string{"0.0.0.0:7843", ":7843", "192.168.1.2:7843"} {
		if CheckLoopback(a) == nil {
			t.Errorf("%s accepted", a)
		}
	}
}

func TestAPIIsHostLocal(t *testing.T) {
	f := &fake{ready: true}
	h := Handler(f, Runtime{Home: "/srv/hachidori"})
	cases := []struct {
		name    string
		method  string
		path    string
		host    string
		headers map[string]string
		code    int
	}{
		{"loopback ip", "GET", "/v1/status", "127.0.0.1:7843", nil, 200},
		{"localhost", "GET", "/v1/status", "localhost:7843", nil, 200},
		{"ipv6 loopback", "GET", "/health", "[::1]:7843", nil, 200},
		{"no origin header (non-browser caller)", "POST", "/v1/decide", "127.0.0.1:7843", nil, 200},
		{"same origin", "POST", "/v1/decide", "127.0.0.1:7843", map[string]string{"Origin": "http://127.0.0.1:7843"}, 200},
		// DNS rebinding: the page's own name resolves to 127.0.0.1.
		{"rebound status", "GET", "/v1/status", "attacker.example:7843", nil, 403},
		{"rebound health", "GET", "/health", "attacker.example", nil, 403},
		{"rebound decide", "POST", "/v1/decide", "attacker.example:7843", nil, 403},
		// A no-preflight cross-site POST from a page in the operator's browser.
		{"cross-origin post", "POST", "/v1/decide", "127.0.0.1:7843", map[string]string{"Origin": "https://attacker.example"}, 403},
		{"opaque origin post", "POST", "/v1/decide/batch", "127.0.0.1:7843", map[string]string{"Origin": "null"}, 403},
		{"cross-site fetch metadata", "POST", "/v1/decide", "127.0.0.1:7843", map[string]string{"Sec-Fetch-Site": "cross-site"}, 403},
	}
	for _, c := range cases {
		b := ""
		if c.method == "POST" {
			b = body
			if strings.HasSuffix(c.path, "/batch") {
				b = `{"schema":"hachidori.v1","requests":[` + body + `]}`
			}
		}
		req := httptest.NewRequest(c.method, c.path, strings.NewReader(b))
		req.Host = c.host
		for k, v := range c.headers {
			req.Header.Set(k, v)
		}
		rec := httptest.NewRecorder()
		before := f.calls
		h.ServeHTTP(rec, req)
		if rec.Code != c.code {
			t.Errorf("%s: status %d, want %d (%s)", c.name, rec.Code, c.code, rec.Body)
		}
		if c.code == 403 {
			if f.calls != before {
				t.Errorf("%s: a refused request reached the worker", c.name)
			}
			if strings.Contains(rec.Body.String(), "/srv/hachidori") {
				t.Errorf("%s: refusal leaks the home path: %s", c.name, rec.Body)
			}
		}
	}
}

// Worker text reaches API callers (possibly remote, behind the reverse
// tunnel) only through the shared redaction policy: the class and the useful
// part of the detail survive; local paths and credentials do not.
func TestWorkerErrorDetailIsRedactedAndBounded(t *testing.T) {
	const homeDir, profile = `C:\Users\alice\Hachidori`, `C:\Users\alice`
	t.Setenv("HOME", profile)
	t.Setenv("USERPROFILE", profile)
	repr := `OSError: [Errno 2] No such file or directory: 'C:\\Users\\alice\\Hachidori\\cache\\x'`
	cases := []struct {
		name  string
		err   error
		code  int
		class string
		keep  string
	}{
		{"request error with doubled backslashes", &worker.RequestError{Class: api.ErrInferenceFailed, Message: repr}, 500, api.ErrInferenceFailed, "<HACHIDORI_HOME>"},
		{"request error with a credential url", &worker.RequestError{Class: api.ErrInferenceFailed,
			Message: "HTTPError: https://user:pw-example@hub.example.invalid/m?token=not-a-real-token"}, 500, api.ErrInferenceFailed, "hub.example.invalid"},
		{"worker failure", &worker.Failure{Class: worker.ClassModelLoad, Message: repr}, 502, api.ErrWorkerFailure, "model_load"},
		{"plain error", errors.New("cache C:/Users/alice/x unreadable"), 502, api.ErrWorkerFailure, "<USERPROFILE>"},
		{"useful detail is kept", &worker.RequestError{Class: api.ErrInferenceFailed, Message: "OutOfMemoryError: CUDA out of memory"}, 500, api.ErrInferenceFailed, "CUDA out of memory"},
	}
	for _, c := range cases {
		h := Handler(&fake{ready: true, err: c.err}, Runtime{Home: homeDir})
		for _, path := range []string{"/v1/decide", "/v1/decide/batch"} {
			b := body
			if strings.HasSuffix(path, "/batch") {
				b = `{"schema":"hachidori.v1","requests":[` + body + `]}`
			}
			rec, m := do(h, "POST", path, b)
			got := rec.Body.String()
			if rec.Code != c.code || errClass(m) != c.class {
				t.Errorf("%s %s: %d %s", c.name, path, rec.Code, got)
			}
			for _, leak := range []string{"alice", "pw-example", "not-a-real-token"} {
				if strings.Contains(got, leak) {
					t.Errorf("%s %s: response carries %q: %s", c.name, path, leak, got)
				}
			}
			if msg, _ := m["error"].(map[string]any)["message"].(string); !strings.Contains(msg, c.keep) {
				t.Errorf("%s %s: message lost %q: %q", c.name, path, c.keep, msg)
			}
		}
	}

	long := &worker.RequestError{Class: api.ErrInferenceFailed, Message: strings.Repeat("x", 1<<20)}
	rec, m := do(Handler(&fake{ready: true, err: long}, Runtime{}), "POST", "/v1/decide", body)
	if msg := m["error"].(map[string]any)["message"].(string); rec.Code != 500 || len(msg) > maxErrorDetail+len("...[truncated]") {
		t.Fatalf("detail not bounded: %d bytes, status %d", len(msg), rec.Code)
	}
}
