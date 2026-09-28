package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

type fake struct {
	ready bool
	err   error
	calls int
}

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
	h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(b)))
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
