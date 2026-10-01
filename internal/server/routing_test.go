package server

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// routedFake is a two-resident Decider whose residents answer by script and
// that applies a routing policy. Direct calls (DecideOn) and the default
// route (Decide) are counted per resident, so a test can prove a direct
// request never reached the router or another resident.
type routedFake struct {
	fullFake
	router *route.Router
	conf   map[string]float64 // per resident confidence of its answers
	fail   map[string]error
	direct map[string]int // DecideOn calls per resident
	def    int            // default-route Decide calls
}

func (r *routedFake) Identity(model string) (api.Served, bool) {
	switch model {
	case "m1":
		return api.Served{Model: "m1", Provider: "p1"}, true
	case "m2":
		return api.Served{Model: "m2", Provider: "p2"}, true
	}
	return api.Served{}, false
}

func (r *routedFake) DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error) {
	r.direct[model]++
	if err := r.fail[model]; err != nil {
		return nil, 0, err
	}
	var out [][]api.Result
	for _, it := range items {
		var rs []api.Result
		for _, q := range it.Questions {
			c := r.conf[model]
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: model + "-" + q.Choices[0], Confidence: c,
				Probabilities: map[string]float64{model + "-" + q.Choices[0]: c, "other": 1 - c}})
		}
		out = append(out, rs)
	}
	return out, 2, nil
}

func (r *routedFake) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	r.def++
	return r.fullFake.Decide(items)
}

func (r *routedFake) AutoRouter() *route.Router { return r.router }
func (r *routedFake) RoutingStatus() *route.Status {
	if r.router == nil {
		return nil
	}
	st := r.router.Status()
	return &st
}

func newRoutedFake(t *testing.T, withPolicy bool) (*routedFake, func(method, path, body string) (int, map[string]any)) {
	t.Helper()
	snap := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, Errors: map[string]int64{}, QueueLimit: 64}
	f := &routedFake{fullFake: fullFake{fake: fake{ready: true}, snap: snap}, conf: map[string]float64{"m1": 0.55, "m2": 0.95},
		fail: map[string]error{}, direct: map[string]int{}}
	if withPolicy {
		one := 0.7
		p := route.Policy{Schema: route.PolicySchema, ID: "p", Families: map[string]string{"hot": "f"},
			Rules:   []route.Rule{{Family: "f", First: "m1", Handoff: &route.Handoff{To: "m2", When: route.When{ConfidenceBelow: &one}}}},
			Default: route.Rule{First: "m1"}}
		r, err := route.New(p, f)
		if err != nil {
			t.Fatal(err)
		}
		f.router = r
	}
	h := Handler(f, Runtime{ModelID: "m1", Device: "cpu"})
	return f, func(method, path, body string) (int, map[string]any) {
		rec, m := do(h, method, path, body)
		return rec.Code, m
	}
}

const (
	hotQ  = `{"id":"hot","type":"choice","instructions":"i","choices":["x","y"]}`
	coldQ = `{"id":"cold","type":"choice","instructions":"i","choices":["x","y"]}`
)

func req(extra string, qs ...string) string {
	return `{"schema":"hachidori.v1","state":"s","questions":[` + strings.Join(qs, ",") + `]` + extra + `}`
}

func TestAutoRouteHandsOffOnlySelectedQuestions(t *testing.T) {
	f, call := newRoutedFake(t, true)
	code, m := call("POST", "/v1/decide", req(`,"route":"auto"`, hotQ, coldQ))
	if code != 200 {
		t.Fatalf("%d %v", code, m)
	}
	results := m["results"].([]any)
	choice := func(i int) string { return results[i].(map[string]any)["choice"].(string) }
	// hot: m1 answered with 0.55 < 0.7, handed to m2. cold: default rule, m1 kept.
	if choice(0) != "m2-x" || choice(1) != "m1-x" {
		t.Fatalf("results %v", results)
	}
	routing := m["routing"].(map[string]any)
	rs := routing["results"].([]any)
	hot, cold := rs[0].(map[string]any), rs[1].(map[string]any)
	if hot["served"].(map[string]any)["model"] != "m2" || hot["reason"] != api.ReasonHandoffLowConf ||
		hot["first_path"].(map[string]any)["model"] != "m1" || hot["profile"] != "f" {
		t.Fatalf("hot %v", hot)
	}
	if cold["served"].(map[string]any)["model"] != "m1" || cold["reason"] != api.ReasonFirstPathOnly || cold["first_path"] != nil {
		t.Fatalf("cold %v", cold)
	}
	if routing["handoffs"] != float64(1) || len(routing["providers"].([]any)) != 2 || routing["mode"] != "auto" ||
		routing["policy"].(map[string]any)["id"] != "p" {
		t.Fatalf("routing %v", routing)
	}
	if m["served"] != nil {
		t.Fatal("a routed response names its residents per result, not one served")
	}
	if timing := m["timing"].(map[string]any); timing["inference_ms"] != float64(4) {
		t.Fatalf("timing %v", timing)
	}
	if f.def != 0 {
		t.Fatal("the default route must not be used by a routed request")
	}
	if !reflect.DeepEqual(f.direct, map[string]int{"m1": 1, "m2": 1}) {
		t.Fatalf("worker calls %v", f.direct)
	}
}

func TestAutoRouteBatchRoutesEveryRequestAndReportsAggregateProviders(t *testing.T) {
	_, call := newRoutedFake(t, true)
	b := `{"schema":"hachidori.v1","route":"auto","requests":[` + req("", hotQ) + `,` + req(`,"route":"auto"`, coldQ) + `]}`
	code, m := call("POST", "/v1/decide/batch", b)
	if code != 200 {
		t.Fatalf("%d %v", code, m)
	}
	resp := m["responses"].([]any)
	first := resp[0].(map[string]any)
	if first["results"].([]any)[0].(map[string]any)["choice"] != "m2-x" || first["routing"].(map[string]any)["handoffs"] != float64(1) {
		t.Fatalf("%v", first)
	}
	if first["routing"].(map[string]any)["providers"] != nil || first["timing"] != nil {
		t.Fatalf("a batch entry carries no timing or providers: %v", first)
	}
	if resp[1].(map[string]any)["results"].([]any)[0].(map[string]any)["choice"] != "m1-x" {
		t.Fatalf("%v", resp[1])
	}
	agg := m["routing"].(map[string]any)
	if agg["handoffs"] != float64(1) || len(agg["providers"].([]any)) != 2 || agg["results"] != nil {
		t.Fatalf("aggregate %v", agg)
	}
}

func TestDirectRequestsNeverTouchTheRouter(t *testing.T) {
	f, call := newRoutedFake(t, true)
	for _, model := range []string{"m1", "m2"} {
		code, m := call("POST", "/v1/decide", req(`,"model":"`+model+`"`, hotQ))
		if code != 200 || m["served"].(map[string]any)["model"] != model || m["routing"] != nil {
			t.Fatalf("%s: %d %v", model, code, m)
		}
		// Low confidence on m1 is exactly what auto would hand off; a direct
		// request keeps the answer of the resident it named.
		if got := m["results"].([]any)[0].(map[string]any)["choice"]; got != model+"-x" {
			t.Fatalf("%s answered %v", model, got)
		}
	}
	if !reflect.DeepEqual(f.direct, map[string]int{"m1": 1, "m2": 1}) {
		t.Fatalf("a direct request is one call on its own resident: %v", f.direct)
	}
	// A direct request to a resident that is down fails; it is not rerouted.
	f.fail["m2"] = &worker.RequestError{Class: api.ErrNotReady, Message: "starting"}
	code, m := call("POST", "/v1/decide", req(`,"model":"m2"`, hotQ))
	if code != 503 || errClass(m) != api.ErrNotReady || f.direct["m1"] != 1 {
		t.Fatalf("%d %v %v", code, m, f.direct)
	}
	// The default route is untouched by routing.
	if code, m := call("POST", "/v1/decide", req("", hotQ)); code != 200 || m["routing"] != nil || m["served"] != nil || f.def != 1 {
		t.Fatalf("%d %v def=%d", code, m, f.def)
	}
}

func TestAutoRouteCannotBeCombinedWithADirectModel(t *testing.T) {
	f, call := newRoutedFake(t, true)
	code, m := call("POST", "/v1/decide", req(`,"route":"auto","model":"m1"`, hotQ))
	if code != 400 || errClass(m) != api.ErrRequestInvalid {
		t.Fatalf("%d %v", code, m)
	}
	code, m = call("POST", "/v1/decide/batch", `{"schema":"hachidori.v1","model":"m1","requests":[`+req(`,"route":"auto"`, hotQ)+`]}`)
	if code != 400 || errClass(m) != api.ErrRequestInvalid {
		t.Fatalf("batch %d %v", code, m)
	}
	code, m = call("POST", "/v1/decide", req(`,"route":"smart"`, hotQ))
	if code != 400 || errClass(m) != api.ErrRequestInvalid {
		t.Fatalf("unknown route %d %v", code, m)
	}
	if len(f.direct) != 0 || f.def != 0 {
		t.Fatalf("a refused request reached a resident: %v %d", f.direct, f.def)
	}
}

func TestAutoRouteWithoutAPolicyIsRefusedNotAnsweredByTheDefaultResident(t *testing.T) {
	f, call := newRoutedFake(t, false)
	code, m := call("POST", "/v1/decide", req(`,"route":"auto"`, hotQ))
	if code != 424 || errClass(m) != api.ErrRoutingFailed || !strings.Contains(m["error"].(map[string]any)["message"].(string), route.CodeNoPolicy) {
		t.Fatalf("%d %v", code, m)
	}
	if f.def != 0 || len(f.direct) != 0 {
		t.Fatalf("a refused routed request reached a resident: %v %d", f.direct, f.def)
	}
	// A single worker (no AutoRouting at all) refuses the same way.
	h := Handler(&fake{ready: true}, Runtime{ModelID: "m1"})
	if rec, m := do(h, "POST", "/v1/decide", req(`,"route":"auto"`, hotQ)); rec.Code != 424 || errClass(m) != api.ErrRoutingFailed {
		t.Fatalf("%d %v", rec.Code, m)
	}
}

func TestRoutedFailureIsAnExplicitErrorWithNoResults(t *testing.T) {
	f, call := newRoutedFake(t, true)
	f.fail["m2"] = &worker.RequestError{Class: api.ErrNotReady, Message: "model m2: starting"}
	code, m := call("POST", "/v1/decide", req(`,"route":"auto"`, hotQ, coldQ))
	msg, _ := m["error"].(map[string]any)["message"].(string)
	if code != 424 || errClass(m) != api.ErrRoutingFailed || !strings.HasPrefix(msg, route.CodeRequiredHandoffFailed+": model m2") {
		t.Fatalf("%d %v", code, m)
	}
	if m["results"] != nil || m["routing"] != nil {
		t.Fatalf("a routing failure carries no results: %v", m)
	}
	// One unavailable resident does not corrupt the other: m1 is still served.
	if code, m := call("POST", "/v1/decide", req(`,"model":"m1"`, hotQ)); code != 200 || m["served"].(map[string]any)["model"] != "m1" {
		t.Fatalf("%d %v", code, m)
	}
	f.fail["m1"] = errors.New("down")
	code, m = call("POST", "/v1/decide", req(`,"route":"auto"`, coldQ))
	if msg, _ := m["error"].(map[string]any)["message"].(string); code != 424 || !strings.HasPrefix(msg, route.CodeFirstPathFailed+": model m1") {
		t.Fatalf("%d %v", code, m)
	}
}

func TestStatusReportsRoutingCountersOnlyWhenAPolicyIsBound(t *testing.T) {
	_, call := newRoutedFake(t, true)
	if _, m := call("GET", "/v1/status", ""); m["routing"] == nil || m["routing"].(map[string]any)["requests"] != float64(0) {
		t.Fatalf("%v", m)
	}
	call("POST", "/v1/decide", req(`,"route":"auto"`, hotQ, coldQ))
	_, m := call("GET", "/v1/status", "")
	st := m["routing"].(map[string]any)
	if st["requests"] != float64(1) || st["handoffs"] != float64(1) || st["questions"] != float64(2) ||
		st["reasons"].(map[string]any)[api.ReasonHandoffLowConf] != float64(1) {
		t.Fatalf("%v", st)
	}
	var m1 map[string]any
	for _, p := range st["providers"].([]any) {
		if p.(map[string]any)["model"] == "m1" {
			m1 = p.(map[string]any)
		}
	}
	if m1["first_path_questions"] != float64(2) || m1["final_results"] != float64(1) || m1["inference_ms_total"] != float64(2) {
		t.Fatalf("%v", m1)
	}
	_, nm := newRoutedFake(t, false)
	if _, m := nm("GET", "/v1/status", ""); m["routing"] != nil {
		t.Fatalf("no policy, no routing section: %v", m)
	}
}

func TestOpenAPIRoutingMatchesHandlers(t *testing.T) {
	d := loadDoc(t)
	c := checker{t, d}
	validate := func(what string, schema map[string]any, v any) {
		t.Helper()
		if errs := c.validate(schema, v); len(errs) > 0 {
			raw, _ := json.Marshal(v)
			t.Errorf("%s violates the document: %v\n%s", what, errs, raw)
		}
	}
	respSchema := func(path string, code string) map[string]any {
		return at(t, d, "paths", path, "post", "responses", code, "content", "application/json", "schema").(map[string]any)
	}
	_, call := newRoutedFake(t, true)

	auto := req(`,"route":"auto"`, hotQ, coldQ)
	var rq any
	_ = json.Unmarshal([]byte(auto), &rq)
	validate("routed request", c.schema("DecideRequest"), rq)
	batch := `{"schema":"hachidori.v1","route":"auto","requests":[` + req("", hotQ) + `]}`
	_ = json.Unmarshal([]byte(batch), &rq)
	validate("routed batch", c.schema("BatchRequest"), rq)
	// route is const "auto": the document refuses anything else.
	_ = json.Unmarshal([]byte(req(`,"route":"smart"`, hotQ)), &rq)
	if errs := c.validate(c.schema("DecideRequest"), rq); len(errs) == 0 {
		t.Error("the document accepts a route other than auto")
	}

	_, m := call("POST", "/v1/decide", auto)
	validate("routed decide response", respSchema("/v1/decide", "200"), m)
	_, m = call("POST", "/v1/decide/batch", batch)
	validate("routed batch response", respSchema("/v1/decide/batch", "200"), m)
	_, m = call("GET", "/v1/status", "")
	validate("status with routing", at(t, d, "paths", "/v1/status", "get", "responses", "200", "content", "application/json", "schema").(map[string]any), m)

	// Every reason code the router can produce is documented, and only those.
	enum := at(t, d, "components", "schemas", "RoutedResult", "properties", "reason", "enum").([]any)
	var documented []string
	for _, e := range enum {
		documented = append(documented, e.(string))
	}
	want := []string{api.ReasonFirstPathKept, api.ReasonFirstPathOnly, api.ReasonHandoffAlways, api.ReasonHandoffChoice,
		api.ReasonHandoffFailed, api.ReasonHandoffLowConf, api.ReasonHandoffLowMargin}
	if !sameSet(documented, want) {
		t.Errorf("documented reasons %v, implemented %v", documented, want)
	}
}

func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	m := map[string]bool{}
	for _, s := range a {
		m[s] = true
	}
	for _, s := range b {
		if !m[s] {
			return false
		}
	}
	return true
}
