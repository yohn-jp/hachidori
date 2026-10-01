package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// directPost sends one HTTP request through the production API handler bound
// to the resident set and decodes the JSON answer.
func directPost(t *testing.T, h http.Handler, method, path, body string) (int, map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = server.DefaultListen
	h.ServeHTTP(rec, req)
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("%s %s: %d %q", method, path, rec.Code, rec.Body)
	}
	return rec.Code, m
}

func decideBody(model, state string) string {
	sel := ""
	if model != "" {
		sel = `,"model":` + strconv.Quote(model)
	}
	return `{"schema":"hachidori.v1","state":` + strconv.Quote(state) + sel +
		`,"questions":[{"id":"q","type":"choice","instructions":"i","choices":["a","b"]}]}`
}

// answer is the fake worker's report of who served: model, pid, sequence.
func answer(t *testing.T, m map[string]any) (model string, pid, seq int) {
	t.Helper()
	rs, _ := m["results"].([]any)
	if len(rs) != 1 {
		t.Fatalf("response %v", m)
	}
	parts := strings.SplitN(rs[0].(map[string]any)["choice"].(string), "|", 4)
	pid, _ = strconv.Atoi(parts[1])
	seq, _ = strconv.Atoi(parts[2])
	return parts[0], pid, seq
}

func servedOf(m map[string]any) (model, provider string, ok bool) {
	s, ok := m["served"].(map[string]any)
	if !ok {
		return "", "", false
	}
	return s["model"].(string), s["provider"].(string), true
}

func errorOf(m map[string]any) (class, msg string) {
	e, _ := m["error"].(map[string]any)
	class, _ = e["class"].(string)
	msg, _ = e["message"].(string)
	return
}

// With both residents READY a caller can target either by catalog identity,
// alternate between them without reloading either, and is told which resident
// answered. A caller that names no model keeps the default route and the
// response it always had.
func TestDirectSelectionAlternatesWithoutReload(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	s.Start()
	bothReady(t, s)
	h := server.Handler(s, s.Runtime())
	pids := map[string]int{modelA: residentState(s, modelA).PID, modelB: residentState(s, modelB).PID}
	seqs := map[string]int{}
	for i := range 5 {
		for _, m := range []string{modelA, modelB} {
			code, body := directPost(t, h, "POST", "/v1/decide", decideBody(m, "s"+strconv.Itoa(i)))
			if code != 200 {
				t.Fatalf("%s: %d %v", m, code, body)
			}
			got, pid, seq := answer(t, body)
			sm, sp, ok := servedOf(body)
			if got != m || pid != pids[m] || !ok || sm != m || sp != "fake-"+m {
				t.Fatalf("direct %s: worker %s pid %d (want %d), served %q/%q/%v", m, got, pid, pids[m], sm, sp, ok)
			}
			if seq != seqs[m]+1 {
				t.Fatalf("%s sequence %d after %d: the model was reloaded", m, seq, seqs[m])
			}
			seqs[m] = seq
		}
	}
	for _, m := range []string{modelA, modelB} {
		if st := residentState(s, m); st.Starts != 1 || st.PID != pids[m] || st.Requests != 5 {
			t.Fatalf("%s: starts %d pid %d (want %d) requests %d", m, st.Starts, st.PID, pids[m], st.Requests)
		}
	}

	// A batch is served by one resident, named on the batch.
	batch := `{"schema":"hachidori.v1","model":` + strconv.Quote(modelB) + `,"requests":[` + decideBody("", "b1") + `,` + decideBody("", "b2") + `]}`
	code, body := directPost(t, h, "POST", "/v1/decide/batch", batch)
	if sm, _, ok := servedOf(body); code != 200 || !ok || sm != modelB {
		t.Fatalf("batch: %d %v", code, body)
	}
	for _, r := range body["responses"].([]any) {
		if got, _, _ := answer(t, r.(map[string]any)); got != modelB {
			t.Fatalf("batch entry served by %s", got)
		}
	}
	// Requests of one batch that name different residents are invalid.
	mixed := `{"schema":"hachidori.v1","requests":[` + decideBody(modelA, "x") + `,` + decideBody(modelB, "y") + `]}`
	if code, body := directPost(t, h, "POST", "/v1/decide/batch", mixed); code != 400 {
		t.Fatalf("mixed batch: %d %v", code, body)
	}

	// No selector: the default resident, with the unchanged response shape.
	code, body = directPost(t, h, "POST", "/v1/decide", decideBody("", "legacy"))
	if got, _, _ := answer(t, body); code != 200 || got != modelA {
		t.Fatalf("default route: %d %v", code, body)
	}
	if _, _, ok := servedOf(body); ok {
		t.Fatalf("default route reports served: %v", body)
	}
	if st := residentState(s, modelA); st.Starts != 1 || st.PID != pids[modelA] {
		t.Fatalf("default route restarted the default resident: %+v", st)
	}
}

// A direct request never lands on another resident: an unknown identity, an
// arbitrary repository string, a stopped resident and a failed resident all
// fail explicitly, and the resident that could have answered is not touched.
func TestDirectSelectionIsStrict(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	s.Start()
	bothReady(t, s)
	h := server.Handler(s, s.Runtime())
	before := residentState(s, modelA).Requests

	for _, bad := range []struct {
		model, class string
		code         int
	}{
		{"openjev", api.ErrRequestInvalid, 400},                         // not a resident catalog identity
		{"manjunathshiva/opendecider-nano", api.ErrRequestInvalid, 400}, // repository string
		{"manjunathshiva/opendecider-nano@abc", api.ErrRequestInvalid, 400},
		{" " + modelB, api.ErrRequestInvalid, 400},
	} {
		code, body := directPost(t, h, "POST", "/v1/decide", decideBody(bad.model, "x"))
		if class, _ := errorOf(body); code != bad.code || class != bad.class {
			t.Fatalf("model %q: %d %v", bad.model, code, body)
		}
	}

	// Operator-stopped resident: explicit not_ready naming it; the other keeps serving.
	if err := s.StopResident(modelB); err != nil {
		t.Fatal(err)
	}
	code, body := directPost(t, h, "POST", "/v1/decide", decideBody(modelB, "x"))
	class, msg := errorOf(body)
	if code != 503 || class != api.ErrNotReady || !strings.Contains(msg, modelB) {
		t.Fatalf("stopped resident: %d %v", code, body)
	}
	if _, _, ok := servedOf(body); ok {
		t.Fatalf("a failed direct request reports a resident: %v", body)
	}
	if got := residentState(s, modelA).Requests; got != before {
		t.Fatalf("a request for %s reached %s (%d -> %d)", modelB, modelA, before, got)
	}
	if code, body := directPost(t, h, "POST", "/v1/decide", decideBody(modelA, "x")); code != 200 {
		t.Fatalf("healthy resident: %d %v", code, body)
	}
	if st := residentState(s, modelB); st.Errors[api.ErrNotReady] != 1 || st.Starts != 1 {
		t.Fatalf("stopped resident was started by a direct request: %+v", st)
	}
}

func TestDirectToFailedResidentFailsAndOthersStayHealthy(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "fatal"))
	s.Start()
	waitFor(t, "B failed", func() bool { return residentState(s, modelB).State == worker.StateFailed })
	waitFor(t, "A ready", func() bool { return residentState(s, modelA).State == worker.StateReady })
	h := server.Handler(s, s.Runtime())

	code, body := directPost(t, h, "POST", "/v1/decide", decideBody(modelB, "x"))
	if class, _ := errorOf(body); code != 503 || class != api.ErrNotReady {
		t.Fatalf("failed resident: %d %v", code, body)
	}
	code, body = directPost(t, h, "POST", "/v1/decide", decideBody(modelA, "x"))
	if got, _, _ := answer(t, body); code != 200 || got != modelA {
		t.Fatalf("healthy resident: %d %v", code, body)
	}

	// /v1/status lists every resident with its own state: B failed, A ready.
	code, st := directPost(t, h, "GET", "/v1/status", "")
	rs, _ := st["residents"].([]any)
	if code != 200 || len(rs) != 2 {
		t.Fatalf("status: %d %v", code, st)
	}
	want := map[string]string{modelA: worker.StateReady, modelB: worker.StateFailed}
	for _, r := range rs {
		r := r.(map[string]any)
		w := r["status"].(map[string]any)["worker"].(map[string]any)
		m := r["model"].(string)
		if w["state"] != want[m] || r["provider"] != "fake-"+m {
			t.Fatalf("resident %s: %v", m, r)
		}
	}
	if w := st["worker"].(map[string]any); w["state"] != worker.StateReady {
		t.Fatalf("a failed resident changed the default resident's status: %v", w)
	}
}

// A single worker serves direct selection of the model it runs and nothing
// else: it never claims another model.
func TestSingleWorkerDirectSelection(t *testing.T) {
	m := residentMember(t, modelA, "ok")
	sup := worker.NewSupervisor(m.Config, noRestart)
	lc := worker.NewLifecycle(t.Context(), sup)
	lc.Start()
	t.Cleanup(lc.Stop)
	waitFor(t, "ready", sup.Ready)
	h := server.Handler(sup, server.Runtime{ModelID: modelA})

	code, body := directPost(t, h, "POST", "/v1/decide", decideBody(modelA, "x"))
	if sm, _, ok := servedOf(body); code != 200 || !ok || sm != modelA {
		t.Fatalf("own model: %d %v", code, body)
	}
	code, body = directPost(t, h, "POST", "/v1/decide", decideBody(modelB, "x"))
	if class, _ := errorOf(body); code != 400 || class != api.ErrRequestInvalid {
		t.Fatalf("other model: %d %v", code, body)
	}
	if st := sup.Snapshot(); st.Requests != 1 {
		t.Fatalf("the single worker answered for another model: %+v", st)
	}
}
