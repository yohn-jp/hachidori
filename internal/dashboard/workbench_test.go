package dashboard

import (
	"bytes"
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

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/question"
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
	block chan struct{}
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
	}
	f.bodies = append(f.bodies, b)
	f.paths = append(f.paths, r.Method+" "+r.URL.Path)
	reply, block := f.reply, f.block
	f.mu.Unlock()
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
