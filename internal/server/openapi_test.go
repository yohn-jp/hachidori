package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func loadDoc(t *testing.T) map[string]any {
	t.Helper()
	var d map[string]any
	if err := json.Unmarshal(openAPIDocument, &d); err != nil {
		t.Fatalf("document is not JSON: %v", err)
	}
	return d
}

func at(t *testing.T, v any, path ...string) any {
	t.Helper()
	for _, p := range path {
		m, ok := v.(map[string]any)
		if !ok {
			t.Fatalf("%v: not an object at %q", path, p)
		}
		if v, ok = m[p]; !ok {
			t.Fatalf("%v: missing %q", path, p)
		}
	}
	return v
}

// --- a deliberately small JSON Schema evaluator ---------------------------

// The document uses a closed vocabulary of JSON Schema keywords. check fails
// on any keyword it does not evaluate, so a constraint added to the document
// can never be silently ignored by these tests.
var annotationKeywords = map[string]bool{"description": true, "format": true, "title": true}

type checker struct {
	t   *testing.T
	doc map[string]any
}

func (c checker) resolve(s map[string]any) map[string]any {
	r, ok := s["$ref"].(string)
	if !ok {
		return s
	}
	const prefix = "#/components/schemas/"
	if !strings.HasPrefix(r, prefix) {
		c.t.Fatalf("unsupported $ref %q", r)
	}
	return at(c.t, c.doc, "components", "schemas", strings.TrimPrefix(r, prefix)).(map[string]any)
}

// validate returns the violations of v against schema s.
func (c checker) validate(s map[string]any, v any) []string {
	s = c.resolve(s)
	var errs []string
	fail := func(f string, a ...any) { errs = append(errs, fmt.Sprintf(f, a...)) }
	for k, want := range s {
		switch k {
		case "$ref", "description", "format", "title":
			// annotations and the resolved reference
		case "type":
			if !typeOK(want.(string), v) {
				fail("want type %v", want)
				return errs
			}
		case "const":
			if !reflect.DeepEqual(want, v) {
				fail("want const %v, got %v", want, v)
			}
		case "enum":
			found := false
			for _, e := range want.([]any) {
				found = found || reflect.DeepEqual(e, v)
			}
			if !found {
				fail("%v not in enum %v", v, want)
			}
		case "minLength", "maxLength":
			if str, ok := v.(string); ok {
				n, lim := utf8.RuneCountInString(str), int(want.(float64))
				if k == "minLength" && n < lim || k == "maxLength" && n > lim {
					fail("%s %d violated by %d characters", k, lim, n)
				}
			}
		case "minimum", "maximum":
			if f, ok := v.(float64); ok && (k == "minimum" && f < want.(float64) || k == "maximum" && f > want.(float64)) {
				fail("%s %v violated by %v", k, want, f)
			}
		case "minItems", "maxItems":
			if a, ok := v.([]any); ok {
				lim := int(want.(float64))
				if k == "minItems" && len(a) < lim || k == "maxItems" && len(a) > lim {
					fail("%s %d violated by %d items", k, lim, len(a))
				}
			}
		case "uniqueItems":
			if a, ok := v.([]any); ok && want == true {
				seen := map[string]bool{}
				for _, e := range a {
					b, _ := json.Marshal(e)
					if seen[string(b)] {
						fail("duplicate item %s", b)
					}
					seen[string(b)] = true
				}
			}
		case "items":
			if a, ok := v.([]any); ok {
				for i, e := range a {
					for _, m := range c.validate(want.(map[string]any), e) {
						fail("[%d]: %s", i, m)
					}
				}
			}
		case "required":
			if m, ok := v.(map[string]any); ok {
				for _, r := range want.([]any) {
					if _, has := m[r.(string)]; !has {
						fail("missing required %q", r)
					}
				}
			}
		case "properties", "additionalProperties":
			// evaluated together below
		case "allOf":
			for _, sub := range want.([]any) {
				errs = append(errs, c.validate(sub.(map[string]any), v)...)
			}
		default:
			c.t.Fatalf("test evaluator does not support keyword %q", k)
		}
	}
	if m, ok := v.(map[string]any); ok {
		props, _ := s["properties"].(map[string]any)
		for k, val := range m {
			if p, ok := props[k]; ok {
				for _, e := range c.validate(p.(map[string]any), val) {
					fail("%s: %s", k, e)
				}
				continue
			}
			switch ap := s["additionalProperties"].(type) {
			case bool:
				if !ap {
					fail("unknown property %q", k)
				}
			case map[string]any:
				for _, e := range c.validate(ap, val) {
					fail("%s: %s", k, e)
				}
			}
		}
	}
	return errs
}

var evaluated = map[string]bool{"$ref": true, "type": true, "const": true, "enum": true, "minLength": true, "maxLength": true,
	"minimum": true, "maximum": true, "minItems": true, "maxItems": true, "uniqueItems": true, "items": true, "required": true,
	"properties": true, "additionalProperties": true, "allOf": true}

// lint fails on a schema keyword that validate does not evaluate.
func lint(t *testing.T, where string, v any) {
	t.Helper()
	s, ok := v.(map[string]any)
	if !ok {
		return
	}
	for k, sub := range s {
		switch {
		case annotationKeywords[k]:
		case !evaluated[k]:
			t.Errorf("%s: keyword %q is not evaluated by the tests", where, k)
		case k == "properties":
			for n, p := range sub.(map[string]any) {
				lint(t, where+"."+n, p)
			}
		case k == "items" || k == "additionalProperties":
			lint(t, where+"."+k, sub)
		case k == "allOf":
			for _, e := range sub.([]any) {
				lint(t, where+".allOf", e)
			}
		}
	}
}

func typeOK(typ string, v any) bool {
	switch typ {
	case "string":
		_, ok := v.(string)
		return ok
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "number":
		_, ok := v.(float64)
		return ok
	case "integer":
		f, ok := v.(float64)
		return ok && f == math.Trunc(f)
	case "array":
		_, ok := v.([]any)
		return ok
	case "object":
		_, ok := v.(map[string]any)
		return ok
	}
	return false
}

func (c checker) schema(name string) map[string]any {
	return at(c.t, c.doc, "components", "schemas", name).(map[string]any)
}

func (c checker) limit(schema string, path ...string) int {
	return int(at(c.t, c.schema(schema), path...).(float64))
}

func asJSON(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// --- the served document ---------------------------------------------------

func TestOpenAPIIsServedLocally(t *testing.T) {
	// Not ready, and a runtime whose identity must not leak into the API schema.
	rt := Runtime{Home: "/srv/hachidori", Runtime: "0.0.0-runtime", ModelID: "model-xyz", Model: "org-repo/rev-123", Device: "cuda"}
	h := Handler(&fake{}, rt)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", OpenAPIPath, nil)
	req.Host = DefaultListen
	h.ServeHTTP(rec, req)
	if rec.Code != 200 || rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("%d %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	if !bytes.Equal(rec.Body.Bytes(), openAPIDocument) {
		t.Fatal("served document differs from the built document")
	}
	var d map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &d); err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{rt.Home, rt.Runtime, rt.ModelID, rt.Model, "laya", "cuda"} {
		if strings.Contains(strings.ToLower(rec.Body.String()), strings.ToLower(leak)) {
			t.Errorf("document carries runtime/provider detail %q", leak)
		}
	}

	// Existing boundary: host-local, and no cross-origin POST (a POST is not a route).
	rec = httptest.NewRecorder()
	req = httptest.NewRequest("GET", OpenAPIPath, nil)
	req.Host = "attacker.example:7843"
	h.ServeHTTP(rec, req)
	if rec.Code != 403 || strings.Contains(rec.Body.String(), `"openapi"`) {
		t.Fatalf("rebound host: %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", OpenAPIPath, nil)
	req.Host = DefaultListen
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST %s: %d", OpenAPIPath, rec.Code)
	}
}

func TestOpenAPIStructure(t *testing.T) {
	d := loadDoc(t)
	if v, _ := d["openapi"].(string); !strings.HasPrefix(v, "3.1.") {
		t.Fatalf("openapi = %q", v)
	}
	if at(t, d, "info", "title") == "" || at(t, d, "info", "version") != api.SchemaV1 {
		t.Fatalf("info = %v", d["info"])
	}
	if at(t, d, "servers").([]any)[0].(map[string]any)["url"] != "http://"+DefaultListen {
		t.Fatalf("servers = %v", d["servers"])
	}

	// The documented surface is exactly the public one.
	var ops []string
	ids := map[string]bool{}
	for path, item := range at(t, d, "paths").(map[string]any) {
		for method, o := range item.(map[string]any) {
			op := o.(map[string]any)
			ops = append(ops, strings.ToUpper(method)+" "+path)
			if id, _ := op["operationId"].(string); id == "" || ids[id] {
				t.Errorf("%s %s: operationId %q missing or duplicated", method, path, id)
			} else {
				ids[id] = true
			}
			if _, ok := op["responses"].(map[string]any); !ok {
				t.Errorf("%s %s: no responses", method, path)
			}
			if _, hasBody := op["requestBody"]; hasBody != (method == "post") {
				t.Errorf("%s %s: requestBody presence", method, path)
			}
		}
	}
	sort.Strings(ops)
	want := []string{"GET /health", "GET /openapi.json", "GET /v1/status", "POST /v1/decide", "POST /v1/decide/batch"}
	if !reflect.DeepEqual(ops, want) {
		t.Fatalf("operations = %v, want %v", ops, want)
	}

	// Every documented operation is implemented, and other methods are not.
	h := Handler(&fake{ready: true}, Runtime{})
	for _, op := range ops {
		method, path, _ := strings.Cut(op, " ")
		rec, _ := do(h, method, path, "{}")
		if rec.Code == http.StatusNotFound || rec.Code == http.StatusMethodNotAllowed {
			t.Errorf("%s documented but answers %d", op, rec.Code)
		}
	}

	// Every $ref resolves and every component schema is used.
	used := map[string]bool{}
	var walk func(v any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			if r, ok := x["$ref"].(string); ok {
				name := strings.TrimPrefix(r, "#/components/schemas/")
				if name == r {
					t.Errorf("unsupported $ref %q", r)
				}
				at(t, d, "components", "schemas", name)
				used[name] = true
			}
			for _, e := range x {
				walk(e)
			}
		case []any:
			for _, e := range x {
				walk(e)
			}
		}
	}
	walk(d["paths"])
	// Components reach each other; follow references until stable.
	for changed := true; changed; {
		changed = false
		for name := range used {
			before := len(used)
			walk(at(t, d, "components", "schemas", name))
			changed = changed || len(used) != before
		}
	}
	for name := range at(t, d, "components", "schemas").(map[string]any) {
		if !used[name] {
			t.Errorf("component schema %q is unreferenced", name)
		}
	}

	// The evaluator understands every keyword of every schema.
	for name, s := range at(t, d, "components", "schemas").(map[string]any) {
		lint(t, name, s)
	}
	for path, item := range at(t, d, "paths").(map[string]any) {
		for method, op := range item.(map[string]any) {
			for code, r := range op.(map[string]any)["responses"].(map[string]any) {
				for _, media := range r.(map[string]any)["content"].(map[string]any) {
					lint(t, method+" "+path+" "+code, media.(map[string]any)["schema"])
				}
			}
		}
	}
}

// --- schemas against the Go types -----------------------------------------

func jsonFields(t reflect.Type) (props map[string]reflect.Type, required map[string]bool) {
	props, required = map[string]reflect.Type{}, map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		name, opts, _ := strings.Cut(t.Field(i).Tag.Get("json"), ",")
		props[name] = t.Field(i).Type
		required[name] = !strings.Contains(opts, "omitempty")
	}
	return
}

func TestOpenAPISchemasMatchGoTypes(t *testing.T) {
	d := loadDoc(t)
	c := checker{t, d}
	cases := []struct {
		schema   string
		typ      reflect.Type
		optional []string // properties the contract relaxes versus the Go tag
	}{
		{"Question", reflect.TypeFor[api.Question](), nil},
		{"DecideRequest", reflect.TypeFor[api.DecideRequest](), nil},
		{"BatchItem", reflect.TypeFor[api.DecideRequest](), []string{"schema"}},
		{"Result", reflect.TypeFor[api.Result](), nil},
		{"Timing", reflect.TypeFor[api.Timing](), nil},
		{"DecideResponse", reflect.TypeFor[api.DecideResponse](), nil},
		{"BatchRequest", reflect.TypeFor[api.BatchRequest](), nil},
		{"BatchResponse", reflect.TypeFor[api.BatchResponse](), nil},
		{"ErrorBody", reflect.TypeFor[api.ErrorBody](), nil},
		{"ErrorInfo", reflect.TypeFor[api.ErrorInfo](), nil},
		{"Health", reflect.TypeFor[api.Health](), nil},
		{"Status", reflect.TypeFor[Status](), nil},
		{"Runtime", reflect.TypeFor[Runtime](), nil},
		{"Served", reflect.TypeFor[api.Served](), nil},
		{"ResidentStatus", reflect.TypeFor[ResidentStatus](), nil},
		{"Worker", reflect.TypeFor[worker.Snapshot](), nil},
		{"Failure", reflect.TypeFor[worker.FailureView](), nil},
	}
	covered := map[string]bool{}
	for _, tc := range cases {
		covered[tc.schema] = true
		s := c.schema(tc.schema)
		props, required := jsonFields(tc.typ)
		for _, o := range tc.optional {
			required[o] = false
		}
		doc := s["properties"].(map[string]any)
		for name, ft := range props {
			p, ok := doc[name].(map[string]any)
			if !ok {
				t.Errorf("%s: Go field %q is not documented", tc.schema, name)
				continue
			}
			if got := schemaKind(c, p); !kindMatches(got, ft) {
				t.Errorf("%s.%s: documented as %s, Go type %s", tc.schema, name, got, ft)
			}
		}
		for name := range doc {
			if _, ok := props[name]; !ok {
				t.Errorf("%s: documented property %q has no Go field", tc.schema, name)
			}
		}
		var req []string
		for _, r := range s["required"].([]any) {
			req = append(req, r.(string))
		}
		var wantReq []string
		for n, r := range required {
			if r {
				wantReq = append(wantReq, n)
			}
		}
		sort.Strings(req)
		sort.Strings(wantReq)
		if !reflect.DeepEqual(req, wantReq) {
			t.Errorf("%s: required %v, Go requires %v", tc.schema, req, wantReq)
		}
	}
	for name := range at(t, d, "components", "schemas").(map[string]any) {
		if !covered[name] {
			t.Errorf("component %q has no Go type in this test", name)
		}
	}
}

// schemaKind is the JSON type a property schema denotes.
func schemaKind(c checker, p map[string]any) string {
	r := c.resolve(p)
	if k, ok := r["type"].(string); ok {
		return k
	}
	return "?"
}

func kindMatches(doc string, t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return doc == "string"
	case reflect.Bool:
		return doc == "boolean"
	case reflect.Float32, reflect.Float64:
		return doc == "number"
	case reflect.Int, reflect.Int32, reflect.Int64:
		return doc == "integer"
	case reflect.Slice:
		return doc == "array"
	case reflect.Map, reflect.Struct:
		return doc == "object"
	}
	return false
}

// --- limits and request validation against the handlers -----------------

func TestOpenAPILimitsMatchHandlers(t *testing.T) {
	d := loadDoc(t)
	c := checker{t, d}
	// The documented limits are the implemented ones.
	for _, l := range []struct {
		got  int
		want int
		name string
	}{
		{c.limit("DecideRequest", "properties", "state", "maxLength"), api.MaxStateBytes, "state"},
		{c.limit("DecideRequest", "properties", "questions", "maxItems"), api.MaxQuestions, "questions"},
		{c.limit("DecideRequest", "properties", "questions", "minItems"), 1, "min questions"},
		{c.limit("Question", "properties", "instructions", "maxLength"), api.MaxInstructionSize, "instructions"},
		{c.limit("Question", "properties", "choices", "maxItems"), api.MaxChoices, "choices"},
		{c.limit("Question", "properties", "choices", "minItems"), 2, "min choices"},
		{c.limit("BatchRequest", "properties", "requests", "maxItems"), api.MaxBatchRequests, "batch"},
		{c.limit("BatchRequest", "properties", "requests", "minItems"), 1, "min batch"},
	} {
		if l.got != l.want {
			t.Errorf("%s: documented %d, implemented %d", l.name, l.got, l.want)
		}
	}

	choices := func(n int) []any {
		out := make([]any, n)
		for i := range out {
			out[i] = "c" + strconv.Itoa(i)
		}
		return out
	}
	question := func(id string) map[string]any {
		return map[string]any{"id": id, "type": "choice", "instructions": "i", "choices": choices(2)}
	}
	questions := func(n int) []any {
		out := make([]any, n)
		for i := range out {
			out[i] = question("q" + strconv.Itoa(i))
		}
		return out
	}
	decide := func() map[string]any {
		return map[string]any{"schema": api.SchemaV1, "state": "s", "questions": questions(1)}
	}
	with := func(mutate func(m map[string]any)) map[string]any {
		m := decide()
		mutate(m)
		return m
	}
	withQ := func(mutate func(q map[string]any)) map[string]any {
		return with(func(m map[string]any) { q := question("q"); mutate(q); m["questions"] = []any{q} })
	}

	type tc struct {
		name string
		body map[string]any
	}
	decideCases := []tc{
		{"minimal", decide()},
		{"options accepted", with(func(m map[string]any) { m["options"] = map[string]any{"x": 1} })},
		{"descriptions", withQ(func(q map[string]any) { q["descriptions"] = map[string]any{"c0": "d"} })},
		{"max questions", with(func(m map[string]any) { m["questions"] = questions(api.MaxQuestions) })},
		{"too many questions", with(func(m map[string]any) { m["questions"] = questions(api.MaxQuestions + 1) })},
		{"no questions", with(func(m map[string]any) { m["questions"] = []any{} })},
		{"max state", with(func(m map[string]any) { m["state"] = strings.Repeat("a", api.MaxStateBytes) })},
		{"state too long", with(func(m map[string]any) { m["state"] = strings.Repeat("a", api.MaxStateBytes+1) })},
		{"empty state", with(func(m map[string]any) { m["state"] = "" })},
		{"missing state", with(func(m map[string]any) { delete(m, "state") })},
		{"missing schema", with(func(m map[string]any) { delete(m, "schema") })},
		{"wrong schema", with(func(m map[string]any) { m["schema"] = "hachidori.v2" })},
		{"unknown field", with(func(m map[string]any) { m["bogus"] = 1 })},
		{"max instructions", withQ(func(q map[string]any) { q["instructions"] = strings.Repeat("a", api.MaxInstructionSize) })},
		{"instructions too long", withQ(func(q map[string]any) { q["instructions"] = strings.Repeat("a", api.MaxInstructionSize+1) })},
		{"empty instructions", withQ(func(q map[string]any) { q["instructions"] = "" })},
		{"max choices", withQ(func(q map[string]any) { q["choices"] = choices(api.MaxChoices) })},
		{"too many choices", withQ(func(q map[string]any) { q["choices"] = choices(api.MaxChoices + 1) })},
		{"one choice", withQ(func(q map[string]any) { q["choices"] = choices(1) })},
		{"duplicate choices", withQ(func(q map[string]any) { q["choices"] = []any{"a", "a"} })},
		{"empty choice label", withQ(func(q map[string]any) { q["choices"] = []any{"a", ""} })},
		{"non-choice type", withQ(func(q map[string]any) { q["type"] = "scalar" })},
		{"missing type", withQ(func(q map[string]any) { delete(q, "type") })},
		{"missing id", withQ(func(q map[string]any) { delete(q, "id") })},
		{"empty id", withQ(func(q map[string]any) { q["id"] = "" })},
		{"unknown question field", withQ(func(q map[string]any) { q["bogus"] = 1 })},
		{"non-string description", withQ(func(q map[string]any) { q["descriptions"] = map[string]any{"c0": 1} })},
	}
	h := Handler(&fake{ready: true}, Runtime{})
	post := func(path string, v any) int {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		rec, _ := do(h, "POST", path, string(b))
		return rec.Code
	}
	// Everything the document admits as a question type is accepted.
	typ := c.schema("Question")["properties"].(map[string]any)["type"].(map[string]any)
	admitted := []any{typ["const"]}
	if e, ok := typ["enum"].([]any); ok {
		admitted = e
	}
	if len(admitted) != 1 || admitted[0] != "choice" {
		t.Errorf("question type admits %v; only choice is implemented", admitted)
	}
	for _, v := range admitted {
		if got := post("/v1/decide", withQ(func(q map[string]any) { q["type"] = v })); got != 200 {
			t.Errorf("documented question type %v answered %d", v, got)
		}
	}
	for _, x := range decideCases {
		valid := len(c.validate(c.schema("DecideRequest"), asJSON(t, x.body))) == 0
		want := map[bool]int{true: 200, false: 400}[valid]
		if got := post("/v1/decide", x.body); got != want {
			t.Errorf("decide %s: document says valid=%v, handler answered %d", x.name, valid, got)
		}
		// The same entry inside a batch obeys the same rules.
		batch := map[string]any{"schema": api.SchemaV1, "requests": []any{x.body}}
		valid = len(c.validate(c.schema("BatchRequest"), asJSON(t, batch))) == 0
		want = map[bool]int{true: 200, false: 400}[valid]
		if got := post("/v1/decide/batch", batch); got != want {
			t.Errorf("batch of %s: document says valid=%v, handler answered %d", x.name, valid, got)
		}
	}

	batchCases := []tc{
		{"one request", map[string]any{"schema": api.SchemaV1, "requests": []any{decide()}}},
		{"max requests", map[string]any{"schema": api.SchemaV1, "requests": repeat(decide(), api.MaxBatchRequests)}},
		{"too many requests", map[string]any{"schema": api.SchemaV1, "requests": repeat(decide(), api.MaxBatchRequests+1)}},
		{"no requests", map[string]any{"schema": api.SchemaV1, "requests": []any{}}},
		{"missing schema", map[string]any{"requests": []any{decide()}}},
		{"unknown field", map[string]any{"schema": api.SchemaV1, "requests": []any{decide()}, "bogus": 1}},
		{"entry without schema", map[string]any{"schema": api.SchemaV1, "requests": []any{with(func(m map[string]any) { delete(m, "schema") })}}},
		{"entry with wrong schema", map[string]any{"schema": api.SchemaV1, "requests": []any{with(func(m map[string]any) { m["schema"] = "x" })}}},
	}
	for _, x := range batchCases {
		valid := len(c.validate(c.schema("BatchRequest"), asJSON(t, x.body))) == 0
		want := map[bool]int{true: 200, false: 400}[valid]
		if got := post("/v1/decide/batch", x.body); got != want {
			t.Errorf("batch %s: document says valid=%v, handler answered %d", x.name, valid, got)
		}
	}
}

func repeat(v map[string]any, n int) []any {
	out := make([]any, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// --- responses against the document ------------------------------------

type fullFake struct {
	fake
	snap worker.Snapshot
}

func (f *fullFake) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	if f.err != nil {
		return nil, 0, f.err
	}
	var out [][]api.Result
	for _, it := range items {
		var rs []api.Result
		for _, q := range it.Questions {
			p := map[string]float64{}
			for _, c := range q.Choices {
				p[c] = 1 / float64(len(q.Choices))
			}
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: p[q.Choices[0]], Probabilities: p})
		}
		out = append(out, rs)
	}
	return out, 3.5, nil
}
func (f *fullFake) Snapshot() worker.Snapshot { return f.snap }

func TestOpenAPIResponsesMatchHandlers(t *testing.T) {
	d := loadDoc(t)
	c := checker{t, d}
	snap := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, PID: 42, Starts: 2, Restarts: 1,
		ReadySince: "2026-01-02T03:04:05Z", Info: worker.Info{"device": "cpu", "load_ms": 12.5},
		Accelerator: map[string]any{"used_mb": 1.5}, AcceleratorStale: true,
		LastFailure: &worker.FailureView{Class: worker.ClassCrash, Message: "exited", Stderr: []string{"x"}},
		Requests:    7, Errors: map[string]int64{api.ErrWorkerFailure: 1}, QueueDepth: 1, QueueLimit: 64, LatencyP50MS: 1.5, LatencyP95MS: 2.5}
	f := &fullFake{fake: fake{ready: true}, snap: snap}
	h := Handler(f, Runtime{Home: "/h", Runtime: "r", ModelID: "m", Model: "o/r", Device: "cpu"})

	// responseSchema is the documented schema for a status of an operation.
	responseSchema := func(path, method, code string) map[string]any {
		r := at(t, d, "paths", path, method, "responses", code, "content", "application/json", "schema")
		return r.(map[string]any)
	}
	check := func(what string, schema map[string]any, rec *httptest.ResponseRecorder) {
		t.Helper()
		var v any
		if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
			t.Fatalf("%s: %v: %s", what, err, rec.Body)
		}
		if errs := c.validate(schema, v); len(errs) > 0 {
			t.Errorf("%s: response violates the document: %v\n%s", what, errs, rec.Body)
		}
	}

	two := `{"id":"a","type":"choice","instructions":"i","choices":["x","y"],"descriptions":{"x":"d"}}`
	one := `{"schema":"hachidori.v1","state":"s","questions":[` + two + `]}`
	rec, _ := do(h, "POST", "/v1/decide", one)
	check("decide", responseSchema("/v1/decide", "post", "200"), rec)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"timing"`) {
		t.Fatalf("decide: %d %s", rec.Code, rec.Body)
	}
	rec, _ = do(h, "POST", "/v1/decide/batch", `{"schema":"hachidori.v1","requests":[`+one+`,`+one+`]}`)
	check("batch", responseSchema("/v1/decide/batch", "post", "200"), rec)
	var batch map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &batch)
	if _, has := batch["responses"].([]any)[0].(map[string]any)["timing"]; has {
		t.Error("batch entries carry timing, which the document says they omit")
	}

	rec, _ = do(h, "GET", "/v1/status", "")
	check("status", responseSchema("/v1/status", "get", "200"), rec)
	rec, _ = do(h, "GET", "/health", "")
	check("health 200", responseSchema("/health", "get", "200"), rec)
	f.ready = false
	f.snap = worker.Snapshot{State: worker.StateStarting, Errors: map[string]int64{}}
	rec, _ = do(h, "GET", "/health", "")
	check("health 503", responseSchema("/health", "get", "503"), rec)
	rec, _ = do(h, "GET", "/v1/status", "")
	check("status (minimal snapshot)", responseSchema("/v1/status", "get", "200"), rec)
	f.ready = true

	// Every documented error status is produced by exactly its documented class.
	errs := map[string]error{
		api.ErrRequestInvalid:  nil,
		api.ErrNotReady:        &worker.RequestError{Class: api.ErrNotReady, Message: "m"},
		api.ErrCapacity:        &worker.RequestError{Class: api.ErrCapacity, Message: "m"},
		api.ErrInferenceFailed: &worker.RequestError{Class: api.ErrInferenceFailed, Message: "m"},
		api.ErrWorkerFailure:   &worker.Failure{Class: worker.ClassCrash, Message: "m"},
	}
	for class, status := range statusFor {
		for _, path := range []string{"/v1/decide", "/v1/decide/batch"} {
			body := one
			if strings.HasSuffix(path, "/batch") {
				body = `{"schema":"hachidori.v1","requests":[` + one + `]}`
			}
			if class == api.ErrRequestInvalid {
				body = `{`
			}
			e, ok := errs[class]
			if !ok {
				t.Fatalf("no trigger for documented class %q", class)
			}
			f.err = e
			rec, _ := do(h, "POST", path, body)
			if rec.Code != status {
				t.Errorf("%s %s: %d, documented %d", path, class, rec.Code, status)
			}
			check(path+" "+class, responseSchema(path, "post", strconv.Itoa(status)), rec)
		}
	}
	f.err = nil
	for _, path := range []string{"/v1/decide", "/v1/decide/batch"} {
		codes := map[string]bool{}
		for code := range at(t, d, "paths", path, "post", "responses").(map[string]any) {
			codes[code] = true
		}
		want := map[string]bool{"200": true, "403": true}
		for _, s := range statusFor {
			want[strconv.Itoa(s)] = true
		}
		if !reflect.DeepEqual(codes, want) {
			t.Errorf("%s documents statuses %v, implemented %v", path, codes, want)
		}
	}

	// The documented 403 is the host-local refusal.
	req := httptest.NewRequest("POST", "/v1/decide", strings.NewReader(one))
	req.Host = DefaultListen
	req.Header.Set("Origin", "https://attacker.example")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 403 || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/plain") {
		t.Errorf("403: %d %q", w.Code, w.Header().Get("Content-Type"))
	}
	at(t, d, "paths", "/v1/decide", "post", "responses", "403", "content", "text/plain")
}

// routerFake is a Decider with two residents, enough to pin the documented
// direct-selection and multi-resident status shapes against the handlers.
type routerFake struct {
	fullFake
	residents []ResidentStatus
}

func (r *routerFake) Identity(model string) (api.Served, bool) {
	for _, s := range r.residents {
		if s.Model == model {
			return api.Served{Model: s.Model, Provider: s.Provider}, true
		}
	}
	return api.Served{}, false
}

func (r *routerFake) DecideOn(_ string, items []worker.Item) ([][]api.Result, float64, error) {
	return r.fullFake.Decide(items)
}

func (r *routerFake) ResidentStatuses() []ResidentStatus { return r.residents }

func TestOpenAPIDirectSelectionAndResidentsMatchHandlers(t *testing.T) {
	d := loadDoc(t)
	c := checker{t, d}
	snap := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, PID: 42, Errors: map[string]int64{}, QueueLimit: 64}
	rs := func(model, provider string, def bool) ResidentStatus {
		return ResidentStatus{Model: model, Provider: provider, Default: def, Running: true,
			Status: Status{Schema: api.SchemaV1, Runtime: Runtime{ModelID: model, Device: "cuda"}, Worker: snap}}
	}
	f := &routerFake{fullFake: fullFake{fake: fake{ready: true}, snap: snap},
		residents: []ResidentStatus{rs("m1", "p1", true), rs("m2", "p2", false)}}
	h := Handler(f, Runtime{ModelID: "m1", Device: "cuda"})
	schema := func(path, method string) map[string]any {
		return at(t, d, "paths", path, method, "responses", "200", "content", "application/json", "schema").(map[string]any)
	}
	q := `{"id":"a","type":"choice","instructions":"i","choices":["x","y"]}`
	one := func(model string) string {
		sel := ""
		if model != "" {
			sel = `,"model":"` + model + `"`
		}
		return `{"schema":"hachidori.v1","state":"s","questions":[` + q + `]` + sel + `}`
	}
	validate := func(what string, schema map[string]any, body string) {
		t.Helper()
		var v any
		if err := json.Unmarshal([]byte(body), &v); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if errs := c.validate(schema, v); len(errs) > 0 {
			t.Errorf("%s violates the document: %v\n%s", what, errs, body)
		}
	}
	// Requests with a selector are valid per the document and accepted.
	validate("direct request", c.schema("DecideRequest"), one("m2"))
	batch := `{"schema":"hachidori.v1","model":"m2","requests":[` + one("") + `]}`
	validate("direct batch", c.schema("BatchRequest"), batch)

	rec, m := do(h, "POST", "/v1/decide", one("m2"))
	if served, _ := m["served"].(map[string]any); rec.Code != 200 || served["model"] != "m2" || served["provider"] != "p2" {
		t.Fatalf("direct decide: %d %s", rec.Code, rec.Body)
	}
	validate("direct decide response", schema("/v1/decide", "post"), rec.Body.String())
	rec, m = do(h, "POST", "/v1/decide/batch", batch)
	if served, _ := m["served"].(map[string]any); rec.Code != 200 || served["model"] != "m2" {
		t.Fatalf("direct batch: %d %s", rec.Code, rec.Body)
	}
	validate("direct batch response", schema("/v1/decide/batch", "post"), rec.Body.String())
	rec, m = do(h, "POST", "/v1/decide", one(""))
	if _, has := m["served"]; rec.Code != 200 || has {
		t.Fatalf("default route: %d %s", rec.Code, rec.Body)
	}
	if rec, m = do(h, "POST", "/v1/decide", one("m3")); rec.Code != 400 || errClass(m) != api.ErrRequestInvalid {
		t.Fatalf("unknown resident: %d %s", rec.Code, rec.Body)
	}
	if rec, m = do(h, "POST", "/v1/decide", one("a/b")); rec.Code != 400 {
		t.Fatalf("repository selector: %d %s", rec.Code, rec.Body)
	}

	rec, m = do(h, "GET", "/v1/status", "")
	validate("multi-resident status", schema("/v1/status", "get"), rec.Body.String())
	if res, _ := m["residents"].([]any); len(res) != 2 {
		t.Fatalf("status residents: %s", rec.Body)
	}
}
