package question

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
)

func def() Definition {
	return Definition{Schema: Schema, ID: "scope_expansion", Version: 2, Type: "choice",
		Instructions: "Did the agent leave scope?", Choices: []string{"yes", "no"},
		Descriptions: map[string]string{"yes": "touched other files", "no": "stayed in scope"}}
}

func TestCompileExact(t *testing.T) {
	d := def()
	if err := d.Validate(); err != nil {
		t.Fatal(err)
	}
	want := api.Question{ID: "scope_expansion", Type: "choice", Instructions: "Did the agent leave scope?",
		Choices: []string{"yes", "no"}, Descriptions: map[string]string{"yes": "touched other files", "no": "stayed in scope"}}
	got := d.Compile()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("compile = %+v, want %+v", got, want)
	}
	// Compiled question is independent of the definition's slices/maps.
	got.Choices[0], got.Descriptions["yes"] = "x", "x"
	if d.Choices[0] != "yes" || d.Descriptions["yes"] != "touched other files" {
		t.Fatal("compile aliases definition storage")
	}
	d.Descriptions = map[string]string{}
	if d.Compile().Descriptions != nil {
		t.Fatal("empty descriptions must compile to none")
	}
	req := api.DecideRequest{Schema: api.SchemaV1, State: "s", Questions: []api.Question{def().Compile()}}
	if err := req.Validate(); err != nil {
		t.Fatalf("compiled question rejected by v1: %v", err)
	}
}

func TestValidateMatchesAPI(t *testing.T) {
	cases := map[string]func(*Definition){
		"type":             func(d *Definition) { d.Type = "score" },
		"empty instr":      func(d *Definition) { d.Instructions = " \n" },
		"long instr":       func(d *Definition) { d.Instructions = strings.Repeat("x", api.MaxInstructionSize+1) },
		"one choice":       func(d *Definition) { d.Choices = []string{"yes"} },
		"too many choices": func(d *Definition) { d.Choices = make([]string, api.MaxChoices+1) },
		"dup choice":       func(d *Definition) { d.Choices = []string{"yes", "yes"}; d.Descriptions = nil },
		"empty choice":     func(d *Definition) { d.Choices = []string{"yes", ""}; d.Descriptions = nil },
		"unknown describe": func(d *Definition) { d.Descriptions = map[string]string{"maybe": "m"} },
	}
	for name, mut := range cases {
		d := def()
		mut(&d)
		err := d.Validate()
		if err == nil {
			t.Errorf("%s: accepted", name)
			continue
		}
		// Same rule, same message as the v1 request validator.
		req := api.DecideRequest{Schema: api.SchemaV1, State: "s", Questions: []api.Question{d.Compile()}}
		apiErr := req.Validate()
		if apiErr == nil || apiErr.Error() != err.Error() {
			t.Errorf("%s: definition error %q, api error %v", name, err, apiErr)
		}
	}
	for name, mut := range map[string]func(*Definition){
		"schema":     func(d *Definition) { d.Schema = "" },
		"no id":      func(d *Definition) { d.ID = "" },
		"no version": func(d *Definition) { d.Version = 0 },
		"neg ver":    func(d *Definition) { d.Version = -1 },
	} {
		d := def()
		mut(&d)
		if d.Validate() == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestDigestDeterministic(t *testing.T) {
	a, b := def(), def()
	// Map insertion order must not matter.
	b.Descriptions = map[string]string{}
	b.Descriptions["no"] = "stayed in scope"
	b.Descriptions["yes"] = "touched other files"
	for i := 0; i < 20; i++ {
		if a.Digest() != b.Digest() || a.Digest() != def().Digest() {
			t.Fatal("digest not deterministic")
		}
	}
	if !strings.HasPrefix(a.Digest(), "sha256:") || len(a.Digest()) != 7+64 {
		t.Fatalf("digest format %q", a.Digest())
	}
	if id := a.Identity(); id.ID != a.ID || id.Version != a.Version || id.Digest != a.Digest() {
		t.Fatalf("identity %+v", id)
	}
	// Absent and empty descriptions are the same content.
	c, e := def(), def()
	c.Descriptions, e.Descriptions = nil, map[string]string{}
	if c.Digest() != e.Digest() {
		t.Fatal("nil vs empty descriptions changed digest")
	}
}

func TestDigestSemanticChanges(t *testing.T) {
	base := def().Digest()
	for name, mut := range map[string]func(*Definition){
		"id":           func(d *Definition) { d.ID = "scope_expansion_v" },
		"version":      func(d *Definition) { d.Version = 3 },
		"instructions": func(d *Definition) { d.Instructions += " " },
		"choice label": func(d *Definition) { d.Choices = []string{"yes", "No"}; d.Descriptions = nil },
		"choice order": func(d *Definition) { d.Choices = []string{"no", "yes"} },
		"add choice":   func(d *Definition) { d.Choices = append(d.Choices, "unclear") },
		"description":  func(d *Definition) { d.Descriptions["yes"] = "changed" },
		"drop desc":    func(d *Definition) { delete(d.Descriptions, "no") },
		"no desc":      func(d *Definition) { d.Descriptions = nil },
	} {
		d := def()
		d.Descriptions = map[string]string{"yes": "touched other files", "no": "stayed in scope"}
		mut(&d)
		if d.Digest() == base {
			t.Errorf("%s: digest unchanged", name)
		}
	}
}

func TestFormattingAndPathNoise(t *testing.T) {
	dir := t.TempDir()
	compact := `{"schema":"hachidori.question.v1","id":"q","version":1,"type":"choice","instructions":"i?","choices":["yes","no"],"descriptions":{"yes":"y","no":"n"}}`
	pretty := "{\n  \"descriptions\": {\"no\": \"n\",\n \"yes\": \"y\"},\n\t\"choices\" : [ \"yes\" , \"no\" ],\n  \"instructions\": \"i\\u003f\",\n  \"type\": \"choice\", \"version\": 1, \"id\": \"q\",\n  \"schema\": \"hachidori.question.v1\"\n}\n"
	os.MkdirAll(filepath.Join(dir, "a"), 0o755)
	os.MkdirAll(filepath.Join(dir, "b", "nested"), 0o755)
	p1, p2 := filepath.Join(dir, "a", "one.json"), filepath.Join(dir, "b", "nested", "other-name.json")
	os.WriteFile(p1, []byte(compact), 0o644)
	os.WriteFile(p2, []byte(pretty), 0o644)
	s1, err := Load(p1)
	if err != nil {
		t.Fatal(err)
	}
	s2, err := Load(p2)
	if err != nil {
		t.Fatal(err)
	}
	d1, d2 := s1.Definitions()[0], s2.Definitions()[0]
	if d1.Digest() != d2.Digest() || !reflect.DeepEqual(d1.Compile(), d2.Compile()) {
		t.Fatalf("formatting/path changed identity: %s vs %s", d1.Digest(), d2.Digest())
	}
	if strings.Contains(string(d1.Canonical()), dir) || strings.Contains(string(d1.Canonical()), "one.json") {
		t.Fatal("path leaked into canonical form")
	}
}

func TestParseRejects(t *testing.T) {
	for name, doc := range map[string]string{
		"unknown field": `{"schema":"hachidori.question.v1","id":"q","version":1,"type":"choice","instructions":"i","choices":["a","b"],"extends":"x"}`,
		"string ver":    `{"schema":"hachidori.question.v1","id":"q","version":"1","type":"choice","instructions":"i","choices":["a","b"]}`,
		"no version":    `{"schema":"hachidori.question.v1","id":"q","type":"choice","instructions":"i","choices":["a","b"]}`,
		"trailing":      `{"schema":"hachidori.question.v1","id":"q","version":1,"type":"choice","instructions":"i","choices":["a","b"]} {}`,
		"bad schema":    `{"schema":"hachidori.v1","id":"q","version":1,"type":"choice","instructions":"i","choices":["a","b"]}`,
	} {
		if _, err := Parse([]byte(doc)); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestSetResolve(t *testing.T) {
	dir := t.TempDir()
	v1, v2 := def(), def()
	v1.Version, v2.Version = 1, 2
	v2.Instructions = "changed"
	for name, d := range map[string]Definition{"v1.json": v1, "v2.json": v2} {
		os.WriteFile(filepath.Join(dir, name), d.Canonical(), 0o644)
	}
	os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignored"), 0o644)
	s, err := Load(dir)
	if err != nil || s.Len() != 2 {
		t.Fatal(s, err)
	}
	got, err := s.Resolve(Ref{ID: v2.ID, Version: 2})
	if err != nil || got.Instructions != "changed" {
		t.Fatal(got, err)
	}
	if _, err := s.Resolve(Ref{ID: v1.ID, Version: 1, Digest: v1.Digest()}); err != nil {
		t.Fatal(err)
	}
	for name, r := range map[string]Ref{
		"missing id":      {ID: "nope", Version: 1},
		"missing version": {ID: v1.ID, Version: 9},
		"digest mismatch": {ID: v1.ID, Version: 1, Digest: v2.Digest()},
	} {
		if _, err := s.Resolve(r); err == nil {
			t.Errorf("%s resolved", name)
		}
	}
	var none *Set
	if _, err := none.Resolve(Ref{ID: "q", Version: 1}); err == nil {
		t.Error("nil set resolved")
	}
	// Same (id, version) in two files fails, even with identical content.
	dup := filepath.Join(t.TempDir(), "copy.json")
	os.WriteFile(dup, v1.Canonical(), 0o644)
	if _, err := Load(dir, dup); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate accepted: %v", err)
	}
}

func TestRepositoryDefinitions(t *testing.T) {
	s, err := Load("../../testdata/questions")
	if err != nil || s.Len() == 0 {
		t.Fatal(s, err)
	}
}
