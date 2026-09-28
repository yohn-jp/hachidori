package api

import (
	"strings"
	"testing"
)

func valid() DecideRequest {
	return DecideRequest{Schema: SchemaV1, State: "s", Questions: []Question{
		{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}, Descriptions: map[string]string{"yes": "y"}}}}
}

func TestValidate(t *testing.T) {
	if err := func() *DecideRequest { r := valid(); return &r }().Validate(); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*DecideRequest){
		"schema":           func(r *DecideRequest) { r.Schema = "v0" },
		"empty state":      func(r *DecideRequest) { r.State = " " },
		"large state":      func(r *DecideRequest) { r.State = strings.Repeat("x", MaxStateBytes+1) },
		"no questions":     func(r *DecideRequest) { r.Questions = nil },
		"type":             func(r *DecideRequest) { r.Questions[0].Type = "score" },
		"one choice":       func(r *DecideRequest) { r.Questions[0].Choices = []string{"yes"} },
		"dup choice":       func(r *DecideRequest) { r.Questions[0].Choices = []string{"a", "a"} },
		"no instructions":  func(r *DecideRequest) { r.Questions[0].Instructions = "" },
		"dup id":           func(r *DecideRequest) { r.Questions = append(r.Questions, r.Questions[0]) },
		"unknown describe": func(r *DecideRequest) { r.Questions[0].Descriptions = map[string]string{"maybe": "m"} },
	}
	for name, mut := range cases {
		r := valid()
		r.Questions = append([]Question(nil), r.Questions...)
		mut(&r)
		if r.Validate() == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestBatchValidate(t *testing.T) {
	r := valid()
	r.Schema = ""
	b := BatchRequest{Schema: SchemaV1, Requests: []DecideRequest{r}}
	if err := b.Validate(); err != nil {
		t.Fatal(err)
	}
	if (&BatchRequest{Schema: SchemaV1}).Validate() == nil {
		t.Fatal("empty batch accepted")
	}
}
