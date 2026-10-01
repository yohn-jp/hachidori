package api

import (
	"encoding/json"
	"strings"
	"testing"
)

func ptrTo(s string) *string { return &s }

// The route selector is optional and additive: absent it the wire form is the
// v1 one; present it is exactly "auto" and never combined with a direct model.
func TestRouteSelector(t *testing.T) {
	r := valid()
	if b, _ := json.Marshal(r); strings.Contains(string(b), "route") {
		t.Fatalf("an unrouted request carries a route field: %s", b)
	}
	r.Route = RouteAuto
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"smart", "AUTO", " auto", "laya-base"} {
		r.Route = bad
		if r.Validate() == nil {
			t.Errorf("route %q accepted", bad)
		}
	}
	m := "laya-base"
	r.Route, r.Model = RouteAuto, &m
	if r.Validate() == nil {
		t.Error("route auto with a direct model accepted: a direct request is strict")
	}

	for _, tc := range []struct {
		batch string
		items []string
		want  string
		ok    bool
	}{
		{"", []string{"", ""}, "", true},
		{"auto", []string{"", ""}, "auto", true},
		{"", []string{"auto", "auto"}, "auto", true},
		{"", []string{"", "auto"}, "auto", true},
		{"auto", []string{"smart"}, "", false},
		{"", []string{"auto", "smart"}, "", false},
	} {
		b := BatchRequest{Schema: SchemaV1, Route: tc.batch}
		for _, m := range tc.items {
			r := valid()
			r.Route = m
			b.Requests = append(b.Requests, r)
		}
		got, err := b.RouteMode()
		if (err == nil) != tc.ok || got != tc.want || (b.Validate() == nil) != tc.ok {
			t.Errorf("batch %+v: route %q err %v", tc, got, err)
		}
	}
	// A routed batch cannot also target a resident, on the batch or on an entry.
	b := BatchRequest{Schema: SchemaV1, Route: RouteAuto, Model: ptrTo("laya-base"), Requests: []DecideRequest{valid()}}
	if b.Validate() == nil {
		t.Error("routed batch with a batch-level model accepted")
	}
	e := valid()
	e.Model = ptrTo("laya-base")
	b = BatchRequest{Schema: SchemaV1, Route: RouteAuto, Requests: []DecideRequest{e}}
	if b.Validate() == nil {
		t.Error("routed batch with an entry-level model accepted")
	}
}

// A response without routing keeps the v1 wire form exactly.
func TestUnroutedResponseWireFormIsUnchanged(t *testing.T) {
	resp := DecideResponse{Schema: SchemaV1, Results: []Result{}}
	b, _ := json.Marshal(resp)
	if strings.Contains(string(b), "routing") {
		t.Fatalf("%s", b)
	}
	if b, _ := json.Marshal(BatchResponse{Schema: SchemaV1}); strings.Contains(string(b), "routing") {
		t.Fatalf("%s", b)
	}
}
