package server

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/requesthistory"
)

func TestRegisteredStateDecisionAndHistory(t *testing.T) {
	f := &fake{ready: true}
	rt := Runtime{Home: t.TempDir()}
	history := requesthistory.New()
	h := HandlerSinceWithHistory(f, rt, time.Now(), history)
	rec, m := do(h, "POST", "/v1/states", `{"schema":"hachidori.v1","state":"policy"}`)
	if rec.Code != 200 {
		t.Fatalf("registration: %d %v", rec.Code, m)
	}
	ref := m["state_ref"].(string)
	if ref != home.StateRef("policy") {
		t.Fatal("identity mismatch")
	}
	rec, m = do(h, "POST", "/v1/states", `{"schema":"hachidori.v1","state":"policy"}`)
	if rec.Code != 200 || m["state_ref"] != ref {
		t.Fatalf("duplicate: %d %v", rec.Code, m)
	}
	request := fmt.Sprintf(`{"schema":"hachidori.v1","state_ref":%q,"questions":[{"id":"q","type":"choice","instructions":"i","choices":["yes","no"]}]}`, ref)
	rec, m = do(h, "POST", "/v1/decide", request)
	if rec.Code != 200 || m["state_ref"] != ref || f.calls != 1 {
		t.Fatalf("reference decision: %d %v", rec.Code, m)
	}
	view := history.List()
	if len(view.Entries) != 1 || view.Entries[0].StateRef != ref || view.Entries[0].StateBytes != len("policy") || view.Entries[0].State != "completed" {
		t.Fatalf("history: %+v", view.Entries)
	}
	rec, m = do(h, "POST", "/v1/decide", strings.Replace(request, `"state_ref":"`+ref+`"`, `"state":"other","state_ref":"`+ref+`"`, 1))
	if rec.Code != 400 || errClass(m) != "request_invalid" || f.calls != 1 {
		t.Fatalf("exclusivity: %d %v", rec.Code, m)
	}
	rec, m = do(h, "POST", "/v1/decide", strings.Replace(request, `"state_ref":`, `"state":"","state_ref":`, 1))
	if rec.Code != 400 || errClass(m) != "request_invalid" || f.calls != 1 {
		t.Fatalf("empty inline conflict: %d %v", rec.Code, m)
	}
	rec, m = do(h, "POST", "/v1/decide", strings.Replace(request, ref, home.StateRef("missing"), 1))
	if rec.Code != 400 || errClass(m) != "request_invalid" || f.calls != 1 {
		t.Fatalf("unknown: %d %v", rec.Code, m)
	}
	rec, m = do(h, "POST", "/v1/decide", body)
	if rec.Code != 200 || m["state_ref"] != home.StateRef("s") {
		t.Fatalf("inline: %d %v", rec.Code, m)
	}
}
