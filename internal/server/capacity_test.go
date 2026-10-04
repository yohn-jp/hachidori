package server

import (
	"encoding/json"
	"github.com/yohn-jp/hachidori/internal/home"
	"net/http"
	"os"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func TestCapacityDenialProjectsAuthoritativeLimit(t *testing.T) {
	for _, endpoint := range []string{"/v1/decide", "/v1/decide/batch"} {
		t.Run(endpoint, func(t *testing.T) {
			info := &api.CapacityInfo{Metric: "input_tokens", Limit: 100, Observed: 101}
			d := &fake{ready: true, err: &worker.RequestError{Class: api.ErrCapacity, Message: "input exceeds capacity", Capacity: info}}
			input := body
			if endpoint == "/v1/decide/batch" {
				input = `{"schema":"hachidori.v1","requests":[` + body + `]}`
			}
			rec, _ := do(Handler(d, Runtime{}), "POST", endpoint, input)
			if rec.Code != http.StatusTooManyRequests {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			var response api.ErrorBody
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if response.Error.Capacity == nil || *response.Error.Capacity != *info || response.Error.Class != api.ErrCapacity {
				t.Fatalf("error = %+v", response.Error)
			}
		})
	}
}

func TestResolveCapacityProfilePinsServingDTypeAndSource(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	if err := os.MkdirAll(h.Path("state"), 0755); err != nil {
		t.Fatal(err)
	}
	model := home.ModelManifest{ID: "clef-flash", Provider: "clef", Repo: "repo", Revision: "rev", Files: map[string]string{"weights": "digest"}}
	rm := home.RuntimeManifest{Identity: "runtime"}
	a := home.Active{Device: "cuda"}
	p := home.CapacityProfile{CapacityTarget: home.CapacityTarget{Runtime: rm.EnvironmentID(), ModelID: model.ID, Provider: model.Provider, Repo: model.Repo, Revision: model.Revision, SourceFilesSHA256: home.SourceOf(model).FilesSHA256, Device: "cuda", DType: "bfloat16"}, MaxStateTokens: 10, MaxInputTokens: 20, MaxBatchItems: 2, MaxBatchPaddedTokens: 40, RequiredGPUHeadroomBytes: 100}
	b, err := json.Marshal(home.CapacityProfiles{Schema: home.CapacityProfilesSchema, Profiles: []home.CapacityProfile{p}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(h.Path("state", home.CapacityProfilesFile), b, 0600); err != nil {
		t.Fatal(err)
	}
	if got, reason := resolveCapacityProfile(h, a, rm, model, model, nil, ""); got == nil || reason != "" {
		t.Fatalf("default dtype profile: %v %s", got, reason)
	}
	if got, reason := resolveCapacityProfile(h, a, rm, model, model, nil, "float32"); got != nil || reason == "" {
		t.Fatalf("different dtype accepted: %v %s", got, reason)
	}
	changed := model
	changed.Revision = "other"
	if got, reason := resolveCapacityProfile(h, a, rm, model, changed, nil, ""); got != nil || reason == "" {
		t.Fatalf("different revision accepted: %v %s", got, reason)
	}
	a.Device = "cpu"
	if got, reason := resolveCapacityProfile(h, a, rm, model, model, nil, ""); got != nil || reason != "" {
		t.Fatalf("CPU requires GPU profile: %v %s", got, reason)
	}
}
