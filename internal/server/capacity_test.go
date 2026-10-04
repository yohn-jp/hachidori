package server

import (
	"encoding/json"
	"net/http"
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
