package dashboard

import (
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// The Runtime page restates the routing counters of the status document:
// the policy identity, handoff counts and each resident's contribution. It
// shows nothing when no policy is bound.
func TestRuntimeShowsRoutingCountersFromTheStatusDocument(t *testing.T) {
	e := newEnv(t)
	if strings.Contains(e.get(t, "/").Body.String(), `id="routing"`) {
		t.Fatal("a runtime without a routing policy shows a routing section")
	}

	ready := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, Errors: map[string]int64{}, QueueLimit: 64}
	st := server.Status{Schema: "hachidori.v1", Runtime: server.Runtime{ModelID: "laya-base", Device: "cuda"}, Worker: ready,
		Residents: []server.ResidentStatus{residentStatus("laya-base", "laya", true, true, ready), residentStatus("opendecider-nano", "opendecider", false, true, ready)},
		Routing: &route.Status{Policy: api.PolicyRef{ID: "wave3", SHA256: "sha256:abc"}, Requests: 7, Failures: 1, Questions: 20, Handoffs: 3, HandoffFailures: 2,
			Reasons: map[string]int64{api.ReasonHandoffLowConf: 3, api.ReasonFirstPathKept: 17},
			Providers: []route.ProviderStatus{
				{Model: "laya-base", Provider: "laya", FirstPathQuestions: 20, FinalResults: 17, Calls: 7, InferenceMSTotal: 140},
				{Model: "opendecider-nano", Provider: "opendecider", HandoffQuestions: 3, FinalResults: 3, Calls: 3, InferenceMSTotal: 30}},
			Calibration: &route.Calibration{EvidenceSHA256: "sha256:evidence"}}}
	e.d.cfg.Status = func() server.Status { return st }

	body := e.get(t, "/").Body.String()
	section := regexp.MustCompile(`(?s)<section[^>]*id="routing".*?</section>`).FindString(body)
	if section == "" {
		t.Fatal("no routing section")
	}
	for _, want := range []string{"wave3", "sha256:abc", "sha256:evidence", api.ReasonHandoffLowConf, api.ReasonFirstPathKept, "opendecider-nano", "laya-base"} {
		if !strings.Contains(section, want) {
			t.Errorf("routing section lacks %q:\n%s", want, section)
		}
	}
	row := regexp.MustCompile(`(?s)<tr data-model="opendecider-nano">.*?</tr>`).FindString(section)
	for _, want := range []string{">3<", "30"} {
		if !strings.Contains(row, want) {
			t.Errorf("opendecider-nano row lacks %q:\n%s", want, row)
		}
	}
}
