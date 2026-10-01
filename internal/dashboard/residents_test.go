package dashboard

import (
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func residentStatus(model, provider string, def, running bool, w worker.Snapshot) server.ResidentStatus {
	rt := server.Runtime{ModelID: model, Device: "cuda"}
	return server.ResidentStatus{Model: model, Provider: provider, Default: def, Running: running,
		Status: server.Status{Runtime: rt, Worker: w}}
}

func residentRow(t *testing.T, body, model string) string {
	t.Helper()
	m := regexp.MustCompile(`(?s)<tr id="resident-` + regexp.QuoteMeta(model) + `".*?</tr>`).FindString(body)
	if m == "" {
		t.Fatalf("no row for resident %s in the Runtime page", model)
	}
	return m
}

// The Runtime page shows every resident with its own state, restating the
// status document's residents; a failed resident does not change another's
// row, and a single worker shows no resident table.
func TestRuntimeShowsEachResidentIndependently(t *testing.T) {
	e := newEnv(t)
	if strings.Contains(e.get(t, "/").Body.String(), `id="residents"`) {
		t.Fatal("a single worker shows a resident table")
	}

	ready := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, PID: 4242, Starts: 1, Requests: 9,
		Errors: map[string]int64{}, QueueLimit: 64, LatencyP50MS: 25.3, LatencyP95MS: 30.5,
		Info: worker.Info{"device": "cuda:0", "dtype": "torch.float16", "load_ms": 812.5, "warmup_ms": 90.1}}
	failed := worker.Snapshot{State: worker.StateFailed, Phase: "loading", PID: 0, Starts: 1, Errors: map[string]int64{"not_ready": 3},
		QueueLimit: 64, LastFailure: &worker.FailureView{Class: worker.ClassModelLoad, Message: "no weights"}}
	st := server.Status{Schema: "hachidori.v1", Runtime: server.Runtime{ModelID: "laya-base", Device: "cuda"}, Worker: ready,
		Residents: []server.ResidentStatus{
			residentStatus("laya-base", "laya", true, true, ready),
			residentStatus("opendecider-nano", "opendecider", false, true, failed),
		}}
	e.d.cfg.Status = func() server.Status { return st }

	body := e.get(t, "/").Body.String()
	a, b := residentRow(t, body, "laya-base"), residentRow(t, body, "opendecider-nano")
	for _, want := range []string{`data-state="ready"`, "READY", "4242", "cuda:0 torch.float16", "812.5 / 90.1 ms", "default route", "laya"} {
		if !strings.Contains(a, want) {
			t.Errorf("laya-base row lacks %q:\n%s", want, a)
		}
	}
	for _, want := range []string{`data-state="failed"`, "phase loading", worker.ClassModelLoad, "opendecider"} {
		if !strings.Contains(b, want) {
			t.Errorf("opendecider-nano row lacks %q:\n%s", want, b)
		}
	}
	if strings.Contains(a, "failed") || strings.Contains(b, "READY") || strings.Contains(b, "default route") {
		t.Fatalf("a resident's state leaked into another's row:\n%s\n%s", a, b)
	}
	if !strings.Contains(body, "Resident opendecider-nano failed") {
		t.Fatal("the failed resident is not named in the attention list")
	}

	// An operator-stopped resident is shown as stopped, not failed.
	st.Residents[1] = residentStatus("opendecider-nano", "opendecider", false, false, worker.Snapshot{State: worker.StateStopped, Errors: map[string]int64{}})
	b = residentRow(t, e.get(t, "/").Body.String(), "opendecider-nano")
	if !strings.Contains(b, `data-state="stopped"`) || strings.Contains(b, "failed") {
		t.Fatalf("stopped resident row:\n%s", b)
	}
}
