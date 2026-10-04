package dashboard

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/requesthistory"
)

func TestRequestHistoryWorkspaceProjectsBoundedAPIHistory(t *testing.T) {
	e := newEnv(t)
	store := requesthistory.New()
	e.d.cfg.RequestHistory = store

	started := time.Now().UTC()
	entry := store.Begin("/v1/decide/batch", started, requesthistory.Identity{
		Runtime: "runtime-1", Model: "model-1", Device: "cuda:0", Variant: "variant-1",
	})
	entry.SetInput(map[string]any{"requests": []string{"first", "second"}}, 24, 2, 2)
	entry.Admit()
	entry.Queued(started.Add(time.Millisecond))
	entry.Started(started.Add(3 * time.Millisecond))
	entry.Complete(started.Add(9*time.Millisecond), map[string]any{"responses": []string{"yes", "no"}}, http.StatusOK)

	page := e.get(t, "/requests")
	if page.Code != http.StatusOK {
		t.Fatalf("GET /requests: %d %s", page.Code, page.Body)
	}
	for _, want := range []string{"Request history", `id="request-queue"`, `id="request-flight"`, `id="request-items"`,
		"Started at", "Completed at", `fetch("/api/requests")`} {
		if !strings.Contains(page.Body.String(), want) {
			t.Errorf("request screen lacks %q", want)
		}
	}
	templateFile, err := pageFS.ReadFile("requests.html")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(templateFile), "innerHTML") || !strings.Contains(string(templateFile), "textContent") {
		t.Error("request screen renders request content as HTML")
	}

	list := e.get(t, "/api/requests")
	if list.Code != http.StatusOK {
		t.Fatalf("GET /api/requests: %d %s", list.Code, list.Body)
	}
	var view requesthistory.View
	if err := json.Unmarshal(list.Body.Bytes(), &view); err != nil {
		t.Fatal(err)
	}
	if view.QueueDepth != 0 || view.InFlight != 0 || view.ActiveItems != 0 || len(view.Entries) != 1 {
		t.Fatalf("request history view %+v", view)
	}
	if got := view.Entries[0]; got.Endpoint != "/v1/decide/batch" || got.State != "completed" || got.ItemCount != 2 || got.Runtime.Model != "model-1" {
		t.Fatalf("request summary %+v", got)
	}

	detail := e.get(t, "/api/requests/"+view.Entries[0].ID)
	if detail.Code != http.StatusOK || !strings.Contains(detail.Body.String(), `"variant":"variant-1"`) ||
		!strings.Contains(detail.Body.String(), `"received_at":`) || !strings.Contains(detail.Body.String(), `"started_at":`) ||
		!strings.Contains(detail.Body.String(), `"completed_at":`) ||
		!strings.Contains(detail.Body.String(), `"requests":["first","second"]`) || !strings.Contains(detail.Body.String(), `"responses":["yes","no"]`) {
		t.Fatalf("request detail: %d %s", detail.Code, detail.Body)
	}
	missing := e.get(t, "/api/requests/missing")
	if missing.Code != http.StatusNotFound {
		t.Errorf("unknown request detail: %d", missing.Code)
	}
}
