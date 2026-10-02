package dashboard

import (
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

type fakeResidency struct {
	mu    sync.Mutex
	ids   []string
	saves [][]string
	err   error
}

func (f *fakeResidency) Residents() ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.ids), f.err
}

func (f *fakeResidency) SetResidents(ids []string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.saves = append(f.saves, slices.Clone(ids))
	if f.err == nil {
		f.ids = slices.Clone(ids)
	}
	return f.err
}

// pairInventory is the catalog pair on cuda with laya-base active, plus one
// model that is not materialized.
func pairInventory() setup.Inventory {
	inv := modelsInventory()
	inv.Models = []setup.ModelEntry{
		{ID: "laya-base", Provider: "laya", Repo: "convaiinnovations/laya", Revision: "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851", Files: 5, Materialized: true, Active: true},
		{ID: "opendecider-nano", Provider: "opendecider", Repo: "manjunathshiva/opendecider-nano", Revision: "7e42a1508d2beef44717d044831e87f2fc4db9f2", Files: 7, Materialized: true},
		{ID: "laya-absent", Provider: "laya", Repo: "example/absent", Revision: "fedcba0987654321fedcba0987654321fedcba09", Files: 1},
	}
	return inv
}

func withResidency(e *env, fm *fakeModels, fr *fakeResidency) {
	cfg := e.d.cfg
	cfg.Models, cfg.Residency = fm, fr
	e.d = New(cfg)
}

func selectionRow(t *testing.T, body, model string) string {
	t.Helper()
	m := regexp.MustCompile(`(?s)<tr data-model="` + regexp.QuoteMeta(model) + `">.*?</tr>`).FindString(body)
	if m == "" {
		t.Fatalf("no resident-selection row for %s", model)
	}
	return m
}

// Settings > Models & runtimes offers an explicit resident selection over the
// catalog models only: the active model is the always-on default resident,
// a materialized model can be selected, and one that is not materialized cannot
// be newly selected. Viewing acts on nothing.
func TestResidentSelectionControl(t *testing.T) {
	e := newEnv(t)
	fm := &fakeModels{state: ModelsState{Inventory: pairInventory()}}
	withResidency(e, fm, &fakeResidency{})
	body := e.get(t, "/models").Body.String()
	if !strings.Contains(body, `id="resident-selection"`) || !strings.Contains(body, `action="/models/residents"`) {
		t.Fatal("no resident-selection form in Models & runtimes")
	}
	def := selectionRow(t, body, "laya-base")
	for _, want := range []string{`value="laya-base"`, ` checked`, ` disabled`, "default resident"} {
		if !strings.Contains(def, want) {
			t.Errorf("active model row lacks %q: %s", want, def)
		}
	}
	nano := selectionRow(t, body, "opendecider-nano")
	if strings.Contains(nano, "checked") || strings.Contains(nano, "disabled") {
		t.Errorf("a materialized model cannot be selected: %s", nano)
	}
	absent := selectionRow(t, body, "laya-absent")
	if !strings.Contains(absent, " disabled") || strings.Contains(absent, "checked") || !strings.Contains(absent, "not materialized") {
		t.Errorf("an unmaterialized model can be newly selected: %s", absent)
	}
	e.rt.mu.Lock()
	lc := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(fm.calls) != 0 || len(lc) != 0 {
		t.Fatalf("viewing acted: maintenance=%v lifecycle=%v", fm.calls, lc)
	}
}

// Saving the selection only stores it: no maintenance or lifecycle action, a
// restart-later message, and the same token/origin protection as every POST.
// Without the authority there is no route.
func TestResidentSelectionSaveOnlyStores(t *testing.T) {
	e := newEnv(t)
	if rec := e.post(t, "/models/residents", url.Values{"resident": {"opendecider-nano"}}); rec.Code != http.StatusNotFound && rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("route without the authority: %d", rec.Code)
	}
	fm, fr := &fakeModels{state: ModelsState{Inventory: pairInventory()}}, &fakeResidency{}
	withResidency(e, fm, fr)

	if rec := e.post(t, "/models/residents", url.Values{"token": {"forged"}, "resident": {"opendecider-nano"}}); rec.Code != http.StatusForbidden || len(fr.saves) != 0 {
		t.Fatalf("forged token: %d saves %v", rec.Code, fr.saves)
	}
	rec := e.post(t, "/models/residents", url.Values{"return": {"models"}, "resident": {" opendecider-nano ", "laya-base"}})
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/models" {
		t.Fatalf("save: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if len(fr.saves) != 1 || !slices.Equal(fr.saves[0], []string{"opendecider-nano", "laya-base"}) {
		t.Fatalf("saved %v", fr.saves)
	}
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, "next start") {
		t.Fatalf("action %+v", a)
	}
	e.rt.mu.Lock()
	lc := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(fm.calls) != 0 || len(lc) != 0 {
		t.Fatalf("saving acted: maintenance=%v lifecycle=%v", fm.calls, lc)
	}

	// A refusal by the authority is shown and nothing else happens.
	fr.err = errBad
	e.post(t, "/models/residents", url.Values{"resident": {"someone/else"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "refused") {
		t.Fatalf("refusal not reported: %+v", a)
	}
}

var errBad = &refusal{}

type refusal struct{}

func (*refusal) Error() string { return "refused: not a catalog model" }

// A selection that is not what the runtime was started with is shown as next
// start / restart intent per model, and the explicit restart is offered; the
// running set is read from the status document, never from the selection.
func TestResidentSelectionIsNextStartIntentNotRunningState(t *testing.T) {
	e := newEnv(t) // one worker: laya-base on cuda
	inv := pairInventory()
	fm := &fakeModels{state: ModelsState{Inventory: inv, RestartRequired: true, ResidencyChanged: true}}
	withResidency(e, fm, &fakeResidency{ids: []string{"opendecider-nano", "laya-absent"}})
	body := e.get(t, "/models").Body.String()

	if !strings.Contains(body, `id="restart-required"`) || !strings.Contains(body, "The resident selection changed") {
		t.Error("a pending selection does not require a restart")
	}
	nano := selectionRow(t, body, "opendecider-nano")
	if !strings.Contains(nano, "selected · applies on next start") || !strings.Contains(nano, " checked") || strings.Contains(nano, `badge tone-ok">resident<`) {
		t.Errorf("selected but not running: %s", nano)
	}
	absent := selectionRow(t, body, "laya-absent")
	if !strings.Contains(absent, " checked") || strings.Contains(absent, " disabled") || !strings.Contains(absent, "will fail to start and is never replaced by another model") {
		t.Errorf("selected but not materialized is not surfaced: %s", absent)
	}
	if def := selectionRow(t, body, "laya-base"); !strings.Contains(def, "default resident") || strings.Contains(def, "starts with the runtime") {
		t.Errorf("the running default: %s", def)
	}

	// Once the restart bound both, nothing is pending and the rows say so; a
	// resident that is bound but no longer selected is removed on restart.
	ready := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, PID: 7}
	st := server.Status{Schema: "hachidori.v1", Runtime: server.Runtime{ModelID: "laya-base", Device: "cuda"}, Worker: ready,
		Residents: []server.ResidentStatus{
			residentStatus("laya-base", "laya", true, true, ready),
			residentStatus("opendecider-nano", "opendecider", false, true, ready),
		}}
	cfg := e.d.cfg
	cfg.Status = func() server.Status { return st }
	e.d = New(cfg)
	fm.state.RestartRequired, fm.state.ResidencyChanged = false, false
	body = e.get(t, "/models").Body.String()
	if strings.Contains(body, `id="restart-required"`) || !strings.Contains(selectionRow(t, body, "opendecider-nano"), `badge tone-ok">resident<`) {
		t.Error("a bound selection is still pending")
	}
	fr := cfg.Residency.(*fakeResidency)
	fr.mu.Lock()
	fr.ids = nil
	fr.mu.Unlock()
	body = e.get(t, "/models").Body.String()
	if row := selectionRow(t, body, "opendecider-nano"); !strings.Contains(row, "resident · removed on restart") || strings.Contains(row, " checked") {
		t.Errorf("a deselected running resident: %s", row)
	}
}

// The active model changing under a running worker keeps its own message; the
// resident control never presents the active pair as running state.
func TestResidentControlKeepsActivationRestartMessage(t *testing.T) {
	e := newEnv(t)
	inv := pairInventory()
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "opendecider-nano", Device: "cuda"}
	inv.Models[0].Active, inv.Models[1].Active = false, true
	withResidency(e, &fakeModels{state: ModelsState{Inventory: inv, RestartRequired: true}}, &fakeResidency{})
	body := e.get(t, "/models").Body.String()
	if !strings.Contains(body, "The active runtime/model changed.") || strings.Contains(body, "The resident selection changed") {
		t.Error("the activation restart message was replaced")
	}
	if row := selectionRow(t, body, "opendecider-nano"); !strings.Contains(row, "default resident") || !strings.Contains(row, "starts with the runtime") || !strings.Contains(row, " disabled") {
		t.Errorf("the newly active model: %s", row)
	}
}
