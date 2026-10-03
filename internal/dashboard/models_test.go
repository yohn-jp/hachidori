package dashboard

import (
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/setup"
)

// The existing dashboard fake implements the new controller operation as a
// no-op so the other workspace tests keep exercising their original paths.
func (f *fakeModels) StartDesiredState(DesiredStateRequest) error { return nil }

type desiredStateFake struct {
	*fakeModels
	intents []DesiredStateRequest
	err     error
}

func (f *desiredStateFake) StartDesiredState(r DesiredStateRequest) error {
	f.intents = append(f.intents, r)
	f.calls = append(f.calls, "desired-state")
	return f.err
}

func desiredStateInventory() setup.Inventory {
	inv := modelsInventory()
	inv.Variants = []setup.VariantEntry{{ID: "variant-123", SourceID: "laya-base", Certification: "accepted", SourceMaterialized: true}}
	return inv
}

func TestModelsDesiredStateUsesOneSubmitForTargetDeviceAndResidents(t *testing.T) {
	e := newEnv(t)
	fm := &desiredStateFake{fakeModels: &fakeModels{state: ModelsState{Inventory: desiredStateInventory()}}}
	withResidency(e, fm, &fakeResidency{ids: []string{"laya-other"}})
	withSettings(e, &fakeSettings{loc: "en"}, nil)

	rec := e.get(t, "/models")
	if rec.Code != 200 {
		t.Fatalf("GET /models: %d %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{
		`action="/models/desired-state"`, `id="desired-state-form"`, `value="source:laya-base"`,
		`value="variant:variant-123"`, `name="device_mode"`, `value="pinned"`, `name="device_override"`,
		`form="desired-state-form" name="resident"`, `name="allow_provision"`, `Auto uses the active activation`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("desired-state form lacks %q", want)
		}
	}

	form := url.Values{
		"target": {"source:laya-base"}, "device_mode": {"auto"}, "device_override": {"cuda"},
		"resident": {"laya-other"}, "allow_provision": {"1"},
	}
	if rec := e.post(t, "/models/desired-state", form); rec.Code != 303 || rec.Header().Get("Location") != "/models" {
		t.Fatalf("source request: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if len(fm.intents) != 1 {
		t.Fatalf("source submit called backend %d times", len(fm.intents))
	}
	wantSource := DesiredStateRequest{Model: "laya-base", DeviceMode: "auto", Residents: []string{"laya-other"}, AllowProvision: true}
	if got := fm.intents[0]; got.Model != wantSource.Model || got.Variant != wantSource.Variant || got.DeviceMode != wantSource.DeviceMode || got.Device != wantSource.Device ||
		strings.Join(got.Residents, ",") != strings.Join(wantSource.Residents, ",") || got.AllowProvision != wantSource.AllowProvision {
		t.Fatalf("source intent %+v, want %+v", got, wantSource)
	}

	form = url.Values{
		"target": {"variant:variant-123"}, "device_mode": {"pinned"}, "device_override": {"cpu"},
		"resident": {"laya-other"},
	}
	if rec := e.post(t, "/models/desired-state", form); rec.Code != 303 {
		t.Fatalf("variant request: %d", rec.Code)
	}
	if len(fm.intents) != 2 {
		t.Fatalf("variant submit called backend %d times", len(fm.intents)-1)
	}
	got := fm.intents[1]
	if got.Model != "laya-base" || got.Variant != "variant-123" || got.DeviceMode != "pinned" || got.Device != "cpu" {
		t.Fatalf("variant intent %+v", got)
	}
}

func TestModelsDesiredStateRejectsUnknownTargetBeforeBackendCall(t *testing.T) {
	e := newEnv(t)
	fm := &desiredStateFake{fakeModels: &fakeModels{state: ModelsState{Inventory: desiredStateInventory()}}}
	withModels(e, fm)

	if rec := e.post(t, "/models/desired-state", url.Values{"target": {"source:outside"}, "device_mode": {"auto"}}); rec.Code != 303 {
		t.Fatalf("unknown target: %d", rec.Code)
	}
	if len(fm.intents) != 0 {
		t.Fatalf("unknown target reached backend: %+v", fm.intents)
	}
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "no longer available") {
		t.Fatalf("unknown target outcome %+v", a)
	}
}

func TestModelsProjectsDesiredDeviceResolutionAndRollback(t *testing.T) {
	e := newEnv(t)
	fm := &desiredStateFake{fakeModels: &fakeModels{state: ModelsState{
		Inventory: desiredStateInventory(),
		Busy: &ModelOp{Kind: "desired_state", Model: "laya-base", Target: "source laya-base", DeviceMode: "auto", ResolvedDevice: "cuda",
			Device: "cuda", ActualDevice: "cpu", Plan: []string{"validating", "rebinding", "proving"}, Phases: []string{"validating", "rebinding"}, Phase: "rebinding"},
		Last: &ModelOp{Kind: "desired_state", Model: "laya-base", Target: "source laya-base", DeviceMode: "pinned", RequestedDevice: "cuda",
			ResolvedDevice: "cuda", ActualDevice: "cuda", Plan: []string{"validating", "rebinding", "rolling_back"},
			Phases: []string{"validating", "rebinding", "rolling_back"}, Phase: "rolling_back", FailurePhase: "rebinding",
			Failure: "the previous serving target and resident set were restored", Started: time.Now().Add(-time.Second), Finished: time.Now()},
	}}}
	withModels(e, fm)
	withSettings(e, &fakeSettings{loc: "en"}, nil)
	body := e.get(t, "/models").Body.String()
	for _, want := range []string{
		`data-device-mode="auto"`, "Auto · resolved to cuda · worker reported cpu",
		`data-device-mode="pinned"`, "Pinned override · cuda · worker reported cuda",
		`class="failed">Rebinding runtime`, `class="done">Rolling back`,
		"the previous serving target and resident set were restored",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("desired-state operation projection lacks %q", want)
		}
	}
}
