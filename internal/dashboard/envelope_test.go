package dashboard

import (
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/tuning"
)

const gib = uint64(1) << 30

// fullDevice puts the test runtime on an RTX 3060 whose 12 GiB are all in use.
func fullDevice(e *env) {
	e.rt.mu.Lock()
	e.rt.snap.Accelerator["memory_free"] = float64(0)
	e.rt.mu.Unlock()
}

// budgetSettings is a settings authority with the memory-budget capability.
type budgetSettings struct {
	fakeSettings
	budget uint64
	sets   []uint64
}

func (b *budgetSettings) MemoryBudget() (uint64, error) { return b.budget, nil }
func (b *budgetSettings) SetMemoryBudget(v uint64) error {
	b.budget = v
	b.sets = append(b.sets, v)
	return nil
}

// The Tuning envelope is the device the runtime reports, with the Auto budget
// as the default; the operator enters no hardware identifier.
func TestTuningEnvelopeComesFromTheActiveDeviceWithAutoMargin(t *testing.T) {
	e, _, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	target, budget := section(body, `id="tuning-target"`, `</dd>`), section(body, `id="tuning-budget"`, `</dd>`)
	if !has(target, "NVIDIA GeForce RTX 3060", "12.00 GiB") {
		t.Errorf("target device: %s", target)
	}
	want := tuning.Bytes(tuning.AutoBudget(12 * gib))
	if !has(budget, `data-budget="auto"`, "Auto · "+want) || want == "12.00 GiB" {
		t.Errorf("budget is not Auto with a margin (%s): %s", want, budget)
	}
	// No form field asks for a device, PCI id or byte count.
	if strings.Contains(workspace(body), `name="device"`) {
		t.Error("Tuning asks the operator for a device identifier")
	}
}

// A plan whose memory exceeds the budget is shown as blocked and opens the
// preservation editor; nothing is saved, built or changed for the operator.
func TestTuningOverBudgetPlanIsBlockedNotRewritten(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	if !strings.Contains(body, `data-fit="unknown"`) || strings.Contains(body, `id="tuning-over-budget"`) ||
		strings.Contains(section(body, `id="tuning-advanced"`, `<summary>`), " open") {
		t.Error("without a memory figure the fit is not unknown, or the editor is open")
	}
	// A measured figure inside the budget fits.
	ft.impact = TuningImpact{Memory: ImpactValue{State: Measured, Value: "VRAM reserved 8.00 GiB", Basis: "probe"}, MemoryUsage: tuning.Usage{Bytes: 8 * gib}}
	if body := e.get(t, "/tuning").Body.String(); !strings.Contains(body, `data-fit="within"`) || !strings.Contains(body, "Within budget") {
		t.Error("a measured figure inside the budget is not within")
	}
	// A measured full device exceeds the Auto budget.
	ft.impact = TuningImpact{Memory: ImpactValue{State: Measured, Value: "VRAM reserved 12.00 GiB", Basis: "probe"}, MemoryUsage: tuning.Usage{Bytes: 12 * gib}}
	body = e.get(t, "/tuning").Body.String()
	if !has(body, `data-fit="exceeds"`, `id="tuning-over-budget"`, "Exceeds budget", `id="tuning-advanced" open`) {
		t.Error("an over-budget plan is not blocked with the editor open")
	}
	// An estimate that is only a lower bound inside the budget is not a fit.
	ft.impact = TuningImpact{Memory: ImpactValue{State: Estimated, Value: "at least 4 GiB", Basis: "weights"}, MemoryUsage: tuning.Usage{Bytes: 4 * gib, LowerBound: true}}
	if body := e.get(t, "/tuning").Body.String(); !strings.Contains(body, `data-fit="unknown"`) || strings.Contains(body, "Within budget") {
		t.Error("a lower-bound estimate is claimed to fit")
	}
	// A memory figure whose value the page cannot state is never used.
	ft.impact = TuningImpact{Memory: ImpactValue{State: "VERIFIED", Value: "1 GiB", Basis: "x"}, MemoryUsage: tuning.Usage{Bytes: 20 * gib}}
	if body := e.get(t, "/tuning").Body.String(); !strings.Contains(body, `data-fit="unknown"`) {
		t.Error("an unstated memory figure decided the fit")
	}
	if len(ft.profiles) != 0 || len(ft.builds) != 0 {
		t.Error("rendering changed or built a profile")
	}
}

// The normal Tuning surface is intent: semantic regions are an Advanced
// disclosure, closed unless the operator has a reason to act.
func TestTuningNormalSurfaceIsIntentFirst(t *testing.T) {
	e, _, _ := tuningEnv(t)
	body := workspace(e.get(t, "/tuning").Body.String())
	adv := strings.Index(body, `data-disclosure="advanced" id="tuning-advanced"`)
	if adv < 0 || strings.Index(body, `id="tuning-region-table"`) < adv {
		t.Fatal("semantic regions are not inside the Advanced disclosure")
	}
	normal := body[:adv]
	for _, want := range []string{"Source", "Target device", "Memory budget", "Quality objective", "Current plan", "semantic regions kept at source precision"} {
		if !strings.Contains(normal, want) {
			t.Errorf("normal surface lacks %q", want)
		}
	}
	if n := strings.Count(body, `class="btn primary"`); n != 1 || !strings.Contains(body, `formaction="/tuning/build" data-pending`) {
		t.Errorf("Tuning has %d primary actions, want Build candidate only", n)
	}
	if strings.Count(body, `name="objective"`) != 1 {
		t.Error("the objective control is repeated")
	}
}

// The budget override is stored only through the settings authority; it is
// refused above the device total and changes no profile.
func TestTuningBudgetOverrideIsExplicit(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	bs := &budgetSettings{}
	cfg := e.d.cfg
	cfg.Settings = bs
	e.d = New(cfg)
	if rec := e.post(t, "/tuning/budget", url.Values{"source": {"clef-flash"}, "budget_mode": {"explicit"}, "budget_gib": {"6"}}); rec.Code != http.StatusSeeOther || bs.budget != 6*gib {
		t.Fatalf("override: %d %d", rec.Code, bs.budget)
	}
	body := e.get(t, "/tuning").Body.String()
	if !has(section(body, `id="tuning-budget"`, `</dd>`), `data-budget="explicit"`, "Override · 6.00 GiB") {
		t.Error("the override is not shown")
	}
	e.post(t, "/tuning/budget", url.Values{"budget_mode": {"explicit"}, "budget_gib": {"13"}})
	if a := e.lastAction(t); a.OK || bs.budget != 6*gib {
		t.Errorf("a budget above the device was stored: %+v %d", a, bs.budget)
	}
	e.post(t, "/tuning/budget", url.Values{"budget_mode": {"auto"}, "budget_gib": {"6"}})
	if bs.budget != 0 {
		t.Error("Auto did not clear the override")
	}
	if len(ft.profiles) != 0 {
		t.Error("changing the budget saved a profile")
	}
}

// A candidate measured above the budget is not a normal candidate: no Apply
// or activation is offered or accepted, and it routes back to Tuning.
func TestForgeBlocksOutOfEnvelopeCandidates(t *testing.T) {
	e, fm, fv := forgeEnv(t, variantInventory())
	fm.state.RestartRequired = false
	id := "clef-flash--r--aaaaaaaaaaaa"
	fm.state.Forge.Probes = []ProbeRow{{Variant: id, Device: "cuda", Result: "passed", VRAMBytes: 12 * gib}}
	body := e.get(t, "/forge").Body.String()
	c := card(t, body, id)
	if !has(c, `data-fit="exceeds"`, `data-envelope-blocked`, `data-return-tuning`, `href="/tuning?source=clef-flash"`, "Over memory budget") || has(c, `action="/forge/apply"`) {
		t.Errorf("over-budget candidate card:\n%s", c)
	}
	if has(section(body, `id="forge-advanced"`, `</details>`), `name="variant" value="`+id+`"><button type="submit" class="btn">Activate`) {
		t.Error("the low-level section activates an over-budget candidate")
	}
	for _, op := range []string{"/forge/apply", "/forge/activate"} {
		e.post(t, op, url.Values{"variant": {id}, "model": {"clef-flash"}, "device": {"cuda"}})
		if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "exceeds the memory budget") {
			t.Errorf("%s of an over-budget candidate: %+v", op, a)
		}
	}
	if len(fv.calls) != 0 {
		t.Errorf("an over-budget candidate reached the backend: %v", fv.calls)
	}
	// Measured inside the budget: Apply is offered again.
	fm.state.Forge.Probes[0].VRAMBytes = 6 * gib
	if c := card(t, e.get(t, "/forge").Body.String(), id); !has(c, `data-fit="within"`, `action="/forge/apply"`) {
		t.Errorf("an in-budget candidate is not applicable:\n%s", c)
	}
	// Without a measurement the fit is NOT_CHECKED, never invented.
	fm.state.Forge.Probes = nil
	if c := card(t, e.get(t, "/forge").Body.String(), id); !has(c, `data-fit="unknown"`, `data-state="NOT_CHECKED"`, `action="/forge/apply"`) {
		t.Errorf("an unmeasured candidate:\n%s", c)
	}
}

// Forge builds a saved tuning profile through the one backend operation;
// the operator is not handed lifecycle steps.
func TestForgeBuildsASavedTuningProfile(t *testing.T) {
	e, _, fv := forgeEnv(t, variantInventory())
	ft := &fakeTuning{analysis: tuningAnalysis(t)}
	withTuning(e, ft)
	p := expectedProfile(t, ft.analysis, "balanced", tuning.RegionFeedForward)
	if err := ft.SaveProfile(p, ft.analysis); err != nil {
		t.Fatal(err)
	}
	body := e.get(t, "/forge").Body.String()
	form := section(body, `id="forge-intent-form"`, `</form>`)
	if !has(form, `<option value="tuning:`+p.ID()+`" selected>Tuned · `, "Built-in · clef-flash-w4a16-rtn-g128", "Build candidate") {
		t.Errorf("Forge does not offer the saved profile first:\n%s", form)
	}
	for _, banned := range []string{"/forge/preflight", "/forge/probe", "/forge/certify", "/forge/optimize", `class="forge-flow"`} {
		if strings.Contains(body, banned) {
			t.Errorf("Forge hands the operator a lifecycle step: %s", banned)
		}
	}
	if n := strings.Count(form, `class="btn primary"`); n != 1 {
		t.Errorf("the build form has %d primary actions", n)
	}
	e.post(t, "/forge/build-evaluate", url.Values{"source": {"clef-flash"}, "profile": {"tuning:" + p.ID()}, "dataset": {"/data/eval.jsonl"}, "provisioning": {"auto"}})
	if len(fv.buildRequests) != 1 || fv.buildRequests[0].TuningProfile != p.ID() || fv.buildRequests[0].Profile != "" {
		t.Fatalf("build request %+v", fv.buildRequests)
	}
	// The dataset used is now a known resource, offered by name.
	if !strings.Contains(e.get(t, "/forge").Body.String(), `<option value="/data/eval.jsonl">eval.jsonl</option>`) {
		t.Error("a used dataset is not offered again by name")
	}
}

// Every evaluation resource is named by its meaning; no page shows the
// generic file chooser.
func TestResourceInputsAreSemanticallyNamed(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	withPathPicker(e, &fakePathPicker{})
	for page, wants := range map[string][]string{
		"/forge":       {"Dataset", "Evaluation questions", "Certification policy", "Add dataset…", "Add question file…", "Add question folder…", "Add policy…"},
		"/experiments": {"Dataset", "Evaluation questions", "Add dataset…", "Add question file…"},
	} {
		body := e.get(t, page).Body.String()
		for _, w := range wants {
			if !strings.Contains(body, w) {
				t.Errorf("%s lacks %q", page, w)
			}
		}
	}
	for _, page := range []string{"/forge", "/experiments", "/workbench", "/errors"} {
		if body := e.get(t, page).Body.String(); strings.Contains(body, "Choose file") || strings.Contains(body, "Choose folder") {
			t.Errorf("%s shows a generic file chooser", page)
		}
	}
}

// Runtime has one primary action for each state, and memory pressure is its
// own axis: a READY runtime on a full device is an attention state.
func TestRuntimeActionHierarchyAndMemoryPressure(t *testing.T) {
	e := newEnv(t)
	panel := func() string {
		body := e.get(t, "/").Body.String()
		return section(body, `<section class="readiness-panel`, `</section>`)
	}
	p := panel()
	if n := strings.Count(p, `class="btn primary"`); n != 1 || !has(p, `data-primary-action="workbench"`, `href="/workbench">Open Workbench`) {
		t.Errorf("ready runtime: %d primary actions:\n%s", n, p)
	}
	if strings.Contains(p, `action="/runtime/start"`) {
		t.Error("a running runtime offers Start")
	}
	if !has(p, `id="runtime-controls"`, `action="/runtime/restart"`, `action="/runtime/stop"`) || strings.Contains(p, `<details class="inline runtime-controls" id="runtime-controls" open`) {
		t.Error("Restart and Stop are not secondary controls")
	}
	if !has(p, `data-pressure="ok"`) {
		t.Error("normal memory is not ok")
	}

	fullDevice(e)
	body := e.get(t, "/").Body.String()
	p = section(body, `<section class="readiness-panel`, `</section>`)
	if !has(p, "READY", `data-pressure="bad"`, "Above safe budget", "12.00 GiB / 12.00 GiB") {
		t.Errorf("full device is not an attention state beside READY:\n%s", p)
	}
	if !has(body, "GPU memory is above the safe budget", `id="shell-memory"`) {
		t.Error("full device memory is not in the attention list and the shell")
	}

	e.rt.mu.Lock()
	e.rt.snap.State, e.rt.snap.Ready = "failed", false
	e.rt.mu.Unlock()
	if p := panel(); strings.Count(p, `class="btn primary"`) != 1 || !has(p, `data-primary-action="restart"`) || strings.Contains(p, "Open Workbench") {
		t.Errorf("failed runtime:\n%s", p)
	}
	e.rt.mu.Lock()
	e.rt.run = false
	e.rt.mu.Unlock()
	if p := panel(); strings.Count(p, `class="btn primary"`) != 1 || !has(p, `data-primary-action="start"`) || has(p, `action="/runtime/stop"`) {
		t.Errorf("stopped runtime:\n%s", p)
	}
}
