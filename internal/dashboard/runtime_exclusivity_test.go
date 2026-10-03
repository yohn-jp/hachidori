package dashboard

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/tuning"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// stopServing makes the test runtime a bound but stopped one: no worker, no
// accelerator stats, nothing resident.
func stopServing(e *env) {
	e.rt.mu.Lock()
	e.rt.run = false
	e.rt.snap = worker.Snapshot{State: worker.StateStopped}
	e.rt.mu.Unlock()
}

func panel(e *env, t *testing.T) string {
	t.Helper()
	body := e.get(t, "/").Body.String()
	return section(body, `<section class="readiness-panel`, `</section>`)
}

// Stopped is a legitimate stable state: it is not an attention item, the
// runtime page names it and offers the explicit Start, and it says that
// planning stays available.
func TestStoppedRuntimeIsStableAndOffersStart(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	stopServing(e)
	if a := alerts(e.d.view("Runtime", "runtime")); len(a) != 0 {
		t.Fatalf("a stopped runtime is flagged: %+v", a)
	}
	p := panel(e, t)
	if !has(p, `data-primary-action="start"`, `action="/runtime/start"`, "STOPPED", "Planning, Tuning and Forge stay available") || strings.Contains(p, `action="/runtime/stop"`) {
		t.Errorf("stopped runtime panel:\n%s", p)
	}
	for _, page := range []string{"/", "/forge", "/models", "/tuning"} {
		body := e.get(t, page).Body.String()
		if strings.Contains(body, "needs attention") || strings.Contains(body, `class="readiness tone-bad"`) {
			t.Errorf("%s presents a stopped runtime as a problem", page)
		}
	}
}

// A running runtime shows Stop where the operator can see it, not behind a
// disclosure.
func TestRunningRuntimeShowsStopDirectly(t *testing.T) {
	e := newEnv(t)
	p := panel(e, t)
	controls := section(p, `id="runtime-controls"`, `</div>`)
	if !has(controls, `action="/runtime/stop"`, `class="btn danger">Stop</button>`) || strings.Contains(p, "<details") && strings.Contains(section(p, "<details", "</details>"), "/runtime/stop") {
		t.Errorf("Stop is not directly available:\n%s", p)
	}
}

// While model engineering owns the accelerator the runtime reads as an
// intentional pause: not STOPPED (the operator did not stop it), not a
// failure, and it offers neither Start nor Stop, since the controller brings
// serving back by itself.
func TestPauseForForgeIsNeitherStopNorFailure(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	stopServing(e)
	plan := []string{"resolve_inputs", "preflight", "build", "resolving", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
	fm.state.Busy = &ModelOp{Kind: "forge_build_evaluate", Model: "clef-flash", Plan: plan, Phases: plan[:6], Phase: "probe", Started: time.Now()}
	fm.state.Pause = &ModelPause{Owner: "forge_build_evaluate"}

	p := panel(e, t)
	if !has(p, "PAUSED", "Paused — Forge is using the GPU", `readiness-panel tone-active`) {
		t.Errorf("the pause is not projected:\n%s", p)
	}
	for _, not := range []string{"STOPPED", `action="/runtime/start"`, `action="/runtime/stop"`, "tone-bad", "FAILED"} {
		if strings.Contains(p, not) {
			t.Errorf("a pause shows %q:\n%s", not, p)
		}
	}
	body := e.get(t, "/forge").Body.String()
	shell := section(body, `id="shell-status"`, `</div>`)
	if !has(shell, "PAUSED", "tone-active") || strings.Contains(shell, "needs attention") {
		t.Errorf("shell:\n%s", shell)
	}
	if got := shellOf(e.d.view("Runtime", "runtime")); !got.Paused || got.Word != "PAUSED" {
		t.Errorf("shell status %+v", got)
	}

	// An operator Stop and a failure stay what they are.
	fm.state.Pause, fm.state.Busy = nil, nil
	if p := panel(e, t); !has(p, "STOPPED", `action="/runtime/start"`) || strings.Contains(p, "PAUSED") {
		t.Errorf("operator stop:\n%s", p)
	}
	e.rt.mu.Lock()
	e.rt.snap = worker.Snapshot{State: worker.StateFailed, Phase: "loading", LastFailure: &worker.FailureView{Class: worker.ClassModelLoad, Message: "boom"}}
	e.rt.mu.Unlock()
	if a := alerts(e.d.view("Runtime", "runtime")); len(a) == 0 || a[0].Level != "bad" {
		t.Errorf("a failure is not an alert: %+v", a)
	}
}

// planningEnv hosts the Models, Forge and Tuning authorities over a runtime
// that is serving.
func planningEnv(t *testing.T) (*env, *fakeModels) {
	t.Helper()
	e, fm, _ := forgeEnv(t, variantInventory())
	withTuning(e, &fakeTuning{analysis: tuningAnalysis(t)})
	return e, fm
}

var everyPage = []string{"/", "/models", "/forge", "/tuning", "/workbench", "/experiments", "/errors", "/settings", "/diagnostics"}

// The operation is one compact headline in the shell of every workspace; its
// phase strip, phase number, progress and execution facts render only on Forge.
func TestForgeOperationDetailBelongsToForgeAndTheHeadlineToEveryPage(t *testing.T) {
	e, fm := planningEnv(t)
	withSettings(e, &fakeSettings{}, nil)
	plan := []string{"resolve_inputs", "preflight", "build", "resolving", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
	fm.state.Busy = &ModelOp{Kind: "forge_build_evaluate", Device: "cuda", Model: "clef-flash", Target: "source clef-flash recipe clef-flash-w4a16-rtn-g128",
		Plan: plan, Phases: plan[:7], Phase: "reference_run", Step: "probing", Detail: "case 12", Item: 12, Items: 40, DeviceMode: "auto", ResolvedDevice: "cuda",
		Started: time.Now().Add(-time.Minute)}
	for _, page := range everyPage {
		rec := e.get(t, page)
		if rec.Code != 200 {
			t.Fatalf("%s: %d", page, rec.Code)
		}
		body := rec.Body.String()
		head := section(body, `id="shell-operation"`, `</span>`)
		if !has(head, "RUNNING", "Build &amp; evaluate", "Reference run", `href="/forge"`) {
			t.Errorf("%s lacks the compact headline: %q", page, head)
		}
		for _, detail := range []string{`id="models-busy"`, `class="stages"`, "phase 7 of 11", `class="bar progress`, "case 12", "data-device-mode"} {
			if page == "/forge" {
				if !strings.Contains(body, detail) {
					t.Errorf("/forge lacks the operation detail %q", detail)
				}
			} else if strings.Contains(body, detail) {
				t.Errorf("%s repeats the Forge operation detail %q", page, detail)
			}
		}
	}
	// The same operation, rendered for the live refresh of any page.
	live := e.get(t, "/live").Body.String()
	if strings.Contains(live, `class="stages"`) || !strings.Contains(live, `id="shell-operation"`) {
		t.Error("the live fragment carries the operation detail or lacks the headline")
	}
}

// A finished Forge operation keeps its outcome elsewhere as one line that
// points at Forge; its phases and diagnostics are Forge's.
func TestFinishedForgeOperationIsCompactOutsideForge(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	plan := []string{"resolve_inputs", "preflight", "build"}
	fm.state.Last = &ModelOp{Kind: "forge_build_evaluate", Model: "clef-flash", Plan: plan, Phases: plan, Phase: "build", Failure: "boom", FailurePhase: "build",
		Diagnostic: "forge_build_evaluate-20261002T000000Z-0123abcd", Started: time.Now().Add(-time.Minute), Finished: time.Now()}
	for _, page := range []string{"/", "/models"} {
		body := e.get(t, page).Body.String()
		last := section(body, `id="models-last"`, "</div>")
		if !has(last, "FAILED", `href="/forge">Details`) || has(last, `class="stages"`) || has(last, "forge-diagnostic-last") {
			t.Errorf("%s: %s", page, last)
		}
	}
	if body := e.get(t, "/forge").Body.String(); !has(body, `class="stages"`, "Failed in phase") {
		t.Error("Forge lacks the finished operation's detail")
	}
}

// ---- planning with serving stopped ----

type deviceModels struct {
	*fakeModels
	obs DeviceObservation
}

func (d deviceModels) Device() DeviceObservation { return d.obs }

// With no worker the device capacity comes from the host observation, so the
// Tuning target, the Auto budget and Forge's envelope exist while serving is
// stopped; and the observation is capacity, never a candidate's measurement.
func TestPlanningAndBudgetWorkWithServingStopped(t *testing.T) {
	e, fm := planningEnv(t)
	stopServing(e)
	withModels(e, deviceModels{fm, DeviceObservation{Name: "NVIDIA GeForce RTX 3060", TotalBytes: 12 * gib}})

	body := e.get(t, "/tuning").Body.String()
	target, budget := section(body, `id="tuning-target"`, `</dd>`), section(body, `id="tuning-budget"`, `</dd>`)
	want := tuning.Bytes(tuning.AutoBudget(12 * gib))
	if !has(target, "NVIDIA GeForce RTX 3060", "12.00 GiB") || !has(budget, `data-budget="auto"`, "Auto · "+want) {
		t.Errorf("stopped Tuning envelope:\n%s\n%s", target, budget)
	}
	v := e.d.view("Forge", "forge")
	env := e.d.envelopeOf(v)
	if env.TotalBytes != 12*gib || env.BudgetBytes == 0 || env.Device != "NVIDIA GeForce RTX 3060" {
		t.Fatalf("envelope %+v", env)
	}
	// No candidate memory was measured by observing the device.
	row := variantInventory().Variants[1]
	if fit := variantFit(v, env, VariantRow{VariantEntry: row}); fit.State != NotChecked || fit.Value != "" {
		t.Errorf("a candidate's fit while stopped is %+v, want NOT_CHECKED without a value", fit)
	}
	if page := e.get(t, "/forge").Body.String(); strings.Contains(section(page, "data-fit-memory", "</dd>"), "MEASURED") {
		t.Error("Forge shows measured memory for a candidate nothing measured")
	}

	// A worker that reports the accelerator stays authoritative.
	e2, fm2 := planningEnv(t)
	withModels(e2, deviceModels{fm2, DeviceObservation{Name: "other", TotalBytes: 4 * gib}})
	if got := e2.d.envelopeOf(e2.d.view("Forge", "forge")); got.TotalBytes != 12*gib {
		t.Errorf("the worker's own report was replaced: %+v", got)
	}
	// Nothing known stays unknown.
	e3, fm3 := planningEnv(t)
	stopServing(e3)
	withModels(e3, deviceModels{fm3, DeviceObservation{Pending: true}})
	if got := e3.d.envelopeOf(e3.d.view("Forge", "forge")); got.TotalBytes != 0 || got.BudgetBytes != 0 {
		t.Errorf("a pending observation produced a figure: %+v", got)
	}
}

// ---- shared layout contract ----

// Short explanatory text and controls take the width of their section; only
// .prose carries a reading measure. The contract is stated on the shared rules,
// not on any one screen.
func TestSharedLayoutDoesNotNarrowShortTextOrControls(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	css := strings.Split(body, "</style>")[0]
	for _, sel := range []string{".lede", ".note", ".empty"} {
		if r := cssRule(t, css, sel); strings.Contains(r, "max-width") || strings.Contains(r, "width:") {
			t.Errorf("%s is narrower than its section: %s", sel, r)
		}
	}
	if r := cssRule(t, css, ".prose"); !strings.Contains(r, "max-width: var(--measure)") {
		t.Errorf(".prose lost the reading measure: %s", r)
	}
	if strings.Contains(css, ".intent dd select") || strings.Contains(css, ".intent dd input") {
		t.Error("controls inside .intent are capped narrower than their column")
	}
	// Label/value grids give the value the rest of the row; they are not a
	// fixed ratio that wastes a wide window.
	for _, sel := range []string{".operator-row", ".instrument-heading"} {
		if r := cssRule(t, css, sel); !strings.Contains(r, "grid-template-columns: minmax(8rem, 13rem) minmax(0, 1fr)") {
			t.Errorf("%s does not give its value the available width: %s", sel, r)
		}
	}
}

// No template breaks a line by hand or sets an ad-hoc width on a text
// container: layout comes from the shared rules.
func TestTemplatesHaveNoHardBreaksOrAdHocTextWidths(t *testing.T) {
	files, err := filepath.Glob("*.html")
	if err != nil || len(files) == 0 {
		t.Fatalf("templates: %v %v", files, err)
	}
	adHoc := regexp.MustCompile(`style="[^"]*(?:width|max-width|min-width)\s*:\s*[0-9.]+(?:px|rem|em|ch)`)
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		src := string(b)
		if strings.Contains(src, "<br") {
			t.Errorf("%s breaks a line with <br>", f)
		}
		for _, m := range adHoc.FindAllString(src, -1) {
			// The one deliberate control size: a short numeric field.
			if !strings.Contains(m, `max-width:7rem`) || f != "tuning.html" {
				t.Errorf("%s sets an ad-hoc width: %s", f, m)
			}
		}
	}
}

// The state vocabulary and the operation headline do not split at arbitrary
// boundaries.
func TestSemanticLabelsDoNotWrapMidLabel(t *testing.T) {
	css := strings.Split(newEnv(t).get(t, "/").Body.String(), "</style>")[0]
	for _, sel := range []string{".readiness", ".semantic-state", ".badge", ".op-headline"} {
		if r := cssRule(t, css, sel); !strings.Contains(r, "white-space: nowrap") {
			t.Errorf("%s can split mid-label: %s", sel, r)
		}
	}
}
