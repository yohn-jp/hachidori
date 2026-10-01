package dashboard

import (
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func statuses(sts []stage) string {
	var out []string
	for _, s := range sts {
		out = append(out, s.Name+":"+s.Status)
	}
	return strings.Join(out, " ")
}

// An operation's stages follow what it really entered: done before the
// current phase, pending after it, failed at the failing phase, and skipped
// (not pending forever) for a phase a finished action never needed.
func TestOpStages(t *testing.T) {
	plan := []string{"preparing", "runtime", "model", "publish"}
	now := time.Now()
	for name, tc := range map[string]struct {
		op   ModelOp
		want string
	}{
		"running": {ModelOp{Plan: plan, Phases: []string{"preparing", "runtime"}, Phase: "runtime"},
			"preparing:done runtime:current model:pending publish:pending"},
		"failed": {ModelOp{Plan: plan, Phases: []string{"preparing", "runtime"}, Phase: "runtime", Finished: now, Failure: "boom", FailurePhase: "runtime"},
			"preparing:done runtime:failed model:pending publish:pending"},
		"done": {ModelOp{Plan: plan, Phases: plan, Phase: "publish", Finished: now},
			"preparing:done runtime:done model:done publish:done"},
		"done without needing a phase": {ModelOp{Plan: []string{"runtime", "model", "activation"}, Phases: []string{"runtime", "model"}, Phase: "model", Finished: now},
			"runtime:done model:done activation:skipped"},
		"no plan": {ModelOp{Phases: []string{"model"}, Phase: "model"}, "model:current"},
	} {
		if got := statuses(opStages(tc.op)); got != tc.want {
			t.Errorf("%s: %s, want %s", name, got, tc.want)
		}
	}
	if got := phasePosition(ModelOp{Plan: plan, Phase: "model"}); got != "3 of 4" {
		t.Errorf("position %q", got)
	}
	if phasePosition(ModelOp{Plan: plan}) != "" {
		t.Error("a position without a phase")
	}
}

// The worker's startup is the supervisor's own phase sequence; a failed worker
// shows the phase it failed in, a ready one all of it done, and a stopped
// runtime nothing.
func TestWorkerStages(t *testing.T) {
	for name, tc := range map[string]struct {
		w    worker.Snapshot
		want string
	}{
		"spawning":   {worker.Snapshot{State: worker.StateStarting, Phase: "spawning"}, "spawning:current importing:pending loading:pending warming:pending ready:pending"},
		"loading":    {worker.Snapshot{State: worker.StateStarting, Phase: "loading"}, "spawning:done importing:done loading:current warming:pending ready:pending"},
		"warming":    {worker.Snapshot{State: worker.StateStarting, Phase: "warming"}, "spawning:done importing:done loading:done warming:current ready:pending"},
		"ready":      {worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true}, "spawning:done importing:done loading:done warming:done ready:done"},
		"failed":     {worker.Snapshot{State: worker.StateFailed, Phase: "loading"}, "spawning:done importing:done loading:failed warming:pending ready:pending"},
		"restarting": {worker.Snapshot{State: worker.StateRestarting, Phase: "ready"}, "spawning:current importing:pending loading:pending warming:pending ready:pending"},
		"stopped":    {worker.Snapshot{State: worker.StateStopped, Phase: "ready"}, ""},
		"preflight":  {worker.Snapshot{State: worker.StateFailed, Phase: worker.PhasePreflight}, ""},
	} {
		if got := statuses(workerStages(tc.w)); got != tc.want {
			t.Errorf("%s: %s, want %s", name, got, tc.want)
		}
	}
}

func TestBytesAndElapsed(t *testing.T) {
	for n, want := range map[int64]string{0: "0 B", 1023: "1023 B", 1024: "1.0 KiB", 412 << 20: "412.0 MiB", 3<<30 + 512<<20: "3.5 GiB"} {
		if got := bytesIn(n); got != want {
			t.Errorf("bytesIn(%d) = %q, want %q", n, got, want)
		}
	}
	now := time.Now()
	if since(time.Time{}) != "" || took(ModelOp{Started: now}) != "" {
		t.Error("an unknown time was rendered")
	}
	if got := took(ModelOp{Started: now, Finished: now.Add(95 * time.Second)}); got != "1m 35s" {
		t.Errorf("took = %q", got)
	}
}

// Only a step with a total is a percentage.
func TestModelOpDeterminate(t *testing.T) {
	if o := (ModelOp{Done: 50, Total: 200}); !o.Determinate() || o.Percent() != 25 {
		t.Errorf("determinate step: %v %v", o.Determinate(), o.Percent())
	}
	if o := (ModelOp{Done: 50}); o.Determinate() || o.Percent() != 0 {
		t.Errorf("a step without a total: %v %v", o.Determinate(), o.Percent())
	}
	if o := (ModelOp{Done: 500, Total: 200}); o.Percent() != 100 {
		t.Errorf("overshoot not clamped: %v", o.Percent())
	}
}

// Every dynamic label and hint the page looks up has a Japanese entry, the
// same guarantee the literal template messages have.
func TestOperationMessagesAreCatalogued(t *testing.T) {
	var msgs []string
	for _, m := range []map[string]string{phaseLabels, stepLabels, workerPhaseLabels, failureHints} {
		for _, v := range m {
			msgs = append(msgs, v)
		}
	}
	msgs = append(msgs, "Read the worker output below and in Diagnostics.")
	for _, m := range msgs {
		if !i18n.Japanese.Has(m) {
			t.Errorf("%q has no Japanese entry", m)
		}
	}
	for _, class := range []string{worker.ClassStartup, worker.ClassProviderInit, worker.ClassDevice, worker.ClassModelLoad, worker.ClassWarmup,
		worker.ClassPreflight, worker.ClassCrash, worker.ClassUnresponsive, worker.ClassStartTimeout, worker.ClassProtocolError} {
		if failureHints[class] == "" {
			t.Errorf("class %s has no hint", class)
		}
	}
}

func TestFailureOf(t *testing.T) {
	lines := []string{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
	w := worker.Snapshot{State: worker.StateFailed, Phase: "loading",
		LastFailure: &worker.FailureView{Class: worker.ClassModelLoad, Message: "no weights", Stderr: lines}}
	f := failureOf(w)
	if f == nil || f.Phase != "loading" || f.PhaseWord != "Loading model" || f.Hint != failureHints[worker.ClassModelLoad] ||
		len(f.Stderr) != failureTail || f.Stderr[len(f.Stderr)-1] != "10" {
		t.Fatalf("failure %+v", f)
	}
	w.LastFailure.Class = "something_new"
	if f := failureOf(w); f.Hint != "Read the worker output below and in Diagnostics." {
		t.Errorf("unknown class hint %q", f.Hint)
	}
	// A worker that recovered, one being started again, and one that never
	// failed have nothing to act on.
	w.Ready, w.State = true, worker.StateReady
	if failureOf(w) != nil || failureOf(worker.Snapshot{State: worker.StateStarting}) != nil {
		t.Error("a failure was reported for a healthy worker")
	}
	w.Ready, w.State = false, worker.StateStarting
	if failureOf(w) != nil {
		t.Error("the previous failure is shown while the worker is starting again")
	}
}

func TestNextOf(t *testing.T) {
	rt := server.Runtime{ModelID: "laya-base", Device: "cuda"}
	inv := modelsInventory()
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "opendecider-nano", Device: "cuda"}
	inv.Models[0].Active = false
	inv.Models = append(inv.Models, setup.ModelEntry{ID: "opendecider-nano", Active: true})
	n := nextOf(ModelsState{Inventory: inv}, rt)
	if n.Model != "opendecider-nano" || n.Device != "cuda" || !n.Differs {
		t.Fatalf("next %+v", n)
	}
	if n := nextOf(ModelsState{Inventory: inv}, server.Runtime{ModelID: "opendecider-nano", Device: "cuda"}); n.Differs {
		t.Error("the same pair differs")
	}
	if n := nextOf(ModelsState{Inventory: inv}, server.Runtime{ModelID: "opendecider-nano", Device: "cpu"}); !n.Differs {
		t.Error("a different device is not different")
	}
	if n := nextOf(ModelsState{}, rt); n.Model != "" || n.Differs {
		t.Errorf("no activation: %+v", n)
	}
}

// The Runtime page tells the operator which phase the worker failed in and
// what to do, with the worker's own output; the active pair that this build
// cannot start is named; and an activation waiting for a restart is shown
// beside the runtime that is configured, which is not claimed to be serving.
func TestRuntimePageExplainsAFailedStartAndTheNextStart(t *testing.T) {
	e := newEnv(t)
	e.rt.mu.Lock()
	e.rt.snap = worker.Snapshot{State: worker.StateFailed, Phase: "spawning", Starts: 1,
		LastFailure: &worker.FailureView{Class: worker.ClassStartup, Message: "worker exited: exit status 2",
			Stderr: []string{"usage: hachidori_worker.py [-h]", "hachidori_worker.py: error: unrecognized arguments: --provider laya"}}}
	e.rt.run = false
	e.rt.mu.Unlock()
	inv := modelsInventory()
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "opendecider-nano", Device: "cuda"}
	inv.Models[0].Active = false
	inv.Models = append(inv.Models, setup.ModelEntry{ID: "opendecider-nano", Provider: "opendecider", Materialized: true, Active: true})
	inv.ActiveProblem = "runtime cu128-old was materialized by an older Hachidori"
	withModels(e, &fakeModels{state: ModelsState{Inventory: inv}})

	body := e.get(t, "/").Body.String()
	for _, want := range []string{
		`id="worker-failure"`, "Worker failed while: Starting worker process", "worker_startup: worker exited: exit status 2",
		"unrecognized arguments: --provider laya", "Materialized models are unaffected.", "materialize and activate the current one in Settings", `href="/diagnostics#worker-failure"`,
		`id="worker-stages"`, `<li class="failed">Starting worker process</li>`,
		`<span class="sub">configured for this runtime, not serving</span>`,
		`id="next-start"`, "opendecider-nano · cuda", "cannot start", `id="active-problem"`, "materialized by an older Hachidori",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("Runtime page lacks %q", want)
		}
	}
	if strings.Contains(body, "serving now") {
		t.Error("a failed runtime is said to be serving")
	}
	if strings.Contains(body, `class="bar progress indeterminate"`) {
		t.Error("a failed worker is shown as still working")
	}

	// While it loads, the lifecycle moves and the bar is indeterminate: the
	// supervisor reports phases, never a total.
	e.rt.mu.Lock()
	e.rt.snap = worker.Snapshot{State: worker.StateStarting, Phase: "loading"}
	e.rt.run = true
	e.rt.mu.Unlock()
	body = e.get(t, "/live").Body.String()
	if !strings.Contains(body, `<li class="current" aria-current="step">Loading model</li>`) || !strings.Contains(body, `<li class="done">Importing provider</li>`) ||
		!strings.Contains(body, `class="bar progress indeterminate" role="progressbar" aria-label="Loading model"`) || strings.Contains(body, `aria-valuenow`) {
		t.Errorf("loading phase not shown as an indeterminate, current step:\n%s", body)
	}
}

// A determinate step shows bytes and a percentage; one without a total shows
// its step name and an indeterminate bar, and never a percentage.
func TestOperationProgressRendering(t *testing.T) {
	e := newEnv(t)
	busy := &ModelOp{Kind: "materialize", Device: "cuda", Model: "opendecider-nano", Plan: []string{"preparing", "runtime", "model", "publish"},
		Phases: []string{"preparing", "runtime", "model"}, Phase: "model", Step: "downloading", Detail: "model.safetensors", Item: 2, Items: 7,
		Done: 200 << 20, Total: 800 << 20, Started: time.Now().Add(-3 * time.Minute)}
	fm := &fakeModels{state: ModelsState{Inventory: modelsInventory(), Busy: busy}}
	withModels(e, fm)
	body := e.get(t, "/settings").Body.String()
	for _, want := range []string{"RUNNING", "running for 3m", `<li class="done">Preparing</li>`, `<li class="done">Runtime</li>`, `<li class="current" aria-current="step">Model</li>`,
		`<li class="pending">Publishing</li>`, "Downloading", "(2 of 7)", "200.0 MiB of 800.0 MiB (25%)",
		`role="progressbar" aria-label="Downloading" aria-valuemin="0" aria-valuemax="100" aria-valuenow="25"><span style="width:25.0%">`} {
		if !strings.Contains(body, want) {
			t.Errorf("determinate step lacks %q", want)
		}
	}

	busy.Phase, busy.Phases, busy.Step, busy.Detail, busy.Done, busy.Total, busy.Item, busy.Items =
		"runtime", []string{"preparing", "runtime"}, "materializing", "installing the locked packages (cu128)", 0, 0, 0, 0
	body = e.get(t, "/settings").Body.String()
	if !strings.Contains(body, "Materializing") || !strings.Contains(body, "installing the locked packages (cu128)") ||
		!strings.Contains(body, `class="bar progress indeterminate" role="progressbar" aria-label="Materializing"`) {
		t.Error("indeterminate step not shown as such")
	}
	if i := strings.Index(body, `id="models-busy"`); strings.Contains(body[i:i+2500], "aria-valuenow") || strings.Contains(body[i:i+2500], "%)") {
		t.Error("a step without a total shows a percentage")
	}

	// Between the phase's start and its first report there is no step yet.
	busy.Step, busy.Detail = "", ""
	body = e.get(t, "/settings").Body.String()
	if !strings.Contains(body, "phase 2 of 4 · Working") || !strings.Contains(body, `class="bar progress indeterminate" role="progressbar" aria-label="Working"`) {
		t.Error("a phase without a step yet is not shown as working")
	}

	// The same operation is visible on the Runtime page.
	if b := e.get(t, "/").Body.String(); !strings.Contains(b, `id="models-busy"`) || !strings.Contains(b, "RUNNING") {
		t.Error("the Runtime page does not show the operation in flight")
	}
}

// A finished operation keeps its outcome on screen; a success is DONE with how
// long it took, and a failure names its phase, step, cause and where to look.
func TestFinishedOperationOutcomeRendering(t *testing.T) {
	e := newEnv(t)
	now := time.Now()
	fm := &fakeModels{state: ModelsState{Inventory: modelsInventory(),
		Last: &ModelOp{Kind: "activate", Device: "cuda", Model: "opendecider-nano", Plan: []string{"runtime", "model", "activation"},
			Phases: []string{"runtime", "model", "activation"}, Phase: "activation", Started: now.Add(-12 * time.Second), Finished: now}}}
	withModels(e, fm)
	body := e.get(t, "/settings").Body.String()
	for _, want := range []string{`id="models-last"`, "DONE", "took 12s", `<li class="done">Activation</li>`} {
		if !strings.Contains(body, want) {
			t.Errorf("success lacks %q", want)
		}
	}
	if strings.Contains(body, "Failed in phase") {
		t.Error("a success reports a failure")
	}
	fm.state.Last = &ModelOp{Kind: "materialize", Device: "cuda", Model: "opendecider-nano", Plan: []string{"preparing", "runtime", "model", "publish"},
		Phases: []string{"preparing", "runtime"}, Phase: "runtime", Failure: "uv sync: exit status 1", FailurePhase: "runtime", FailureStep: "materializing",
		Log: "C:\\Hachidori\\logs\\setup.log", Started: now.Add(-time.Minute), Finished: now}
	body = e.get(t, "/settings").Body.String()
	for _, want := range []string{"FAILED", `<li class="failed">Runtime</li>`, "Failed in phase <strong>Runtime</strong> · Materializing", "uv sync: exit status 1",
		"The active runtime and model were not changed.", `href="/diagnostics"`, `C:\Hachidori\logs\setup.log`} {
		if !strings.Contains(body, want) {
			t.Errorf("failure lacks %q", want)
		}
	}
}

// The Settings manager names an active pair that this build cannot start.
func TestSettingsShowsAnActivePairThatCannotStart(t *testing.T) {
	e := newEnv(t)
	inv := modelsInventory()
	inv.ActiveProblem = "runtime cu128-old was materialized by an older Hachidori (worker aaaa, this build bbbb)"
	withModels(e, &fakeModels{state: ModelsState{Inventory: inv}})
	body := e.get(t, "/settings").Body.String()
	if !strings.Contains(body, `id="active-problem"`) || !strings.Contains(body, "CANNOT START") || !strings.Contains(body, "materialized by an older Hachidori") {
		t.Error("Settings does not name the active pair that cannot start")
	}
}
