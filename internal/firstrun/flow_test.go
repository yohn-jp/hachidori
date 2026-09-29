package firstrun

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// fakeRT stands in for the worker binding: it reports whatever worker state
// the test sets, exactly as the supervisor's snapshot would.
type fakeRT struct {
	mu      sync.Mutex
	running bool
	state   string
	phase   string
	starts  int
}

func (r *fakeRT) Start() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.running {
		return false
	}
	r.running, r.state, r.phase = true, worker.StateStarting, "spawning"
	r.starts++
	return true
}
func (r *fakeRT) Stop()    { r.mu.Lock(); r.running, r.state = false, worker.StateStopped; r.mu.Unlock() }
func (r *fakeRT) Restart() { r.Stop(); r.Start() }
func (r *fakeRT) Running() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running
}
func (r *fakeRT) set(state, phase string) {
	r.mu.Lock()
	r.state, r.phase = state, phase
	r.mu.Unlock()
}
func (r *fakeRT) Status() server.Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	return server.Status{Runtime: server.Runtime{Home: "h", Runtime: "cpu-abc", ModelID: setup.DefaultModel, Model: "repo/rev", Device: "cpu"},
		Worker: worker.Snapshot{State: r.state, Phase: r.phase, Ready: r.state == worker.StateReady}}
}

type fakePicker struct {
	mu   sync.Mutex
	path string
	err  error
}

func (p *fakePicker) PickFolder(context.Context, string) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.path, p.err
}
func (p *fakePicker) set(path string) { p.mu.Lock(); p.path, p.err = path, nil; p.mu.Unlock() }

type fixture struct {
	t            *testing.T
	base         string // where the user's storage folders live
	loc          home.Locator
	picker       *fakePicker
	rt           *fakeRT
	ctl          *app.Controller
	flow         *Flow
	env          Env
	plan         Plan
	failRemember atomic.Bool

	mu        sync.Mutex
	installed map[string]bool
	devices   []string
	models    []string
	setupErr  error
	gate      chan struct{} // when non-nil, setup blocks until it is closed
	setups    atomic.Int32
	opens     atomic.Int32
}

func newFixture(t *testing.T, plan Plan) *fixture {
	t.Helper()
	f := &fixture{t: t, base: t.TempDir(), rt: &fakeRT{state: worker.StateStopped}, picker: &fakePicker{},
		installed: map[string]bool{}, plan: plan}
	f.loc = home.Locator{Path: filepath.Join(t.TempDir(), "bootstrap", "bootstrap.json")}
	f.env = Env{
		Load: func(root string) (home.Active, error) {
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.installed[root] {
				return home.Active{Runtime: "cpu-abc", ModelID: setup.DefaultModel, Model: "repo/rev", Device: "cpu"}, nil
			}
			return home.Active{}, errors.New("no active runtime")
		},
		FreeSpace: func(string) (uint64, bool) { return 123 << 30, true },
	}
	selected := ""
	if plan.Mode == ModeLaunch || plan.Mode == ModeResume {
		selected = plan.Home
	}
	f.ctl = app.New(app.Config{
		Home: selected,
		Open: func(string) (app.Runtime, error) { f.opens.Add(1); return f.rt, nil },
		Setup: func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error {
			f.setups.Add(1)
			f.mu.Lock()
			f.devices = append(f.devices, device)
			f.models = append(f.models, model)
			gate, serr := f.gate, f.setupErr
			f.mu.Unlock()
			onPhase(setup.PhasePreparing)
			io.WriteString(log, "setup log line\n")
			if gate != nil {
				<-gate
			}
			onPhase(setup.PhaseRuntime)
			if serr != nil {
				return serr
			}
			onPhase(setup.PhaseModel)
			onPhase(setup.PhaseActivation)
			if err := os.MkdirAll(filepath.Join(root, "runtime"), 0o755); err != nil {
				return err
			}
			f.mu.Lock()
			f.installed[root] = true
			f.mu.Unlock()
			return nil
		},
		Installed: func(root string) bool { return f.env.IsInstalled(root) },
	})
	f.flow = New(Config{Ctl: f.ctl, Plan: plan, Picker: f.picker, Env: f.env,
		Remember: func(root string) (home.Home, error) {
			if f.failRemember.Load() {
				return home.Home{}, errors.New("access denied")
			}
			return f.loc.Save(root)
		}})
	return f
}

func (f *fixture) dir(name string) string {
	f.t.Helper()
	p := filepath.Join(f.base, name)
	if err := os.MkdirAll(p, 0o755); err != nil {
		f.t.Fatal(err)
	}
	return p
}

func (f *fixture) selectDir(p string) Validation {
	f.t.Helper()
	f.picker.set(p)
	v, err := f.flow.Select(context.Background())
	if err != nil {
		f.t.Fatalf("Select: %v", err)
	}
	return v
}

func (f *fixture) waitStage(want string) View {
	f.t.Helper()
	var v View
	for i := 0; i < 400; i++ {
		if v = f.flow.View(); v.Stage == want {
			return v
		}
		time.Sleep(5 * time.Millisecond)
	}
	f.t.Fatalf("stage %q never reached; last view %+v", want, v)
	return v
}

func (f *fixture) remembered() (string, bool) {
	f.t.Helper()
	b, found, err := f.loc.Load()
	if err != nil {
		f.t.Fatalf("locator: %v", err)
	}
	return b.Home, found
}

func (f *fixture) waitRemembered(want string) {
	f.t.Helper()
	for i := 0; i < 400; i++ {
		if got, found := f.remembered(); found && got == want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	got, found := f.remembered()
	f.t.Fatalf("remembered %q %v, want %q", got, found, want)
}

// Proofs 1-3: startup decisions come from real home discovery results.
func TestDecideStartup(t *testing.T) {
	loc := home.Locator{Path: filepath.Join(t.TempDir(), "bootstrap.json")}
	installed := map[string]bool{}
	env := Env{Load: func(root string) (home.Active, error) {
		if installed[root] {
			return home.Active{Runtime: "r"}, nil
		}
		return home.Active{}, errors.New("no active runtime")
	}}
	lookup := func() Plan {
		h, found, err := loc.Lookup()
		d := home.Discovery{Source: home.SourceUnconfigured}
		if found && err == nil {
			d = home.Discovery{Home: h, Source: home.SourceLocator}
		}
		return Decide(d, err, env)
	}

	// 1. No bootstrap: first run.
	if p := lookup(); p.Mode != ModeFirstRun {
		t.Fatalf("no bootstrap: %+v", p)
	}
	// 2. A configured home: resume until it holds a valid runtime, then launch.
	root := filepath.Join(t.TempDir(), "H")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := loc.Save(root); err != nil {
		t.Fatal(err)
	}
	if p := lookup(); p.Mode != ModeResume || p.Home != root {
		t.Fatalf("configured, not installed: %+v", p)
	}
	installed[root] = true
	if p := lookup(); p.Mode != ModeLaunch || p.Home != root {
		t.Fatalf("configured and installed: %+v", p)
	}
	// 3. The stored home vanished: recovery naming it, never a fresh install.
	if err := os.RemoveAll(root); err != nil {
		t.Fatal(err)
	}
	if p := lookup(); p.Mode != ModeMissing || p.Home != root {
		t.Fatalf("missing stored home: %+v", p)
	}
	// A malformed locator is a diagnostic recovery, not a first run.
	if err := os.WriteFile(loc.Path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if p := lookup(); p.Mode != ModeInvalid || p.Detail == "" {
		t.Fatalf("malformed locator: %+v", p)
	}
}

// Proofs 4 and 6: picker -> validation -> setup -> commit to the locator ->
// start -> READY with the canonical identity; the device is passed unchanged.
func TestSelectValidateInstallReady(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	dir := f.dir("Models")
	v := f.selectDir(dir)
	if !v.OK() || v.Home != dir || v.Nested || v.Existing != nil || v.FreeBytes == nil || *v.FreeBytes != 123<<30 {
		t.Fatalf("validation %+v", v)
	}
	if _, found := f.remembered(); found {
		t.Fatal("selecting a folder alone must not write the bootstrap locator")
	}
	f.gate = make(chan struct{})
	if err := f.flow.Install("cpu"); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageInstalling)
	var inst View
	for i := 0; i < 400 && (len(inst.Phases) == 0 || inst.Phases[0].Status != "current"); i++ {
		time.Sleep(5 * time.Millisecond)
		inst = f.flow.View()
	}
	if got := inst.Phases; len(got) != 4 || got[0].Status != "current" || got[1].Status != "pending" {
		t.Fatalf("phases must show only real progress: %+v", got)
	}
	if _, found := f.remembered(); found {
		t.Fatal("bootstrap written before setup succeeded")
	}
	close(f.gate)
	f.waitStage(StageStarting)
	f.waitRemembered(dir)
	if f.rt.starts != 1 || f.opens.Load() != 1 {
		t.Fatalf("starts %d opens %d", f.rt.starts, f.opens.Load())
	}
	f.rt.set(worker.StateStarting, "warming")
	if v := f.flow.View(); v.Stage != StageStarting || !strings.Contains(v.Worker, "warming") {
		t.Fatalf("warming view %+v", v)
	}
	if f.flow.Done() {
		t.Fatal("desktop home shown before READY")
	}
	f.rt.set(worker.StateReady, "ready")
	v2 := f.waitStage(StageReady)
	if v2.Identity == nil || v2.Identity.ModelID != setup.DefaultModel || v2.Identity.Device != "cpu" || v2.Identity.Runtime != "cpu-abc" {
		t.Fatalf("identity %+v", v2.Identity)
	}
	if !f.flow.Done() {
		t.Fatal("READY must hand over to the desktop home")
	}
	if len(f.devices) != 1 || f.devices[0] != "cpu" || f.models[0] != setup.DefaultModel {
		t.Fatalf("setup got devices %v models %v", f.devices, f.models)
	}
}

// Proof 5: locations that cannot be written or are files block setup.
func TestUnusableLocationBlocksSetup(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	file := filepath.Join(f.base, "afile")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := map[string]string{
		"file":         file,
		"below a file": filepath.Join(file, "sub"),
		"relative":     "relative/path",
		"empty":        "",
	}
	for name, p := range cases {
		v := f.selectDir(p)
		if v.OK() || v.Problem == "" {
			t.Errorf("%s: %+v accepted", name, v)
		}
		if err := f.flow.Install("cpu"); err == nil {
			t.Errorf("%s: install started", name)
		}
	}
	if f.setups.Load() != 0 {
		t.Fatalf("setup ran %d times for unusable locations", f.setups.Load())
	}
	if _, found := f.remembered(); found {
		t.Fatal("unusable location was remembered")
	}
	// A folder that does not exist yet but can be created is fine.
	if v := f.selectDir(filepath.Join(f.base, "nope", "deeper")); !v.OK() {
		t.Fatalf("creatable folder rejected: %+v", v)
	}
}

func TestWriteProbeFailureBlocksSetup(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	dir := f.dir("blocked")
	f.flow.cfg.Env.ProbeWritable = func(path string) error {
		if path != dir {
			t.Fatalf("probe path %q, want %q", path, dir)
		}
		return os.ErrPermission
	}
	v := f.selectDir(dir)
	if v.OK() || !strings.Contains(v.Problem, "cannot write") {
		t.Fatalf("unwritable folder: %+v", v)
	}
	if err := f.flow.Install("cpu"); err == nil || f.setups.Load() != 0 {
		t.Fatalf("install err %v setups %d", err, f.setups.Load())
	}
}

// A non-empty foreign folder is never mixed with Hachidori's directories:
// a Hachidori subfolder is used, and everything is created inside it.
func TestForeignFolderGetsHachidoriSubfolder(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	dir := f.dir("Documents")
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("mine"), 0o644); err != nil {
		t.Fatal(err)
	}
	v := f.selectDir(dir)
	want := filepath.Join(dir, HomeSubdir)
	if !v.OK() || v.Home != want || !v.Nested {
		t.Fatalf("validation %+v", v)
	}
	if err := f.flow.Install("cpu"); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageStarting)
	f.waitRemembered(want)
	// Proof 14: nothing but the foreign file and the home escapes into the
	// chosen folder or its parent.
	assertOnly(t, dir, "notes.txt", HomeSubdir)
	assertOnly(t, f.base, "Documents")
}

func assertOnly(t *testing.T, dir string, names ...string) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, e := range ents {
		got[e.Name()] = true
	}
	for _, n := range names {
		delete(got, n)
	}
	if len(got) != 0 {
		t.Fatalf("%s contains unexpected entries %v", dir, got)
	}
}

// Proofs 7 and 13: a failed setup stays failed and not-installed, keeps its
// diagnostics, never remembers the home, and never switches device.
func TestSetupFailureKeepsDiagnosticsAndDevice(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	f.setupErr = errors.New("cuda: driver too old")
	dir := f.dir("Models")
	f.selectDir(dir)
	if err := f.flow.Install("cuda"); err != nil {
		t.Fatal(err)
	}
	v := f.waitStage(StageFailed)
	if v.Failure == nil || v.Failure.Source != app.SourceSetup || !strings.Contains(v.Failure.Message, "driver too old") {
		t.Fatalf("failure %+v", v.Failure)
	}
	if !v.CanRetry || !v.CanChange {
		t.Fatalf("recovery actions %+v", v)
	}
	if last := v.Phases[1]; last.Name != "runtime" || last.Status != "failed" {
		t.Fatalf("phases %+v", v.Phases)
	}
	if f.env.IsInstalled(dir) {
		t.Fatal("failed setup reads as installed")
	}
	if _, found := f.remembered(); found {
		t.Fatal("failed setup was remembered: a relaunch would claim a configured desktop")
	}
	if f.rt.starts != 0 || f.opens.Load() != 0 {
		t.Fatal("runtime started after failed setup")
	}
	b, err := os.ReadFile(filepath.Join(dir, "logs", "setup.log"))
	if err != nil || !strings.Contains(string(b), "setup log line") || v.SetupLog == "" {
		t.Fatalf("setup log %q %v (%q)", b, err, v.SetupLog)
	}
	if got := f.devices; len(got) != 1 || got[0] != "cuda" {
		t.Fatalf("devices %v: CUDA must never be replaced by CPU", got)
	}
}

// Proof 8: one setup at a time; Retry after a failure runs exactly one more.
func TestRetryDoesNotDuplicateSetup(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	f.setupErr = errors.New("network down")
	f.gate = make(chan struct{})
	f.selectDir(f.dir("Models"))
	if err := f.flow.Install("cpu"); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageInstalling)
	for _, err := range []error{f.flow.Install("cpu"), f.flow.Retry(), f.flow.Change()} {
		if err == nil {
			t.Fatal("action accepted while setup is running")
		}
	}
	if _, err := f.flow.Select(context.Background()); err == nil {
		t.Fatal("selection changed while setup is running")
	}
	close(f.gate)
	f.waitStage(StageFailed)
	if f.setups.Load() != 1 {
		t.Fatalf("setups %d", f.setups.Load())
	}
	f.mu.Lock()
	f.setupErr, f.gate = nil, nil
	f.mu.Unlock()
	if err := f.flow.Retry(); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageStarting)
	if f.setups.Load() != 2 || f.rt.starts != 1 {
		t.Fatalf("setups %d starts %d", f.setups.Load(), f.rt.starts)
	}
	if err := f.flow.Retry(); !errors.Is(err, ErrNotRetry) {
		t.Fatalf("retry with nothing failed: %v", err)
	}
}

// Proof 9: changing the location before a successful activation is safe, and
// after activation it is refused without touching the bootstrap record.
func TestChangeLocationBeforeActivation(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	f.setupErr = errors.New("disk full")
	a, b := f.dir("A"), f.dir("B")
	f.selectDir(a)
	if err := f.flow.Install("cpu"); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageFailed)
	if err := f.flow.Change(); err != nil {
		t.Fatal(err)
	}
	if v := f.flow.View(); v.Stage != StageSelect || v.Selection != nil {
		t.Fatalf("after Change: %+v", v)
	}
	if _, found := f.remembered(); found {
		t.Fatal("changing location wrote the bootstrap locator")
	}
	f.mu.Lock()
	f.setupErr = nil
	f.mu.Unlock()
	f.selectDir(b)
	if err := f.flow.Install("cpu"); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageStarting)
	f.waitRemembered(b)
	if f.env.IsInstalled(a) {
		t.Fatal("abandoned location must not read as installed")
	}
	// After activation the location cannot be changed from the wizard.
	f.picker.set(a)
	if _, err := f.flow.Select(context.Background()); !errors.Is(err, ErrActive) {
		t.Fatalf("Select after activation: %v", err)
	}
	if err := f.flow.Change(); !errors.Is(err, ErrActive) {
		t.Fatalf("Change after activation: %v", err)
	}
	if got, _ := f.remembered(); got != b {
		t.Fatalf("locator changed to %q", got)
	}
}

// Proof 10: an existing valid home is recognised and never installed over.
func TestExistingHomeIsRecognisedNotOverwritten(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeMissing, Home: `Z:\gone`})
	dir := f.dir("Old")
	for _, d := range []string{"state", "runtime"} {
		if err := os.MkdirAll(filepath.Join(dir, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	f.installed[dir] = true
	v := f.selectDir(dir)
	if !v.OK() || v.Existing == nil || v.Existing.ModelID != setup.DefaultModel || v.Home != dir || v.Nested {
		t.Fatalf("validation %+v", v)
	}
	if err := f.flow.Install("cuda"); !errors.Is(err, ErrExisting) || f.setups.Load() != 0 {
		t.Fatalf("install over existing: %v (setups %d)", err, f.setups.Load())
	}
	if err := f.flow.UseExisting(); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageStarting)
	f.waitRemembered(dir)
	if f.setups.Load() != 0 || f.rt.starts != 1 {
		t.Fatalf("setups %d starts %d", f.setups.Load(), f.rt.starts)
	}
}

func TestUseExistingRequiresAnExistingInstallation(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	if err := f.flow.UseExisting(); !errors.Is(err, ErrNoSelection) {
		t.Fatal(err)
	}
	f.selectDir(f.dir("Empty"))
	if err := f.flow.UseExisting(); !errors.Is(err, ErrNoExisting) {
		t.Fatal(err)
	}
}

// Recovery: a missing stored home offers Locate/Choose with an explanation
// and does not install anything by itself.
func TestMissingHomeRecoveryView(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeMissing, Home: `D:\Hachidori`})
	v := f.flow.View()
	if v.Stage != StageSelect || v.Mode != ModeMissing || !strings.Contains(v.Notice, `D:\Hachidori`) || v.State != app.Unconfigured {
		t.Fatalf("view %+v", v)
	}
	if f.setups.Load() != 0 || f.opens.Load() != 0 {
		t.Fatal("recovery installed or started something")
	}
	if err := f.flow.Install("cpu"); !errors.Is(err, ErrNoSelection) {
		t.Fatal(err)
	}
}

// A picker cancel changes nothing.
func TestPickerCancelKeepsSelection(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	f.selectDir(f.dir("Keep"))
	f.picker.mu.Lock()
	f.picker.err = desktop.ErrPickCancelled
	f.picker.mu.Unlock()
	if _, err := f.flow.Select(context.Background()); !errors.Is(err, desktop.ErrPickCancelled) {
		t.Fatal(err)
	}
	if v := f.flow.View(); v.Selection == nil || !strings.HasSuffix(v.Selection.Picked, "Keep") {
		t.Fatalf("selection lost: %+v", v.Selection)
	}
}

// If the bootstrap record cannot be written after a successful setup, the
// runtime is not started and the error is shown with Retry.
func TestRememberFailureAfterSetup(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	f.failRemember.Store(true)
	f.selectDir(f.dir("Models"))
	if err := f.flow.Install("cpu"); err != nil {
		t.Fatal(err)
	}
	v := f.waitStage(StageFailed)
	if v.Failure == nil || v.Failure.Source != "bootstrap" || !strings.Contains(v.Failure.Message, "access denied") || !v.CanRetry {
		t.Fatalf("view %+v", v)
	}
	if f.rt.starts != 0 {
		t.Fatal("runtime started without a remembered home")
	}
	f.failRemember.Store(false)
	if err := f.flow.Retry(); err != nil {
		t.Fatal(err)
	}
	f.waitStage(StageStarting)
	if f.setups.Load() != 1 || f.rt.starts != 1 {
		t.Fatalf("retry re-ran setup: setups %d starts %d", f.setups.Load(), f.rt.starts)
	}
}

// Normal launch: the desktop home takes over as soon as the runtime is bound;
// a start that fails to open is shown with Retry instead of exiting.
func TestLaunchModeHandover(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "H")
	f := newFixture(t, Plan{Mode: ModeLaunch, Home: dir})
	f.installed[dir] = true
	if f.flow.Done() {
		t.Fatal("done before the runtime is bound")
	}
	if err := f.ctl.Start(); err != nil {
		t.Fatal(err)
	}
	if !f.flow.Done() {
		t.Fatal("launch mode must show the desktop home once the runtime is bound")
	}
}

func TestInvalidInputs(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	if err := f.flow.Install("tpu"); !errors.Is(err, ErrDevice) {
		t.Fatal(err)
	}
	if err := f.flow.Install("cpu"); !errors.Is(err, ErrNoSelection) {
		t.Fatal(err)
	}
}
