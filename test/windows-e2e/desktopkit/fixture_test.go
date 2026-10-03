package desktopkit

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// The fixture must be a home the production authorities accept, in a folder
// whose name has spaces and non-ASCII characters.
func TestInstallFixtureIsAcceptedByTheProductionAuthorities(t *testing.T) {
	root := filepath.Join(t.TempDir(), "Hachidori Data", "ハチドリ")
	inst, err := InstallFixture(root)
	if err != nil {
		t.Fatal(err)
	}
	a, rm, mm, err := inst.Home.LoadActive()
	if err != nil {
		t.Fatalf("LoadActive: %v", err)
	}
	if a.Device != FixtureDevice || a.ModelID != FixtureModel || rm.Identity != inst.Runtime || mm.ID != FixtureModel {
		t.Errorf("activation %+v runtime %s model %s", a, rm.Identity, mm.ID)
	}
	if !(firstrun.Env{}).IsInstalled(root) {
		t.Error("the first-run flow does not see the fixture as installed")
	}
	if !firstrun.Validate(root, firstrun.Env{}).OK() {
		t.Errorf("the fixture home does not validate: %+v", firstrun.Validate(root, firstrun.Env{}))
	}
	if rec, ok := setup.AssessHome(inst.Home); !ok || rec.State != setup.CompatCurrent {
		t.Errorf("the runtime is not the one this build requires: %+v (ok=%v)", rec, ok)
	}
	if fi, err := os.Stat(inst.Python); err != nil || fi.Size() == 0 {
		t.Errorf("stand-in interpreter: %v", err)
	}
}

// The stand-in worker must be a real child process the supervisor starts through
// the production launch configuration, and the controller must take the home to
// READY with it (the path the packaged executable takes, in process).
func TestFixtureHomeStartsThroughTheControllerWithOneStandInWorker(t *testing.T) {
	inst, err := InstallFixture(filepath.Join(t.TempDir(), "home dir"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, rt, err := server.WorkerConfig(inst.Home, io.Discard)
	if err != nil {
		t.Fatalf("WorkerConfig: %v", err)
	}
	if err := cfg.Preflight(); err != nil {
		t.Fatalf("launch preflight refused the fixture runtime: %v", err)
	}
	if rt.ModelID != FixtureModel || rt.Device != FixtureDevice {
		t.Fatalf("runtime %+v", rt)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctl := app.New(app.Config{Home: inst.Home.Root, Open: app.WorkerRuntime(ctx, io.Discard, worker.DefaultPolicy, nil)})
	if err := ctl.Bind(); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if snap := ctl.Snapshot(); snap.Status != nil && snap.Status.Worker.PID != 0 {
		t.Fatalf("Bind started a worker (pid %d); serving starts only on Start", snap.Status.Worker.PID)
	}
	if err := ctl.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := Poll(30*time.Second, "controller READY", func() (bool, error) {
		s := ctl.Snapshot()
		return s.State == app.Ready, nil
	}); err != nil {
		t.Fatal(err)
	}
	snap := ctl.Snapshot()
	if snap.Status == nil || snap.Status.Worker.PID == 0 || !snap.Status.Worker.Ready {
		t.Fatalf("status %+v", snap.Status)
	}
	closeCtx, closeCancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer closeCancel()
	if err := ctl.Close(closeCtx); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestStandInWorkerAnswersTheProtocol(t *testing.T) {
	inst, err := InstallFixture(filepath.Join(t.TempDir(), "h"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, _, err := server.WorkerConfig(inst.Home, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	cfg.StartTimeout, cfg.RequestTimeout = 30*time.Second, 10*time.Second
	var phases []string
	p, err := worker.Start(context.Background(), cfg, func(ph string) { phases = append(phases, ph) })
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	defer p.Close()
	if len(phases) != 3 || p.PID == 0 || p.Info["stand_in"] != true {
		t.Fatalf("phases %v pid %d info %v", phases, p.PID, p.Info)
	}
	res, pid, err := p.Decide([]worker.Item{{State: "s", Questions: []api.Question{{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}}}})
	if err != nil || len(res) != 1 || res[0][0].Choice != "yes" || int(pid) != p.PID {
		t.Fatalf("decide %+v pid %v err %v", res, pid, err)
	}
	if _, err := p.Stats(); err != nil {
		t.Fatalf("stats: %v", err)
	}
}
