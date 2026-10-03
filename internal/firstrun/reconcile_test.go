package firstrun

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Exercise the composed launch -> observer -> HTTP view -> retry -> Bind ->
// dashboard path, without invoking setup or starting the serving worker.
func TestDesktopUpgradeReconciliation(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "failure-retry"}[fail], func(t *testing.T) {
			root := t.TempDir()
			var stale, failing atomic.Bool
			stale.Store(true)
			failing.Store(fail)
			var calls, opens atomic.Int32
			entered, release := make(chan struct{}, 2), make(chan struct{}, 2)
			defer func() { release <- struct{}{} }()
			rt := &fakeRT{state: worker.StateStopped}
			ctl := app.New(app.Config{Home: root, Installed: func(string) bool { return true },
				Setup: func(string, string, string, io.Writer, *setup.Observer) error {
					t.Error("first-run setup invoked")
					return errors.New("unexpected setup")
				},
				Open: func(string) (app.Runtime, error) { opens.Add(1); return rt, nil },
				Maintenance: app.Maintenance{
					Assess: func(string) (setup.Reconciliation, bool) {
						state := setup.CompatCurrent
						if stale.Load() {
							state = setup.CompatStale
						}
						return setup.Reconciliation{State: state}, true
					},
					Reconcile: func(_ string, _ io.Writer, obs *setup.Observer) (setup.Reconciliation, bool, error) {
						calls.Add(1)
						obs.OnPhase(setup.PhasePreparing)
						obs.OnPhase(setup.PhaseRuntime)
						obs.OnProgress(setup.Progress{Step: setup.StepVerify, Detail: "existing runtime", Done: 17})
						entered <- struct{}{}
						<-release
						if failing.Load() {
							return setup.Reconciliation{}, false, errors.New("private python probe deadline exceeded")
						}
						obs.OnPhase(setup.PhaseModel)
						obs.OnPhase(setup.PhasePublish)
						obs.OnPhase(setup.PhaseActivation)
						stale.Store(false)
						return setup.Reconciliation{State: setup.CompatCurrent}, true, nil
					},
				},
			})
			flow := New(Config{Ctl: ctl, Plan: Plan{Mode: ModeLaunch, Home: root}, Env: Env{Load: func(string) (home.Active, error) { return home.Active{}, nil }}})
			h := NewHandler(flow, func() http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "dashboard") })
			})
			token := tokenOf(t, h)
			if err := ctl.Bind(); err != nil {
				t.Fatal(err)
			}
			select {
			case <-entered:
			case <-time.After(3 * time.Second):
				t.Fatal("reconcile not entered")
			}
			w := serve(h, "GET", "/wizard/state", nil, nil)
			var v View
			if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
				t.Fatal(err)
			}
			if v.Stage != StageInstalling || v.Operation == nil || v.Operation.Kind != app.OpReconcile || v.Dashboard || v.Selection != nil {
				t.Fatalf("upgrade view: %+v", v)
			}
			if v.Step == nil || v.Step.Total != 0 || v.Step.Done != 17 || v.Step.Detail != "existing runtime" {
				t.Fatalf("unknown-total progress: %+v", v.Step)
			}
			if len(v.Phases) != 6 || v.Phases[1].Status != "current" || v.Phases[3].Name != "publish" {
				t.Fatalf("phases: %+v", v.Phases)
			}
			snap := ctl.Snapshot()
			if !v.Operation.Started.Equal(snap.Operation.Started) || !v.Operation.Activity.Equal(snap.Operation.Activity) || v.ElapsedSeconds < 0 || v.IdleSeconds < 0 || v.SetupLog != app.SetupLogPath(root) {
				t.Fatalf("timing/log: %+v", v)
			}
			page := serve(h, "GET", "/", nil, nil).Body.String()
			for _, want := range []string{"Updating the Hachidori runtime", "st.total > 0", "removeAttribute(\"aria-valuenow\")", "v.dashboard", "location.replace"} {
				if !strings.Contains(page, want) {
					t.Errorf("page missing %q", want)
				}
			}
			release <- struct{}{}
			wait := func(pred func() bool) {
				t.Helper()
				deadline := time.Now().Add(3 * time.Second)
				for !pred() {
					if time.Now().After(deadline) {
						t.Fatalf("timed out: %+v", ctl.Snapshot())
					}
					time.Sleep(time.Millisecond)
				}
			}
			if fail {
				wait(func() bool { return ctl.Snapshot().State == app.Failed })
				v = flow.View()
				if !v.CanRetry || v.CanChange || v.Failure == nil || v.Failure.Phase != "runtime" || v.Failure.Step != "verifying" || !strings.Contains(v.Failure.Message, "deadline exceeded") || v.Phases[1].Status != "failed" || opens.Load() != 0 {
					t.Fatalf("failure: %+v", v)
				}
				failing.Store(false)
				w = serve(h, "POST", "/wizard/retry", url.Values{"token": {token}}, nil)
				if w.Code != 200 {
					t.Fatalf("retry: %d %s", w.Code, w.Body)
				}
				select {
				case <-entered:
				case <-time.After(3 * time.Second):
					t.Fatal("retry not entered")
				}
				release <- struct{}{}
			}
			wait(func() bool { return ctl.Snapshot().Status != nil })
			if !flow.View().Dashboard || !flow.Done() || rt.Running() || opens.Load() != 1 {
				t.Fatalf("handoff: %+v", ctl.Snapshot())
			}
			if got := serve(h, "GET", "/", nil, nil).Body.String(); got != "dashboard" {
				t.Fatalf("dashboard: %s", got)
			}
			before := calls.Load()
			if err := ctl.Bind(); err != nil {
				t.Fatal(err)
			}
			if calls.Load() != before {
				t.Fatal("compatible runtime reconciled again")
			}
		})
	}
}

type snapshotController struct {
	Controller
	snapshot app.Snapshot
}

func (c snapshotController) Snapshot() app.Snapshot { return c.snapshot }

func TestReconcileTimingAndDeterminateProjection(t *testing.T) {
	now := time.Now()
	op := &app.Operation{Kind: app.OpReconcile, Started: now.Add(-5 * time.Minute), Activity: now.Add(-3 * time.Minute), Progress: &setup.Progress{Step: setup.StepDownload, Done: 30, Total: 120}}
	ctl := snapshotController{snapshot: app.Snapshot{State: app.Installing, Home: "configured", Operation: op}}
	flow := New(Config{Ctl: ctl, Plan: Plan{Mode: ModeLaunch}})
	v := flow.View()
	if !v.Stale || v.ElapsedSeconds < 300 || v.IdleSeconds < 180 || v.Step == nil || v.Step.Total != 120 || v.Step.Done != 30 {
		t.Fatalf("stale/determinate projection: %+v", v)
	}
	ctl.snapshot.Operation = &app.Operation{Kind: app.OpBind, Started: op.Started, Activity: op.Activity}
	ctl.snapshot.State = app.Starting
	flow = New(Config{Ctl: ctl, Plan: Plan{Mode: ModeLaunch}})
	if v = flow.View(); !v.Stale || v.Operation.Kind != app.OpBind || v.Step != nil {
		t.Fatalf("silent Bind must remain observable: %+v", v)
	}
}

func TestOperationTimingAndStaleActivity(t *testing.T) {
	now := time.Now()
	op := &app.Operation{Started: now.Add(-5 * time.Minute), Activity: now.Add(-3 * time.Minute)}
	elapsed, idle, stale := operationTiming(op, now)
	if elapsed != 300 || idle != 180 || !stale {
		t.Fatalf("timing: %v %v %v", elapsed, idle, stale)
	}
	op.Activity = now.Add(-time.Second)
	if _, idle, stale = operationTiming(op, now); idle != 1 || stale {
		t.Fatal("recent activity marked stale")
	}
	op.Finished = now
	if _, _, stale = operationTiming(op, now.Add(time.Hour)); stale {
		t.Fatal("finished operation marked stalled")
	}
}
