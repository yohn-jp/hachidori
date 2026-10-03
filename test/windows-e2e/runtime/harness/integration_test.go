package harness

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// The portable proof that the fixture and the harness drive the real product
// path: the real application controller opens the fixture home, the real
// supervisor starts the real hachidori_worker.py (the one this build embeds
// and delivers) on the fixture's stub packages, and a typed decision comes
// back over the real HTTP handler from the worker's inference path. Only
// hachidori.exe and the Windows desktop shell are not part of it; the Windows
// shards add them.

type stack struct {
	fix     Fixture
	mu      sync.Mutex
	binding *app.WorkerBinding
	handler http.Handler
	client  *Client
	ctl     *app.Controller
}

// ServeHTTP serves the API of the current binding, as the desktop's swap
// handler does across a rebind (a Restart after the supervisor gave up).
func (s *stack) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	h := s.handler
	s.mu.Unlock()
	h.ServeHTTP(w, r)
}

func (s *stack) current() *app.WorkerBinding {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.binding
}

func needPosixPython(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("the Windows shards exercise the same path through hachidori.exe")
	}
	py, err := HostPython()
	if err != nil {
		t.Skip(err)
	}
	return py
}

func newStack(t *testing.T, device string) *stack {
	t.Helper()
	py := needPosixPython(t)
	fix, err := NewHome(filepath.Join(t.TempDir(), "home"), device, py)
	if err != nil {
		t.Fatal(err)
	}
	s := &stack{fix: fix}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	logf, err := os.OpenFile(fix.Home.Path("logs", "worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { logf.Close() })
	open := app.WorkerRuntime(ctx, logf, worker.DefaultPolicy, func(b *app.WorkerBinding) {
		s.mu.Lock()
		s.binding, s.handler = b, server.HandlerSince(b, b.Info, b.Started)
		s.mu.Unlock()
	})
	s.ctl = app.New(app.Config{Home: fix.Home.Root, Open: open})
	t.Cleanup(func() {
		c, stop := context.WithTimeout(context.Background(), 20*time.Second)
		defer stop()
		_ = s.ctl.Close(c)
	})
	if err := s.ctl.Bind(); err != nil {
		t.Fatalf("bind: %v", err)
	}
	api := httptest.NewServer(s)
	t.Cleanup(api.Close)
	s.client = NewClient(strings.TrimPrefix(api.URL, "http://"), "")
	return s
}

func TestFixtureServesTypedInferenceThroughTheRealControllerAndWorker(t *testing.T) {
	s := newStack(t, "cpu")
	root := s.fix.Home.Root
	before, err := s.fix.Persisted()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.ctl.Start(); err != nil {
		t.Fatal(err)
	}
	st, err := s.client.WaitReady(BoundReady)
	if err != nil {
		t.Fatal(err)
	}
	if st.Runtime.ModelID != s.fix.ModelID || st.Runtime.Device != "cpu" || st.Runtime.Runtime != s.fix.RuntimeID {
		t.Fatalf("identity %+v", st.Runtime)
	}
	if info := st.Worker.Info; info["provider"] != "laya" || !strings.Contains(fmt.Sprint(info["provider_version"]), "e2e-fixture") ||
		!strings.Contains(fmt.Sprint(info["torch_version"]), "e2e-fixture") || info["device"] != "cpu" {
		t.Fatalf("worker did not report the fixture provider on the cpu: %v", info)
	}
	if st.Runtime.Worker == nil || st.Runtime.Worker.SHA256 == "" {
		t.Fatalf("no worker build identity: %+v", st.Runtime.Worker)
	}
	// The worker script is the one this build delivers into the home, content addressed.
	script := s.fix.Home.Path("workers", st.Runtime.Worker.SHA256, "hachidori_worker.py")
	if sum, err := FileSHA256(script); err != nil || sum != st.Runtime.Worker.SHA256 {
		t.Fatalf("delivered worker %s: %v %v", script, sum, err)
	}
	if err := s.expectOneWorker(root, st.Worker.PID); err != nil {
		t.Fatal(err)
	}

	state, choices := "the plan says beta, then beta again; alpha once", []string{"alpha", "beta"}
	res, err := s.client.Decide(TypedRequest(state, "q1", choices))
	if err != nil || res.Code != 200 {
		t.Fatalf("decide: %+v %v", res, err)
	}
	if err := CheckAnswer(res.Response, "q1", state, choices); err != nil {
		t.Fatal(err)
	}

	// Stop, Start, Restart: each a new worker, never two, none after Stop.
	if err := s.ctl.Stop(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.client.WaitWorkerState(worker.StateStopped, BoundStop); err != nil {
		t.Fatal(err)
	}
	if err := WaitGone("stopped worker", 30*time.Second, st.Worker.PID); err != nil {
		t.Fatal(err)
	}
	if err := s.expectNoWorker(root); err != nil {
		t.Fatal(err)
	}
	if err := s.ctl.Start(); err != nil {
		t.Fatal(err)
	}
	st2, err := s.client.WaitReady(BoundReady)
	if err != nil || st2.Worker.PID == st.Worker.PID {
		t.Fatalf("start after stop: pid %d -> %d, %v", st.Worker.PID, st2.Worker.PID, err)
	}
	if err := s.ctl.Restart(); err != nil {
		t.Fatal(err)
	}
	var st3 server.Status
	if err := Eventually(BoundReady, poll, "a new worker after restart", func() (bool, string, error) {
		x, err := s.client.WaitReady(BoundReady)
		st3 = x
		return err == nil && x.Worker.PID != st2.Worker.PID, fmt.Sprintf("pid %d", x.Worker.PID), nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := WaitGone("restarted-away worker", 30*time.Second, st2.Worker.PID); err != nil {
		t.Fatal(err)
	}
	if err := s.expectOneWorker(root, st3.Worker.PID); err != nil {
		t.Fatal(err)
	}

	// Close (the Quit path of the controller) leaves nothing owned.
	c, stop := context.WithTimeout(context.Background(), 30*time.Second)
	defer stop()
	if err := s.ctl.Close(c); err != nil {
		t.Fatal(err)
	}
	if err := WaitGone("closed worker", 30*time.Second, st3.Worker.PID); err != nil {
		t.Fatal(err)
	}
	if err := s.expectNoWorker(root); err != nil {
		t.Fatal(err)
	}
	after, err := s.fix.Persisted()
	if err != nil {
		t.Fatal(err)
	}
	if d := before.Diff(after); d != "" {
		t.Fatal(d)
	}
}

func (s *stack) expectOneWorker(root string, pid int) error {
	ws, err := WorkersOf(root)
	if err != nil {
		return err
	}
	if len(ws) != 1 || ws[0].PID != pid {
		return fmt.Errorf("worker processes %v, want exactly [%d]", PIDs(ws), pid)
	}
	return nil
}

func (s *stack) expectNoWorker(root string) error {
	ws, err := WorkersOf(root)
	if err != nil {
		return err
	}
	if len(ws) != 0 {
		return fmt.Errorf("worker processes %v remain", PIDs(ws))
	}
	return nil
}

// A requested CUDA device the (fixture) torch cannot provide is a device
// failure of the real worker script: the runtime is failed, nothing answers,
// and no cpu worker is ever started in its place.
func TestRequestedCUDAWithoutAGPUFailsAndNeverFallsBackToCPU(t *testing.T) {
	s := newStack(t, "cuda")
	if err := s.ctl.Start(); err != nil {
		t.Fatal(err)
	}
	st, err := s.client.WaitWorkerState(worker.StateFailed, BoundReady)
	if err != nil {
		t.Fatal(err)
	}
	if st.Runtime.Device != "cuda" || st.Worker.LastFailure == nil || st.Worker.LastFailure.Class != worker.ClassDevice {
		t.Fatalf("status %+v", st)
	}
	if code, h, err := s.client.Health(); err != nil || code != 503 || h.Ready {
		t.Fatalf("health %d %+v %v", code, h, err)
	}
	res, err := s.client.Decide(TypedRequest("anything", "q", []string{"a", "b"}))
	if err != nil || res.Code == 200 {
		t.Fatalf("a failed cuda runtime must not answer: %+v %v", res, err)
	}
	if err := s.expectNoWorker(s.fix.Home.Root); err != nil {
		t.Fatal(err)
	}
	snap := s.ctl.Snapshot()
	if snap.State != app.Failed || snap.Failure == nil || snap.Failure.Class != worker.ClassDevice {
		t.Fatalf("controller state %s failure %+v", snap.State, snap.Failure)
	}
	var a struct{ Device string }
	if err := ReadJSONFile(s.fix.Home.Path("state", "active-runtime.json"), &a); err != nil || a.Device != "cuda" {
		t.Fatalf("the activation record must still request cuda: %+v %v", a, err)
	}
}

// The dashboard's own forms and Diagnostics page over the real supervisor: a
// killed worker is reported as recovering and returns to READY with a new pid;
// the fourth kill within the window ends automatic recovery, which stays
// stopped until Restart.
func TestDashboardShowsBoundedRecoveryOverTheRealSupervisor(t *testing.T) {
	s := newStack(t, "cpu")
	dash := dashboard.New(dashboard.Config{
		APIAddr: s.client.APIAddr, Status: func() server.Status { return s.current().Status() }, Lifecycle: s.current().Lifecycle,
		Doctor: func(io.Writer) bool { return true }, Tunnel: tunnel.NewManager("ssh"),
		PrefsPath: filepath.Join(s.fix.Home.Root, "state", "dashboard.json"),
	})
	srv := httptest.NewServer(dash)
	t.Cleanup(srv.Close)
	s.client.DashAddr = strings.TrimPrefix(srv.URL, "http://")

	sess := &Session{T: t, Fix: s.fix, Client: s.client}
	Check(t, "Start", sess.Client.RuntimeAction("start"))
	st, err := sess.WaitReady("Start", 0)
	Check(t, "READY", err)
	pid := st.Worker.PID
	for i := 1; i <= 3; i++ {
		Check(t, "kill", sess.KillWorker(pid))
		rec, err := sess.AwaitRecovering()
		Check(t, fmt.Sprintf("kill %d recovering", i), err)
		if rec.Worker.LastFailure == nil || rec.Worker.LastFailure.Class != worker.ClassCrash {
			t.Fatalf("kill %d: last failure %+v", i, rec.Worker.LastFailure)
		}
		next, err := sess.WaitReady(fmt.Sprintf("recovery %d", i), pid)
		Check(t, "recovered", err)
		if next.Worker.Restarts != i || next.Worker.Starts != i+1 {
			t.Fatalf("kill %d: restarts %d starts %d", i, next.Worker.Restarts, next.Worker.Starts)
		}
		Check(t, "no recovery notice once READY", sess.ExpectNoRecoveryNotice())
		pid = next.Worker.PID
	}
	Check(t, "fourth kill", sess.KillWorker(pid))
	failed, err := sess.AwaitGaveUp()
	Check(t, "gave up", err)
	starts := failed.Worker.Starts
	// Bounded: three automatic restarts after the first start, no fifth start.
	if starts != 4 {
		t.Fatalf("starts = %d, want 4 (1 start + 3 bounded restarts)", starts)
	}
	Check(t, "no automatic restart after giving up", Holds(3*time.Second, 100*time.Millisecond, "failed stays failed", func() (bool, string, error) {
		again, err := sess.Client.Status()
		if err != nil {
			return false, "", err
		}
		ws, err := sess.Workers()
		return again.Worker.State == worker.StateFailed && again.Worker.Starts == starts && len(ws) == 0, fmt.Sprintf("%+v %v", again.Worker, PIDs(ws)), err
	}))
	Check(t, "Restart", sess.Client.RuntimeAction("restart"))
	back, err := sess.WaitReady("Restart", pid)
	Check(t, "READY after Restart", err)
	if back.Worker.Restarts != 0 {
		t.Fatalf("restarts after Restart: %d", back.Worker.Restarts)
	}
	Check(t, "no recovery notice after Restart", sess.ExpectNoRecoveryNotice())
}

// A damaged delivered worker script is an observable startup failure that is
// not retried by itself; the operator's Restart makes the controller open the
// home again, which delivers this build's script over the damaged copy.
func TestDamagedWorkerScriptFailsThenRestartReplacesIt(t *testing.T) {
	s := newStack(t, "cpu")
	sess := &Session{T: t, Fix: s.fix, Client: s.client}
	Check(t, "Start", s.ctl.Start())
	st, err := sess.WaitReady("Start", 0)
	Check(t, "READY", err)
	digest := st.Runtime.Worker.SHA256
	script := sess.WorkerScriptPath(st)
	Check(t, "Stop", s.ctl.Stop())
	_, err = s.client.WaitWorkerState(worker.StateStopped, BoundStop)
	Check(t, "stopped", err)
	Check(t, "damage", os.WriteFile(script, []byte("this is not a worker\n"), 0o644))

	Check(t, "Start on a damaged script", s.ctl.Start())
	failed, err := s.client.WaitWorkerState(worker.StateFailed, BoundReady)
	Check(t, "startup failure", err)
	if lf := failed.Worker.LastFailure; lf == nil || lf.Class != worker.ClassStartup {
		t.Fatalf("last failure %+v", lf)
	}
	Check(t, "Restart", s.ctl.Restart())
	again, err := sess.WaitReady("Restart", st.Worker.PID)
	Check(t, "READY", err)
	sess.ExpectWorkerScript(again)
	if sum, err := FileSHA256(script); err != nil || sum != digest {
		t.Fatalf("script not replaced: %s %v", sum, err)
	}
}

func TestHoldsReportsTheFirstViolation(t *testing.T) {
	n := 0
	err := Holds(time.Second, time.Millisecond, "inv", func() (bool, string, error) { n++; return n < 3, "n=3", nil })
	if err == nil || !strings.Contains(err.Error(), "violated: n=3") {
		t.Fatalf("%v", err)
	}
	if err := Holds(20*time.Millisecond, 5*time.Millisecond, "inv", func() (bool, string, error) { return true, "", nil }); err != nil {
		t.Fatal(err)
	}
}
