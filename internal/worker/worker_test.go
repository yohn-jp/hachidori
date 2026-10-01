package worker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
)

// TestHelperWorker is not a real test: it is re-executed as a fake worker
// speaking the private protocol, selected by HACHIDORI_FAKE_MODE.
func TestHelperWorker(t *testing.T) {
	mode := os.Getenv("HACHIDORI_FAKE_MODE")
	if mode == "" {
		return
	}
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	switch mode {
	case "fatal":
		emit(map[string]any{"event": "phase", "phase": "loading"})
		emit(map[string]any{"event": "fatal", "class": ClassModelLoad, "message": "no weights"})
		os.Exit(3)
	case "die":
		fmt.Fprintln(os.Stderr, "Traceback: boom")
		os.Exit(1)
	case "garbage":
		fmt.Println("Warning: CUDA requested but not available")
		time.Sleep(time.Minute)
	case "hang_start":
		time.Sleep(time.Minute)
	case "huge_stderr":
		// One line the scanner accepts but the tail must bound, then one
		// beyond the scanner limit. A worker whose stderr is no longer read
		// blocks in these writes and never reaches ready.
		fmt.Fprintln(os.Stderr, strings.Repeat("a", 100<<10))
		fmt.Fprintln(os.Stderr, strings.Repeat("b", 2<<20))
		fmt.Fprintln(os.Stderr, "after the long line")
	}
	for _, ph := range []string{"importing", "loading", "warming"} {
		emit(map[string]any{"event": "phase", "phase": ph})
	}
	emit(map[string]any{"event": "ready", "info": map[string]any{"provider": "fake", "device": "cpu"}})
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		var req struct {
			ID    int64  `json:"id"`
			Op    string `json:"op"`
			Items []Item `json:"items"`
		}
		_ = json.Unmarshal(sc.Bytes(), &req)
		if mode == "hold_decide" && req.Op == "decide" {
			// Stay inside the inference until the test releases it, so a
			// test can observe the worker while a request is in flight.
			dir := os.Getenv("HACHIDORI_FAKE_HOLD")
			_ = os.WriteFile(filepath.Join(dir, "started"), nil, 0o644)
			for {
				if _, err := os.Stat(filepath.Join(dir, "release")); err == nil {
					break
				}
				time.Sleep(2 * time.Millisecond)
			}
		}
		switch {
		case mode == "oversized_line" && req.Op == "decide":
			// One protocol line beyond the Go side's limit, then nothing.
			_, _ = os.Stdout.WriteString(strings.Repeat("x", 17<<20) + "\n")
			time.Sleep(time.Minute)
		case req.Op == "shutdown":
			emit(map[string]any{"id": req.ID, "ok": true})
			os.Exit(0)
		case mode == "crash_on_decide" && req.Op == "decide":
			os.Exit(7)
		case mode == "hang_on_decide" && req.Op == "decide":
			time.Sleep(time.Minute)
		case req.Op == "stats":
			emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{"memory_total": 100}})
		case req.Op == "decide" && req.Items[0].State == "invalid":
			emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": "request_invalid", "message": "bad"}})
		case req.Op == "decide":
			var results [][]api.Result
			for _, it := range req.Items {
				var rs []api.Result
				for _, q := range it.Questions {
					rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: 0.9,
						Probabilities: map[string]float64{q.Choices[0]: 0.9, q.Choices[1]: 0.1}})
				}
				results = append(results, rs)
			}
			// pid lets tests prove requests reuse one resident process.
			emit(map[string]any{"id": req.ID, "ok": true, "results": results, "inference_ms": float64(os.Getpid())})
		}
	}
	os.Exit(0)
}

func fakeConfig(t *testing.T, mode string) Config {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return Config{
		Python:         exe,
		Args:           []string{"-test.run=^TestHelperWorker$"},
		Env:            []string{"HACHIDORI_FAKE_MODE=" + mode},
		StartTimeout:   5 * time.Second,
		RequestTimeout: 2 * time.Second,
	}
}

var item = Item{State: "s", Questions: []api.Question{{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}}}

func TestLifecycleReadyAndResident(t *testing.T) {
	var phases []string
	p, err := Start(context.Background(), fakeConfig(t, "ok"), func(ph string) { phases = append(phases, ph) })
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if fmt.Sprint(phases) != "[importing loading warming]" {
		t.Fatalf("phases = %v", phases)
	}
	for i := 0; i < 3; i++ {
		res, pid, err := p.Decide([]Item{item, item})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 2 || res[0][0].Choice != "yes" {
			t.Fatalf("results = %+v", res)
		}
		if int(pid) != p.PID {
			t.Fatalf("request served by pid %v, want resident %d", pid, p.PID)
		}
	}
}

func TestStartupFailureClasses(t *testing.T) {
	cases := map[string]string{
		"fatal":      ClassModelLoad,
		"die":        ClassCrash, // died after hello, before ready
		"garbage":    ClassProtocolError,
		"hang_start": ClassStartTimeout,
	}
	for mode, want := range cases {
		cfg := fakeConfig(t, mode)
		cfg.StartTimeout = time.Second
		_, err := Start(context.Background(), cfg, nil)
		var f *Failure
		if !errors.As(err, &f) || f.Class != want {
			t.Errorf("%s: err = %v, want class %s", mode, err, want)
		}
	}
	cfg := fakeConfig(t, "ok")
	cfg.Python = filepath.Join(t.TempDir(), "missing-python")
	_, err := Start(context.Background(), cfg, nil)
	var f *Failure
	if !errors.As(err, &f) || f.Class != ClassStartup {
		t.Errorf("missing interpreter: err = %v", err)
	}
}

func TestDieCapturesStderr(t *testing.T) {
	_, err := Start(context.Background(), fakeConfig(t, "die"), nil)
	var f *Failure
	if !errors.As(err, &f) || len(f.Stderr) == 0 || f.Stderr[0] != "Traceback: boom" {
		t.Fatalf("err = %#v", err)
	}
}

func TestStderrLongLineKeepsWorkerDrainedAndTailBounded(t *testing.T) {
	var log strings.Builder
	cfg := fakeConfig(t, "huge_stderr")
	cfg.StartTimeout = 3 * time.Second
	cfg.Log = &lockedWriter{w: &log}
	p, err := Start(context.Background(), cfg, nil)
	if err != nil {
		t.Fatalf("a long stderr line stalled the worker: %v", err)
	}
	defer p.Close()
	if _, _, err := p.Decide([]Item{item}); err != nil {
		t.Fatal(err)
	}
	var sawNotice bool
	for _, l := range p.tail.lines() {
		if len(l) > maxTailLine+len("...") {
			t.Fatalf("tail keeps a %d byte line", len(l))
		}
		sawNotice = sawNotice || strings.Contains(l, "exceeded the capture limit")
	}
	if !sawNotice {
		t.Errorf("tail does not say a line was dropped: %q", p.tail.lines())
	}
	if !strings.Contains(log.String(), strings.Repeat("a", 100<<10)) {
		t.Error("the worker log lost the line the tail truncated")
	}
}

type lockedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (l *lockedWriter) Write(b []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(b)
}

func TestOversizedProtocolLineIsAProtocolFailure(t *testing.T) {
	cfg := fakeConfig(t, "oversized_line")
	// Long enough that falling back to the request timeout would be visible
	// as the wrong failure class.
	cfg.RequestTimeout = 20 * time.Second
	p, err := Start(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = p.Decide([]Item{item})
	var f *Failure
	if !errors.As(err, &f) || f.Class != ClassProtocolError || !strings.Contains(f.Message, "unreadable") {
		t.Fatalf("err = %v, want a protocol failure naming the unreadable output", err)
	}
	select {
	case <-p.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("worker not reaped after the protocol failure")
	}
}

func TestSnapshotDoesNotWaitBehindAnInference(t *testing.T) {
	dir := t.TempDir()
	cfg := fakeConfig(t, "hold_decide")
	cfg.Env = append(cfg.Env, "HACHIDORI_FAKE_HOLD="+dir)
	s := NewSupervisor(cfg, Policy{QueueDepth: 2})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()
	release := func() { _ = os.WriteFile(filepath.Join(dir, "release"), nil, 0o644) }
	t.Cleanup(func() { release(); cancel(); <-done })
	waitState(t, s, StateReady)

	idle := s.Snapshot()
	if idle.AcceleratorStale || idle.Accelerator["memory_total"] != float64(100) {
		t.Fatalf("idle snapshot: stale=%v accelerator=%v", idle.AcceleratorStale, idle.Accelerator)
	}

	decided := make(chan error, 1)
	go func() { _, _, err := s.Decide([]Item{item}); decided <- err }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(filepath.Join(dir, "started")); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the worker never received the request")
		}
		time.Sleep(2 * time.Millisecond)
	}

	// The inference is held until release(). A snapshot taken now must come
	// back by itself, current except for the stale accelerator numbers.
	snapped := make(chan Snapshot, 1)
	go func() { snapped <- s.Snapshot() }()
	var busy Snapshot
	select {
	case busy = <-snapped:
	case <-time.After(5 * time.Second):
		t.Fatal("Snapshot waited for the in-flight inference")
	}
	if busy.State != StateReady || !busy.Ready || busy.QueueDepth != 1 {
		t.Fatalf("runtime state is not current: %+v", busy)
	}
	if !busy.AcceleratorStale || busy.Accelerator["memory_total"] != float64(100) {
		t.Fatalf("busy snapshot: stale=%v accelerator=%v, want the last known numbers marked stale", busy.AcceleratorStale, busy.Accelerator)
	}

	release()
	if err := <-decided; err != nil {
		t.Fatal(err)
	}
	if fresh := s.Snapshot(); fresh.AcceleratorStale || fresh.QueueDepth != 0 {
		t.Fatalf("after the inference: stale=%v depth=%d", fresh.AcceleratorStale, fresh.QueueDepth)
	}
}

func TestCrashIsWorkerFailureNotResult(t *testing.T) {
	p, err := Start(context.Background(), fakeConfig(t, "crash_on_decide"), nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = p.Decide([]Item{item})
	var f *Failure
	if !errors.As(err, &f) || f.Class != ClassCrash {
		t.Fatalf("err = %v, want worker crash", err)
	}
}

func TestUnresponsiveWorkerIsKilled(t *testing.T) {
	cfg := fakeConfig(t, "hang_on_decide")
	cfg.RequestTimeout = 300 * time.Millisecond
	p, err := Start(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = p.Decide([]Item{item})
	var f *Failure
	if !errors.As(err, &f) || f.Class != ClassUnresponsive {
		t.Fatalf("err = %v", err)
	}
	select {
	case <-p.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("worker not reaped")
	}
}

func TestRequestErrorKeepsWorker(t *testing.T) {
	p, err := Start(context.Background(), fakeConfig(t, "ok"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	_, _, err = p.Decide([]Item{{State: "invalid", Questions: item.Questions}})
	var re *RequestError
	if !errors.As(err, &re) || re.Class != api.ErrRequestInvalid {
		t.Fatalf("err = %v", err)
	}
	if _, _, err := p.Decide([]Item{item}); err != nil {
		t.Fatalf("worker unusable after request error: %v", err)
	}
}

func waitState(t *testing.T, s *Supervisor, want string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for s.State() != want {
		if time.Now().After(deadline) {
			t.Fatalf("state = %s, want %s", s.State(), want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestSupervisorNotReadyUntilWarm(t *testing.T) {
	cfg := fakeConfig(t, "hang_start")
	s := NewSupervisor(cfg, Policy{QueueDepth: 1})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()
	time.Sleep(200 * time.Millisecond)
	if s.Ready() {
		t.Fatal("ready before warmup")
	}
	_, _, err := s.Decide([]Item{item})
	var re *RequestError
	if !errors.As(err, &re) || re.Class != api.ErrNotReady {
		t.Fatalf("err = %v", err)
	}
	cancel()
	<-done
}

func TestSupervisorRestartsWithinBudget(t *testing.T) {
	s := NewSupervisor(fakeConfig(t, "crash_on_decide"), Policy{MaxRestarts: 1, Window: time.Minute, Backoff: 10 * time.Millisecond, QueueDepth: 2})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()
	waitState(t, s, StateReady)
	_, _, err := s.Decide([]Item{item})
	if _, ok := err.(*Failure); !ok {
		t.Fatalf("err = %v", err)
	}
	// one restart allowed
	deadline := time.Now().Add(5 * time.Second)
	for !(s.Ready() && s.Snapshot().Starts == 2) {
		if time.Now().After(deadline) {
			t.Fatalf("no restart: state %s starts %d", s.State(), s.Snapshot().Starts)
		}
		time.Sleep(10 * time.Millisecond)
	}
	_, _, _ = s.Decide([]Item{item})
	<-done // budget exhausted: supervisor gives up
	if s.State() != StateFailed || s.LastFailure().Class != ClassCrash {
		t.Fatalf("state %s failure %+v", s.State(), s.LastFailure())
	}
}

func TestSupervisorRunAfterGivingUpHasAFreshRestartBudget(t *testing.T) {
	s := NewSupervisor(fakeConfig(t, "crash_on_decide"), Policy{MaxRestarts: 1, Window: time.Hour, Backoff: 10 * time.Millisecond, QueueDepth: 2})
	crash := func(ctx context.Context) {
		t.Helper()
		done := make(chan struct{})
		go func() { s.Run(ctx); close(done) }()
		waitState(t, s, StateReady)
		_, _, _ = s.Decide([]Item{item})
		// The first exit is retried within the budget; the exit after that
		// exhausts it.
		deadline := time.Now().Add(5 * time.Second)
		for !(s.Ready() && s.Snapshot().Restarts == 1) {
			if time.Now().After(deadline) {
				t.Fatalf("no restart: state %s", s.State())
			}
			time.Sleep(10 * time.Millisecond)
		}
		_, _, _ = s.Decide([]Item{item})
		<-done
		if s.State() != StateFailed {
			t.Fatalf("state %s, want failed (budget exhausted)", s.State())
		}
	}
	crash(context.Background())
	// The operator's Restart runs the supervisor again. Its budget is fresh,
	// so the first exit is retried again instead of ending in failed at once.
	crash(context.Background())
	if got := s.Snapshot().Starts; got != 4 {
		t.Fatalf("starts = %d, want 4 (two per run)", got)
	}
}

func TestSupervisorStopDuringStartupIsNotAFailure(t *testing.T) {
	s := NewSupervisor(fakeConfig(t, "hang_start"), Policy{QueueDepth: 1})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.Run(ctx)
	if s.State() != StateStopped {
		t.Fatalf("state %s, want stopped", s.State())
	}
	if f := s.LastFailure(); f != nil {
		t.Fatalf("a stop requested during startup was recorded as a failure: %+v", f)
	}
	if s.Snapshot().LastFailure != nil {
		t.Fatal("the status document reports the stop as a failure")
	}
}

func TestSupervisorDoesNotRetryStartupFailure(t *testing.T) {
	s := NewSupervisor(fakeConfig(t, "fatal"), DefaultPolicy)
	s.Run(context.Background())
	if s.State() != StateFailed || s.LastFailure().Class != ClassModelLoad || s.Snapshot().Starts != 1 {
		t.Fatalf("state %s failure %+v", s.State(), s.LastFailure())
	}
}

func TestLifecycleStartStopRestart(t *testing.T) {
	s := NewSupervisor(fakeConfig(t, "ok"), Policy{QueueDepth: 2})
	l := NewLifecycle(context.Background(), s)
	defer l.Stop()
	if !l.Start() {
		t.Fatal("first start refused")
	}
	if l.Start() {
		t.Fatal("second start while running must not spawn another worker")
	}
	waitState(t, s, StateReady)
	pid := s.Snapshot().PID
	l.Stop()
	if l.Running() || s.State() != StateStopped || s.Ready() {
		t.Fatalf("after stop: running %v state %s", l.Running(), s.State())
	}
	if _, _, err := s.Decide([]Item{item}); err == nil {
		t.Fatal("decide succeeded while stopped")
	}
	l.Restart()
	waitState(t, s, StateReady)
	if s.Snapshot().PID == pid || s.Snapshot().Starts != 2 {
		t.Fatalf("restart did not start a new worker: pid %d starts %d", s.Snapshot().PID, s.Snapshot().Starts)
	}
	l.Restart()
	waitState(t, s, StateReady)
	if s.Snapshot().Starts != 3 || !l.Running() {
		t.Fatalf("starts %d", s.Snapshot().Starts)
	}
}

func TestLifecycleStartAfterFailure(t *testing.T) {
	s := NewSupervisor(fakeConfig(t, "fatal"), DefaultPolicy)
	l := NewLifecycle(context.Background(), s)
	l.Start()
	waitState(t, s, StateFailed)
	for l.Running() {
		time.Sleep(10 * time.Millisecond)
	}
	if !l.Start() {
		t.Fatal("start after failure refused")
	}
	l.Stop()
	if s.Snapshot().Starts != 2 {
		t.Fatalf("starts %d", s.Snapshot().Starts)
	}
}

func TestPercentile(t *testing.T) {
	xs := []float64{5, 1, 4, 2, 3, 6, 7, 8, 9, 10}
	if Percentile(xs, 50) != 5 || Percentile(xs, 95) != 10 || Percentile(nil, 50) != 0 {
		t.Fatal(Percentile(xs, 50), Percentile(xs, 95))
	}
}

// TestPythonWorkerProtocol runs the real worker script against stub torch/laya
// modules, checking stdout ownership, request grouping and error mapping
// without loading a model. Skipped when no host python3 is available.
func TestPythonWorkerProtocol(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("torch.py", `__version__ = "stub"
class version: cuda = None
class cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def device_count(): return 0
`)
	write("laya.py", `__version__ = "stub"
class _Dev:
    type = "cpu"
    index = None
    def __str__(self): return "cpu"
class Agent:
    device = _Dev()
    dtype = "float32"
    def predict_batch(self, states, questions):
        print("library noise on stdout")
        out = []
        for s in states:
            if s == "boom":
                raise RuntimeError("kaboom")
            if s == "invalid":
                raise ValueError("bad question")
            ans = {}
            for qid, q in questions.items():
                labels = list(q["criteria"])
                ans[qid] = {"choice": labels[0], "answer_confidence": 0.75,
                            "probabilities": {l: (0.75 if i == 0 else 0.25) for i, l in enumerate(labels)}}
            out.append({"answers": ans})
        return out
def load(path, device=None, expected_sha256=None):
    assert expected_sha256 == {"model.safetensors": "abc"}
    return Agent()
`)
	write("manifest.json", `{"files": {"model.safetensors": "abc"}}`)
	script, _ := filepath.Abs("py/hachidori_worker.py")
	cfg := Config{
		Python:         python,
		Args:           []string{"-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[2:]; exec(open(sys.argv[0]).read())", dir, script, "--model-dir", dir, "--device", "cpu", "--manifest", filepath.Join(dir, "manifest.json")},
		Env:            []string{"PYTHONNOUSERSITE=1"},
		StartTimeout:   20 * time.Second,
		RequestTimeout: 10 * time.Second,
	}
	p, err := Start(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.Info["provider"] != "laya" {
		t.Fatalf("info = %v", p.Info)
	}
	q2 := []api.Question{{ID: "other", Type: "choice", Instructions: "x", Choices: []string{"a", "b", "c"}}}
	res, _, err := p.Decide([]Item{item, {State: "t", Questions: q2}, item})
	if err != nil {
		t.Fatal(err)
	}
	if res[0][0].Choice != "yes" || res[1][0].Choice != "a" || res[2][0].ID != "q" || res[0][0].Confidence != 0.75 {
		t.Fatalf("results = %+v", res)
	}
	_, _, err = p.Decide([]Item{{State: "invalid", Questions: item.Questions}})
	var re *RequestError
	if !errors.As(err, &re) || re.Class != api.ErrRequestInvalid {
		t.Fatalf("invalid: err = %v", err)
	}
	_, _, err = p.Decide([]Item{{State: "boom", Questions: item.Questions}})
	if !errors.As(err, &re) || re.Class != api.ErrInferenceFailed {
		t.Fatalf("boom: err = %v", err)
	}
	if _, _, err := p.Decide([]Item{item}); err != nil {
		t.Fatal(err)
	}

	// CUDA requested on a host without CUDA is a device failure, never a CPU fallback.
	cfg.Args[len(cfg.Args)-3] = "cuda"
	_, err = Start(context.Background(), cfg, nil)
	var f *Failure
	if !errors.As(err, &f) || f.Class != ClassDevice {
		t.Fatalf("cuda: err = %v", err)
	}
}
