package worker

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	"github.com/yohn-jp/hachidori/internal/home"
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
	case "phase_die":
		// Reports how far it got, then exits with an argparse-like status.
		emit(map[string]any{"event": "phase", "phase": "importing"})
		emit(map[string]any{"event": "phase", "phase": "loading"})
		fmt.Fprintln(os.Stderr, "loading failed")
		os.Exit(2)
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
	provider := "fake"
	if strings.HasPrefix(mode, "clef") {
		provider = "clef"
	}
	emit(map[string]any{"event": "ready", "info": map[string]any{"provider": provider, "device": "cpu"}})
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	registered := ""
	for sc.Scan() {
		var req struct {
			ID       int64  `json:"id"`
			Op       string `json:"op"`
			Items    []Item `json:"items"`
			StateRef string `json:"state_ref"`
			State    string `json:"state"`
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
		case mode == "hold_shutdown" && req.Op == "shutdown":
			// A graceful shutdown that takes time: it waits for the test to
			// release it, then marks that it exited on its own.
			dir := os.Getenv("HACHIDORI_FAKE_HOLD")
			_ = os.WriteFile(filepath.Join(dir, "shutdown"), nil, 0o644)
			for {
				if _, err := os.Stat(filepath.Join(dir, "release")); err == nil {
					break
				}
				time.Sleep(2 * time.Millisecond)
			}
			_ = os.WriteFile(filepath.Join(dir, "clean_exit"), nil, 0o644)
			emit(map[string]any{"id": req.ID, "ok": true})
			os.Exit(0)
		case req.Op == "shutdown":
			emit(map[string]any{"id": req.ID, "ok": true})
			os.Exit(0)
		case (mode == "crash_on_decide" || mode == "clef_crash") && req.Op == "decide":
			os.Exit(7)
		case mode == "hang_on_decide" && req.Op == "decide":
			time.Sleep(time.Minute)
		case req.Op == "resident_register":
			if req.StateRef != home.StateRef(req.State) {
				emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": "request_invalid", "message": "State reference mismatch"}})
			} else {
				registered = req.StateRef
				emit(map[string]any{"id": req.ID, "ok": true, "result": map[string]any{"supported": true, "artifact": "fake", "effective_prefix": "fake", "payload_bytes": 2}})
			}
		case req.Op == "resident_decide" && (registered == "" || req.Items[0].StateRef != registered):
			emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": "request_invalid", "message": "resident mismatch"}})
		case req.Op == "stats":
			emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{"memory_total": 100}})
		case req.Op == "decide" && (req.Items[0].State == "invalid" || mode == "clef_error"):
			emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": "request_invalid", "message": "bad"}})
		case req.Op == "decide" || req.Op == "resident_decide":
			var results [][]api.Result
			for _, it := range req.Items {
				var rs []api.Result
				for _, q := range it.Questions {
					confidence := 0.9
					if strings.HasPrefix(mode, "clef") && strings.HasSuffix(it.State, "-1") {
						confidence = 0.8
					}
					rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: q.Choices[0], Confidence: confidence,
						Probabilities: map[string]float64{q.Choices[0]: confidence, q.Choices[1]: 1 - confidence}})
				}
				results = append(results, rs)
			}
			// pid lets tests prove requests reuse one resident process.
			emit(map[string]any{"id": req.ID, "ok": true, "results": results, "inference_ms": float64(os.Getpid())})
		}
	}
	os.Exit(0)
}

func fakeConfig(t testing.TB, mode string) Config {
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

// Everything a worker wrote before it exited is applied before the exit is
// judged: its phases are reported and a fatal class is not replaced by a
// generic startup failure. The exit used to race with the queued messages.
func TestStartupExitAppliesMessagesWrittenBeforeIt(t *testing.T) {
	for i := 0; i < 10; i++ {
		var phases []string
		// A slow observer lets the process exit while its later messages
		// are still queued, so the exit and the queue are both ready.
		_, err := Start(context.Background(), fakeConfig(t, "phase_die"), func(ph string) {
			phases = append(phases, ph)
			time.Sleep(20 * time.Millisecond)
		})
		var f *Failure
		if !errors.As(err, &f) || f.Class != ClassCrash || f.Message != "worker exited: exit status 2" ||
			fmt.Sprint(phases) != "[importing loading]" || len(f.Stderr) != 1 || f.Stderr[0] != "loading failed" {
			t.Fatalf("run %d: err = %#v phases = %v", i, err, phases)
		}
		_, err = Start(context.Background(), fakeConfig(t, "fatal"), func(string) { time.Sleep(20 * time.Millisecond) })
		if !errors.As(err, &f) || f.Class != ClassModelLoad || f.Message != "no weights" {
			t.Fatalf("run %d: fatal: err = %#v", i, err)
		}
	}
}

// A worker that dies before it says hello is a startup failure (for example
// an unusable command line), and its stderr names why.
func TestExitBeforeHelloIsAStartupFailureWithStderr(t *testing.T) {
	s := NewSupervisor(Config{Python: os.Args[0], Args: []string{"-test.run=^TestHelperNoHello$"},
		Env: []string{"HACHIDORI_FAKE_NOHELLO=1"}, StartTimeout: 5 * time.Second, RequestTimeout: time.Second}, DefaultPolicy)
	s.Run(context.Background())
	snap := s.Snapshot()
	f := snap.LastFailure
	if snap.State != StateFailed || snap.Phase != "spawning" || f == nil || f.Class != ClassStartup ||
		f.Message != "worker exited: exit status 2" || len(f.Stderr) == 0 ||
		!strings.Contains(f.Stderr[len(f.Stderr)-1], "unrecognized arguments: --provider laya") {
		t.Fatalf("snapshot = %+v failure = %+v", snap, f)
	}
}

// TestHelperNoHello is not a real test: a fake worker with a command line it
// does not understand, as argparse reports it.
func TestHelperNoHello(t *testing.T) {
	if os.Getenv("HACHIDORI_FAKE_NOHELLO") == "" {
		return
	}
	fmt.Fprintln(os.Stderr, "usage: hachidori_worker.py [-h] --model-dir MODEL_DIR")
	fmt.Fprintln(os.Stderr, "hachidori_worker.py: error: unrecognized arguments: --provider laya")
	os.Exit(2)
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
	if busy.State != StateReady || !busy.Ready || busy.QueueDepth != 0 || busy.InFlight != 1 {
		t.Fatalf("runtime state is not current: %+v", busy)
	}
	if !busy.AcceleratorStale || busy.Accelerator["memory_total"] != float64(100) {
		t.Fatalf("busy snapshot: stale=%v accelerator=%v, want the last known numbers marked stale", busy.AcceleratorStale, busy.Accelerator)
	}

	release()
	if err := <-decided; err != nil {
		t.Fatal(err)
	}
	if fresh := s.Snapshot(); fresh.AcceleratorStale || fresh.QueueDepth != 0 || fresh.InFlight != 0 {
		t.Fatalf("after the inference: stale=%v depth=%d", fresh.AcceleratorStale, fresh.QueueDepth)
	}
}

type eventObserver struct {
	mu          sync.Mutex
	queuedCount int
	startCount  int
	finishCount int
}

func (o *eventObserver) Queued(time.Time) {
	o.mu.Lock()
	o.queuedCount++
	o.mu.Unlock()
}

func (o *eventObserver) Started(time.Time) {
	o.mu.Lock()
	o.startCount++
	o.mu.Unlock()
}

func (o *eventObserver) Finished(time.Time) {
	o.mu.Lock()
	o.finishCount++
	o.mu.Unlock()
}

func (o *eventObserver) counts() (int, int, int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.queuedCount, o.startCount, o.finishCount
}

func TestDecideObservedSeparatesSupervisorQueueFromWorkerStart(t *testing.T) {
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

	first, second := &eventObserver{}, &eventObserver{}
	firstDone := make(chan error, 1)
	go func() { _, _, err := s.DecideObserved([]Item{item}, first); firstDone <- err }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(filepath.Join(dir, "started")); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the worker never received the first request")
		}
		time.Sleep(2 * time.Millisecond)
	}

	secondDone := make(chan error, 1)
	go func() { _, _, err := s.DecideObserved([]Item{item}, second); secondDone <- err }()
	deadline = time.Now().Add(5 * time.Second)
	for {
		queued, started, _ := second.counts()
		if queued == 1 {
			if started != 0 {
				t.Fatalf("second request started while the first worker call was held: %d starts", started)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the second request was not admitted to the supervisor queue")
		}
		time.Sleep(time.Millisecond)
	}
	if snapshot := s.Snapshot(); snapshot.QueueDepth != 1 || snapshot.InFlight != 1 {
		t.Fatalf("snapshot while one worker call and one request are waiting: queue=%d in-flight=%d", snapshot.QueueDepth, snapshot.InFlight)
	}
	if q, in, fin := first.counts(); q != 1 || in != 1 || fin != 0 {
		t.Fatalf("first observer counts queued=%d started=%d finished=%d, want 1/1/0", q, in, fin)
	}

	release()
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	if err := <-secondDone; err != nil {
		t.Fatal(err)
	}
	if q, in, fin := second.counts(); q != 1 || in != 1 || fin != 1 {
		t.Fatalf("second observer counts queued=%d started=%d finished=%d, want 1/1/1", q, in, fin)
	}
	if q, in, fin := first.counts(); q != 1 || in != 1 || fin != 1 {
		t.Fatalf("first observer completion counts queued=%d started=%d finished=%d, want 1/1/1", q, in, fin)
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
	// one restart allowed: wait for the second worker lifetime, read in one
	// snapshot so a stale READY of the exited worker cannot pair with the
	// restarted worker's count
	deadline := time.Now().Add(5 * time.Second)
	for st := s.Snapshot(); !(st.State == StateReady && st.Starts == 2); st = s.Snapshot() {
		if time.Now().After(deadline) {
			t.Fatalf("no restart: state %s starts %d", st.State, st.Starts)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, _, err := s.Decide([]Item{item}); !errors.As(err, new(*Failure)) {
		t.Fatalf("the restarted worker did not crash: err = %v", err)
	}
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
		if _, _, err := s.Decide([]Item{item}); !errors.As(err, new(*Failure)) {
			t.Fatalf("the first worker did not crash: err = %v", err)
		}
		// The first exit is retried within the budget; the exit after that
		// exhausts it. State and restart count are read in one snapshot: a
		// stale READY of the exited worker must not pair with the count the
		// supervisor records when it observes that exit.
		deadline := time.Now().Add(5 * time.Second)
		for st := s.Snapshot(); !(st.State == StateReady && st.Restarts == 1); st = s.Snapshot() {
			if time.Now().After(deadline) {
				t.Fatalf("no restart: state %s restarts %d", st.State, st.Restarts)
			}
			time.Sleep(10 * time.Millisecond)
		}
		if _, _, err := s.Decide([]Item{item}); !errors.As(err, new(*Failure)) {
			t.Fatalf("the restarted worker did not crash: err = %v", err)
		}
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

// stubTorch is the part of torch the worker's device policy touches.
const stubTorch = `__version__ = "stub"
class version: cuda = None
class cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def device_count(): return 0
`

// pythonWorker returns the launch configuration of the real worker script for
// provider, run with dir first on sys.path so that its stub torch and provider
// modules are imported instead of the real ones.
func pythonWorker(t *testing.T, dir, provider, device string, extra ...string) Config {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	script, _ := filepath.Abs("py/hachidori_worker.py")
	return Config{
		Python: python,
		Args: append([]string{"-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[2:]; exec(open(sys.argv[0]).read())",
			dir, script, "--model-dir", dir, "--device", device, "--manifest", filepath.Join(dir, "manifest.json"), "--provider", provider}, extra...),
		Env:            []string{"PYTHONNOUSERSITE=1", "USERNAME=hachidori"},
		StartTimeout:   20 * time.Second,
		RequestTimeout: 10 * time.Second,
	}
}

func writeFiles(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestPythonWorkerProtocol runs the real worker script against stub torch/laya
// modules, checking stdout ownership, request grouping and error mapping
// without loading a model. Skipped when no host python3 is available.
func TestPythonWorkerProtocol(t *testing.T) {
	dir := t.TempDir()
	writeFiles(t, dir, map[string]string{"torch.py": stubTorch, "laya.py": `__version__ = "stub"
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
`, "manifest.json": `{"id": "laya-base", "revision": "r1", "files": {"model.safetensors": "abc"}}`})
	cfg := pythonWorker(t, dir, "laya", "cpu")
	p, err := Start(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.Info["provider"] != "laya" || p.Info["provider_version"] != "stub" || p.Info["laya_version"] != "stub" ||
		p.Info["model_id"] != "laya-base" || p.Info["model_revision"] != "r1" || p.Info["device"] != "cpu" || p.Info["dtype"] != "float32" {
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
	_, err = Start(context.Background(), pythonWorker(t, dir, "laya", "cuda"), nil)
	var f *Failure
	if !errors.As(err, &f) || f.Class != ClassDevice {
		t.Fatalf("cuda: err = %v", err)
	}
}

// openDeciderStub imitates the opendecider package surface the worker adapter
// uses: load() of a local directory, system_one_batch(), and the encoder's
// parameters (for the device and dtype the model actually has). The stub scores
// each option from the position of its name so that results are deterministic.
const openDeciderStub = `__version__ = "stub"
class _Dev:
    type = "cpu"
    index = None
    def __str__(self): return "cpu"
import os
class _Param:
    device = _Dev()
    def __init__(self, dtype): self.dtype = dtype
class _Enc:
    def __init__(self, dtype): self.dtype = dtype
    def parameters(self): return iter([_Param(self.dtype)])
class _Impl:
    def __init__(self, dtype): self.enc = _Enc(dtype)
class Model:
    calls = []
    def __init__(self, dtype): self.impl = _Impl(dtype)
    def system_one_batch(self, states, questions):
        Model.calls.append((len(states), len(questions)))
        out = []
        for s in states:
            if s == "boom":
                raise RuntimeError("kaboom")
            if s == "invalid":
                raise ValueError("question 'x': choice question needs 'criteria' with at least 2 options")
            answers = {}
            for qid, q in questions.items():
                opts = list(q["criteria"])
                top = opts[-1] if s == "last" else opts[0]
                probs = {o: (0.6 if o == top else 0.4 / (len(opts) - 1)) for o in opts}
                a = {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}
                if s == "long":
                    a["truncated"] = True
                answers[qid] = a
            out.append({"model": "stub", "answers": answers})
        return out
def load(path, device=None, revision=None, dtype=None, base_url=None):
    assert dtype in ("float32", "bfloat16") and device == "cpu", (device, dtype)
    assert os.path.exists(os.path.join(path, "opendecider.json"))
    # STUB_IGNORE_DTYPE imitates a package that silently keeps float32.
    return Model("torch.float32" if os.environ.get("STUB_IGNORE_DTYPE") else "torch." + dtype)
`

func openDeciderDir(t *testing.T) string {
	dir := t.TempDir()
	files := map[string]string{"opendecider.json": `{"kind":"nano"}`, "model.safetensors": "weights", "head.safetensors": "head"}
	manifest := map[string]any{"id": "opendecider-nano", "revision": "r2", "files": map[string]string{}}
	for name, body := range files {
		sum := sha256.Sum256([]byte(body))
		manifest["files"].(map[string]string)[name] = hex.EncodeToString(sum[:])
	}
	mb, _ := json.Marshal(manifest)
	writeFiles(t, dir, files)
	writeFiles(t, dir, map[string]string{"torch.py": stubTorch, "opendecider.py": openDeciderStub, "manifest.json": string(mb)})
	return dir
}

// TestPythonWorkerOpenDecider runs the real worker script with the opendecider
// adapter against a stub package: the typed-choice contract is the same as for
// Laya, the model's pinned files are verified before it loads, the status
// reports the provider, model, device and dtype that were actually loaded, and
// a requested device is never replaced.
func TestPythonWorkerOpenDecider(t *testing.T) {
	dir := openDeciderDir(t)
	p, err := Start(context.Background(), pythonWorker(t, dir, "opendecider", "cpu"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.Info["provider"] != "opendecider" || p.Info["provider_version"] != "stub" || p.Info["opendecider_version"] != "stub" ||
		p.Info["model_id"] != "opendecider-nano" || p.Info["model_revision"] != "r2" ||
		p.Info["device"] != "cpu" || p.Info["dtype"] != "torch.float32" {
		t.Fatalf("info = %v", p.Info)
	}
	if _, ok := p.Info["laya_version"]; ok {
		t.Fatalf("laya_version reported by another provider: %v", p.Info)
	}
	q2 := []api.Question{{ID: "other", Type: "choice", Instructions: "x", Choices: []string{"a", "b", "c"}, Descriptions: map[string]string{"a": "first"}}}
	res, _, err := p.Decide([]Item{item, {State: "last", Questions: q2}, item})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 || res[0][0].ID != "q" || res[0][0].Choice != "yes" || res[0][0].Type != "choice" ||
		res[1][0].ID != "other" || res[1][0].Choice != "c" || res[1][0].Confidence != 0.6 {
		t.Fatalf("results = %+v", res)
	}
	var sum float64
	for _, pr := range res[1][0].Probabilities {
		sum += pr
	}
	if len(res[1][0].Probabilities) != 3 || sum < 0.999 || sum > 1.001 {
		t.Fatalf("probabilities = %v", res[1][0].Probabilities)
	}
	// Overlong state is scored (upstream truncates the state, never an option).
	if _, _, err := p.Decide([]Item{{State: "long", Questions: item.Questions}}); err != nil {
		t.Fatal(err)
	}
	var re *RequestError
	if _, _, err := p.Decide([]Item{{State: "invalid", Questions: item.Questions}}); !errors.As(err, &re) || re.Class != api.ErrRequestInvalid {
		t.Fatalf("invalid: err = %v", err)
	}
	if _, _, err := p.Decide([]Item{{State: "boom", Questions: item.Questions}}); !errors.As(err, &re) || re.Class != api.ErrInferenceFailed {
		t.Fatalf("boom: err = %v", err)
	}

	// CUDA requested on a host without CUDA is a device failure, never a CPU fallback.
	var f *Failure
	if _, err := Start(context.Background(), pythonWorker(t, dir, "opendecider", "cuda"), nil); !errors.As(err, &f) || f.Class != ClassDevice {
		t.Fatalf("cuda: err = %v", err)
	}

	// A pinned file that does not match its digest is a model-load failure before
	// anything is loaded.
	writeFiles(t, dir, map[string]string{"model.safetensors": "tampered"})
	if _, err := Start(context.Background(), pythonWorker(t, dir, "opendecider", "cpu"), nil); !errors.As(err, &f) || f.Class != ClassModelLoad ||
		!strings.Contains(f.Message, "model.safetensors") {
		t.Fatalf("tampered: err = %v", err)
	}
}

// A provider the worker does not know is refused before the model is touched.
func TestPythonWorkerRejectsUnknownProvider(t *testing.T) {
	dir := openDeciderDir(t)
	if _, err := Start(context.Background(), pythonWorker(t, dir, "openjev", "cpu"), nil); err == nil {
		t.Fatal("unknown provider accepted")
	}
}

// Issue #136: OpenDecider runs in float32 unless bfloat16 is explicitly
// requested; status reports the dtype the encoder is actually in; a package
// that does not honour the requested dtype is a model-load failure, never a
// silent float32 (or CPU) run; only OpenDecider accepts --dtype.
func TestPythonWorkerOpenDeciderDType(t *testing.T) {
	dir := openDeciderDir(t)
	for dtype, want := range map[string]string{"": "torch.float32", "float32": "torch.float32", "bfloat16": "torch.bfloat16"} {
		var extra []string
		if dtype != "" {
			extra = []string{"--dtype", dtype}
		}
		p, err := Start(context.Background(), pythonWorker(t, dir, "opendecider", "cpu", extra...), nil)
		if err != nil {
			t.Fatalf("dtype %q: %v", dtype, err)
		}
		if p.Info["dtype"] != want || p.Info["device"] != "cpu" {
			t.Fatalf("dtype %q: info = %v", dtype, p.Info)
		}
		// The typed closed-choice contract is unchanged by the dtype.
		res, _, err := p.Decide([]Item{item})
		if err != nil || len(res) != 1 || res[0][0].Type != "choice" || res[0][0].Choice != "yes" {
			t.Fatalf("dtype %q: %+v %v", dtype, res, err)
		}
		p.Close()
	}

	var f *Failure
	cfg := pythonWorker(t, dir, "opendecider", "cpu", "--dtype", "bfloat16")
	cfg.Env = append(cfg.Env, "STUB_IGNORE_DTYPE=1")
	if _, err := Start(context.Background(), cfg, nil); !errors.As(err, &f) || f.Class != ClassModelLoad ||
		!strings.Contains(f.Message, "torch.float32 instead of requested bfloat16") {
		t.Fatalf("ignored dtype: err = %v", err)
	}
	if _, err := Start(context.Background(), pythonWorker(t, dir, "opendecider", "cpu", "--dtype", "float16"), nil); err == nil {
		t.Fatal("unsupported dtype accepted")
	}
	if _, err := Start(context.Background(), pythonWorker(t, dir, "laya", "cpu", "--dtype", "bfloat16"), nil); err == nil {
		t.Fatal("laya accepted --dtype")
	}
}

// Stop leaves the API reporting not ready from the moment it begins. While
// the worker shuts down gracefully the supervisor is not ready, a request is
// not_ready (not a worker failure), and a status read does not talk to the
// worker: the worker finishes its own shutdown instead of being killed.
func TestStopIsNotReadyWhileTheWorkerShutsDown(t *testing.T) {
	dir := t.TempDir()
	cfg := fakeConfig(t, "hold_shutdown")
	cfg.Env = append(cfg.Env, "HACHIDORI_FAKE_HOLD="+dir)
	s := NewSupervisor(cfg, Policy{MaxRestarts: 3, Window: time.Minute, QueueDepth: 4})
	l := NewLifecycle(context.Background(), s)
	l.Start()
	waitState(t, s, StateReady)

	stopped := make(chan struct{})
	go func() { l.Stop(); close(stopped) }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(filepath.Join(dir, "shutdown")); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the worker never received the shutdown")
		}
		time.Sleep(2 * time.Millisecond)
	}
	// The worker is inside its graceful shutdown.
	if s.Ready() || s.State() == StateReady {
		t.Fatalf("stopping supervisor reports ready (state %s)", s.State())
	}
	var re *RequestError
	if _, _, err := s.Decide([]Item{item}); !errors.As(err, &re) || re.Class != api.ErrNotReady {
		t.Fatalf("a request during the stop: %v, want not_ready", err)
	}
	if snap := s.Snapshot(); snap.Ready || snap.State == StateReady {
		t.Fatalf("status during the stop: %+v", snap)
	}
	if err := os.WriteFile(filepath.Join(dir, "release"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	<-stopped
	if _, err := os.Stat(filepath.Join(dir, "clean_exit")); err != nil {
		t.Fatal("the worker was killed during its graceful shutdown instead of exiting on its own")
	}
	if s.State() != StateStopped || s.LastFailure() != nil {
		t.Fatalf("after the stop: state %s failure %+v", s.State(), s.LastFailure())
	}
}
