package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// ---- fake resident worker: a separate OS process per resident ----

const fakeResidentEnv = "HACHIDORI_APP_FAKE_RESIDENT"

// fakeResidentWorker speaks the worker protocol as one resident model. The
// spec is "<model>|<mode>". It loads once (at start) and answers every
// request with its own model, PID and request sequence, so a test can see
// which process served a call and whether it reloaded. A request state of
// "sleep:<ms>[:tag]" delays the answer; "crash" kills the process.
func fakeResidentWorker(spec string) {
	model, mode, _ := strings.Cut(spec, "|")
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	fmt.Fprintln(os.Stderr, "loading "+model)
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	if mode == "fatal" {
		emit(map[string]any{"event": "phase", "phase": "loading"})
		emit(map[string]any{"event": "fatal", "class": worker.ClassModelLoad, "message": "no weights for " + model})
		os.Exit(3)
	}
	emit(map[string]any{"event": "ready", "info": map[string]any{"provider": model, "pid": os.Getpid()}})
	dec := json.NewDecoder(os.Stdin)
	seq := 0
	for {
		var req struct {
			ID    int64  `json:"id"`
			Op    string `json:"op"`
			Items []struct {
				State string `json:"state"`
			} `json:"items"`
		}
		if dec.Decode(&req) != nil {
			return
		}
		switch req.Op {
		case "shutdown":
			emit(map[string]any{"id": req.ID, "ok": true})
			return
		case "decide":
			var results [][]api.Result
			for _, it := range req.Items {
				if it.State == "crash" {
					os.Exit(9)
				}
				if ms, ok := strings.CutPrefix(it.State, "sleep:"); ok {
					ms, _, _ = strings.Cut(ms, ":")
					n, _ := strconv.Atoi(ms)
					time.Sleep(time.Duration(n) * time.Millisecond)
				}
				seq++
				results = append(results, []api.Result{{ID: "q", Type: "choice",
					Choice: fmt.Sprintf("%s|%d|%d|%s", model, os.Getpid(), seq, it.State), Confidence: 1}})
			}
			emit(map[string]any{"id": req.ID, "ok": true, "results": results, "inference_ms": 1.0})
		default:
			emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{"requests": seq}})
		}
	}
}

const (
	modelA = setup.DefaultModel    // laya-base: the default route
	modelB = setup.OpenDeciderNano // opendecider-nano
)

func residentMember(t *testing.T, model, mode string) ResidentMember {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return ResidentMember{Model: model, Provider: "fake-" + model, Info: server.Runtime{Model: model, ModelID: model, Device: "cuda"},
		Config: worker.Config{Python: exe, Args: []string{"-test.run=^$"}, Env: []string{fakeResidentEnv + "=" + model + "|" + mode},
			StartTimeout: 10 * time.Second, RequestTimeout: 5 * time.Second}}
}

func newResidentSet(t *testing.T, policy worker.Policy, members ...ResidentMember) *ResidentSet {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	s, err := NewResidentSet(ctx, policy, members[0].Model, members)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Stop)
	return s
}

var noRestart = worker.Policy{MaxRestarts: 0, Window: time.Minute, QueueDepth: 8}

func item(state string) []worker.Item { return []worker.Item{{State: state, Questions: nil}} }

// decideOn returns the fake worker's answer: model, pid, request sequence.
func decideOn(t *testing.T, s *ResidentSet, model, state string) (gotModel string, pid, seq int) {
	t.Helper()
	res, _, err := s.DecideOn(model, item(state))
	if err != nil {
		t.Fatalf("decide on %q: %v", model, err)
	}
	parts := strings.SplitN(res[0][0].Choice, "|", 4)
	pid, _ = strconv.Atoi(parts[1])
	seq, _ = strconv.Atoi(parts[2])
	return parts[0], pid, seq
}

func residentState(s *ResidentSet, model string) worker.Snapshot {
	r, _ := s.Resident(model)
	return r.Supervisor.Snapshot()
}

func bothReady(t *testing.T, s *ResidentSet) {
	t.Helper()
	waitFor(t, "both residents ready", func() bool {
		for _, m := range s.Models() {
			if residentState(s, m).State != worker.StateReady {
				return false
			}
		}
		return true
	})
}

func newController(t *testing.T, s *ResidentSet) *Controller {
	t.Helper()
	c := New(Config{Home: t.TempDir(), Installed: func(string) bool { return true },
		Open: func(string) (Runtime, error) { return s, nil }})
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = c.Close(ctx)
	})
	return c
}

// ---- tests ----

// Both models are resident at once: two processes, two PIDs, each READY on its
// own, and alternating calls between them reload neither.
func TestTwoResidentsAreIndependentProcesses(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	c := newController(t, s)
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	bothReady(t, s)
	waitFor(t, "application ready", func() bool { return c.Snapshot().State == Ready })

	a, b := residentState(s, modelA), residentState(s, modelB)
	if a.PID == 0 || b.PID == 0 || a.PID == b.PID || a.PID == os.Getpid() || b.PID == os.Getpid() {
		t.Fatalf("pids %d %d (test %d)", a.PID, b.PID, os.Getpid())
	}
	var lastSeq = map[string]int{}
	for i := range 6 {
		for _, m := range []string{modelA, modelB} {
			got, pid, seq := decideOn(t, s, m, "x"+strconv.Itoa(i))
			want := a.PID
			if m == modelB {
				want = b.PID
			}
			if got != m || pid != want {
				t.Fatalf("call on %s served by %s pid %d, want pid %d", m, got, pid, want)
			}
			if seq != lastSeq[m]+1 { // one loaded model: its request sequence never restarts
				t.Fatalf("%s sequence %d after %d: the model was reloaded", m, seq, lastSeq[m])
			}
			lastSeq[m] = seq
		}
	}
	for _, m := range []string{modelA, modelB} {
		if st := residentState(s, m); st.Starts != 1 || st.Requests != 6 || st.Errors[api.ErrNotReady] != 0 {
			t.Fatalf("%s: starts %d requests %d errors %v", m, st.Starts, st.Requests, st.Errors)
		}
	}
	// The unrouted (existing) caller gets the default model.
	if got, _, _ := decideOn(t, s, "", "legacy"); got != modelA {
		t.Fatalf("default route served by %s", got)
	}
	if res, _, err := s.Decide(item("compat")); err != nil || !strings.HasPrefix(res[0][0].Choice, modelA+"|") {
		t.Fatalf("server.Decider route: %v %v", res, err)
	}
	if _, _, err := s.DecideOn("not-resident", item("x")); err == nil {
		t.Fatal("unknown model accepted")
	}

	snap := c.Snapshot()
	if len(snap.Residents) != 2 || !snap.Residents[0].Default || snap.Residents[0].Model != modelA || snap.Residents[1].Model != modelB {
		t.Fatalf("residents %+v", snap.Residents)
	}
	if snap.Status.Worker.PID != a.PID || snap.Status.Runtime.ModelID != modelA {
		t.Fatalf("compat status is not the default resident's: %+v", snap.Status)
	}
	for _, r := range snap.Residents {
		if !r.Running || r.Status.Worker.State != worker.StateReady || r.Status.Worker.PID == 0 {
			t.Fatalf("resident %+v", r)
		}
	}
}

// One resident crashing leaves the other running, serving and untouched; the
// crashed one is restarted only by an explicit per-resident action.
func TestResidentCrashDoesNotAffectOther(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	c := newController(t, s)
	c.Start()
	bothReady(t, s)
	before := residentState(s, modelA)

	if _, _, err := s.DecideOn(modelB, item("crash")); err == nil {
		t.Fatal("crash request succeeded")
	}
	waitFor(t, "B failed", func() bool { return residentState(s, modelB).State == worker.StateFailed })

	a := residentState(s, modelA)
	if a.State != worker.StateReady || a.PID != before.PID || a.Starts != 1 || a.LastFailure != nil {
		t.Fatalf("A disturbed by B's crash: %+v", a)
	}
	if _, pid, _ := decideOn(t, s, modelA, "still"); pid != before.PID {
		t.Fatalf("A answered from pid %d", pid)
	}
	if _, _, err := s.DecideOn(modelB, item("x")); err == nil || !strings.Contains(err.Error(), api.ErrNotReady) {
		t.Fatalf("failed resident must refuse as not_ready, got %v", err)
	}

	snap := c.Snapshot()
	if snap.State != Failed || snap.Failure == nil || snap.Failure.Model != modelB || snap.Failure.Provider != "fake-"+modelB ||
		snap.Failure.Class != worker.ClassCrash {
		t.Fatalf("state %s failure %+v: must name the failed resident", snap.State, snap.Failure)
	}
	if !strings.Contains(snap.Failure.Error(), modelB) {
		t.Fatalf("failure text %q does not name the model", snap.Failure.Error())
	}
	if snap.Residents[0].Status.Worker.State != worker.StateReady {
		t.Fatalf("A is not ready in the snapshot: %+v", snap.Residents[0])
	}
	if snap.Recovery == nil || snap.Recovery.State != RecoveryGaveUp || snap.Recovery.Cause.Model != modelB {
		t.Fatalf("recovery %+v", snap.Recovery)
	}

	// Explicit restart of B only.
	if err := c.RestartResident(modelB); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "B ready again", func() bool { return residentState(s, modelB).State == worker.StateReady })
	if a := residentState(s, modelA); a.PID != before.PID || a.Starts != 1 {
		t.Fatalf("B's restart restarted A: %+v", a)
	}
	waitFor(t, "application ready", func() bool { return c.Snapshot().State == Ready })
}

// The supervisor's bounded restart budget is per resident: B burning its own
// budget never spends A's.
func TestRestartBudgetIsPerResident(t *testing.T) {
	policy := worker.Policy{MaxRestarts: 2, Window: time.Minute, Backoff: 5 * time.Millisecond, QueueDepth: 8}
	s := newResidentSet(t, policy, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	s.Start()
	bothReady(t, s)
	apid := residentState(s, modelA).PID
	for range 3 { // two automatic restarts, then the budget is spent
		waitFor(t, "B ready", func() bool { return residentState(s, modelB).State == worker.StateReady })
		if _, _, err := s.DecideOn(modelB, item("crash")); err == nil {
			t.Fatal("crash request succeeded")
		}
	}
	waitFor(t, "B gave up", func() bool { return residentState(s, modelB).State == worker.StateFailed })
	b, a := residentState(s, modelB), residentState(s, modelA)
	if b.Starts != 3 || b.Restarts != 2 {
		t.Fatalf("B starts %d restarts %d", b.Starts, b.Restarts)
	}
	if a.Starts != 1 || a.Restarts != 0 || a.State != worker.StateReady || a.PID != apid {
		t.Fatalf("A's lifecycle changed: %+v", a)
	}
}

// A resident that fails to load never reports READY, names the model and
// provider, and does not stop the other resident from reaching READY.
func TestResidentStartupFailureIsIsolatedAndNamed(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "fatal"))
	c := newController(t, s)
	c.Start()
	waitFor(t, "A ready and B failed", func() bool {
		return residentState(s, modelA).State == worker.StateReady && residentState(s, modelB).State == worker.StateFailed
	})
	snap := c.Snapshot()
	if snap.State != Failed || snap.Failure.Model != modelB || snap.Failure.Provider != "fake-"+modelB ||
		snap.Failure.Class != worker.ClassModelLoad || !strings.Contains(snap.Failure.Message, "no weights for "+modelB) {
		t.Fatalf("state %s failure %+v", snap.State, snap.Failure)
	}
	if snap.Recovery != nil {
		t.Fatalf("a startup failure is not recovery: %+v", snap.Recovery)
	}
	if st := residentState(s, modelB); st.Ready || st.PID != 0 {
		t.Fatalf("B fabricated readiness: %+v", st)
	}
	if got, _, _ := decideOn(t, s, modelA, "x"); got != modelA {
		t.Fatalf("A answered as %s", got)
	}
}

// A member that is refused before launch (for example a model that is not
// materialized) is failed with the cause; the default resident is unaffected.
func TestResidentPreflightFailureNamesModel(t *testing.T) {
	missing := residentMember(t, modelB, "ok")
	cause := fmt.Errorf("model %s (provider %s): not materialized", modelB, missing.Provider)
	missing.Config = worker.Config{Preflight: func() error { return cause }}
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), missing)
	c := newController(t, s)
	c.Start()
	waitFor(t, "A ready", func() bool { return residentState(s, modelA).State == worker.StateReady })
	waitFor(t, "B failed", func() bool { return residentState(s, modelB).State == worker.StateFailed })
	snap := c.Snapshot()
	if snap.State != Failed || snap.Failure.Class != worker.ClassPreflight || snap.Failure.Model != modelB ||
		!strings.Contains(snap.Failure.Message, "not materialized") {
		t.Fatalf("state %s failure %+v", snap.State, snap.Failure)
	}
	if residentState(s, modelB).Starts != 1 || residentState(s, modelB).PID != 0 {
		t.Fatalf("a refused launch started a process: %+v", residentState(s, modelB))
	}
}

// Stopping or restarting one resident neither stops nor restarts the other,
// and the stopped one is not revived implicitly.
func TestStopAndRestartOneResident(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	c := newController(t, s)
	c.Start()
	bothReady(t, s)
	a0, b0 := residentState(s, modelA), residentState(s, modelB)

	if err := c.StopResident(modelB); err != nil {
		t.Fatal(err)
	}
	snap := c.Snapshot()
	if snap.Residents[1].Running || snap.Residents[1].Status.Worker.State != worker.StateStopped {
		t.Fatalf("B not stopped: %+v", snap.Residents[1])
	}
	if snap.State != Ready || snap.OperatorStopped || snap.Failure != nil {
		t.Fatalf("an operator-stopped resident must not fail the application: %s %+v", snap.State, snap.Failure)
	}
	if a := residentState(s, modelA); a.PID != a0.PID || a.Starts != 1 || a.State != worker.StateReady {
		t.Fatalf("A touched by B's stop: %+v", a)
	}
	if _, _, err := s.DecideOn(modelB, item("x")); err == nil || !strings.Contains(err.Error(), api.ErrNotReady) {
		t.Fatalf("stopped resident served: %v", err)
	}
	if _, pid, _ := decideOn(t, s, modelA, "x"); pid != a0.PID {
		t.Fatalf("A served by pid %d", pid)
	}
	time.Sleep(50 * time.Millisecond)
	if residentState(s, modelB).State != worker.StateStopped {
		t.Fatal("stopped resident came back by itself")
	}

	if err := c.StartResident(modelB); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "B ready", func() bool { return residentState(s, modelB).State == worker.StateReady })
	if b := residentState(s, modelB); b.PID == b0.PID || b.Starts != 2 {
		t.Fatalf("B did not get a new process: %+v (was pid %d)", b, b0.PID)
	}
	if err := c.StartResident(modelB); err != nil || residentState(s, modelB).Starts != 2 {
		t.Fatalf("start of a running resident: %v starts %d", err, residentState(s, modelB).Starts)
	}

	if err := c.RestartResident(modelA); err != nil {
		t.Fatal(err)
	}
	bothReady(t, s)
	if a, b := residentState(s, modelA), residentState(s, modelB); a.PID == a0.PID || a.Starts != 2 || b.Starts != 2 {
		t.Fatalf("restart of A: A %+v B %+v", a, b)
	}

	// Stopping the last running resident is the operator stopping the worker.
	c.StopResident(modelA)
	c.StopResident(modelB)
	if s := c.Snapshot(); !s.OperatorStopped || s.State != Installed {
		t.Fatalf("all residents stopped: %s operator_stopped %v", s.State, s.OperatorStopped)
	}
}

// Whole-set Stop and Close stop every worker; none survives.
func TestSetStopAndCloseStopEveryWorker(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	c := newController(t, s)
	c.Start()
	bothReady(t, s)
	if err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	for _, m := range s.Models() {
		if st := residentState(s, m); st.State != worker.StateStopped || st.PID != 0 {
			t.Fatalf("%s after stop: %+v", m, st)
		}
	}
	if sn := c.Snapshot(); sn.State != Installed || !sn.OperatorStopped {
		t.Fatalf("after stop: %s", sn.State)
	}
	if err := c.Restart(); err != nil {
		t.Fatal(err)
	}
	bothReady(t, s)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := c.Close(ctx); err != nil {
		t.Fatal(err)
	}
	for _, m := range s.Models() {
		if st := residentState(s, m); st.State != worker.StateStopped {
			t.Fatalf("%s after close: %+v", m, st)
		}
	}
}

// Requests to one resident are serialized by its own worker; two residents
// serve in parallel.
func TestPerWorkerSerializationAndCrossResidentParallelism(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	s.Start()
	bothReady(t, s)

	const n, each = 6, 40
	start := time.Now()
	var wg sync.WaitGroup
	seqs := make([]int, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, _, err := s.DecideOn(modelA, item("sleep:"+strconv.Itoa(each)+":"+strconv.Itoa(i)))
			if err != nil {
				t.Error(err)
				return
			}
			parts := strings.SplitN(res[0][0].Choice, "|", 4)
			seqs[i], _ = strconv.Atoi(parts[2])
			if !strings.HasSuffix(parts[3], ":"+strconv.Itoa(i)) { // each caller got its own answer
				t.Errorf("caller %d got %s", i, res[0][0].Choice)
			}
		}()
	}
	wg.Wait()
	if d := time.Since(start); d < n*each*time.Millisecond {
		t.Fatalf("%d requests of %dms finished in %s: one worker served concurrently", n, each, d)
	}
	seen := map[int]bool{}
	for _, q := range seqs {
		seen[q] = true
	}
	if len(seen) != n {
		t.Fatalf("worker sequence numbers %v: requests were not served one at a time", seqs)
	}

	// A is busy with a slow request; B answers meanwhile.
	slow := make(chan error, 1)
	go func() { _, _, err := s.DecideOn(modelA, item("sleep:400")); slow <- err }()
	waitFor(t, "A busy", func() bool { return residentState(s, modelA).QueueDepth > 0 })
	t0 := time.Now()
	if got, _, _ := decideOn(t, s, modelB, "fast"); got != modelB {
		t.Fatalf("served by %s", got)
	}
	if d := time.Since(t0); d > 300*time.Millisecond {
		t.Fatalf("B waited %s behind A", d)
	}
	if err := <-slow; err != nil {
		t.Fatal(err)
	}
}

// Snapshots, per-resident actions and decisions race freely across residents.
func TestResidentSetConcurrentUse(t *testing.T) {
	s := newResidentSet(t, worker.Policy{MaxRestarts: 5, Window: time.Minute, Backoff: time.Millisecond, QueueDepth: 64},
		residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	c := newController(t, s)
	c.Start()
	bothReady(t, s)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for _, m := range []string{modelA, modelB} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				s.DecideOn(m, item("x"))
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			c.Snapshot()
		}
	}()
	for range 3 {
		if err := c.RestartResident(modelB); err != nil {
			t.Error(err)
		}
	}
	close(stop)
	wg.Wait()
	bothReady(t, s)
	if a := residentState(s, modelA); a.Starts != 1 {
		t.Fatalf("A restarted: %+v", a)
	}
}

func TestResidentActionRejections(t *testing.T) {
	// A single binding is not a set.
	e := newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	if err := e.c.StopResident(modelA); !errors.Is(err, ErrNoResidents) {
		t.Fatalf("single binding: %v", err)
	}

	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	c := newController(t, s)
	if err := c.StartResident(modelA); !errors.Is(err, ErrNoResidents) { // nothing bound yet
		t.Fatalf("unbound: %v", err)
	}
	c.Start()
	bothReady(t, s)
	if err := c.StopResident("not-a-model"); !errors.Is(err, ErrUnknownResident) {
		t.Fatalf("unknown resident: %v", err)
	}
	if s := c.Snapshot(); s.Operation != nil || s.Last == nil || s.Last.Kind != OpStart {
		t.Fatalf("a rejected action changed state: %+v", s.Operation)
	}
	// An action in flight on one resident refuses another action.
	if err := c.StopResident(modelB); err != nil {
		t.Fatal(err)
	}
	if s := c.Snapshot(); s.Last.Kind != OpStop || s.Last.Model != modelB {
		t.Fatalf("last operation %+v", s.Last)
	}
}

func TestNewResidentSetValidation(t *testing.T) {
	ctx := context.Background()
	m := func(id string) ResidentMember { return ResidentMember{Model: id} }
	for name, tc := range map[string]struct {
		def     string
		members []ResidentMember
	}{
		"empty":           {"x", nil},
		"duplicate":       {"a", []ResidentMember{m("a"), m("a")}},
		"unknown default": {"c", []ResidentMember{m("a"), m("b")}},
		"no identity":     {"a", []ResidentMember{m("")}},
	} {
		if _, err := NewResidentSet(ctx, noRestart, tc.def, tc.members); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestAggregateResidents(t *testing.T) {
	st := func(state string) server.Status {
		return server.Status{Worker: worker.Snapshot{State: state, Ready: state == worker.StateReady}}
	}
	r := func(model, state string) ResidentStatus { return ResidentStatus{Model: model, Status: st(state)} }
	fail := &worker.FailureView{Class: worker.ClassCrash, Message: "boom"}
	for _, tc := range []struct {
		name    string
		rs      []ResidentStatus
		want    string
		culprit string
	}{
		{"all ready", []ResidentStatus{r("a", "ready"), r("b", "ready")}, "ready", ""},
		{"one starting", []ResidentStatus{r("a", "ready"), r("b", "starting")}, "starting", "b"},
		{"restarting beats starting", []ResidentStatus{r("a", "starting"), r("b", "restarting")}, "restarting", "b"},
		{"failed beats all", []ResidentStatus{r("a", "failed"), r("b", "restarting")}, "failed", "a"},
		{"default stopped, other ready", []ResidentStatus{r("a", "stopped"), r("b", "ready")}, "ready", ""},
		{"stopped is ignored beside failed", []ResidentStatus{r("a", "stopped"), r("b", "failed")}, "failed", "b"},
		{"all stopped", []ResidentStatus{r("a", "stopped"), r("b", "stopped")}, "stopped", ""},
	} {
		tc.rs[len(tc.rs)-1].Status.Worker.LastFailure = fail
		got, culprit := aggregateResidents(tc.rs[0].Status, tc.rs)
		if got.Worker.State != tc.want || got.Worker.Ready != (tc.want == "ready") {
			t.Errorf("%s: aggregate %s ready %v, want %s", tc.name, got.Worker.State, got.Worker.Ready, tc.want)
		}
		if name := func() string {
			if culprit == nil {
				return ""
			}
			return culprit.Model
		}(); name != tc.culprit {
			t.Errorf("%s: culprit %q, want %q", tc.name, name, tc.culprit)
		}
	}
}

func TestPrefixWriterTagsEachResidentLine(t *testing.T) {
	var buf bytes.Buffer
	var mu sync.Mutex
	a := &prefixWriter{mu: &mu, w: &buf, prefix: "[a] "}
	b := &prefixWriter{mu: &mu, w: &buf, prefix: "[b] "}
	var wg sync.WaitGroup
	for _, w := range []*prefixWriter{a, b} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 50 {
				fmt.Fprintln(w, "line", i)
			}
		}()
	}
	wg.Wait()
	for _, l := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if !strings.HasPrefix(l, "[a] line ") && !strings.HasPrefix(l, "[b] line ") {
			t.Fatalf("interleaved line %q", l)
		}
	}
}
