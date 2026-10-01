package app

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/worker"
)

// residencyEnv is a controller whose Open builds a real ResidentSet (separate
// fake worker processes) from a desired selection that changes under it, as
// the settings authority does for the desktop. It counts opens and keeps every
// set it built.
type residencyEnv struct {
	t *testing.T
	c *Controller

	mu      sync.Mutex
	desired []string
	broken  map[string]bool // members whose artifacts are missing
	sets    []*ResidentSet
}

func (e *residencyEnv) want() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.desired)
}

func (e *residencyEnv) setDesired(ids ...string) {
	e.mu.Lock()
	e.desired = ids
	e.mu.Unlock()
}

func (e *residencyEnv) opens() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.sets)
}

func (e *residencyEnv) set(i int) *ResidentSet {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.sets[i]
}

func newResidencyEnv(t *testing.T, withSelection bool) *residencyEnv {
	t.Helper()
	e := &residencyEnv{t: t, broken: map[string]bool{}}
	open := func(string) (Runtime, error) {
		ids := append([]string{modelA}, additionalResidents(modelA, e.want())...)
		var members []ResidentMember
		for _, id := range ids {
			m := residentMember(t, id, "ok")
			e.mu.Lock()
			missing := e.broken[id]
			e.mu.Unlock()
			if missing {
				cause := fmt.Errorf("model %s (provider %s): not materialized", id, m.Provider)
				m.Config = worker.Config{Preflight: func() error { return cause }}
			}
			members = append(members, m)
		}
		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		s, err := NewResidentSet(ctx, noRestart, modelA, members)
		if err != nil {
			return nil, err
		}
		t.Cleanup(s.Stop)
		e.mu.Lock()
		e.sets = append(e.sets, s)
		e.mu.Unlock()
		return s, nil
	}
	cfg := Config{Home: t.TempDir(), Installed: func(string) bool { return true }, Open: open}
	if withSelection {
		cfg.Residents = e.want
	}
	e.c = New(cfg)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = e.c.Close(ctx)
	})
	return e
}

func pids(s *ResidentSet) map[string]int {
	out := map[string]int{}
	for _, m := range s.Models() {
		out[m] = residentState(s, m).PID
	}
	return out
}

// Changing the desired residents while the runtime runs is restart-required
// intent: no worker is touched, Start does not apply it, and only an explicit
// Restart opens the new set (both models, distinct processes); clearing it
// returns to one resident the same way.
func TestResidencyChangeAppliesOnlyOnExplicitRestart(t *testing.T) {
	e := newResidencyEnv(t, true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	first := e.set(0)
	bothReady(t, first)
	if snap := e.c.Snapshot(); snap.RestartRequired || snap.ResidencyChanged || snap.State != Ready || len(snap.Residents) != 1 {
		t.Fatalf("one resident: %+v", snap)
	}
	before := pids(first)

	e.setDesired(modelB)
	snap := e.c.Snapshot()
	if !snap.RestartRequired || !snap.ResidencyChanged || snap.State != Ready || len(snap.Residents) != 1 {
		t.Fatalf("selection pending: restart %v residency %v state %s residents %d", snap.RestartRequired, snap.ResidencyChanged, snap.State, len(snap.Residents))
	}
	if err := e.c.Start(); err != nil || e.opens() != 1 {
		t.Fatalf("Start applied the selection: opens %d err %v", e.opens(), err)
	}
	if !first.Running() || !slices.Equal(first.Models(), []string{modelA}) || pids(first)[modelA] != before[modelA] {
		t.Fatalf("a running worker was mutated: %v %v", first.Models(), pids(first))
	}

	if err := e.c.Restart(); err != nil {
		t.Fatal(err)
	}
	second := e.set(1)
	bothReady(t, second)
	if e.opens() != 2 || first.Running() || !slices.Equal(second.Models(), []string{modelA, modelB}) || second.Default().Model != modelA {
		t.Fatalf("restart: opens %d old running %v members %v", e.opens(), first.Running(), second.Models())
	}
	p := pids(second)
	if p[modelA] == 0 || p[modelB] == 0 || p[modelA] == p[modelB] {
		t.Fatalf("residents share a process: %v", p)
	}
	snap = e.c.Snapshot()
	if snap.RestartRequired || snap.ResidencyChanged || snap.State != Ready || len(snap.Residents) != 2 ||
		!snap.Residents[0].Default || snap.Residents[0].Model != modelA || snap.Residents[1].Model != modelB {
		t.Fatalf("after restart: %+v", snap)
	}
	for _, r := range snap.Residents {
		if !r.Running || !r.Status.Worker.Ready || r.Status.Worker.PID != p[r.Model] {
			t.Fatalf("resident status %+v", r)
		}
	}

	// Back to one resident, again only on an explicit restart.
	e.setDesired()
	if !e.c.Snapshot().ResidencyChanged || e.opens() != 2 {
		t.Fatal("clearing was not restart-required intent")
	}
	if err := e.c.Restart(); err != nil {
		t.Fatal(err)
	}
	third := e.set(2)
	bothReady(t, third)
	if !slices.Equal(third.Models(), []string{modelA}) || second.Running() || e.c.Snapshot().ResidencyChanged {
		t.Fatalf("one-resident restart: %v old running %v", third.Models(), second.Running())
	}
}

// The active model is always the default resident: selecting it as an extra
// adds nothing and is never a pending change.
func TestSelectingTheActiveModelAddsNoResident(t *testing.T) {
	e := newResidencyEnv(t, true)
	e.c.Start()
	bothReady(t, e.set(0))
	e.setDesired(modelA, modelA)
	if snap := e.c.Snapshot(); snap.ResidencyChanged || snap.RestartRequired {
		t.Fatalf("the default resident is a pending change: %+v", snap)
	}
	if err := e.c.Restart(); err != nil || !slices.Equal(e.set(e.opens()-1).Models(), []string{modelA}) {
		t.Fatalf("restart %v", err)
	}
}

// A runtime the operator stopped applies the new selection on its next Start,
// which is the next-start semantics; the selection alone starts nothing.
func TestResidencyAppliesOnNextStartAfterStop(t *testing.T) {
	e := newResidencyEnv(t, true)
	e.c.Start()
	first := e.set(0)
	bothReady(t, first)
	if err := e.c.Stop(); err != nil {
		t.Fatal(err)
	}
	e.setDesired(modelB)
	if e.opens() != 1 || first.Running() || e.c.Snapshot().State != Installed {
		t.Fatalf("selecting started or opened something: opens %d state %s", e.opens(), e.c.Snapshot().State)
	}
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	second := e.set(1)
	bothReady(t, second)
	if !slices.Equal(second.Models(), []string{modelA, modelB}) {
		t.Fatalf("members %v", second.Models())
	}
}

// A selected resident whose artifacts are missing is reported for that
// resident, by name; it is not a pending change once bound, it is never
// replaced by another model, and the default resident keeps serving.
func TestSelectedResidentWithMissingArtifactsIsReportedNotSubstituted(t *testing.T) {
	e := newResidencyEnv(t, true)
	e.broken[modelB] = true
	e.setDesired(modelB)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	s := e.set(0)
	waitFor(t, "A ready and B failed", func() bool {
		return residentState(s, modelA).State == worker.StateReady && residentState(s, modelB).State == worker.StateFailed
	})
	snap := e.c.Snapshot()
	if snap.State != Failed || snap.Failure == nil || snap.Failure.Model != modelB || snap.Failure.Class != worker.ClassPreflight {
		t.Fatalf("state %s failure %+v", snap.State, snap.Failure)
	}
	if snap.ResidencyChanged || len(snap.Residents) != 2 || snap.Residents[1].Status.Worker.Ready || snap.Residents[1].Status.Worker.PID != 0 {
		t.Fatalf("snapshot %+v", snap)
	}
	if got, _, _ := decideOn(t, s, modelA, "x"); got != modelA {
		t.Fatalf("A answered as %s", got)
	}
	if _, _, err := s.DecideOn(modelB, item("x")); err == nil {
		t.Fatal("the missing resident answered")
	}
}

// Without a selection source (serve, dashboard, tests) the controller never
// reports or applies residency: existing behavior is unchanged.
func TestNoSelectionSourceNeverReportsResidencyChange(t *testing.T) {
	e := newResidencyEnv(t, false)
	e.c.Start()
	bothReady(t, e.set(0))
	e.setDesired(modelB)
	if snap := e.c.Snapshot(); snap.ResidencyChanged || snap.RestartRequired {
		t.Fatalf("%+v", snap)
	}
	if err := e.c.Restart(); err != nil || e.opens() != 1 {
		t.Fatalf("restart rebound without a selection source: opens %d err %v", e.opens(), err)
	}
}
