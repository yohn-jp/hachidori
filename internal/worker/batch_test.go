package worker

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
)

func clefSupervisor(t testing.TB, mode string, policy Policy) (*Supervisor, func()) {
	return startBatchSupervisor(t, fakeConfig(t, mode), policy)
}

func startBatchSupervisor(t testing.TB, cfg Config, policy Policy) (*Supervisor, func()) {
	t.Helper()
	s := NewSupervisor(cfg, policy)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()
	deadline := time.Now().Add(5 * time.Second)
	for !s.Ready() {
		if time.Now().After(deadline) {
			t.Fatalf("worker not ready: %+v", s.Snapshot())
		}
		time.Sleep(time.Millisecond)
	}
	return s, func() { cancel(); <-done }
}

func awaitBatchCalls(t *testing.T, s *Supervisor, n int64) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for s.BatchMetrics().BatchCalls < n {
		if time.Now().After(deadline) {
			t.Fatalf("batch calls=%d want %d", s.BatchMetrics().BatchCalls, n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestClefBatchWindowAndDemultiplex(t *testing.T) {
	s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 8, BatchWindow: 200 * time.Millisecond, BatchItems: 3})
	defer stop()
	var wg sync.WaitGroup
	failures := make(chan error, 3)
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			it := item
			it.State = fmt.Sprintf("state-%d", i)
			result, _, err := s.Decide([]Item{it})
			if err != nil {
				failures <- err
				return
			}
			want := 0.9
			if i == 1 {
				want = 0.8
			}
			if len(result) != 1 || result[0][0].Choice != "yes" || result[0][0].Confidence != want {
				failures <- fmt.Errorf("bad result for item %d: %+v", i, result)
			}
		}(i)
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	snap := s.BatchMetrics()
	if snap.BatchCalls != 1 || snap.BatchItems != 3 || snap.BatchSizeDistribution[3] != 1 {
		t.Fatalf("full batch: %+v", snap)
	}
	if _, ok := s.Snapshot().Accelerator["batching"].(map[string]any); !ok {
		t.Fatal("batch evidence absent from worker status")
	}
}

func TestClefBatchTimerZeroAndCompatibility(t *testing.T) {
	for _, window := range []time.Duration{0, 20 * time.Millisecond} {
		t.Run(window.String(), func(t *testing.T) {
			s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 4, BatchWindow: window, BatchItems: 4})
			defer stop()
			if _, _, err := s.Decide([]Item{item}); err != nil {
				t.Fatal(err)
			}
			awaitBatchCalls(t, s, 1)
			other := item
			other.Questions = append([]api.Question(nil), item.Questions...)
			other.Questions[0].Choices = []string{"no", "yes"}
			// Different choice identity must not join the first schema.
			if _, _, err := s.Decide([]Item{other}); err != nil {
				t.Fatal(err)
			}
			awaitBatchCalls(t, s, 2)
		})
	}
}

func TestClefBatchErrorFanout(t *testing.T) {
	s, stop := clefSupervisor(t, "clef_error", Policy{QueueDepth: 4, BatchWindow: time.Second, BatchItems: 2})
	defer stop()
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() { _, _, err := s.Decide([]Item{item}); errs <- err }()
	}
	for i := 0; i < 2; i++ {
		err := <-errs
		var request *RequestError
		if !errors.As(err, &request) || request.Class != api.ErrRequestInvalid {
			t.Fatalf("error=%v", err)
		}
	}
	if got := s.BatchMetrics().BatchCalls; got != 1 {
		t.Fatalf("failed combined calls=%d", got)
	}
}

func TestClefBatchPartitionsSchemasAndLengths(t *testing.T) {
	for _, different := range []string{"schema", "length"} {
		t.Run(different, func(t *testing.T) {
			s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 4, BatchWindow: 25 * time.Millisecond, BatchItems: 4})
			defer stop()
			other := item
			if different == "schema" {
				other.Questions = append([]api.Question(nil), item.Questions...)
				other.Questions[0].Choices = []string{"no", "yes"}
			} else {
				other.State = strings.Repeat("a", 500)
			}
			var wg sync.WaitGroup
			for _, it := range []Item{item, other} {
				wg.Add(1)
				go func(it Item) {
					defer wg.Done()
					if _, _, err := s.Decide([]Item{it}); err != nil {
						t.Error(err)
					}
				}(it)
			}
			wg.Wait()
			if got := s.BatchMetrics().BatchCalls; got != 2 {
				t.Fatalf("%s: %d calls", different, got)
			}
		})
	}
}

func TestClefBatchWorkLimit(t *testing.T) {
	_, work, _ := batchMetadata([]Item{item})
	s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 4, BatchWindow: time.Second, BatchItems: 4, BatchWork: work})
	defer stop()
	if _, _, err := s.Decide([]Item{item}); err != nil {
		t.Fatal(err)
	}
	if s.BatchMetrics().BatchCalls != 1 {
		t.Fatal("work budget did not flush immediately")
	}
}

func TestClefBatchCrashReleasesAndRestarts(t *testing.T) {
	s, stop := clefSupervisor(t, "clef_crash", Policy{QueueDepth: 4, BatchWindow: time.Second, BatchItems: 2, MaxRestarts: 1, Backoff: time.Millisecond, Window: time.Minute})
	defer stop()
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() { _, _, err := s.Decide([]Item{item}); errs <- err }()
	}
	for i := 0; i < 2; i++ {
		err := <-errs
		var f *Failure
		if !errors.As(err, &f) || f.Class != ClassCrash {
			t.Fatalf("crash error %v", err)
		}
	}
	deadline := time.Now().Add(3 * time.Second)
	for s.Snapshot().Starts < 2 {
		if time.Now().After(deadline) {
			t.Fatal("worker did not restart")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestClefBatchStopReleasesWaiting(t *testing.T) {
	s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 1, BatchWindow: time.Hour, BatchItems: 8})
	done := make(chan error, 1)
	go func() { _, _, err := s.Decide([]Item{item}); done <- err }()
	deadline := time.Now().Add(time.Second)
	for len(s.queue) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(s.queue) != 1 {
		t.Fatal("request did not queue")
	}
	_, _, err := s.Decide([]Item{item})
	var re *RequestError
	if !errors.As(err, &re) || re.Class != api.ErrCapacity {
		t.Fatalf("capacity error=%v", err)
	}
	stop()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("stop returned success")
		}
	case <-time.After(time.Second):
		t.Fatal("queued caller stranded")
	}
}
