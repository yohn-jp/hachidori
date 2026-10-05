package worker

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
)

func TestResidentStateTimeoutDoesNotReturnResults(t *testing.T) {
	cfg := fakeConfig(t, "clef_hang_resident")
	cfg.RequestTimeout = 300 * time.Millisecond
	s, stop := startBatchSupervisor(t, cfg, Policy{QueueDepth: 4, BatchWindow: 0, BatchItems: 4})
	defer stop()
	it := item
	it.State = "shared"
	it.StateRef = home.StateRef(it.State)
	_, _, err := s.Decide([]Item{it})
	var failure *Failure
	if !errors.As(err, &failure) || failure.Class != ClassUnresponsive {
		t.Fatalf("timeout: %v", err)
	}
}

func TestCombinedStateErrorIsolation(t *testing.T) {
	s, stop := clefSupervisor(t, "clef_invalid_question", Policy{QueueDepth: 8, BatchWindow: time.Second, BatchItems: 2})
	defer stop()
	var wg sync.WaitGroup
	for _, id := range []string{"good", "bad"} {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			it := item
			it.State = "shared"
			it.StateRef = home.StateRef(it.State)
			it.Questions = []api.Question{{ID: id, Type: "choice", Instructions: "choose", Choices: []string{"yes", "no"}}}
			got, _, err := s.Decide([]Item{it})
			if id == "good" && (err != nil || len(got) != 1 || got[0][0].ID != id) {
				t.Errorf("good: %v %v", got, err)
			}
			if id == "bad" {
				var failure *RequestError
				if !errors.As(err, &failure) || failure.Class != api.ErrRequestInvalid {
					t.Errorf("bad: %v", err)
				}
			}
		}(id)
	}
	wg.Wait()
}

func TestCombinedStateCapacityIsolatesRequests(t *testing.T) {
	s, stop := clefSupervisor(t, "clef_capacity", Policy{QueueDepth: 8, BatchWindow: time.Second, BatchItems: 2})
	defer stop()
	var wg sync.WaitGroup
	for _, id := range []string{"first", "second"} {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			it := item
			it.State = "shared"
			it.StateRef = home.StateRef(it.State)
			it.Questions = []api.Question{{ID: id, Type: "choice", Instructions: "choose", Choices: []string{"yes", "no"}}}
			got, _, err := s.Decide([]Item{it})
			if err != nil || len(got) != 1 || len(got[0]) != 1 || got[0][0].ID != id {
				t.Errorf("%s: %v %v", id, got, err)
			}
		}(id)
	}
	wg.Wait()
	if metrics := s.BatchMetrics(); metrics.BatchCalls != 1 {
		t.Fatal(metrics)
	}
}

func TestSameStateDifferentQuestionScheduler(t *testing.T) {
	s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 8, BatchWindow: 0, BatchItems: 2})
	defer stop()
	if err := s.SetBatchWindow(100 * time.Millisecond); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for _, id := range []string{"first", "second"} {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			it := item
			it.State = "shared"
			it.StateRef = home.StateRef("shared")
			it.Questions = []api.Question{{ID: id, Type: "choice", Instructions: "choose", Choices: []string{"yes", "no"}}}
			got, _, err := s.Decide([]Item{it})
			if err != nil || len(got) != 1 || len(got[0]) != 1 || got[0][0].ID != id || got[0][0].Choice != "yes" {
				t.Errorf("%s: %v %v", id, got, err)
			}
		}(id)
	}
	wg.Wait()
	if metrics := s.BatchMetrics(); metrics.BatchCalls != 1 || metrics.BatchItems != 2 {
		t.Fatal(fmt.Sprint(metrics))
	}
	if got := s.Snapshot().BatchWindowMS; got == nil || *got != 100 {
		t.Fatalf("status batch_window_ms=%v, want 100", got)
	}
}

func TestSupervisorBatchWindowChangeWakesPendingCollector(t *testing.T) {
	s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 4, BatchWindow: time.Second, BatchItems: 4})
	defer stop()
	if got := s.Snapshot().BatchWindowMS; got == nil || *got != 1000 {
		t.Fatalf("initial batch_window_ms=%v, want 1000", got)
	}
	for _, window := range []time.Duration{-time.Millisecond, time.Microsecond, MaxBatchWindow + time.Millisecond} {
		if err := s.SetBatchWindow(window); err == nil {
			t.Errorf("accepted invalid batch window %s", window)
		}
	}
	if got := s.BatchWindow(); got != time.Second {
		t.Fatalf("invalid changes modified window to %s", got)
	}
	it := item
	it.State = "shared"
	it.StateRef = home.StateRef(it.State)
	done := make(chan error, 1)
	go func() {
		_, _, err := s.Decide([]Item{it})
		done <- err
	}()
	deadline := time.Now().Add(time.Second)
	for s.Snapshot().QueueDepth == 0 {
		if time.Now().After(deadline) {
			t.Fatal("request did not enter the batching queue")
		}
		time.Sleep(time.Millisecond)
	}
	if got := s.BatchMetrics().BatchCalls; got != 0 {
		t.Fatalf("request flushed before window change: calls=%d", got)
	}
	if err := s.SetBatchWindow(0); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("zero window did not release the pending request")
	}
	if got := s.BatchMetrics().BatchCalls; got != 1 {
		t.Fatalf("batch calls=%d, want 1", got)
	}
}
