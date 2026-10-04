package worker

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
)

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
	s, stop := clefSupervisor(t, "clef", Policy{QueueDepth: 8, BatchWindow: 100 * time.Millisecond, BatchItems: 2})
	defer stop()
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
}
