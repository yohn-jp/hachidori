package worker

import (
	"encoding/json"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
)

// BenchmarkStateWorkload compares scheduler costs with the same State and
// Questions. Without HACHIDORI_STATE_BENCH_CONFIG this is protocol-fake
// evidence, not CUDA throughput or numerical-equivalence certification.
// The optional JSON file supplies {"worker": <Config>, "state": "..."} for
// the pinned Clef runtime; its referenced cases then exercise #270 residency.
func BenchmarkStateWorkload(b *testing.B) {
	var cfg struct {
		Worker Config `json:"worker"`
		State  string `json:"state"`
	}
	if path := os.Getenv("HACHIDORI_STATE_BENCH_CONFIG"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			b.Fatal(err)
		}
		if err = json.Unmarshal(data, &cfg); err != nil {
			b.Fatal(err)
		}
		if cfg.Worker.Python == "" || cfg.State == "" {
			b.Fatal("worker and state required")
		}
	}
	if cfg.State == "" {
		cfg.State = "fixed policy and context"
	}
	question := func(id string) api.Question {
		return api.Question{ID: id, Type: "choice", Instructions: "Is this policy applicable?", Choices: []string{"yes", "no"}}
	}
	for _, mode := range []string{"inline", "referenced", "resident_repeat", "same_state_coalesced"} {
		b.Run(mode, func(b *testing.B) {
			policy := Policy{QueueDepth: 16, BatchWindow: 10 * time.Millisecond, BatchItems: 4}
			var s *Supervisor
			var stop func()
			if cfg.Worker.Python != "" {
				s, stop = startBatchSupervisor(b, cfg.Worker, policy)
			} else {
				s, stop = clefSupervisor(b, "clef", policy)
			}
			defer stop()
			ref := ""
			if mode != "inline" {
				ref = home.StateRef(cfg.State)
			}
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				var wg sync.WaitGroup
				for i := 0; i < 4; i++ {
					if mode == "resident_repeat" && i > 0 {
						wg.Wait()
					}
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						id := "q"
						if mode == "same_state_coalesced" {
							id = string(rune('a' + i))
						}
						result, _, err := s.Decide([]Item{{State: cfg.State, StateRef: ref, Questions: []api.Question{question(id)}}})
						if err != nil || len(result) != 1 || len(result[0]) != 1 {
							b.Errorf("decide: %v %v", err, result)
						}
					}(i)
				}
				wg.Wait()
			}
			b.StopTimer()
			metrics := s.BatchMetrics()
			b.ReportMetric(float64(metrics.BatchCalls), "scheduler_calls")
			b.ReportMetric(float64(metrics.BatchItems), "requests")
		})
	}
}
