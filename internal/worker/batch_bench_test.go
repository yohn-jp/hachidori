package worker

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"testing"
	"time"
)

// BenchmarkClefBatch compares fixed bursts across windows and limits. By
// default it uses a fake worker (scheduling evidence, NOT GPU evidence).
// HACHIDORI_BATCH_BENCH_CONFIG can point to JSON containing
// {"worker":{"Python":"...","Args":[...],"Env":[...],"Dir":"...",
// "RequestTimeout":...},"items":[{"state":"...","questions":[...]}]}.
// The worker must use the pinned runtime and accepted immutable model. Run:
// go test ./internal/worker -run '^$' -bench BenchmarkClefBatch -benchtime=10x
func BenchmarkClefBatch(b *testing.B) {
	var config struct {
		Worker Config `json:"worker"`
		Items  []Item `json:"items"`
	}
	if path := os.Getenv("HACHIDORI_BATCH_BENCH_CONFIG"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			b.Fatal(err)
		}
		if err := json.Unmarshal(data, &config); err != nil {
			b.Fatal(err)
		}
		if config.Worker.Python == "" || len(config.Items) == 0 {
			b.Fatal("worker and items required")
		}
	}
	fixed := []Item{item}
	if len(config.Items) > 0 {
		fixed = config.Items
	}
	questions := 0
	for i := 0; i < 8; i++ {
		questions += len(fixed[i%len(fixed)].Questions)
	}
	for _, window := range []time.Duration{0, 10 * time.Millisecond, 25 * time.Millisecond, 50 * time.Millisecond, 100 * time.Millisecond} {
		for _, limit := range []int{2, 8} {
			for _, work := range []int{512, 32768} {
				b.Run(fmt.Sprintf("window=%s/items=%d/work=%d", window, limit, work), func(b *testing.B) {
					policy := Policy{QueueDepth: 64, BatchWindow: window, BatchItems: limit, BatchWork: work}
					var s *Supervisor
					var stop func()
					if config.Worker.Python != "" {
						s, stop = startBatchSupervisor(b, config.Worker, policy)
					} else {
						s, stop = clefSupervisor(b, "clef", policy)
					}
					defer stop()
					samples := make([]float64, 0, b.N*8)
					initial := s.Snapshot()
					b.ResetTimer()
					for iteration := 0; iteration < b.N; iteration++ {
						var wg sync.WaitGroup
						var mu sync.Mutex
						for i := 0; i < 8; i++ {
							wg.Add(1)
							go func(it Item) {
								defer wg.Done()
								start := time.Now()
								result, _, err := s.Decide([]Item{it})
								if err != nil || len(result) != 1 {
									b.Errorf("decide: %v %v", err, result)
									return
								}
								mu.Lock()
								samples = append(samples, float64(time.Since(start).Microseconds())/1000)
								mu.Unlock()
							}(fixed[i%len(fixed)])
						}
						wg.Wait()
					}
					b.StopTimer()
					sort.Float64s(samples)
					if len(samples) == 0 {
						return
					}
					percentile := func(p float64) float64 { return samples[int(math.Ceil(float64(len(samples))*p))-1] }
					elapsed := b.Elapsed().Seconds()
					snap := s.Snapshot()
					metrics := s.BatchMetrics()
					b.ReportMetric(percentile(.5), "p50_ms")
					b.ReportMetric(percentile(.95), "p95_ms")
					b.ReportMetric(percentile(.99), "p99_ms")
					b.ReportMetric(float64(len(samples))/elapsed, "requests/s")
					b.ReportMetric(float64(questions*b.N)/elapsed, "questions/s")
					b.ReportMetric(float64(metrics.BatchItems)/float64(metrics.BatchCalls), "items/forward")
					b.ReportMetric(float64(metrics.BatchCalls), "worker_calls")
					b.ReportMetric(metrics.BatchWindowWaitMS/float64(metrics.BatchCalls), "batch_wait_ms")
					b.ReportMetric(float64(metrics.BatchLastWork), "last_estimated_bytes")
					if peak, ok := snap.Accelerator["memory_peak_allocated"].(float64); ok {
						b.ReportMetric(peak, "peak_accelerator_bytes")
					}
					if stats, ok := snap.Accelerator["clef_batch"].(map[string]any); ok {
						if n, ok := stats["forwards"].(float64); ok {
							before := 0.0
							if old, ok := initial.Accelerator["clef_batch"].(map[string]any); ok {
								before, _ = old["forwards"].(float64)
							}
							b.ReportMetric(n-before, "model_forwards")
						}
					}
				})
			}
		}
	}
}
