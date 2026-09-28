package doctor

import (
	"context"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// TestCertifyRealProvider is the real-provider certification path. It runs
// only when HACHIDORI_CERT_HOME points at a materialized home, and checks the
// golden path against the real private worker, Laya and the active device:
//
//	HACHIDORI_CERT_HOME=<home> go test ./internal/doctor -run TestCertifyRealProvider -v
func TestCertifyRealProvider(t *testing.T) {
	root := os.Getenv("HACHIDORI_CERT_HOME")
	if root == "" {
		t.Skip("HACHIDORI_CERT_HOME not set: real provider NOT certified")
	}
	h, err := home.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	cfg, rt, err := server.WorkerConfig(h, os.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	sup := worker.NewSupervisor(cfg, worker.Policy{QueueDepth: 8})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { sup.Run(ctx); close(done) }()
	defer func() { cancel(); <-done }()
	srv := httptest.NewServer(server.Handler(sup, rt))
	defer srv.Close()
	c := client.New(srv.URL)

	if hl, err := c.Health(); err != nil || hl.Ready {
		t.Fatalf("ready before warmup completed: %+v %v", hl, err)
	}
	deadline := time.Now().Add(10 * time.Minute)
	for !sup.Ready() {
		if sup.State() == worker.StateFailed || time.Now().After(deadline) {
			t.Fatalf("worker not ready: %s %+v", sup.State(), sup.LastFailure())
		}
		time.Sleep(200 * time.Millisecond)
	}
	snap := sup.Snapshot()
	t.Logf("provider: %v", snap.Info)
	if dev, _ := snap.Info["device"].(string); len(dev) < len(rt.Device) || dev[:len(rt.Device)] != rt.Device {
		t.Fatalf("device %q, want %q", dev, rt.Device)
	}
	pid := snap.PID
	var lat []float64
	for i := 0; i < 20; i++ {
		resp, err := c.Decide(SmokeRequest)
		if err != nil || len(resp.Results) != 1 {
			t.Fatalf("decide %d: %v", i, err)
		}
		lat = append(lat, resp.Timing.InferenceMS)
	}
	snap = sup.Snapshot()
	if snap.PID != pid || snap.Starts != 1 {
		t.Fatalf("worker not resident: pid %d -> %d, starts %d", pid, snap.PID, snap.Starts)
	}
	t.Logf("warm inference over 20 requests: p50 %.1f ms, p95 %.1f ms; load %v ms, warmup %v ms; accelerator %v",
		worker.Percentile(lat, 50), worker.Percentile(lat, 95), snap.Info["load_ms"], snap.Info["warmup_ms"], snap.Accelerator)
}
