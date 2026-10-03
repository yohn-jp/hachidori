package worker

import (
	"context"
	"encoding/json"
	"sort"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
)

// Worker states exposed through status.
const (
	StateStarting   = "starting"
	StateReady      = "ready"
	StateRestarting = "restarting"
	StateFailed     = "failed"
	StateStopped    = "stopped"
)

// PhasePreflight is the supervisor's phase while it checks that a launch is
// possible, before any process exists.
const PhasePreflight = "preflight"

// Policy bounds restarts after a worker that was READY dies.
// Startup failures (import, device, model load, warmup) are deterministic and
// are not retried: the supervisor reports them and stays failed.
type Policy struct {
	MaxRestarts int           // within Window
	Window      time.Duration // sliding window for MaxRestarts
	Backoff     time.Duration // delay before each restart
	QueueDepth  int           // max requests waiting or in flight
	BatchWindow time.Duration // maximum wait for compatible Clef requests
	BatchItems  int           // maximum Clef items per call (zero uses conservative default)
	BatchWork   int           // maximum estimated bytes per call (zero uses conservative default)
}

// DefaultPolicy is the conservative first-milestone policy.
var DefaultPolicy = Policy{MaxRestarts: 3, Window: 10 * time.Minute, Backoff: 2 * time.Second, QueueDepth: 64, BatchWindow: 10 * time.Millisecond, BatchItems: 8, BatchWork: 32768}

// Supervisor owns one resident worker and its restart policy.
type Supervisor struct {
	cfg              Config
	policy           Policy
	queue            chan struct{}
	batch            chan batchRequest
	batchCtx         context.Context
	batchCalls       int64
	batchItems       int64
	batchQueueWaitMS float64
	batchWaitMS      float64
	batchLastSize    int
	batchLastWork    int
	batchSizes       map[int]int64

	mu        sync.Mutex
	state     string
	phase     string
	proc      *Process
	info      Info
	lastFail  *Failure
	starts    int
	restarts  []time.Time
	readyAt   time.Time
	requests  int64
	errors    map[string]int64
	latencies []float64 // recent inference_ms
}

// NewSupervisor creates a supervisor; call Run to start the worker.
func NewSupervisor(cfg Config, policy Policy) *Supervisor {
	if policy.BatchItems <= 0 {
		policy.BatchItems = 8
	}
	if policy.BatchWork <= 0 {
		policy.BatchWork = 32768
	}
	if policy.BatchWindow < 0 {
		policy.BatchWindow = 0
	}
	return &Supervisor{cfg: cfg, policy: policy, queue: make(chan struct{}, policy.QueueDepth),
		state: StateStarting, errors: map[string]int64{}, batchSizes: map[int]int64{}}
}

// Run starts the worker and keeps it resident until ctx is cancelled.
// Run may be called again after it returns (see Lifecycle).
func (s *Supervisor) Run(ctx context.Context) {
	s.mu.Lock()
	if s.state == StateStopped || s.state == StateFailed {
		s.state = StateStarting
	}
	// Every Run is an operator action (Lifecycle.Start or Restart, or the
	// first start): the bounded restart budget begins afresh, as it does
	// when the desktop binds a new supervisor after the operator's Restart.
	// Without this, a Restart after the supervisor gave up leaves the old
	// window full and the next exit is not retried at all.
	s.restarts = nil
	s.mu.Unlock()
	for {
		s.mu.Lock()
		s.starts++
		s.phase = PhasePreflight
		s.mu.Unlock()
		if s.cfg.Preflight != nil {
			if err := s.cfg.Preflight(); err != nil {
				s.mu.Lock()
				s.lastFail = &Failure{Class: ClassPreflight, Message: err.Error()}
				s.state = StateFailed
				s.mu.Unlock()
				return
			}
		}
		s.mu.Lock()
		s.phase = "spawning"
		s.mu.Unlock()
		p, err := Start(ctx, s.cfg, func(ph string) { s.mu.Lock(); s.phase = ph; s.mu.Unlock() })
		if err != nil {
			s.mu.Lock()
			if ctx.Err() != nil {
				// A stop requested during startup is the operator's
				// decision, not a worker failure: it must not appear as the
				// last failure or raise a failure alert.
				s.state = StateStopped
			} else {
				s.lastFail = asFailure(err)
				s.state = StateFailed
			}
			s.mu.Unlock()
			return
		}
		s.mu.Lock()
		s.proc, s.info, s.state, s.phase, s.readyAt = p, p.Info, StateReady, "ready", time.Now()
		if p.Info["provider"] == "clef" {
			s.batch = make(chan batchRequest, s.policy.QueueDepth)
			s.batchCtx = ctx
			go s.collectBatches(ctx, p, s.batch)
		} else {
			s.batch, s.batchCtx = nil, nil
		}
		s.mu.Unlock()

		select {
		case <-ctx.Done():
			// Not ready from the moment the stop begins: while the worker
			// shuts down gracefully, requests are not_ready and status reads
			// no longer talk to it (a request on its closed stdin would kill
			// it and turn the stop into a worker failure).
			s.mu.Lock()
			s.state, s.proc = StateStopped, nil
			s.mu.Unlock()
			p.Close()
			return
		case <-p.Done():
		}

		now := time.Now()
		s.mu.Lock()
		s.proc = nil
		s.lastFail = p.ExitFailure()
		kept := s.restarts[:0]
		for _, t := range s.restarts {
			if now.Sub(t) < s.policy.Window {
				kept = append(kept, t)
			}
		}
		s.restarts = kept
		if len(s.restarts) >= s.policy.MaxRestarts {
			s.state = StateFailed
			s.mu.Unlock()
			return
		}
		s.restarts = append(s.restarts, now)
		s.state = StateRestarting
		s.mu.Unlock()

		select {
		case <-ctx.Done():
			s.mu.Lock()
			s.state = StateStopped
			s.mu.Unlock()
			return
		case <-time.After(s.policy.Backoff):
		}
	}
}

// Ready reports whether the worker is initialized, warmed and serving.
func (s *Supervisor) Ready() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == StateReady
}

// State returns the current worker state.
func (s *Supervisor) State() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

// LastFailure returns the most recent worker failure, if any.
func (s *Supervisor) LastFailure() *Failure {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastFail
}

// Decide runs items on the resident worker.
func (s *Supervisor) Decide(items []Item) ([][]api.Result, float64, error) {
	select {
	case s.queue <- struct{}{}:
		defer func() { <-s.queue }()
	default:
		s.count(api.ErrCapacity)
		return nil, 0, &RequestError{Class: api.ErrCapacity, Message: "request queue is full"}
	}
	s.mu.Lock()
	p := s.proc
	state := s.state
	batch, batchCtx := s.batch, s.batchCtx
	s.mu.Unlock()
	if p == nil || state != StateReady {
		s.count(api.ErrNotReady)
		return nil, 0, &RequestError{Class: api.ErrNotReady, Message: "worker state is " + state}
	}
	var res [][]api.Result
	var ms float64
	var err error
	key, work, length := batchMetadata(items)
	if batch != nil {
		req := batchRequest{items: items, key: key, work: work, length: length, reply: make(chan batchReply, 1), started: make(chan struct{}), queued: time.Now()}
		select {
		case batch <- req:
		case <-batchCtx.Done():
			err = &RequestError{Class: api.ErrNotReady, Message: "worker stopped"}
		case <-p.Done():
			err = p.ExitFailure()
		}
		if err == nil {
			answer, waitErr := waitBatch(req, batchCtx, p)
			res, ms, err = answer.results, answer.ms, answer.err
			if waitErr != nil {
				err = waitErr
			}
		}
	} else {
		res, ms, err = p.Decide(items)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests++
	if err != nil {
		switch e := err.(type) {
		case *RequestError:
			s.errors[e.Class]++
		default:
			s.errors[api.ErrWorkerFailure]++
		}
		return nil, 0, err
	}
	s.latencies = append(s.latencies, ms)
	if len(s.latencies) > 1024 {
		s.latencies = s.latencies[len(s.latencies)-1024:]
	}
	return res, ms, nil
}

// Trial runs one operation of a tuning trial session on the resident worker. It
// returns the worker's PID with the result so that a caller can prove it is
// still talking to the one worker whose model it has been changing: a worker
// that was restarted holds a fresh source model, not the trial state.
func (s *Supervisor) Trial(op string, args map[string]any) (json.RawMessage, int, error) {
	s.mu.Lock()
	p := s.proc
	state := s.state
	s.mu.Unlock()
	if p == nil || state != StateReady {
		return nil, 0, &RequestError{Class: api.ErrNotReady, Message: "worker state is " + state}
	}
	res, err := p.Call(op, args)
	return res, p.PID, err
}

func (s *Supervisor) count(class string) {
	s.mu.Lock()
	s.errors[class]++
	s.mu.Unlock()
}

// Snapshot is the supervisor's contribution to /v1/status.
type Snapshot struct {
	State       string         `json:"state"`
	Phase       string         `json:"phase"`
	Ready       bool           `json:"ready"`
	PID         int            `json:"pid,omitempty"`
	Starts      int            `json:"starts"`
	Restarts    int            `json:"restarts_in_window"`
	ReadySince  string         `json:"ready_since,omitempty"`
	Info        Info           `json:"provider,omitempty"`
	Accelerator map[string]any `json:"accelerator,omitempty"`
	// AcceleratorStale is set when Accelerator was taken before the request
	// now in flight (it is never fetched behind a running inference); every
	// other field of the snapshot is current.
	AcceleratorStale bool             `json:"accelerator_stale,omitempty"`
	LastFailure      *FailureView     `json:"last_failure,omitempty"`
	Requests         int64            `json:"requests"`
	Errors           map[string]int64 `json:"errors"`
	QueueDepth       int              `json:"queue_depth"`
	QueueLimit       int              `json:"queue_limit"`
	LatencyP50MS     float64          `json:"inference_p50_ms"`
	LatencyP95MS     float64          `json:"inference_p95_ms"`
}

// FailureView is the JSON form of a Failure.
type FailureView struct {
	Class   string   `json:"class"`
	Message string   `json:"message"`
	Stderr  []string `json:"stderr_tail,omitempty"`
}

// Snapshot reports current supervisor state; accelerator stats are queried
// from the worker when it is idle and are the last known ones (marked stale)
// while an inference is in flight, so a snapshot never waits for it.
func (s *Supervisor) Snapshot() Snapshot {
	s.mu.Lock()
	snap := Snapshot{State: s.state, Phase: s.phase, Ready: s.state == StateReady, Starts: s.starts,
		Restarts: len(s.restarts), Info: s.info, Requests: s.requests, Errors: map[string]int64{},
		QueueDepth: len(s.queue), QueueLimit: cap(s.queue)}
	batch := s.batchMetricsLocked()
	clef := s.info["provider"] == "clef"
	for k, v := range s.errors {
		snap.Errors[k] = v
	}
	if s.lastFail != nil {
		snap.LastFailure = &FailureView{Class: s.lastFail.Class, Message: s.lastFail.Message, Stderr: s.lastFail.Stderr}
	}
	if !s.readyAt.IsZero() && snap.Ready {
		snap.ReadySince = s.readyAt.UTC().Format(time.RFC3339)
	}
	snap.LatencyP50MS, snap.LatencyP95MS = Percentile(s.latencies, 50), Percentile(s.latencies, 95)
	p := s.proc
	s.mu.Unlock()
	if p != nil {
		snap.PID = p.PID
		if st, stale, err := p.TryStats(); err == nil && len(st) > 0 {
			snap.Accelerator, snap.AcceleratorStale = st, stale
		}
	}
	if clef {
		// Stats are volatile, unlike provider identity. Never mutate the
		// Process's cached stats map while constructing a status snapshot.
		stats := make(map[string]any, len(snap.Accelerator)+1)
		for key, value := range snap.Accelerator {
			stats[key] = value
		}
		stats["batching"] = map[string]any{
			"calls": batch.BatchCalls, "items": batch.BatchItems,
			"window_wait_ms": batch.BatchWindowWaitMS, "request_queue_wait_ms": batch.BatchQueueWaitMS,
			"last_size": batch.BatchLastSize, "last_estimated_work_bytes": batch.BatchLastWork,
			"size_distribution": batch.BatchSizeDistribution,
		}
		snap.Accelerator = stats
	}
	return snap
}

// Percentile returns the nearest-rank percentile of xs (0 when empty).
func Percentile(xs []float64, pct float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	rank := int(pct/100*float64(len(c))+0.999999) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(c) {
		rank = len(c) - 1
	}
	return c[rank]
}

func asFailure(err error) *Failure {
	if f, ok := err.(*Failure); ok {
		return f
	}
	return &Failure{Class: ClassStartup, Message: err.Error()}
}
