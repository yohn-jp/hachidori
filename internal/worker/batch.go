package worker

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
)

// BatchMetrics is a worker-owned observation for benchmarks and status evidence.
type BatchMetrics struct {
	BatchCalls            int64
	BatchItems            int64
	BatchWindowWaitMS     float64
	BatchQueueWaitMS      float64
	BatchLastSize         int
	BatchLastWork         int
	BatchSizeDistribution map[int]int64
}

func (s *Supervisor) batchMetricsLocked() BatchMetrics {
	m := BatchMetrics{s.batchCalls, s.batchItems, s.batchWaitMS, s.batchQueueWaitMS,
		s.batchLastSize, s.batchLastWork, make(map[int]int64, len(s.batchSizes))}
	for size, count := range s.batchSizes {
		m.BatchSizeDistribution[size] = count
	}
	return m
}

func (s *Supervisor) BatchMetrics() BatchMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.batchMetricsLocked()
}

// batchRequest retains the original result boundary across a combined worker call.
type batchRequest struct {
	items    []Item
	key      string
	work     int
	length   int
	reply    chan batchReply
	started  chan struct{}
	queued   time.Time
	observer ExecutionObserver
}

type batchReply struct {
	results [][]api.Result
	ms      float64
	err     error
}

func batchMetadata(items []Item) (string, int, int) {
	if len(items) == 0 {
		return "", 0, 0
	}
	encoded, _ := json.Marshal(items[0].Questions)
	key := string(encoded)
	if len(items) == 1 && items[0].StateRef != "" {
		key = "state:" + items[0].StateRef
	}
	work, longest := 0, 0
	for _, item := range items {
		q, _ := json.Marshal(item.Questions)
		if items[0].StateRef != "" && len(items) == 1 {
			if item.StateRef != items[0].StateRef {
				return "", 0, 0
			}
		} else if string(q) != key {
			return "", 0, 0
		}
		length := len(item.State) + len(q)
		if length > longest {
			longest = length
		}
		work += length
	}
	return key, work, longest
}

// collectBatches is the sole Process.Decide caller for a Clef supervisor.
// An in-flight forward cannot be cancelled; queued callers observe ctx/done.
func (s *Supervisor) collectBatches(ctx context.Context, p *Process, incoming <-chan batchRequest) {
	var pending []batchRequest
	var residentRef string // scoped to this Process generation
	var opened time.Time
	var timer *time.Timer
	var tick <-chan time.Time
	stopTimer := func() {
		if timer != nil {
			timer.Stop()
			timer = nil
			tick = nil
		}
	}
	flush := func() {
		stopTimer()
		if len(pending) == 0 {
			return
		}
		if ctx.Err() != nil {
			pending = nil
			return
		}
		select {
		case <-p.Done():
			pending = nil
			return
		default:
		}
		dispatch := time.Now()
		items := make([]Item, 0)
		work := 0
		for _, req := range pending {
			work += req.work
		}
		s.mu.Lock()
		s.batchWaitMS += float64(dispatch.Sub(opened).Microseconds()) / 1000
		for _, req := range pending {
			s.batchQueueWaitMS += float64(dispatch.Sub(req.queued).Microseconds()) / 1000
		}
		s.batchLastSize = batchCount(pending)
		s.batchLastWork = work
		s.batchSizes[batchCount(pending)]++
		s.mu.Unlock()
		for _, req := range pending {
			items = append(items, req.items...)
			close(req.started)
		}
		execution, counts, coalesced := coalesceStates(pending, items)
		registrationRejected := false
		notify := func(at time.Time) {
			for _, req := range pending {
				if req.observer != nil {
					req.observer.Started(at)
				}
			}
		}
		execute := func(batch []Item, started func(time.Time)) ([][]api.Result, float64, error) {
			if len(batch) != 1 || batch[0].StateRef == "" || p.Info["provider"] != "clef" {
				return p.DecideObserved(batch, started)
			}
			if residentRef != batch[0].StateRef {
				started(time.Now())
				raw, err := p.Call("resident_register", map[string]any{
					"state_ref": batch[0].StateRef, "state": batch[0].State, "questions": batch[0].Questions})
				if err != nil {
					residentRef = ""
					registrationRejected = true
					return nil, 0, err
				}
				var metadata struct {
					Supported bool `json:"supported"`
				}
				if err := json.Unmarshal(raw, &metadata); err != nil {
					return nil, 0, err
				}
				if !metadata.Supported {
					return p.Decide(batch)
				}
				residentRef = batch[0].StateRef
				return p.DecideResidentObserved(batch, nil)
			}
			return p.DecideResidentObserved(batch, started)
		}
		results, ms, err := execute(execution, notify)
		// A combined shape can exceed capacity even when its constituent
		// requests fit. The authoritative worker denies that shape before a
		// forward; retry each original item independently, preserving errors.
		var isolated []batchReply
		var rejection *RequestError
		if coalesced && errors.As(err, &rejection) && (rejection.Class == api.ErrCapacity || (registrationRejected && rejection.Class == api.ErrRequestInvalid)) {
			isolated = make([]batchReply, len(pending))
			for i, item := range items {
				part, elapsed, failure := execute([]Item{item}, func(time.Time) {})
				isolated[i] = batchReply{results: part, ms: elapsed, err: failure}
			}
		}
		finished := time.Now()
		for _, req := range pending {
			if req.observer != nil {
				req.observer.Finished(finished)
			}
		}
		s.mu.Lock()
		s.batchCalls++
		s.batchItems += int64(len(items))
		s.mu.Unlock()
		if err == nil {
			if len(results) != len(execution) {
				err = &RequestError{Class: api.ErrWorkerFailure, Message: "batched worker result count mismatch"}
			} else if coalesced {
				parts := make([][]api.Result, len(counts))
				offset := 0
				for i, n := range counts {
					if offset+n > len(results[0]) {
						err = &RequestError{Class: api.ErrWorkerFailure, Message: "coalesced result count mismatch"}
						break
					}
					parts[i] = results[0][offset : offset+n]
					offset += n
				}
				if err == nil && offset != len(results[0]) {
					err = &RequestError{Class: api.ErrWorkerFailure, Message: "coalesced result count mismatch"}
				}
				if err == nil {
					results = make([][]api.Result, len(parts))
					copy(results, parts)
				}
			}
		}
		offset := 0
		for i, req := range pending {
			if isolated != nil {
				req.reply <- isolated[i]
				offset += len(req.items)
				continue
			}
			var part [][]api.Result
			if err == nil {
				part = results[offset : offset+len(req.items)]
			}
			req.reply <- batchReply{part, ms, err}
			offset += len(req.items)
		}
		pending = nil
		opened = time.Time{}
	}
	for {
		select {
		case <-ctx.Done():
			stopTimer()
			return
		case <-p.Done():
			stopTimer()
			return
		case <-tick:
			flush()
		case req := <-incoming:
			if len(pending) > 0 && s.policy.BatchWindow > 0 && !time.Now().Before(pending[0].queued.Add(s.policy.BatchWindow)) {
				flush()
			}
			if ctx.Err() != nil {
				stopTimer()
				return
			}
			select {
			case <-p.Done():
				stopTimer()
				return
			default:
			}
			if len(pending) > 0 && (pending[0].key != req.key ||
				pending[0].length > 2*req.length || req.length > 2*pending[0].length ||
				batchCount(pending)+len(req.items) > s.policy.BatchItems ||
				batchWork(pending)+req.work > s.policy.BatchWork) {
				flush()
			}
			if len(pending) == 0 {
				opened = time.Now()
			}
			pending = append(pending, req)
			if len(pending) == 1 && s.policy.BatchWindow > 0 {
				timer = time.NewTimer(max(0, time.Until(req.queued.Add(s.policy.BatchWindow))))
				tick = timer.C
			}
			if s.policy.BatchWindow == 0 || req.key == "" || batchCount(pending) >= s.policy.BatchItems ||
				batchWork(pending) >= s.policy.BatchWork {
				flush()
			}
		}
	}
}

// A dispatched forward must finish (or hit the Process timeout) before its
// caller returns. Only requests still waiting in the collector can be released
// immediately when the supervisor stops or the worker exits.
// coalesceStates combines only separately arrived, registered single-State
// requests. Duplicate question IDs cannot share one Clef record without
// changing its encoding, so those calls retain their independent boundary.
func coalesceStates(pending []batchRequest, items []Item) ([]Item, []int, bool) {
	if len(pending) < 2 || len(items) != len(pending) || items[0].StateRef == "" {
		return items, nil, false
	}
	seen := make(map[string]bool)
	combined := Item{State: items[0].State, StateRef: items[0].StateRef}
	counts := make([]int, len(items))
	for i, item := range items {
		if item.StateRef != combined.StateRef || item.State != combined.State {
			return items, nil, false
		}
		counts[i] = len(item.Questions)
		for _, q := range item.Questions {
			if seen[q.ID] || len(combined.Questions) == api.MaxQuestions {
				return items, nil, false
			}
			seen[q.ID] = true
			combined.Questions = append(combined.Questions, q)
		}
	}
	return []Item{combined}, counts, true
}

func waitBatch(req batchRequest, ctx context.Context, p *Process) (batchReply, error) {
	select {
	case answer := <-req.reply:
		return answer, nil
	case <-ctx.Done():
		select {
		case <-req.started:
			return <-req.reply, nil
		default:
			return batchReply{}, &RequestError{Class: api.ErrNotReady, Message: "worker stopped"}
		}
	case <-p.Done():
		select {
		case <-req.started:
			return <-req.reply, nil
		default:
			return batchReply{}, p.ExitFailure()
		}
	}
}

func batchCount(requests []batchRequest) int {
	n := 0
	for _, r := range requests {
		n += len(r.items)
	}
	return n
}

func batchWork(requests []batchRequest) int {
	n := 0
	for _, r := range requests {
		n += r.work
	}
	return n
}
