package trial

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Worker error classes of the tuning trial operations (hachidori_worker.py
// TrialError).
const (
	classUnsupported  = "trial_unsupported"
	classIncompatible = "trial_incompatible"
	classStateLost    = "trial_state_lost"
)

// PolicyOfRepresentation is the policy a module representation realizes. It is
// the one place the observed representation of a module is read back as a
// tuning policy.
func PolicyOfRepresentation(rep string) string {
	switch rep {
	case RepresentationDense:
		return home.PolicySourcePrecision
	case RepresentationPacked:
		return home.PolicyW4A16
	}
	return rep // "mixed" or unknown: never equal to a requested policy
}

// Caller is the resident worker's trial operation channel (worker.Supervisor).
type Caller interface {
	Trial(op string, args map[string]any) (json.RawMessage, int, error)
}

// WorkerBackend drives a tuning trial session on a resident worker. It binds to
// the worker process that opened the session: if the supervisor ever answers
// from another process, that process holds a freshly loaded source model, not
// the trial state, and every operation reports the state as lost.
type WorkerBackend struct {
	Caller Caller
	pid    int
}

var _ Backend = (*WorkerBackend)(nil)

func (b *WorkerBackend) call(op string, args map[string]any, out any) error {
	res, pid, err := b.Caller.Trial(op, args)
	if err != nil {
		var re *worker.RequestError
		if errors.As(err, &re) {
			switch re.Class {
			case classUnsupported:
				return &UnsupportedError{Reason: re.Message}
			case classIncompatible:
				return &IncompatibleError{Reason: re.Message}
			case classStateLost:
				return &StateLostError{Err: errors.New(re.Message)}
			}
		}
		return err
	}
	if b.pid != 0 && pid != b.pid {
		return &StateLostError{Err: fmt.Errorf("the worker process changed (pid %d, then %d): the trial model state is gone", b.pid, pid)}
	}
	b.pid = pid
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(res, out); err != nil {
		return fmt.Errorf("%s: unreadable worker result: %w", op, err)
	}
	return nil
}

// Open implements Backend.
func (b *WorkerBackend) Open(_ context.Context, groups map[string][]string) (Opened, error) {
	var r struct {
		Device         string   `json:"device"`
		DType          string   `json:"dtype"`
		Backend        string   `json:"backend"`
		Modules        []Member `json:"modules"`
		CanonicalBytes int64    `json:"canonical_bytes"`
		Transform      string   `json:"transform"`
	}
	b.pid = 0
	if err := b.call("trial_open", map[string]any{"groups": groups}, &r); err != nil {
		return Opened{}, err
	}
	if r.Transform != TransformImplementation {
		return Opened{}, fmt.Errorf("the worker's trial transformation is %q, this build expects %q", r.Transform, TransformImplementation)
	}
	sort.Slice(r.Modules, func(i, j int) bool { return r.Modules[i].Module < r.Modules[j].Module })
	return Opened{Device: r.Device, DType: r.DType, Backend: r.Backend, Modules: r.Modules, CanonicalBytes: r.CanonicalBytes}, nil
}

// Transform implements Backend.
func (b *WorkerBackend) Transform(_ context.Context, c ComponentIdentity) (Built, error) {
	var r struct {
		Bytes  int64  `json:"bytes"`
		Digest string `json:"digest"`
	}
	err := b.call("trial_transform", map[string]any{"identity": map[string]any{"id": c.ID(), "members": c.Members, "transformation": c.Transformation}}, &r)
	if err != nil {
		return Built{}, err
	}
	if r.Digest == "" {
		return Built{}, errors.New("the worker built a component without a content digest")
	}
	return Built{Bytes: r.Bytes, Digest: r.Digest}, nil
}

// Release implements Backend.
func (b *WorkerBackend) Release(_ context.Context, component string) error {
	return b.call("trial_release", map[string]any{"component": component}, nil)
}

// Validate implements Backend.
func (b *WorkerBackend) Validate(_ context.Context, reps []Replacement) error {
	return b.call("trial_validate", map[string]any{"replacements": reps}, nil)
}

func (b *WorkerBackend) assemble(op string, reps []Replacement) (Applied, error) {
	var r struct {
		BytesToGPU    int64 `json:"bytes_to_gpu"`
		BytesReleased int64 `json:"bytes_released"`
		Modules       int   `json:"modules"`
	}
	if err := b.call(op, map[string]any{"replacements": reps}, &r); err != nil {
		return Applied{}, err
	}
	return Applied{BytesToGPU: r.BytesToGPU, BytesReleased: r.BytesReleased, Modules: r.Modules}, nil
}

// Apply implements Backend.
func (b *WorkerBackend) Apply(_ context.Context, reps []Replacement) (Applied, error) {
	return b.assemble("trial_apply", reps)
}

// Reconstruct implements Backend.
func (b *WorkerBackend) Reconstruct(_ context.Context, reps []Replacement) (Applied, error) {
	return b.assemble("trial_reconstruct", reps)
}

// State implements Backend.
func (b *WorkerBackend) State(_ context.Context) (Observed, error) {
	var r struct {
		Groups       map[string]string `json:"groups"`
		GPUAllocated *int64            `json:"gpu_allocated"`
		HostRSS      *int64            `json:"host_rss"`
	}
	if err := b.call("trial_state", nil, &r); err != nil {
		return Observed{}, err
	}
	o := Observed{Policies: make(map[string]string, len(r.Groups)), GPUAllocated: r.GPUAllocated, HostRSS: r.HostRSS}
	for g, rep := range r.Groups {
		o.Policies[g] = PolicyOfRepresentation(rep)
	}
	return o, nil
}

// Close implements Backend. The worker itself belongs to whoever started it.
func (b *WorkerBackend) Close(context.Context) error { return nil }
