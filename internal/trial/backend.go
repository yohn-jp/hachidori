package trial

import (
	"context"
	"errors"
	"fmt"
)

// Backend is the resident execution the session drives: the worker that holds
// the canonical source in system RAM, builds transformed components there and
// replaces the GPU representation of a bounded set of groups in place. Every
// method that can change what the GPU model computes is all-or-nothing:
//
//   - Validate never mutates;
//   - Apply and Reconstruct either leave the model exactly as asked or exactly
//     as it was, or report a *StateLostError when they cannot guarantee either;
//   - a failed Transform leaves no partial component behind.
//
// The model object, its module identities and the canonical source are never
// replaced or written; only the per-module weight representation changes.
type Backend interface {
	// Open reports the resident source: its device and dtype, every Linear
	// module's source tensor facts, the canonical bytes held in RAM and what the
	// numerical backend is. groups map every plan group to its modules, so the
	// backend can report per-group state.
	Open(ctx context.Context, groups map[string][]string) (Opened, error)
	// Transform builds the component in system RAM from the canonical source and
	// reports the exact bytes and a content digest of what it built.
	Transform(ctx context.Context, c ComponentIdentity) (Built, error)
	// Release drops a component from system RAM.
	Release(ctx context.Context, component string) error
	// Validate checks that the replacements can be applied in place, without
	// changing anything: module identity, shape, dtype, device, representation,
	// required buffers, execution wrapper and kernel.
	Validate(ctx context.Context, reps []Replacement) error
	// Apply replaces the representation of exactly the named groups.
	Apply(ctx context.Context, reps []Replacement) (Applied, error)
	// Reconstruct rebuilds the representation of every named group from system
	// RAM unconditionally. It is the explicit slower path, used when a delta is
	// unsupported or the observed state cannot be trusted.
	Reconstruct(ctx context.Context, reps []Replacement) (Applied, error)
	// State reads back what the model actually is: the representation of every
	// group, observed on the modules themselves, and the accelerator and host
	// memory in use.
	State(ctx context.Context) (Observed, error)
	// Close ends the session's hold on RAM and accelerator.
	Close(ctx context.Context) error
}

// Opened is what the resident worker reports at the start of a session.
type Opened struct {
	Device         string
	DType          string
	Backend        string // the numerical stack identity: material to component identity
	Modules        []Member
	CanonicalBytes int64
}

// Built is a transformed component in system RAM.
type Built struct {
	Bytes  int64
	Digest string // sha256 over the component's tensors
}

// Replacement asks for one group to take a representation: the component that
// provides it (empty for source precision, which is the canonical source) and
// the modules it covers.
type Replacement struct {
	Group     string   `json:"group"`
	Policy    string   `json:"policy"`
	Component string   `json:"component,omitempty"`
	Modules   []string `json:"modules"`
	// Transformation is the exact contract the component was built under, so the
	// backend can refuse a replacement whose representation it cannot execute.
	Transformation Transformation `json:"transformation"`
}

// Applied is the measured cost of one assembly.
type Applied struct {
	BytesToGPU    int64
	BytesReleased int64
	Modules       int
}

// Observed is the model's actual state.
type Observed struct {
	// Policies is the policy each group's modules observably realize. A group
	// whose modules are in different representations reports "mixed".
	Policies map[string]string
	// GPUAllocated is the accelerator bytes the worker's tensors occupy; HostRSS
	// the worker process's resident RAM. Nil when the worker cannot report them:
	// they are never estimated.
	GPUAllocated *int64
	HostRSS      *int64
}

// UnsupportedError says a delta cannot be applied in place for a reason that an
// explicit full reconstruction can resolve.
type UnsupportedError struct {
	Group  string
	Reason string
}

func (e *UnsupportedError) Error() string {
	if e.Group == "" {
		return "the delta cannot be replaced in place: " + e.Reason
	}
	return fmt.Sprintf("group %s cannot be replaced in place: %s", e.Group, e.Reason)
}

// IncompatibleError says a replacement would produce a representation that is
// not the requested one (shape, dtype, device, wrapper or kernel mismatch). It
// is never worked around.
type IncompatibleError struct {
	Group  string
	Reason string
}

func (e *IncompatibleError) Error() string {
	if e.Group == "" {
		return "the replacement is incompatible with the requested representation: " + e.Reason
	}
	return fmt.Sprintf("group %s is incompatible with the requested representation: %s", e.Group, e.Reason)
}

// StateLostError says the backend could not guarantee the model is either as
// asked or as it was.
type StateLostError struct{ Err error }

func (e *StateLostError) Error() string {
	return "the resident model state is no longer known: " + e.Err.Error()
}
func (e *StateLostError) Unwrap() error { return e.Err }

// ErrBroken is returned by a session that could not restore a known-valid
// state. It must be closed; its worker is torn down rather than reused.
var ErrBroken = errors.New("the tuning session could not restore a valid model state and must be closed")

// Evaluator measures the assembled trial through the real evaluation contract.
type Evaluator interface {
	Evaluate(ctx context.Context, t Assembled) (Measurement, error)
}

// Assembled is what the evaluator is told about the model it is measuring.
type Assembled struct {
	PlanSHA256 string
	Policies   map[string]string
}
