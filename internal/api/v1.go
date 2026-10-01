// Package api defines the versioned public HTTP contract of Hachidori.
//
// The schema identifier travels in every request and response body so that
// callers and the runtime can detect incompatible peers explicitly.
package api

import (
	"fmt"
	"strings"
)

// SchemaV1 identifies the v1 decide contract.
const SchemaV1 = "hachidori.v1"

// Request limits (architecture §17).
const (
	MaxStateBytes      = 64 << 10
	MaxQuestions       = 32
	MaxChoices         = 64
	MaxBatchRequests   = 64
	MaxInstructionSize = 4 << 10
)

// Question is one typed semantic question. Only the "choice" type exists in v1.
type Question struct {
	ID           string            `json:"id"`
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Choices      []string          `json:"choices"`
	Descriptions map[string]string `json:"descriptions,omitempty"`
}

// DecideRequest asks every question against one state.
//
// Model optionally names the resident that must answer, by its stable
// Hachidori catalog model ID (never a repository or revision). Omitted, the
// request takes the default route exactly as before; named, it is served by
// that resident or fails explicitly, and is never redirected to another.
type DecideRequest struct {
	Schema    string         `json:"schema"`
	State     string         `json:"state"`
	Questions []Question     `json:"questions"`
	Options   map[string]any `json:"options,omitempty"`
	Model     string         `json:"model,omitempty"`
}

// Result is one typed observation. Confidence is the calibrated probability
// mass on the reported choice (max p).
type Result struct {
	ID            string             `json:"id"`
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

// DecideResponse carries one result per question, in request order.
//
// Served identifies the resident that answered a directly targeted request;
// it is omitted for the default route.
type DecideResponse struct {
	Schema  string   `json:"schema"`
	Results []Result `json:"results"`
	Timing  *Timing  `json:"timing,omitempty"` // omitted inside batch responses
	Served  *Served  `json:"served,omitempty"`
}

// Served is the Hachidori catalog identity of the resident that answered:
// stable provenance only, with no provider prompt or tokenization detail.
type Served struct {
	Model    string `json:"model"`
	Provider string `json:"provider"`
}

// Timing reports server-side latency. InferenceMS is time spent inside the
// resident worker; TotalMS includes queueing.
type Timing struct {
	InferenceMS float64 `json:"inference_ms"`
	TotalMS     float64 `json:"total_ms"`
}

// BatchRequest batches independent decide requests.
//
// Model optionally targets every request of the batch at one resident (see
// DecideRequest.Model). A batch is served by one resident, so a request that
// names a different model is invalid.
type BatchRequest struct {
	Schema   string          `json:"schema"`
	Requests []DecideRequest `json:"requests"`
	Model    string          `json:"model,omitempty"`
}

// BatchResponse is aligned with BatchRequest.Requests by index.
type BatchResponse struct {
	Schema    string           `json:"schema"`
	Responses []DecideResponse `json:"responses"`
	Timing    Timing           `json:"timing"`
	Served    *Served          `json:"served,omitempty"`
}

// Error classes (architecture §15).
const (
	ErrRequestInvalid  = "request_invalid"
	ErrInferenceFailed = "inference_failed"
	ErrNotReady        = "not_ready"
	ErrWorkerFailure   = "worker_failure"
	ErrCapacity        = "capacity"
)

// ErrorBody is the structured error envelope.
type ErrorBody struct {
	Schema string    `json:"schema"`
	Error  ErrorInfo `json:"error"`
}

// ErrorInfo describes one failure.
type ErrorInfo struct {
	Class   string `json:"class"`
	Message string `json:"message"`
}

// Health is the minimal /health payload.
type Health struct {
	Ready bool   `json:"ready"`
	State string `json:"state"`
}

// Validate checks a decide request against the v1 contract.
func (r *DecideRequest) Validate() error {
	if r.Schema != SchemaV1 {
		return fmt.Errorf("schema must be %q, got %q", SchemaV1, r.Schema)
	}
	if len(r.State) > MaxStateBytes {
		return fmt.Errorf("state exceeds %d bytes", MaxStateBytes)
	}
	if strings.TrimSpace(r.State) == "" {
		return fmt.Errorf("state must not be empty")
	}
	if err := validModelRef(r.Model); err != nil {
		return err
	}
	if len(r.Questions) == 0 || len(r.Questions) > MaxQuestions {
		return fmt.Errorf("questions must contain 1..%d entries", MaxQuestions)
	}
	seen := map[string]bool{}
	for i, q := range r.Questions {
		if q.ID == "" {
			return fmt.Errorf("questions[%d]: id is required", i)
		}
		if seen[q.ID] {
			return fmt.Errorf("questions[%d]: duplicate id %q", i, q.ID)
		}
		seen[q.ID] = true
		if err := q.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// Validate checks one question's type, instructions, choices and
// descriptions against the v1 contract. The id must be checked by the caller
// (presence and uniqueness are properties of the enclosing request).
func (q Question) Validate() error {
	if q.Type != "choice" {
		return fmt.Errorf("question %q: type must be \"choice\"", q.ID)
	}
	if strings.TrimSpace(q.Instructions) == "" || len(q.Instructions) > MaxInstructionSize {
		return fmt.Errorf("question %q: instructions must be 1..%d bytes", q.ID, MaxInstructionSize)
	}
	if len(q.Choices) < 2 || len(q.Choices) > MaxChoices {
		return fmt.Errorf("question %q: choices must contain 2..%d labels", q.ID, MaxChoices)
	}
	labels := map[string]bool{}
	for _, c := range q.Choices {
		if c == "" || labels[c] {
			return fmt.Errorf("question %q: choice labels must be non-empty and unique", q.ID)
		}
		labels[c] = true
	}
	for k := range q.Descriptions {
		if !labels[k] {
			return fmt.Errorf("question %q: description for unknown choice %q", q.ID, k)
		}
	}
	return nil
}

// Validate checks a batch request against the v1 contract.
func (b *BatchRequest) Validate() error {
	if b.Schema != SchemaV1 {
		return fmt.Errorf("schema must be %q, got %q", SchemaV1, b.Schema)
	}
	if len(b.Requests) == 0 || len(b.Requests) > MaxBatchRequests {
		return fmt.Errorf("requests must contain 1..%d entries", MaxBatchRequests)
	}
	if err := validModelRef(b.Model); err != nil {
		return err
	}
	for i := range b.Requests {
		if b.Requests[i].Schema == "" {
			b.Requests[i].Schema = SchemaV1
		}
		if err := b.Requests[i].Validate(); err != nil {
			return fmt.Errorf("requests[%d]: %w", i, err)
		}
	}
	_, err := b.Target()
	return err
}

// Target is the model a batch is directed at: its own Model, or the one
// every request names, "" for the default route. Requests naming different
// models are invalid: one batch is served by one resident.
func (b *BatchRequest) Target() (string, error) {
	target := b.Model
	for i, r := range b.Requests {
		switch {
		case r.Model == "" || r.Model == target:
		case target == "":
			target = r.Model
		default:
			return "", fmt.Errorf("requests[%d]: model %q differs from model %q of the same batch", i, r.Model, target)
		}
	}
	return target, nil
}

// maxModelRef bounds a model reference; catalog IDs are short.
const maxModelRef = 128

// validModelRef checks the shape of a model selector: a stable catalog
// identity token. Whether it names a resident is the runtime's decision.
func validModelRef(m string) error {
	if len(m) > maxModelRef {
		return fmt.Errorf("model must be at most %d bytes", maxModelRef)
	}
	for _, c := range m {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return fmt.Errorf("model must be a catalog model ID (letters, digits, '-', '_', '.')")
		}
	}
	return nil
}
