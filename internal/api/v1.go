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
// Hachidori catalog model ID (never a repository or revision). Omitted (nil),
// the request takes the default route exactly as before; named, it is served
// by that resident or fails explicitly, and is never redirected to another.
// An explicitly empty selector is invalid input, not an omitted one.
//
// Route optionally selects deterministic policy routing ("auto") instead of
// the default route. It cannot be combined with Model: a direct request is
// strict and a routed request is answered per the runtime's routing policy.
type DecideRequest struct {
	Schema    string         `json:"schema"`
	State     string         `json:"state"`
	Questions []Question     `json:"questions"`
	Options   map[string]any `json:"options,omitempty"`
	Model     *string        `json:"model,omitempty"`
	Route     string         `json:"route,omitempty"`
}

// RouteAuto selects the runtime's deterministic routing policy: each question
// is answered by its first-path resident and handed off to the alternate
// resident only when the policy requires it.
const RouteAuto = "auto"

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
// it is omitted for the default route. Routing is present only for a routed
// request and says which resident produced each final result and why.
type DecideResponse struct {
	Schema  string   `json:"schema"`
	Results []Result `json:"results"`
	Timing  *Timing  `json:"timing,omitempty"` // omitted inside batch responses
	Served  *Served  `json:"served,omitempty"`
	Routing *Routing `json:"routing,omitempty"`
}

// Stable routing reason codes (RoutedResult.Reason). They are part of the
// public contract; policy changes never rename them.
const (
	ReasonFirstPathOnly    = "first_path_only"                // the policy defines no handoff for the question
	ReasonFirstPathKept    = "first_path_kept"                // a handoff is defined and its condition was not met
	ReasonHandoffAlways    = "handoff_always"                 // the policy hands this question off unconditionally
	ReasonHandoffChoice    = "handoff_choice"                 // the first-path choice is one the policy hands off
	ReasonHandoffLowConf   = "handoff_low_confidence"         // first-path confidence is below the policy threshold
	ReasonHandoffLowMargin = "handoff_low_margin"             // first-path top-two margin is below the policy threshold
	ReasonHandoffFailed    = "handoff_failed_kept_first_path" // an optional handoff failed; the first-path result is kept
)

// Routing is the provenance of a routed request. Policy identifies the exact
// policy that decided; Results has one entry per question in question order.
// Providers reports each resident's latency contribution to this request and
// is present on a /v1/decide response and on a batch response, not on the
// entries of a batch.
type Routing struct {
	Mode      string           `json:"mode"`
	Policy    PolicyRef        `json:"policy"`
	Results   []RoutedResult   `json:"results,omitempty"`
	Handoffs  int              `json:"handoffs"`
	Providers []ProviderTiming `json:"providers,omitempty"`
}

// PolicyRef names a routing policy by its id and canonical content digest.
type PolicyRef struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

// RoutedResult is the routing provenance of one final result. Served is the
// resident that produced the final result. FirstPath is present only when the
// result was handed off and records the replaced first-path observation.
type RoutedResult struct {
	ID        string     `json:"id"`
	Served    Served     `json:"served"`
	Reason    string     `json:"reason"`
	Profile   string     `json:"profile,omitempty"`
	FirstPath *FirstPath `json:"first_path,omitempty"`
}

// FirstPath is the first-path observation that a handoff replaced.
type FirstPath struct {
	Model      string  `json:"model"`
	Provider   string  `json:"provider"`
	Choice     string  `json:"choice"`
	Confidence float64 `json:"confidence"`
}

// ProviderTiming is one resident's contribution to a routed request: the
// worker calls it served, the questions in them and their inference time.
type ProviderTiming struct {
	Model       string  `json:"model"`
	Provider    string  `json:"provider"`
	Calls       int     `json:"calls"`
	Questions   int     `json:"questions"`
	InferenceMS float64 `json:"inference_ms"`
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
//
// Route optionally routes every request of the batch by policy (see
// DecideRequest.Route); a request that names a different route is invalid.
type BatchRequest struct {
	Schema   string          `json:"schema"`
	Requests []DecideRequest `json:"requests"`
	Model    *string         `json:"model,omitempty"`
	Route    string          `json:"route,omitempty"`
}

// BatchResponse is aligned with BatchRequest.Requests by index.
type BatchResponse struct {
	Schema    string           `json:"schema"`
	Responses []DecideResponse `json:"responses"`
	Timing    Timing           `json:"timing"`
	Served    *Served          `json:"served,omitempty"`
	Routing   *Routing         `json:"routing,omitempty"`
}

// Error classes (architecture §15).
const (
	ErrRequestInvalid  = "request_invalid"
	ErrInferenceFailed = "inference_failed"
	ErrNotReady        = "not_ready"
	ErrWorkerFailure   = "worker_failure"
	ErrCapacity        = "capacity"
	// ErrRoutingFailed is a routed request that could not be answered by its
	// policy: the first-path resident or a required handoff target was
	// unavailable or failed, or no routing policy is configured. It never
	// carries results; a weaker result is never substituted.
	ErrRoutingFailed = "routing_failed"
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
	if err := validRoute(r.Route, ModelRef(r.Model)); err != nil {
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
	if err := validRoute(b.Route, ModelRef(b.Model)); err != nil {
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
	target, err := b.Target()
	if err != nil {
		return err
	}
	route, err := b.RouteMode()
	if err != nil {
		return err
	}
	return validRoute(route, target)
}

// RouteMode is the routing a batch asks for: its own Route, or the one every
// request names, "" for the default route. Requests naming different routes
// are invalid: one batch is routed one way.
func (b *BatchRequest) RouteMode() (string, error) {
	mode := b.Route
	for i, r := range b.Requests {
		switch {
		case r.Route == "" || r.Route == mode:
		case mode == "":
			mode = r.Route
		default:
			return "", fmt.Errorf("requests[%d]: route %q differs from route %q of the same batch", i, r.Route, mode)
		}
	}
	return mode, nil
}

// Target is the model a batch is directed at: its own Model, or the one
// every request names, "" for the default route. Requests naming different
// models are invalid: one batch is served by one resident.
func (b *BatchRequest) Target() (string, error) {
	target := ModelRef(b.Model)
	for i, r := range b.Requests {
		switch m := ModelRef(r.Model); {
		case m == "" || m == target:
		case target == "":
			target = m
		default:
			return "", fmt.Errorf("requests[%d]: model %q differs from model %q of the same batch", i, m, target)
		}
	}
	return target, nil
}

// ModelRef is the value of an optional model selector, "" when omitted.
func ModelRef(m *string) string {
	if m == nil {
		return ""
	}
	return *m
}

// validRoute checks a routing selector: absent or "auto", and never together
// with a direct model, because a direct request is strict.
func validRoute(route, model string) error {
	switch route {
	case "":
		return nil
	case RouteAuto:
		if model != "" {
			return fmt.Errorf("route %q cannot be combined with model %q: a direct request is answered by that resident only", route, model)
		}
		return nil
	}
	return fmt.Errorf("route must be %q when present, got %q", RouteAuto, route)
}

// maxModelRef bounds a model reference; catalog IDs are short.
const maxModelRef = 128

// validModelRef checks the shape of an optional model selector: omitted (nil),
// or a non-empty stable catalog identity token. An explicitly empty selector
// is invalid, never an omitted one. Whether it names a resident is the
// runtime's decision.
func validModelRef(sel *string) error {
	if sel == nil {
		return nil
	}
	m := *sel
	if m == "" {
		return fmt.Errorf("model must not be empty; omit it to use the default resident")
	}
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
