// Package requesthistory keeps a bounded, process-local projection of decide
// requests for the host dashboard. It is deliberately separate from saved
// experiment evidence: requests are never written to disk or diagnostics.
package requesthistory

import (
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	// MaxRecords bounds the number of recent request records kept in memory.
	MaxRecords = 128
	// MaxBytes bounds retained request and response bodies. Each API request is
	// already bounded by the HTTP and worker protocol limits.
	MaxBytes = 32 << 20
	// MaxMetadataStringBytes bounds caller-controlled identity and error text
	// retained alongside the bounded request and response bodies.
	MaxMetadataStringBytes = 1024
)

// Identity identifies the runtime that received one request.
type Identity struct {
	Runtime string `json:"runtime,omitempty"`
	Model   string `json:"model,omitempty"`
	Device  string `json:"device,omitempty"`
	Variant string `json:"variant,omitempty"`
}

// Summary is the request-level list projection. ItemCount describes inputs
// inside one HTTP request; queue depth and in-flight counts describe HTTP
// requests and are reported separately by View.
type Summary struct {
	ID             string     `json:"id"`
	Endpoint       string     `json:"endpoint"`
	State          string     `json:"state"`
	ReceivedAt     time.Time  `json:"received_at"`
	StartedAt      *time.Time `json:"started_at,omitempty"`
	CompletedAt    *time.Time `json:"completed_at,omitempty"`
	StateBytes     int        `json:"state_bytes"`
	TokenCount     *int       `json:"token_count,omitempty"`
	QuestionCount  int        `json:"question_count"`
	ItemCount      int        `json:"item_count"`
	ContentLimited bool       `json:"content_limited,omitempty"`
	QueueWaitMS    float64    `json:"queue_wait_ms"`
	ServiceMS      float64    `json:"service_ms"`
	TotalMS        float64    `json:"total_ms"`
	HTTPStatus     int        `json:"http_status,omitempty"`
	ErrorClass     string     `json:"error_class,omitempty"`
	ErrorMessage   string     `json:"error_message,omitempty"`
	Runtime        Identity   `json:"runtime"`
}

// Detail adds the exact decoded request and the typed response projection.
// Input and Output are JSON values so the dashboard can display them without
// changing the public decide response.
type Detail struct {
	Summary
	Input  json.RawMessage `json:"input,omitempty"`
	Output json.RawMessage `json:"output,omitempty"`
}

// View reports distinct live request metrics and the retained recent history.
type View struct {
	QueueDepth  int       `json:"queue_depth"`
	InFlight    int       `json:"in_flight"`
	ActiveItems int       `json:"active_items"`
	Entries     []Summary `json:"entries"`
}

type record struct {
	summary     Summary
	input       json.RawMessage
	output      json.RawMessage
	bytes       int
	queued      time.Time
	executingAt time.Time
	waiting     bool
	executing   bool
	retained    bool
}

// Store is a bounded in-memory request history. It does not persist request
// content under HACHIDORI_HOME or include it in diagnostic exports.
type Store struct {
	mu          sync.RWMutex
	next        uint64
	entries     map[string]*record
	order       []string // oldest first
	bytes       int
	queueDepth  int
	inFlight    int
	activeItems int
}

// Request is the lifecycle handle owned by one HTTP request handler.
type Request struct {
	store *Store
	entry *record
}

// New returns an empty bounded request history.
func New() *Store { return &Store{entries: map[string]*record{}} }

// Begin creates a request record at the time the HTTP request was received.
func (s *Store) Begin(endpoint string, received time.Time, identity Identity) *Request {
	endpoint, endpointLimited := boundedString(endpoint)
	identity, identityLimited := boundedIdentity(identity)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.next++
	id := received.UTC().Format("20060102T150405.000000000Z") + "-" + strconv.FormatUint(s.next, 10)
	for len(s.order) >= MaxRecords {
		s.evictLocked(s.order[0])
	}
	r := &record{summary: Summary{ID: id, Endpoint: endpoint, State: "received", ReceivedAt: received.UTC(), Runtime: identity,
		ContentLimited: endpointLimited || identityLimited}, retained: true}
	s.entries[id] = r
	s.order = append(s.order, id)
	s.trimLocked()
	return &Request{store: s, entry: r}
}

// SetInput stores the decoded input and its aggregate sizes. The input JSON
// is captured before it reaches the worker so detail remains inspectable if
// execution later fails.
func (r *Request) SetInput(input any, stateBytes, questions, items int) {
	data, err := json.Marshal(input)
	if err != nil {
		data = nil
	}
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	r.entry.summary.StateBytes = stateBytes
	r.entry.summary.QuestionCount = questions
	r.entry.summary.ItemCount = items
	if r.entry.retained {
		r.setInputLocked(data)
	}
}

// SetIdentity updates the execution identity when a named resident is
// selected by the request or by routing.
func (r *Request) SetIdentity(identity Identity) {
	identity, limited := boundedIdentity(identity)
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	r.entry.summary.Runtime = identity
	r.entry.summary.ContentLimited = r.entry.summary.ContentLimited || limited
}

// Admit marks a decoded and valid request as admitted before queue entry.
func (r *Request) Admit() {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	if r.entry.summary.State == "received" {
		r.setStateLocked("admitted")
	}
}

// Queued marks the request after the worker supervisor has admitted it to
// its bounded request queue.
func (r *Request) Queued(at time.Time) {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	if isTerminal(r.entry.summary.State) {
		return
	}
	if !r.entry.waiting && !r.entry.executing {
		r.entry.waiting = true
		r.entry.queued = at.UTC()
		r.store.queueDepth++
		r.store.activeItems += r.entry.summary.ItemCount
	}
	if r.entry.summary.State == "admitted" {
		r.setStateLocked("queued")
	}
}

// Started marks the point at which the worker process is about to receive the
// inference call. Repeated notifications from routed multi-stage requests do
// not reset the first start time.
func (r *Request) Started(at time.Time) {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	if isTerminal(r.entry.summary.State) {
		return
	}
	if r.entry.waiting {
		r.store.queueDepth--
		r.store.activeItems -= r.entry.summary.ItemCount
		r.entry.waiting = false
		if !r.entry.queued.IsZero() {
			r.entry.summary.QueueWaitMS += elapsedMS(r.entry.queued, at)
			r.entry.queued = time.Time{}
		}
	}
	if !r.entry.executing {
		r.entry.executing = true
		r.entry.executingAt = at.UTC()
		r.store.inFlight++
		r.store.activeItems += r.entry.summary.ItemCount
	}
	if r.entry.summary.StartedAt == nil {
		started := at.UTC()
		r.entry.summary.StartedAt = &started
		r.setStateLocked("running")
	}
}

// Finished marks the end of one worker call. Auto-routed requests may run
// more than one resident call; the request lifecycle stays running between
// calls while in-flight execution reflects only the active worker call.
func (r *Request) Finished(at time.Time) {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	if r.entry.executing {
		r.entry.summary.ServiceMS += elapsedMS(r.entry.executingAt, at)
		r.entry.executingAt = time.Time{}
		r.store.inFlight--
		r.store.activeItems -= r.entry.summary.ItemCount
		r.entry.executing = false
	}
}

// WasStarted reports whether this request crossed into worker execution.
func (r *Request) WasStarted() bool {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()
	return r.entry.summary.StartedAt != nil
}

// Complete stores a successful typed response and terminal timing.
func (r *Request) Complete(at time.Time, output any, status int) {
	data, err := json.Marshal(output)
	if err != nil {
		data = nil
	}
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	r.setStateLocked("completed")
	r.entry.summary.HTTPStatus = status
	r.finishLocked(at, data)
}

// Reject stores an input/admission denial with its typed reason.
func (r *Request) Reject(at time.Time, status int, class, message string) {
	r.finishError(at, "rejected", status, class, message)
}

// Fail stores an execution failure with its typed reason.
func (r *Request) Fail(at time.Time, status int, class, message string) {
	r.finishError(at, "failed", status, class, message)
}

// Timeout stores a worker timeout as a distinct terminal lifecycle state.
func (r *Request) Timeout(at time.Time, status int, class, message string) {
	r.finishError(at, "timed_out", status, class, message)
}

func (r *Request) finishError(at time.Time, state string, status int, class, message string) {
	class, classLimited := boundedString(class)
	message, messageLimited := boundedString(message)
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	r.setStateLocked(state)
	r.entry.summary.HTTPStatus = status
	r.entry.summary.ErrorClass = class
	r.entry.summary.ErrorMessage = message
	r.entry.summary.ContentLimited = r.entry.summary.ContentLimited || classLimited || messageLimited
	r.finishLocked(at, nil)
}

func (r *Request) finishLocked(at time.Time, output []byte) {
	r.clearActivityLocked(at)
	completed := at.UTC()
	r.entry.summary.CompletedAt = &completed
	r.entry.summary.TotalMS = elapsedMS(r.entry.summary.ReceivedAt, at)
	if r.entry.retained && len(output) > 0 {
		if r.store.makeRoomLocked(r.entry, len(output)) {
			r.entry.output = output
			r.entry.bytes += len(output)
			r.store.bytes += len(output)
		} else {
			r.entry.summary.ContentLimited = true
		}
	}
	r.store.trimLocked()
}

func (r *Request) clearActivityLocked(at time.Time) {
	if r.entry.waiting {
		r.store.queueDepth--
		r.store.activeItems -= r.entry.summary.ItemCount
		r.entry.waiting = false
	}
	if r.entry.executing {
		r.entry.summary.ServiceMS += elapsedMS(r.entry.executingAt, at)
		r.entry.executingAt = time.Time{}
		r.store.inFlight--
		r.store.activeItems -= r.entry.summary.ItemCount
		r.entry.executing = false
	}
}

func (r *Request) setInputLocked(data []byte) {
	if len(data) == 0 {
		return
	}
	if r.store.makeRoomLocked(r.entry, len(data)) {
		r.entry.input = data
		r.entry.bytes += len(data)
		r.store.bytes += len(data)
	} else {
		r.entry.summary.ContentLimited = true
	}
}

// List returns newest-first summaries and live counts. QueueDepth counts only
// waiting requests; InFlight counts executing requests; ActiveItems counts
// their semantic input items independently.
func (s *Store) List() View {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v := View{QueueDepth: s.queueDepth, InFlight: s.inFlight, ActiveItems: s.activeItems, Entries: make([]Summary, 0, len(s.order))}
	for i := len(s.order) - 1; i >= 0; i-- {
		r := s.entries[s.order[i]]
		if r == nil {
			continue
		}
		summary := r.summary
		v.Entries = append(v.Entries, summary)
	}
	return v
}

// Get returns one retained request detail.
func (s *Store) Get(id string) (Detail, bool) {
	s.mu.RLock()
	r, ok := s.entries[id]
	if !ok {
		s.mu.RUnlock()
		return Detail{}, false
	}
	summary, input, output := r.summary, r.input, r.output
	s.mu.RUnlock()
	// Payloads are immutable after assignment. Copy them after releasing the
	// store lock so a large detail read cannot hold up worker-start observers.
	d := Detail{Summary: summary, Input: append(json.RawMessage(nil), input...), Output: append(json.RawMessage(nil), output...)}
	return d, true
}

func (s *Store) trimLocked() {
	for len(s.order) > MaxRecords {
		s.evictLocked(s.order[0])
	}
}

func (s *Store) makeRoomLocked(owner *record, size int) bool {
	if size > MaxBytes {
		return false
	}
	for s.bytes+size > MaxBytes {
		removed := false
		for _, id := range s.order {
			r := s.entries[id]
			if r != nil && r != owner && isTerminal(r.summary.State) {
				s.evictLocked(id)
				removed = true
				break
			}
		}
		if !removed {
			return false
		}
	}
	return true
}

func (s *Store) evictLocked(id string) {
	r := s.entries[id]
	if r != nil {
		s.bytes -= r.bytes
		r.bytes = 0
		r.input = nil
		r.output = nil
		r.retained = false
		delete(s.entries, id)
	}
	for i, have := range s.order {
		if have == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
}

func (r *Request) setStateLocked(next string) {
	r.entry.summary.State = next
}

func isTerminal(state string) bool {
	switch state {
	case "completed", "rejected", "timed_out", "failed":
		return true
	default:
		return false
	}
}

func elapsedMS(start, end time.Time) float64 { return float64(end.Sub(start).Microseconds()) / 1000 }

func boundedIdentity(identity Identity) (Identity, bool) {
	var limited bool
	identity.Runtime, limited = boundedString(identity.Runtime)
	var fieldLimited bool
	identity.Model, fieldLimited = boundedString(identity.Model)
	limited = limited || fieldLimited
	identity.Device, fieldLimited = boundedString(identity.Device)
	limited = limited || fieldLimited
	identity.Variant, fieldLimited = boundedString(identity.Variant)
	limited = limited || fieldLimited
	return identity, limited
}

// boundedString copies retained metadata so a short prefix cannot keep an
// arbitrarily large input string alive through its backing allocation.
func boundedString(value string) (string, bool) {
	if len(value) <= MaxMetadataStringBytes {
		return strings.Clone(value), false
	}
	end := MaxMetadataStringBytes
	for end > 0 && !utf8.ValidString(value[:end]) {
		end--
	}
	return strings.Clone(value[:end]), true
}
