package eval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
)

// EvidenceSchema identifies the Decision Evidence format written by eval and
// benchmark. Consumers must reject reports with a different schema.
const EvidenceSchema = "hachidori.evidence.v1"

// volatileProvider are provider status keys that are measurements of one
// worker start rather than identity; they are excluded from the snapshot.
var volatileProvider = map[string]bool{"load_ms": true, "warmup_ms": true}

// Served is the served runtime/model identity snapshotted from the
// endpoint's GET /v1/status document. It is never supplied by the caller.
//
// Runtime is the status "runtime" object and Provider the status
// "worker.provider" object, both copied verbatim (minus per-start timings),
// so identity fields added to those objects by the status authority are
// recorded without changes here. Digest covers exactly Runtime and Provider.
// UptimeS, WorkerPID and WorkerStarts are provenance only.
type Served struct {
	StatusSchema string         `json:"status_schema"`
	Runtime      map[string]any `json:"runtime"`
	Provider     map[string]any `json:"provider"`
	Digest       string         `json:"identity_sha256"`
	UptimeS      int            `json:"uptime_s"`
	WorkerPID    int            `json:"worker_pid,omitempty"`
	WorkerStarts int            `json:"worker_starts"`
}

// StatusSource is the endpoint status authority (GET /v1/status).
type StatusSource interface {
	Status() (json.RawMessage, error)
}

// Snapshot reads the served identity from the endpoint status. It fails when
// the status does not identify a ready worker with a model and provider.
func Snapshot(s StatusSource) (Served, error) {
	raw, err := s.Status()
	if err != nil {
		return Served{}, fmt.Errorf("status: %w", err)
	}
	var st struct {
		Schema  string         `json:"schema"`
		Runtime map[string]any `json:"runtime"`
		UptimeS int            `json:"uptime_s"`
		Worker  struct {
			Ready    bool           `json:"ready"`
			PID      int            `json:"pid"`
			Starts   int            `json:"starts"`
			Provider map[string]any `json:"provider"`
		} `json:"worker"`
	}
	if err := json.Unmarshal(raw, &st); err != nil {
		return Served{}, fmt.Errorf("status: %w", err)
	}
	if st.Schema != api.SchemaV1 {
		return Served{}, fmt.Errorf("status: schema %q, want %q", st.Schema, api.SchemaV1)
	}
	if m, _ := st.Runtime["model"].(string); m == "" {
		return Served{}, errors.New("status does not identify the served model")
	}
	if !st.Worker.Ready || len(st.Worker.Provider) == 0 {
		return Served{}, errors.New("status does not identify a ready provider")
	}
	prov := map[string]any{}
	for k, v := range st.Worker.Provider {
		if !volatileProvider[k] {
			prov[k] = v
		}
	}
	id, err := json.Marshal(struct {
		Runtime  map[string]any `json:"runtime"`
		Provider map[string]any `json:"provider"`
	}{st.Runtime, prov})
	if err != nil {
		return Served{}, err
	}
	sum := sha256.Sum256(id)
	return Served{StatusSchema: st.Schema, Runtime: st.Runtime, Provider: prov, Digest: hex.EncodeToString(sum[:]),
		UptimeS: st.UptimeS, WorkerPID: st.Worker.PID, WorkerStarts: st.Worker.Starts}, nil
}

// Model is the served model identity as reported by status.
func (s *Served) Model() string {
	if s == nil {
		return ""
	}
	m, _ := s.Runtime["model"].(string)
	return m
}

// Compatible reports whether end describes the same served identity as s
// with no runtime restart in between.
func (s *Served) Compatible(end *Served) bool {
	return s != nil && end != nil && s.Digest == end.Digest && end.UptimeS >= s.UptimeS
}

// QuestionSHA256 is the digest of the exact question as sent on the wire.
// It makes inline questions unambiguous across question revisions.
func QuestionSHA256(q api.Question) string {
	b, _ := json.Marshal(q)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// RequestError is one failed request or response-contract violation. It is
// never scored: semantic correctness only applies to returned results.
type RequestError struct {
	Phase      string `json:"phase"` // warmup | pass | status
	Pass       int    `json:"pass,omitempty"`
	CaseID     string `json:"case_id,omitempty"`
	QuestionID string `json:"question_id,omitempty"`
	Class      string `json:"class"`
	Message    string `json:"message"`
}

func (e RequestError) String() string {
	id := e.CaseID
	if id == "" {
		id = e.Phase
	}
	if e.QuestionID != "" {
		id += "/" + e.QuestionID
	}
	return fmt.Sprintf("%s: %s: %s", id, e.Class, e.Message)
}

// Error classes for failures that do not carry an endpoint error class.
const (
	ErrClassTransport     = "transport"
	ErrClassMissingResult = "missing_result"
	ErrClassStatus        = "status"
)

func classify(err error) (string, string) {
	var ae *client.APIError
	if errors.As(err, &ae) {
		return ae.Class, ae.Message
	}
	return ErrClassTransport, err.Error()
}
