package trial

import (
	"errors"
	"fmt"

	"github.com/yohn-jp/hachidori/internal/eval"
)

// Evaluation modes. A trial is measured fast and ephemerally; certification of
// a finalist is a different evaluation of a different object (a clean-loaded
// immutable Variant) and is never produced by this package.
const (
	EvalModeTrial = "trial-fast"
)

// Measurement statuses.
const (
	MeasurementMeasured   = "measured"
	MeasurementNotChecked = "not_checked"
)

// QuestionBinding is the identity of one question as sent.
type QuestionBinding struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

// Measurement is what the trial evaluation actually measured on the assembled
// model, bound to the exact dataset and inputs it was measured on. Every figure
// is nil unless it was measured: nothing is estimated or carried over from
// another run.
type Measurement struct {
	Mode   string `json:"mode"`
	Status string `json:"status"`
	// Dataset binding: the same identities eval.ModelRun carries.
	DatasetSHA256 string            `json:"dataset_sha256"`
	InputSHA256   string            `json:"input_sha256"`
	SentSHA256    string            `json:"sent_sha256"`
	Questions     []QuestionBinding `json:"questions"`
	Labelled      bool              `json:"labelled"`
	// Protocol actually run.
	Cases     int `json:"cases"`
	Warmup    int `json:"warmup_requests"`
	Passes    int `json:"passes"`
	Requests  int `json:"requests"`
	Succeeded int `json:"succeeded"`
	Errors    int `json:"errors"`
	// Figures.
	Accuracy       *float64 `json:"accuracy,omitempty"`
	MacroF1        *float64 `json:"macro_f1,omitempty"`
	MeanConfidence *float64 `json:"mean_confidence,omitempty"`
	LatencyP50MS   *float64 `json:"request_latency_p50_ms,omitempty"`
	LatencyP95MS   *float64 `json:"request_latency_p95_ms,omitempty"`
	// ResidentStable says the same worker answered throughout the run.
	ResidentStable bool `json:"resident_stable"`
	// Served is what the worker reported itself to be: the evidence that the
	// measured model was the ephemeral trial of the source, not a Variant.
	Served ServedExecution `json:"served"`
	// GPUAllocatedBytes and HostRSSBytes are the worker's own readings after
	// the run, when it reported them.
	GPUAllocatedBytes *int64 `json:"gpu_allocated_bytes,omitempty"`
	HostRSSBytes      *int64 `json:"host_rss_bytes,omitempty"`
}

// ServedExecution is the execution the measured worker reported.
type ServedExecution struct {
	Model        string `json:"model"`
	Execution    string `json:"execution"`
	WorkerPID    int    `json:"worker_pid,omitempty"`
	Device       string `json:"device,omitempty"`
	DType        string `json:"dtype,omitempty"`
	VariantID    string `json:"variant_id,omitempty"`
	StatusDigest string `json:"identity_sha256,omitempty"`
}

// ExecutionTrial is the execution value a trial-mode worker reports. A Variant
// reports "variant" and the pinned source "source"; the three are therefore
// distinguishable in every status and evidence surface.
const ExecutionTrial = "trial"

// MeasurementOf projects a real resident run, produced by the same evaluation
// contract Forge and certification use, onto trial measurement. A run that was
// not stable, or that answered nothing, is not a measurement.
func MeasurementOf(run eval.ResidentRun) (Measurement, error) {
	r := run.Run
	m := Measurement{Mode: EvalModeTrial, Status: MeasurementMeasured,
		DatasetSHA256: r.DatasetSHA256, InputSHA256: r.InputSHA256, SentSHA256: r.SentSHA256, Labelled: run.Labelled,
		Cases: r.Cases, Warmup: r.WarmupRequests, Passes: r.Passes, Requests: r.Requests, Succeeded: r.Succeeded, Errors: r.ErrorCount,
		ResidentStable: r.ResidentStable}
	for _, q := range r.Questions {
		m.Questions = append(m.Questions, QuestionBinding{ID: q.ID, SHA256: q.SHA256})
	}
	if !r.ResidentStable {
		return Measurement{}, errors.New("the worker was not provably the same throughout the trial evaluation; nothing is recorded")
	}
	if r.Succeeded == 0 {
		return Measurement{}, fmt.Errorf("the trial evaluation answered none of its %d requests", r.Requests)
	}
	m.Served = servedOf(r)
	if run.Labelled {
		m.Accuracy, m.MacroF1 = r.Quality.Accuracy, r.Quality.MacroF1
	}
	m.MeanConfidence = r.Quality.MeanConfidence
	if r.RequestLatency.N > 0 {
		p50, p95 := r.RequestLatency.P50, r.RequestLatency.P95
		m.LatencyP50MS, m.LatencyP95MS = &p50, &p95
	}
	m.GPUAllocatedBytes, m.HostRSSBytes = r.Memory.PeakAlloc, r.Memory.PeakHostRSS
	return m, nil
}

func servedOf(r eval.ModelRun) ServedExecution {
	s := ServedExecution{Model: r.Model, WorkerPID: r.Identity.WorkerPID, StatusDigest: r.Identity.Digest}
	str := func(k string) string { v, _ := r.Identity.Provider[k].(string); return v }
	s.Execution, s.Device, s.DType, s.VariantID = str("execution"), str("device"), str("dtype"), str("variant_id")
	return s
}

// Validate refuses a measurement that could be mistaken for anything other than
// an ephemeral trial measurement of the source.
func (m Measurement) Validate() error {
	switch {
	case m.Mode != EvalModeTrial:
		return fmt.Errorf("measurement mode %q: only %q measurements are trial evidence", m.Mode, EvalModeTrial)
	case m.Status != MeasurementMeasured && m.Status != MeasurementNotChecked:
		return fmt.Errorf("measurement status %q is unknown", m.Status)
	case m.Status == MeasurementMeasured && (m.DatasetSHA256 == "" || m.InputSHA256 == "" || m.SentSHA256 == "" || len(m.Questions) == 0):
		return errors.New("a measured trial must bind its dataset, inputs and questions")
	case m.Status == MeasurementMeasured && !m.ResidentStable:
		return errors.New("a measured trial must come from one stable worker")
	case m.Status == MeasurementMeasured && m.Served.Execution != ExecutionTrial:
		return fmt.Errorf("the measured worker reports execution %q, not %q: it was not the ephemeral trial", m.Served.Execution, ExecutionTrial)
	case m.Served.VariantID != "":
		return fmt.Errorf("the measured worker ran variant %s; a trial measures the source model", m.Served.VariantID)
	}
	return nil
}
