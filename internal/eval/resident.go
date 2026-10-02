package eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/question"
)

// ComparisonSchema identifies the resident cross-model comparison evidence.
// It is a separate, additive document: Decision Evidence
// (hachidori.evidence.v1), history entries and every single-model eval or
// benchmark report are unchanged and stay readable exactly as before.
const ComparisonSchema = "hachidori.resident-comparison.v1"

// ErrClassServedMismatch is the class of a directly targeted request that was
// answered by a different resident than the one named, or that carried no
// served provenance. Such an answer is never scored.
const ErrClassServedMismatch = "served_mismatch"

// ServedMismatchError reports a direct request that was not provably answered
// by the targeted resident.
type ServedMismatchError struct {
	Want string
	Got  *api.Served
}

func (e *ServedMismatchError) Error() string {
	if e.Got == nil {
		return fmt.Sprintf("request targeted %q but the response carries no served provenance", e.Want)
	}
	return fmt.Sprintf("request targeted %q but was served by %q (provider %q)", e.Want, e.Got.Model, e.Got.Provider)
}

// Alignment states and codes. Alignment is about whether the runs received
// the same inputs; it never says which model is better.
const (
	AlignAligned = "aligned" // every run received identical dataset, questions and inputs
	AlignRefused = "refused" // at least one identity differs; the runs are not comparable like for like

	IncompatInputs   = "input_mismatch"
	IncompatProtocol = "protocol_mismatch"
)

// AlignmentIssue is one explicit reason the runs are not aligned.
type AlignmentIssue struct {
	Code     string `json:"code"`
	Model    string `json:"model,omitempty"`
	Question string `json:"question,omitempty"`
	Detail   string `json:"detail"`
}

// Alignment is the identity check of the runs of one comparison. When
// Status is aligned, every run used the same dataset, the same question
// identities and byte-identical normalized requests in the same order.
type Alignment struct {
	Status          string           `json:"status"`
	DatasetSHA256   string           `json:"dataset_sha256"`
	InputSHA256     string           `json:"input_sha256"`
	SentSHA256      string           `json:"sent_sha256"`
	Incompatibility []AlignmentIssue `json:"incompatibility"`
}

// QuestionRecord is the exact identity of one question as sent.
type QuestionRecord struct {
	ID         string             `json:"id"`
	SHA256     string             `json:"sha256"`
	Definition *question.Identity `json:"definition,omitempty"`
}

// Startup is cold start metadata of the resident's worker as reported by its
// status (provider load_ms and warmup_ms). It is read once, before any
// request of the comparison, and is never mixed into request latency. Null
// when the provider does not report it.
type Startup struct {
	LoadMS   *float64 `json:"load_ms"`
	WarmupMS *float64 `json:"warmup_ms"`
	Source   string   `json:"source"`
}

// MemorySample is one reading of the resident's own accelerator statistics,
// in bytes. Allocated and Reserved are the resident worker's own tensors and
// cache; Free and Total are device-wide.
type MemorySample struct {
	Phase     string `json:"phase"`
	Allocated *int64 `json:"allocated"`
	Reserved  *int64 `json:"reserved"`
	Free      *int64 `json:"free"`
	Total     *int64 `json:"total"`
	Stale     bool   `json:"stale,omitempty"`
	// HostRSS is the resident set size of the resident's own worker process
	// (host RAM), when the worker reports it. Null otherwise: never estimated.
	HostRSS *int64 `json:"host_rss_bytes,omitempty"`
}

// Memory is the accelerator memory evidence of one resident. Resident is the
// reading after warmup (before the run when warmup was not requested). Peak
// is the highest sampled value: it is a sampled observation, not a
// continuous maximum, and covers only the listed Samples. Available is false
// when the resident reports no accelerator statistics (a CPU resident).
type Memory struct {
	Available     bool           `json:"available"`
	Method        string         `json:"method"`
	Samples       []MemorySample `json:"samples"`
	ResidentAlloc *int64         `json:"resident_allocated_bytes"`
	ResidentRsrv  *int64         `json:"resident_reserved_bytes"`
	PeakAlloc     *int64         `json:"peak_allocated_bytes"`
	PeakRsrv      *int64         `json:"peak_reserved_bytes"`
	// HostRSS is the worker process RAM at the resident reading and
	// PeakHostRSS the highest sampled value; null when never reported.
	HostRSS     *int64 `json:"host_rss_bytes,omitempty"`
	PeakHostRSS *int64 `json:"peak_host_rss_bytes,omitempty"`
}

const memoryMethod = "accelerator statistics of this resident's own worker from /v1/status, sampled before the run, " +
	"after warmup, after every pass and after all runs; peak is the highest sampled value"

// ResidentObservation is one scored Decision Evidence observation with the
// provenance needed to audit it: the exact input it answered and the
// resident that answered. ServedModel and ServedProvider are what the
// response itself reported, IdentitySHA256 the resident identity digest of
// the run it belongs to.
type ResidentObservation struct {
	Observation
	InputSHA256    string `json:"input_sha256"`
	InputChars     int    `json:"input_chars"`
	InputBytes     int    `json:"input_bytes"`
	ServedModel    string `json:"served_model"`
	ServedProvider string `json:"served_provider"`
	IdentitySHA256 string `json:"identity_sha256"`
}

// ModelRun is the evidence of one resident's pass over the dataset.
//
// Identity is the resident's status identity before any request of the
// comparison and IdentityEnd after the last one. ResidentStable is true only
// when they describe the same worker (same identity, PID, start count, load
// and warmup timing, uptime not reset): evidence that the resident was not
// stopped, restarted or reloaded during the whole comparison.
//
// Requests counts the scored-pass requests (cases x passes); warmup requests
// are separate and never in the latency. SentSHA256 digests every normalized
// request sent (warmup and passes, in order), model selector excluded.
type ModelRun struct {
	Model          string  `json:"model"`
	Provider       string  `json:"provider"`
	StartedAt      string  `json:"started_at"`
	Identity       Served  `json:"identity"`
	IdentityEnd    Served  `json:"identity_end"`
	ResidentStable bool    `json:"resident_stable"`
	Startup        Startup `json:"startup"`
	Memory         Memory  `json:"memory"`

	DatasetSHA256  string           `json:"dataset_sha256"`
	InputSHA256    string           `json:"input_sha256"`
	SentSHA256     string           `json:"sent_sha256"`
	Questions      []QuestionRecord `json:"questions"`
	Cases          int              `json:"cases"`
	WarmupRequests int              `json:"warmup_requests"`
	Passes         int              `json:"passes"`

	Requests      int            `json:"requests"`
	Succeeded     int            `json:"succeeded"`
	ErrorCount    int            `json:"error_count"`
	ErrorsByClass map[string]int `json:"errors_by_class"`
	Errors        []RequestError `json:"errors"`

	Quality          Quality               `json:"quality"`
	PerQuestion      []Slice               `json:"per_question"`
	PerFamily        []Slice               `json:"per_family"`
	RequestLatency   Latency               `json:"request_latency"`
	InferenceLatency Latency               `json:"inference_latency"`
	LengthBuckets    []LengthBucket        `json:"length_buckets"`
	Observations     []ResidentObservation `json:"observations"`
}

// Evidence is the run as a plain Decision Evidence report, for the tools
// that read hachidori.evidence.v1 (Errors workspace, history, replay). It
// carries the resident identity but none of the comparison-only provenance.
func (m ModelRun) Evidence(endpoint, dataset string, defs []question.Identity) Report {
	r := Report{Schema: EvidenceSchema, Endpoint: endpoint, Dataset: dataset, DatasetSHA256: m.DatasetSHA256,
		Definitions: defs, StartedAt: m.StartedAt, Served: &m.Identity, ServedEnd: &m.IdentityEnd,
		ServedConsistent: m.ResidentStable, Cases: m.Cases, Errors: append([]RequestError{}, m.Errors...),
		WarmupRequests: m.WarmupRequests, Passes: m.Passes, RequestLatency: m.RequestLatency,
		ServerInference: m.InferenceLatency, PerQuestion: map[string]QuestionStats{}, Results: []Observation{}}
	for _, o := range m.Observations {
		r.Results = append(r.Results, o.Observation)
	}
	r.Observations = len(r.Results)
	r.ChoiceAccuracy, r.MeanConfidence, r.ECE = score(r.Results)
	groups := map[string][]Observation{}
	for _, o := range r.Results {
		groups[o.QuestionID] = append(groups[o.QuestionID], o)
	}
	for id, g := range groups {
		acc, conf, ece := score(g)
		r.PerQuestion[id] = QuestionStats{N: len(g), Accuracy: acc, MeanConfidence: conf, ECE: ece}
	}
	return r
}

// Declared are the analysis controls and protocol of the comparison. They are
// recorded so every number can be reproduced and never separated from the
// thresholds, bucket edges and families it was computed with. Models is the
// order of the passes; the passes run one resident after the other, each
// resident staying loaded throughout.
type Declared struct {
	Models         []string          `json:"models"`
	Warmup         int               `json:"warmup"`
	Passes         int               `json:"passes"`
	HighConfidence float64           `json:"high_confidence_threshold"`
	Thresholds     []float64         `json:"thresholds"`
	LengthEdges    []int             `json:"length_edges_chars"`
	Families       map[string]string `json:"families,omitempty"`
}

// ComparisonReport is the descriptive evidence of several simultaneously
// resident models evaluated on exactly the same dataset and question
// identities. It states no winner, ranking, pass or fail: it records what each
// resident did. Alignment says whether the runs are comparable like for like.
type ComparisonReport struct {
	Schema          string              `json:"schema"`
	Endpoint        string              `json:"endpoint"`
	Dataset         string              `json:"dataset"`
	DatasetSHA256   string              `json:"dataset_sha256"`
	Definitions     []question.Identity `json:"question_definitions,omitempty"`
	StartedAt       string              `json:"started_at"`
	Declared        Declared            `json:"declared"`
	Alignment       Alignment           `json:"alignment"`
	ResidentsStable bool                `json:"residents_stable"`
	Runs            []ModelRun          `json:"runs"`
}

// ResidentOptions control a resident comparison. Warmup requests are excluded
// from latency and Passes repeats the dataset for latency, as for a
// benchmark; semantic scoring uses pass 1. Models are the catalog IDs of the
// resident targets, in pass order. Zero values of the declared controls mean
// the defaults.
type ResidentOptions struct {
	Options
	Models         []string
	HighConfidence float64
	Thresholds     []float64
	LengthEdges    []int
	// Families maps a question id to a measurement family name; questions
	// without an entry are grouped as "unassigned".
	Families map[string]string
}

// UnassignedFamily groups the questions that declare no family.
const UnassignedFamily = "unassigned"

func (o ResidentOptions) declared(minModels int) (Declared, error) {
	d := Declared{Models: append([]string{}, o.Models...), Warmup: o.Warmup, Passes: max(o.Passes, 1),
		HighConfidence: o.HighConfidence, Thresholds: o.Thresholds, LengthEdges: o.LengthEdges, Families: o.Families}
	if d.HighConfidence == 0 {
		d.HighConfidence = DefaultHighConfidence
	}
	if d.Thresholds == nil {
		d.Thresholds = append([]float64{}, DefaultThresholds...)
	}
	if d.LengthEdges == nil {
		d.LengthEdges = append([]int{}, DefaultLengthEdges...)
	}
	if len(d.Models) < minModels {
		if minModels == 1 {
			return d, errors.New("a resident run needs a model")
		}
		return d, errors.New("a resident comparison needs at least two models")
	}
	seen := map[string]bool{}
	for _, m := range d.Models {
		if m == "" || seen[m] {
			return d, fmt.Errorf("models must be distinct non-empty catalog IDs, got %v", d.Models)
		}
		seen[m] = true
	}
	if d.Warmup < 0 {
		return d, errors.New("warmup must not be negative")
	}
	if math.IsNaN(d.HighConfidence) || d.HighConfidence <= 0 || d.HighConfidence > 1 {
		return d, fmt.Errorf("high-confidence threshold must be within (0, 1], got %v", d.HighConfidence)
	}
	if err := ValidateThresholds(d.Thresholds); err != nil {
		return d, err
	}
	if err := ValidateEdges(d.LengthEdges); err != nil {
		return d, err
	}
	return d, nil
}

// inputInfo is the normalized input of one case: the request exactly as sent
// to every model, without the model selector.
type inputInfo struct {
	CaseID string
	SHA256 string
	Chars  int
	Bytes  int
}

// requestSHA256 is the digest of a decide request with its model selector
// cleared: the normalized input every resident must receive identically.
func requestSHA256(r api.DecideRequest) string {
	r.Model = nil
	b, _ := json.Marshal(r)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func inputsOf(cases []Case) ([]inputInfo, string) {
	out := make([]inputInfo, len(cases))
	h := sha256.New()
	for i, c := range cases {
		out[i] = inputInfo{CaseID: c.ID, SHA256: requestSHA256(c.Request()), Chars: utf8.RuneCountInString(c.State), Bytes: len(c.State)}
		fmt.Fprintf(h, "%s\t%s\n", c.ID, out[i].SHA256)
	}
	return out, hex.EncodeToString(h.Sum(nil))
}

// directDecider targets every request at one resident and refuses any answer
// that is not provably that resident's. It never retries elsewhere.
type directDecider struct {
	d     Decider
	model string
	sent  hash.Hash
}

func (t *directDecider) Decide(req api.DecideRequest) (api.DecideResponse, error) {
	fmt.Fprintln(t.sent, requestSHA256(req))
	model := t.model
	req.Model = &model
	resp, err := t.d.Decide(req)
	if err != nil {
		return resp, err
	}
	if resp.Served == nil || resp.Served.Model != t.model {
		return api.DecideResponse{}, &ServedMismatchError{Want: t.model, Got: resp.Served}
	}
	return resp, nil
}

// residentState is one resident as described by the endpoint status.
type residentState struct {
	Model    string
	Provider string
	Served   Served
	Startup  Startup
	Accel    map[string]any
	Stale    bool
}

// readResident reads one resident's own status document from the endpoint:
// the entry of status.residents for a multi-resident runtime, or the top-level
// document of a single worker when that worker runs exactly this model. It
// fails, and never starts anything, when the resident is absent, stopped or
// not ready.
func readResident(src StatusSource, model string) (residentState, error) {
	raw, err := src.Status()
	if err != nil {
		return residentState{}, fmt.Errorf("status: %w", err)
	}
	var doc struct {
		Residents []struct {
			Model    string          `json:"model"`
			Provider string          `json:"provider"`
			Running  bool            `json:"running"`
			Status   json.RawMessage `json:"status"`
		} `json:"residents"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return residentState{}, fmt.Errorf("status: %w", err)
	}
	sub, provider := raw, ""
	if len(doc.Residents) > 0 {
		found := false
		for _, r := range doc.Residents {
			if r.Model != model {
				continue
			}
			if !r.Running {
				return residentState{}, fmt.Errorf("resident %q is not running", model)
			}
			sub, provider, found = r.Status, r.Provider, true
		}
		if !found {
			return residentState{}, fmt.Errorf("model %q is not a resident of the endpoint", model)
		}
	}
	var st struct {
		Runtime map[string]any `json:"runtime"`
		Worker  struct {
			Provider         map[string]any `json:"provider"`
			Accelerator      map[string]any `json:"accelerator"`
			AcceleratorStale bool           `json:"accelerator_stale"`
		} `json:"worker"`
	}
	if err := json.Unmarshal(sub, &st); err != nil {
		return residentState{}, fmt.Errorf("status: %w", err)
	}
	if id, _ := st.Runtime["model_id"].(string); id != model {
		return residentState{}, fmt.Errorf("status serves model %q, not %q", id, model)
	}
	served, err := Snapshot(rawStatus(sub))
	if err != nil {
		return residentState{}, fmt.Errorf("resident %q: %w", model, err)
	}
	if provider == "" {
		provider, _ = st.Worker.Provider["provider"].(string)
	}
	num := func(k string) *float64 {
		if v, ok := st.Worker.Provider[k].(float64); ok {
			return &v
		}
		return nil
	}
	return residentState{Model: model, Provider: provider, Served: served,
		Startup: Startup{LoadMS: num("load_ms"), WarmupMS: num("warmup_ms"),
			Source: "status worker.provider of the resident, read before the first request of the comparison"},
		Accel: st.Worker.Accelerator, Stale: st.Worker.AcceleratorStale}, nil
}

type rawStatus json.RawMessage

func (r rawStatus) Status() (json.RawMessage, error) { return json.RawMessage(r), nil }

func sameTiming(a, b *float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// stable reports whether two readings describe the same uninterrupted worker.
func stable(a, b residentState) bool {
	return a.Served.Compatible(&b.Served) && a.Served.WorkerPID == b.Served.WorkerPID &&
		a.Served.WorkerStarts == b.Served.WorkerStarts &&
		sameTiming(a.Startup.LoadMS, b.Startup.LoadMS) && sameTiming(a.Startup.WarmupMS, b.Startup.WarmupMS)
}

func (s residentState) memorySample(phase string) MemorySample {
	i := func(k string) *int64 {
		if v, ok := s.Accel[k].(float64); ok {
			n := int64(v)
			return &n
		}
		return nil
	}
	return MemorySample{Phase: phase, Allocated: i("memory_allocated"), Reserved: i("memory_reserved"),
		Free: i("memory_free"), Total: i("memory_total"), Stale: s.Stale, HostRSS: i("host_rss_bytes")}
}

func buildMemory(samples []MemorySample) Memory {
	m := Memory{Method: memoryMethod, Samples: samples}
	peak := func(cur *int64, v *int64) *int64 {
		if v != nil && (cur == nil || *v > *cur) {
			x := *v
			return &x
		}
		return cur
	}
	var resident *MemorySample
	for i, s := range samples {
		if s.Allocated != nil || s.Reserved != nil {
			m.Available = true
		}
		m.PeakAlloc, m.PeakRsrv = peak(m.PeakAlloc, s.Allocated), peak(m.PeakRsrv, s.Reserved)
		m.PeakHostRSS = peak(m.PeakHostRSS, s.HostRSS)
		switch {
		case s.Phase == "after_warmup":
			resident = &samples[i]
		case s.Phase == "before_run" && resident == nil:
			resident = &samples[i]
		}
	}
	if resident != nil {
		m.ResidentAlloc, m.ResidentRsrv = resident.Allocated, resident.Reserved
		m.HostRSS = resident.HostRSS
	}
	return m
}

// RunResidents evaluates every model of opt.Models, one after the other,
// against the same cases through direct resident targeting, without stopping,
// activating, restarting or reloading anything: it only sends decide requests
// naming a model and reads status. Every resident must already be running and
// ready; if one is not, nothing is evaluated.
//
// All residents are identified before the first request and again after the
// last, so a restart or reload of any resident at any time during the
// comparison is detected and marked (ResidentStable), not hidden. The report
// is returned together with ErrMisaligned when the runs turn out not to be
// aligned; the caller decides how to present it. No verdict is produced.
func RunResidents(e Endpoint, cases []Case, datasetSHA256 string, opt ResidentOptions) (ComparisonReport, error) {
	return runResidents(e, cases, datasetSHA256, opt, 2)
}

// runResidents is RunResidents for at least minModels models. A single-model
// run (minModels 1, see RunResident) is not an alignment: it has no second run
// to align with and reports no alignment status.
func runResidents(e Endpoint, cases []Case, datasetSHA256 string, opt ResidentOptions, minModels int) (ComparisonReport, error) {
	dec, err := opt.declared(minModels)
	if err != nil {
		return ComparisonReport{}, err
	}
	if len(cases) == 0 {
		return ComparisonReport{}, errors.New("no cases")
	}
	known := map[string]bool{}
	for _, c := range cases {
		for _, q := range c.Questions {
			known[q.ID] = true
		}
	}
	for id := range dec.Families {
		if !known[id] {
			return ComparisonReport{}, fmt.Errorf("family declared for question %q, which is not in the dataset", id)
		}
	}
	// Identify and check every resident before sending anything.
	before := map[string]residentState{}
	memory := map[string][]MemorySample{}
	for _, m := range dec.Models {
		st, err := readResident(e, m)
		if err != nil {
			return ComparisonReport{}, fmt.Errorf("cannot identify resident %q: %w", m, err)
		}
		before[m] = st
		memory[m] = append(memory[m], st.memorySample("before_run"))
	}
	inputs, inputSHA := inputsOf(cases)
	cmp := ComparisonReport{Schema: ComparisonSchema, DatasetSHA256: datasetSHA256, Definitions: definitions(cases),
		StartedAt: time.Now().UTC().Format(time.RFC3339), Declared: dec}
	families := func(o Observation) string {
		if f, ok := dec.Families[o.QuestionID]; ok {
			return f
		}
		return UnassignedFamily
	}
	byQuestion := func(o Observation) string { return o.QuestionID }
	type pending struct {
		run    ModelRun
		status []RequestError
	}
	var runs []pending
	for _, m := range dec.Models {
		pin := &directDecider{d: e, model: m, sent: sha256.New()}
		var statusErrs []RequestError
		hook := func(phase string) {
			st, err := readResident(e, m)
			if err != nil {
				statusErrs = append(statusErrs, RequestError{Phase: "status", Class: ErrClassStatus, Message: phase + ": " + err.Error()})
				return
			}
			memory[m] = append(memory[m], st.memorySample(phase))
		}
		r, det := runDetailed(pin, cases, Options{Warmup: dec.Warmup, Passes: dec.Passes}, hook)
		b := before[m]
		run := ModelRun{Model: m, Provider: b.Provider, StartedAt: r.StartedAt, Identity: b.Served, Startup: b.Startup,
			DatasetSHA256: datasetSHA256, InputSHA256: inputSHA, SentSHA256: hex.EncodeToString(pin.sent.Sum(nil)),
			Questions: questionRecords(cases), Cases: len(cases), WarmupRequests: dec.Warmup, Passes: dec.Passes,
			Requests: len(cases) * dec.Passes, Succeeded: len(det.samples),
			Quality:        QualityOf(r.Results, dec.HighConfidence, dec.Thresholds),
			PerQuestion:    slices(r.Results, byQuestion, dec.HighConfidence, dec.Thresholds),
			PerFamily:      slices(r.Results, families, dec.HighConfidence, dec.Thresholds),
			RequestLatency: r.RequestLatency, InferenceLatency: r.ServerInference,
			LengthBuckets: lengthBuckets(dec.LengthEdges, inputs, r, det), Observations: []ResidentObservation{}}
		for i, o := range r.Results {
			in := inputs[det.caseIdx[i]]
			sv := det.served[i] // non-nil: a direct answer without served provenance is an error, never scored
			run.Observations = append(run.Observations, ResidentObservation{Observation: o, InputSHA256: in.SHA256,
				InputChars: in.Chars, InputBytes: in.Bytes, ServedModel: sv.Model, ServedProvider: sv.Provider,
				IdentitySHA256: b.Served.Digest})
		}
		run.Errors = r.Errors
		runs = append(runs, pending{run: run, status: statusErrs})
	}
	// Identify every resident again after the last request: a restart or
	// reload anywhere in the comparison shows here.
	cmp.ResidentsStable = true
	for i, m := range dec.Models {
		run := &runs[i].run
		run.Errors = append(run.Errors, runs[i].status...)
		end, err := readResident(e, m)
		if err != nil {
			run.Errors = append(run.Errors, RequestError{Phase: "status", Class: ErrClassStatus, Message: "after_run: " + err.Error()})
			run.IdentityEnd = run.Identity
			run.ResidentStable = false
		} else {
			memory[m] = append(memory[m], end.memorySample("after_run"))
			run.IdentityEnd = end.Served
			run.ResidentStable = stable(before[m], end)
		}
		run.Memory = buildMemory(memory[m])
		run.ErrorCount, run.ErrorsByClass = len(run.Errors), map[string]int{}
		for _, re := range run.Errors {
			run.ErrorsByClass[re.Class]++
		}
		cmp.ResidentsStable = cmp.ResidentsStable && run.ResidentStable
		cmp.Runs = append(cmp.Runs, *run)
	}
	if len(cmp.Runs) < 2 {
		return cmp, nil
	}
	cmp.Alignment = Align(cmp.Runs)
	if cmp.Alignment.Status != AlignAligned {
		return cmp, ErrMisaligned
	}
	return cmp, nil
}

// ErrMisaligned is returned with a comparison whose runs did not receive the
// same dataset, question identities or inputs.
var ErrMisaligned = errors.New("runs are not aligned: dataset, question or input identity differs")

func questionRecords(cases []Case) []QuestionRecord {
	seen := map[string]bool{}
	var out []QuestionRecord
	for _, c := range cases {
		for i, q := range c.Questions {
			rec := QuestionRecord{ID: q.ID, SHA256: QuestionSHA256(q)}
			if len(c.Definitions) == len(c.Questions) {
				d := c.Definitions[i]
				rec.Definition = &d
			}
			b, _ := json.Marshal(rec)
			if !seen[string(b)] {
				seen[string(b)] = true
				out = append(out, rec)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		return out[i].SHA256 < out[j].SHA256
	})
	return out
}

// Align checks that every run received the identical dataset, protocol,
// question identities and normalized inputs as the first. It never silently
// aligns: any difference is listed and the status is refused. It is a pure
// function of the runs.
func Align(runs []ModelRun) Alignment {
	a := Alignment{Status: AlignAligned, Incompatibility: []AlignmentIssue{}}
	if len(runs) < 2 {
		a.Status = AlignRefused
		a.Incompatibility = append(a.Incompatibility, AlignmentIssue{Code: IncompatProtocol, Detail: "a comparison needs at least two runs"})
		return a
	}
	ref := runs[0]
	a.DatasetSHA256, a.InputSHA256, a.SentSHA256 = ref.DatasetSHA256, ref.InputSHA256, ref.SentSHA256
	add := func(code, model, q, detail string) {
		a.Incompatibility = append(a.Incompatibility, AlignmentIssue{Code: code, Model: model, Question: q, Detail: detail})
	}
	for _, r := range runs[1:] {
		if r.DatasetSHA256 != ref.DatasetSHA256 || r.DatasetSHA256 == "" {
			add(IncompatDataset, r.Model, "", fmt.Sprintf("dataset sha256 %q differs from %q of %s", r.DatasetSHA256, ref.DatasetSHA256, ref.Model))
		}
		if r.Cases != ref.Cases {
			add(IncompatCases, r.Model, "", fmt.Sprintf("%d cases, %d in %s", r.Cases, ref.Cases, ref.Model))
		}
		if r.WarmupRequests != ref.WarmupRequests || r.Passes != ref.Passes {
			add(IncompatProtocol, r.Model, "", fmt.Sprintf("warmup %d passes %d, %d and %d in %s", r.WarmupRequests, r.Passes,
				ref.WarmupRequests, ref.Passes, ref.Model))
		}
		if r.InputSHA256 != ref.InputSHA256 {
			add(IncompatInputs, r.Model, "", fmt.Sprintf("normalized input digest %s differs from %s of %s", r.InputSHA256, ref.InputSHA256, ref.Model))
		}
		if r.SentSHA256 != ref.SentSHA256 {
			add(IncompatInputs, r.Model, "", fmt.Sprintf("digest of the requests actually sent %s differs from %s of %s", r.SentSHA256, ref.SentSHA256, ref.Model))
		}
		ra, rb := questionIndex(ref.Questions), questionIndex(r.Questions)
		ids := map[string]bool{}
		for id := range ra {
			ids[id] = true
		}
		for id := range rb {
			ids[id] = true
		}
		sorted := make([]string, 0, len(ids))
		for id := range ids {
			sorted = append(sorted, id)
		}
		sort.Strings(sorted)
		for _, id := range sorted {
			x, inA := ra[id]
			y, inB := rb[id]
			switch {
			case !inB:
				add(IncompatOnlyInA, r.Model, id, "question is only in "+ref.Model)
			case !inA:
				add(IncompatOnlyInB, r.Model, id, "question is only in "+r.Model)
			case x != y:
				add(IncompatQuestionIdentity, r.Model, id, "same question id but a different question or Question Definition identity")
			}
		}
	}
	if len(a.Incompatibility) > 0 {
		a.Status = AlignRefused
	}
	return a
}

func questionIndex(qs []QuestionRecord) map[string]string {
	sets := map[string][]string{}
	for _, q := range qs {
		s := q.SHA256
		if q.Definition != nil {
			s += "|" + q.Definition.String() + "#" + q.Definition.Digest
		}
		sets[q.ID] = append(sets[q.ID], s)
	}
	out := map[string]string{}
	for id, s := range sets {
		sort.Strings(s)
		out[id] = strings.Join(s, "\n")
	}
	return out
}

// ValidateComparison checks the internal consistency of a comparison report
// and recomputes its alignment from the runs: a report that claims aligned
// runs whose identities differ is rejected. A report that marks itself
// refused is consistent and is returned as such; it is never aligned by this
// check.
func ValidateComparison(c ComparisonReport) error {
	if c.Schema != ComparisonSchema {
		return fmt.Errorf("comparison schema %q, want %q", c.Schema, ComparisonSchema)
	}
	if len(c.Runs) < 2 {
		return errors.New("a comparison has at least two runs")
	}
	seen := map[string]bool{}
	for i, r := range c.Runs {
		at := fmt.Sprintf("runs[%d]", i)
		switch {
		case r.Model == "" || seen[r.Model]:
			return fmt.Errorf("%s: model must be present and distinct", at)
		case i >= len(c.Declared.Models) || c.Declared.Models[i] != r.Model:
			return fmt.Errorf("%s: model %q is not the declared pass order %v", at, r.Model, c.Declared.Models)
		case r.Identity.Digest == "":
			return fmt.Errorf("%s: resident identity is missing", at)
		case r.Quality.N != len(r.Observations):
			return fmt.Errorf("%s: quality.n is %d but there are %d observations", at, r.Quality.N, len(r.Observations))
		case r.ErrorCount != len(r.Errors):
			return fmt.Errorf("%s: error_count is %d but there are %d errors", at, r.ErrorCount, len(r.Errors))
		}
		seen[r.Model] = true
		for j, o := range r.Observations {
			if o.ServedModel != r.Model || o.IdentitySHA256 != r.Identity.Digest {
				return fmt.Errorf("%s.observations[%d]: served by %q, not the run's resident %q", at, j, o.ServedModel, r.Model)
			}
			if o.InputSHA256 == "" {
				return fmt.Errorf("%s.observations[%d]: input_sha256 is missing", at, j)
			}
		}
	}
	got := Align(c.Runs)
	if got.Status != c.Alignment.Status {
		return fmt.Errorf("alignment is %q but the runs recompute to %q (%d incompatibilities)", c.Alignment.Status, got.Status, len(got.Incompatibility))
	}
	return nil
}

// DecodeComparison parses a comparison strictly: another schema, unknown
// fields, trailing data or an inconsistent report are rejected.
func DecodeComparison(data []byte) (ComparisonReport, error) {
	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return ComparisonReport{}, fmt.Errorf("not a resident comparison: %w", err)
	}
	if probe.Schema != ComparisonSchema {
		return ComparisonReport{}, fmt.Errorf("comparison schema %q, want %q", probe.Schema, ComparisonSchema)
	}
	var c ComparisonReport
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return ComparisonReport{}, fmt.Errorf("malformed %s: %w", ComparisonSchema, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return ComparisonReport{}, fmt.Errorf("malformed %s: trailing data", ComparisonSchema)
	}
	if err := ValidateComparison(c); err != nil {
		return ComparisonReport{}, fmt.Errorf("malformed %s: %w", ComparisonSchema, err)
	}
	return c, nil
}

// LoadComparison reads and decodes a comparison report file.
func LoadComparison(path string) (ComparisonReport, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return ComparisonReport{}, err
	}
	c, err := DecodeComparison(b)
	if err != nil {
		return c, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// ResidentSummary renders a short human-readable side-by-side of the runs. It
// states measurements only.
func ResidentSummary(w io.Writer, c ComparisonReport) {
	f := func(p *float64) string {
		if p == nil {
			return "n/a"
		}
		return fmt.Sprintf("%.4f", *p)
	}
	ms := func(p *float64) string {
		if p == nil {
			return "n/a"
		}
		return fmt.Sprintf("%.0f ms", *p)
	}
	mib := func(p *int64) string {
		if p == nil {
			return "n/a"
		}
		return fmt.Sprintf("%.0f MiB", float64(*p)/(1<<20))
	}
	fmt.Fprintf(w, "dataset        %s (sha256 %.12s)\n", c.Dataset, c.DatasetSHA256)
	fmt.Fprintf(w, "endpoint       %s\n", c.Endpoint)
	fmt.Fprintf(w, "alignment      %s", c.Alignment.Status)
	if c.Alignment.Status == AlignAligned {
		fmt.Fprintf(w, " (inputs sha256 %.12s)", c.Alignment.InputSHA256)
	}
	fmt.Fprintf(w, "\nresidents      stable=%v (no stop/restart/reload observed)\n", c.ResidentsStable)
	for _, i := range c.Alignment.Incompatibility {
		fmt.Fprintf(w, "  incompatible: %s %s %s: %s\n", i.Code, i.Model, i.Question, i.Detail)
	}
	for _, r := range c.Runs {
		q := r.Quality
		fmt.Fprintf(w, "\n[%s] provider %s  runtime %s  stable=%v\n", r.Model, r.Provider, r.Identity.Runtime["model"], r.ResidentStable)
		fmt.Fprintf(w, "  startup        load %s  warmup %s (cold start metadata, not request latency)\n", ms(r.Startup.LoadMS), ms(r.Startup.WarmupMS))
		fmt.Fprintf(w, "  requests       %d (ok %d, errors %d)  observations %d\n", r.Requests, r.Succeeded, r.ErrorCount, q.N)
		fmt.Fprintf(w, "  accuracy %s  macro-F1 %s  mean conf %s\n", f(q.Accuracy), f(q.MacroF1), f(q.MeanConfidence))
		fmt.Fprintf(w, "  ECE(%d) %s  Brier %s  NLL %s\n", q.Calibration.ECEBins, f(q.Calibration.ECE), f(q.Calibration.Brier), f(q.Calibration.NLL))
		fmt.Fprintf(w, "  high-confidence errors @%.2f  %d of %d high-confidence (rate of all %s)\n", q.HighConfidence.Threshold,
			q.HighConfidence.Errors, q.HighConfidence.HighConfidenceN, f(q.HighConfidence.RateOfObservations))
		fmt.Fprintf(w, "  latency        request p50 %.1f p95 %.1f ms  inference p50 %.1f p95 %.1f ms (warm, n=%d)\n",
			r.RequestLatency.P50, r.RequestLatency.P95, r.InferenceLatency.P50, r.InferenceLatency.P95, r.RequestLatency.N)
		fmt.Fprintf(w, "  memory         resident %s (reserved %s)  peak sampled %s (reserved %s)\n",
			mib(r.Memory.ResidentAlloc), mib(r.Memory.ResidentRsrv), mib(r.Memory.PeakAlloc), mib(r.Memory.PeakRsrv))
		fmt.Fprintln(w, "  coverage       threshold  covered  coverage  conditional accuracy")
		for _, t := range q.Thresholds {
			fmt.Fprintf(w, "                 %-9.2f  %-7d  %-8s  %s\n", t.Threshold, t.Covered, f(t.Coverage), f(t.ConditionalAccuracy))
		}
		fmt.Fprintln(w, "  input length   bucket              cases  req p50   req p95   inf p50   inf p95   acc")
		for _, b := range r.LengthBuckets {
			fmt.Fprintf(w, "                 %-18s  %-5d  %-8.1f  %-8.1f  %-8.1f  %-8.1f  %s\n", b.Label, b.Cases,
				b.RequestLatency.P50, b.RequestLatency.P95, b.InferenceLatency.P50, b.InferenceLatency.P95, f(b.Accuracy))
		}
		for _, s := range r.PerQuestion {
			fmt.Fprintf(w, "  question %-28s n=%-4d acc=%s ece=%s\n", s.Key, s.N, f(s.Accuracy), f(s.Calibration.ECE))
		}
		for _, e := range r.Errors {
			fmt.Fprintf(w, "  error: %s\n", e.String())
		}
	}
}
