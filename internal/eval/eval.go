// Package eval runs caller-side evaluation against a Hachidori endpoint.
//
// Datasets and expected labels are read locally; only state and questions
// are sent to the endpoint. Metrics are computed locally, and the report is
// versioned Decision Evidence that identifies the served runtime/model from
// endpoint status and can be replayed against the same dataset digest.
package eval

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Case is one dataset line. Expected maps question id -> expected choice and
// never leaves the caller.
type Case struct {
	ID        string            `json:"id"`
	State     string            `json:"state"`
	Questions []api.Question    `json:"questions"`
	Expected  map[string]string `json:"expected"`
}

// Load reads and validates a JSONL dataset, returning its cases and SHA-256.
func Load(path string) ([]Case, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	h := sha256.New()
	sc := bufio.NewScanner(io.TeeReader(f, h))
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	var cases []Case
	ids := map[string]bool{}
	for n := 1; sc.Scan(); n++ {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var c Case
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			return nil, "", fmt.Errorf("%s:%d: %w", path, n, err)
		}
		if c.ID == "" || ids[c.ID] {
			return nil, "", fmt.Errorf("%s:%d: case id must be present and unique", path, n)
		}
		ids[c.ID] = true
		req := c.Request()
		if err := req.Validate(); err != nil {
			return nil, "", fmt.Errorf("%s:%d (%s): %w", path, n, c.ID, err)
		}
		for _, q := range c.Questions {
			exp, ok := c.Expected[q.ID]
			if !ok {
				return nil, "", fmt.Errorf("%s:%d (%s): no expected label for question %q", path, n, c.ID, q.ID)
			}
			if !contains(q.Choices, exp) {
				return nil, "", fmt.Errorf("%s:%d (%s): expected %q is not a choice of %q", path, n, c.ID, exp, q.ID)
			}
		}
		cases = append(cases, c)
	}
	if err := sc.Err(); err != nil {
		return nil, "", err
	}
	if len(cases) == 0 {
		return nil, "", fmt.Errorf("%s: no cases", path)
	}
	return cases, hex.EncodeToString(h.Sum(nil)), nil
}

// Request is the inference-only projection of a case: no expected labels.
func (c Case) Request() api.DecideRequest {
	return api.DecideRequest{Schema: api.SchemaV1, State: c.State, Questions: c.Questions}
}

// Decider is the endpoint decide surface.
type Decider interface {
	Decide(api.DecideRequest) (api.DecideResponse, error)
}

// Endpoint is the endpoint surface eval needs: decisions plus the status
// authority that identifies what served them.
type Endpoint interface {
	Decider
	StatusSource
}

// Observation is one scored (case, question) pair of Decision Evidence.
// QuestionSHA256 is the digest of the question exactly as sent;
// QuestionDefinition identifies a reusable question definition and is
// absent for inline questions. Probabilities is the full returned
// distribution.
type Observation struct {
	CaseID             string             `json:"case_id"`
	QuestionID         string             `json:"question_id"`
	QuestionSHA256     string             `json:"question_sha256"`
	QuestionDefinition string             `json:"question_definition,omitempty"`
	Expected           string             `json:"expected"`
	Choice             string             `json:"choice"`
	Confidence         float64            `json:"confidence"`
	Probabilities      map[string]float64 `json:"probabilities"`
	Correct            bool               `json:"correct"`
	RequestMS          float64            `json:"request_ms"`
	InferenceMS        *float64           `json:"inference_ms,omitempty"`
}

// QuestionStats is per-question performance.
type QuestionStats struct {
	N              int     `json:"n"`
	Accuracy       float64 `json:"accuracy"`
	MeanConfidence float64 `json:"mean_confidence"`
	ECE            float64 `json:"ece"`
}

// Latency summarizes latencies in milliseconds.
type Latency struct {
	N    int     `json:"n"`
	P50  float64 `json:"p50_ms"`
	P95  float64 `json:"p95_ms"`
	Mean float64 `json:"mean_ms"`
}

// Report is the locally computed Decision Evidence of one run. Served is
// the identity snapshotted from endpoint status before the run and ServedEnd
// after it; ServedConsistent is false when they differ, the runtime
// restarted, or the end snapshot could not be taken, in which case the
// observations must not be attributed to a single served model.
type Report struct {
	Schema           string                   `json:"schema"`
	Endpoint         string                   `json:"endpoint"`
	Dataset          string                   `json:"dataset"`
	DatasetSHA256    string                   `json:"dataset_sha256"`
	StartedAt        string                   `json:"started_at"`
	Served           *Served                  `json:"served"`
	ServedEnd        *Served                  `json:"served_end"`
	ServedConsistent bool                     `json:"served_consistent"`
	Cases            int                      `json:"cases"`
	Observations     int                      `json:"observations"`
	Errors           []RequestError           `json:"errors"`
	ChoiceAccuracy   float64                  `json:"choice_accuracy"`
	MeanConfidence   float64                  `json:"mean_confidence"`
	ECE              float64                  `json:"ece"`
	WarmupRequests   int                      `json:"warmup_requests"`
	Passes           int                      `json:"passes"`
	RequestLatency   Latency                  `json:"request_latency"`
	ServerInference  Latency                  `json:"server_inference_latency"`
	PerQuestion      map[string]QuestionStats `json:"per_question"`
	Results          []Observation            `json:"results"`
}

// Options control a run. Warmup requests are sent first and excluded from
// latency; Passes repeats the dataset for latency (accuracy uses pass 1).
//
// QuestionDefinition, when set, returns the identity of the reusable
// question definition a case question was resolved from ("" for inline
// questions). It is the seam for caller-side question definitions; eval
// records the value and never derives one itself.
type Options struct {
	Warmup             int
	Passes             int
	QuestionDefinition func(c Case, q api.Question) string
}

// Run evaluates cases against d. It fails before sending any decide request
// when the endpoint status does not identify the served runtime and model.
func Run(d Endpoint, cases []Case, opt Options) (Report, error) {
	r := Report{Schema: EvidenceSchema, Cases: len(cases), StartedAt: time.Now().UTC().Format(time.RFC3339),
		PerQuestion: map[string]QuestionStats{}, WarmupRequests: opt.Warmup, Passes: max(opt.Passes, 1),
		Errors: []RequestError{}, Results: []Observation{}}
	start, err := Snapshot(d)
	if err != nil {
		return r, fmt.Errorf("cannot identify served runtime: %w", err)
	}
	r.Served = &start
	for i := 0; i < opt.Warmup; i++ {
		c := cases[i%len(cases)]
		if _, err := d.Decide(c.Request()); err != nil {
			cls, msg := classify(err)
			r.Errors = append(r.Errors, RequestError{Phase: "warmup", CaseID: c.ID, Class: cls, Message: msg})
		}
	}
	var lat, inf []float64
	for pass := 0; pass < r.Passes; pass++ {
		for _, c := range cases {
			t0 := time.Now()
			resp, err := d.Decide(c.Request())
			ms := float64(time.Since(t0).Microseconds()) / 1000
			if err != nil {
				cls, msg := classify(err)
				r.Errors = append(r.Errors, RequestError{Phase: "pass", Pass: pass + 1, CaseID: c.ID, Class: cls, Message: msg})
				continue
			}
			lat = append(lat, ms)
			var infMS *float64
			if resp.Timing != nil {
				inf = append(inf, resp.Timing.InferenceMS)
				v := resp.Timing.InferenceMS
				infMS = &v
			}
			if pass > 0 {
				continue
			}
			byID := map[string]api.Result{}
			for _, res := range resp.Results {
				byID[res.ID] = res
			}
			for _, q := range c.Questions {
				res, ok := byID[q.ID]
				if !ok {
					r.Errors = append(r.Errors, RequestError{Phase: "pass", Pass: 1, CaseID: c.ID, QuestionID: q.ID,
						Class: ErrClassMissingResult, Message: "response has no result for this question"})
					continue
				}
				o := Observation{CaseID: c.ID, QuestionID: q.ID, QuestionSHA256: QuestionSHA256(q),
					Expected: c.Expected[q.ID], Choice: res.Choice, Confidence: res.Confidence,
					Probabilities: res.Probabilities, Correct: res.Choice == c.Expected[q.ID],
					RequestMS: ms, InferenceMS: infMS}
				if opt.QuestionDefinition != nil {
					o.QuestionDefinition = opt.QuestionDefinition(c, q)
				}
				r.Results = append(r.Results, o)
			}
		}
	}
	if end, err := Snapshot(d); err != nil {
		r.Errors = append(r.Errors, RequestError{Phase: "status", Class: ErrClassStatus, Message: err.Error()})
	} else {
		r.ServedEnd = &end
		r.ServedConsistent = start.Compatible(&end)
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
	r.RequestLatency, r.ServerInference = summarize(lat), summarize(inf)
	return r, nil
}

func score(obs []Observation) (acc, conf, ece float64) {
	if len(obs) == 0 {
		return 0, 0, 0
	}
	c := make([]float64, len(obs))
	ok := make([]bool, len(obs))
	for i, o := range obs {
		c[i], ok[i] = o.Confidence, o.Correct
		conf += o.Confidence
		if o.Correct {
			acc++
		}
	}
	n := float64(len(obs))
	return acc / n, conf / n, ECE(c, ok, 15)
}

// ECE is expected calibration error with equal-width bins; the first bin is
// closed on the left, matching laya.common.ece_score.
func ECE(conf []float64, correct []bool, bins int) float64 {
	if len(conf) == 0 {
		return 0
	}
	n := float64(len(conf))
	e := 0.0
	for b := 0; b < bins; b++ {
		lo, hi := float64(b)/float64(bins), float64(b+1)/float64(bins)
		var cnt, sc, sa float64
		for i, c := range conf {
			in := c > lo && c <= hi
			if b == 0 {
				in = c >= lo && c <= hi
			}
			if in {
				cnt++
				sc += c
				if correct[i] {
					sa++
				}
			}
		}
		if cnt > 0 {
			e += cnt / n * math.Abs(sc/cnt-sa/cnt)
		}
	}
	return e
}

func summarize(xs []float64) Latency {
	l := Latency{N: len(xs)}
	if len(xs) == 0 {
		return l
	}
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	l.Mean = sum / float64(len(xs))
	l.P50, l.P95 = worker.Percentile(xs, 50), worker.Percentile(xs, 95)
	return l
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// Summary renders a short human-readable report.
func Summary(w io.Writer, r Report) {
	fmt.Fprintf(w, "dataset        %s (sha256 %s)\n", r.Dataset, r.DatasetSHA256[:12])
	fmt.Fprintf(w, "endpoint       %s\n", r.Endpoint)
	if r.Served != nil {
		fmt.Fprintf(w, "served         model %s (identity %s)\n", r.Served.Model(), r.Served.Digest[:12])
	}
	if !r.ServedConsistent {
		fmt.Fprintln(w, "served         INCONSISTENT: identity changed or could not be re-checked during the run")
	}
	fmt.Fprintf(w, "cases          %d  observations %d  errors %d\n", r.Cases, r.Observations, len(r.Errors))
	fmt.Fprintf(w, "choice acc     %.4f\n", r.ChoiceAccuracy)
	fmt.Fprintf(w, "mean conf      %.4f\n", r.MeanConfidence)
	fmt.Fprintf(w, "ECE (15 bins)  %.4f\n", r.ECE)
	fmt.Fprintf(w, "request p50    %.1f ms  p95 %.1f ms  (n=%d, warmup %d excluded)\n",
		r.RequestLatency.P50, r.RequestLatency.P95, r.RequestLatency.N, r.WarmupRequests)
	fmt.Fprintf(w, "inference p50  %.1f ms  p95 %.1f ms\n", r.ServerInference.P50, r.ServerInference.P95)
	ids := make([]string, 0, len(r.PerQuestion))
	for id := range r.PerQuestion {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		q := r.PerQuestion[id]
		fmt.Fprintf(w, "  %-32s n=%-4d acc=%.4f conf=%.4f ece=%.4f\n", id, q.N, q.Accuracy, q.MeanConfidence, q.ECE)
	}
	for _, e := range r.Errors {
		fmt.Fprintf(w, "error: %s\n", e.String())
	}
}
