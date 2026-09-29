// Package eval runs caller-side evaluation against a Hachidori endpoint.
//
// Datasets and expected labels are read locally; only state and questions
// are sent to the endpoint. Question Definition references are resolved and
// compiled locally before any request. Metrics are computed locally.
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
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Case is one dataset line. Expected maps question id -> expected choice and
// never leaves the caller.
//
// A case carries either inline Questions or QuestionRefs to caller-side
// Question Definitions, not both. Load resolves QuestionRefs into Questions
// (compiled api.Question) and records the resolved identities in
// Definitions; neither the references nor the identities are sent.
type Case struct {
	ID           string              `json:"id"`
	State        string              `json:"state"`
	Questions    []api.Question      `json:"questions,omitempty"`
	QuestionRefs []question.Ref      `json:"question_refs,omitempty"`
	Expected     map[string]string   `json:"expected"`
	Definitions  []question.Identity `json:"-"`
}

// Load reads and validates a JSONL dataset, returning its cases and SHA-256.
// defs resolves question_refs and may be nil for inline-only datasets. Every
// reference is resolved before Load returns, so unresolved, duplicate or
// incompatible definitions fail before any inference. Within one dataset a
// question id must always mean the same thing: one definition version, or
// inline questions only.
func Load(path string, defs *question.Set) ([]Case, string, error) {
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
	bound := map[string]string{} // question id -> "inline" or definition id@version
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
		if err := resolve(&c, defs, bound); err != nil {
			return nil, "", fmt.Errorf("%s:%d (%s): %w", path, n, c.ID, err)
		}
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

// resolve compiles c.QuestionRefs into c.Questions and checks that every
// question id is bound to a single source across the dataset.
func resolve(c *Case, defs *question.Set, bound map[string]string) error {
	if len(c.QuestionRefs) > 0 {
		if len(c.Questions) > 0 {
			return fmt.Errorf("questions and question_refs are mutually exclusive")
		}
		seen := map[string]bool{}
		for _, r := range c.QuestionRefs {
			if seen[r.ID] {
				return fmt.Errorf("duplicate question reference %q", r.ID)
			}
			seen[r.ID] = true
			d, err := defs.Resolve(r)
			if err != nil {
				return err
			}
			c.Questions = append(c.Questions, d.Compile())
			c.Definitions = append(c.Definitions, d.Identity())
		}
		c.QuestionRefs = nil
	}
	for i, q := range c.Questions {
		src := "inline"
		if c.Definitions != nil {
			src = c.Definitions[i].String()
		}
		if prev, ok := bound[q.ID]; ok && prev != src {
			return fmt.Errorf("question %q is %s here but %s in an earlier case", q.ID, src, prev)
		}
		bound[q.ID] = src
	}
	return nil
}

// Request is the inference-only projection of a case: no expected labels,
// no definition references or identities.
func (c Case) Request() api.DecideRequest {
	return api.DecideRequest{Schema: api.SchemaV1, State: c.State, Questions: c.Questions}
}

// Decider is the endpoint surface eval needs.
type Decider interface {
	Decide(api.DecideRequest) (api.DecideResponse, error)
}

// Observation is one scored (case, question) pair.
type Observation struct {
	CaseID     string  `json:"case_id"`
	QuestionID string  `json:"question_id"`
	Expected   string  `json:"expected"`
	Choice     string  `json:"choice"`
	Confidence float64 `json:"confidence"`
	Correct    bool    `json:"correct"`
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

// Report is the locally computed evaluation evidence.
type Report struct {
	Endpoint        string                   `json:"endpoint"`
	Dataset         string                   `json:"dataset"`
	DatasetSHA256   string                   `json:"dataset_sha256"`
	Definitions     []question.Identity      `json:"question_definitions,omitempty"`
	StartedAt       string                   `json:"started_at"`
	Cases           int                      `json:"cases"`
	Observations    int                      `json:"observations"`
	Errors          []string                 `json:"errors"`
	ChoiceAccuracy  float64                  `json:"choice_accuracy"`
	MeanConfidence  float64                  `json:"mean_confidence"`
	ECE             float64                  `json:"ece"`
	WarmupRequests  int                      `json:"warmup_requests"`
	Passes          int                      `json:"passes"`
	RequestLatency  Latency                  `json:"request_latency"`
	ServerInference Latency                  `json:"server_inference_latency"`
	PerQuestion     map[string]QuestionStats `json:"per_question"`
	Results         []Observation            `json:"results"`
}

// Options control a run. Warmup requests are sent first and excluded from
// latency; Passes repeats the dataset for latency (accuracy uses pass 1).
type Options struct {
	Warmup int
	Passes int
}

// Run evaluates cases against d.
func Run(d Decider, cases []Case, opt Options) Report {
	r := Report{Cases: len(cases), StartedAt: time.Now().UTC().Format(time.RFC3339), PerQuestion: map[string]QuestionStats{},
		WarmupRequests: opt.Warmup, Passes: max(opt.Passes, 1), Errors: []string{}, Definitions: definitions(cases)}
	for i := 0; i < opt.Warmup; i++ {
		if _, err := d.Decide(cases[i%len(cases)].Request()); err != nil {
			r.Errors = append(r.Errors, fmt.Sprintf("warmup: %v", err))
		}
	}
	var lat, inf []float64
	for pass := 0; pass < r.Passes; pass++ {
		for _, c := range cases {
			t0 := time.Now()
			resp, err := d.Decide(c.Request())
			ms := float64(time.Since(t0).Microseconds()) / 1000
			if err != nil {
				r.Errors = append(r.Errors, fmt.Sprintf("%s: %v", c.ID, err))
				continue
			}
			lat = append(lat, ms)
			if resp.Timing != nil {
				inf = append(inf, resp.Timing.InferenceMS)
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
					r.Errors = append(r.Errors, fmt.Sprintf("%s: missing result for %s", c.ID, q.ID))
					continue
				}
				r.Results = append(r.Results, Observation{CaseID: c.ID, QuestionID: q.ID, Expected: c.Expected[q.ID],
					Choice: res.Choice, Confidence: res.Confidence, Correct: res.Choice == c.Expected[q.ID]})
			}
		}
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
	return r
}

// definitions lists the distinct resolved definitions used by cases, sorted
// by question id.
func definitions(cases []Case) []question.Identity {
	seen := map[question.Identity]bool{}
	var out []question.Identity
	for _, c := range cases {
		for _, d := range c.Definitions {
			if !seen[d] {
				seen[d] = true
				out = append(out, d)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
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
	for _, d := range r.Definitions {
		fmt.Fprintf(w, "definition     %s (%s)\n", d, d.Digest)
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
		fmt.Fprintf(w, "error: %s\n", e)
	}
}
