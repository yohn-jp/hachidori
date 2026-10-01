// Package server exposes the public v1 HTTP contract on top of the supervised worker.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// DefaultListen is the default loopback-only bind address.
const DefaultListen = "127.0.0.1:7843"

// maxBody bounds request bodies: a full batch of maximal states plus questions.
const maxBody = api.MaxBatchRequests*(api.MaxStateBytes+64<<10) + 1<<20

// Decider is what the HTTP layer needs from the runtime.
type Decider interface {
	Decide(items []worker.Item) ([][]api.Result, float64, error)
	Ready() bool
	State() string
	Snapshot() worker.Snapshot
}

// Runtime describes the active runtime for status.
type Runtime struct {
	Home    string `json:"home"`
	Runtime string `json:"runtime"`
	ModelID string `json:"model_id"` // catalog model identity
	Model   string `json:"model"`    // model directory: <repo>/<revision>
	Device  string `json:"device"`
}

// Handler builds the public HTTP handler.
func Handler(d Decider, rt Runtime) http.Handler { return HandlerSince(d, rt, time.Now()) }

// HandlerSince is Handler with an explicit serving start time, so that other
// host surfaces can report the same uptime via StatusBody.
func HandlerSince(d Decider, rt Runtime, started time.Time) http.Handler {
	return hostLocal(routes(d, rt, started))
}

func routes(d Decider, rt Runtime, started time.Time) *http.ServeMux {
	mux := http.NewServeMux()
	sc := redact.New(rt.Home)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		h := api.Health{Ready: d.Ready(), State: d.State()}
		code := http.StatusOK
		if !h.Ready {
			code = http.StatusServiceUnavailable
		}
		writeJSON(w, code, h)
	})
	mux.HandleFunc("GET "+OpenAPIPath, serveOpenAPI)
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, StatusBody(d, rt, started))
	})
	mux.HandleFunc("POST /v1/decide", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		var req api.DecideRequest
		if !decode(w, r, &req) {
			return
		}
		if err := req.Validate(); err != nil {
			writeErr(w, api.ErrRequestInvalid, err.Error())
			return
		}
		res, ms, err := d.Decide([]worker.Item{{State: req.State, Questions: req.Questions}})
		if err != nil {
			writeWorkerErr(w, sc, err)
			return
		}
		writeJSON(w, http.StatusOK, api.DecideResponse{Schema: api.SchemaV1, Results: res[0],
			Timing: &api.Timing{InferenceMS: ms, TotalMS: msSince(t0)}})
	})
	mux.HandleFunc("POST /v1/decide/batch", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		var req api.BatchRequest
		if !decode(w, r, &req) {
			return
		}
		if err := req.Validate(); err != nil {
			writeErr(w, api.ErrRequestInvalid, err.Error())
			return
		}
		items := make([]worker.Item, len(req.Requests))
		for i, q := range req.Requests {
			items[i] = worker.Item{State: q.State, Questions: q.Questions}
		}
		res, ms, err := d.Decide(items)
		if err != nil {
			writeWorkerErr(w, sc, err)
			return
		}
		out := api.BatchResponse{Schema: api.SchemaV1, Timing: api.Timing{InferenceMS: ms, TotalMS: msSince(t0)}}
		for _, rs := range res {
			out.Responses = append(out.Responses, api.DecideResponse{Schema: api.SchemaV1, Results: rs})
		}
		writeJSON(w, http.StatusOK, out)
	})
	return mux
}

// hostLocal enforces the same host-local boundary as the dashboard on the
// public API: the Host header must name a loopback address (a DNS-rebinding
// page resolves its own name to 127.0.0.1 and would otherwise read
// /v1/status, which carries the HACHIDORI_HOME path), and a state-changing
// request must not come from another web origin (a page in the operator's
// browser can send a no-preflight POST to /v1/decide). Callers of this API
// are not browsers, and the API sends no CORS headers, so nothing legitimate
// is refused.
func hostLocal(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !LoopbackHost(r.Host) {
			http.Error(w, "host-local: Host must be a loopback address", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if o := r.Header.Get("Origin"); o != "" && o != "http://"+r.Host {
				http.Error(w, "cross-origin request refused", http.StatusForbidden)
				return
			}
			if s := r.Header.Get("Sec-Fetch-Site"); s != "" && s != "same-origin" && s != "none" {
				http.Error(w, "cross-site request refused", http.StatusForbidden)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// LoopbackHost reports whether a Host header value (host or host:port) names
// a loopback address. It is the one host-local policy shared by the API, the
// dashboard and the first-run screen.
func LoopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Status is the GET /v1/status document. The host dashboard renders this
// same document rather than keeping its own view of the runtime.
type Status struct {
	Schema  string          `json:"schema"`
	Runtime Runtime         `json:"runtime"`
	UptimeS int             `json:"uptime_s"`
	Worker  worker.Snapshot `json:"worker"`
}

// StatusBody builds the status document for a runtime serving since started.
func StatusBody(d Decider, rt Runtime, started time.Time) Status {
	return Status{Schema: api.SchemaV1, Runtime: rt, UptimeS: int(time.Since(started).Seconds()), Worker: d.Snapshot()}
}

// CheckLoopback refuses non-loopback binds: remote exposure needs an
// authentication model that this milestone does not implement.
func CheckLoopback(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return err
	}
	if host == "localhost" {
		return nil
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("refusing to bind %s: only loopback addresses are supported (use SSH forwarding for remote callers)", addr)
	}
	return nil
}

// WorkerConfig builds the launch configuration of the private worker from
// the active runtime and the activated catalog model, verifying the worker's
// pinned digest first. It only resolves what setup already materialized and
// activated; it never downloads anything. The worker verifies every model
// file against the manifest's digests before loading.
func WorkerConfig(h home.Home, log io.Writer) (worker.Config, Runtime, error) {
	a, rm, mm, err := h.LoadActive()
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	model, err := setup.ActiveModel(a)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	if mm.Repo != model.Repo || mm.Revision != model.Revision || (mm.ID != "" && mm.ID != model.ID) || !maps.Equal(mm.Files, model.Files) {
		return worker.Config{}, Runtime{}, fmt.Errorf("model %s: materialized manifest does not match its catalog entry (run `hachidori setup`)", model.ID)
	}
	if !rm.Spec.Provides(model.Provider) {
		return worker.Config{}, Runtime{}, fmt.Errorf("runtime %s does not carry provider %s needed by model %s (run `hachidori setup`)", a.Runtime, model.Provider, model.ID)
	}
	python := h.PythonExe(a, rm)
	if _, err := os.Stat(python); err != nil {
		return worker.Config{}, Runtime{}, fmt.Errorf("private python missing: %w", err)
	}
	script := h.WorkerScript(a)
	got, err := setup.FileSHA256(script)
	if err != nil || got != rm.Worker["worker/hachidori_worker.py"] {
		return worker.Config{}, Runtime{}, fmt.Errorf("worker script %s does not match runtime manifest", script)
	}
	modelDir := h.ModelDir(a)
	cfg := worker.Config{
		Python: python,
		Args: []string{"-I", "-X", "utf8", script, "--model-dir", modelDir, "--device", a.Device,
			"--manifest", filepath.Join(modelDir, "hachidori-model.json"), "--provider", model.Provider},
		Env:            h.Env(filepath.Dir(python), true),
		Dir:            h.Path("state"),
		Log:            log,
		StartTimeout:   10 * time.Minute,
		RequestTimeout: 2 * time.Minute,
	}
	return cfg, Runtime{Home: h.Root, Runtime: a.Runtime, ModelID: model.ID, Model: a.Model, Device: a.Device}, nil
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeErr(w, api.ErrRequestInvalid, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}

// maxErrorDetail bounds the worker-supplied detail of an error response.
const maxErrorDetail = 1024

// writeWorkerErr answers a failed request. The error class is the stable,
// public part; the detail is the worker's own text (Python exception text
// for an inference failure) passed through the shared redaction policy and
// bounded, because the caller may be a remote host behind the reverse tunnel
// and the text can carry local paths or credential-bearing URLs. The operator
// sees the unredacted text in the worker log and the dashboard.
func writeWorkerErr(w http.ResponseWriter, sc redact.Scrubber, err error) {
	var re *worker.RequestError
	if errors.As(err, &re) {
		writeErr(w, re.Class, sc.Line(re.Message, maxErrorDetail))
		return
	}
	var f *worker.Failure
	if errors.As(err, &f) {
		writeErr(w, api.ErrWorkerFailure, sc.Line(f.Class+": "+f.Message, maxErrorDetail))
		return
	}
	writeErr(w, api.ErrWorkerFailure, sc.Line(err.Error(), maxErrorDetail))
}

var statusFor = map[string]int{
	api.ErrRequestInvalid:  http.StatusBadRequest,
	api.ErrInferenceFailed: http.StatusInternalServerError,
	api.ErrNotReady:        http.StatusServiceUnavailable,
	api.ErrWorkerFailure:   http.StatusBadGateway,
	api.ErrCapacity:        http.StatusTooManyRequests,
}

func writeErr(w http.ResponseWriter, class, msg string) {
	code, ok := statusFor[class]
	if !ok {
		code = http.StatusInternalServerError
	}
	writeJSON(w, code, api.ErrorBody{Schema: api.SchemaV1, Error: api.ErrorInfo{Class: class, Message: msg}})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000 }
