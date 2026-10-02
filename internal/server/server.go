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
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/route"
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

// Router is implemented by a Decider that keeps several residents. DecideOn
// serves items on exactly the named resident (a stable catalog model ID) and
// nothing else: a model that is not resident is a request_invalid error, one
// that is not ready is not_ready for itself, and neither is ever answered by
// another resident. Identity reports the catalog identity of a resident.
type Router interface {
	DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error)
	Identity(model string) (api.Served, bool)
}

// AutoRouting is implemented by a Decider that can route by policy. A nil
// router means no routing policy is configured, and a routed request is then
// refused with routing_failed; it is never answered by the default resident.
type AutoRouting interface {
	AutoRouter() *route.Router
}

// RoutingReporter is implemented by a Decider that applies a routing policy;
// the status document then carries its counters.
type RoutingReporter interface {
	RoutingStatus() *route.Status
}

// Residents is implemented by a Decider that can report every resident; the
// status document then lists all of them.
type Residents interface {
	ResidentStatuses() []ResidentStatus
}

// Runtime describes the active runtime for status.
//
// ModelID is always the semantic source model identity. When the resident
// executes a derived variant, Variant says which one and how it is quantized;
// it is execution provenance, never a different model identity, and it is
// absent for the upstream source artifact.
type Runtime struct {
	Home    string   `json:"home"`
	Runtime string   `json:"runtime"`
	ModelID string   `json:"model_id"` // catalog model identity
	Model   string   `json:"model"`    // model directory: <repo>/<revision>
	Device  string   `json:"device"`
	Variant *Variant `json:"variant,omitempty"`
}

// Variant is the identity of the variant a resident executes, from its
// manifest, with the certification state the activation was admitted under.
// The device and dtype the worker actually reports are in worker.provider.
type Variant struct {
	ID             string `json:"id"`
	Recipe         string `json:"recipe"`
	Scheme         string `json:"scheme"`
	Bits           int    `json:"bits"`
	DType          string `json:"dtype"` // compute dtype of everything not quantized
	Format         string `json:"format"`
	Engine         string `json:"engine"`
	EngineVersion  string `json:"engine_version"`
	ManifestSHA256 string `json:"manifest_sha256"`
	// Certification is accepted, or experimental/uncertified for a variant
	// launched on an explicit operator request without an accepted record.
	Certification string        `json:"certification"`
	Source        VariantSource `json:"source"`
}

// VariantSource is the immutable source identity of a variant.
type VariantSource struct {
	ID       string `json:"id"`
	Provider string `json:"provider"`
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
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
		items := []worker.Item{{State: req.State, Questions: req.Questions}}
		if req.Route == api.RouteAuto {
			out, rtr, err := decideRouted(d, items)
			if err != nil {
				writeWorkerErr(w, sc, err)
				return
			}
			writeJSON(w, http.StatusOK, api.DecideResponse{Schema: api.SchemaV1, Results: out.Items[0].Results,
				Timing:  &api.Timing{InferenceMS: out.InferenceMS, TotalMS: msSince(t0)},
				Routing: rtr.Routing(out.Items[0], out.Providers)})
			return
		}
		res, ms, served, err := decideTargeted(d, rt, api.ModelRef(req.Model), items)
		if err != nil {
			writeWorkerErr(w, sc, err)
			return
		}
		writeJSON(w, http.StatusOK, api.DecideResponse{Schema: api.SchemaV1, Results: res[0],
			Timing: &api.Timing{InferenceMS: ms, TotalMS: msSince(t0)}, Served: served})
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
		if mode, _ := req.RouteMode(); mode == api.RouteAuto { // validated above
			out, rtr, err := decideRouted(d, items)
			if err != nil {
				writeWorkerErr(w, sc, err)
				return
			}
			resp := api.BatchResponse{Schema: api.SchemaV1, Timing: api.Timing{InferenceMS: out.InferenceMS, TotalMS: msSince(t0)},
				Routing: &api.Routing{Mode: api.RouteAuto, Policy: rtr.Ref(), Handoffs: out.Handoffs, Providers: out.Providers}}
			for _, it := range out.Items {
				resp.Responses = append(resp.Responses, api.DecideResponse{Schema: api.SchemaV1, Results: it.Results, Routing: rtr.Routing(it, nil)})
			}
			writeJSON(w, http.StatusOK, resp)
			return
		}
		target, _ := req.Target() // validated above
		res, ms, served, err := decideTargeted(d, rt, target, items)
		if err != nil {
			writeWorkerErr(w, sc, err)
			return
		}
		out := api.BatchResponse{Schema: api.SchemaV1, Timing: api.Timing{InferenceMS: ms, TotalMS: msSince(t0)}, Served: served}
		for _, rs := range res {
			out.Responses = append(out.Responses, api.DecideResponse{Schema: api.SchemaV1, Results: rs})
		}
		writeJSON(w, http.StatusOK, out)
	})
	return mux
}

// decideRouted routes items by the runtime's routing policy. A runtime
// without a policy refuses the request; it never falls back to the default
// resident.
func decideRouted(d Decider, items []worker.Item) (route.Outcome, *route.Router, error) {
	var rtr *route.Router
	if a, ok := d.(AutoRouting); ok {
		rtr = a.AutoRouter()
	}
	if rtr == nil {
		return route.Outcome{}, nil, &worker.RequestError{Class: api.ErrRoutingFailed,
			Message: route.CodeNoPolicy + ": no routing policy is configured for this runtime (serve with --resident and --routing-policy); the request was not answered by any resident"}
	}
	out, err := rtr.Decide(items)
	return out, rtr, err
}

// decideTargeted runs items on the resident the caller named, or on the
// default route when model is empty (the unchanged compatibility path, which
// reports no provenance). A named model is served by that resident only: a
// Router answers for its own members; a single-worker Decider answers only
// for the model it serves. Anything else is a request_invalid error, never a
// different resident.
func decideTargeted(d Decider, rt Runtime, model string, items []worker.Item) ([][]api.Result, float64, *api.Served, error) {
	if model == "" {
		res, ms, err := d.Decide(items)
		return res, ms, nil, err
	}
	if r, ok := d.(Router); ok {
		served, known := r.Identity(model)
		if !known {
			return nil, 0, nil, &worker.RequestError{Class: api.ErrRequestInvalid, Message: "model " + model + " is not resident"}
		}
		res, ms, err := r.DecideOn(model, items)
		if err != nil {
			return nil, 0, nil, err
		}
		return res, ms, &served, nil
	}
	if model != rt.ModelID {
		return nil, 0, nil, &worker.RequestError{Class: api.ErrRequestInvalid, Message: "model " + model + " is not resident"}
	}
	served := api.Served{Model: rt.ModelID}
	if m, err := setup.LookupModel(rt.ModelID); err == nil {
		served.Provider = m.Provider
	}
	res, ms, err := d.Decide(items)
	if err != nil {
		return nil, 0, nil, err
	}
	return res, ms, &served, nil
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
//
// Runtime and Worker are the default resident's (the compatibility route,
// unchanged). Residents lists every resident of a multi-resident runtime,
// default first, each with its own runtime identity and worker snapshot
// (state, PID, device, dtype, load and warmup timing, counters, queue,
// latency and accelerator evidence); it is absent for a single worker.
// Routing is the deterministic routing policy and its counters (per reason
// code and per resident); it is absent when no policy is configured.
type Status struct {
	Schema    string           `json:"schema"`
	Runtime   Runtime          `json:"runtime"`
	UptimeS   int              `json:"uptime_s"`
	Worker    worker.Snapshot  `json:"worker"`
	Residents []ResidentStatus `json:"residents,omitempty"`
	Routing   *route.Status    `json:"routing,omitempty"`
}

// ResidentStatus is one resident's view in Status.Residents: the status of
// that resident's own supervisor, under its stable catalog identity.
type ResidentStatus struct {
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Default  bool   `json:"default,omitempty"`
	Running  bool   `json:"running"`
	Status   Status `json:"status"`
}

// StatusBody builds the status document for a runtime serving since started.
func StatusBody(d Decider, rt Runtime, started time.Time) Status {
	st := Status{Schema: api.SchemaV1, Runtime: rt, UptimeS: int(time.Since(started).Seconds()), Worker: d.Snapshot()}
	if r, ok := d.(Residents); ok {
		st.Residents = r.ResidentStatuses()
	}
	if r, ok := d.(RoutingReporter); ok {
		st.Routing = r.RoutingStatus()
	}
	return st
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
	return workerConfig(h, a, rm, mm, log)
}

// ResidentConfig is WorkerConfig for one resident of a multi-resident set:
// the catalog model modelID served by the active runtime on the active
// device. The empty ID and the active model's own ID are the default
// resident, identical to WorkerConfig. Any other ID must be a catalog model
// that is already materialized in the home; the activation record is only
// read, never changed, and nothing is downloaded. The requested device is the
// activation record's for every resident: there is no per-resident device and
// no fallback to another one.
func ResidentConfig(h home.Home, modelID string, log io.Writer) (worker.Config, Runtime, error) {
	a, rm, mm, err := h.LoadActive()
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	active, err := setup.ActiveModel(a)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	if modelID == "" || modelID == active.ID {
		return workerConfig(h, a, rm, mm, log)
	}
	model, err := setup.LookupModel(modelID)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	ra := home.Active{Runtime: a.Runtime, ModelID: model.ID, Model: setup.ModelDirName(model), Device: a.Device}
	var rmm home.ModelManifest
	if err := home.ReadJSON(filepath.Join(h.ModelDir(ra), "hachidori-model.json"), &rmm); err != nil {
		return worker.Config{}, Runtime{}, fmt.Errorf("model %s is not materialized in this home (materialize it first): %w", model.ID, err)
	}
	return workerConfig(h, ra, rm, rmm, log)
}

// EnvOpenDeciderDType is the explicit evaluation control for the dtype of an
// OpenDecider resident (#136). Unset or empty keeps the provider default
// (float32); the only other accepted value is bfloat16. It is read at launch,
// never persisted, and applies to no other provider. The dtype a resident is
// actually on is always reported by /v1/status (worker.provider.dtype).
const EnvOpenDeciderDType = "HACHIDORI_OPENDECIDER_DTYPE"

// EnvClefDType is the explicit control for the dtype of a Clef resident that
// runs the upstream source model: bfloat16 (the default, what the release
// ships) or float32, the high-precision reference. It is read at launch, never
// persisted, and refused for a variant, which executes at the dtype its
// manifest declares.
const EnvClefDType = "HACHIDORI_CLEF_DTYPE"

// providerDType resolves the dtype control of model's provider. An unsupported
// value is a launch error, never silently replaced by another dtype.
func providerDType(model home.ModelManifest, variant bool) (string, error) {
	env := ""
	switch model.Provider {
	case setup.ProviderOpenDecider:
		env = EnvOpenDeciderDType
	case home.ProviderClef:
		env = EnvClefDType
	default:
		return "", nil
	}
	v := os.Getenv(env)
	if v == "" {
		return "", nil
	}
	if v != "float32" && v != "bfloat16" {
		return "", fmt.Errorf("%s=%q: want float32 or bfloat16", env, v)
	}
	if variant {
		return "", fmt.Errorf("%s=%q: a variant executes at the dtype its manifest declares", env, v)
	}
	return v, nil
}

// workerConfig is the launch configuration of the model an activation-shaped
// record names, on the runtime rm.
func workerConfig(h home.Home, a home.Active, rm home.RuntimeManifest, mm home.ModelManifest, log io.Writer) (worker.Config, Runtime, error) {
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
	args := []string{"-I", "-X", "utf8", script, "--model-dir", modelDir, "--device", a.Device,
		"--manifest", filepath.Join(modelDir, "hachidori-model.json"), "--provider", model.Provider}
	status := Runtime{Home: h.Root, Runtime: a.Runtime, ModelID: model.ID, Model: a.Model, Device: a.Device}
	variant, err := launchVariant(h, a, rm, model)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	if variant != nil {
		vdir := h.VariantDir(model.ID, variant.ID)
		args = append(args, "--variant-dir", vdir, "--variant-manifest", filepath.Join(vdir, home.VariantManifestFile))
		status.Variant = variant
	}
	dtype, err := providerDType(model, variant != nil)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	if dtype != "" {
		args = append(args, "--dtype", dtype)
	}
	cfg := worker.Config{
		Python:         python,
		Args:           args,
		Env:            h.Env(filepath.Dir(python), true),
		Dir:            h.Path("state"),
		Log:            log,
		StartTimeout:   10 * time.Minute,
		RequestTimeout: 2 * time.Minute,
		// The worker script is half of the launch contract (its command line
		// and protocol); a runtime materialized by an older build keeps its
		// older script. The supervisor refuses such a launch with the cause
		// instead of starting a process that can only die in argument
		// parsing. The binding itself still comes up, so the dashboard can
		// show why and offer the recovery (Materialize, Activate, Restart).
		Preflight: func() error { return setup.CheckWorkerContract(a, rm) },
	}
	return cfg, status, nil
}

// launchVariant resolves the variant an activation record selects and checks
// everything that must hold at every launch: its manifest and identity, its
// link to exactly this catalog model, the runtime's provider, and its
// certification state. Nothing falls back: a record that names a variant that
// cannot be launched fails the launch with the cause, and the source artifact
// is never started in its place. The worker verifies every artifact digest
// before it loads anything.
func launchVariant(h home.Home, a home.Active, rm home.RuntimeManifest, model home.ModelManifest) (*Variant, error) {
	v, ok, err := h.LoadVariant(a)
	if err != nil {
		return nil, err
	}
	if !ok {
		if a.Experimental {
			return nil, errors.New("the activation record is marked experimental but names no variant")
		}
		return nil, nil
	}
	if err := v.CheckSource(model); err != nil {
		return nil, err
	}
	if !rm.Spec.Provides(v.Provider) {
		return nil, fmt.Errorf("runtime %s does not carry provider %s needed by variant %s (run `hachidori setup`)", a.Runtime, v.Provider, v.ID)
	}
	st := eval.ResolveCertification(h, v)
	cert := st.State
	switch {
	case st.State == eval.StateAccepted:
	case st.State == eval.StateUncertified && a.Experimental:
		cert = eval.StateExperimental
	default:
		return nil, fmt.Errorf("variant %s: certification state is %s; an accepted certification record is required (or an explicit experimental activation of an uncertified variant)", v.ID, st.State)
	}
	return &Variant{ID: v.ID, Recipe: v.Recipe.Name, Scheme: v.Weights.Scheme, Bits: v.Weights.Bits, DType: v.Weights.DType,
		Format: v.Weights.Format, Engine: v.Optimizer.Engine, EngineVersion: v.Optimizer.Version, ManifestSHA256: v.ManifestSHA256(),
		Certification: cert, Source: VariantSource{ID: v.Source.ID, Provider: v.Source.Provider, Repo: v.Source.Repo, Revision: v.Source.Revision}}, nil
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
	api.ErrRoutingFailed:   http.StatusFailedDependency,
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
