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
	"github.com/yohn-jp/hachidori/internal/requesthistory"
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

// ObservedDecider is an optional Decider capability used to capture the
// actual worker queue and dispatch boundaries without wrapping inference.
type ObservedDecider interface {
	DecideObserved(items []worker.Item, observer worker.ExecutionObserver) ([][]api.Result, float64, error)
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

// ObservedRouter is the observer-aware form of Router.
type ObservedRouter interface {
	DecideOnObserved(model string, items []worker.Item, observer worker.ExecutionObserver) ([][]api.Result, float64, error)
}

// RequestIdentityProvider reports a resident's static execution identity
// without asking it for worker statistics.
type RequestIdentityProvider interface {
	RequestIdentity(model string) (requesthistory.Identity, bool)
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
//
// Runtime is the dependency runtime identity: the immutable Python/CUDA
// environment the worker runs in, derived only from its material contract.
// RuntimeDirectory is where it lives when that is not its identity (a runtime
// materialized under the previous identity scheme). Worker is the Hachidori
// worker implementation delivered by this build, observed separately: a
// worker-only update changes Worker and leaves Runtime as it was.
type Runtime struct {
	Home             string       `json:"home"`
	Runtime          string       `json:"runtime"`
	RuntimeDirectory string       `json:"runtime_directory,omitempty"`
	Worker           *WorkerBuild `json:"worker_build,omitempty"`
	ModelID          string       `json:"model_id"` // catalog model identity
	Model            string       `json:"model"`    // model directory: <repo>/<revision>
	Device           string       `json:"device"`
	Variant          *Variant     `json:"variant,omitempty"`
}

// WorkerBuild is the identity of the worker implementation that runs: the
// digest of the delivered script and the worker/runtime ABI it requires. ABI
// is compared with the runtime's declared ABI at launch; the digest is
// evidence only.
type WorkerBuild struct {
	SHA256 string `json:"sha256"`
	ABI    string `json:"abi"`
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
	return HandlerSinceWithHistory(d, rt, started, requesthistory.New())
}

// HandlerSinceWithHistory shares one bounded request history with the
// dashboard that reads it. The worker and API behavior remains the same when
// history is not queried.
func HandlerSinceWithHistory(d Decider, rt Runtime, started time.Time, requests *requesthistory.Store) http.Handler {
	if requests == nil {
		requests = requesthistory.New()
	}
	return hostLocal(routes(d, rt, started, requests))
}

func routes(d Decider, rt Runtime, started time.Time, requests *requesthistory.Store) *http.ServeMux {
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
	mux.HandleFunc("POST /v1/states", func(w http.ResponseWriter, r *http.Request) {
		var input api.RegisterState
		if err := decode(w, r, &input); err != nil {
			writeErr(w, api.ErrRequestInvalid, err.Error())
			return
		}
		ref, err := (home.Home{Root: rt.Home}).RegisterState(input.State)
		if err != nil {
			writeErr(w, api.ErrRequestInvalid, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, api.StateReference{StateRef: ref})
	})
	mux.HandleFunc("POST /v1/decide", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		var req api.DecideRequest
		if err := decode(w, r, &req); err != nil {
			msg := "invalid JSON body: " + err.Error()
			entry := requests.Begin("/v1/decide", t0, requestIdentity(rt))
			entry.Reject(time.Now(), http.StatusBadRequest, api.ErrRequestInvalid, msg)
			writeErr(w, api.ErrRequestInvalid, msg)
			return
		}
		entry := requests.Begin("/v1/decide", t0, requestIdentity(rt))
		entry.SetInput(req, len(req.State), len(req.Questions), 1)
		if req.Route == api.RouteAuto {
			entry.SetIdentity(requestBaseIdentity(rt))
		} else {
			entry.SetIdentity(requestedIdentity(d, rt, api.ModelRef(req.Model)))
		}
		if err := req.Validate(); err != nil {
			entry.Reject(time.Now(), http.StatusBadRequest, api.ErrRequestInvalid, err.Error())
			writeErr(w, api.ErrRequestInvalid, err.Error())
			return
		}
		stateRef := req.StateRef
		if stateRef != "" {
			state, err := (home.Home{Root: rt.Home}).ResolveState(stateRef)
			if err != nil {
				entry.Reject(time.Now(), http.StatusBadRequest, api.ErrRequestInvalid, err.Error())
				writeErr(w, api.ErrRequestInvalid, err.Error())
				return
			}
			req.State = state
		} else {
			stateRef = home.StateRef(req.State)
		}
		entry.SetStateRef(stateRef, len(req.State))
		entry.Admit()
		observer := requestObserver{entry: entry, decider: d, runtime: rt}
		items := []worker.Item{{State: req.State, StateRef: req.StateRef, Questions: req.Questions}}
		if req.Route == api.RouteAuto {
			out, rtr, err := decideRouted(d, items, observer)
			if err != nil {
				finishRequestError(entry, sc, err)
				writeWorkerErr(w, sc, err)
				return
			}
			resp := api.DecideResponse{Schema: api.SchemaV1, Results: out.Items[0].Results, StateRef: stateRef,
				Timing:  &api.Timing{InferenceMS: out.InferenceMS, TotalMS: msSince(t0)},
				Routing: rtr.Routing(out.Items[0], out.Providers)}
			entry.Complete(time.Now(), resp, http.StatusOK)
			writeJSON(w, http.StatusOK, resp)
			return
		}
		res, ms, served, err := decideTargeted(d, rt, api.ModelRef(req.Model), items, observer)
		if err != nil {
			finishRequestError(entry, sc, err)
			writeWorkerErr(w, sc, err)
			return
		}
		resp := api.DecideResponse{Schema: api.SchemaV1, Results: res[0], StateRef: stateRef,
			Timing: &api.Timing{InferenceMS: ms, TotalMS: msSince(t0)}, Served: served}
		entry.Complete(time.Now(), resp, http.StatusOK)
		writeJSON(w, http.StatusOK, resp)
	})
	mux.HandleFunc("POST /v1/decide/batch", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		var req api.BatchRequest
		if err := decode(w, r, &req); err != nil {
			msg := "invalid JSON body: " + err.Error()
			entry := requests.Begin("/v1/decide/batch", t0, requestIdentity(rt))
			entry.Reject(time.Now(), http.StatusBadRequest, api.ErrRequestInvalid, msg)
			writeErr(w, api.ErrRequestInvalid, msg)
			return
		}
		entry := requests.Begin("/v1/decide/batch", t0, requestIdentity(rt))
		stateBytes, questionCount := 0, 0
		for _, item := range req.Requests {
			stateBytes += len(item.State)
			questionCount += len(item.Questions)
		}
		entry.SetInput(req, stateBytes, questionCount, len(req.Requests))
		mode, _ := req.RouteMode()
		if mode == api.RouteAuto {
			entry.SetIdentity(requestBaseIdentity(rt))
		} else {
			target, _ := req.Target()
			entry.SetIdentity(requestedIdentity(d, rt, target))
		}
		if err := req.Validate(); err != nil {
			entry.Reject(time.Now(), http.StatusBadRequest, api.ErrRequestInvalid, err.Error())
			writeErr(w, api.ErrRequestInvalid, err.Error())
			return
		}
		items := make([]worker.Item, len(req.Requests))
		refs := make([]string, len(req.Requests))
		for i, q := range req.Requests {
			if q.StateRef != "" {
				state, err := (home.Home{Root: rt.Home}).ResolveState(q.StateRef)
				if err != nil {
					entry.Reject(time.Now(), http.StatusBadRequest, api.ErrRequestInvalid, err.Error())
					writeErr(w, api.ErrRequestInvalid, err.Error())
					return
				}
				q.State = state
				refs[i] = q.StateRef
			} else {
				refs[i] = home.StateRef(q.State)
			}
			items[i] = worker.Item{State: q.State, StateRef: q.StateRef, Questions: q.Questions}
		}
		entry.Admit()
		observer := requestObserver{entry: entry, decider: d, runtime: rt}
		if mode == api.RouteAuto { // validated above
			out, rtr, err := decideRouted(d, items, observer)
			if err != nil {
				finishRequestError(entry, sc, err)
				writeWorkerErr(w, sc, err)
				return
			}
			resp := api.BatchResponse{Schema: api.SchemaV1, Timing: api.Timing{InferenceMS: out.InferenceMS, TotalMS: msSince(t0)},
				Routing: &api.Routing{Mode: api.RouteAuto, Policy: rtr.Ref(), Handoffs: out.Handoffs, Providers: out.Providers}}
			for i, it := range out.Items {
				resp.Responses = append(resp.Responses, api.DecideResponse{Schema: api.SchemaV1, Results: it.Results, StateRef: refs[i], Routing: rtr.Routing(it, nil)})
			}
			entry.Complete(time.Now(), resp, http.StatusOK)
			writeJSON(w, http.StatusOK, resp)
			return
		}
		target, _ := req.Target() // validated above
		res, ms, served, err := decideTargeted(d, rt, target, items, observer)
		if err != nil {
			finishRequestError(entry, sc, err)
			writeWorkerErr(w, sc, err)
			return
		}
		out := api.BatchResponse{Schema: api.SchemaV1, Timing: api.Timing{InferenceMS: ms, TotalMS: msSince(t0)}, Served: served}
		for i, rs := range res {
			out.Responses = append(out.Responses, api.DecideResponse{Schema: api.SchemaV1, Results: rs, StateRef: refs[i]})
		}
		entry.Complete(time.Now(), out, http.StatusOK)
		writeJSON(w, http.StatusOK, out)
	})
	return mux
}

// decideRouted routes items by the runtime's routing policy. A runtime
// without a policy refuses the request; it never falls back to the default
// resident.
func decideRouted(d Decider, items []worker.Item, observer worker.ExecutionObserver) (route.Outcome, *route.Router, error) {
	var rtr *route.Router
	if a, ok := d.(AutoRouting); ok {
		rtr = a.AutoRouter()
	}
	if rtr == nil {
		return route.Outcome{}, nil, &worker.RequestError{Class: api.ErrRoutingFailed,
			Message: route.CodeNoPolicy + ": no routing policy is configured for this runtime (serve with --resident and --routing-policy); the request was not answered by any resident"}
	}
	out, err := rtr.DecideObserved(items, observer)
	return out, rtr, err
}

// decideTargeted runs items on the resident the caller named, or on the
// default route when model is empty (the unchanged compatibility path, which
// reports no provenance). A named model is served by that resident only: a
// Router answers for its own members; a single-worker Decider answers only
// for the model it serves. Anything else is a request_invalid error, never a
// different resident.
func decideTargeted(d Decider, rt Runtime, model string, items []worker.Item, observer worker.ExecutionObserver) ([][]api.Result, float64, *api.Served, error) {
	if model == "" {
		res, ms, err := decideObserved(d, items, observer)
		return res, ms, nil, err
	}
	if r, ok := d.(Router); ok {
		served, known := r.Identity(model)
		if !known {
			return nil, 0, nil, &worker.RequestError{Class: api.ErrRequestInvalid, Message: "model " + model + " is not resident"}
		}
		if target, ok := observer.(worker.TargetObserver); ok {
			target.Target(model)
		}
		var res [][]api.Result
		var ms float64
		var err error
		if observed, ok := d.(ObservedRouter); ok {
			res, ms, err = observed.DecideOnObserved(model, items, observer)
		} else {
			observeFallback(observer)
			res, ms, err = r.DecideOn(model, items)
			observeFinished(observer)
		}
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
	res, ms, err := decideObserved(d, items, observer)
	if err != nil {
		return nil, 0, nil, err
	}
	return res, ms, &served, nil
}

func decideObserved(d Decider, items []worker.Item, observer worker.ExecutionObserver) ([][]api.Result, float64, error) {
	if observed, ok := d.(ObservedDecider); ok {
		return observed.DecideObserved(items, observer)
	}
	observeFallback(observer)
	res, ms, err := d.Decide(items)
	observeFinished(observer)
	return res, ms, err
}

func observeFallback(observer worker.ExecutionObserver) {
	if observer != nil {
		at := time.Now()
		observer.Queued(at)
		observer.Started(at)
	}
}

func observeFinished(observer worker.ExecutionObserver) {
	if observer != nil {
		observer.Finished(time.Now())
	}
}

func requestIdentity(rt Runtime) requesthistory.Identity {
	id := requesthistory.Identity{Runtime: rt.Runtime, Model: rt.ModelID, Device: rt.Device}
	if rt.Variant != nil {
		id.Variant = rt.Variant.ID
	}
	return id
}

func requestBaseIdentity(rt Runtime) requesthistory.Identity {
	return requesthistory.Identity{Runtime: rt.Runtime, Device: rt.Device}
}

func requestedIdentity(d Decider, rt Runtime, model string) requesthistory.Identity {
	if model == "" {
		return requestIdentity(rt)
	}
	if p, ok := d.(RequestIdentityProvider); ok {
		if identity, found := p.RequestIdentity(model); found {
			return identity
		}
	}
	id := requestIdentity(rt)
	id.Model = model
	if model != rt.ModelID {
		id.Variant = ""
	}
	return id
}

type requestObserver struct {
	entry   *requesthistory.Request
	decider Decider
	runtime Runtime
}

func (o requestObserver) Queued(at time.Time)   { o.entry.Queued(at) }
func (o requestObserver) Started(at time.Time)  { o.entry.Started(at) }
func (o requestObserver) Finished(at time.Time) { o.entry.Finished(at) }
func (o requestObserver) Target(model string) {
	o.entry.SetIdentity(requestedIdentity(o.decider, o.runtime, model))
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
//
// override is an explicit dtype of an execution target (SourceConfig). It takes
// the place of the environment control and is refused for a variant and for a
// provider with no dtype control.
func providerDType(model home.ModelManifest, variant bool, override string) (string, error) {
	env := ""
	switch model.Provider {
	case setup.ProviderOpenDecider:
		env = EnvOpenDeciderDType
	case home.ProviderClef:
		env = EnvClefDType
	default:
		if override != "" {
			return "", fmt.Errorf("dtype %q: provider %s has no dtype control", override, model.Provider)
		}
		return "", nil
	}
	v, name := os.Getenv(env), env
	if override != "" {
		v, name = override, "dtype"
	}
	if v == "" {
		return "", nil
	}
	if v != "float32" && v != "bfloat16" {
		return "", fmt.Errorf("%s=%q: want float32 or bfloat16", name, v)
	}
	if variant {
		return "", fmt.Errorf("%s=%q: a variant executes at the dtype its manifest declares", name, v)
	}
	return v, nil
}

// workerConfig is the launch configuration of the model an activation-shaped
// record names, on the runtime rm.
func workerConfig(h home.Home, a home.Active, rm home.RuntimeManifest, mm home.ModelManifest, log io.Writer) (worker.Config, Runtime, error) {
	return workerConfigFor(h, a, rm, mm, log, false, "")
}

// ProbeConfig is the launch configuration of an isolated probe worker for one
// persisted variant on an explicit device: the same worker script, arguments,
// environment and variant-digest verification as serving it, resolved from the
// variant and the device's runtime instead of from the activation record. It
// reads only: nothing is activated, no certification is required (the
// variant's actual state is reported, never implied) and nothing falls back to
// the source or to another device.
func ProbeConfig(h home.Home, device, variantID string, log io.Writer) (worker.Config, Runtime, error) {
	model, v, err := setup.FindVariant(h, variantID)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	a, rm, mm, err := targetRecord(h, device, model)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	a.Variant = v.ID
	return workerConfigFor(h, a, rm, mm, log, true, "")
}

// SourceConfig is ProbeConfig for the pinned source model modelID on an
// explicit device: the exact-target launch of a source execution session. dtype
// is the explicit reference dtype of a provider with a dtype control (empty
// for the others). The activation record is neither read nor changed, the
// environment dtype control is not consulted, and nothing falls back to
// another model, dtype or device.
func SourceConfig(h home.Home, device, modelID, dtype string, log io.Writer) (worker.Config, Runtime, error) {
	model, err := setup.LookupModel(modelID)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	a, rm, mm, err := targetRecord(h, device, model)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	return workerConfigFor(h, a, rm, mm, log, false, dtype)
}

// targetRecord resolves the activation-shaped record of model on device from
// the device's materialized runtime and the materialized source, without the
// activation record.
func targetRecord(h home.Home, device string, model home.ModelManifest) (home.Active, home.RuntimeManifest, home.ModelManifest, error) {
	spec, err := setup.Desired(device)
	if err != nil {
		return home.Active{}, home.RuntimeManifest{}, home.ModelManifest{}, err
	}
	name := setup.RuntimeDirFor(h, spec)
	var rm home.RuntimeManifest
	if err := home.ReadJSON(filepath.Join(h.Path("runtime", name), "manifest.json"), &rm); err != nil || setup.CheckRuntimeManifest(name, rm, spec) != nil {
		return home.Active{}, home.RuntimeManifest{}, home.ModelManifest{}, fmt.Errorf("the %s runtime %s is not materialized for this build (run `hachidori setup --device %s`)", device, spec.ID(), device)
	}
	a := home.Active{Runtime: name, ModelID: model.ID, Model: setup.ModelDirName(model), Device: device}
	var mm home.ModelManifest
	if err := home.ReadJSON(filepath.Join(h.ModelDir(a), "hachidori-model.json"), &mm); err != nil {
		return home.Active{}, home.RuntimeManifest{}, home.ModelManifest{}, fmt.Errorf("source model %s is not materialized in this home: %w", model.ID, err)
	}
	return a, rm, mm, nil
}

// workerConfigFor is workerConfig; probe launches a variant whatever its
// certification state is (the state is still reported).
func workerConfigFor(h home.Home, a home.Active, rm home.RuntimeManifest, mm home.ModelManifest, log io.Writer, probe bool, dtypeOverride string) (worker.Config, Runtime, error) {
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
	// The worker is delivered by this build beside the dependency runtime and
	// runs in it; it is not part of the runtime.
	delivered, err := setup.DeliverWorker(h)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	script := delivered.Path
	modelDir := h.ModelDir(a)
	args := setup.PythonArgs(script, "--model-dir", modelDir, "--device", a.Device,
		"--manifest", filepath.Join(modelDir, "hachidori-model.json"), "--provider", model.Provider)
	status := Runtime{Home: h.Root, Runtime: rm.EnvironmentID(), ModelID: model.ID, Model: a.Model, Device: a.Device,
		Worker: &WorkerBuild{SHA256: delivered.SHA256, ABI: delivered.ABI}}
	if a.Runtime != status.Runtime {
		status.RuntimeDirectory = a.Runtime
	}
	variant, err := launchVariant(h, a, rm, model, probe)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	if variant != nil {
		vdir := h.VariantDir(model.ID, variant.ID)
		args = append(args, "--variant-dir", vdir, "--variant-manifest", filepath.Join(vdir, home.VariantManifestFile))
		status.Variant = variant
	}
	dtype, err := providerDType(model, variant != nil, dtypeOverride)
	if err != nil {
		return worker.Config{}, Runtime{}, err
	}
	if dtype != "" {
		args = append(args, "--dtype", dtype)
	}
	if model.Provider == "clef" {
		profile, profileErr := resolveCapacityProfile(h, a, rm, model, mm, variant, dtype)
		if profileErr != "" {
			args = append(args, "--capacity-profile-error", profileErr)
		} else if profile != nil {
			encoded, err := json.Marshal(profile)
			if err != nil {
				return worker.Config{}, Runtime{}, fmt.Errorf("encode capacity profile: %w", err)
			}
			args = append(args, "--capacity-profile", string(encoded))
		}
	}
	cfg := worker.Config{
		Python:         python,
		Args:           args,
		Env:            h.Env(filepath.Dir(python), true),
		Dir:            h.Path("state"),
		Log:            log,
		StartTimeout:   10 * time.Minute,
		RequestTimeout: 2 * time.Minute,
		// The worker of this build runs only in the dependency runtime this
		// build requires (or one that declares exactly that environment),
		// and only if that runtime was derived for the worker's ABI. The
		// supervisor refuses any other launch with the cause. The binding
		// itself still comes up, so the dashboard can show why; reconciliation
		// materializes and activates the required runtime.
		Preflight: func() error { return setup.CheckRuntimeCompatibility(a, rm) },
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
func launchVariant(h home.Home, a home.Active, rm home.RuntimeManifest, model home.ModelManifest, probe bool) (*Variant, error) {
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
	case probe:
		// A probe is not an activation: it loads the persisted variant
		// whatever its state, and reports that state as it is.
	case st.State == eval.StateAccepted:
	case st.State == eval.StateUncertified && a.Experimental && len(st.Problems) == 0:
		// Experimental is only for a variant with no record at all; records
		// that exist but do not verify never admit it.
		cert = eval.StateExperimental
	default:
		problems := ""
		if len(st.Problems) > 0 {
			problems = " (records not trusted: " + strings.Join(st.Problems, "; ") + ")"
		}
		return nil, fmt.Errorf("variant %s: certification state is %s%s; an accepted certification record is required (or an explicit experimental activation of a variant with no certification record)", v.ID, st.State, problems)
	}
	return &Variant{ID: v.ID, Recipe: v.Recipe.Name, Scheme: v.Weights.Scheme, Bits: v.Weights.Bits, DType: v.Weights.DType,
		Format: v.Weights.Format, Engine: v.Optimizer.Engine, EngineVersion: v.Optimizer.Version, ManifestSHA256: v.ManifestSHA256(),
		Certification: cert, Source: VariantSource{ID: v.Source.ID, Provider: v.Source.Provider, Repo: v.Source.Repo, Revision: v.Source.Revision}}, nil
}

func decode(w http.ResponseWriter, r *http.Request, v any) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	return nil
}

// maxErrorDetail bounds the worker-supplied detail of an error response.
const maxErrorDetail = requesthistory.MaxMetadataStringBytes

// writeWorkerErr answers a failed request. The error class is the stable,
// public part; the detail is the worker's own text (Python exception text
// for an inference failure) passed through the shared redaction policy and
// bounded, because the caller may be a remote host behind the reverse tunnel
// and the text can carry local paths or credential-bearing URLs. The operator
// sees the unredacted text in the worker log and the dashboard.
func writeWorkerErr(w http.ResponseWriter, sc redact.Scrubber, err error) {
	class, message := workerError(sc, err)
	var re *worker.RequestError
	if errors.As(err, &re) && re.Class == api.ErrCapacity && re.Capacity != nil {
		writeJSON(w, statusFor[api.ErrCapacity], api.ErrorBody{Schema: api.SchemaV1,
			Error: api.ErrorInfo{Class: class, Message: message, Capacity: re.Capacity}})
		return
	}
	writeErr(w, class, message)
}

func workerError(sc redact.Scrubber, err error) (string, string) {
	var re *worker.RequestError
	if errors.As(err, &re) {
		return re.Class, sc.Line(re.Message, maxErrorDetail)
	}
	var f *worker.Failure
	if errors.As(err, &f) {
		return api.ErrWorkerFailure, sc.Line(f.Class+": "+f.Message, maxErrorDetail)
	}
	return api.ErrWorkerFailure, sc.Line(err.Error(), maxErrorDetail)
}

func finishRequestError(entry *requesthistory.Request, sc redact.Scrubber, err error) {
	class, message := workerError(sc, err)
	code, ok := statusFor[class]
	if !ok {
		code = http.StatusInternalServerError
	}
	var failure *worker.Failure
	if errors.As(err, &failure) && failure.Class == worker.ClassUnresponsive {
		entry.Timeout(time.Now(), code, class, message)
		return
	}
	if !entry.WasStarted() {
		entry.Reject(time.Now(), code, class, message)
		return
	}
	entry.Fail(time.Now(), code, class, message)
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
