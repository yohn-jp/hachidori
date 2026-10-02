package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/doctor"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/redact"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// A probe answers one question: can this exact persisted variant load and
// produce a valid typed decision? It starts an isolated worker on the
// variant's published artifact through the normal provider/worker path, waits
// for READY, asks one small fixed typed question, records what it observed and
// stops the worker.
//
// A probe is not a certification and not an activation. It never writes a
// certification record, never changes the activation record, never binds or
// replaces a resident, never falls back to the source or to another device,
// and leaves the active resident set exactly as it found it: its worker is its
// own process and is gone when it returns.

// ProbeSchema identifies a probe record.
const ProbeSchema = "hachidori.variant-probe.v1"

// Probe results.
const (
	ProbePassed = "passed"
	ProbeFailed = "failed"
)

// Probe phases: where a failed probe stopped.
const (
	ProbePhasePreflight  = "preflight"  // readiness, including source and variant integrity
	ProbePhaseConfigure  = "configure"  // resolving the launch of the isolated worker
	ProbePhaseStart      = "start"      // the worker importing, loading and warming up
	ProbePhaseProvenance = "provenance" // the worker reported something other than the requested variant/device
	ProbePhaseDecide     = "decide"     // the typed decision request
	ProbePhaseValidate   = "validate"   // the typed decision contract
	ProbePhaseDone       = "done"
)

// ProbeCertificationEffect states what a probe does to certification, in the
// record itself.
const ProbeCertificationEffect = "none: a probe is not a certification, an activation or a route change"

// ProbeRecord is the persisted outcome of one probe. Result "passed" means
// only that this variant loaded and answered one valid typed decision on this
// device at this time; it carries no fidelity claim.
type ProbeRecord struct {
	Schema string `json:"schema"`
	Result string `json:"result"`
	// Phase is the last phase reached; for a failed probe, where it failed.
	Phase                 string         `json:"phase"`
	Model                 string         `json:"model"`
	Variant               string         `json:"variant"`
	Provider              string         `json:"provider"`
	Runtime               string         `json:"runtime"`
	Device                string         `json:"device"`
	SourceRevision        string         `json:"source_revision"`
	VariantManifestSHA256 string         `json:"variant_manifest_sha256"`
	Recipe                string         `json:"recipe"`
	Scheme                string         `json:"scheme"`
	Certification         string         `json:"variant_certification,omitempty"` // the variant's state when probed, as reported by the worker's launch
	Execution             ProbeExecution `json:"execution"`
	Timing                ProbeTiming    `json:"timing"`
	Decision              *ProbeDecision `json:"decision,omitempty"`
	Resources             ProbeResources `json:"resources"`
	PreflightOutcome      string         `json:"preflight_outcome,omitempty"`
	Error                 string         `json:"error,omitempty"`
	WorkerClass           string         `json:"worker_failure_class,omitempty"`
	StderrTail            []string       `json:"worker_stderr_tail,omitempty"`
	TornDown              bool           `json:"worker_stopped"`
	StartedAt             string         `json:"started_at"`
	FinishedAt            string         `json:"finished_at"`
	Effect                string         `json:"certification_effect"`
}

// ProbeExecution is what the worker reported about how it actually runs.
type ProbeExecution struct {
	ModelID         string `json:"model_id,omitempty"`
	VariantID       string `json:"variant_id,omitempty"`
	Provider        string `json:"provider,omitempty"`
	ProviderVersion string `json:"provider_version,omitempty"`
	Device          string `json:"device,omitempty"`
	DeviceName      string `json:"device_name,omitempty"`
	DType           string `json:"dtype,omitempty"`
	Quantization    string `json:"quantization,omitempty"` // the variant's declared scheme
	TorchVersion    string `json:"torch_version,omitempty"`
	TorchCUDA       string `json:"torch_cuda,omitempty"`
	PythonVersion   string `json:"python_version,omitempty"`
	PID             int    `json:"pid,omitempty"`
}

// ProbeTiming is wall time observed here and timings the worker measured.
// Warmup is the worker's own warm-up inference, distinct from the request.
type ProbeTiming struct {
	StartupMS float64 `json:"startup_ms"`          // spawn to READY, as seen here
	LoadMS    float64 `json:"load_ms,omitempty"`   // the worker's model load
	WarmupMS  float64 `json:"warmup_ms,omitempty"` // the worker's warm-up inference
	RequestMS float64 `json:"request_ms,omitempty"`
	Inference float64 `json:"inference_ms,omitempty"` // inside the worker
}

// ProbeDecision is the one typed decision the probe asked for.
type ProbeDecision struct {
	Question      string             `json:"question"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Valid         bool               `json:"valid"`
}

// ProbeResources are the observations available. A zero value is "not
// observed", never "zero use".
type ProbeResources struct {
	HostRSSBytes    uint64 `json:"worker_host_rss_bytes,omitempty"`
	VRAMAllocated   uint64 `json:"vram_allocated_bytes,omitempty"`
	VRAMReserved    uint64 `json:"vram_reserved_bytes,omitempty"`
	VRAMFree        uint64 `json:"vram_free_bytes,omitempty"`
	VRAMTotal       uint64 `json:"vram_total_bytes,omitempty"`
	HostRAMTotal    uint64 `json:"host_ram_total_bytes,omitempty"`
	HostRAMAvail    uint64 `json:"host_ram_available_bytes,omitempty"`
	WeightFileBytes uint64 `json:"variant_weight_bytes,omitempty"`
}

// ProbeError is a probe that failed after it began. It wraps the cause; the
// record says how far it got.
type ProbeError struct {
	Phase string
	Err   error
}

func (e *ProbeError) Error() string { return e.Err.Error() }
func (e *ProbeError) Unwrap() error { return e.Err }

// ProbeParams names the exact thing probed: a variant and the explicit device.
type ProbeParams struct {
	Variant string
	Device  string
}

// ProbeDeps are the replaceable parts of Probe.
type ProbeDeps struct {
	// Config resolves the isolated worker's launch (server.ProbeConfig).
	Config func(h home.Home, device, variant string, log io.Writer) (worker.Config, server.Runtime, error)
	// Start launches it and waits for READY (worker.Start).
	Start func(ctx context.Context, cfg worker.Config, onPhase func(string)) (*worker.Process, error)
	// Request is the one typed request asked; doctor.SmokeRequest by default.
	Request *api.DecideRequest
	// Preflight carries the observation overrides of the preflight.
	Preflight optimize.PreflightDeps
	Now       func() time.Time
}

func (d ProbeDeps) withDefaults() ProbeDeps {
	if d.Config == nil {
		d.Config = server.ProbeConfig
	}
	if d.Start == nil {
		d.Start = worker.Start
	}
	if d.Request == nil {
		r := doctor.SmokeRequest
		d.Request = &r
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	return d
}

// Probe loads the persisted variant in an isolated worker and asks one typed
// question. The returned record is always populated as far as the probe got
// and is persisted under the home; the error is the primary failure, if any,
// and is never replaced by a failure to persist the record.
func Probe(ctx context.Context, h home.Home, p ProbeParams, deps ProbeDeps, log io.Writer, obs *setup.Observer) (rec ProbeRecord, err error) {
	deps = deps.withDefaults()
	rec = ProbeRecord{Schema: ProbeSchema, Variant: p.Variant, Device: p.Device, Phase: ProbePhasePreflight, Effect: ProbeCertificationEffect,
		StartedAt: deps.Now().UTC().Format(time.RFC3339Nano)}
	scrub := redact.New(h.Root)
	var proc *worker.Process
	// Whatever happens, the probe worker is torn down and the outcome is
	// recorded. Failing to record never replaces the probe's own failure; for a
	// pass it is reported, since the pass could not be recorded.
	defer func() {
		if proc != nil {
			proc.Close()
			rec.TornDown = true
		}
		rec.FinishedAt = deps.Now().UTC().Format(time.RFC3339Nano)
		if serr := SaveProbe(h, rec); serr != nil && err == nil {
			err = fmt.Errorf("the probe passed but its record could not be saved: %w", serr)
		}
	}()
	fail := func(phase string, e error) (ProbeRecord, error) {
		rec.Result, rec.Phase, rec.Error = ProbeFailed, phase, scrub.Line(e.Error(), 2048)
		return rec, &ProbeError{Phase: phase, Err: e}
	}
	model, v, ferr := setup.FindVariant(h, p.Variant)
	if ferr != nil {
		return fail(ProbePhasePreflight, ferr)
	}
	rec.Model, rec.Provider, rec.SourceRevision = model.ID, model.Provider, model.Revision
	rec.VariantManifestSHA256, rec.Recipe, rec.Scheme = v.ManifestSHA256(), v.Recipe.Name, v.Weights.Scheme
	rec.Execution.Quantization = v.Weights.Scheme
	if spec, serr := setup.Desired(p.Device); serr == nil {
		rec.Runtime = spec.ID()
	}

	// Integrity first: the source and the variant are verified against their
	// digests, the runtime and the device are checked, before anything loads.
	rep := optimize.Preflight(ctx, h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: p.Variant, Device: p.Device}, deps.Preflight, obs)
	_ = SavePreflight(h, rep)
	rec.PreflightOutcome = rep.Outcome
	if rep.Blocked() {
		return fail(ProbePhasePreflight, &setup.PreflightError{Report: rep})
	}
	obs.Phase(setup.PhaseProbing)

	rec.Phase = ProbePhaseConfigure
	cfg, rt, cerr := deps.Config(h, p.Device, p.Variant, log)
	if cerr != nil {
		return fail(ProbePhaseConfigure, cerr)
	}
	if rt.Variant != nil {
		rec.Certification = rt.Variant.Certification
	}
	if cfg.Preflight != nil {
		if perr := cfg.Preflight(); perr != nil {
			return fail(ProbePhaseConfigure, perr)
		}
	}

	rec.Phase = ProbePhaseStart
	obs.Step(setup.StepProbe, "starting an isolated worker for variant "+p.Variant)
	t0 := deps.Now()
	var serr error
	proc, serr = deps.Start(ctx, cfg, func(ph string) { obs.Step(setup.StepProbe, "worker "+ph) })
	rec.Timing.StartupMS = ms(deps.Now().Sub(t0))
	if serr != nil {
		proc = nil
		var wf *worker.Failure
		if errors.As(serr, &wf) {
			rec.WorkerClass = wf.Class
			rec.StderrTail = scrubTail(scrub, wf.Stderr)
		}
		if ctx.Err() != nil {
			serr = fmt.Errorf("probe cancelled: %w (%w)", ctx.Err(), serr)
		}
		return fail(ProbePhaseStart, serr)
	}
	info := proc.Info
	rec.Execution = executionOf(info, proc.PID, v.Weights.Scheme)
	rec.Timing.LoadMS, rec.Timing.WarmupMS = num(info, "load_ms"), num(info, "warmup_ms")
	obs.Step(setup.StepProbe, "worker ready")

	// What loaded is what was asked for: the variant, on the device, with no
	// substitution of the source or of cpu.
	rec.Phase = ProbePhaseProvenance
	if got := str(info, "variant_id"); got != v.ID {
		return fail(ProbePhaseProvenance, fmt.Errorf("the probe worker reports variant %q, not %q: it did not load the persisted variant (the source is never substituted)", got, v.ID))
	}
	if got := str(info, "model_id"); got != model.ID {
		return fail(ProbePhaseProvenance, fmt.Errorf("the probe worker reports source model %q, not %q", got, model.ID))
	}
	if dev := str(info, "device"); dev != p.Device && !(p.Device == "cuda" && len(dev) >= 4 && dev[:4] == "cuda") {
		return fail(ProbePhaseProvenance, fmt.Errorf("device %q was requested but the probe worker is on %q; there is no fallback to another device", p.Device, dev))
	}

	rec.Phase = ProbePhaseDecide
	obs.Step(setup.StepProbe, "one typed decision")
	req := *deps.Request
	t1 := deps.Now()
	results, inferenceMS, derr := proc.Decide([]worker.Item{{State: req.State, Questions: req.Questions}})
	rec.Timing.RequestMS, rec.Timing.Inference = ms(deps.Now().Sub(t1)), inferenceMS
	observeResources(&rec, proc, deps, v, h)
	if derr != nil {
		var wf *worker.Failure
		if errors.As(derr, &wf) {
			rec.WorkerClass = wf.Class
			rec.StderrTail = scrubTail(scrub, wf.Stderr)
		}
		return fail(ProbePhaseDecide, derr)
	}

	rec.Phase = ProbePhaseValidate
	if len(results) != 1 || len(results[0]) != len(req.Questions) {
		return fail(ProbePhaseValidate, fmt.Errorf("the worker answered %d results for one item of %d questions", len(results), len(req.Questions)))
	}
	for i, q := range req.Questions {
		res := results[0][i]
		d := &ProbeDecision{Question: q.ID, Choice: res.Choice, Confidence: res.Confidence, Probabilities: res.Probabilities}
		if i == 0 {
			rec.Decision = d
		}
		if verr := res.Validate(q); verr != nil {
			return fail(ProbePhaseValidate, verr)
		}
		d.Valid = true
	}
	rec.Result, rec.Phase = ProbePassed, ProbePhaseDone
	return rec, nil
}

func executionOf(info worker.Info, pid int, scheme string) ProbeExecution {
	return ProbeExecution{ModelID: str(info, "model_id"), VariantID: str(info, "variant_id"), Provider: str(info, "provider"), ProviderVersion: str(info, "provider_version"),
		Device: str(info, "device"), DeviceName: str(info, "device_name"), DType: str(info, "dtype"), Quantization: scheme,
		TorchVersion: str(info, "torch_version"), TorchCUDA: str(info, "torch_cuda"), PythonVersion: str(info, "python_version"), PID: pid}
}

// observeResources records the RAM/VRAM observations that exist. The worker's
// own statistics are asked after the request, so they include its working set.
func observeResources(rec *ProbeRecord, proc *worker.Process, deps ProbeDeps, v home.VariantManifest, h home.Home) {
	if stats, err := proc.Stats(); err == nil {
		rec.Resources.HostRSSBytes = u64(stats, "host_rss_bytes")
		rec.Resources.VRAMAllocated, rec.Resources.VRAMReserved = u64(stats, "memory_allocated"), u64(stats, "memory_reserved")
		rec.Resources.VRAMFree, rec.Resources.VRAMTotal = u64(stats, "memory_free"), u64(stats, "memory_total")
	}
	host := setup.DefaultHost()
	if deps.Preflight.Host != nil {
		host = *deps.Preflight.Host
	}
	if m := host.Mem(); m.TotalKnown {
		rec.Resources.HostRAMTotal = m.Total
		if m.AvailableKnown {
			rec.Resources.HostRAMAvail = m.Available
		}
	}
	var w uint64
	for _, rel := range v.VariantFileNames() {
		if fi, err := statSize(h.VariantDir(v.Source.ID, v.ID), rel); err == nil {
			w += fi
		}
	}
	rec.Resources.WeightFileBytes = w
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func str(m map[string]any, k string) string { v, _ := m[k].(string); return v }

func num(m map[string]any, k string) float64 { v, _ := m[k].(float64); return v }

func u64(m map[string]any, k string) uint64 {
	if v, _ := m[k].(float64); v > 0 {
		return uint64(v)
	}
	return 0
}

// scrubTail bounds and redacts worker stderr lines for a record.
func scrubTail(s redact.Scrubber, lines []string) []string {
	const maxLines, maxLine = 40, 512
	if len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = s.Line(l, maxLine)
	}
	return out
}
