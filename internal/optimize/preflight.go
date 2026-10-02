package optimize

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// PreflightRequest names the expensive operation being gated. Every identity
// is a catalog identity: a model, a recipe or a variant ID.
type PreflightRequest struct {
	Kind    string // setup.PreflightMaterialize | Optimize | Probe | Certify
	Model   string // catalog model ID (optimize/materialize); a variant's own source for probe/certify
	Recipe  string // optimize: recipe name
	Variant string // probe/certify: variant ID
	// Device is the device the operation will run on: "cuda" or "cpu". It is
	// the optimizer's own cpu for an optimization unless a serving device is
	// being checked ahead of time. It is never defaulted or substituted.
	Device string
	// Verified: the caller has just verified the artifacts' digests itself
	// (Build does, right before it calls Preflight), so they are not hashed
	// twice.
	Verified bool
	// Quick skips hashing the artifacts. The digest findings are then UNKNOWN,
	// never PASS.
	Quick bool
	// ReferenceDType is the dtype of the high-precision reference of a
	// certification ("float32", or "" / "bfloat16": the release's own). It
	// sizes the reference's lower bound.
	ReferenceDType string
}

// PreflightDeps are the replaceable observations of Preflight.
type PreflightDeps struct {
	// Host observes disk and RAM (setup.DefaultHost when zero).
	Host *setup.Host
	// Accelerator observes torch/CUDA with the runtime's private interpreter
	// (setup.ProbeAccelerator when nil).
	Accelerator func(ctx context.Context, h home.Home, python string) (setup.AcceleratorFacts, error)
	Now         func() time.Time
}

// lowestWeightRatio is the smallest fraction of the source's 16-bit weights
// a 4-bit weight-only variant can occupy. It is a bound, not an estimate: the
// real output is larger by the preserved modules, the scales and the carried
// files.
const lowestWeightRatio = 4

// Preflight inspects everything Hachidori can know before the expensive
// operation req starts and reports it as typed findings. It never starts the
// operation, never downloads, never writes beyond a probe file in the target
// directory, and never changes any record. Failure to inspect something is
// a finding (blocker or unknown), not an error.
func Preflight(ctx context.Context, h home.Home, req PreflightRequest, deps PreflightDeps, obs *setup.Observer) setup.PreflightReport {
	now := deps.Now
	if now == nil {
		now = time.Now
	}
	host := setup.DefaultHost()
	if deps.Host != nil {
		host = *deps.Host
	}
	probeAccel := deps.Accelerator
	if probeAccel == nil {
		probeAccel = setup.ProbeAccelerator
	}
	obs.Phase(setup.PhasePreflight)
	rep := setup.NewPreflightReport(req.Kind, req.Model, now())
	rep.Recipe, rep.Variant, rep.Device = req.Recipe, req.Variant, req.Device
	// The identities are bound before anything is inspected: the findings
	// are conclusions about exactly these.
	b := PreflightBindingOf(h, req)
	rep.Binding = &b
	p := &preflight{ctx: ctx, h: h, req: req, host: host, accel: probeAccel, obs: obs, rep: rep}
	p.run()
	rep.NotMeasured = append(rep.NotMeasured, "network reachability and remote object sizes (the preflight performs no network access)")
	return *rep
}

// PreflightBindingOf derives the binding of the preflight target req from the
// authorities as they are now: the catalog model and its pinned identity, the
// materialized source manifest, the variant manifest, the recipe, the Runtime
// Spec of the operation's runtime and its materialized manifest, the device
// and the reference dtype. It reads manifests only and never hashes an
// artifact. A part that does not resolve is left empty, so a report whose
// target resolved differently then and now never matches.
//
// Preflight records it in the report; a reader recomputes it for the report's
// own target (PreflightRequestOf) and the report is current only while both
// are equal.
func PreflightBindingOf(h home.Home, req PreflightRequest) setup.PreflightBinding {
	b := setup.PreflightBinding{Kind: req.Kind, Device: req.Device}
	if req.Kind == setup.PreflightCertify {
		b.ReferenceDType = req.ReferenceDType
	}
	var m home.ModelManifest
	resolved := false
	switch req.Kind {
	case setup.PreflightProbe, setup.PreflightCertify:
		b.Variant = req.Variant
		if mm, v, err := setup.FindVariant(h, req.Variant); err == nil {
			m, resolved = mm, true
			b.VariantManifestSHA256, b.Recipe, b.RecipeSHA256 = v.ManifestSHA256(), v.Recipe.Name, v.RecipeSHA256
		}
	default:
		b.Model = req.Model
		if mm, err := setup.LookupModel(req.Model); err == nil {
			m, resolved = mm, true
		}
	}
	if resolved {
		src := home.SourceOf(m)
		b.Model, b.Provider, b.SourceRepo, b.SourceRevision, b.SourceFilesSHA256 = src.ID, src.Provider, src.Repo, src.Revision, src.FilesSHA256
		b.SourceManifestSHA256 = manifestDigest(h.Path("models", filepath.FromSlash(setup.ModelDirName(m)), "hachidori-model.json"))
		if req.Kind == setup.PreflightOptimize {
			b.Recipe = req.Recipe
			if r, err := LookupRecipe(m.ID, req.Recipe); err == nil {
				b.Recipe, b.RecipeSHA256 = r.Name, r.SHA256()
			}
		}
	}
	var spec home.RuntimeSpec
	var err error
	switch {
	case req.Kind == setup.PreflightOptimize:
		spec, err = setup.DesiredOptimizer()
	case req.Device != "":
		spec, err = setup.Desired(req.Device)
	default:
		return b
	}
	if err == nil {
		b.Runtime = spec.ID()
		b.RuntimeManifestSHA256 = manifestDigest(h.Path("runtime", spec.ID(), "manifest.json"))
	}
	return b
}

// PreflightRequestOf is the target a recorded report was requested for: what
// PreflightBindingOf is recomputed from when the report is read back.
func PreflightRequestOf(r setup.PreflightReport) PreflightRequest {
	req := PreflightRequest{Kind: r.Kind, Model: r.Model, Recipe: r.Recipe, Variant: r.Variant, Device: r.Device}
	if r.Binding != nil {
		req.ReferenceDType = r.Binding.ReferenceDType
	}
	return req
}

// manifestDigest is the SHA-256 of a manifest file, "" when it does not exist
// and a marker that matches no digest when it cannot be read.
func manifestDigest(path string) string {
	d, err := setup.FileSHA256(path)
	switch {
	case err == nil:
		return d
	case errors.Is(err, fs.ErrNotExist):
		return ""
	default:
		return "unreadable"
	}
}

// RequirePreflight runs Preflight and refuses the operation with a
// *setup.PreflightError when it found a blocker. The report is returned in
// either case so the caller can record it.
func RequirePreflight(ctx context.Context, h home.Home, req PreflightRequest, deps PreflightDeps, obs *setup.Observer) (setup.PreflightReport, error) {
	rep := Preflight(ctx, h, req, deps, obs)
	if rep.Blocked() {
		return rep, &setup.PreflightError{Report: rep}
	}
	return rep, nil
}

type preflight struct {
	ctx   context.Context
	h     home.Home
	req   PreflightRequest
	host  setup.Host
	accel func(ctx context.Context, h home.Home, python string) (setup.AcceleratorFacts, error)
	obs   *setup.Observer
	rep   *setup.PreflightReport

	model    home.ModelManifest
	modelOK  bool // the catalog model resolves
	srcDir   string
	srcReady bool // the source is materialized and its manifest matches the catalog
	variant  home.VariantManifest
	varDir   string
	varOK    bool
	// bytes known from the artifacts on disk
	srcWeightBytes uint64
	varWeightBytes uint64
	// python is the private interpreter of the runtime the operation uses, ""
	// when there is none.
	python string
	rm     home.RuntimeManifest
	spec   home.RuntimeSpec
	rtOK   bool
}

func (p *preflight) add(id, area string, st setup.FindingStatus, summary string, facts map[string]any) {
	p.rep.Add(setup.Finding{ID: id, Area: area, Status: st, Summary: summary, Facts: facts})
}

func (p *preflight) run() {
	p.resolveModel()
	if p.modelOK {
		p.sourceIdentity()
	}
	switch p.req.Kind {
	case setup.PreflightOptimize:
		p.optimizerRuntime()
		p.recipe()
		p.storageFor(p.h.VariantsDir(p.req.Model), p.optimizeDisk)
		p.memoryFor("source weights loaded at their stored 16-bit precision", p.srcWeightBytes, "RAM")
		p.accelerator()
	case setup.PreflightMaterialize:
		p.servingRuntime(false)
		p.storageFor(p.h.Path("models"), p.materializeDisk)
		p.accelerator()
	case setup.PreflightProbe, setup.PreflightCertify:
		p.variantIdentity()
		p.servingRuntime(true)
		p.storageFor(p.h.Path("state"), p.recordsDisk)
		p.servingMemory()
		p.accelerator()
		if p.req.Kind == setup.PreflightCertify {
			p.certificationState()
		}
	default:
		p.add("request.kind", setup.AreaIdentity, setup.FindingBlocker, fmt.Sprintf("unknown preflight kind %q", p.req.Kind), nil)
	}
}

func (p *preflight) resolveModel() {
	id := p.req.Model
	if p.req.Variant != "" && (p.req.Kind == setup.PreflightProbe || p.req.Kind == setup.PreflightCertify) {
		// A variant names its source: resolving it also validates its manifest
		// identity and its link to the catalog.
		m, v, err := setup.FindVariant(p.h, p.req.Variant)
		if err != nil {
			p.add("variant.manifest", setup.AreaIdentity, setup.FindingBlocker, err.Error(), map[string]any{"variant": p.req.Variant})
			return
		}
		p.model, p.variant, p.varDir, p.varOK, p.modelOK = m, v, p.h.VariantDir(m.ID, v.ID), true, true
		p.rep.Model = m.ID
		return
	}
	m, err := setup.LookupModel(id)
	if err != nil {
		p.add("source.catalog", setup.AreaIdentity, setup.FindingBlocker, err.Error(), map[string]any{"model": id})
		return
	}
	if p.req.Kind != setup.PreflightMaterialize && !setup.SupportsVariants(m) {
		p.add("source.catalog", setup.AreaIdentity, setup.FindingBlocker,
			fmt.Sprintf("model %s has no variants (only System One models are optimized)", m.ID), map[string]any{"model": m.ID, "provider": m.Provider})
		return
	}
	p.model, p.modelOK = m, true
	p.rep.Model = m.ID
}

// sourceIdentity checks the pinned source: manifest, identity, revision and
// every file digest.
func (p *preflight) sourceIdentity() {
	m := p.model
	p.srcDir = p.h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	facts := map[string]any{"model": m.ID, "provider": m.Provider, "repo": m.Repo, "revision": m.Revision, "files": len(m.Files)}
	cond, err := setup.InspectSource(p.srcDir, m)
	facts["condition"] = string(cond)
	if p.req.Kind == setup.PreflightMaterialize {
		// Materialization downloads an absent source, and verifies and
		// reuses a present one, failing on a present one that does not
		// verify. A present source is therefore checked as strictly as for
		// any other operation: what would make materialization fail is a
		// blocker now, not a pass because a manifest file exists.
		switch cond {
		case setup.SourceAbsent:
			p.add("source.manifest", setup.AreaIdentity, setup.FindingPass,
				fmt.Sprintf("catalog model %s pins %s@%s with %d files; it is not materialized yet", m.ID, m.Repo, m.Revision[:12], len(m.Files)), facts)
			return
		case setup.SourcePresent:
		default:
			p.add("source.manifest", setup.AreaIdentity, setup.FindingBlocker,
				fmt.Sprintf("a source of %s exists but is not the pinned catalog model (%s: %v); materialization would refuse to reuse it; Repair (Settings, Models & runtimes) rebuilds it", m.ID, cond, err), facts)
			return
		}
	} else if err != nil {
		p.add("source.manifest", setup.AreaIdentity, setup.FindingBlocker,
			fmt.Sprintf("source %s is not materialized as the pinned catalog model: %v (materialize it with `hachidori setup --model %s`)", m.ID, err, m.ID), facts)
		return
	}
	p.srcReady = true
	p.srcWeightBytes = weightBytes(p.srcDir, sortedKeys(m.Files))
	facts["weight_bytes"] = p.srcWeightBytes
	p.add("source.manifest", setup.AreaIdentity, setup.FindingPass,
		fmt.Sprintf("source %s matches the catalog: %s@%s, %d pinned files", m.ID, m.Repo, m.Revision[:12], len(m.Files)), facts)
	switch {
	case p.req.Verified:
		p.add("source.digests", setup.AreaIdentity, setup.FindingPass, "every pinned source file was verified against its digest immediately before this check", facts)
	case p.req.Quick:
		p.add("source.digests", setup.AreaIdentity, setup.FindingUnknown, "source file digests were not hashed (quick preflight); the operation verifies them before it uses them", facts)
	default:
		p.obs.Step(setup.StepVerify, "source "+m.ID)
		if err := setup.VerifyModel(p.srcDir, m); err != nil {
			p.srcReady = false
			p.add("source.digests", setup.AreaIdentity, setup.FindingBlocker, fmt.Sprintf("source %s failed verification: %v (repair it with `hachidori setup --model %s`)", m.ID, err, m.ID), facts)
		} else {
			p.add("source.digests", setup.AreaIdentity, setup.FindingPass, fmt.Sprintf("all %d pinned source files match their SHA-256", len(m.Files)), facts)
		}
	}
}

// variantIdentity checks the variant: its manifest, its lineage to the pinned
// source and every artifact digest.
func (p *preflight) variantIdentity() {
	if !p.varOK {
		return
	}
	v := p.variant
	facts := map[string]any{"variant": v.ID, "source": v.Source.ID, "source_revision": v.Source.Revision, "manifest_sha256": v.ManifestSHA256(),
		"recipe": v.Recipe.Name, "scheme": v.Weights.Scheme, "dtype": v.Weights.DType, "engine": v.Optimizer.Engine, "engine_version": v.Optimizer.Version}
	p.add("variant.manifest", setup.AreaIdentity, setup.FindingPass,
		fmt.Sprintf("variant %s is a valid manifest linked to source %s@%s (recipe %s, %s)", v.ID, v.Source.ID, v.Source.Revision[:12], v.Recipe.Name, v.Weights.Scheme), facts)
	p.varWeightBytes = weightBytes(p.varDir, v.VariantFileNames())
	facts["weight_bytes"] = p.varWeightBytes
	if !p.srcReady {
		return
	}
	p.add("variant.lineage", setup.AreaIdentity, setup.FindingPass,
		fmt.Sprintf("variant %s derives from exactly the materialized source %s (revision and file digests match)", v.ID, p.model.ID), facts)
	switch {
	case p.req.Verified:
		p.add("variant.digests", setup.AreaIdentity, setup.FindingPass, "every variant artifact was verified against its digest immediately before this check", facts)
	case p.req.Quick:
		p.add("variant.digests", setup.AreaIdentity, setup.FindingUnknown, "variant artifact digests were not hashed (quick preflight); the operation verifies them before it uses them", facts)
	default:
		p.obs.Step(setup.StepVerify, "variant "+v.ID)
		if err := setup.VerifyVariantArtifacts(p.h, p.model, v, p.obs); err != nil {
			p.add("variant.digests", setup.AreaIdentity, setup.FindingBlocker, fmt.Sprintf("variant %s failed verification: %v", v.ID, err), facts)
		} else {
			p.add("variant.digests", setup.AreaIdentity, setup.FindingPass,
				fmt.Sprintf("all %d variant artifacts match their digests and the preserved modules are intact", len(v.Files)), facts)
		}
	}
}

// installedVersions picks the named distributions out of a runtime manifest's
// verified package list.
func installedVersions(installed []string, names ...string) map[string]any {
	out := map[string]any{}
	for _, d := range installed {
		name, ver, _ := strings.Cut(d, "==")
		n := strings.ToLower(strings.ReplaceAll(name, "_", "-"))
		if slices.Contains(names, n) {
			out[n] = ver
		}
	}
	return out
}

var relevantDists = []string{"torch", "transformers", "compressed-tensors", "llmcompressor", "accelerate", "safetensors", "numpy"}

// optimizerRuntime checks the optimizer runtime that builds variants.
func (p *preflight) optimizerRuntime() {
	spec, err := setup.DesiredOptimizer()
	if err != nil {
		p.add("runtime.optimizer", setup.AreaRuntime, setup.FindingBlocker, "no optimizer runtime exists for this platform: "+err.Error(), nil)
		return
	}
	facts := map[string]any{"runtime": spec.ID(), "python": spec.Python, "torch": spec.Torch, "engine": setup.OptimizerEngine, "engine_version": setup.OptimizerEngineVersion}
	rt, err := setup.FindOptimizer(p.h)
	if err != nil {
		if _, serr := os.Stat(p.h.Path("runtime", spec.ID())); serr != nil {
			p.add("runtime.optimizer", setup.AreaRuntime, setup.FindingWarning,
				fmt.Sprintf("the optimizer runtime %s is not materialized; `variant optimize` materializes it first, which needs network access once", spec.ID()), facts)
			return
		}
		p.add("runtime.optimizer", setup.AreaRuntime, setup.FindingBlocker, err.Error(), facts)
		return
	}
	p.rm, p.python, p.rtOK = rt.Manifest, rt.Python, true
	facts["python_version"] = rt.Manifest.PythonVersion
	facts["installed"] = installedVersions(rt.Manifest.Installed, relevantDists...)
	p.add("runtime.optimizer", setup.AreaRuntime, setup.FindingPass,
		fmt.Sprintf("optimizer runtime %s is materialized with its pinned worker script (python %s)", rt.ID, rt.Manifest.PythonVersion), facts)
	have := installedVersions(rt.Manifest.Installed, "llmcompressor")["llmcompressor"]
	if have != setup.OptimizerEngineVersion {
		p.add("runtime.engine", setup.AreaRuntime, setup.FindingBlocker,
			fmt.Sprintf("the optimizer runtime has llmcompressor %v installed, the pinned engine is %s", have, setup.OptimizerEngineVersion), facts)
		return
	}
	p.add("runtime.engine", setup.AreaRuntime, setup.FindingPass, fmt.Sprintf("optimizer engine llmcompressor %s is installed as pinned", setup.OptimizerEngineVersion), facts)
}

// servingRuntime checks the runtime that serves the model on the requested
// device. required: the runtime must already exist (probe, certify).
func (p *preflight) servingRuntime(required bool) {
	if p.req.Device == "" {
		p.add("runtime.device", setup.AreaRuntime, setup.FindingBlocker, "no device was requested: name cuda or cpu; there is no default and no fallback", nil)
		return
	}
	spec, err := setup.Desired(p.req.Device)
	if err != nil {
		p.add("runtime.device", setup.AreaRuntime, setup.FindingBlocker, err.Error(), map[string]any{"device": p.req.Device})
		return
	}
	facts := map[string]any{"device": p.req.Device, "runtime": spec.ID(), "python": spec.Python, "provider": spec.Provider, "torch": spec.Torch, "flavor": spec.Flavor}
	if p.modelOK && !spec.Provides(p.model.Provider) {
		p.add("runtime.provider", setup.AreaRuntime, setup.FindingBlocker,
			fmt.Sprintf("runtime %s does not carry provider %s needed by model %s", spec.ID(), p.model.Provider, p.model.ID), facts)
		return
	}
	if p.req.Device == "cuda" && spec.Flavor == "cpu" {
		p.add("runtime.compat", setup.AreaRuntime, setup.FindingBlocker,
			"cuda was requested but the runtime for it is a cpu torch build; there is no fallback to cpu", facts)
		return
	}
	var rm home.RuntimeManifest
	dir := p.h.Path("runtime", spec.ID())
	if err := home.ReadJSON(filepath.Join(dir, "manifest.json"), &rm); err != nil || rm.Identity != spec.ID() || rm.Spec != spec {
		st, why := setup.FindingWarning, "is not materialized; materialize it with `hachidori setup` first"
		if required {
			st = setup.FindingBlocker
		}
		if err == nil {
			st, why = setup.FindingBlocker, "has a manifest that does not match its Runtime Spec"
		}
		p.add("runtime.serving", setup.AreaRuntime, st, fmt.Sprintf("the %s runtime %s %s", p.req.Device, spec.ID(), why), facts)
		return
	}
	if rm.Spec.Worker != setup.WorkerDigest() {
		p.add("runtime.worker", setup.AreaRuntime, setup.FindingBlocker,
			fmt.Sprintf("runtime %s was materialized by an older Hachidori (worker %.12s, this build %.12s); materialize the current runtime", spec.ID(), rm.Spec.Worker, setup.WorkerDigest()), facts)
		return
	}
	p.rm, p.spec, p.rtOK = rm, spec, true
	p.python = filepath.Join(dir, filepath.FromSlash(rm.PythonRelPath))
	facts["python_version"] = rm.PythonVersion
	facts["installed"] = installedVersions(rm.Installed, relevantDists...)
	p.add("runtime.serving", setup.AreaRuntime, setup.FindingPass,
		fmt.Sprintf("the %s runtime %s is materialized (python %s, torch %s, provider %s)", p.req.Device, spec.ID(), rm.PythonVersion, spec.Torch, spec.Provider), facts)
}

// accelerator reports the requested device against what torch observes.
func (p *preflight) accelerator() {
	dev := p.req.Device
	switch {
	case dev == "" && p.req.Kind == setup.PreflightOptimize:
		p.add("accelerator.device", setup.AreaAccelerator, setup.FindingPass, "optimization runs on cpu by contract; no accelerator is requested", map[string]any{"requested": "cpu"})
		return
	case dev == "cpu":
		p.add("accelerator.device", setup.AreaAccelerator, setup.FindingPass, "cpu was requested explicitly; no accelerator is used", map[string]any{"requested": "cpu"})
		return
	case dev == "":
		return // servingRuntime already refused the missing device
	case dev != "cuda":
		return // servingRuntime already refused the unknown device
	}
	facts := map[string]any{"requested": "cuda"}
	if !p.rtOK {
		p.add("accelerator.device", setup.AreaAccelerator, setup.FindingUnknown, "cuda was requested but no runtime is materialized to observe it with", facts)
		return
	}
	p.obs.Step(setup.StepVerify, "torch/CUDA of the private runtime")
	f, err := p.accel(p.ctx, p.h, p.python)
	if err != nil {
		p.add("accelerator.device", setup.AreaAccelerator, setup.FindingUnknown, "the CUDA device could not be observed: "+err.Error(), facts)
		return
	}
	facts["torch"], facts["torch_cuda"], facts["cuda_available"] = f.Torch, f.TorchCUDA, f.CUDAAvailable
	if f.Error != "" {
		facts["error"] = f.Error
		p.add("accelerator.device", setup.AreaAccelerator, setup.FindingUnknown, "torch could not report CUDA: "+f.Error, facts)
		return
	}
	if !f.CUDAAvailable {
		p.add("accelerator.device", setup.AreaAccelerator, setup.FindingBlocker,
			fmt.Sprintf("cuda was requested but torch %s (CUDA %s) reports no usable CUDA device; Hachidori never falls back to cpu", f.Torch, f.TorchCUDA), facts)
		return
	}
	facts["device_name"], facts["device_count"], facts["capability"] = f.DeviceName, f.DeviceCount, f.Capability
	facts["vram_total_bytes"], facts["vram_free_bytes"] = f.VRAMTotal, f.VRAMFree
	p.add("accelerator.device", setup.AreaAccelerator, setup.FindingPass,
		fmt.Sprintf("cuda was requested and torch %s (CUDA %s) sees %s with %s of %s VRAM free", f.Torch, f.TorchCUDA, f.DeviceName, setup.Bytes(f.VRAMFree), setup.Bytes(f.VRAMTotal)), facts)
	if p.req.Kind == setup.PreflightProbe || p.req.Kind == setup.PreflightCertify {
		p.vramFit(f)
	}
}

// vramFit compares the variant's known weight bytes with the observed VRAM.
// Whether the model fits with its activations is not known.
func (p *preflight) vramFit(f setup.AcceleratorFacts) {
	lower := p.varWeightBytes
	facts := map[string]any{"lower_bound_bytes": lower, "vram_total_bytes": f.VRAMTotal, "vram_free_bytes": f.VRAMFree,
		"basis": "variant weight files; activations, the joint head and the CUDA context are not included"}
	switch {
	case lower == 0:
		p.add("accelerator.vram", setup.AreaAccelerator, setup.FindingUnknown, "the variant's weight size is not known; whether it fits in VRAM is not known", facts)
	case f.VRAMTotal > 0 && f.VRAMTotal < lower:
		p.add("accelerator.vram", setup.AreaAccelerator, setup.FindingBlocker,
			fmt.Sprintf("the variant's weights alone (%s) exceed the device's total VRAM (%s)", setup.Bytes(lower), setup.Bytes(f.VRAMTotal)), facts)
	case f.VRAMFree > 0 && f.VRAMFree < lower:
		p.add("accelerator.vram", setup.AreaAccelerator, setup.FindingWarning,
			fmt.Sprintf("the variant's weights (%s) exceed the VRAM free right now (%s)", setup.Bytes(lower), setup.Bytes(f.VRAMFree)), facts)
	default:
		p.add("accelerator.vram", setup.AreaAccelerator, setup.FindingUnknown,
			fmt.Sprintf("the weights (%s) are below the observed VRAM (%s free of %s), but whether the model fits with activations is not known until it loads", setup.Bytes(lower), setup.Bytes(f.VRAMFree), setup.Bytes(f.VRAMTotal)), facts)
	}
}

// memoryFor compares a known lower bound with host RAM. It can refuse
// (installed RAM below the bound) or warn (available RAM below it); it never
// reports that the operation fits.
func (p *preflight) memoryFor(basis string, lower uint64, what string) {
	m := p.host.Mem()
	facts := map[string]any{"lower_bound_bytes": lower, "basis": basis, "total_known": m.TotalKnown, "available_known": m.AvailableKnown}
	if m.TotalKnown {
		facts["total_bytes"] = m.Total
	}
	if m.AvailableKnown {
		facts["available_bytes"] = m.Available
	}
	switch {
	case !m.TotalKnown:
		p.add("memory.fit", setup.AreaMemory, setup.FindingUnknown, "host RAM cannot be read on this platform; whether the operation fits is not known", facts)
	case lower > 0 && m.Total < lower:
		p.add("memory.fit", setup.AreaMemory, setup.FindingBlocker,
			fmt.Sprintf("the operation needs at least %s of %s (%s) but the host has %s installed", setup.Bytes(lower), what, basis, setup.Bytes(m.Total)), facts)
	case lower > 0 && m.AvailableKnown && m.Available < lower:
		p.add("memory.fit", setup.AreaMemory, setup.FindingWarning,
			fmt.Sprintf("the operation needs at least %s of %s (%s); %s are available now of %s installed", setup.Bytes(lower), what, basis, setup.Bytes(m.Available), setup.Bytes(m.Total)), facts)
	case lower == 0:
		p.add("memory.fit", setup.AreaMemory, setup.FindingUnknown,
			fmt.Sprintf("host RAM is %s; the operation's requirement is not known, so whether it fits is not known", setup.Bytes(m.Total)), facts)
	default:
		avail := "available RAM is not reported"
		if m.AvailableKnown {
			avail = setup.Bytes(m.Available) + " available"
		}
		p.add("memory.fit", setup.AreaMemory, setup.FindingUnknown,
			fmt.Sprintf("host RAM is %s (%s); the lower bound %s is below it, but the operation's peak use is not known, so whether it fits is not known until it runs", setup.Bytes(m.Total), avail, setup.Bytes(lower)), facts)
	}
}

// servingMemory sizes the RAM a probe or a certification's runs need.
func (p *preflight) servingMemory() {
	switch {
	case p.req.Kind == setup.PreflightCertify:
		src := p.srcWeightBytes
		basis := "the high-precision reference source weights at their stored 16-bit precision"
		if p.req.ReferenceDType == "float32" {
			src, basis = src*2, "the reference source weights widened to float32"
		}
		p.memoryFor(basis, src, "RAM (the cpu reference run)")
	case p.req.Device == "cpu":
		p.memoryFor("variant weight files loaded on the cpu", p.varWeightBytes, "RAM")
	default:
		// On cuda the weights live in VRAM (accelerator.vram); the host
		// needs RAM to stage them but its size is not known.
		p.memoryFor("host RAM while loading; the variant's weights are checked against VRAM", 0, "RAM")
	}
}

// storageFor checks that dir (or the directory that will hold it) is writable
// and describes the free capacity against need.
func (p *preflight) storageFor(dir string, need func(free uint64, freeKnown bool, facts map[string]any)) {
	facts := map[string]any{"path": dir}
	if err := setup.WritableDir(dir); err != nil {
		p.add("storage.writable", setup.AreaStorage, setup.FindingBlocker, fmt.Sprintf("%s is not writable: %v", dir, err), facts)
		return
	}
	p.add("storage.writable", setup.AreaStorage, setup.FindingPass, dir+" is writable", facts)
	free, ok := p.host.FreeDiskAt(dir)
	if ok {
		facts["free_bytes"] = free
	}
	need(free, ok, facts)
}

// optimizeDisk: the staged variant is at least a quarter of the source's
// 16-bit weights and at most their size.
func (p *preflight) optimizeDisk(free uint64, known bool, facts map[string]any) {
	upper := p.srcWeightBytes
	lower := upper / lowestWeightRatio
	facts["lower_bound_bytes"], facts["upper_bound_bytes"] = lower, upper
	facts["basis"] = fmt.Sprintf("the staged variant is between 1/%d and the whole of the source's weight files", lowestWeightRatio)
	switch {
	case !known:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingUnknown, "free disk space cannot be read for this volume; whether the variant fits is not known", facts)
	case upper == 0:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingUnknown, fmt.Sprintf("%s is free; the source's size is not known", setup.Bytes(free)), facts)
	case free < lower:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingBlocker,
			fmt.Sprintf("%s is free but even the smallest possible variant of this source needs %s", setup.Bytes(free), setup.Bytes(lower)), facts)
	case free < upper:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingWarning,
			fmt.Sprintf("%s is free; the variant needs between %s and %s", setup.Bytes(free), setup.Bytes(lower), setup.Bytes(upper)), facts)
	default:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingPass,
			fmt.Sprintf("%s is free, above the largest possible staged variant (%s)", setup.Bytes(free), setup.Bytes(upper)), facts)
	}
}

// materializeDisk: the catalog does not state file sizes, so only an
// interrupted download's remaining bytes are known.
func (p *preflight) materializeDisk(free uint64, known bool, facts map[string]any) {
	remaining, partial := p.remainingDownload()
	facts["partial_bytes"] = partial
	if remaining > 0 {
		facts["remaining_known_bytes"] = remaining
	}
	switch {
	case !known:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingUnknown, "free disk space cannot be read for this volume; whether the model fits is not known", facts)
	case remaining > 0 && free < remaining:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingBlocker,
			fmt.Sprintf("%s is free but the interrupted download still needs %s", setup.Bytes(free), setup.Bytes(remaining)), facts)
	default:
		msg := fmt.Sprintf("%s is free; the catalog states no file sizes, so the model's total size is not known", setup.Bytes(free))
		if partial > 0 {
			msg += fmt.Sprintf(" (%s of an interrupted download are kept and will be resumed)", setup.Bytes(uint64(partial)))
		}
		p.add("storage.capacity", setup.AreaStorage, setup.FindingUnknown, msg, facts)
	}
}

// remainingDownload is the bytes still to fetch of the file whose partial is
// kept, when its total is recorded, and the bytes held in staging.
func (p *preflight) remainingDownload() (remaining uint64, held int64) {
	if !p.modelOK {
		return 0, 0
	}
	stage := p.srcDir + ".staging"
	for _, rel := range sortedKeys(p.model.Files) {
		url := setup.ModelFileURL(p.model, rel)
		if st, ok := setup.InspectPartial(filepath.Join(stage, filepath.FromSlash(rel)), url, p.model.Files[rel]); ok && st.Total > st.Bytes {
			remaining += uint64(st.Total - st.Bytes)
		}
	}
	for _, e := range setup.Inspect(p.h, false).Models {
		if e.ID == p.model.ID {
			held = e.PartialBytes
		}
	}
	return remaining, held
}

// recordsDisk: a probe or a certification writes small records only.
func (p *preflight) recordsDisk(free uint64, known bool, facts map[string]any) {
	const need = 64 << 20
	facts["lower_bound_bytes"] = uint64(need)
	switch {
	case !known:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingUnknown, "free disk space cannot be read for this volume", facts)
	case free < need:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingBlocker, fmt.Sprintf("%s is free; the records of this operation need room to be written", setup.Bytes(free)), facts)
	default:
		p.add("storage.capacity", setup.AreaStorage, setup.FindingPass, fmt.Sprintf("%s is free; the operation writes small records only", setup.Bytes(free)), facts)
	}
}

// recipe resolves the recipe against the pinned source.
func (p *preflight) recipe() {
	if !p.modelOK {
		return
	}
	r, err := LookupRecipe(p.model.ID, p.req.Recipe)
	if err != nil {
		p.add("recipe.selected", setup.AreaRecipe, setup.FindingBlocker, err.Error(), map[string]any{"model": p.model.ID, "recipe": p.req.Recipe})
		return
	}
	facts := map[string]any{"recipe": r.Name, "sha256": r.SHA256(), "engine": r.Engine, "scheme": r.Scheme, "algorithm": r.Algorithm}
	p.add("recipe.selected", setup.AreaRecipe, setup.FindingPass, fmt.Sprintf("recipe %s (%s, %s) is defined for %s and parses", r.Name, r.Scheme, r.Algorithm, p.model.ID), facts)
	if _, err := weightsOf(r); err != nil {
		p.add("recipe.scheme", setup.AreaRecipe, setup.FindingBlocker, err.Error(), facts)
		return
	}
	if r.Engine != setup.OptimizerEngine {
		p.add("recipe.engine", setup.AreaRecipe, setup.FindingBlocker, fmt.Sprintf("recipe engine %q is not the pinned optimizer engine %q", r.Engine, setup.OptimizerEngine), facts)
		return
	}
	// Carried files must be pinned source files.
	var missing []string
	for _, rel := range r.Carry {
		if _, ok := p.model.Files[rel]; !ok {
			missing = append(missing, rel)
		}
	}
	for _, pm := range r.Preserved {
		if pm.Scope == home.ScopeCarried {
			if _, ok := p.model.Files[pm.Pattern]; !ok {
				missing = append(missing, pm.Pattern)
			}
		}
	}
	if len(missing) > 0 {
		p.add("recipe.carry", setup.AreaRecipe, setup.FindingBlocker, "the recipe carries files the pinned source does not have: "+strings.Join(missing, ", "), facts)
	} else {
		p.add("recipe.carry", setup.AreaRecipe, setup.FindingPass, fmt.Sprintf("all %d carried files are pinned source files", len(r.Carry)), facts)
	}
	p.modules(r)
}

// WeightMapFile is the source's checkpoint index: the pinned list of every
// tensor and the shard that holds it. It is the module inventory available
// before the model is loaded.
const WeightMapFile = "model.safetensors.index.json"

// SourceModules reads the module names of a source from its pinned weight
// map: every tensor "<module>.weight" belongs to the module "<module>". It
// lists modules that have a weight; whether one is a Linear is known only to
// the optimizer once it has loaded the model.
func SourceModules(dir string) ([]string, error) {
	b, err := os.ReadFile(filepath.Join(dir, WeightMapFile))
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("%s: %w", WeightMapFile, err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("%s lists no tensors", WeightMapFile)
	}
	seen := map[string]bool{}
	for name := range idx.WeightMap {
		if mod, ok := strings.CutSuffix(name, ".weight"); ok {
			seen[mod] = true
		}
	}
	return sortedKeys(seen), nil
}

// patternMatcher applies the optimizer's own selector semantics: a "re:"
// prefix is a regular expression matched from the start of the module name,
// anything else is an exact name.
func patternMatcher(pattern string) (func(string) bool, error) {
	if rest, ok := strings.CutPrefix(pattern, "re:"); ok {
		re, err := regexp.Compile(`^(?:` + rest + `)`)
		if err != nil {
			return nil, err
		}
		return re.MatchString, nil
	}
	return func(name string) bool { return name == pattern }, nil
}

// modules resolves the recipe's preserved selectors against the source's
// module inventory, as the optimizer will, and checks that no preserved module
// can end up quantized.
func (p *preflight) modules(r home.Recipe) {
	if !p.srcReady {
		p.add("recipe.modules", setup.AreaRecipe, setup.FindingUnknown, "the source is not available, so the recipe's selectors cannot be resolved against its modules", nil)
		return
	}
	mods, err := SourceModules(p.srcDir)
	if err != nil {
		p.add("recipe.modules", setup.AreaRecipe, setup.FindingUnknown, "the source's module inventory could not be read: "+err.Error(), nil)
		return
	}
	facts := map[string]any{"modules_with_weights": len(mods), "inventory": WeightMapFile}
	preserved := map[string]bool{}
	matched := map[string]int{}
	var unmatched, uncompiled []string
	for _, pm := range r.Preserved {
		if pm.Scope != home.ScopeBackbone {
			continue
		}
		match, err := patternMatcher(pm.Pattern)
		if err != nil {
			uncompiled = append(uncompiled, pm.Pattern)
			continue
		}
		for _, m := range mods {
			if match(m) {
				preserved[m] = true
				matched[pm.Pattern]++
			}
		}
		if matched[pm.Pattern] == 0 {
			unmatched = append(unmatched, pm.Pattern)
		}
	}
	facts["preserved_patterns"] = matched
	facts["preserved_modules"] = len(preserved)
	switch {
	case len(unmatched) > 0:
		facts["unmatched"] = unmatched
		p.add("recipe.modules", setup.AreaRecipe, setup.FindingBlocker,
			fmt.Sprintf("preserved selectors match no module of the pinned source: %s; the optimizer would refuse this recipe after loading the model", strings.Join(unmatched, ", ")), facts)
		return
	case len(uncompiled) > 0:
		facts["unevaluable"] = uncompiled
		p.add("recipe.modules", setup.AreaRecipe, setup.FindingUnknown,
			fmt.Sprintf("selectors %s are not evaluable here (the optimizer's matcher is Python re); they are resolved when the optimizer loads the model", strings.Join(uncompiled, ", ")), facts)
		return
	}
	// Quantization candidates are the weight modules no preserved selector
	// matched. A preserved module among them would mean the exclusion is not
	// applied.
	var targets []string
	for _, m := range mods {
		if !preserved[m] {
			targets = append(targets, m)
		}
	}
	facts["quantization_candidates"] = len(targets)
	for _, m := range targets {
		for _, pm := range r.Preserved {
			if pm.Scope != home.ScopeBackbone {
				continue
			}
			if match, err := patternMatcher(pm.Pattern); err == nil && match(m) {
				p.add("recipe.preserved", setup.AreaRecipe, setup.FindingBlocker, fmt.Sprintf("module %s matches preserved selector %s but is a quantization candidate", m, pm.Pattern), facts)
				return
			}
		}
	}
	if len(targets) == 0 {
		p.add("recipe.modules", setup.AreaRecipe, setup.FindingBlocker, "the recipe preserves every module of the source; nothing would be quantized", facts)
		return
	}
	names := sortedKeys(preserved)
	if len(names) > 8 {
		names = append(names[:8], "...")
	}
	p.add("recipe.modules", setup.AreaRecipe, setup.FindingPass,
		fmt.Sprintf("all preserved selectors resolve against the pinned source's %d weight modules (%d preserved, %d quantization candidates; Linear-ness is known to the optimizer only)",
			len(mods), len(preserved), len(targets)), facts)
	p.add("recipe.preserved", setup.AreaRecipe, setup.FindingPass,
		fmt.Sprintf("the %d preserved modules (%s) are excluded from quantization", len(preserved), strings.Join(names, ", ")), facts)
}

// certificationState reports what is already recorded for the variant. It is
// informational: certification records new evidence and never activates.
func (p *preflight) certificationState() {
	if !p.varOK {
		return
	}
	st := eval.ResolveCertification(p.h, p.variant)
	p.add("certification.state", setup.AreaIdentity, setup.FindingPass,
		fmt.Sprintf("variant %s is currently %s; certification records new evidence and never activates it", p.variant.ID, st.State),
		map[string]any{"variant": p.variant.ID, "state": st.State})
}

func weightBytes(dir string, rels []string) uint64 {
	var n uint64
	for _, rel := range rels {
		if !strings.HasSuffix(rel, ".safetensors") {
			continue
		}
		if fi, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel))); err == nil && fi.Mode().IsRegular() {
			n += uint64(fi.Size())
		}
	}
	return n
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
