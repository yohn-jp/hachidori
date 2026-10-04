package optimize

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/subprocess"
)

// Request is one optimization: a catalog model and the name of a canonical
// recipe for it. Both are catalog identities; there is no repository,
// revision, path or free-form recipe input.
type Request struct {
	Model  string // catalog model ID
	Recipe string // recipe name
	// Device is the optimizer device: "cpu" or "cuda". Empty is the cpu. There
	// is no automatic choice: the device is named, it selects the optimizer
	// runtime flavor, it is recorded in the variant, and a build on cuda that
	// cannot use the accelerator fails instead of running on the cpu.
	Device string
	// CompiledRecipe is set only when Tuning names a stored semantic profile;
	// it is the canonical recipe produced by the tuning compiler.
	CompiledRecipe *home.Recipe
	Tuning         *home.TuningProvenance
	// Plan is the resolved layer-wise policy of every group. It is set exactly
	// when Tuning is layer-wise (schema 2): the build applies the plan and
	// refuses to publish a variant whose applied transformations differ from it.
	Plan *home.TuningPlan
	// Reproduce rebuilds a contract that already has a published variant and
	// compares the new artifacts with it, publishing nothing.
	Reproduce bool
}

// Result describes the outcome of Build.
type Result struct {
	Variant home.VariantManifest
	Dir     string
	// Existing is set when a variant of this build contract was already
	// published and no rebuild was requested: nothing ran.
	Existing bool
	// Reproduced is set by Reproduce when the rebuilt artifacts are
	// byte-identical to the published variant's.
	Reproduced bool
}

// MismatchError is returned by a reproduction whose artifacts differ from the
// published variant's. The difference is surfaced; nothing is published.
type MismatchError struct {
	Variant string
	Files   []string
}

func (e *MismatchError) Error() string {
	return fmt.Sprintf("rebuilding the contract of variant %s produced different artifacts (%s); the published variant is unchanged and the new build was discarded",
		e.Variant, strings.Join(e.Files, ", "))
}

// FatalError is a failure the optimizer itself reported (a fatal protocol
// event): the class names the cause, for example cuda_unavailable,
// cuda_out_of_memory or quantize. A failed build is never retried on another
// device.
type FatalError struct {
	Class   string
	Message string
	Stderr  string // tail of the optimizer's stderr, with its own leading newline
}

func (e *FatalError) Error() string {
	return fmt.Sprintf("optimizer %s: %s%s", e.Class, e.Message, e.Stderr)
}

// CUDAUnavailableError is returned when a build resolved to cuda but the
// pinned CUDA optimizer runtime does not see a usable accelerator. Hachidori
// never continues such a build on the cpu.
type CUDAUnavailableError struct {
	Runtime string
	Reason  string
	Facts   setup.AcceleratorFacts
}

func (e *CUDAUnavailableError) Error() string {
	return fmt.Sprintf("optimizer device cuda is unavailable in runtime %s: %s; the build is not continued on the cpu (use --device cpu for a cpu build)", e.Runtime, e.Reason)
}

// RequireCUDA checks the accelerator facts observed with the private
// interpreter of the cuda optimizer runtime against the runtime's pinned torch:
// torch must be the pinned CUDA build, import cleanly and report a usable CUDA
// device with a name and a VRAM total. It returns a *CUDAUnavailableError
// otherwise.
func RequireCUDA(runtime string, spec home.RuntimeSpec, f setup.AcceleratorFacts) error {
	fail := func(format string, a ...any) error {
		return &CUDAUnavailableError{Runtime: runtime, Reason: fmt.Sprintf(format, a...), Facts: f}
	}
	switch {
	case f.Error != "":
		return fail("torch could not report CUDA: %s", f.Error)
	case f.Torch != spec.Torch:
		return fail("torch %q is not the pinned %s", f.Torch, spec.Torch)
	case f.TorchCUDA == "":
		return fail("torch %s is not a CUDA build", f.Torch)
	case !f.CUDAAvailable:
		return fail("torch %s (CUDA %s) reports no usable CUDA device", f.Torch, f.TorchCUDA)
	case f.DeviceName == "" || f.VRAMTotal == 0:
		return fail("torch %s (CUDA %s) reported no device name or VRAM", f.Torch, f.TorchCUDA)
	}
	return nil
}

// Runner starts the optimizer process. Production uses the optimizer
// runtime's private interpreter; tests substitute a fake optimizer.
type Runner interface {
	// Run starts the optimizer for the given arguments, copies its stdout
	// (protocol events only) into events and its stderr into log, and returns
	// when it exits.
	Run(ctx context.Context, args []string, events io.Writer, log io.Writer) error
}

// processRunner runs the real optimizer: the private interpreter of the
// optimizer runtime, isolated and offline.
type processRunner struct {
	python, script string
	env            []string
	dir            string
}

func (p processRunner) Run(ctx context.Context, args []string, events, log io.Writer) error {
	cmd := exec.CommandContext(ctx, p.python, setup.PythonArgs(append([]string{p.script}, args...)...)...)
	subprocess.Configure(cmd)
	cmd.Env, cmd.Dir = p.env, p.dir
	cmd.Stdout, cmd.Stderr = events, log
	return cmd.Run()
}

// NewProcessRunner is the Runner that starts the optimizer script with the
// given private interpreter, isolated, in env and dir. Build uses it for the
// optimizer runtime; it is exported so that the end-to-end test can run the
// real optimizer from any prepared environment.
func NewProcessRunner(python, script string, env []string, dir string) Runner {
	return processRunner{python: python, script: script, env: env, dir: dir}
}

// Deps are the replaceable parts of Build.
type Deps struct {
	// Runner overrides the optimizer process; nil starts the real one in the
	// optimizer runtime, materializing it first if needed.
	Runner Runner
	// Now overrides the clock (tests).
	Now func() time.Time
	// OptimizerRuntime is the runtime identity recorded when Runner is set.
	OptimizerRuntime string
	// Preflight overrides the observations of the preflight Build runs before
	// any expensive work (host disk and RAM, and the accelerator probe, which
	// Build also uses to check a cuda optimizer runtime after it is materialized).
	Preflight PreflightDeps
}

// Build builds the variant of req from the materialized, verified catalog
// model and publishes it. It follows one path:
//
//	verify the pinned source -> optimizer runtime -> staging -> run the
//	optimizer -> digest every output -> verify preserved modules and scheme
//	-> write the manifest -> atomic publish
//
// A failed, cancelled or interrupted build removes its staging directory and
// leaves nothing selectable: a variant exists only once its manifest is
// published, and the manifest is the last file written. The source model is
// only read.
func Build(ctx context.Context, h home.Home, req Request, deps Deps, log io.Writer, obs *setup.Observer) (Result, error) {
	model, err := setup.LookupModel(req.Model)
	if err != nil {
		return Result{}, err
	}
	if !setup.SupportsVariants(model) {
		return Result{}, fmt.Errorf("model %s has no variants (only System One models are optimized)", model.ID)
	}
	canonical, err := LookupRecipe(model.ID, req.Recipe)
	if err != nil {
		return Result{}, err
	}
	// The optimizer device is concrete from here on: it selects the runtime,
	// is passed to the optimizer, and is recorded. It is never changed again.
	device, err := setup.ResolveOptimizerDevice(req.Device)
	if err != nil {
		return Result{}, err
	}
	src := home.SourceOf(model)
	recipe := canonical
	if req.Tuning == nil {
		if req.CompiledRecipe != nil || req.Plan != nil {
			return Result{}, errors.New("a compiled tuning recipe or plan requires exact tuning provenance")
		}
	} else {
		if req.CompiledRecipe == nil {
			return Result{}, errors.New("tuning provenance requires the canonical recipe produced by its compiler")
		}
		if err := req.Tuning.Validate(src); err != nil {
			return Result{}, fmt.Errorf("tuning profile cannot build this source: %w", err)
		}
		compiled := *req.CompiledRecipe
		if compiled.Name != canonical.Name || compiled.Schema != canonical.Schema || compiled.Engine != canonical.Engine ||
			compiled.Scheme != canonical.Scheme || compiled.Algorithm != canonical.Algorithm ||
			!equalStrings(compiled.Targets, canonical.Targets) || !equalStrings(compiled.Carry, canonical.Carry) ||
			!includesPreserved(compiled.Preserved, canonical.Preserved) {
			return Result{}, errors.New("tuning compiler recipe changes the canonical optimizer contract")
		}
		if err := compiled.Validate(); err != nil {
			return Result{}, fmt.Errorf("compiled tuning recipe: %w", err)
		}
		if err := checkPlan(req, compiled); err != nil {
			return Result{}, err
		}
		recipe = compiled
	}
	weights, err := weightsOf(recipe)
	if err != nil {
		return Result{}, err
	}
	if err := h.Ensure(); err != nil {
		return Result{}, err
	}
	// The pinned source must be intact before it is transformed: every file
	// against its digest.
	if err := setup.Verify(h, setup.KindModel, model.ID, obs); err != nil {
		return Result{}, fmt.Errorf("source model %s: %w (materialize it first with `hachidori setup --model %s`)", model.ID, err, model.ID)
	}
	// Refuse an obviously impossible build before the optimizer runtime is
	// materialized or the multi-GB source is loaded. The source was just
	// verified, so its digests are not hashed again.
	if _, err := RequirePreflight(ctx, h, PreflightRequest{Kind: setup.PreflightOptimize, Model: model.ID, Recipe: recipe.Name, Device: device, Verified: true}, deps.Preflight, obs); err != nil {
		return Result{}, err
	}
	runner := deps.Runner
	runtimeID := deps.OptimizerRuntime
	if runner == nil {
		rt, err := setup.EnsureOptimizer(h, device, log, obs)
		if err != nil {
			return Result{}, err
		}
		if device == home.OptimizerDeviceCUDA {
			// The pinned CUDA runtime is proven to see the accelerator with
			// its own interpreter before anything is loaded; a build that
			// resolved to cuda never continues without it.
			if err := requireCUDARuntime(ctx, h, rt, deps.Preflight.Accelerator, log); err != nil {
				return Result{}, err
			}
		}
		runner = processRunner{python: rt.Python, script: rt.Script, env: rt.Env(h), dir: h.Path("state")}
		runtimeID = rt.ID
	}
	now := deps.Now
	if now == nil {
		now = time.Now
	}
	opt := home.Optimizer{Engine: recipe.Engine, Version: setup.OptimizerEngineVersion, Runtime: runtimeID, Device: device}
	recipeSHA := recipe.SHA256()
	buildID := home.DeriveBuildIDWithTuning(src, model.Provider, opt, recipeSHA, nil, req.Tuning)
	existing, err := findByBuild(h, model, buildID)
	if err != nil {
		return Result{}, err
	}
	if existing != nil && !req.Reproduce {
		return Result{Variant: *existing, Dir: h.VariantDir(model.ID, existing.ID), Existing: true}, nil
	}
	if existing == nil && req.Reproduce {
		// A reproduction compares with a published variant and publishes
		// nothing; without one there is nothing to compare with.
		return Result{}, fmt.Errorf("%s with recipe %s has no published variant to reproduce; build it without -reproduce first", model.ID, recipe.Name)
	}

	// Every phase is reported once, when it is first entered.
	entered := map[setup.Phase]bool{}
	enter := func(p setup.Phase) {
		if !entered[p] {
			entered[p] = true
			obs.Phase(p)
		}
	}
	stageParent := h.VariantsDir(model.ID)
	if err := os.MkdirAll(stageParent, 0o755); err != nil {
		return Result{}, err
	}
	stage := filepath.Join(stageParent, ".staging-"+buildID[:16])
	recipeFile := h.Path("cache", "tmp", "recipe-"+buildID[:16]+".json")
	cleanup := func() {
		_ = os.RemoveAll(stage)
		_ = os.Remove(recipeFile)
	}
	cleanup() // a staging directory left by an interrupted build is never reused
	defer cleanup()
	if err := os.WriteFile(recipeFile, recipe.Canonical(), 0o644); err != nil {
		return Result{}, err
	}

	fmt.Fprintf(log, "optimizing %s (%s@%s) with recipe %s, optimizer runtime %s, optimizer device %s\n", model.ID, model.Repo, model.Revision[:12], recipe.Name, runtimeID, device)
	enter(setup.PhaseStarting)
	engine, err := runOptimizer(ctx, runner, []string{"--source-dir", h.Path("models", filepath.FromSlash(setup.ModelDirName(model))),
		"--recipe", recipeFile, "--out", stage, "--device", opt.Device}, log, enter, obs)
	if err != nil {
		return Result{}, err
	}
	if engine.Engine != recipe.Engine || engine.Version != setup.OptimizerEngineVersion {
		return Result{}, fmt.Errorf("the optimizer reported engine %s %s, the pinned optimizer is %s %s; the build is refused, not recorded under another engine",
			engine.Engine, engine.Version, recipe.Engine, setup.OptimizerEngineVersion)
	}
	if engine.Device != opt.Device {
		return Result{}, fmt.Errorf("the optimizer reported device %q, the build resolved to %q; the build is refused, not recorded under another device",
			engine.Device, opt.Device)
	}
	opt.Versions = engine.Versions

	enter(setup.PhaseVerifying)
	if req.Plan != nil {
		// What the optimizer did to every group is verified against what the plan
		// resolved before anything is digested: a mismatch is never published.
		if err := writeTuningEvidence(stage, model, *req.Plan); err != nil {
			return Result{}, err
		}
	}
	files, err := setup.DigestTree(stage, nil, obs)
	if err != nil {
		return Result{}, fmt.Errorf("digesting the optimizer output: %w", err)
	}
	if _, ok := files[setup.OptimizerReportFile]; !ok {
		return Result{}, errors.New("the optimizer wrote no report")
	}
	for _, rel := range recipe.Carry {
		if _, ok := files[rel]; !ok {
			return Result{}, fmt.Errorf("the optimizer did not carry %s over from the source", rel)
		}
	}
	var tuning *home.TuningProvenance
	if req.Tuning != nil {
		copy := *req.Tuning
		tuning = &copy
	}
	v := home.VariantManifest{Source: src, Provider: model.Provider, Optimizer: opt, Recipe: recipe, Tuning: tuning, Weights: weights, Files: files,
		Creation: home.Creation{CreatedAt: now().UTC().Format(time.RFC3339), Platform: runtime.GOOS + "/" + runtime.GOARCH,
			Command: fmt.Sprintf("hachidori variant optimize --model %s --recipe %s", model.ID, recipe.Name)}}
	v.Seal()
	if err := v.Validate(); err != nil {
		return Result{}, fmt.Errorf("the built variant is not a valid variant: %w", err)
	}
	if v.BuildID != buildID {
		return Result{}, errors.New("the built variant's build id differs from the contract it was built for")
	}
	obs.Step(setup.StepVerify, "scheme, preserved modules and carried files")
	if err := setup.CheckPreserved(stage, model, v); err != nil {
		return Result{}, err
	}

	if existing != nil {
		// Reproduction: compare, publish nothing, whatever the outcome.
		if diff := diffFiles(existing.Files, v.Files); len(diff) > 0 {
			return Result{Variant: *existing, Dir: h.VariantDir(model.ID, existing.ID)}, &MismatchError{Variant: existing.ID, Files: diff}
		}
		return Result{Variant: *existing, Dir: h.VariantDir(model.ID, existing.ID), Existing: true, Reproduced: true}, nil
	}
	final := h.VariantDir(model.ID, v.ID)
	if _, err := os.Stat(final); err == nil {
		// Same identity means same contract and same bytes: already published,
		// but only if the directory holds that variant's published manifest.
		// Without it the directory is not a variant and is never reported as one.
		if pub, rerr := home.ReadVariant(final); rerr != nil || pub.ID != v.ID {
			if rerr == nil {
				rerr = fmt.Errorf("its manifest names variant %s", pub.ID)
			}
			return Result{}, fmt.Errorf("variant directory %s exists but is not a published variant (%v); remove it with `hachidori variant remove %s` and build again", final, rerr, v.ID)
		}
		return Result{Variant: v, Dir: final, Existing: true}, nil
	}
	// The manifest is written last: its presence marks a complete variant.
	enter(setup.PhasePublish)
	obs.Step(setup.StepPublish, "variant "+v.ID)
	if err := writeManifest(stage, v); err != nil {
		return Result{}, err
	}
	if err := os.Rename(stage, final); err != nil {
		return Result{}, fmt.Errorf("publishing variant %s: %w", v.ID, err)
	}
	if _, err := home.ReadVariant(final); err != nil {
		return Result{}, fmt.Errorf("published variant %s failed to read back: %w", v.ID, err)
	}
	fmt.Fprintf(log, "variant %s published\n", v.ID)
	return Result{Variant: v, Dir: final}, nil
}

// requireCUDARuntime observes the CUDA of the materialized cuda optimizer
// runtime with its private interpreter (the accelerator probe the preflight
// uses) and refuses the build unless it is usable. The device name, CUDA
// version and VRAM it saw are logged with the build.
func requireCUDARuntime(ctx context.Context, h home.Home, rt setup.OptimizerRuntime,
	probe func(context.Context, home.Home, string) (setup.AcceleratorFacts, error), log io.Writer) error {
	if probe == nil {
		probe = setup.ProbeAccelerator
	}
	f, err := probe(ctx, h, rt.Python)
	if err != nil {
		return &CUDAUnavailableError{Runtime: rt.ID, Reason: "the CUDA device could not be observed: " + err.Error()}
	}
	if err := RequireCUDA(rt.ID, rt.Manifest.Spec, f); err != nil {
		return err
	}
	fmt.Fprintf(log, "optimizer cuda: torch %s (CUDA %s), %s, %s VRAM total, %s free\n", f.Torch, f.TorchCUDA, f.DeviceName, setup.Bytes(f.VRAMTotal), setup.Bytes(f.VRAMFree))
	return nil
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func includesPreserved(actual, required []home.PreservedModule) bool {
	for _, want := range required {
		found := false
		for _, got := range actual {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func writeManifest(dir string, v home.VariantManifest) error {
	return home.WriteFileAtomic(filepath.Join(dir, home.VariantManifestFile), v.Canonical(), 0o644)
}

// findByBuild returns the published variant of model with the build contract
// buildID, if any.
func findByBuild(h home.Home, model home.ModelManifest, buildID string) (*home.VariantManifest, error) {
	entries, err := os.ReadDir(h.VariantsDir(model.ID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var found *home.VariantManifest
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		v, err := home.ReadVariant(h.VariantDir(model.ID, e.Name()))
		if err != nil || v.BuildID != buildID {
			continue
		}
		if found == nil || v.ID < found.ID {
			vv := v
			found = &vv
		}
	}
	return found, nil
}

func diffFiles(a, b map[string]string) []string {
	var out []string
	for rel, d := range a {
		if b[rel] != d {
			out = append(out, rel)
		}
	}
	for rel := range b {
		if _, ok := a[rel]; !ok {
			out = append(out, rel)
		}
	}
	sort.Strings(out)
	return out
}

// engineFacts are what the optimizer reports about the backend it ran.
type engineFacts struct {
	Engine   string
	Version  string
	Versions map[string]string
	// Device is the device the backend will run on, as the optimizer resolved
	// it before loading anything.
	Device string
}

type event struct {
	Event    string            `json:"event"`
	Phase    string            `json:"phase"`
	Detail   string            `json:"detail"`
	Class    string            `json:"class"`
	Message  string            `json:"message"`
	Engine   string            `json:"engine"`
	Version  string            `json:"version"`
	Versions map[string]string `json:"versions"`
	Device   string            `json:"device"`
}

// phaseOf maps the optimizer's own phases onto operation phases. importing is
// the optimizer starting up and stays in the starting phase.
var phaseOf = map[string]setup.Phase{
	"loading_source": setup.PhaseLoadingSource,
	"preparing":      setup.PhaseResolving,
	"quantizing":     setup.PhaseQuantizing,
	"serializing":    setup.PhaseSerializing,
	"verifying":      setup.PhaseVerifying,
}

// runOptimizer runs the optimizer and reports its phases. Progress inside a
// phase is indeterminate: the backend reports no measurable total, so none is
// shown.
func runOptimizer(ctx context.Context, r Runner, args []string, log io.Writer, enter func(setup.Phase), obs *setup.Observer) (engineFacts, error) {
	pr, pw := io.Pipe()
	var (
		mu       sync.Mutex
		facts    engineFacts
		fatal    *event
		done     bool
		tail     []string
		scanDone = make(chan struct{})
	)
	go func() {
		defer close(scanDone)
		sc := bufio.NewScanner(pr)
		sc.Buffer(make([]byte, 64<<10), 4<<20)
		for sc.Scan() {
			var ev event
			if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
				mu.Lock()
				tail = append(tail, "non-protocol output: "+truncate(sc.Text(), 200))
				mu.Unlock()
				continue
			}
			mu.Lock()
			switch ev.Event {
			case "engine":
				facts.Engine, facts.Version, facts.Versions = ev.Engine, ev.Version, ev.Versions
			case "device":
				facts.Device = ev.Device
			case "phase":
				if ph, ok := phaseOf[ev.Phase]; ok {
					enter(ph)
				}
				obs.Step(stepFor(ev.Phase), ev.Detail)
			case "fatal":
				e := ev
				fatal = &e
			case "done":
				done = true
			}
			mu.Unlock()
		}
		_, _ = io.Copy(io.Discard, pr)
	}()
	var stderr tailWriter
	err := r.Run(ctx, args, pw, io.MultiWriter(log, &stderr))
	_ = pw.Close()
	<-scanDone
	mu.Lock()
	defer mu.Unlock()
	switch {
	case ctx.Err() != nil:
		return facts, fmt.Errorf("optimization cancelled: %w", ctx.Err())
	case fatal != nil:
		return facts, &FatalError{Class: fatal.Class, Message: fatal.Message, Stderr: stderr.context()}
	case err != nil:
		return facts, fmt.Errorf("optimizer exited: %v%s", err, stderr.context())
	case !done:
		return facts, fmt.Errorf("optimizer ended without completing%s", stderr.context())
	case facts.Engine == "":
		return facts, errors.New("optimizer reported no engine")
	}
	return facts, nil
}

func stepFor(phase string) setup.Step {
	switch phase {
	case "verifying":
		return setup.StepVerify
	case "serializing":
		return setup.StepPublish
	}
	return setup.StepMaterialize
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// tailWriter keeps the last lines of the optimizer's stderr for failure
// context.
type tailWriter struct {
	mu    sync.Mutex
	lines []string
	part  string
}

func (t *tailWriter) Write(b []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.part += string(b)
	for {
		i := strings.IndexByte(t.part, '\n')
		if i < 0 {
			break
		}
		t.lines = append(t.lines, truncate(strings.TrimRight(t.part[:i], "\r"), 300))
		t.part = t.part[i+1:]
		if len(t.lines) > 12 {
			t.lines = t.lines[len(t.lines)-12:]
		}
	}
	return len(b), nil
}

func (t *tailWriter) context() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.lines) == 0 {
		return ""
	}
	return "\n  optimizer stderr (tail):\n    " + strings.Join(t.lines, "\n    ")
}
