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
	cmd := exec.CommandContext(ctx, p.python, append([]string{"-I", "-X", "utf8", p.script}, args...)...)
	subprocess.Configure(cmd)
	cmd.Env, cmd.Dir = p.env, p.dir
	cmd.Stdout, cmd.Stderr = events, log
	return cmd.Run()
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
	recipe, err := LookupRecipe(model.ID, req.Recipe)
	if err != nil {
		return Result{}, err
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
	runner := deps.Runner
	runtimeID := deps.OptimizerRuntime
	if runner == nil {
		rt, err := setup.EnsureOptimizer(h, log, obs)
		if err != nil {
			return Result{}, err
		}
		runner = processRunner{python: rt.Python, script: rt.Script, env: rt.Env(h), dir: h.Path("state")}
		runtimeID = rt.ID
	}
	now := deps.Now
	if now == nil {
		now = time.Now
	}
	src := home.SourceOf(model)
	opt := home.Optimizer{Engine: recipe.Engine, Version: setup.OptimizerEngineVersion, Runtime: runtimeID, Device: "cpu"}
	recipeSHA := recipe.SHA256()
	buildID := home.DeriveBuildID(src, model.Provider, opt, recipeSHA, nil)
	existing, err := findByBuild(h, model, buildID)
	if err != nil {
		return Result{}, err
	}
	if existing != nil && !req.Reproduce {
		return Result{Variant: *existing, Dir: h.VariantDir(model.ID, existing.ID), Existing: true}, nil
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

	fmt.Fprintf(log, "optimizing %s (%s@%s) with recipe %s, optimizer runtime %s\n", model.ID, model.Repo, model.Revision[:12], recipe.Name, runtimeID)
	obs.Phase(setup.PhaseStarting)
	engine, err := runOptimizer(ctx, runner, []string{"--source-dir", h.Path("models", filepath.FromSlash(setup.ModelDirName(model))),
		"--recipe", recipeFile, "--out", stage, "--device", opt.Device}, log, obs)
	if err != nil {
		return Result{}, err
	}
	if engine.Engine != recipe.Engine || engine.Version != setup.OptimizerEngineVersion {
		return Result{}, fmt.Errorf("the optimizer reported engine %s %s, the pinned optimizer is %s %s; the build is refused, not recorded under another engine",
			engine.Engine, engine.Version, recipe.Engine, setup.OptimizerEngineVersion)
	}
	opt.Versions = engine.Versions

	obs.Phase(setup.PhaseVerifying)
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
	v := home.VariantManifest{Source: src, Provider: model.Provider, Optimizer: opt, Recipe: recipe, Weights: weights, Files: files,
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
		// Same identity means same contract and same bytes: already published.
		return Result{Variant: v, Dir: final, Existing: true}, nil
	}
	// The manifest is written last: its presence marks a complete variant.
	obs.Phase(setup.PhasePublish)
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
}

// phaseOf maps the optimizer's own phases onto operation phases. importing is
// the optimizer starting up and stays in the starting phase.
var phaseOf = map[string]setup.Phase{
	"loading_source": setup.PhaseLoadingSource,
	"preparing":      setup.PhasePreparing,
	"quantizing":     setup.PhaseQuantizing,
	"serializing":    setup.PhaseSerializing,
	"verifying":      setup.PhaseVerifying,
}

// runOptimizer runs the optimizer and reports its phases. Progress inside a
// phase is indeterminate: the backend reports no measurable total, so none is
// shown.
func runOptimizer(ctx context.Context, r Runner, args []string, log io.Writer, obs *setup.Observer) (engineFacts, error) {
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
				facts = engineFacts{Engine: ev.Engine, Version: ev.Version, Versions: ev.Versions}
			case "phase":
				if ph, ok := phaseOf[ev.Phase]; ok {
					obs.Phase(ph)
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
		return facts, fmt.Errorf("optimizer %s: %s%s", fatal.Class, fatal.Message, stderr.context())
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
