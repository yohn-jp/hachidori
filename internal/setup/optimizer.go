package setup

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/yohn-jp/hachidori/internal/home"
)

// OptimizerRuntime is a verified, materialized optimizer runtime: the separate
// environment that builds System One variants.
type OptimizerRuntime struct {
	ID       string
	Dir      string
	Python   string // absolute path of its private interpreter
	Script   string // absolute path of the delivered hachidori_optimizer.py
	Manifest home.RuntimeManifest
}

// Env is the complete environment for an optimizer process. The optimizer
// never needs the network: the source it transforms is already materialized
// and verified, so it is always offline.
func (o OptimizerRuntime) Env(h home.Home) []string { return h.Env(filepath.Dir(o.Python), true) }

// FindOptimizer returns the optimizer runtime of a concrete device (cpu or
// cuda) if it is already materialized,
// checking its manifest identity (not re-verifying its packages). It never
// materializes anything and never touches the network. The optimizer script is
// delivered by this build beside the runtime (DeliverOptimizer), so an
// optimizer runtime survives a script-only change.
func FindOptimizer(h home.Home, device string) (OptimizerRuntime, error) {
	spec, err := DesiredOptimizer(device)
	if err != nil {
		return OptimizerRuntime{}, err
	}
	name := RuntimeDirFor(h, spec)
	dir := h.Path("runtime", name)
	var m home.RuntimeManifest
	if err := home.ReadJSON(filepath.Join(dir, "manifest.json"), &m); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return OptimizerRuntime{}, fmt.Errorf("the optimizer runtime %s is not materialized (run `hachidori variant optimize --device %s`, which materializes it)", spec.ID(), device)
		}
		return OptimizerRuntime{}, fmt.Errorf("optimizer runtime %s: %w", name, err)
	}
	if err := CheckRuntimeManifest(name, m, spec); err != nil {
		return OptimizerRuntime{}, fmt.Errorf("optimizer runtime %s: %w", name, err)
	}
	d, err := DeliverOptimizer(h)
	if err != nil {
		return OptimizerRuntime{}, err
	}
	return OptimizerRuntime{ID: name, Dir: dir, Python: filepath.Join(dir, filepath.FromSlash(m.PythonRelPath)), Manifest: m, Script: d.Path}, nil
}

// EnsureOptimizer materializes the optimizer runtime of a concrete device (cpu
// or cuda) if it is not present,
// through the same staged, verified and atomically published path as the
// serving runtime, and returns it. This is the one place the optimizer's
// network access happens (the private uv installs the locked packages);
// optimization itself is offline. An existing runtime is verified and reused,
// never modified in place.
func EnsureOptimizer(h home.Home, device string, log io.Writer, obs *Observer) (OptimizerRuntime, error) {
	obs.phase(PhasePreparing)
	spec, err := DesiredOptimizer(device)
	if err != nil {
		return OptimizerRuntime{}, err
	}
	if err := h.Ensure(); err != nil {
		return OptimizerRuntime{}, err
	}
	final := h.Path("runtime", RuntimeDirFor(h, spec))
	if _, err := os.Stat(final); err == nil {
		obs.step(StepVerify, "optimizer runtime "+spec.ID())
		if err := verifyPublished(h, final, spec); err != nil {
			return OptimizerRuntime{}, fmt.Errorf("optimizer runtime %s exists but failed verification; it is never modified in place (remove %s to rematerialize): %w", spec.ID(), final, err)
		}
		return FindOptimizer(h, device)
	} else if !errors.Is(err, os.ErrNotExist) {
		return OptimizerRuntime{}, err
	}
	uv, err := ensureUV(h, log, obs)
	if err != nil {
		return OptimizerRuntime{}, fmt.Errorf("private uv: %w", err)
	}
	obs.phase(PhaseRuntime)
	stage, err := materializeRuntime(h, uv, spec, log, obs)
	if err != nil {
		return OptimizerRuntime{}, fmt.Errorf("optimizer runtime %s: %w", spec.ID(), err)
	}
	obs.step(StepPublish, "optimizer runtime "+spec.ID())
	if err := os.Rename(stage, final); err != nil {
		return OptimizerRuntime{}, fmt.Errorf("optimizer runtime %s: publish: %w", spec.ID(), err)
	}
	obs.step(StepVerify, "published optimizer runtime "+spec.ID())
	if err := verifyPublished(h, final, spec); err != nil {
		return OptimizerRuntime{}, fmt.Errorf("optimizer runtime %s: published runtime failed verification: %w", spec.ID(), err)
	}
	fmt.Fprintf(log, "optimizer runtime %s published\n", spec.ID())
	return FindOptimizer(h, device)
}
