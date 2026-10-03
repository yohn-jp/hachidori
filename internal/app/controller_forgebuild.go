package app

import (
	"context"
	"io"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// BuildAndEvaluate starts one controller-owned Forge transaction. The build
// and certification are visible through the same operation and diagnostic. The
// transaction owns the accelerator exclusively from before its first GPU phase
// to its end (engineering.go): its probe and both runs nest in that ownership
// and none of them restores serving on its own.
func (c *Controller) BuildAndEvaluate(p ForgeBuildEvaluateParams) error {
	var result ForgeBuildEvaluateResult
	return c.async(SetupParams{Device: p.Device}, action{
		kind: OpForgeBuildEvaluate, device: p.Device, model: p.Source,
		target: "source " + p.Source + " recipe " + p.Optimization.Recipe,
		plan:   forgeBuildEvaluatePlan(p.Materialize),
		forge:  &ForgeFailure{Kind: OpForgeBuildEvaluate, Model: p.Source, Recipe: p.Optimization.Recipe, Device: p.Device},
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			c.mu.Lock()
			rt := c.rt
			c.mu.Unlock()
			m := c.cfg.Maintenance
			certify := ForgeCertifyDeps{
				Preflight: m.Preflight,
				Materialize: func(_ context.Context, root, device, model string, log io.Writer, obs *setup.Observer) error {
					return m.Materialize(root, device, model, log, obs)
				},
				Probe: func(ctx context.Context, root string, p ProbeParams, log io.Writer, obs *setup.Observer) (rec ProbeRecord, err error) {
					var probeErr error
					_, _, err = c.leased(ctx, root, rt, p.Device, nil, func() error {
						rec, probeErr = m.Probe(ctx, root, p, log, obs)
						return probeErr
					})
					return rec, err
				},
				Execute: func(ctx context.Context, root string, p ExecuteParams, log io.Writer) (res ExecutionResult, err error) {
					quiesced, restored, err := c.leased(ctx, root, rt, p.Target.Device, nil, func() error {
						res, err = m.Execute(ctx, root, p, log)
						return err
					})
					res.Quiesced, res.Restored = quiesced, restored
					return res, err
				},
			}
			deps := ForgeBuildEvaluateDeps{
				Build:   m.Build,
				Certify: certify,
				OnResolution: func(r ForgeBuildEvaluateResolution) {
					c.mu.Lock()
					if c.op != nil && c.op.Kind == OpForgeBuildEvaluate {
						c.op.Device = r.CandidateDevice.Value
						resolved := r
						c.op.ForgeResolution = &resolved
						c.notify()
					}
					c.mu.Unlock()
				},
			}
			// The whole composed transaction owns the accelerator: serving
			// is stopped before its first GPU phase and stays down through
			// preflight, build, probe and both runs, then comes back once.
			// The devices are those of the resolved plan; a request that
			// does not resolve fails in RunForgeBuildEvaluate without ever
			// touching serving.
			ctx := context.Background()
			h := home.Home{Root: root}
			var devices []string
			if plan, perr := resolveForgeBuildPlan(h, p); perr == nil {
				devices = []string{plan.resolution.CandidateDevice.Value, plan.resolution.ReferenceDevice.Value}
			}
			return c.engineer(ctx, root, rt, devices, func() (err error) {
				result, err = RunForgeBuildEvaluate(ctx, h, p, deps, log, obs)
				return err
			})
		},
		after: func(err error) {
			if c.op == nil || c.op.Kind != OpForgeBuildEvaluate || result.Resolution.Source == "" {
				return
			}
			c.op.Device = result.Resolution.CandidateDevice.Value
			resolved := result.Resolution
			c.op.ForgeResolution = &resolved
		},
		enrich: func(f *ForgeFailure) {
			if result.Resolution.Source == "" {
				return
			}
			f.Device = result.Resolution.CandidateDevice.Value
			f.Variant = result.Variant.ID
		},
	})
}

func forgeBuildEvaluatePlan(materialize bool) []string {
	phases := []string{string(ForgeBuildPhaseResolve), string(ForgeBuildPhasePreflight)}
	if materialize {
		phases = append(phases, string(ForgeBuildPhaseProvision))
	}
	phases = append(phases, string(ForgeBuildPhaseBuild), string(CertPhaseResolving), string(CertPhasePreflight))
	if materialize {
		phases = append(phases, string(ForgeBuildPhaseProvision), string(ForgeBuildPhaseProvision), string(ForgeBuildPhaseProvision))
	}
	return append(phases, string(CertPhaseProbe), string(CertPhaseReference), string(CertPhaseCandidate), string(CertPhaseAligning), string(CertPhaseCertifying), string(CertPhasePersisting))
}
