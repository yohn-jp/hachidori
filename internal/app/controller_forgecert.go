package app

import (
	"context"
	"io"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// CertifyVariant starts a self-contained Forge certification of p.Variant
// (RunForgeCertification) as one controller action. The action is the
// operator-visible state machine: Snapshot.Operation reports each phase when it
// is entered (resolving, preflight, materializing, probe, reference_run,
// candidate_run, aligning, certifying, persisting) and the finished operation is
// completed or carries the failure of the phase it failed in. A front end only
// projects it.
//
// The whole certification is one model-engineering transaction: only the
// Hachidori-owned residents that occupy the accelerator are stopped, before the
// probe, and they stay down through both runs; they are restored (and waited
// for) once, when the operation ends, on success, failure and cancellation
// alike. The activation record, desired residents, routing policy and default
// model are never written; the verdict, accepted or rejected, is evidence only.
func (c *Controller) CertifyVariant(p ForgeCertifyParams) error {
	return c.async(SetupParams{Device: p.Device}, action{kind: OpForgeCertify, device: p.Device, target: setup.KindVariant + " " + p.Variant, needDevice: true,
		plan:  certifyPlan(p.Materialize),
		forge: &ForgeFailure{Kind: OpForgeCertify, Variant: p.Variant, Device: p.Device},
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			c.mu.Lock()
			rt := c.rt
			c.mu.Unlock()
			m := c.cfg.Maintenance
			ctx := context.Background()
			lease := func(device string, fn func() error) error {
				_, _, err := c.leased(ctx, root, rt, device, nil, fn)
				return err
			}
			deps := ForgeCertifyDeps{
				Preflight: m.Preflight,
				Materialize: func(_ context.Context, root, device, model string, log io.Writer, obs *setup.Observer) error {
					return m.Materialize(root, device, model, log, obs)
				},
				Probe: func(ctx context.Context, root string, pp ProbeParams, log io.Writer, obs *setup.Observer) (rec ProbeRecord, err error) {
					err = lease(pp.Device, func() (e error) { rec, e = m.Probe(ctx, root, pp, log, obs); return })
					return rec, err
				},
				Execute: func(ctx context.Context, root string, ep ExecuteParams, log io.Writer) (res ExecutionResult, err error) {
					quiesced, restored, err := c.leased(ctx, root, rt, ep.Target.Device, nil, func() (e error) { res, e = m.Execute(ctx, root, ep, log); return })
					res.Quiesced, res.Restored = quiesced, restored
					return res, err
				},
			}
			// The certification is one model-engineering transaction: it owns
			// the accelerator from before the probe until the last run is
			// done, and the probe and both runs nest in that ownership.
			reference := p.ReferenceDevice
			if reference == "" {
				reference = DefaultReferenceDevice
			}
			return c.engineer(ctx, root, rt, []string{p.Device, reference}, func() error {
				_, err := RunForgeCertification(ctx, home.Home{Root: root}, p, deps, log, obs)
				return err
			})
		}})
}

// certifyPlan is the phases a certification is expected to enter, in order;
// materializing is planned only when the operation may materialize a runtime
// (and is then entered only if one is missing).
func certifyPlan(materialize bool) []string {
	p := plan(OpForgeCertify, "")
	if !materialize {
		return p
	}
	return append(p[:2:2], append([]string{string(CertPhaseMaterializing)}, p[2:]...)...)
}
