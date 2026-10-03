package app

import (
	"context"
	"fmt"
	"io"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// RunTrials starts one tuning trial session as a controller operation. The
// whole session owns the accelerator exclusively, like every GPU
// model-engineering operation: serving is stopped before the worker starts and
// the residents it stopped come back once, after the last trial, whatever the
// outcome. The session's results are the Candidates and trial Evidence it
// recorded under the home; nothing is built, activated or certified.
func (c *Controller) RunTrials(p TrialParams) error {
	return c.async(SetupParams{Device: p.Device}, action{
		kind: OpTuningTrial, device: p.Device, model: p.Source, needDevice: true,
		target: "source " + p.Source + ", " + trialCount(len(p.Profiles)),
		run: func(root string, log io.Writer, obs *setup.Observer) error {
			c.mu.Lock()
			rt := c.rt
			c.mu.Unlock()
			// Everything that can be refused is refused before serving stops.
			if _, _, _, err := resolveTrialPlans(home.Home{Root: root}, p); err != nil {
				return err
			}
			return c.engineer(context.Background(), root, rt, []string{p.Device}, func() error {
				_, err := c.cfg.Maintenance.Trials(context.Background(), root, p, log, obs)
				return err
			})
		},
	})
}

func trialCount(n int) string {
	if n == 1 {
		return "1 profile"
	}
	return fmt.Sprintf("%d profiles", n)
}
