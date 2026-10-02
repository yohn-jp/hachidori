package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// OpExecute is a Forge execution session: exact-target maintenance work that
// shares the accelerator with the serving residents (see Controller.Execute).
const OpExecute = "execute"

// defaultRestoreTimeout bounds how long restoring the serving residents waits
// for them to be READY again; a Clef load is the slow case.
const defaultRestoreTimeout = 15 * time.Minute

// servingState is what an execution session must leave exactly as it found
// it. The resident set itself is restored by the lease; this is the part no
// session may change at all.
type servingState struct {
	activation []byte   // state/active-runtime.json; nil when there is none
	desired    []string // the desired additional residents
	routing    string   // the bound routing policy
}

func (c *Controller) servingState(h home.Home, rt Runtime) servingState {
	var s servingState
	s.activation, _ = os.ReadFile(h.Path("state", "active-runtime.json"))
	if c.cfg.Residents != nil {
		s.desired = slices.Clone(c.cfg.Residents())
	}
	if r, ok := rt.(interface{ RoutingStatus() *route.Status }); ok {
		if st := r.RoutingStatus(); st != nil {
			b, _ := json.Marshal(st.Policy)
			s.routing = string(b)
		}
	}
	return s
}

// changed names what differs between two states, or returns nil.
func (s servingState) changed(then servingState) error {
	switch {
	case !bytes.Equal(s.activation, then.activation):
		return errors.New("the activation record changed during the session")
	case !slices.Equal(s.desired, then.desired):
		return errors.New("the desired resident selection changed during the session")
	case s.routing != then.routing:
		return errors.New("the routing policy changed during the session")
	}
	return nil
}

// lease is the set of Hachidori-owned residents a session stopped so that its
// worker could have the accelerator, and how to bring them back.
type lease struct {
	rt       Runtime
	models   []string // residents of a resident set, in the order they were stopped
	whole    bool     // a single worker binding was stopped
	restored []string
}

// occupies reports whether a runtime on device holds the accelerator.
func occupies(device string) bool { return device != "cpu" }

// quiesce stops exactly the running Hachidori-owned residents that occupy the
// accelerator the target needs, through the resident lifecycle authority. A
// CPU target, a runtime that is not running and residents on the CPU are left
// alone; no process Hachidori does not own is ever looked at. The returned
// lease records what was stopped even when quiescing fails part way.
func (c *Controller) quiesce(rt Runtime, device string) (*lease, error) {
	l := &lease{rt: rt}
	if rt == nil || !rt.Running() || !occupies(device) {
		return l, nil
	}
	if rr, ok := rt.(ResidentRuntime); ok {
		for _, s := range rr.ResidentStatuses() {
			if !s.Running || !occupies(s.Status.Runtime.Device) {
				continue
			}
			if err := rr.StopResident(s.Model); err != nil {
				return l, fmt.Errorf("quiescing resident %s: %w", s.Model, err)
			}
			l.models = append(l.models, s.Model)
		}
		return l, nil
	}
	if occupies(rt.Status().Runtime.Device) {
		rt.Stop()
		l.whole = true
	}
	return l, nil
}

func (l *lease) names() []string {
	if l.whole {
		return []string{"runtime"}
	}
	return slices.Clone(l.models)
}

// restore starts every resident the lease stopped, with the lifecycle it had
// (its own supervisor, config and restart budget), and waits until each is
// READY again. A resident that does not come back is reported by name.
func (l *lease) restore(ctx context.Context) error {
	var errs []error
	if l.whole {
		l.rt.Start()
		if err := l.awaitReady(ctx, ""); err != nil {
			return err
		}
		l.restored = l.names()
		return nil
	}
	rr, _ := l.rt.(ResidentRuntime)
	for _, m := range l.models {
		if _, err := rr.StartResident(m); err != nil {
			errs = append(errs, fmt.Errorf("restarting resident %s: %w", m, err))
		}
	}
	for _, m := range l.models {
		if err := l.awaitReady(ctx, m); err != nil {
			errs = append(errs, err)
			continue
		}
		l.restored = append(l.restored, m)
	}
	return errors.Join(errs...)
}

// awaitReady waits for model (or the single worker, when empty) to be READY,
// judging each poll from one status read.
func (l *lease) awaitReady(ctx context.Context, model string) error {
	for {
		state, name := l.rt.Status().Worker.State, "runtime"
		if rr, ok := l.rt.(ResidentRuntime); ok && model != "" {
			name = model
			for _, s := range rr.ResidentStatuses() {
				if s.Model == model {
					state = s.Status.Worker.State
				}
			}
		}
		switch state {
		case worker.StateReady:
			return nil
		case worker.StateFailed:
			return fmt.Errorf("%s did not come back: its worker failed", name)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s did not come back (%s): %w", name, state, ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// Execute runs one Forge execution session as temporary maintenance work.
//
// It is one controller action (nothing else starts, stops or activates while
// it runs). Before anything is stopped it records the serving state; it then
// stops only the Hachidori-owned residents that occupy the accelerator the
// target needs, runs the session (Maintenance.Execute), and in every outcome —
// success, execution or evaluation failure, cancellation — tears its worker
// down and starts the stopped residents again, waiting for them to be READY.
// The activation record, the desired residents and the routing policy are
// never written; they are compared after the restore and a difference is a
// failure. A failure to restore is returned beside the primary failure, never
// instead of it, and a session whose restore failed is never a success.
func (c *Controller) Execute(ctx context.Context, p ExecuteParams) (ExecutionResult, error) {
	c.mu.Lock()
	if err := c.admit(); err != nil {
		c.mu.Unlock()
		return ExecutionResult{}, err
	}
	op := c.begin(OpExecute, p.Target.Device, p.Target.Model)
	op.Target = p.Target.String()
	root, rt := c.home, c.rt
	c.mu.Unlock()

	res, err := c.execute(ctx, op, root, rt, p)

	c.mu.Lock()
	defer c.mu.Unlock()
	var f *Failure
	if err != nil {
		f = &Failure{Source: SourceSetup, Phase: op.Phase, Message: err.Error()}
	}
	c.finish(op, f)
	return res, err
}

func (c *Controller) phase(op *Operation, ph string) {
	c.mu.Lock()
	op.Phase = ph
	op.Phases = append(op.Phases, ph)
	c.notify()
	c.mu.Unlock()
}

func (c *Controller) execute(ctx context.Context, op *Operation, root string, rt Runtime, p ExecuteParams) (res ExecutionResult, err error) {
	h := home.Home{Root: root}
	before := c.servingState(h, rt)
	log, closeLog := OpenSetupLog(root, OpExecute, p.Target.Device, p.Target.Model, p.Target.String())
	defer closeLog()

	c.phase(op, PhaseExecQuiesce)
	l, err := c.quiesce(rt, p.Target.Device)
	if err == nil {
		c.phase(op, PhaseExecRun)
		res, err = c.cfg.Maintenance.Execute(ctx, root, p, log)
	}

	c.phase(op, PhaseExecRestore)
	timeout := c.cfg.RestoreTimeout
	if timeout <= 0 {
		timeout = defaultRestoreTimeout
	}
	rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	defer cancel()
	rerr := l.restore(rctx)
	if rerr == nil {
		rerr = before.changed(c.servingState(h, rt))
	}
	res.Quiesced, res.Restored = l.names(), l.restored
	if rerr != nil {
		return res, &ExecutionError{Primary: err, Restore: rerr}
	}
	return res, err
}
