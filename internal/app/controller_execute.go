package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
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

// lease is the set of Hachidori-owned residents a model-engineering transaction
// stopped so that its workers could have the accelerator, and how to bring them
// back. It is written by the transaction's goroutine and read by Snapshot.
type lease struct {
	rt Runtime

	mu       sync.Mutex
	models   []string // residents of a resident set, in the order they were stopped
	whole    bool     // a single worker binding was stopped
	restored []string
}

// occupies reports whether a runtime on device holds the accelerator.
func occupies(device string) bool { return device != "cpu" }

// quiesceInto stops exactly the running Hachidori-owned residents that occupy
// the accelerator device needs, through the resident lifecycle authority, and
// records them in l. A CPU target, a runtime that is not running and residents
// on the CPU are left alone; no process Hachidori does not own is ever looked
// at. l records what was stopped even when quiescing fails part way, and a
// resident l already stopped is not running any more, so covering a second
// device never stops anything twice.
func (l *lease) quiesceInto(device string) error {
	rt := l.rt
	if rt == nil || !rt.Running() || !occupies(device) {
		return nil
	}
	if rr, ok := rt.(ResidentRuntime); ok {
		for _, s := range rr.ResidentStatuses() {
			if !s.Running || !occupies(s.Status.Runtime.Device) {
				continue
			}
			if err := rr.StopResident(s.Model); err != nil {
				return fmt.Errorf("quiescing resident %s: %w", s.Model, err)
			}
			l.mu.Lock()
			l.models = append(l.models, s.Model)
			l.mu.Unlock()
		}
		return nil
	}
	if occupies(rt.Status().Runtime.Device) {
		rt.Stop()
		l.mu.Lock()
		l.whole = true
		l.mu.Unlock()
	}
	return nil
}

// stopped reports whether the lease holds anything serving would need back.
func (l *lease) stopped() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.whole || len(l.models) > 0
}

func (l *lease) names() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
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
	log, closeLog := OpenSetupLog(root, OpExecute, p.Target.Device, p.Target.Model, p.Target.String())
	defer closeLog()
	quiesced, restored, err := c.leased(ctx, root, rt, p.Target.Device, func(ph string) { c.phase(op, ph) }, func() (err error) {
		res, err = c.cfg.Maintenance.Execute(ctx, root, p, log)
		return err
	})
	res.Quiesced, res.Restored = quiesced, restored
	return res, err
}

// leased runs fn as model-engineering work on device. It is the one entry the
// execution, probe and composed Forge operations use for GPU work.
//
// Inside a transaction that already owns the accelerator (transact) it only
// makes sure device is covered and runs fn: the serving residents stay down
// until the owning transaction ends, so a nested probe, reference or candidate
// execution can never bring them back in the middle of it. Standalone it is a
// transaction of its own: it records the serving state, stops only the
// Hachidori-owned residents that occupy the accelerator device needs, runs fn,
// and in every outcome starts them again, waits for them to be READY and
// compares the serving state with what it recorded. A failure to restore is
// returned beside fn's own failure (ExecutionError), never instead of it.
// report, when set, is told each phase as it is entered.
func (c *Controller) leased(ctx context.Context, root string, rt Runtime, device string, report func(string), fn func() error) (quiesced, restored []string, err error) {
	return c.transact(ctx, root, rt, report, []string{device}, fn)
}
