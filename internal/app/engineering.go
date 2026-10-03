package app

import (
	"context"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Model-engineering ownership of the accelerator.
//
// Serving and GPU model-engineering execution (a Forge execution session, a
// variant probe, a certification, the composed build and evaluation) are
// mutually exclusive on one accelerator. The controller owns that exclusion as
// one transaction boundary (transact):
//
//  1. the serving state is recorded and the exclusive ownership is taken;
//  2. the Hachidori-owned residents that occupy the accelerator are stopped
//     and waited for, so their resources are released before any GPU work;
//  3. the ownership is held for the whole composed operation, not for one of
//     its phases: a probe, a reference run and a candidate run nested in it
//     only extend the ownership to their device and never restore anything;
//  4. on every exit (success, failure, cancellation) the ownership is
//     released and exactly the residents the transaction stopped are started
//     again.
//
// What the transaction stopped is the only thing it restores. A runtime the
// operator had already stopped is not running, is never stopped by the
// transaction and is therefore never started by it. The ownership does not
// touch Controller.stopped: the operator's intent survives the pause.

// Pause is the intentional pause of serving while model engineering owns the
// accelerator. It is projected only while the ownership has actually stopped a
// serving resident; an operator Stop is OperatorStopped and a crash is Failed.
type Pause struct {
	// Owner is the kind of the operation that owns the accelerator.
	Owner string `json:"owner"`
	// Residents are the serving residents the ownership stopped ("runtime"
	// for a single worker binding).
	Residents []string `json:"residents,omitempty"`
}

// engineering is one held ownership of the accelerator.
type engineering struct {
	c      *Controller
	root   string
	owner  string
	before servingState
	l      *lease

	// restoring is set under Controller.mu once the ownership is being
	// released: the residents are on their way back and the pause is over.
	restoring bool
}

// pause is the projection of the ownership; c.mu must be held.
func (e *engineering) pause() *Pause {
	if e.restoring || !e.l.stopped() {
		return nil
	}
	return &Pause{Owner: e.owner, Residents: e.l.names()}
}

// acquire takes the exclusive ownership of the accelerator for the operation
// owner. The serving state is recorded before anything is stopped.
func (c *Controller) acquire(root string, rt Runtime, owner string) *engineering {
	e := &engineering{c: c, root: root, owner: owner, before: c.servingState(home.Home{Root: root}, rt), l: &lease{rt: rt}}
	c.mu.Lock()
	c.eng = e
	c.notify()
	c.mu.Unlock()
	return e
}

// holding is the ownership in force, nil when there is none.
func (c *Controller) holding() *engineering {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.eng
}

// release gives the accelerator back: the pause ends, the residents the
// transaction stopped are started again and waited for, and the serving state
// is compared with what was recorded. It never starts anything the
// transaction did not stop.
func (e *engineering) release(ctx context.Context) error {
	e.c.mu.Lock()
	e.restoring = true
	e.c.notify()
	e.c.mu.Unlock()

	timeout := e.c.cfg.RestoreTimeout
	if timeout <= 0 {
		timeout = defaultRestoreTimeout
	}
	rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	defer cancel()
	err := e.l.restore(rctx)
	if err == nil {
		err = e.before.changed(e.c.servingState(home.Home{Root: e.root}, e.l.rt))
	}

	e.c.mu.Lock()
	if e.c.eng == e {
		e.c.eng = nil
	}
	e.c.notify()
	e.c.mu.Unlock()
	return err
}

// transact runs fn as model-engineering work that needs the accelerator on
// each of devices. It is the single place where serving is stopped for, and
// restored after, GPU model-engineering work.
//
// A transaction nested in one that already holds the ownership extends it to
// its devices and runs fn; the owner releases it. Otherwise this call is the
// owner: it acquires, covers devices, runs fn and always releases, so a
// failure of fn or of covering a device still restores what was stopped.
func (c *Controller) transact(ctx context.Context, root string, rt Runtime, report func(string), devices []string, fn func() error) (quiesced, restored []string, err error) {
	phase := func(ph string) {
		if report != nil {
			report(ph)
		}
	}
	if e := c.holding(); e != nil {
		for _, d := range devices {
			if err := e.l.quiesceInto(d); err != nil {
				return nil, nil, err
			}
		}
		return nil, nil, fn()
	}

	c.mu.Lock()
	owner := ""
	if c.op != nil {
		owner = c.op.Kind
	}
	c.mu.Unlock()
	e := c.acquire(root, rt, owner)
	released := false
	defer func() {
		if !released { // fn panicked: the serving residents still come back
			_ = e.release(ctx)
		}
	}()

	phase(PhaseExecQuiesce)
	for _, d := range devices {
		if err = e.l.quiesceInto(d); err != nil {
			break
		}
	}
	if err == nil {
		phase(PhaseExecRun)
		err = fn()
	}

	phase(PhaseExecRestore)
	rerr := e.release(ctx)
	released = true
	if rerr != nil {
		return e.l.names(), e.l.restored, &ExecutionError{Primary: err, Restore: rerr}
	}
	return e.l.names(), e.l.restored, err
}

// engineer runs fn as one composed model-engineering transaction: the
// accelerator is owned from before the first GPU phase until fn returns, so
// the phases nested in fn (leased) share one ownership. fn's own error is
// returned with a restore failure beside it, never instead of it.
func (c *Controller) engineer(ctx context.Context, root string, rt Runtime, devices []string, fn func() error) error {
	_, _, err := c.transact(ctx, root, rt, nil, devices, fn)
	return err
}
