package worker

import (
	"context"
	"sync"
)

// Lifecycle is the single start/stop authority over a Supervisor's Run loop.
// Both `serve` and the host dashboard drive the resident worker through it.
type Lifecycle struct {
	sup    *Supervisor
	parent context.Context

	op     sync.Mutex // serializes Start/Stop/Restart
	mu     sync.Mutex // guards cancel/done
	cancel context.CancelFunc
	done   chan struct{}
}

// NewLifecycle binds a lifecycle to sup; every Run it starts ends when parent does.
func NewLifecycle(parent context.Context, sup *Supervisor) *Lifecycle {
	return &Lifecycle{sup: sup, parent: parent}
}

// Start runs the supervisor unless it is already running. It returns false
// when a run was already active (no second worker is ever started).
func (l *Lifecycle) Start() bool {
	l.op.Lock()
	defer l.op.Unlock()
	return l.start()
}

func (l *Lifecycle) start() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.done != nil {
		select {
		case <-l.done:
		default:
			return false
		}
	}
	ctx, cancel := context.WithCancel(l.parent)
	done := make(chan struct{})
	l.cancel, l.done = cancel, done
	go func() { l.sup.Run(ctx); close(done) }()
	return true
}

// Stop shuts the worker down and waits until the Run loop has returned.
func (l *Lifecycle) Stop() {
	l.op.Lock()
	defer l.op.Unlock()
	l.stop()
}

func (l *Lifecycle) stop() {
	l.mu.Lock()
	cancel, done := l.cancel, l.done
	l.mu.Unlock()
	if done == nil {
		return
	}
	cancel()
	<-done
}

// Restart stops the current worker (if any) and starts a new one.
func (l *Lifecycle) Restart() {
	l.op.Lock()
	defer l.op.Unlock()
	l.stop()
	l.start()
}

// Running reports whether a Run loop is active. A supervisor that gave up
// (startup failure, restart budget exhausted) is not running.
func (l *Lifecycle) Running() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.done == nil {
		return false
	}
	select {
	case <-l.done:
		return false
	default:
		return true
	}
}
