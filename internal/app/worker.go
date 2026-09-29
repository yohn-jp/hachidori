package app

import (
	"context"
	"io"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// WorkerBinding is the production Runtime: the same worker.Lifecycle,
// worker.Supervisor and server.StatusBody composition that `serve` and the
// dashboard use. Supervisor is exported so a host can serve the HTTP API
// (server.HandlerSince) against the same supervisor.
type WorkerBinding struct {
	*worker.Lifecycle
	Supervisor *worker.Supervisor
	Info       server.Runtime
	Started    time.Time
}

// Status is the /v1/status document of this binding.
func (b *WorkerBinding) Status() server.Status {
	return server.StatusBody(b.Supervisor, b.Info, b.Started)
}

// WorkerRuntime returns an OpenFunc that builds a WorkerBinding from the
// active runtime (server.WorkerConfig). Every worker it starts ends when
// parent does. onOpen, if non-nil, receives each new binding.
func WorkerRuntime(parent context.Context, log io.Writer, policy worker.Policy, onOpen func(*WorkerBinding)) OpenFunc {
	return func(root string) (Runtime, error) {
		cfg, rt, err := server.WorkerConfig(home.Home{Root: root}, log)
		if err != nil {
			return nil, err
		}
		sup := worker.NewSupervisor(cfg, policy)
		b := &WorkerBinding{Lifecycle: worker.NewLifecycle(parent, sup), Supervisor: sup, Info: rt, Started: time.Now()}
		if onOpen != nil {
			onOpen(b)
		}
		return b, nil
	}
}
