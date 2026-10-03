package harness

import (
	"fmt"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// KillWorker ends the worker of pid and everything it started, as ending the
// python worker in Task Manager does, and waits, bounded, until it is gone.
func (s *Session) KillWorker(pid int) error {
	if err := KillTree(pid); err != nil {
		return err
	}
	return WaitGone("killed worker", BoundStop, pid)
}

// AwaitRecovering waits, bounded, for the observable recovery of an
// unexpected worker exit: the supervisor reports restarting with the crash as
// its last failure, and the dashboard's Diagnostics page says "Recovering from
// an unexpected worker exit". The supervisor holds a 2 second backoff before
// each restart, so the state is present long enough to be observed by polling.
func (s *Session) AwaitRecovering() (server.Status, error) {
	var st server.Status
	err := Eventually(BoundReady, 50*time.Millisecond, "observable recovery (restarting)", func() (bool, string, error) {
		if err := s.exitedErr(); err != nil {
			return false, "", err
		}
		x, err := s.Client.Status()
		if err != nil {
			return false, err.Error(), nil
		}
		if x.Worker.State != worker.StateRestarting {
			return false, fmt.Sprintf("worker %s", x.Worker.State), nil
		}
		page, err := s.Client.Diagnostics()
		if err != nil {
			return false, err.Error(), nil
		}
		st = x
		return RecoveryNotice(page) == "recovering" && strings.Contains(page, "Recovering from an unexpected worker exit"),
			fmt.Sprintf("worker restarting, notice %q", RecoveryNotice(page)), nil
	})
	return st, err
}

// AwaitGaveUp waits, bounded, for the end of automatic recovery: the worker is
// failed, and the Diagnostics page reports "Automatic recovery stopped" and
// Needs attention.
func (s *Session) AwaitGaveUp() (server.Status, error) {
	st, err := s.Client.WaitWorkerState(worker.StateFailed, BoundReady)
	if err != nil {
		return st, err
	}
	page, err := s.Client.Diagnostics()
	if err != nil {
		return st, err
	}
	if RecoveryNotice(page) != "gave_up" || !strings.Contains(page, "Automatic recovery stopped") || !strings.Contains(page, "Needs attention") {
		return st, fmt.Errorf("the Diagnostics page does not report the stopped recovery as needing attention (notice %q)", RecoveryNotice(page))
	}
	return st, nil
}

// ExpectNoRecoveryNotice checks that the Diagnostics page carries no recovery
// notice (a healthy worker, or a startup failure that is not a recovery loop).
func (s *Session) ExpectNoRecoveryNotice() error {
	page, err := s.Client.Diagnostics()
	if err != nil {
		return err
	}
	if n := RecoveryNotice(page); n != "" {
		return fmt.Errorf("the Diagnostics page still reports recovery %q", n)
	}
	return nil
}
