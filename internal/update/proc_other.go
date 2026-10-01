//go:build !windows

package update

import (
	"errors"
	"fmt"
	"os/exec"
	"syscall"
	"time"
)

// StartDetached starts exe with args as an independent process and does not
// wait for it.
func StartDetached(exe string, args []string) error {
	cmd := exec.Command(exe, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Process.Release()
}

// waitExit polls until the process no longer exists.
func waitExit(pid int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if err := syscall.Kill(pid, 0); errors.Is(err, syscall.ESRCH) {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("process %d still running after %s", pid, timeout)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
