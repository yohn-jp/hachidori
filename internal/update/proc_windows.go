//go:build windows

package update

import (
	"errors"
	"fmt"
	"os/exec"
	"syscall"
	"time"

	"golang.org/x/sys/windows"
)

// DETACHED_PROCESS: the child has no console and outlives its parent.
const detachedProcess = 0x00000008

// StartDetached starts exe with args as an independent process and does not
// wait for it.
func StartDetached(exe string, args []string) error {
	cmd := exec.Command(exe, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP | detachedProcess}
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Process.Release()
}

// waitExit waits for the process to end. A process that cannot be opened has
// already exited (or never existed).
func waitExit(pid int, timeout time.Duration) error {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(pid))
	if err != nil {
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
			return nil
		}
		return err
	}
	defer windows.CloseHandle(h)
	ev, err := windows.WaitForSingleObject(h, uint32(timeout.Milliseconds()))
	switch {
	case err != nil:
		return err
	case ev == windows.WAIT_OBJECT_0:
		return nil
	}
	return fmt.Errorf("process %d still running after %s", pid, timeout)
}
