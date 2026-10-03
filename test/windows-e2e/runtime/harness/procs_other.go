//go:build !windows

package harness

import (
	"errors"
	"fmt"
	"os/exec"
	"syscall"
)

// ListProcesses reads the process table with ps. The Windows E2E does not run
// here; this exists so the harness is exercised by portable tests.
func ListProcesses() ([]Process, error) {
	out, err := exec.Command("ps", "-axo", "pid=,ppid=,args=").Output()
	if err != nil {
		return nil, fmt.Errorf("list processes: %w", err)
	}
	return ParsePS(out), nil
}

// Alive reports whether pid exists and is not a zombie.
func Alive(pid int) bool {
	if syscall.Kill(pid, 0) != nil {
		return false
	}
	out, err := exec.Command("ps", "-o", "stat=", "-p", fmt.Sprint(pid)).Output()
	if err != nil {
		return false
	}
	s := string(out)
	return len(s) > 0 && s[0] != 'Z'
}

// KillTree ends pid and its descendants.
func KillTree(pid int) error {
	all, err := ListProcesses()
	if err != nil {
		return err
	}
	for _, d := range Descendants(all, pid) {
		_ = syscall.Kill(d, syscall.SIGKILL)
	}
	return syscall.Kill(pid, syscall.SIGKILL)
}

// KillProcess ends only pid.
func KillProcess(pid int) error { return syscall.Kill(pid, syscall.SIGKILL) }

// RequestQuit is Windows only.
func RequestQuit() error { return errors.ErrUnsupported }

// IsNoWindow is false: there is no shell window off Windows.
func IsNoWindow(error) bool { return false }
