//go:build windows

package harness

import (
	"fmt"
	"os/exec"
	"strings"

	"golang.org/x/sys/windows"

	"github.com/yohn-jp/hachidori/internal/desktop"
)

const stillActive = 259

// ListProcesses reads the process table (id, parent, command line) through
// PowerShell's CIM provider, the one place Windows exposes command lines.
func ListProcesses() ([]Process, error) {
	script := "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " +
		"Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
	out, err := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script).Output()
	if err != nil {
		return nil, fmt.Errorf("list processes: %w", err)
	}
	return ParseProcessJSON(out)
}

// Alive reports whether the process pid exists and has not exited.
func Alive(pid int) bool {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	return code == stillActive
}

// KillTree ends pid and everything it started, as ending the worker in Task
// Manager does for a launcher and its interpreter.
func KillTree(pid int) error {
	out, err := exec.Command("taskkill.exe", "/PID", fmt.Sprint(pid), "/T", "/F").CombinedOutput()
	if err != nil {
		return fmt.Errorf("taskkill /T /F %d: %v: %s", pid, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// KillProcess ends only pid, leaving any child it started (a hard kill of the
// executable, which must not leave its worker behind).
func KillProcess(pid int) error {
	out, err := exec.Command("taskkill.exe", "/PID", fmt.Sprint(pid), "/F").CombinedOutput()
	if err != nil {
		return fmt.Errorf("taskkill /F %d: %v: %s", pid, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// RequestQuit asks the running desktop shell of this user to quit, as the tray
// menu entry Quit Hachidori does. It is a single attempt; see desktop.RequestQuit.
func RequestQuit() error { return desktop.RequestQuit() }

// IsNoWindow reports whether err means there was no shell window to ask yet.
func IsNoWindow(err error) bool { return err == desktop.ErrNoShellWindow }
