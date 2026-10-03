//go:build windows

package desktopkit

import (
	"os"
	"os/exec"
	"strconv"
)

// killTree ends the process and every process it started (the WebView2
// browser, a worker) with taskkill, which is what leaves nothing of the run
// behind.
func killTree(p *os.Process) {
	if err := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(p.Pid)).Run(); err != nil {
		_ = p.Kill()
	}
}
