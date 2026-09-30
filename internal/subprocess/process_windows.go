//go:build windows

// Package subprocess contains the small process policy shared by Hachidori's
// private child launches. It does not configure the top-level CLI process.
package subprocess

import (
	"os/exec"
	"syscall"
)

// Configure keeps a Hachidori-owned child from allocating a visible console
// window while leaving its stdio, environment, directory and exit handling to
// the caller.
func Configure(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
}
