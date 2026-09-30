//go:build !windows

package subprocess

import "os/exec"

// Configure preserves the platform's existing child process behavior.
func Configure(_ *exec.Cmd) {}
