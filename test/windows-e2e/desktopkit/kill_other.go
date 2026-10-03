//go:build !windows

package desktopkit

import "os"

func killTree(p *os.Process) { _ = p.Kill() }
