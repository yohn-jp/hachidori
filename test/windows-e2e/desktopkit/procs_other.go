//go:build !windows

package desktopkit

import "errors"

// Processes is Windows-only; certification mode requires Windows.
func Processes() ([]ProcEntry, error) {
	return nil, errors.New("process snapshots are only implemented on Windows")
}
