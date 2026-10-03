//go:build !windows

package lab

import (
	"io"
	"os"
)

// holdOpen holds path open. Off Windows an open handle does not block a
// rename, so the file-lock scenario is only meaningful (and only run) on
// Windows; this exists so the package builds everywhere.
func holdOpen(path string) (io.Closer, error) { return os.Open(path) }
