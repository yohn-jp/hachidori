//go:build windows

package home

import (
	"errors"
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"
)

func openBootstrapReadFile(path string) (*os.File, error) {
	// os.Open/os.ReadFile on Windows use FILE_SHARE_READ|FILE_SHARE_WRITE
	// but not FILE_SHARE_DELETE. Root.Open uses the Go 1.24 rooted Windows
	// open path, which includes delete sharing, so an atomic rename can replace
	// the locator while a reader still owns the previous file handle.
	root, err := os.OpenRoot(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	defer root.Close()
	return root.Open(filepath.Base(path))
}

func retryableBootstrapReadError(err error) bool {
	return errors.Is(err, windows.ERROR_SHARING_VIOLATION) ||
		errors.Is(err, windows.ERROR_LOCK_VIOLATION)
}

func retryableBootstrapReplaceError(err error) bool {
	return errors.Is(err, windows.ERROR_ACCESS_DENIED) ||
		errors.Is(err, windows.ERROR_SHARING_VIOLATION) ||
		errors.Is(err, windows.ERROR_LOCK_VIOLATION)
}
