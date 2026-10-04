//go:build windows

package home

import (
	"errors"
	"os"
	"path/filepath"
	"unsafe"

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

var replaceFileW = windows.NewLazySystemDLL("kernel32.dll").NewProc("ReplaceFileW")

func replaceBootstrapFile(replacement, replaced string) error {
	// MoveFileEx/os.Rename cannot replace a destination that another process
	// has open, even when that reader grants delete sharing. ReplaceFileW is
	// the Windows replacement primitive whose destination open explicitly
	// requests FILE_SHARE_DELETE, matching the locator's atomic-read contract.
	dst, err := windows.UTF16PtrFromString(replaced)
	if err != nil {
		return err
	}
	src, err := windows.UTF16PtrFromString(replacement)
	if err != nil {
		return err
	}
	ok, _, callErr := replaceFileW.Call(
		uintptr(unsafe.Pointer(dst)),
		uintptr(unsafe.Pointer(src)),
		0, 0, 0, 0,
	)
	if ok != 0 {
		return nil
	}
	if errors.Is(callErr, windows.ERROR_FILE_NOT_FOUND) ||
		errors.Is(callErr, windows.ERROR_PATH_NOT_FOUND) {
		// ReplaceFileW requires an existing destination. Initial creation (or
		// a concurrent disappearance) retains the ordinary rename path.
		return os.Rename(replacement, replaced)
	}
	return callErr
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
