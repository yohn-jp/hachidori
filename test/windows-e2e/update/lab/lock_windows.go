//go:build windows

package lab

import (
	"io"

	"golang.org/x/sys/windows"
)

type handleCloser windows.Handle

func (h handleCloser) Close() error { return windows.CloseHandle(windows.Handle(h)) }

// holdOpen holds path open for reading without FILE_SHARE_DELETE, the way a
// scanner, an indexer or another process holding the executable does: the file
// can still be read and run, but it cannot be renamed or replaced until the
// returned handle is closed.
func holdOpen(path string) (io.Closer, error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	h, err := windows.CreateFile(name, windows.GENERIC_READ, windows.FILE_SHARE_READ, nil,
		windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, err
	}
	return handleCloser(h), nil
}
