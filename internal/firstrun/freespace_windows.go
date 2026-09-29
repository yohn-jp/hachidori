//go:build windows

package firstrun

import "golang.org/x/sys/windows"

// DefaultFreeSpace reports the free bytes available to the current user on
// the volume holding path (GetDiskFreeSpaceEx).
func DefaultFreeSpace(path string) (uint64, bool) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, false
	}
	var avail, total, free uint64
	if err := windows.GetDiskFreeSpaceEx(p, &avail, &total, &free); err != nil {
		return 0, false
	}
	return avail, true
}
