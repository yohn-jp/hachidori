//go:build windows

package desktopkit

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Processes snapshots every process of the machine (pid, parent pid and image
// name) with the Toolhelp API.
func Processes() ([]ProcEntry, error) {
	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, fmt.Errorf("process snapshot: %w", err)
	}
	defer func() { _ = windows.CloseHandle(snap) }()
	var e windows.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	var out []ProcEntry
	for err = windows.Process32First(snap, &e); err == nil; err = windows.Process32Next(snap, &e) {
		out = append(out, ProcEntry{PID: int(e.ProcessID), PPID: int(e.ParentProcessID), Name: windows.UTF16ToString(e.ExeFile[:])})
	}
	return out, nil
}
