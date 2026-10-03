//go:build windows

package desktopkit

import (
	"errors"
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
	if err = windows.Process32First(snap, &e); err != nil {
		return nil, fmt.Errorf("first process: %w", err)
	}
	var out []ProcEntry
	for {
		out = append(out, ProcEntry{PID: int(e.ProcessID), PPID: int(e.ParentProcessID), Name: windows.UTF16ToString(e.ExeFile[:])})
		err = windows.Process32Next(snap, &e)
		if errors.Is(err, windows.ERROR_NO_MORE_FILES) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("next process: %w", err)
		}
	}
	return out, nil
}
