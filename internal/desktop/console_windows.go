//go:build windows

package desktop

import (
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32Console           = windows.NewLazySystemDLL("kernel32.dll")
	procGetConsoleWindow      = kernel32Console.NewProc("GetConsoleWindow")
	procGetConsoleProcessList = kernel32Console.NewProc("GetConsoleProcessList")
)

// HideOwnedConsole hides the console created solely for this process.
// A console shared with an existing terminal is left untouched.
func HideOwnedConsole() bool {
	hwnd, _, _ := procGetConsoleWindow.Call()
	if hwnd == 0 {
		return false
	}

	buf := make([]uint32, 4)
	for {
		n, _, _ := procGetConsoleProcessList.Call(
			uintptr(unsafe.Pointer(&buf[0])),
			uintptr(len(buf)),
		)
		if n == 0 {
			return false
		}
		if n <= uintptr(len(buf)) {
			if !shouldHideOwnedConsole(uint32(os.Getpid()), buf[:int(n)]) {
				return false
			}
			procShowWindow.Call(hwnd, swHide)
			return true
		}
		buf = make([]uint32, int(n))
	}
}
