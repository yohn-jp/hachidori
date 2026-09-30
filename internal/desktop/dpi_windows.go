//go:build windows

package desktop

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	procSetProcessDPIAwarenessContext = user32.NewProc("SetProcessDpiAwarenessContext")
	procSetWindowPos                  = user32.NewProc("SetWindowPos")
	procGetDpiForSystem               = user32.NewProc("GetDpiForSystem")
)

const (
	// DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is the only context that
	// keeps the native shell and its child controls correct across monitors
	// with different scale factors.
	dpiAwarenessContextPerMonitorV2 = ^uintptr(3) // HANDLE(-4)
	defaultDPI                      = 96
	swpNoZOrder                     = 0x0004
	swpNoActivate                   = 0x0010
)

// dpiRect is the suggested logical window rectangle from WM_DPICHANGED.
// Keeping this type local makes the native boundary straightforward to test
// without manufacturing a Windows message loop.
type dpiRect struct {
	left, top, right, bottom int32
}

func enablePerMonitorDPI() error {
	if procSetProcessDPIAwarenessContext.Find() != nil {
		// Before Windows 10 1703 the API is absent; the process keeps the
		// system's default awareness and systemDPI reports 96.
		return nil
	}
	if ok, _, err := procSetProcessDPIAwarenessContext.Call(dpiAwarenessContextPerMonitorV2); ok != 0 {
		return nil
	} else if err != windows.ERROR_ACCESS_DENIED {
		return fmt.Errorf("SetProcessDpiAwarenessContext: %w", err)
	}
	// ERROR_ACCESS_DENIED means a manifest or an earlier owner already set
	// the process context. It is safe to continue with that explicit policy.
	return nil
}

func validDPIChangeRect(r *dpiRect) bool {
	return r != nil && r.right > r.left && r.bottom > r.top
}

func applyDPIChange(hwnd, lParam uintptr) {
	if lParam == 0 {
		return
	}
	// lParam is a RECT* owned by the system for this message.
	r := *(**dpiRect)(unsafe.Pointer(&lParam))
	if !validDPIChangeRect(r) {
		return
	}
	procSetWindowPos.Call(hwnd, 0, uintptr(r.left), uintptr(r.top), uintptr(r.right-r.left), uintptr(r.bottom-r.top), swpNoZOrder|swpNoActivate)
}

// systemDPI is the DPI the initial window is sized for. It is 96 for a
// process that is not DPI aware.
func systemDPI() uint32 {
	if procGetDpiForSystem.Find() != nil {
		return defaultDPI
	}
	if dpi, _, _ := procGetDpiForSystem.Call(); dpi != 0 {
		return uint32(dpi)
	}
	return defaultDPI
}

// scaleForDPI converts a logical (96-DPI) length to physical pixels.
func scaleForDPI(logical int32, dpi uint32) int32 {
	return int32((int64(logical)*int64(dpi) + defaultDPI/2) / defaultDPI)
}
