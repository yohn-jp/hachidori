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
)

const (
	// DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is the only context that
	// keeps the native shell and its child controls correct across monitors
	// with different scale factors.
	dpiAwarenessContextPerMonitorV2 = ^uintptr(3) // HANDLE(-4)
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
	r := (*dpiRect)(unsafe.Pointer(lParam))
	if !validDPIChangeRect(r) {
		return
	}
	procSetWindowPos.Call(hwnd, 0, uintptr(r.left), uintptr(r.top), uintptr(r.right-r.left), uintptr(r.bottom-r.top), swpNoZOrder|swpNoActivate)
}
