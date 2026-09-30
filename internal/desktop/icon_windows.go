//go:build windows

package desktop

import (
	"github.com/yohn-jp/hachidori/internal/winres"
	"golang.org/x/sys/windows"
)

// The window, taskbar and tray use the application icon that cmd/hachidori
// links into the executable (internal/winres). A binary without that
// resource, such as a test binary, keeps the stock application icon.

var (
	procLoadImageW       = user32.NewProc("LoadImageW")
	procGetSystemMetrics = user32.NewProc("GetSystemMetrics")
)

const (
	imageIcon  = 1
	lrShared   = 0x8000
	smCxIcon   = 11
	smCyIcon   = 12
	smCxSmIcon = 49
	smCySmIcon = 50
)

// appIcon returns the executable's icon at the system large or small icon
// size. The handle is shared (LR_SHARED) and is never destroyed.
func appIcon(inst windows.Handle, small bool) uintptr {
	mx, my := uintptr(smCxIcon), uintptr(smCyIcon)
	if small {
		mx, my = smCxSmIcon, smCySmIcon
	}
	cx, _, _ := procGetSystemMetrics.Call(mx)
	cy, _, _ := procGetSystemMetrics.Call(my)
	h, _, _ := procLoadImageW.Call(uintptr(inst), winres.IconGroupID, imageIcon, cx, cy, lrShared)
	if h == 0 {
		h, _, _ = procLoadIconW.Call(0, idiApplication)
	}
	return h
}
