//go:build windows

package desktop

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

// The tray icon is one Shell_NotifyIcon notification icon owned by the main
// window and driven entirely from that window's UI thread. It shows the
// concise state of the application controller (Summary) as its tooltip and
// menu header; it computes nothing itself.

var (
	shell32              = windows.NewLazySystemDLL("shell32.dll")
	procShellNotifyIconW = shell32.NewProc("Shell_NotifyIconW")

	procRegisterWindowMessageW   = user32.NewProc("RegisterWindowMessageW")
	procAllowSetForegroundWindow = user32.NewProc("AllowSetForegroundWindow")
	procFindWindowW              = user32.NewProc("FindWindowW")
	procMessageBoxW              = user32.NewProc("MessageBoxW")
	procSetForegroundWindow      = user32.NewProc("SetForegroundWindow")
	procIsIconic                 = user32.NewProc("IsIconic")
	procCreatePopupMenu          = user32.NewProc("CreatePopupMenu")
	procAppendMenuW              = user32.NewProc("AppendMenuW")
	procTrackPopupMenu           = user32.NewProc("TrackPopupMenu")
	procDestroyMenu              = user32.NewProc("DestroyMenu")
	procGetCursorPos             = user32.NewProc("GetCursorPos")
	procLoadIconW                = user32.NewProc("LoadIconW")
)

const (
	nimAdd    = 0
	nimModify = 1
	nimDelete = 2

	nifMessage = 0x1
	nifIcon    = 0x2
	nifTip     = 0x4
	nifInfo    = 0x10

	niifInfo = 0x1

	idiApplication = 32512

	mfString    = 0x0
	mfGrayed    = 0x1
	mfChecked   = 0x8
	mfSeparator = 0x800

	tpmRightButton = 0x2
	tpmReturnCmd   = 0x100

	wmLButtonUp     = 0x0202
	wmLButtonDblClk = 0x0203
	wmRButtonUp     = 0x0205
	wmContextMenu   = 0x007B
	wmNull          = 0x0000
)

// notifyIconData is NOTIFYICONDATAW (shellapi.h, Vista+ layout).
type notifyIconData struct {
	Size             uint32
	Hwnd             uintptr
	ID               uint32
	Flags            uint32
	CallbackMessage  uint32
	Icon             uintptr
	Tip              [128]uint16
	State            uint32
	StateMask        uint32
	Info             [256]uint16
	TimeoutOrVersion uint32
	InfoTitle        [64]uint16
	InfoFlags        uint32
	GUIDItem         windows.GUID
	BalloonIcon      uintptr
}

type point struct{ X, Y int32 }

// tray is the notification-area icon of one window.
type tray struct {
	hwnd  uintptr
	msg   uint32 // callback message posted to hwnd
	added bool
}

func (t *tray) data() notifyIconData {
	var d notifyIconData
	d.Size = uint32(unsafe.Sizeof(d))
	d.Hwnd = t.hwnd
	d.ID = 1
	return d
}

func copyUTF16(dst []uint16, s string) {
	u, _ := windows.UTF16FromString(s)
	if len(u) > len(dst) {
		u = u[:len(dst)]
		u[len(u)-1] = 0
	}
	copy(dst, u)
}

// add creates the icon. It can fail early in a sign-in session while the
// shell is still starting; the caller retries (TaskbarCreated / refresh).
func (t *tray) add(sum Summary) bool {
	if t.added {
		return true
	}
	d := t.data()
	d.Flags = nifMessage | nifIcon | nifTip
	d.CallbackMessage = t.msg
	d.Icon, _, _ = procLoadIconW.Call(0, idiApplication)
	copyUTF16(d.Tip[:], sum.Tooltip())
	r, _, _ := procShellNotifyIconW.Call(nimAdd, uintptr(unsafe.Pointer(&d)))
	t.added = r != 0
	return t.added
}

func (t *tray) setTip(sum Summary) {
	if !t.added {
		return
	}
	d := t.data()
	d.Flags = nifTip
	copyUTF16(d.Tip[:], sum.Tooltip())
	procShellNotifyIconW.Call(nimModify, uintptr(unsafe.Pointer(&d)))
}

func (t *tray) balloon(title, text string) {
	if !t.added {
		return
	}
	d := t.data()
	d.Flags = nifInfo
	d.InfoFlags = niifInfo
	copyUTF16(d.InfoTitle[:], title)
	copyUTF16(d.Info[:], text)
	procShellNotifyIconW.Call(nimModify, uintptr(unsafe.Pointer(&d)))
}

func (t *tray) remove() {
	if !t.added {
		return
	}
	d := t.data()
	procShellNotifyIconW.Call(nimDelete, uintptr(unsafe.Pointer(&d)))
	t.added = false
}

// popup shows the tray menu at the cursor and returns the chosen entry (0 if
// the menu was dismissed).
func (t *tray) popup(items []MenuItem) MenuID {
	h, _, _ := procCreatePopupMenu.Call()
	if h == 0 {
		return 0
	}
	defer procDestroyMenu.Call(h)
	for _, it := range items {
		if it.Separator {
			procAppendMenuW.Call(h, mfSeparator, 0, 0)
			continue
		}
		flags := uintptr(mfString)
		if it.Disabled {
			flags |= mfGrayed
		}
		if it.Checked {
			flags |= mfChecked
		}
		label, _ := windows.UTF16PtrFromString(it.Label)
		procAppendMenuW.Call(h, flags, uintptr(it.ID), uintptr(unsafe.Pointer(label)))
	}
	var pt point
	procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	// The window must be foreground or the menu will not dismiss on an
	// outside click (documented Shell_NotifyIcon requirement).
	procSetForegroundWindow.Call(t.hwnd)
	cmd, _, _ := procTrackPopupMenu.Call(h, tpmReturnCmd|tpmRightButton,
		uintptr(pt.X), uintptr(pt.Y), 0, t.hwnd, 0)
	procPostMessageW.Call(t.hwnd, wmNull, 0, 0)
	return MenuID(cmd)
}
