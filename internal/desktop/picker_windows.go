//go:build windows

package desktop

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// NativePicker returns the Windows folder picker: the common item dialog
// (IFileOpenDialog with FOS_PICKFOLDERS), i.e. the normal Windows folder
// chooser. It is not a custom file browser.
func NativePicker() FolderPicker { return winPicker{} }

type winPicker struct{}

var procCoCreateInstance = windows.NewLazySystemDLL("ole32.dll").NewProc("CoCreateInstance")

var (
	clsidFileOpenDialog = windows.GUID{Data1: 0xDC1C5A9C, Data2: 0xE88A, Data3: 0x4DDE, Data4: [8]byte{0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7}}
	iidFileOpenDialog   = windows.GUID{Data1: 0xD57C7288, Data2: 0xD4AD, Data3: 0x4768, Data4: [8]byte{0xBE, 0x9D, 0x96, 0x95, 0x32, 0x95, 0x3D, 0x60}}
)

// Vtable slots (IUnknown 0-2, IModalWindow, IFileDialog, IShellItem; see
// ShObjIdl_core.h).
const (
	slotRelease        = 2
	slotDialogShow     = 3
	slotDialogSetOpts  = 9
	slotDialogGetOpts  = 10
	slotDialogSetTitle = 17
	slotDialogResult   = 20
	slotItemDisplay    = 5

	fosPickFolders             = 0x20
	fosForceFileSystem         = 0x40
	fosPathMustExist           = 0x800
	sigdnFileSysPath           = 0x80058000
	hrCancelled        hresult = 0x800704C7
)

func (winPicker) PickFolder(ctx context.Context, title string) (string, error) {
	type result struct {
		path string
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		// The dialog needs its own STA thread for its whole lifetime.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		switch err := windows.CoInitializeEx(0, windows.COINIT_APARTMENTTHREADED); {
		case err == nil, errors.Is(err, syscall.Errno(1)):
			defer windows.CoUninitialize()
		default:
			ch <- result{err: fmt.Errorf("initializing COM for the folder picker: %w", err)}
			return
		}
		p, err := showFolderDialog(title, ownerWindow())
		ch <- result{p, err}
	}()
	select {
	case r := <-ch:
		return r.path, r.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func showFolderDialog(title string, owner uintptr) (string, error) {
	var dlg unsafe.Pointer
	const clsctxInprocServer = 0x1
	if raw, _, _ := procCoCreateInstance.Call(uintptr(unsafe.Pointer(&clsidFileOpenDialog)), 0, clsctxInprocServer,
		uintptr(unsafe.Pointer(&iidFileOpenDialog)), uintptr(unsafe.Pointer(&dlg))); normalizeHRESULT(raw) != sOK {
		hr := normalizeHRESULT(raw)
		return "", fmt.Errorf("creating the folder dialog: %w", hr.errno())
	}
	defer comCall(dlg, slotRelease)

	var opts uint32
	if hr := comCall(dlg, slotDialogGetOpts, uintptr(unsafe.Pointer(&opts))); hr != sOK {
		return "", hr.errno()
	}
	opts |= fosPickFolders | fosForceFileSystem | fosPathMustExist
	if hr := comCall(dlg, slotDialogSetOpts, uintptr(opts)); hr != sOK {
		return "", hr.errno()
	}
	if t, err := windows.UTF16PtrFromString(title); err == nil {
		comCall(dlg, slotDialogSetTitle, uintptr(unsafe.Pointer(t)))
	}
	switch hr := comCall(dlg, slotDialogShow, owner); hr {
	case sOK:
	case hrCancelled:
		return "", ErrPickCancelled
	default:
		return "", fmt.Errorf("showing the folder dialog: %w", hr.errno())
	}
	var item unsafe.Pointer
	if hr := comCall(dlg, slotDialogResult, uintptr(unsafe.Pointer(&item))); hr != sOK || item == nil {
		return "", fmt.Errorf("reading the folder selection: %w", hr.errno())
	}
	defer comCall(item, slotRelease)
	var name *uint16
	if hr := comCall(item, slotItemDisplay, sigdnFileSysPath, uintptr(unsafe.Pointer(&name))); hr != sOK || name == nil {
		return "", fmt.Errorf("the selection is not a file system folder: %w", hr.errno())
	}
	defer windows.CoTaskMemFree(unsafe.Pointer(name))
	return windows.UTF16PtrToString(name), nil
}

// ownerWindow is the desktop shell window, when one is open, so the dialog is
// modal to it.
func ownerWindow() uintptr {
	activeMu.Lock()
	defer activeMu.Unlock()
	if active == nil {
		return 0
	}
	return active.hwnd
}
