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

// NativePicker returns the Windows picker: the common item dialogs
// (IFileOpenDialog, with FOS_PICKFOLDERS for folders, and IFileSaveDialog).
// It also implements PathPicker. It is not a custom file browser.
func NativePicker() FolderPicker { return winPicker{} }

type winPicker struct{}

var procCoCreateInstance = windows.NewLazySystemDLL("ole32.dll").NewProc("CoCreateInstance")

var (
	clsidFileOpenDialog = windows.GUID{Data1: 0xDC1C5A9C, Data2: 0xE88A, Data3: 0x4DDE, Data4: [8]byte{0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7}}
	iidFileOpenDialog   = windows.GUID{Data1: 0xD57C7288, Data2: 0xD4AD, Data3: 0x4768, Data4: [8]byte{0xBE, 0x02, 0x9D, 0x96, 0x95, 0x32, 0xD9, 0x60}}
	clsidFileSaveDialog = windows.GUID{Data1: 0xC0B4E2F3, Data2: 0xBA21, Data3: 0x4773, Data4: [8]byte{0x8D, 0xBA, 0x33, 0x5E, 0xC9, 0x46, 0xEB, 0x8B}}
	iidFileSaveDialog   = windows.GUID{Data1: 0x84BCCD23, Data2: 0x5FDE, Data3: 0x4CDB, Data4: [8]byte{0xAE, 0xA4, 0xAF, 0x64, 0xB8, 0x3D, 0x78, 0xAB}}
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

	fosOverwritePrompt         = 0x2
	fosPickFolders             = 0x20
	fosForceFileSystem         = 0x40
	fosPathMustExist           = 0x800
	fosFileMustExist           = 0x1000
	sigdnFileSysPath           = 0x80058000
	hrCancelled        hresult = 0x800704C7
)

func (winPicker) PickFolder(ctx context.Context, title string) (string, error) {
	return runDialog(ctx, func(owner uintptr) (string, error) { return showFolderDialog(title, owner) })
}

func (winPicker) PickOpen(ctx context.Context, title string) (string, error) {
	return runDialog(ctx, func(owner uintptr) (string, error) { return showFileDialog(title, owner, false) })
}

func (winPicker) PickSave(ctx context.Context, title string) (string, error) {
	return runDialog(ctx, func(owner uintptr) (string, error) { return showFileDialog(title, owner, true) })
}

func runDialog(ctx context.Context, show func(uintptr) (string, error)) (string, error) {
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
			ch <- result{err: fmt.Errorf("initializing COM for the native picker: %w", err)}
			return
		}
		p, err := show(ownerWindow())
		ch <- result{p, err}
	}()
	select {
	case r := <-ch:
		return r.path, r.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func showFileDialog(title string, owner uintptr, save bool) (string, error) {
	clsid, iid := &clsidFileOpenDialog, &iidFileOpenDialog
	if save {
		clsid, iid = &clsidFileSaveDialog, &iidFileSaveDialog
	}
	var dlg unsafe.Pointer
	const clsctxInprocServer = 0x1
	raw, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(clsid)), 0, clsctxInprocServer,
		uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&dlg)))
	if hr := normalizeHRESULT(raw); hr != sOK {
		return "", fmt.Errorf("creating the file dialog: %w", hr.errno())
	}
	defer comCall(dlg, slotRelease)
	var opts uint32
	if hr := comCall(dlg, slotDialogGetOpts, uintptr(unsafe.Pointer(&opts))); hr != sOK {
		return "", hr.errno()
	}
	opts |= fosForceFileSystem
	if save {
		// Choosing a destination writes nothing; the exporter's
		// never-overwrite check decides, so do not offer to replace a file.
		opts &^= fosOverwritePrompt
	} else {
		opts |= fosPathMustExist | fosFileMustExist
	}
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
		return "", fmt.Errorf("showing the file dialog: %w", hr.errno())
	}
	return dialogResultPath(dlg, "file")
}

func dialogResultPath(dlg unsafe.Pointer, kind string) (string, error) {
	var item unsafe.Pointer
	if hr := comCall(dlg, slotDialogResult, uintptr(unsafe.Pointer(&item))); hr != sOK || item == nil {
		return "", fmt.Errorf("reading the selected %s: %w", kind, hr.errno())
	}
	defer comCall(item, slotRelease)
	var name *uint16
	if hr := comCall(item, slotItemDisplay, sigdnFileSysPath, uintptr(unsafe.Pointer(&name))); hr != sOK || name == nil {
		return "", fmt.Errorf("the selection is not a file system %s: %w", kind, hr.errno())
	}
	defer windows.CoTaskMemFree(unsafe.Pointer(name))
	return windows.UTF16PtrToString(name), nil
}

func showFolderDialog(title string, owner uintptr) (string, error) {
	var dlg unsafe.Pointer
	const clsctxInprocServer = 0x1
	raw, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&clsidFileOpenDialog)),
		0,
		clsctxInprocServer,
		uintptr(unsafe.Pointer(&iidFileOpenDialog)),
		uintptr(unsafe.Pointer(&dlg)),
	)
	if hr := normalizeHRESULT(raw); hr != sOK {
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
