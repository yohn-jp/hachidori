//go:build windows

package desktop

import (
	"errors"
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// ErrNoShellWindow is returned by RequestQuit when no desktop shell window of
// this user exists (yet, or any more).
var ErrNoShellWindow = errors.New("no desktop shell window of this user exists")

// RequestQuit asks the desktop shell already running for this user to end its
// session, as Quit Hachidori does: the window and tray are released, the
// application controller shuts the runtime down within its bound and the
// process exits. It posts the private end-session message the shell also posts
// to itself when its context ends, to the window found by the same per-user
// class a second launch uses (Activate), so it needs no new authority and
// reaches only this user's shell.
//
// It is the deterministic Quit used by the Windows E2E certification, which has
// no human to choose the tray menu entry. It returns ErrNoShellWindow without
// retrying; the caller bounds the wait.
func RequestQuit() error {
	uid, err := currentUserID()
	if err != nil {
		return err
	}
	class, err := windows.UTF16PtrFromString(WindowClassName(uid))
	if err != nil {
		return err
	}
	hwnd, _, _ := procFindWindowW.Call(uintptr(unsafe.Pointer(class)), 0)
	if hwnd == 0 {
		return ErrNoShellWindow
	}
	if r, _, e := procPostMessageW.Call(hwnd, wmQuitReq, 0, 0); r == 0 {
		return fmt.Errorf("posting the quit request to the desktop window: %w", e)
	}
	return nil
}
