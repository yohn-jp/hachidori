//go:build windows

package desktop

import (
	"errors"
	"fmt"
	"time"
	"unsafe"

	"github.com/wailsapp/go-webview2/webviewloader"
	"golang.org/x/sys/windows"
)

// Native returns the Windows desktop platform: WebView2 through the pure-Go
// loader of github.com/wailsapp/go-webview2 (no WebView2Loader.dll is
// embedded, written or searched for) and a per-user named mutex.
func Native() Platform { return native{} }

type native struct{}

// RuntimeVersion asks the installed Evergreen WebView2 Runtime for its
// version (registry + client DLL lookup; nothing is downloaded).
func (native) RuntimeVersion() (string, error) {
	return webviewloader.GetAvailableCoreWebView2BrowserVersionString("")
}

func currentUserID() (string, error) {
	u, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		return "", fmt.Errorf("resolving the current user: %w", err)
	}
	return u.User.Sid.String(), nil
}

// AcquireInstance creates the per-user named mutex (see InstanceName). The
// handle is held until release; the kernel also drops it when the process
// exits, so a crashed shell never blocks the next launch.
func (native) AcquireInstance() (func(), error) {
	uid, err := currentUserID()
	if err != nil {
		return nil, err
	}
	name, err := windows.UTF16PtrFromString(InstanceName(uid))
	if err != nil {
		return nil, err
	}
	h, err := windows.CreateMutex(nil, false, name)
	if errors.Is(err, windows.ERROR_ALREADY_EXISTS) {
		if h != 0 {
			_ = windows.CloseHandle(h)
		}
		return nil, ErrAlreadyRunning
	}
	if err != nil {
		return nil, fmt.Errorf("creating the single-instance guard: %w", err)
	}
	return func() { _ = windows.CloseHandle(h) }, nil
}

// Activate finds the running shell's main window by its per-user class (a
// hidden window is still found) and posts it the per-user activate message.
// The first shell takes the mutex before it creates its window, so a very
// early second launch retries briefly.
func (native) Activate() error {
	uid, err := currentUserID()
	if err != nil {
		return err
	}
	class, err := windows.UTF16PtrFromString(WindowClassName(uid))
	if err != nil {
		return err
	}
	msgName, err := windows.UTF16PtrFromString(ActivateMessageName(uid))
	if err != nil {
		return err
	}
	msgID, _, e := procRegisterWindowMessageW.Call(uintptr(unsafe.Pointer(msgName)))
	if msgID == 0 {
		return fmt.Errorf("registering the activate message: %w", e)
	}
	// Let the running shell take the foreground on our behalf.
	procAllowSetForegroundWindow.Call(^uintptr(0)) // ASFW_ANY
	deadline := time.Now().Add(10 * time.Second)
	for {
		hwnd, _, _ := procFindWindowW.Call(uintptr(unsafe.Pointer(class)), 0)
		if hwnd != 0 {
			if r, _, e := procPostMessageW.Call(hwnd, msgID, 0, 0); r == 0 {
				return fmt.Errorf("activating the running desktop window: %w", e)
			}
			return nil
		}
		if time.Now().After(deadline) {
			return errors.New("the running desktop shell has no window to activate")
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// ReportError shows a modal error box, for launches with no visible console.
func (native) ReportError(title, message string) {
	t, _ := windows.UTF16PtrFromString(title)
	m, _ := windows.UTF16PtrFromString(message)
	const mbOK, mbIconError = 0x0, 0x10
	procMessageBoxW.Call(0, uintptr(unsafe.Pointer(m)), uintptr(unsafe.Pointer(t)), mbOK|mbIconError)
}
