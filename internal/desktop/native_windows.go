//go:build windows

package desktop

import (
	"errors"
	"fmt"

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

// AcquireInstance creates the per-user named mutex (see InstanceName). The
// handle is held until release; the kernel also drops it when the process
// exits, so a crashed shell never blocks the next launch.
func (native) AcquireInstance() (func(), error) {
	tok := windows.GetCurrentProcessToken()
	u, err := tok.GetTokenUser()
	if err != nil {
		return nil, fmt.Errorf("resolving the current user: %w", err)
	}
	name, err := windows.UTF16PtrFromString(InstanceName(u.User.Sid.String()))
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
