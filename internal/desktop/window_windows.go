//go:build windows

package desktop

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"github.com/wailsapp/go-webview2/pkg/edge"
	"golang.org/x/sys/windows"
)

var (
	user32               = windows.NewLazySystemDLL("user32.dll")
	procRegisterClassExW = user32.NewProc("RegisterClassExW")
	procUnregisterClassW = user32.NewProc("UnregisterClassW")
	procCreateWindowExW  = user32.NewProc("CreateWindowExW")
	procDefWindowProcW   = user32.NewProc("DefWindowProcW")
	procDestroyWindow    = user32.NewProc("DestroyWindow")
	procShowWindow       = user32.NewProc("ShowWindow")
	procUpdateWindow     = user32.NewProc("UpdateWindow")
	procGetMessageW      = user32.NewProc("GetMessageW")
	procTranslateMessage = user32.NewProc("TranslateMessage")
	procDispatchMessageW = user32.NewProc("DispatchMessageW")
	procPostMessageW     = user32.NewProc("PostMessageW")
	procPostQuitMessage  = user32.NewProc("PostQuitMessage")
	procLoadCursorW      = user32.NewProc("LoadCursorW")
)

const (
	className = "HachidoriDesktopShell"

	wsOverlappedWindow = 0x00CF0000
	cwUseDefault       = 0x80000000
	swShowNormal       = 1
	idcArrow           = 32512
	colorWindow        = 5

	wmDestroy = 0x0002
	wmMove    = 0x0003
	wmSize    = 0x0005
	wmClose   = 0x0010
)

type wndClassEx struct {
	Size       uint32
	Style      uint32
	WndProc    uintptr
	ClsExtra   int32
	WndExtra   int32
	Instance   windows.Handle
	Icon       windows.Handle
	Cursor     windows.Handle
	Background windows.Handle
	MenuName   *uint16
	ClassName  *uint16
	IconSm     windows.Handle
}

type msg struct {
	Hwnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      struct{ X, Y int32 }
	private uint32
}

// shell is the single window of this process (the instance guard allows one
// desktop shell per user; Open refuses a second concurrent window).
type shell struct {
	hwnd       uintptr
	chromium   *edge.Chromium
	controller *edge.ICoreWebView2Controller
	guard      *navigationGuard
	closed     bool
}

var (
	wndProcOnce sync.Once
	wndProcCB   uintptr

	activeMu sync.Mutex
	active   *shell
)

func wndProc(hwnd, m, wp, lp uintptr) uintptr {
	s := active // only touched on the UI thread after Open publishes it
	switch m {
	case wmSize:
		if s != nil && s.chromium != nil {
			s.chromium.Resize()
		}
	case wmMove:
		if s != nil && s.chromium != nil {
			_ = s.chromium.NotifyParentWindowPositionChanged()
		}
	case wmClose:
		// Documented close behaviour: closing the window ends the desktop
		// session. Release the WebView2 controller (browser processes)
		// before the parent window goes away.
		if s != nil {
			s.closeWebView()
		}
		procDestroyWindow.Call(hwnd)
		return 0
	case wmDestroy:
		procPostQuitMessage.Call(0)
		return 0
	}
	r, _, _ := procDefWindowProcW.Call(hwnd, m, wp, lp)
	return r
}

func (s *shell) closeWebView() {
	if s.closed {
		return
	}
	s.closed = true
	if s.chromium != nil {
		s.chromium.ShuttingDown()
	}
	if s.controller != nil {
		comCall(unsafe.Pointer(s.controller), slotControllerClose)
	}
	runtime.KeepAlive(s.guard)
}

// Open implements Platform. It runs the whole window on one locked,
// single-threaded-apartment OS thread and returns after the window is gone.
func (native) Open(ctx context.Context, w Window) (err error) {
	if w.Policy.origin == "" || !w.Policy.AllowNavigation(w.URL) {
		return fmt.Errorf("desktop window URL %q is outside its navigation policy", w.URL)
	}
	activeMu.Lock()
	if active != nil {
		activeMu.Unlock()
		return ErrAlreadyRunning
	}
	s := &shell{}
	active = s
	activeMu.Unlock()
	defer func() {
		activeMu.Lock()
		active = nil
		activeMu.Unlock()
	}()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	switch err := windows.CoInitializeEx(0, windows.COINIT_APARTMENTTHREADED); {
	case err == nil, errors.Is(err, syscall.Errno(1)): // S_OK, S_FALSE
		defer windows.CoUninitialize()
	default:
		return fmt.Errorf("initializing COM for the window thread: %w", err)
	}

	if err := os.MkdirAll(w.DataDir, 0o755); err != nil {
		return fmt.Errorf("WebView2 user data folder: %w", err)
	}

	var inst windows.Handle
	if err := windows.GetModuleHandleEx(0, nil, &inst); err != nil {
		return err
	}
	cls, _ := windows.UTF16PtrFromString(className)
	title, _ := windows.UTF16PtrFromString(w.Title)
	wndProcOnce.Do(func() { wndProcCB = windows.NewCallback(wndProc) })
	cursor, _, _ := procLoadCursorW.Call(0, idcArrow)
	wc := wndClassEx{WndProc: wndProcCB, Instance: inst, Cursor: windows.Handle(cursor),
		Background: windows.Handle(colorWindow + 1), ClassName: cls}
	wc.Size = uint32(unsafe.Sizeof(wc))
	if r, _, e := procRegisterClassExW.Call(uintptr(unsafe.Pointer(&wc))); r == 0 {
		return fmt.Errorf("registering the window class: %w", e)
	}
	defer procUnregisterClassW.Call(uintptr(unsafe.Pointer(cls)), uintptr(inst))

	hwnd, _, e := procCreateWindowExW.Call(0, uintptr(unsafe.Pointer(cls)), uintptr(unsafe.Pointer(title)),
		wsOverlappedWindow, cwUseDefault, cwUseDefault, 1200, 860, 0, 0, uintptr(inst), 0)
	if hwnd == 0 {
		return fmt.Errorf("creating the window: %w", e)
	}
	s.hwnd = hwnd
	destroyed := false
	defer func() {
		if !destroyed {
			s.closeWebView()
			procDestroyWindow.Call(hwnd)
		}
	}()

	c := edge.NewChromium()
	c.DataPath = w.DataDir
	// The binding reports fatal WebView2 errors through this callback and
	// then exits the process; say why before it does.
	c.SetErrorCallback(func(err error) {
		fmt.Fprintln(os.Stderr, "hachidori: WebView2 failure:", err)
	})
	// The dashboard needs no camera, microphone, geolocation, clipboard or
	// notification permission.
	c.SetGlobalPermission(edge.CoreWebView2PermissionStateDeny)
	if !c.Embed(hwnd) {
		return errors.New("embedding WebView2 in the window failed")
	}
	s.controller = c.GetController()
	webview, err := s.controller.GetCoreWebView2()
	if err != nil {
		return err
	}
	if err := lockDown(c); err != nil {
		return err
	}
	s.guard = newNavigationGuard(w.Policy, func(kind, uri string) {
		fmt.Fprintf(os.Stderr, "hachidori: desktop blocked %s to %q (only %s is shown)\n", kind, uri, w.Policy.origin)
	})
	if err := s.guard.attach(unsafe.Pointer(webview)); err != nil {
		return fmt.Errorf("installing the navigation policy: %w", err)
	}
	s.chromium = c
	c.Resize()
	procShowWindow.Call(hwnd, swShowNormal)
	procUpdateWindow.Call(hwnd)
	c.Focus()
	c.Navigate(w.URL)

	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			procPostMessageW.Call(hwnd, wmClose, 0, 0)
		case <-done:
		}
	}()

	var m msg
	for {
		r, _, e := procGetMessageW.Call(uintptr(unsafe.Pointer(&m)), 0, 0, 0)
		switch int32(r) {
		case 0: // WM_QUIT: the window has been destroyed
			destroyed = true
			return nil
		case -1:
			return fmt.Errorf("window message loop: %w", e)
		}
		procTranslateMessage.Call(uintptr(unsafe.Pointer(&m)))
		procDispatchMessageW.Call(uintptr(unsafe.Pointer(&m)))
	}
}

// lockDown turns off every WebView2 feature that could give page content a
// path to native code or a second surface: no web messaging (so no
// JS-to-Go bridge exists at all), no host objects, no DevTools, no default
// context menu, no status bar.
func lockDown(c *edge.Chromium) error {
	st, err := c.GetSettings()
	if err != nil {
		return err
	}
	for _, f := range []func(bool) error{
		st.PutIsWebMessageEnabled,
		st.PutAreHostObjectsAllowed,
		st.PutAreDevToolsEnabled,
		st.PutAreDefaultContextMenusEnabled,
		st.PutIsStatusBarEnabled,
	} {
		if err := f(false); err != nil {
			return fmt.Errorf("configuring WebView2: %w", err)
		}
	}
	return nil
}
