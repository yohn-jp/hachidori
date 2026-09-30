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
	"time"
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
	procIsWindowVisible  = user32.NewProc("IsWindowVisible")
)

const (
	wsOverlappedWindow = 0x00CF0000
	cwUseDefault       = 0x80000000
	swHide             = 0
	swShowNormal       = 1
	swMinimize         = 6
	swRestore          = 9
	idcArrow           = 32512
	colorWindow        = 5

	wmDestroy    = 0x0002
	wmMove       = 0x0003
	wmSize       = 0x0005
	wmClose      = 0x0010
	wmDPIChanged = 0x02E0

	// Private messages, posted to the window from other goroutines.
	wmTray     = 0x8000 + 1 // tray icon callback (WM_APP+1)
	wmRefresh  = 0x8000 + 2 // application state may have changed
	wmQuitReq  = 0x8000 + 3 // end the session (context cancelled)
	wmReload   = 0x8000 + 4 // bounded WebView2 reload after a renderer/navigation failure
	trayNotice = "Hachidori keeps running in the tray. Use the tray icon to open it, or choose Quit Hachidori to exit."
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

	// Resident mode (nil res: no tray, closing the window ends the session).
	res         *Resident
	url         string // dashboard URL the window was opened on
	tray        tray
	activateMsg uint32 // registered per-user activate message
	taskbarMsg  uint32 // "TaskbarCreated": explorer restarted, re-add the icon
	lastLevel   Level

	// WebView2 failure boundary: every failure after the environment was
	// created is recorded, retried a bounded number of times by reloading
	// the loopback dashboard, and then surfaced natively.
	failures FailureTracker
	navURL   string
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
	case wmDPIChanged:
		applyDPIChange(hwnd, lp)
		if s != nil && s.chromium != nil {
			s.chromium.Resize()
		}
		return 0
	case wmSize:
		if s != nil && s.chromium != nil {
			s.chromium.Resize()
		}
	case wmMove:
		if s != nil && s.chromium != nil {
			_ = s.chromium.NotifyParentWindowPositionChanged()
		}
	case wmClose:
		if s != nil && s.res != nil {
			// Resident: closing the window hides it to the tray. The
			// runtime is not touched; Quit is the only way out.
			s.onClose()
			return 0
		}
		// Non-resident close behaviour: closing the window ends the desktop
		// session. Release the WebView2 controller (browser processes)
		// before the parent window goes away.
		if s != nil {
			s.closeWebView()
		}
		procDestroyWindow.Call(hwnd)
		return 0
	case wmQuitReq:
		if s != nil {
			s.quit()
		}
		return 0
	case wmReload:
		if s != nil && s.chromium != nil && !s.closed {
			fmt.Fprintf(os.Stderr, "hachidori: WebView2 reload: %s\n", s.navURL)
			s.chromium.Navigate(s.navURL)
		}
		return 0
	case wmTray:
		if s != nil && s.res != nil {
			switch uint32(lp) & 0xFFFF {
			case wmLButtonUp, wmLButtonDblClk:
				s.perform(s.res.OnActivate())
			case wmRButtonUp, wmContextMenu:
				s.showMenu()
			}
		}
		return 0
	case wmRefresh:
		if s != nil && s.res != nil {
			s.refresh()
		}
		return 0
	case wmDestroy:
		if s != nil {
			s.tray.remove()
		}
		procPostQuitMessage.Call(0)
		return 0
	}
	if s != nil && s.res != nil && m != 0 {
		switch uint32(m) {
		case s.activateMsg:
			// A second launch: show the existing window.
			s.perform(s.res.OnActivate())
			return 0
		case s.taskbarMsg:
			s.tray.added = false
			s.tray.add(s.res.Summary())
			return 0
		}
	}
	r, _, _ := procDefWindowProcW.Call(hwnd, m, wp, lp)
	return r
}

// onClose is the resident WM_CLOSE: hide to the tray (telling the user the
// first time). Without a tray icon there would be no way back to a hidden
// window, so it minimizes to the taskbar instead.
func (s *shell) onClose() {
	if !s.tray.added && !s.tray.add(s.res.Summary()) {
		procShowWindow.Call(s.hwnd, swMinimize)
		return
	}
	s.perform(s.res.OnClose())
}

// webViewFailed records a WebView2 failure and carries out the bounded
// response: a reload (posted, never run inside the WebView2 event), or a
// native notification when reloading cannot help. It runs on the UI thread.
func (s *shell) webViewFailed(f WebViewFailure) {
	fmt.Fprintf(os.Stderr, "hachidori: WebView2 %s failure: %s\n", f.Boundary, f.Detail)
	r := s.failures.Report(f)
	if r.Reload {
		procPostMessageW.Call(s.hwnd, wmReload, 0, 0)
	}
	if r.Notify {
		s.notifyFailure(r.Message)
	}
}

// notifyFailure makes a display failure visible outside the web page: a tray
// balloon, and a message box when the window is showing (so the user is not
// left looking at a blank surface). The message box runs off the UI thread.
func (s *shell) notifyFailure(msg string) {
	fmt.Fprintln(os.Stderr, "hachidori:", msg)
	if s.tray.added {
		s.tray.balloon("Hachidori cannot display its window", msg)
	}
	if v, _, _ := procIsWindowVisible.Call(s.hwnd); v != 0 {
		go native{}.ReportError("Hachidori", msg)
	}
}

// navigation completed event args: IUnknown (0-2), get_IsSuccess, get_WebErrorStatus.
const (
	slotNavDoneGetIsSuccess      = 3
	slotNavDoneGetWebErrorStatus = 4
)

// quit ends the desktop session: the WebView2 controller and tray icon are
// released and the window is destroyed, which ends the message loop. Open then
// returns and the caller shuts the runtime down.
func (s *shell) quit() {
	s.closeWebView()
	s.tray.remove()
	procDestroyWindow.Call(s.hwnd)
}

// perform carries out a Resident decision on the UI thread.
func (s *shell) perform(a Action) {
	switch a {
	case ActionShow:
		s.show(false)
	case ActionShowDiagnostics:
		s.show(true)
	case ActionHide:
		procShowWindow.Call(s.hwnd, swHide)
	case ActionHideWithNotice:
		procShowWindow.Call(s.hwnd, swHide)
		s.tray.balloon("Hachidori is still running", trayNotice)
	case ActionQuit:
		s.quit()
	}
}

// show restores and focuses the one existing window (never a second one).
func (s *shell) show(diagnostics bool) {
	if r, _, _ := procIsIconic.Call(s.hwnd); r != 0 {
		procShowWindow.Call(s.hwnd, swRestore)
	} else {
		procShowWindow.Call(s.hwnd, swShowNormal)
	}
	procSetForegroundWindow.Call(s.hwnd)
	if s.chromium != nil {
		s.chromium.Resize()
		s.chromium.Focus()
		if diagnostics {
			s.chromium.Navigate(s.url + DiagnosticsFragment)
		}
	}
}

// showMenu pops the tray menu and dispatches the choice.
func (s *shell) showMenu() {
	id := s.tray.popup(s.res.Menu())
	if id == 0 {
		return
	}
	a, err := s.res.OnMenu(id)
	if err != nil {
		fmt.Fprintln(os.Stderr, "hachidori: tray:", err)
		s.tray.balloon("Hachidori", err.Error())
		return
	}
	s.perform(a)
}

// refresh updates the tray tooltip from the application state and, on the
// transition into needing attention, says so once (no repeated prompts).
func (s *shell) refresh() {
	sum := s.res.Summary()
	if !s.tray.added {
		s.tray.add(sum)
	}
	s.tray.setTip(sum)
	if sum.Level == LevelAttention && s.lastLevel != LevelAttention {
		s.tray.balloon("Hachidori needs attention", "Open Hachidori to see diagnostics.")
	}
	s.lastLevel = sum.Level
}

func registerMessage(name string) uint32 {
	p, _ := windows.UTF16PtrFromString(name)
	r, _, _ := procRegisterWindowMessageW.Call(uintptr(unsafe.Pointer(p)))
	return uint32(r)
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
	if err := enablePerMonitorDPI(); err != nil {
		return fmt.Errorf("enabling per-monitor DPI awareness: %w", err)
	}
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
	uid, err := currentUserID()
	if err != nil {
		return err
	}
	// Per-user class: a second launch finds this window by it (see Activate).
	cls, _ := windows.UTF16PtrFromString(WindowClassName(uid))
	title, _ := windows.UTF16PtrFromString(w.Title)
	if w.Resident != nil {
		s.res, s.url = w.Resident, w.URL
		s.activateMsg = registerMessage(ActivateMessageName(uid))
		s.taskbarMsg = registerMessage("TaskbarCreated")
		s.tray.msg = wmTray
	}
	wndProcOnce.Do(func() { wndProcCB = windows.NewCallback(wndProc) })
	cursor, _, _ := procLoadCursorW.Call(0, idcArrow)
	wc := wndClassEx{WndProc: wndProcCB, Instance: inst, Cursor: windows.Handle(cursor),
		Background: windows.Handle(colorWindow + 1), ClassName: cls,
		Icon: windows.Handle(appIcon(inst, false)), IconSm: windows.Handle(appIcon(inst, true))}
	wc.Size = uint32(unsafe.Sizeof(wc))
	if r, _, e := procRegisterClassExW.Call(uintptr(unsafe.Pointer(&wc))); r == 0 {
		return fmt.Errorf("registering the window class: %w", e)
	}
	defer procUnregisterClassW.Call(uintptr(unsafe.Pointer(cls)), uintptr(inst))

	// The window is per-monitor DPI aware, so its initial size is physical
	// pixels: scale the logical 1200x860 for the current system DPI.
	dpi := systemDPI()
	hwnd, _, e := procCreateWindowExW.Call(0, uintptr(unsafe.Pointer(cls)), uintptr(unsafe.Pointer(title)),
		wsOverlappedWindow, cwUseDefault, cwUseDefault,
		uintptr(scaleForDPI(1200, dpi)), uintptr(scaleForDPI(860, dpi)), 0, 0, uintptr(inst), 0)
	if hwnd == 0 {
		return fmt.Errorf("creating the window: %w", e)
	}
	s.hwnd, s.navURL = hwnd, w.URL
	s.tray.hwnd = hwnd
	s.tray.icon = uintptr(wc.IconSm)

	// A visible launch must make the native parent presentable before WebView2
	// creates its controller. Creating a controller against a hidden Win32
	// parent can leave WebView2 alive and navigable but never presenting pixels,
	// which is observed by users as a permanent blank window. Background launches
	// intentionally keep the parent hidden and preserve the tray-only contract.
	if !w.StartHidden {
		procShowWindow.Call(hwnd, swShowNormal)
		procUpdateWindow.Call(hwnd)
	}

	destroyed := false
	defer func() {
		s.tray.remove()
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
		// Controller/environment failures. The binding exits the process
		// after most of these, so report natively first (a launch with no
		// console must not just disappear) and end the window session for
		// the ones that return.
		f := ControllerFailure(err)
		fmt.Fprintf(os.Stderr, "hachidori: WebView2 %s failure: %s\n", f.Boundary, f.Detail)
		s.failures.Report(f)
		native{}.ReportError("Hachidori", "WebView2 failed: "+f.Detail+". "+
			"Quit Hachidori and start it again, or run 'hachidori dashboard' and open the printed loopback URL in a browser.")
		if s.hwnd != 0 {
			procPostMessageW.Call(s.hwnd, wmQuitReq, 0, 0)
		}
	})
	c.NavigationCompletedCallback = func(_ *edge.ICoreWebView2, args *edge.ICoreWebView2NavigationCompletedEventArgs) {
		var ok, status int32
		hrOK := comCall(unsafe.Pointer(args), slotNavDoneGetIsSuccess, uintptr(unsafe.Pointer(&ok)))
		hrSt := comCall(unsafe.Pointer(args), slotNavDoneGetWebErrorStatus, uintptr(unsafe.Pointer(&status)))
		if hrOK != sOK || hrSt != sOK {
			fmt.Fprintf(os.Stderr, "hachidori: WebView2 navigation completed (result unreadable: %v, %v): %s\n", hrOK.errno(), hrSt.errno(), w.URL)
			return
		}
		if f, failed := NavigationFailure(ok != 0, status); failed {
			s.webViewFailed(f)
			return
		}
		fmt.Fprintf(os.Stderr, "hachidori: WebView2 navigation completed: success=%v status=%d %s\n", ok != 0, status, w.URL)
	}
	c.ProcessFailedCallback = func(_ *edge.ICoreWebView2, args *edge.ICoreWebView2ProcessFailedEventArgs) {
		kind, err := args.GetProcessFailedKind()
		if err != nil {
			s.webViewFailed(WebViewFailure{Boundary: BoundaryProcess, Reloadable: true,
				Detail: "a WebView2 process failed (kind unreadable: " + err.Error() + ")"})
			return
		}
		s.webViewFailed(ProcessFailure(uint32(kind)))
	}
	// The dashboard needs no camera, microphone, geolocation, clipboard or
	// notification permission.
	c.SetGlobalPermission(edge.CoreWebView2PermissionStateDeny)
	if !c.Embed(hwnd) {
		return errors.New("embedding WebView2 in the window failed")
	}
	s.controller = c.GetController()
	if s.controller == nil {
		return errors.New("WebView2 controller was not created")
	}
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
	// A resident shell starts in the tray only if the tray icon really exists;
	// otherwise a hidden window would be unreachable.
	hidden := false
	if s.res != nil {
		sum := s.res.Summary()
		s.lastLevel = sum.Level
		hidden = s.tray.add(sum) && w.StartHidden
	}
	if !hidden {
		// Visible launches were presented before controller creation. A requested
		// background launch can still become visible here when the tray icon could
		// not be created, preserving the existing reachability fallback.
		if w.StartHidden {
			procShowWindow.Call(hwnd, swShowNormal)
			procUpdateWindow.Call(hwnd)
		}
		c.Resize()
		c.Focus()
	}
	fmt.Fprintf(os.Stderr, "hachidori: WebView2 navigate: %s\n", w.URL)
	c.Navigate(w.URL)

	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			procPostMessageW.Call(hwnd, wmQuitReq, 0, 0)
		case <-done:
		}
	}()
	if s.res != nil {
		// The supervisor owns worker transitions and the controller only
		// signals its own actions, so the tray polls the canonical state and
		// only repaints when the window thread is told to.
		go func() {
			t := time.NewTicker(time.Second)
			defer t.Stop()
			last := s.res.Summary()
			for n := 1; ; n++ {
				select {
				case <-t.C:
					// Also every few seconds regardless, so an icon that
					// could not be created at sign-in is retried.
					if cur := s.res.Summary(); cur != last || n%5 == 0 {
						last = cur
						procPostMessageW.Call(hwnd, wmRefresh, 0, 0)
					}
				case <-done:
					return
				}
			}
		}()
	}

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
