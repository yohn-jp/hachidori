//go:build windows

package desktop

import (
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// This file is the only hand-written WebView2 COM glue. The binding does not
// expose NavigationStarting or NewWindowRequested, which the navigation
// policy needs, so the shell registers two minimal event handlers itself.
// Vtable slot numbers follow the order of the interfaces in WebView2.idl.

const (
	// ICoreWebView2
	slotAddNavigationStarting      = 7
	slotAddFrameNavigationStarting = 17
	slotAddNewWindowRequested      = 44
	// ICoreWebView2NavigationStartingEventArgs
	slotNavArgsGetURI    = 3
	slotNavArgsPutCancel = 8
	// ICoreWebView2NewWindowRequestedEventArgs
	slotNewWinArgsGetURI     = 3
	slotNewWinArgsPutHandled = 6
	// ICoreWebView2Controller
	slotControllerClose = 24

	sOK          = 0
	eNoInterface = 0x80004002
)

var (
	iidIUnknown             = windows.GUID{Data1: 0x00000000, Data2: 0x0000, Data3: 0x0000, Data4: [8]byte{0xC0, 0, 0, 0, 0, 0, 0, 0x46}}
	iidNavigationStartingEH = windows.GUID{Data1: 0x9adbe429, Data2: 0xf36d, Data3: 0x432b, Data4: [8]byte{0x9d, 0xdc, 0xf8, 0x88, 0x1f, 0xbd, 0x76, 0xe3}}
	iidNewWindowRequestedEH = windows.GUID{Data1: 0xd4c185fe, Data2: 0xc81c, Data3: 0x4989, Data4: [8]byte{0x97, 0xaf, 0x2d, 0x3f, 0xa7, 0xab, 0x56, 0x51}}
)

// comCall invokes vtable slot of the COM object obj.
//
//go:uintptrescapes
func comCall(obj unsafe.Pointer, slot int, args ...uintptr) uintptr {
	vtbl := *(*unsafe.Pointer)(obj)
	fn := *(*uintptr)(unsafe.Add(vtbl, uintptr(slot)*unsafe.Sizeof(uintptr(0))))
	r, _, _ := syscall.SyscallN(fn, append([]uintptr{uintptr(obj)}, args...)...)
	return r
}

// eventHandler is a COM object implementing one ICoreWebView2*EventHandler.
// Instances are allocated on the Go heap (never moved) and kept reachable by
// the window for as long as WebView2 may call them; reference counting is
// therefore a no-op.
type eventHandler struct {
	vtbl   *eventHandlerVtbl
	iid    windows.GUID
	invoke func(sender, args unsafe.Pointer)
}

type eventHandlerVtbl struct {
	QueryInterface, AddRef, Release, Invoke uintptr
}

var (
	handlerVtblOnce sync.Once
	handlerVtbl     *eventHandlerVtbl
)

func newEventHandler(iid windows.GUID, invoke func(sender, args unsafe.Pointer)) *eventHandler {
	handlerVtblOnce.Do(func() {
		handlerVtbl = &eventHandlerVtbl{
			QueryInterface: windows.NewCallback(func(this *eventHandler, riid *windows.GUID, out *unsafe.Pointer) uintptr {
				if *riid == iidIUnknown || *riid == this.iid {
					*out = unsafe.Pointer(this)
					return sOK
				}
				*out = nil
				return eNoInterface
			}),
			AddRef:  windows.NewCallback(func(this *eventHandler) uintptr { return 1 }),
			Release: windows.NewCallback(func(this *eventHandler) uintptr { return 1 }),
			Invoke: windows.NewCallback(func(this *eventHandler, sender, args unsafe.Pointer) uintptr {
				this.invoke(sender, args)
				return sOK
			}),
		}
	})
	return &eventHandler{vtbl: handlerVtbl, iid: iid, invoke: invoke}
}

// argsURI reads get_Uri (slot) of an event-args object.
func argsURI(args unsafe.Pointer, slot int) (string, bool) {
	var p *uint16
	if hr := comCall(args, slot, uintptr(unsafe.Pointer(&p))); hr != sOK || p == nil {
		return "", false
	}
	defer windows.CoTaskMemFree(unsafe.Pointer(p))
	return windows.UTF16PtrToString(p), true
}

// navigationGuard enforces Policy on every top-level and frame navigation and
// refuses every new-window request.
type navigationGuard struct {
	policy    Policy
	onBlocked func(kind, uri string)
	nav       *eventHandler
	newWindow *eventHandler
	tokens    [3]int64
}

func newNavigationGuard(p Policy, onBlocked func(kind, uri string)) *navigationGuard {
	g := &navigationGuard{policy: p, onBlocked: onBlocked}
	g.nav = newEventHandler(iidNavigationStartingEH, func(_, args unsafe.Pointer) {
		uri, ok := argsURI(args, slotNavArgsGetURI)
		if ok && g.policy.AllowNavigation(uri) {
			return
		}
		comCall(args, slotNavArgsPutCancel, 1)
		g.onBlocked("navigation", uri)
	})
	g.newWindow = newEventHandler(iidNewWindowRequestedEH, func(_, args unsafe.Pointer) {
		uri, _ := argsURI(args, slotNewWinArgsGetURI)
		if g.policy.AllowNewWindow(uri) {
			return
		}
		// Handled without NewWindow: WebView2 opens nothing and window.open
		// returns null in the page.
		comCall(args, slotNewWinArgsPutHandled, 1)
		g.onBlocked("new window", uri)
	})
	return g
}

// attach registers the guard on an ICoreWebView2. It must succeed before the
// first navigation; any failure keeps the window from showing anything.
func (g *navigationGuard) attach(webview unsafe.Pointer) error {
	for i, reg := range []struct {
		slot int
		h    *eventHandler
	}{
		{slotAddNavigationStarting, g.nav},
		{slotAddFrameNavigationStarting, g.nav},
		{slotAddNewWindowRequested, g.newWindow},
	} {
		if hr := comCall(webview, reg.slot, uintptr(unsafe.Pointer(reg.h)), uintptr(unsafe.Pointer(&g.tokens[i]))); hr != sOK {
			return syscall.Errno(hr)
		}
	}
	return nil
}
