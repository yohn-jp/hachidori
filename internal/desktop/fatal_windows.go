//go:build windows

package desktop

import "golang.org/x/sys/windows"

// ShowFatal reports a startup failure in a native message box, for launches
// with no terminal to read. It needs no window or WebView2.
func ShowFatal(title, message string) {
	t, err1 := windows.UTF16PtrFromString(title)
	m, err2 := windows.UTF16PtrFromString(message)
	if err1 != nil || err2 != nil {
		return
	}
	const mbOK, mbIconError = 0x0, 0x10
	_, _ = windows.MessageBox(0, m, t, mbOK|mbIconError)
}
