//go:build windows

package main

import "github.com/yohn-jp/hachidori/internal/desktop"

// noArgLaunch: on Windows, hachidori.exe with no arguments is the desktop
// product entry point. Fatal startup errors (for example a missing WebView2
// Runtime, where no window can be shown) are reported in a message box, so a
// double-click never ends in a silent exit. A console created only for this
// Explorer launch is hidden; a console shared with an existing terminal is
// never hidden.
var noArgLaunch = func() error {
	desktop.HideOwnedConsole()
	err := runDesktopApp(desktop.Native(), desktop.NativePicker())
	if err != nil {
		desktop.ShowFatal("Hachidori", err.Error())
	}
	return err
}
