package main

import (
	"flag"
	"runtime"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/server"
)

// cmdDesktop is `hachidori desktop`: the same desktop composition as the
// no-argument launch (desktopApp), with flags for the home, listen addresses
// and background (start at sign-in) mode. It has no lifecycle rules of its own.
func cmdDesktop(p desktop.Platform, args []string) error {
	if runtime.GOOS != "windows" {
		return desktop.ErrUnsupported
	}
	prefsPath, err := desktop.PrefsPath()
	if err != nil {
		return err
	}
	return runDesktop(p, desktop.NativePicker(), desktop.NativeStartup(), prefsPath, args)
}

// runDesktop is cmdDesktop with its OS surfaces injected (tests use fakes).
func runDesktop(p desktop.Platform, pk desktop.FolderPicker, st desktop.Startup, prefsPath string, args []string) error {
	fs := flag.NewFlagSet("desktop", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME, then the desktop bootstrap locator)")
	listen := fs.String("listen", server.DefaultListen, "loopback address to bind")
	dashAddr := fs.String("addr", dashboard.DefaultListen, "loopback address of the dashboard")
	sshExe := fs.String("ssh", "ssh", "host ssh client used by the tunnel launcher")
	background := fs.Bool("background", false, "launched at sign-in: honor the Start minimized preference")
	fs.Parse(args)

	a := newDesktopApp(p, pk, st, prefsPath, *homeFlag)
	a.APIAddr, a.DashAddr, a.SSH, a.Background = *listen, *dashAddr, *sshExe, *background
	return a.launch()
}
