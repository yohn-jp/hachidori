//go:build windows

package main

import "testing"

func TestNoArgLaunchIsDesktopOnWindows(t *testing.T) {
	if noArgLaunch == nil {
		t.Fatal("Windows builds must open the desktop with no arguments")
	}
}
