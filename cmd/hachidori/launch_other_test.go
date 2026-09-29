//go:build !windows

package main

import "testing"

// Proof 12: outside Windows the no-argument launch is not a desktop.
func TestNoArgLaunchIsWindowsOnly(t *testing.T) {
	if noArgLaunch != nil {
		t.Fatal("non-Windows builds must not have a no-argument desktop entry point")
	}
}
