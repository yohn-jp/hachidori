//go:build !windows

package main

import (
	"errors"
	"testing"

	"github.com/yohn-jp/hachidori/internal/desktop"
)

// On non-Windows systems the desktop command fails clearly, before flag
// parsing, home resolution or any runtime component starts.
func TestDesktopCommandUnsupportedOutsideWindows(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	err := cmdDesktop(desktop.Native(), []string{"--home", t.TempDir()})
	if !errors.Is(err, desktop.ErrUnsupported) {
		t.Fatalf("err = %v, want ErrUnsupported", err)
	}
}
