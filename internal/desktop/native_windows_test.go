//go:build windows

package desktop

import (
	"errors"
	"testing"
)

// These run only on Windows. They need no WebView2 Runtime, GPU or display.

func TestNativeInstanceGuard(t *testing.T) {
	p := Native()
	release, err := p.AcquireInstance()
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if _, err := p.AcquireInstance(); !errors.Is(err, ErrAlreadyRunning) {
		t.Fatalf("second acquire: err = %v, want ErrAlreadyRunning", err)
	}
	release()
	release, err = p.AcquireInstance()
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	release()
}

func TestNativeRuntimeDetectionDoesNotFail(t *testing.T) {
	// "" (not installed) and a version are both valid outcomes here.
	if _, err := Native().RuntimeVersion(); err != nil {
		t.Fatalf("RuntimeVersion: %v", err)
	}
}
