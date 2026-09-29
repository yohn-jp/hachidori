//go:build windows

package desktop

import (
	"errors"
	"os"
	"strconv"
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

// The startup entry lives only under HKEY_CURRENT_USER, so this needs no
// elevation. It uses its own value name, never the real "Hachidori" entry.
func TestRunKeyStartupIsIdempotentAndRemovable(t *testing.T) {
	s := runKey{name: "HachidoriTest" + strconv.Itoa(os.Getpid())}
	t.Cleanup(func() { _ = s.Disable() })
	if on, err := s.Enabled(); err != nil || on {
		t.Fatalf("initially enabled = %v, %v", on, err)
	}
	for i := 0; i < 2; i++ { // enabling twice is the same as once
		if err := s.Enable(`"C:\x\hachidori.exe" desktop --background`); err != nil {
			t.Fatalf("enable #%d: %v", i, err)
		}
	}
	if on, err := s.Enabled(); err != nil || !on {
		t.Fatalf("after enable: %v, %v", on, err)
	}
	if err := s.Enable(`"C:\y\hachidori.exe" desktop --background`); err != nil { // moved exe is repaired
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ { // disabling twice, and when absent, succeeds
		if err := s.Disable(); err != nil {
			t.Fatalf("disable #%d: %v", i, err)
		}
	}
	if on, err := s.Enabled(); err != nil || on {
		t.Fatalf("after disable: %v, %v", on, err)
	}
}

func TestNativeRuntimeDetectionDoesNotFail(t *testing.T) {
	// "" (not installed) and a version are both valid outcomes here.
	if _, err := Native().RuntimeVersion(); err != nil {
		t.Fatalf("RuntimeVersion: %v", err)
	}
}
