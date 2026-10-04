//go:build windows

package desktop

import "testing"

func TestQuitBeforeControllerReadyIsDeferred(t *testing.T) {
	s := &shell{hwnd: 1}
	s.quit()

	if !s.quitPending {
		t.Fatal("Quit before controller readiness must be recorded as pending")
	}
	if s.closed {
		t.Fatal("Quit before controller readiness must not close WebView2")
	}
}

func TestRepeatedEarlyQuitRemainsIdempotentlyPending(t *testing.T) {
	s := &shell{hwnd: 1}
	s.quit()
	s.quit()

	if !s.quitPending {
		t.Fatal("repeated early Quit must remain pending")
	}
	if s.closed {
		t.Fatal("repeated early Quit must not tear down the parent window")
	}
}
