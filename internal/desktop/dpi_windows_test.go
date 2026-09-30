//go:build windows

package desktop

import "testing"

func TestPerMonitorDPIContext(t *testing.T) {
	if dpiAwarenessContextPerMonitorV2 != ^uintptr(3) {
		t.Fatalf("unexpected per-monitor V2 context: %#x", dpiAwarenessContextPerMonitorV2)
	}
}

func TestDPIChangeRectValidation(t *testing.T) {
	for name, r := range map[string]*dpiRect{
		"nil":        nil,
		"zero":       {},
		"horizontal": {right: 10, bottom: 0},
		"vertical":   {right: 0, bottom: 10},
		"valid":      {left: -20, top: 30, right: 1180, bottom: 890},
	} {
		want := name == "valid"
		if got := validDPIChangeRect(r); got != want {
			t.Errorf("%s: validDPIChangeRect() = %v, want %v", name, got, want)
		}
	}
}
