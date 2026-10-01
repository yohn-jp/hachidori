package main

import (
	"context"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/desktop"
)

// BenchmarkLaunchToWindow is the Go half of the "process start -> first
// visible window" budget in docs/desktop.md: a configured, installed launch
// from entry to the native window request (preflight, home discovery,
// controller start, listeners, settings). The WebView2 half is measured on
// Windows from the shell's startup log marks.
func BenchmarkLaunchToWindow(b *testing.B) {
	var total time.Duration
	for i := 0; i < b.N; i++ {
		f := &fakeDesktop{}
		a, _, _ := installedApp(b, f, false)
		var start time.Time
		f.open = func(_ context.Context, w desktop.Window) error {
			total += time.Since(start)
			if r := w.Resident; r != nil {
				r.Wait()
				r.OnMenu(desktop.MenuQuit)
			}
			return nil
		}
		start = time.Now()
		if err := a.launch(); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(total.Microseconds())/float64(b.N), "µs-to-window/op")
}
