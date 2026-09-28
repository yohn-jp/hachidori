package dashboard

// Presentation-only helpers for page.html. Everything here is derived from
// the view the handlers already build (the /v1/status document, the tunnel
// status, the doctor run and the last action); nothing here reads or changes
// runtime state.

import (
	"fmt"
	"strings"

	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// alert is one item of the "needs attention" list.
type alert struct {
	Level  string // "bad" or "warn"
	Title  string
	Detail string
}

// alerts summarizes the conditions an operator should look at, most severe
// first. It only restates what the status fragments already show in detail.
func alerts(v view) []alert {
	var bad, warn []alert
	w := v.S.Worker
	switch {
	case !v.Running:
		bad = append(bad, alert{"bad", "Runtime is not running",
			"The inference API stays bound and reports not ready until the runtime is started."})
	case w.State == worker.StateFailed:
		bad = append(bad, alert{"bad", "Worker failed", "Phase " + w.Phase + "."})
	}
	if f := w.LastFailure; f != nil {
		a := alert{"bad", "Last worker failure: " + f.Class, f.Message}
		if w.Ready {
			a.Level, a.Title = "warn", "Recovered from worker failure: "+f.Class
		}
		if a.Level == "bad" {
			bad = append(bad, a)
		} else {
			warn = append(warn, a)
		}
	}
	if w.QueueLimit > 0 && w.QueueDepth >= w.QueueLimit {
		warn = append(warn, alert{"warn", "Inference queue is full",
			fmt.Sprintf("%d of %d slots in use.", w.QueueDepth, w.QueueLimit)})
	}
	if m := gpuMem(w.Accelerator); m != nil && m.Used >= 95 {
		warn = append(warn, alert{"warn", "GPU memory pressure", fmt.Sprintf("%.1f%% of device memory in use.", m.Used)})
	}
	switch t := v.Tunnel; {
	case t.State == tunnel.StateExited:
		bad = append(bad, alert{"bad", "SSH tunnel exited", t.LastError})
	case t.LastError != "":
		warn = append(warn, alert{"warn", "SSH tunnel error", t.LastError})
	}
	if d := v.Doctor; !d.Running && !d.Finished.IsZero() && !d.OK {
		bad = append(bad, alert{"bad", "Doctor found problems", "See the doctor report below."})
	}
	return append(bad, warn...)
}

// memView is the GPU-memory breakdown derived from the accelerator stats, as
// percentages of device total: tensors (allocated), the PyTorch cache
// (reserved - allocated), everything else on the device (total - free -
// reserved: CUDA context, other processes) and free.
type memView struct {
	Alloc, Cached, Other, Free float64
	Used                       float64 // (total - free) / total
	Level                      string  // "ok", "warn", "bad"
}

func gpuMem(m map[string]any) *memView {
	total, ok1 := num(m, "memory_total")
	free, ok2 := num(m, "memory_free")
	if !ok1 || !ok2 || total <= 0 {
		return nil
	}
	alloc, _ := num(m, "memory_allocated")
	res, _ := num(m, "memory_reserved")
	pct := func(x float64) float64 { return max(0, min(100, 100*x/total)) }
	v := &memView{Alloc: pct(alloc), Cached: pct(res - alloc), Free: pct(free), Used: pct(total - free)}
	v.Other = max(0, 100-v.Alloc-v.Cached-v.Free)
	switch {
	case v.Used >= 95:
		v.Level = "bad"
	case v.Used >= 85:
		v.Level = "warn"
	default:
		v.Level = "ok"
	}
	return v
}

func num(m map[string]any, key string) (float64, bool) {
	switch v := m[key].(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	}
	return 0, false
}

// ratio is a/b as a clamped percentage (0 when b is not positive).
func ratio(a, b int) float64 {
	if b <= 0 {
		return 0
	}
	return max(0, min(100, 100*float64(a)/float64(b)))
}

func errTotal(m map[string]int64) int64 {
	var n int64
	for _, v := range m {
		n += v
	}
	return n
}

// uptime renders a duration in seconds compactly ("2d 3h", "1h 04m", "5m 09s").
func uptime(s int) string {
	d, h, m := s/86400, s/3600%24, s/60%60
	switch {
	case d > 0:
		return fmt.Sprintf("%dd %dh", d, h)
	case h > 0:
		return fmt.Sprintf("%dh %02dm", h, m)
	case m > 0:
		return fmt.Sprintf("%dm %02ds", m, s%60)
	}
	return fmt.Sprintf("%ds", s)
}

type doctorLine struct {
	Kind string // "pass", "fail", "skip" or "" for continuation lines
	Text string
}

type doctorReport struct {
	Lines            []doctorLine
	Pass, Fail, Skip int
}

// doctorOut splits doctor output (one "STATUS name [owner] detail" line per
// check) so the page can tint each check and count outcomes.
func doctorOut(out string) doctorReport {
	var r doctorReport
	for _, l := range strings.Split(strings.TrimRight(out, "\n"), "\n") {
		k := ""
		switch f := strings.Fields(l); {
		case len(f) == 0:
		case f[0] == "PASS":
			k = "pass"
			r.Pass++
		case f[0] == "FAIL":
			k = "fail"
			r.Fail++
		case f[0] == "SKIP":
			k = "skip"
			r.Skip++
		}
		r.Lines = append(r.Lines, doctorLine{k, l})
	}
	return r
}
