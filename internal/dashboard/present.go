package dashboard

// Presentation-only helpers for the workstation templates. Everything here is derived from
// the view the handlers already build (the /v1/status document, the tunnel
// status, the doctor run and the last action); nothing here reads or changes
// runtime state.

import (
	"fmt"
	"sort"
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
	w, t := v.S.Worker, v.Lang.T
	switch {
	case !v.Running:
		bad = append(bad, alert{"bad", t("Runtime is not running"),
			t("The inference API stays bound and reports not ready until the runtime is started.")})
	case w.State == worker.StateFailed:
		bad = append(bad, alert{"bad", t("Worker failed"), t("Phase %s.", w.Phase)})
	}
	// Another resident that failed is named; the default resident's own
	// failure is already the alert above, and healthy residents add nothing.
	for _, r := range v.S.Residents {
		if !r.Default && r.Running && r.Status.Worker.State == worker.StateFailed {
			bad = append(bad, alert{"bad", t("Resident %s failed", r.Model), t("Phase %s.", r.Status.Worker.Phase)})
		}
	}
	if f := w.LastFailure; f != nil {
		a := alert{"bad", t("Last worker failure: %s", f.Class), f.Message}
		if w.Ready {
			a.Level, a.Title = "warn", t("Recovered from worker failure: %s", f.Class)
		}
		if a.Level == "bad" {
			bad = append(bad, a)
		} else {
			warn = append(warn, a)
		}
	}
	if w.QueueLimit > 0 && w.QueueDepth >= w.QueueLimit {
		warn = append(warn, alert{"warn", t("Inference queue is full"),
			t("%d of %d slots in use.", w.QueueDepth, w.QueueLimit)})
	}
	if m := gpuMem(w.Accelerator); m != nil && m.Used >= 95 {
		warn = append(warn, alert{"warn", t("GPU memory pressure"), t("%.1f%% of device memory in use.", m.Used)})
	}
	switch tn := v.Tunnel; {
	case tn.State == tunnel.StateExited:
		bad = append(bad, alert{"bad", t("SSH tunnel exited"), tn.LastError})
	case tn.LastError != "":
		warn = append(warn, alert{"warn", t("SSH tunnel error"), tn.LastError})
	}
	if d := v.Doctor; !d.Running && !d.Finished.IsZero() && !d.OK {
		bad = append(bad, alert{"bad", t("Doctor found problems"), t("See the doctor report in Diagnostics.")})
	}
	return append(bad, warn...)
}

// shellStatus is the compact runtime identity the workstation shell shows on
// every page. It restates the /v1/status document and the attention list; it
// is not a second readiness authority.
type shellStatus struct {
	Word      string // "READY", else the worker state
	Tone      string // ok | warn | bad | idle
	Model     string // selected model id
	Provider  string // provider and version
	Device    string // device and dtype
	GPU       string // accelerator name
	Memory    string // GPU memory in use, when the device reports it
	Attention int    // items in the needs-attention list
}

func shellOf(v view) shellStatus {
	w := v.S.Worker
	s := shellStatus{Word: w.State, Tone: stateTone(w), Model: v.S.Runtime.ModelID, Attention: len(alerts(v)),
		Provider: join(opt(w.Info, "provider"), opt(w.Info, "provider_version")),
		Device:   join(opt(w.Info, "device"), opt(w.Info, "dtype")), GPU: opt(w.Info, "device_name")}
	if w.Ready {
		s.Word = "READY"
	}
	if m := gpuMem(w.Accelerator); m != nil {
		s.Memory = fmt.Sprintf("%.0f%% GPU memory used", m.Used)
	}
	return s
}

// stateTone is the status vocabulary shared by every workspace.
func stateTone(w worker.Snapshot) string {
	switch {
	case w.Ready:
		return "ok"
	case w.State == worker.StateFailed:
		return "bad"
	case w.State == worker.StateStarting || w.State == worker.StateRestarting:
		return "warn"
	}
	return "idle"
}

// opt is one value of a free-form worker map, empty when absent.
func opt(m map[string]any, key string) string {
	if v, ok := m[key]; ok && v != nil {
		return fmt.Sprint(v)
	}
	return ""
}

func join(parts ...string) string {
	var out []string
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, " ")
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

// ratio64 is a/b as a clamped percentage (0 when b is not positive).
func ratio64(a, b int64) float64 {
	if b <= 0 {
		return 0
	}
	return max(0, min(100, 100*float64(a)/float64(b)))
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

// probRow is one choice of a typed result with its probability.
type probRow struct {
	Choice string
	P      float64
	Chosen bool
}

// probRows orders a result's probabilities most likely first (ties by
// label), marking the reported choice.
func probRows(probs map[string]float64, choice string) []probRow {
	rows := make([]probRow, 0, len(probs))
	for c, p := range probs {
		rows = append(rows, probRow{Choice: c, P: p, Chosen: c == choice})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].P != rows[j].P {
			return rows[i].P > rows[j].P
		}
		return rows[i].Choice < rows[j].Choice
	})
	return rows
}

// distView is one rendered probability distribution. Expected is set only
// when the distribution comes from evidence that carries an expected label.
type distView struct {
	Rows     []probRow
	Expected string
}

func distOf(probs map[string]float64, choice, expected string) distView {
	return distView{Rows: probRows(probs, choice), Expected: expected}
}
