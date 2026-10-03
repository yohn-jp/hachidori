package dashboard

// Presentation-only helpers for the workstation templates. Everything here is derived from
// the view the handlers already build (the /v1/status document, the tunnel
// status, the doctor run and the last action); nothing here reads or changes
// runtime state.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// SemanticState is the shared presentation vocabulary. It describes the
// provenance of a displayed value; it does not resolve or change that value.
type SemanticState string

const (
	Intent     SemanticState = "INTENT"
	Auto       SemanticState = "AUTO"
	Overridden SemanticState = "OVERRIDDEN"
	Measured   SemanticState = "MEASURED"
	Estimated  SemanticState = "ESTIMATED"
	Preserved  SemanticState = "PRESERVED"
	NotChecked SemanticState = "NOT_CHECKED"
)

func (s SemanticState) Tone() string {
	switch s {
	case Intent, Auto, Overridden:
		return "active"
	case Measured, Preserved:
		return "ok"
	case Estimated:
		return "warn"
	default:
		return "idle"
	}
}

// ControlProjection projects a backend-supplied intent, resolved Auto value,
// or explicit pinned value. Mode is Intent, Auto or Overridden. Value and
// Reason are literal facts, never translated or computed by the browser.
type ControlProjection struct {
	Label         string
	Mode          SemanticState
	Value, Reason string
}

// DisclosureProjection identifies a native disclosure. Callers compose their
// own controls inside operator-advanced-open and facts inside operator-details-open or
// operator-evidence-open, followed by operator-disclosure-close. No raw HTML is passed as data.
type DisclosureProjection struct {
	ID, Label string
	// Open renders the disclosure expanded: only when the operator has a
	// reason to act inside it.
	Open bool
}

// SectionProjection supplies the heading of a document/instrument section.
// Callers compose content between instrument-section-open and its close.
type SectionProjection struct {
	ID, Label, Description string
}

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
	// A runtime that is not running is not a problem by itself: stopped is a
	// stable state the operator chose (or the desktop opened in), and a pause
	// for model engineering is intentional. Only failures need attention, and a
	// worker the supervisor gave up on is listed through its last failure.
	if v.Running && w.State == worker.StateFailed {
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
	if p := memPressureOf(w.Accelerator); p != nil && p.Level != "ok" {
		a := alert{"warn", t("GPU memory pressure"), t("%.1f%% of device memory in use.", p.Pct)}
		if p.Level == "bad" {
			a = alert{"bad", t("GPU memory is above the safe budget"),
				t("%.1f%% of device memory in use; the Auto budget keeps %s free. Requests can fail or slow down. Plan a smaller candidate in Tuning.", p.Pct, p.Margin)}
			bad = append(bad, a)
		} else {
			warn = append(warn, a)
		}
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

// memPressure is device memory occupancy as its own axis, independent of
// READY: a ready worker on a full device is still an attention state. Level
// is bad above the Auto budget (inside the safety margin tuning keeps free),
// warn from 80%, else ok.
type memPressure struct {
	Used, Total, Margin string
	Pct                 float64
	Level               string // ok | warn | bad
	Word                string // catalog message ID
}

func memPressureOf(m map[string]any) *memPressure {
	total, ok1 := num(m, "memory_total")
	free, ok2 := num(m, "memory_free")
	if !ok1 || !ok2 || total <= 0 {
		return nil
	}
	used := max(total-free, 0)
	p := &memPressure{Used: tuning.Bytes(uint64(used)), Total: tuning.Bytes(uint64(total)), Pct: max(0, min(100, 100*used/total)), Level: "ok", Word: "Normal"}
	budget := tuning.AutoBudget(uint64(total))
	p.Margin = tuning.Bytes(uint64(total) - budget)
	switch {
	case budget > 0 && uint64(used) > budget:
		p.Level, p.Word = "bad", "Above safe budget"
	case p.Pct >= 80:
		p.Level, p.Word = "warn", "High"
	}
	return p
}

// runtimeAction is the one primary Runtime action the current state makes
// valid: Start a stopped runtime, Restart a failed one or one whose next
// start differs, open the Workbench when it is ready, nothing while it
// starts. Kind is start | restart | workbench | "".
type runtimeAction struct {
	Kind, Label, Reason string
}

func runtimeActionOf(v view) runtimeAction {
	w := v.S.Worker
	switch {
	case v.pause() != nil && !v.Running:
		// Serving resumes by itself when the operation ends; Start would be
		// refused while it owns the accelerator, so none is offered.
		return runtimeAction{"", "", pauseReason(v.pause())}
	case !v.Running:
		return runtimeAction{"start", "Start", "Serving is stopped. Planning, Tuning and Forge stay available; Start loads the active model."}
	case w.State == worker.StateFailed:
		return runtimeAction{"restart", "Restart", "The worker failed."}
	case v.Next != nil && v.Next.Model != "" && (v.Next.Differs || v.Next.VariantDiffers) && v.Next.Problem == "":
		return runtimeAction{"restart", "Restart to apply", "The next start differs from what serves now."}
	case w.Ready:
		return runtimeAction{"workbench", "Open Workbench", ""}
	}
	return runtimeAction{}
}

// pause is the intentional pause of serving this render read, nil when none.
func (v view) pause() *ModelPause {
	if v.models == nil {
		return nil
	}
	return v.models.Pause
}

// pauseReason is the operator's statement of why serving is paused (a catalog
// message ID): who owns the accelerator, and that serving returns by itself.
func pauseReason(p *ModelPause) string {
	if opOwner(p.Owner) == "forge" {
		return "Paused — Forge is using the GPU. Serving resumes automatically when it finishes."
	}
	return "Paused — model engineering is using the GPU. Serving resumes automatically when it finishes."
}

// shellStatus is the compact runtime identity the workstation shell shows on
// every page. It restates the /v1/status document and the attention list; it
// is not a second readiness authority.
type shellStatus struct {
	Word       string // "READY", else the worker state
	Tone       string // ok | warn | bad | idle
	Model      string // selected model id
	Provider   string // provider and version
	Device     string // device and dtype
	GPU        string // accelerator name
	Memory     string // GPU memory in use, when the device reports it
	MemoryTone string // ok | warn | bad: memory pressure, independent of Tone
	Attention  int    // items in the needs-attention list
	// Paused: serving is intentionally down while model engineering owns the
	// accelerator (Word is then PAUSED).
	Paused bool
	// Artifact is the execution artifact the running worker serves: SOURCE or
	// VARIANT <exact id>; empty while no worker runs.
	Artifact string
}

func shellOf(v view) shellStatus {
	w := v.S.Worker
	s := shellStatus{Word: w.State, Tone: stateTone(w), Model: v.S.Runtime.ModelID, Attention: len(alerts(v)),
		Provider: join(opt(w.Info, "provider"), opt(w.Info, "provider_version")),
		Device:   join(opt(w.Info, "device"), opt(w.Info, "dtype")), GPU: opt(w.Info, "device_name")}
	if w.Ready {
		s.Word = "READY"
	}
	if v.pause() != nil && !v.Running {
		s.Word, s.Tone, s.Paused = "PAUSED", "active", true
	}
	switch {
	case !v.Running:
	case v.S.Runtime.Variant != nil:
		s.Artifact = "VARIANT " + v.S.Runtime.Variant.ID
	case v.S.Runtime.ModelID != "":
		s.Artifact = "SOURCE"
	}
	if p := memPressureOf(w.Accelerator); p != nil {
		s.Memory, s.MemoryTone = fmt.Sprintf("%.0f%% GPU memory used", p.Pct), p.Level
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

// modelTitle is a catalog model's display name: the words of its catalog ID
// ("clef-flash" is "Clef Flash"). A catalog ID is a human slug, not an opaque
// identity, so this is presentation of the catalog's own name.
func modelTitle(id string) string {
	words := strings.FieldsFunc(id, func(r rune) bool { return r == '-' || r == '_' })
	for i, w := range words {
		words[i] = strings.ToUpper(w[:1]) + w[1:]
	}
	return strings.Join(words, " ")
}

// variantTitle is the operator's name of a variant, "Clef Flash · W4A16", from
// the structured source model and weight scheme its manifest records. It is ""
// unless both are known: the label is never reconstructed from the opaque
// variant ID, and a caller without one shows the exact ID.
func variantTitle(sourceID, scheme string) string {
	if sourceID == "" || scheme == "" {
		return ""
	}
	return modelTitle(sourceID) + " · " + scheme
}

// variantLabels names every variant of an inventory by its title. Variants of
// the same source and scheme would read identically, so exactly those carry the
// first eight digits of their manifest digest to tell them apart (or, with no
// digest to use, their exact ID); the exact ID stays in each variant's details
// and evidence.
func variantLabels(vs []setup.VariantEntry) map[string]string {
	count := map[string]int{}
	for _, v := range vs {
		count[variantTitle(v.SourceID, v.Scheme)]++
	}
	out := make(map[string]string, len(vs))
	for _, v := range vs {
		t := variantTitle(v.SourceID, v.Scheme)
		switch {
		case t == "":
			out[v.ID] = v.ID
		case count[t] > 1 && v.ManifestSHA256 == "":
			out[v.ID] = v.ID // nothing structured tells them apart: the exact identity does
		case count[t] > 1:
			out[v.ID] = t + " · #" + v.ManifestSHA256[:min(8, len(v.ManifestSHA256))]
		default:
			out[v.ID] = t
		}
	}
	return out
}
