package dashboard

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/yohn-jp/hachidori/internal/tuning"
)

// The resource envelope projection: the target device and the memory budget
// Tuning plans against and Forge applies against. The device facts are the
// ones a serving worker reports (the last known accelerator stats) and, while
// none does (serving stopped, or paused for model engineering), the device
// capacity the host observes without a worker and without loading a model
// (DeviceObserver). The budget is Auto (tuning.AutoBudget) unless the operator
// stored an explicit override. Nothing here measures or invents a figure: an
// unknown device stays unknown, device capacity is never presented as a
// candidate's measured memory, and every fit that cannot be decided is
// NOT_CHECKED.

// MemoryBudgets is the optional settings capability that stores the explicit
// tuning memory budget (bytes; 0 is Auto).
type MemoryBudgets interface {
	MemoryBudget() (uint64, error)
	SetMemoryBudget(uint64) error
}

// EnvelopeView is the envelope as the workspaces show it.
type EnvelopeView struct {
	tuning.Envelope
	// Editable: the budget override can be stored.
	Editable bool
	// OverrideGiB is the explicit override as typed, "" under Auto.
	OverrideGiB string
	Err         string
}

// Total and Budget are display strings; "" when unknown.
func (e EnvelopeView) Total() string  { return bytesOrEmpty(e.TotalBytes) }
func (e EnvelopeView) Budget() string { return bytesOrEmpty(e.BudgetBytes) }

func bytesOrEmpty(b uint64) string {
	if b == 0 {
		return ""
	}
	return tuning.Bytes(b)
}

func (d *Dashboard) memoryBudgets() MemoryBudgets {
	if mb, ok := d.cfg.Settings.(MemoryBudgets); ok {
		return mb
	}
	return nil
}

// envelopeOf derives the envelope of the device the runtime reports.
func (d *Dashboard) envelopeOf(v view) EnvelopeView {
	w := v.S.Worker
	total, _ := num(w.Accelerator, "memory_total")
	device := opt(w.Info, "device_name")
	if device == "" {
		device = opt(w.Info, "device")
	}
	if total <= 0 {
		// No worker reports the accelerator: capacity does not require a
		// model to be resident.
		if o, ok := d.cfg.Models.(DeviceObserver); ok {
			if obs := o.Device(); obs.TotalBytes > 0 {
				total = float64(obs.TotalBytes)
				if opt(w.Info, "device_name") == "" {
					device = obs.Name
				}
			}
		}
	}
	var override uint64
	ev := EnvelopeView{}
	if mb := d.memoryBudgets(); mb != nil {
		ev.Editable = true
		var err error
		if override, err = mb.MemoryBudget(); err != nil {
			ev.Err = "The saved memory budget cannot be read: " + err.Error()
			override = 0
		}
	}
	e, err := tuning.NewEnvelope(device, uint64(max(total, 0)), override)
	if err != nil {
		// A stored override larger than the device: show the device with
		// no budget rather than silently replacing the override.
		ev.Err = err.Error()
		e = tuning.Envelope{Device: device, TotalBytes: uint64(max(total, 0)), Auto: false}
	}
	ev.Envelope = e
	if override > 0 {
		ev.OverrideGiB = strconv.FormatFloat(float64(override)/(1<<30), 'f', -1, 64)
	}
	return ev
}

// FitView is one memory figure placed against the envelope.
type FitView struct {
	Fit   tuning.Fit
	State SemanticState // MEASURED, ESTIMATED or NOT_CHECKED
	Value string        // the figure, "" when none
	Basis string
}

// Blocked is a fit that rules a candidate out of normal Apply.
func (f FitView) Blocked() bool { return f.Fit == tuning.FitExceeds }

// Tone is the presentation tone of the fit.
func (f FitView) Tone() string {
	switch f.Fit {
	case tuning.FitWithin:
		return "ok"
	case tuning.FitExceeds:
		return "bad"
	}
	return "idle"
}

// fitOf classifies one figure. state is the figure's provenance; an unknown
// figure is NOT_CHECKED whatever was passed.
func fitOf(e tuning.Envelope, u tuning.Usage, state SemanticState, basis string) FitView {
	f := FitView{Fit: e.Classify(u), State: state, Basis: basis}
	if u.Bytes == 0 {
		f.State = NotChecked
		return f
	}
	f.Value = tuning.Bytes(u.Bytes)
	if u.LowerBound {
		f.Value = "≥ " + f.Value
	}
	return f
}

// parseGiB reads an operator budget in GiB; "" and "auto" are Auto (0).
func parseGiB(s string) (uint64, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" || s == "auto" {
		return 0, nil
	}
	g, err := strconv.ParseFloat(s, 64)
	if err != nil || g <= 0 || g > 1<<20 {
		return 0, fmt.Errorf("the memory budget %q is not a positive number of GiB", s)
	}
	return uint64(g * (1 << 30)), nil
}

// tuningBudget stores the explicit memory budget (or Auto). It changes no
// profile, candidate or activation.
func (d *Dashboard) tuningBudget(w http.ResponseWriter, r *http.Request) {
	source, profile := r.PostFormValue("source"), r.PostFormValue("profile")
	mb := d.memoryBudgets()
	err := errors.New("the memory budget cannot be stored by this host")
	if mb != nil {
		var b uint64
		mode := r.PostFormValue("budget_mode")
		if b, err = parseGiB(r.PostFormValue("budget_gib")); err == nil {
			if mode != "explicit" {
				b = 0
			} else if b == 0 {
				err = errors.New("enter the explicit memory budget in GiB")
			}
		}
		if err == nil {
			v := d.statusView("Tuning", "tuning")
			if e := d.envelopeOf(v); b > 0 && e.TotalBytes > 0 && b > e.TotalBytes {
				err = fmt.Errorf("the memory budget (%s) exceeds the device's total memory (%s)", tuning.Bytes(b), tuning.Bytes(e.TotalBytes))
			}
		}
		if err == nil {
			err = mb.SetMemoryBudget(b)
		}
	}
	if err != nil {
		d.remember("set memory budget", err, "")
	} else {
		d.remember("set memory budget", nil, "the memory budget was saved. No profile or candidate changed")
	}
	http.Redirect(w, r, tuningLocation(source, profile), http.StatusSeeOther)
}
