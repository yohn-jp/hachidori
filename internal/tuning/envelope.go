package tuning

import "fmt"

// The resource envelope is the memory a candidate may occupy on the target
// device. It is the top-level constraint of tuning: a candidate that does not
// stay inside it is not a normal, applicable result, however it scores.
//
// The envelope is derived from facts the host reports about the device (its
// total memory) and an operator budget. The Auto budget keeps a deterministic
// safety margin, so a device that is full (100% occupancy) is never inside the
// envelope. Nothing here measures anything: callers pass observed or estimated
// usage and say which it is.

const (
	// AutoMarginFraction is the share of device memory the Auto budget keeps
	// free for the CUDA context, allocator fragmentation and other processes.
	AutoMarginFraction = 0.10
	// AutoMarginMinBytes is the smallest Auto margin, for small devices.
	AutoMarginMinBytes uint64 = 1 << 30
)

// AutoBudget is the Auto memory budget of a device with total bytes: the total
// less max(10%, 1 GiB). It is 0 (unknown) when the total is unknown or not
// larger than the margin.
func AutoBudget(total uint64) uint64 {
	margin := max(uint64(float64(total)*AutoMarginFraction), AutoMarginMinBytes)
	if total <= margin {
		return 0
	}
	return total - margin
}

// Envelope is the resource envelope of one target device.
type Envelope struct {
	// Device names the target device as the host reports it ("" if unknown).
	Device string
	// TotalBytes is the device's total memory, 0 when it is not known.
	TotalBytes uint64
	// BudgetBytes is the memory a candidate may occupy, 0 when unknown.
	BudgetBytes uint64
	// Auto is true when BudgetBytes is the Auto budget, false for an explicit
	// operator override.
	Auto bool
}

// NewEnvelope derives the envelope of a device. override 0 selects the Auto
// budget; an explicit override must not exceed the device's known total.
func NewEnvelope(device string, total, override uint64) (Envelope, error) {
	e := Envelope{Device: device, TotalBytes: total, Auto: override == 0}
	if e.Auto {
		e.BudgetBytes = AutoBudget(total)
		return e, nil
	}
	if total > 0 && override > total {
		return Envelope{}, fmt.Errorf("the memory budget (%s) exceeds the device's total memory (%s)", Bytes(override), Bytes(total))
	}
	e.BudgetBytes = override
	return e, nil
}

// Known reports whether the envelope has a budget to compare against.
func (e Envelope) Known() bool { return e.BudgetBytes > 0 }

// Fit is how a memory figure relates to the envelope.
type Fit string

const (
	// FitWithin: the measured usage is inside the budget.
	FitWithin Fit = "within"
	// FitExceeds: the usage (or a lower bound of it) is above the budget.
	FitExceeds Fit = "exceeds"
	// FitUnknown: the budget or the usage is unknown, or only a lower bound
	// that is inside the budget is known.
	FitUnknown Fit = "unknown"
)

// Usage is one memory figure of a candidate. LowerBound marks a figure that
// is known to understate the usage (for example the weights alone).
type Usage struct {
	Bytes      uint64
	LowerBound bool
}

// Classify places a usage figure against the envelope. An exceeding lower
// bound is decisive; an inside lower bound is not.
func (e Envelope) Classify(u Usage) Fit {
	switch {
	case !e.Known() || u.Bytes == 0:
		return FitUnknown
	case u.Bytes > e.BudgetBytes:
		return FitExceeds
	case u.LowerBound:
		return FitUnknown
	}
	return FitWithin
}

// Bytes formats a byte count in GiB with two decimals.
func Bytes(b uint64) string { return fmt.Sprintf("%.2f GiB", float64(b)/(1<<30)) }
