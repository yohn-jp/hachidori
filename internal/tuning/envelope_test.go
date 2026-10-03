package tuning

import "testing"

const gib = uint64(1) << 30

func TestAutoBudgetKeepsSafetyMargin(t *testing.T) {
	rtx3060 := 12 * gib
	b := AutoBudget(rtx3060)
	if b >= rtx3060 || rtx3060-b != uint64(float64(rtx3060)*AutoMarginFraction) {
		t.Fatalf("AutoBudget(12 GiB) = %d, want total less 10%%", b)
	}
	if got := AutoBudget(4 * gib); got != 3*gib {
		t.Fatalf("AutoBudget(4 GiB) = %d, want the 1 GiB minimum margin", got)
	}
	if AutoBudget(0) != 0 || AutoBudget(gib) != 0 {
		t.Fatal("an unknown or tiny device has no Auto budget")
	}
}

func TestFullDeviceIsNeverInsideAutoEnvelope(t *testing.T) {
	e, err := NewEnvelope("NVIDIA GeForce RTX 3060", 12*gib, 0)
	if err != nil || !e.Auto || !e.Known() {
		t.Fatalf("envelope = %+v, %v", e, err)
	}
	if fit := e.Classify(Usage{Bytes: 12 * gib}); fit != FitExceeds {
		t.Fatalf("100%% occupancy classified %s, want exceeds", fit)
	}
	if fit := e.Classify(Usage{Bytes: 8 * gib}); fit != FitWithin {
		t.Fatalf("8 GiB measured classified %s, want within", fit)
	}
}

func TestClassifyLowerBoundAndUnknown(t *testing.T) {
	e, _ := NewEnvelope("gpu", 12*gib, 0)
	if fit := e.Classify(Usage{Bytes: 4 * gib, LowerBound: true}); fit != FitUnknown {
		t.Fatalf("inside lower bound = %s, want unknown", fit)
	}
	if fit := e.Classify(Usage{Bytes: 11 * gib, LowerBound: true}); fit != FitExceeds {
		t.Fatalf("exceeding lower bound = %s, want exceeds", fit)
	}
	if fit := e.Classify(Usage{}); fit != FitUnknown {
		t.Fatalf("no figure = %s, want unknown", fit)
	}
	if fit := (Envelope{}).Classify(Usage{Bytes: gib}); fit != FitUnknown {
		t.Fatalf("unknown envelope = %s, want unknown", fit)
	}
}

func TestExplicitBudget(t *testing.T) {
	e, err := NewEnvelope("gpu", 12*gib, 6*gib)
	if err != nil || e.Auto || e.BudgetBytes != 6*gib {
		t.Fatalf("envelope = %+v, %v", e, err)
	}
	if e.Classify(Usage{Bytes: 7 * gib}) != FitExceeds {
		t.Fatal("usage above an explicit budget must exceed")
	}
	if _, err := NewEnvelope("gpu", 12*gib, 13*gib); err == nil {
		t.Fatal("a budget above the device total must be refused")
	}
	// Without a known device total an explicit budget still applies.
	if e, err := NewEnvelope("", 0, 6*gib); err != nil || !e.Known() {
		t.Fatalf("explicit budget on an unknown device = %+v, %v", e, err)
	}
}
