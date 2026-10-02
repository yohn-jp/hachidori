package setup

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

func TestPreflightReportOutcomeFollowsTheFindings(t *testing.T) {
	r := NewPreflightReport(PreflightOptimize, "m", time.Unix(0, 0))
	if r.Schema != PreflightSchema || len(r.NotMeasured) == 0 {
		t.Fatalf("new report %+v", r)
	}
	step := func(st FindingStatus, want string) {
		t.Helper()
		r.Add(Finding{ID: string(st), Area: AreaMemory, Status: st})
		if r.Outcome != want {
			t.Fatalf("after %s the outcome is %s, want %s (%+v)", st, r.Outcome, want, r.Counts)
		}
	}
	r.finalize()
	if r.Outcome != OutcomeReady {
		t.Fatalf("an empty report is %s", r.Outcome)
	}
	step(FindingPass, OutcomeReady)
	step(FindingUnknown, OutcomeAttention) // an unknown is never ready
	step(FindingWarning, OutcomeAttention)
	step(FindingBlocker, OutcomeBlocked)
	if !r.Blocked() || len(r.Blockers()) != 1 || r.Counts != (Counts{Pass: 1, Warning: 1, Blocker: 1, Unknown: 1}) {
		t.Fatalf("report %+v", r)
	}
	err := &PreflightError{Report: *r}
	if pe, ok := AsPreflightError(err); !ok || pe != err || !strings.Contains(err.Error(), "before any expensive work") {
		t.Fatalf("error %v", err)
	}
}

func TestDefaultHostObservesDiskAndMemoryWhereThePlatformReportsThem(t *testing.T) {
	h := DefaultHost()
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		if m := h.Mem(); !m.TotalKnown || m.Total == 0 {
			t.Fatalf("memory %+v", m)
		}
	}
	// A directory that does not exist yet is measured on its nearest existing ancestor.
	dir := filepath.Join(t.TempDir(), "not", "yet")
	if free, ok := h.FreeDiskAt(dir); runtime.GOOS != "plan9" && (!ok || free == 0) {
		t.Fatalf("free disk %d ok=%v", free, ok)
	}
	// The zero Host observes nothing: unknown, never zero.
	var zero Host
	if _, ok := zero.FreeDiskAt(dir); ok || zero.Mem().TotalKnown {
		t.Fatal("the zero host reported an observation")
	}
}

func TestWritableDirRefusesWhatCannotBeWritten(t *testing.T) {
	dir := t.TempDir()
	if err := WritableDir(filepath.Join(dir, "a", "b")); err != nil {
		t.Fatal(err)
	}
	if entries, _ := os.ReadDir(filepath.Join(dir, "a", "b")); len(entries) != 0 {
		t.Fatalf("the probe file was left behind: %v", entries)
	}
	f := filepath.Join(dir, "file")
	os.WriteFile(f, []byte("x"), 0o644)
	if err := WritableDir(filepath.Join(f, "sub")); err == nil {
		t.Fatal("a directory under a regular file was reported writable")
	}
}

// The accelerator probe runs the runtime's private interpreter isolated and
// reads its JSON; an interpreter that cannot run is an error, not a fact.
func TestProbeAcceleratorRunsThePrivateInterpreter(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	h.Ensure()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// The test binary answers as "python" (see TestMain); its probe output is
	// valid JSON without CUDA facts.
	py := filepath.Join(t.TempDir(), filepath.FromSlash(pythonRelPath()))
	os.MkdirAll(filepath.Dir(py), 0o755)
	if err := copyFile(exe, py); err != nil {
		t.Fatal(err)
	}
	f, err := ProbeAccelerator(context.Background(), h, py)
	if err != nil || f.CUDAAvailable {
		t.Fatalf("facts %+v err %v", f, err)
	}
	if _, err := ProbeAccelerator(context.Background(), h, filepath.Join(t.TempDir(), "missing-python")); err == nil {
		t.Fatal("a missing interpreter produced facts")
	}
}
