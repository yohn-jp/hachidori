package app

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/setup"
)

func writeSetupLog(t *testing.T, root, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(root, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(SetupLogPath(root), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The console is the setup log's own newest section of the operation's kind,
// bounded to the diagnostics' tail length and scrubbed by the same policy as a
// Forge diagnostic's log tail: it never shows more than a diagnostic would.
func TestConsoleTailIsBoundedRedactedAndOfTheNewestSection(t *testing.T) {
	root := t.TempDir()
	var b strings.Builder
	b.WriteString("== 2026-10-03T09:00:00Z forge_build_evaluate cuda clef-flash \nOLD SECTION LINE\n")
	b.WriteString("== 2026-10-03T09:30:00Z optimize cpu clef-flash \nother operation output\n")
	b.WriteString("== 2026-10-03T10:00:00Z forge_build_evaluate cuda clef-flash \n")
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&b, "step %d done\n", i)
	}
	fmt.Fprintf(&b, "loading %s\n", filepath.Join(root, "models", "clef-flash"))
	b.WriteString("fetch https://user:hunter2@example.invalid/x?token=abc123def456 HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\n")
	b.WriteString("Authorization: Bearer abcdefghijklmnop\n")
	b.WriteString("long " + strings.Repeat("x", 4000) + "\n")
	writeSetupLog(t, root, b.String())

	lines, at := ConsoleTail(root, OpForgeBuildEvaluate)
	if len(lines) != diagnostics.MaxForgeLogLines {
		t.Fatalf("tail has %d lines, want the bound %d", len(lines), diagnostics.MaxForgeLogLines)
	}
	all := strings.Join(lines, "\n")
	for _, absent := range []string{"OLD SECTION LINE", "other operation output", "step 0 done", "hunter2", "abc123def456", "hf_abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnop", root} {
		if strings.Contains(all, absent) {
			t.Errorf("console shows %q", absent)
		}
	}
	for _, present := range []string{"step 399 done", "<redacted>", "<HACHIDORI_HOME>"} {
		if present != "" && !strings.Contains(all, present) {
			t.Errorf("console lacks %q", present)
		}
	}
	for _, l := range lines {
		if len(l) > diagnostics.MaxForgeLineBytes+len("...[truncated]") {
			t.Fatalf("a console line has %d bytes", len(l))
		}
	}
	if at.IsZero() || time.Since(at) > time.Minute {
		t.Errorf("activity time %v is not the log's own write time", at)
	}
}

func TestConsoleTailWithoutALogIsEmpty(t *testing.T) {
	if lines, at := ConsoleTail(t.TempDir(), OpForgeBuildEvaluate); len(lines) != 0 || !at.IsZero() {
		t.Errorf("no log: %v %v", lines, at)
	}
	if lines, at := ConsoleTail("", OpForgeBuildEvaluate); len(lines) != 0 || !at.IsZero() {
		t.Errorf("no home: %v %v", lines, at)
	}
}

// Activity is the backend's own report: the operation's start, then each phase
// and step it enters. Nothing advances it between reports.
func TestOperationActivityIsWhatTheBackendReported(t *testing.T) {
	e := newMaintEnv(t)
	entered, release := make(chan struct{}), make(chan struct{})
	e.c.cfg.Maintenance.Materialize = func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		obs.OnPhase(setup.PhasePreparing)
		close(entered)
		<-release
		time.Sleep(5 * time.Millisecond)
		obs.OnProgress(setup.Progress{Step: setup.StepDownload, Detail: "model.safetensors", Done: 1})
		return nil
	}
	if err := e.c.Materialize(SetupParams{Device: "cuda", Model: "opendecider-nano"}); err != nil {
		t.Fatal(err)
	}
	<-entered
	op := e.c.Snapshot().Operation
	if op == nil || op.Activity.IsZero() || op.Activity.Before(op.Started) {
		t.Fatalf("a phase was reported but activity is %v (started %v)", op.Activity, op.Started)
	}
	first := op.Activity
	time.Sleep(20 * time.Millisecond)
	if again := e.c.Snapshot().Operation.Activity; !again.Equal(first) {
		t.Errorf("activity moved without a report: %v -> %v", first, again)
	}
	close(release)
	waitIdle(t, e.c)
	last := e.c.Snapshot().Maintenance
	if last == nil || !last.Activity.After(first) {
		t.Errorf("a step report did not advance activity: %v then %v", first, last)
	}
}
