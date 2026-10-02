package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// stdoutOf runs f and returns what it printed to stdout.
func stdoutOf(t *testing.T, f func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	done := make(chan string)
	go func() { var b bytes.Buffer; io.Copy(&b, r); done <- b.String() }()
	defer func() { os.Stdout = old }()
	f()
	w.Close()
	os.Stdout = old
	return <-done
}

func TestForgePreflightThroughTheCLI(t *testing.T) {
	h, _ := forgeSource(t)
	var code int
	out := stdoutOf(t, func() { code = run([]string{"forge", "preflight", "optimize", "-home", h.Root, "-json"}, nil) })
	var rep setup.PreflightReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("not machine readable: %v\n%s", err, out)
	}
	if rep.Schema != setup.PreflightSchema || rep.Kind != setup.PreflightOptimize || rep.Model != setup.ClefFlash || len(rep.Findings) == 0 {
		t.Fatalf("report %+v", rep)
	}
	// Whatever the host, an unknown RAM fit is never a pass, and a report with
	// an unknown is never "ready".
	for _, f := range rep.Findings {
		if f.ID == "memory.fit" && f.Status == setup.FindingPass {
			t.Fatal("memory.fit passed")
		}
	}
	if (rep.Blocked()) != (code != 0) || rep.Outcome == setup.OutcomeReady {
		t.Fatalf("exit %d for outcome %s", code, rep.Outcome)
	}
	// The report is recorded where the desktop reads it.
	if got, ok := app.LatestPreflight(h, app.PreflightTarget{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: rep.Binding.Recipe}); !ok || got.CreatedAt != rep.CreatedAt || rep.Binding.Recipe == "" {
		t.Fatalf("latest preflight %+v ok=%v", got, ok)
	}

	// A blocked preflight exits non-zero and still prints the report.
	out = stdoutOf(t, func() {
		code = run([]string{"forge", "preflight", "probe", "-home", h.Root, "-variant", "clef-flash--none--000000000000", "-device", "cpu"}, nil)
	})
	if code == 0 || !strings.Contains(out, "BLOCKED") || !strings.Contains(out, "BLOCKER") {
		t.Fatalf("exit %d output:\n%s", code, out)
	}
	for _, args := range [][]string{{"forge"}, {"forge", "nope"}, {"forge", "preflight"}, {"forge", "preflight", "nope"}, {"forge", "preflight", "probe", "-home", h.Root},
		{"forge", "probe", "-home", h.Root}, {"forge", "diagnostics"}, {"forge", "diagnostics", "nope", "-home", h.Root},
		{"forge", "execute"}, {"forge", "execute", "-home", h.Root, "-device", "cuda", "/nonexistent.jsonl"}} {
		if code := run(args, nil); code == 0 {
			t.Errorf("%v succeeded", args)
		}
	}
}

// A failed command leaves a diagnostic that the CLI can list, show and export,
// the latest one by default and a chosen one by its stable identity.
func TestForgeDiagnosticsThroughTheCLI(t *testing.T) {
	h, _ := forgeSource(t)
	// certify evaluate with missing run files fails after its identities are known.
	if code := run([]string{"certify", "evaluate", "-home", h.Root, "-variant", "clef-flash--none--000000000000", "-reference", "nope.json", "-candidate", "nope.json"}, nil); code != 1 {
		t.Fatalf("exit %d", code)
	}
	list := diagnostics.ListForge(h.Root, diagnostics.ForgeFilter{})
	if len(list) != 1 || list[0].Kind != app.OpCertify {
		t.Fatalf("diagnostics %+v", list)
	}
	id := list[0].ID

	var code int
	out := stdoutOf(t, func() { code = run([]string{"forge", "diagnostics", "list", "-home", h.Root}, nil) })
	if code != 0 || !strings.Contains(out, id) {
		t.Fatalf("list exit %d:\n%s", code, out)
	}
	for _, args := range [][]string{{"forge", "diagnostics", "show", "-home", h.Root}, {"forge", "diagnostics", "show", "-home", h.Root, id}} {
		out = stdoutOf(t, func() { code = run(args, nil) })
		var d diagnostics.ForgeDiagnostic
		if err := json.Unmarshal([]byte(out), &d); code != 0 || err != nil || d.ID != id || d.Operation.Kind != app.OpCertify || strings.Contains(out, h.Root) {
			t.Fatalf("%v exit %d err %v:\n%s", args, code, err, out)
		}
	}
	dir := t.TempDir()
	if code := run([]string{"forge", "diagnostics", "export", "-home", h.Root, "-out", dir}, nil); code != 0 {
		t.Fatalf("export exit %d", code)
	}
	if b, err := os.ReadFile(filepath.Join(dir, id+".json")); err != nil || !strings.Contains(string(b), diagnostics.ForgeSchema) {
		t.Fatalf("exported file: %v", err)
	}
	for _, args := range [][]string{{"forge", "diagnostics", "export", "-home", h.Root}, {"forge", "diagnostics", "show", "-home", h.Root, "optimize-20200101T000000Z-00000000"},
		{"forge", "diagnostics", "show", "-home", h.Root, "-kind", "probe"}, {"forge", "diagnostics", "show", "-home", h.Root, "../../x"}} {
		if code := run(args, nil); code == 0 {
			t.Errorf("%v succeeded", args)
		}
	}
}

// The desktop's Forge projection restates the controller's records: findings
// that are not a pass, probes with their identity, diagnostics.
func TestDesktopForgeStateIsARestatement(t *testing.T) {
	rep := setup.NewPreflightReport(setup.PreflightProbe, setup.ClefFlash, time.Unix(0, 0))
	rep.Add(setup.Finding{ID: "source.digests", Area: setup.AreaIdentity, Status: setup.FindingPass, Summary: "ok"})
	rep.Add(setup.Finding{ID: "memory.fit", Area: setup.AreaMemory, Status: setup.FindingUnknown, Summary: "not known"})
	got := forgeState(app.ForgeState{
		Preflights:  []app.RecordedPreflight{{PreflightReport: *rep, Evidence: setup.EvidenceCurrent}},
		Probes:      []app.ProbeRecord{{Variant: "v", Device: "cuda", Result: app.ProbePassed, VariantManifestSHA256: "sha", Decision: &app.ProbeDecision{Choice: "yes", Confidence: 0.8}}},
		Diagnostics: []diagnostics.ForgeSummary{{ID: "probe-20261002T000000Z-0123abcd", Kind: "probe"}},
	})
	if len(got.Preflights) != 1 || got.Preflights[0].Outcome != setup.OutcomeAttention || len(got.Preflights[0].Findings) != 1 || got.Preflights[0].Findings[0].ID != "memory.fit" ||
		got.Preflights[0].Pass != 1 || got.Preflights[0].Unknown != 1 {
		t.Fatalf("preflights %+v", got.Preflights)
	}
	if len(got.Probes) != 1 || got.Probes[0].ManifestSHA256 != "sha" || got.Probes[0].Choice != "yes" || len(got.Diagnostics) != 1 {
		t.Fatalf("state %+v", got)
	}
}

// A preflight that is not current evidence is never projected as READY: its
// state replaces the outcome and the reason is the first finding.
func TestDesktopForgeStateNeverShowsStalePreflightAsReady(t *testing.T) {
	rep := setup.NewPreflightReport(setup.PreflightProbe, setup.ClefFlash, time.Unix(0, 0))
	rep.Add(setup.Finding{ID: "source.digests", Area: setup.AreaIdentity, Status: setup.FindingPass, Summary: "ok"})
	if rep.Outcome != setup.OutcomeReady {
		t.Fatal("fixture is not ready")
	}
	got := forgeState(app.ForgeState{Preflights: []app.RecordedPreflight{
		{PreflightReport: *rep, Evidence: setup.EvidenceStale, Stale: []string{"runtime", "device"}},
		{PreflightReport: *rep, Evidence: setup.EvidenceLegacy},
		{PreflightReport: *rep, Evidence: setup.EvidenceCurrent},
	}})
	for i, want := range []string{setup.EvidenceStale, setup.EvidenceLegacy} {
		row := got.Preflights[i]
		if row.Outcome != want || len(row.Findings) != 1 || row.Findings[0].ID != "preflight.evidence" || row.Findings[0].Status != want ||
			!strings.Contains(row.Findings[0].Summary, "not current") {
			t.Fatalf("%s row %+v", want, row)
		}
	}
	if !strings.Contains(got.Preflights[0].Findings[0].Summary, "runtime, device") {
		t.Fatalf("stale reason %q", got.Preflights[0].Findings[0].Summary)
	}
	if got.Preflights[2].Outcome != setup.OutcomeReady || len(got.Preflights[2].Findings) != 0 {
		t.Fatalf("current row %+v", got.Preflights[2])
	}
}
