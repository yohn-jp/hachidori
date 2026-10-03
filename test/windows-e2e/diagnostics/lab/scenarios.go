package lab

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
	"github.com/yohn-jp/hachidori/test/windows-e2e/diagnostics/bundle"
	"github.com/yohn-jp/hachidori/test/windows-e2e/diagnostics/fixture"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/disposable"
)

// Candidate is the identity of the executable under certification.
type Candidate struct {
	Path      string // the certified file; it is copied, never run in place
	File      string // its file name
	SHA256    string
	Size      int64
	GOOS      string
	GOARCH    string
	GoVersion string // "" when the manifest records none
}

// export is one completed Export bundle against the real executable.
type export struct {
	scratch  *disposable.Scratch
	home     string
	profile  string
	exe      string
	info     fixture.Info
	canaries fixture.Canaries
	status   server.Status
	zipPath  string
	bundle   *bundle.Bundle
	files    []string
	h        *host
}

var bundleName = regexp.MustCompile(`^hachidori-diagnostics-\d{8}T\d{6}Z\.zip$`)

// runExport builds the disposable installation, starts a copy of the candidate
// on it, exports one bundle and parses it.
func runExport(t *testing.T, r Reporter, cand Candidate) *export {
	t.Helper()
	if sum, err := disposable.FileSHA256(cand.Path); err != nil || sum != cand.SHA256 {
		t.Fatalf("the candidate is not the certified one: %v %s", err, sum)
	}
	s := disposable.NewScratch(t)
	e := &export{scratch: s, home: s.Path("home"), profile: s.Path("profile"), exe: s.Path("bin", cand.File)}
	if sum, err := disposable.CopyFile(cand.Path, e.exe); err != nil || sum != cand.SHA256 {
		t.Fatalf("copy the candidate: %v %s", err, sum)
	}
	tmp := s.Path("tmp")
	for _, d := range []string{e.profile, tmp} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	var err error
	if e.info, err = fixture.Materialize(e.home); err != nil {
		t.Fatalf("materialize the disposable home: %v", err)
	}
	if e.canaries, err = fixture.Plant(e.info, e.profile); err != nil {
		t.Fatalf("plant the canaries: %v", err)
	}
	for _, p := range []string{e.home, e.profile, e.exe} {
		if !s.Contains(p) {
			t.Fatalf("%s is outside the scratch folder", p)
		}
	}

	e.h = startDashboard(t, e.exe, e.home, fixture.Environment(e.profile, tmp, e.canaries))
	st := e.h.waitFailedWorker()
	e.status = st
	r.Logf("worker state %s in phase %s after %d start(s); failure class %s", st.Worker.State, st.Worker.Phase, st.Worker.Starts, failureClass(st.Worker.LastFailure))

	e.files = e.h.exportBundle()
	var zips []string
	for _, f := range e.files {
		if strings.HasSuffix(f, ".zip") {
			zips = append(zips, f)
		}
	}
	if len(e.files) != 1 || len(zips) != 1 || !bundleName.MatchString(zips[0]) {
		t.Fatalf("state/diagnostics should hold exactly one hachidori-diagnostics-<time>.zip, holds %v", e.files)
	}
	e.zipPath = filepath.Join(e.home, "state", "diagnostics", zips[0])
	raw, err := readCapped(e.zipPath, bundle.MaxZipBytes)
	if err != nil {
		t.Fatalf("read the bundle: %v", err)
	}
	if e.bundle, err = bundle.Parse(raw); err != nil {
		t.Fatalf("the bundle is not the documented archive: %v", err)
	}
	r.Logf("bundle %s: %d bytes, entries %v", zips[0], len(raw), e.bundle.Names)
	for _, name := range []string{bundle.ManifestFile, bundle.FactsFile} {
		r.Attach("diagnostics-"+name, indent(e.bundle.Entries[name]), e.home)
	}
	r.Attach("diagnostics-worker-log-tail.txt", e.bundle.Entries[bundle.LogFile], e.home)
	r.Attach("diagnostics-host-output.txt", []byte(e.h.stderr.String()), e.home)
	return e
}

func failureClass(f *worker.FailureView) string {
	if f == nil {
		return "(none)"
	}
	return f.Class
}

// BundleSchemaIdentity proves W11.1 and W11.2 against the real candidate:
// Export bundle writes one bounded .zip under state/diagnostics holding exactly
// manifest.json, facts.json and worker-log-tail.txt, in the two versioned
// schemas; the facts carry the executable name and SHA-256 of the candidate,
// the runtime, model and device of the home, and the worker and recovery state
// of the failure the home produces.
func BundleSchemaIdentity(t *testing.T, r Reporter, cand Candidate) {
	t.Helper()
	e := runExport(t, r, cand)
	b := e.bundle
	if p := b.Problems(); len(p) != 0 {
		t.Errorf("the bundle violates the documented contract: %v", p)
	}
	if len(b.Raw) > bundle.MaxZipBytes {
		t.Errorf("the bundle is %d bytes", len(b.Raw))
	}

	// Identity: the facts name exactly the candidate.
	f := b.Facts
	if x := f.App.Executable; x.Name != cand.File || x.SHA256 != cand.SHA256 || x.Bytes != cand.Size {
		t.Errorf("facts executable %+v, candidate %s sha256=%s size=%d", x, cand.File, cand.SHA256, cand.Size)
	}
	if sum, _ := disposable.FileSHA256(e.exe); sum != f.App.Executable.SHA256 {
		t.Errorf("the facts' SHA-256 is not that of the file that ran: %s", sum)
	}
	if f.System.OS != cand.GOOS || f.System.Arch != cand.GOARCH || f.System.CPUs < 1 {
		t.Errorf("facts system %+v, candidate %s/%s", f.System, cand.GOOS, cand.GOARCH)
	}
	if cand.GoVersion != "" && f.App.GoVersion != cand.GoVersion {
		t.Errorf("facts go_version %q, candidate built with %q", f.App.GoVersion, cand.GoVersion)
	}

	// Runtime facts: the home's runtime, model and device, and the worker build
	// this executable delivers.
	rt := f.Runtime
	if rt.Runtime != e.info.Runtime || rt.ModelID != e.info.ModelID || rt.Model != e.info.ModelDir || rt.Device != "cpu" {
		t.Errorf("facts runtime %+v, home %+v", rt, e.info)
	}
	if want := setup.BuildWorker(); rt.WorkerSHA256 != want.SHA256 || rt.WorkerABI != home.WorkerABIServing || rt.WorkerABI != want.ABI {
		t.Errorf("facts worker build %s/%s, this source's worker %s/%s", rt.WorkerSHA256, rt.WorkerABI, want.SHA256, want.ABI)
	}
	if f.WebView2.Version != "" {
		t.Errorf("a dashboard-hosted export must not claim a WebView2 version, got %q", f.WebView2.Version)
	}

	// Worker and recovery facts: the bundle agrees with the status the executable
	// served, and the recovery state follows the Diagnostics rule.
	w := e.status.Worker
	fw := f.Worker
	if fw.State != worker.StateFailed || fw.State != w.State || fw.Ready || fw.Starts < 1 || fw.QueueLimit != w.QueueLimit {
		t.Errorf("facts worker %+v, served status %+v", fw, w)
	}
	wantRecovery := ""
	switch {
	case fw.State == worker.StateRestarting:
		wantRecovery = "recovering"
	case fw.State == worker.StateFailed && fw.Restarts > 0:
		wantRecovery = "gave_up"
	}
	if fw.Recovery != wantRecovery {
		t.Errorf("recovery %q for state %s with %d restarts, want %q", fw.Recovery, fw.State, fw.Restarts, wantRecovery)
	}
	if fw.LastFailure == nil || fw.LastFailure.Class != worker.ClassStartup || fw.LastFailure.Message == "" || w.LastFailure == nil || fw.LastFailure.Class != w.LastFailure.Class {
		t.Errorf("facts last failure %+v, served %+v", fw.LastFailure, w.LastFailure)
	}
	r.Logf("identity %s sha256=%s; runtime %s model %s device %s; worker %s recovery %q; %d log lines",
		f.App.Executable.Name, f.App.Executable.SHA256, rt.Runtime, rt.ModelID, rt.Device, fw.State, fw.Recovery, len(b.LogLines))
}

// BundleSecrecy proves W11.4 (and the bound of W11.1) against the real
// candidate: with canary secrets planted in the process environment, an SSH
// key and known_hosts, semantic request/question/state text, a dataset, a
// credentials file, a model binary and the worker log, none of them appears
// anywhere in the bundle, the log tail is bounded to Hachidori's own prefixed
// lines with the home path and credentials scrubbed, and the sources the export
// read still hold their canaries (so the absence is not vacuous).
func BundleSecrecy(t *testing.T, r Reporter, cand Candidate) {
	t.Helper()
	e := runExport(t, r, cand)
	b := e.bundle

	// Non-vacuous: every canary file still holds its canary after the export.
	for _, s := range e.canaries.Sources {
		data, err := os.ReadFile(s.Path)
		if err != nil || !strings.Contains(string(data), s.Needle) {
			t.Errorf("the planted source %s no longer holds its canary: %v", s.Path, err)
		}
	}
	if len(e.canaries.Env) != 4 || len(e.canaries.List) < 15 {
		t.Fatalf("too few canaries planted: %d environment, %d total", len(e.canaries.Env), len(e.canaries.List))
	}

	var forbidden []bundle.Forbidden
	for _, c := range e.canaries.Needles() {
		forbidden = append(forbidden, bundle.Forbidden{Label: c.Label, Needle: c.Needle})
	}
	forbidden = append(forbidden,
		bundle.Forbidden{Label: "the HACHIDORI_HOME path", Needle: e.home},
		bundle.Forbidden{Label: "the user profile path", Needle: e.profile},
		bundle.Forbidden{Label: "the scratch folder", Needle: e.scratch.Root},
		bundle.Forbidden{Label: "an SSH private key block", Needle: "BEGIN OPENSSH PRIVATE KEY"},
		bundle.Forbidden{Label: "an SSH public key", Needle: "ssh-ed25519 AAAA"},
		bundle.Forbidden{Label: "a credentials file entry", Needle: "machine canary.example"},
	)
	for _, kv := range e.canaries.Env {
		name, _, _ := strings.Cut(kv, "=")
		forbidden = append(forbidden, bundle.Forbidden{Label: "environment variable name " + name, Needle: name})
	}
	if found := b.Scan(forbidden); len(found) != 0 {
		for _, f := range found {
			t.Errorf("LEAK: %s", f)
		}
	}
	if p := b.Problems(); len(p) != 0 {
		t.Errorf("the bundle violates the documented contract: %v", p)
	}

	// The log tail: bounded, only Hachidori's own prefixed lines, newest kept,
	// the home path and credentials replaced.
	lines := b.LogLines
	if len(lines) == 0 || len(lines) > bundle.MaxLogLines {
		t.Fatalf("log tail of %d lines, want 1..%d", len(lines), bundle.MaxLogLines)
	}
	var seqs []int
	var sawHome, sawURL, sawTrunc bool
	seq := regexp.MustCompile(`^\[worker\] seq=(\d{4}) heartbeat$`)
	for _, l := range lines {
		if m := seq.FindStringSubmatch(l); m != nil {
			n, _ := strconv.Atoi(m[1])
			seqs = append(seqs, n)
		}
		sawHome = sawHome || strings.Contains(l, "model directory "+"<HACHIDORI_HOME>")
		sawURL = sawURL || (strings.Contains(l, "https://<redacted>@host.example") && strings.Contains(l, "token=<redacted>"))
		sawTrunc = sawTrunc || strings.HasSuffix(l, "...[truncated]")
	}
	if len(seqs) == 0 || seqs[len(seqs)-1] != e.canaries.LogSeqLast {
		t.Errorf("the tail must end with the newest planted line (seq %d), sequence %v", e.canaries.LogSeqLast, tailOf(seqs))
	}
	for i := 1; i < len(seqs); i++ {
		if seqs[i] != seqs[i-1]+1 {
			t.Errorf("the tail is not in order and gap-free around seq %d", seqs[i])
			break
		}
	}
	if len(seqs) > 0 && seqs[0] <= e.canaries.LogSeqLast-bundle.MaxLogLines {
		t.Errorf("the tail holds seq %d, older than the %d-line bound allows", seqs[0], bundle.MaxLogLines)
	}
	if !sawHome || !sawURL || !sawTrunc {
		t.Errorf("scrubbing evidence missing from the tail: home placeholder %v, URL credentials %v, truncation %v", sawHome, sawURL, sawTrunc)
	}
	if strings.Contains(string(b.Entries[bundle.FactsFile]), e.home) {
		t.Error("facts.json names the home path")
	}
	r.Logf("%d canaries planted (%d in the environment); 0 found in %d entries and the raw archive; log tail %d lines",
		len(e.canaries.List), len(e.canaries.Env), len(b.Names), len(lines))
}

func tailOf(s []int) []int {
	if len(s) > 5 {
		return s[len(s)-5:]
	}
	return s
}
