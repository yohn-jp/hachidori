package lab

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/update"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/disposable"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/releasestub"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/stubapp"
)

// Candidate is the certified executable: the one the workflow built once. A
// scenario copies it; it never runs the update against the file itself.
type Candidate struct {
	Path   string
	SHA256 string
}

// replaceRig is a rig whose disposable executable is a copy of the candidate,
// with a newer release (the E2E test binary, standing in as the new
// application) already downloaded and verified.
type replaceRig struct {
	*rig
	cand   Candidate
	newSHA string
	hold   *exec.Cmd
}

// startHold starts the stand-in for the running application the helper waits
// for.
func (rr *replaceRig) startHold() {
	rr.t.Helper()
	self, err := os.Executable()
	if err != nil {
		rr.t.Fatal(err)
	}
	cmd := exec.Command(self, stubapp.RoleHold)
	if err := cmd.Start(); err != nil {
		rr.t.Fatalf("start the stand-in application: %v", err)
	}
	rr.hold = cmd
	rr.t.Cleanup(func() { rr.quit() })
}

// quit ends the stand-in application, as Quit ends the real one.
func (rr *replaceRig) quit() {
	if rr.hold == nil {
		return
	}
	_ = rr.hold.Process.Kill()
	_, _ = rr.hold.Process.Wait()
	rr.hold = nil
}

// helperStarter starts the real helper detached with the production starter.
// The only change to Service.Install's arguments is --pid: Install names the
// test process, which never exits, so the helper is pointed at the stand-in
// application the scenario ends.
func (rr *replaceRig) helperStarter(exe string, args []string) error {
	if rr.hold == nil {
		rr.t.Fatal("the helper was started with no stand-in application running")
	}
	args, err := withFlag(args, "--pid", itoa(rr.hold.Process.Pid))
	if err != nil {
		return err
	}
	return update.StartDetached(exe, args)
}

func newReplaceRig(t *testing.T, r Reporter, cand Candidate) *replaceRig {
	t.Helper()
	if sum, err := disposable.FileSHA256(cand.Path); err != nil || sum != cand.SHA256 {
		t.Fatalf("the candidate is not the certified one: %v %s", err, sum)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	newExe, err := os.ReadFile(self)
	if err != nil || len(newExe) < 2 || string(newExe[:2]) != "MZ" {
		t.Fatalf("the stand-in new release (this test binary) is not a Windows executable: %v", err)
	}
	rr := &replaceRig{cand: cand, newSHA: shaOf(newExe)}
	rr.rig = newRig(t, r, rigOptions{exeFrom: cand.Path, helperStarter: rr.helperStarter})
	if rr.exeSHA != cand.SHA256 {
		t.Fatalf("the disposable copy %s differs from the candidate %s", rr.exeSHA, cand.SHA256)
	}
	// Nothing the helper restarts may reach a real profile or home.
	t.Setenv("LOCALAPPDATA", rr.profile)
	t.Setenv("APPDATA", filepath.Join(rr.profile, "Roaming"))
	t.Setenv("USERPROFILE", rr.profile)
	t.Setenv("HOME", rr.profile)
	t.Setenv("HACHIDORI_HOME", rr.home)
	// Processes started from the disposable executable or the helper copy must
	// be gone before the scratch folder is removed.
	t.Cleanup(func() {
		_ = killImage(rr.exe)
		for _, f := range regularFiles(filepath.Join(rr.updatesDir(), "helper")) {
			_ = killImage(filepath.Join(rr.updatesDir(), "helper", filepath.FromSlash(f)))
		}
	})

	rr.seedHome()
	rr.seedInstalled(tagInst)
	rr.stub.Publish(tagNewer, newExe, func(r *releasestub.Release) { r.MetadataDigest = true })
	rr.setChannel(update.Development)
	rr.check()
	if op := rr.download(tagNewer); op.Failure != nil {
		t.Fatalf("download of the new release: %+v", op.Failure)
	}
	st := rr.svc.Status()
	if st.Ready == nil || st.ReadyProblem != "" || st.Ready.SHA256 != rr.newSHA || st.Ready.Target != rr.exe {
		t.Fatalf("the new release is not ready to install: %+v (%s)", st.Ready, st.ReadyProblem)
	}
	page := rr.get(updatesPage)
	requireContains(t, "the Updates page with a ready update", page, `id="updates-install"`, "matches the release checksum", rr.exe)
	if buttonDisabled(t, page, "Restart & update") {
		t.Fatal("Restart & update must be available for a verified update")
	}
	return rr
}

// restartAndUpdate is the operator's Restart & update while the application
// runs: the helper is started from the disposable copy, then the application
// quits so it may replace the executable. It returns the helper's report.
func (rr *replaceRig) restartAndUpdate() update.Result {
	rr.t.Helper()
	rr.startHold()
	since := time.Now()
	nStarts := len(rr.started)
	rr.post(updatesPage+"/install", nil)
	if len(rr.started) != nStarts+1 {
		rr.t.Fatalf("Restart & update started %d helpers", len(rr.started)-nStarts)
	}
	start := rr.started[len(rr.started)-1]
	helper, args := start[0], start[1:]
	if !disposable.Within(filepath.Join(rr.updatesDir(), "helper"), helper) {
		rr.t.Errorf("the helper %s is not in the update area", helper)
	}
	if sum, err := disposable.FileSHA256(helper); err != nil || sum != rr.cand.SHA256 {
		rr.t.Errorf("the helper is not a copy of the candidate: %v %s", err, sum)
	}
	if len(args) < 7 || args[0] != "apply-update" || !contains2(args, "--target", rr.exe) || !contains2(args, "--home", rr.home) || !contains2(args, "--restart-home", rr.home) {
		rr.t.Errorf("helper arguments: %v", args)
	}
	// The application is still running: nothing may have been replaced.
	if sum, _ := disposable.FileSHA256(rr.exe); sum != rr.cand.SHA256 {
		rr.t.Fatal("the executable changed while the application was still running")
	}
	rr.quit()
	res := rr.awaitResult(since)
	if b, err := os.ReadFile(filepath.Join(rr.updatesDir(), "result.json")); err == nil {
		rr.r.Attach("update-helper-result.json", b, rr.home)
	}
	return res
}

func contains2(args []string, name, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name && strings.EqualFold(args[i+1], value) {
			return true
		}
	}
	return false
}

// awaitResult waits for the helper's report written after since.
func (rr *replaceRig) awaitResult(since time.Time) update.Result {
	rr.t.Helper()
	var res update.Result
	until(rr.t, "the replacement helper's result", longWait, func() bool {
		r, err := update.LoadResult(rr.home)
		if err != nil || !r.Time.After(since) {
			return false
		}
		res = r
		return true
	})
	return res
}

// awaitReopened waits for the executable at the replaced path to have been
// started again and returns what it recorded.
func (rr *replaceRig) awaitReopened() stubapp.Marker {
	rr.t.Helper()
	var m stubapp.Marker
	until(rr.t, "the replaced executable to be reopened", shortWait, func() bool {
		got, err := stubapp.ReadMarker(rr.exe)
		if err != nil {
			return false
		}
		m = got
		return true
	})
	return m
}

func (rr *replaceRig) requireNoStaging() {
	rr.t.Helper()
	if disposable.Exists(rr.exe + ".new") {
		rr.t.Errorf("%s.new was left behind", filepath.Base(rr.exe))
	}
	for _, f := range rr.stagedFiles() {
		rr.t.Errorf("staged file left in the update area: %s", f)
	}
}

// requireUnchanged fails when the promised persisted state differs from before.
func (rr *replaceRig) requireUnchanged(before persisted) {
	rr.t.Helper()
	if d := before.diff(rr.snapshot(before.exact)); len(d) != 0 {
		rr.t.Errorf("promised persisted state changed: %v", d)
	}
}

// requireCandidateIntact proves the scenario never touched the certified file.
func (rr *replaceRig) requireCandidateIntact() {
	rr.t.Helper()
	if sum, err := disposable.FileSHA256(rr.cand.Path); err != nil || sum != rr.cand.SHA256 {
		rr.t.Fatalf("the candidate itself changed: %v %s", err, sum)
	}
}

// ReplaceAndRestart proves W18.4 against the real candidate: Restart & update
// hands the verified update to the candidate's own helper, which replaces the
// executable of a disposable copy after the application quit, keeps the
// previous executable as <exe>.old, reopens the same path (the new bytes run
// with the same home), reports UPDATED, cleans its staging, and leaves the
// home, models, settings, history and evidence exactly as they were.
func ReplaceAndRestart(t *testing.T, r Reporter, cand Candidate) {
	t.Helper()
	rr := newReplaceRig(t, r, cand)
	before := rr.snapshot(true) // nothing else runs: the settings file must be byte-identical

	res := rr.restartAndUpdate()
	if res.Outcome != update.OutcomeApplied || res.Tag != tagNewer || res.SHA256 != rr.newSHA || !strings.EqualFold(res.Target, rr.exe) {
		t.Fatalf("the helper reported %+v", res)
	}
	if sum, err := disposable.FileSHA256(rr.exe); err != nil || sum != rr.newSHA {
		t.Fatalf("the executable at the replaced path is not the update: %v %s", err, sum)
	}
	if sum, err := disposable.FileSHA256(rr.exe + ".old"); err != nil || sum != rr.cand.SHA256 {
		t.Errorf("the previous executable was not kept as .old: %v %s", err, sum)
	}
	rr.requireNoStaging()
	if rr.readyRecord() || disposable.Exists(filepath.Join(rr.updatesDir(), tagNewer)) {
		t.Error("the ready record or the staged release was left after a successful update")
	}

	m := rr.awaitReopened()
	if !strings.EqualFold(filepath.Clean(m.Exe), filepath.Clean(rr.exe)) || m.SHA256 != rr.newSHA || !strings.EqualFold(filepath.Clean(m.Home), filepath.Clean(rr.home)) {
		t.Errorf("the reopened executable: %+v", m)
	}
	rr.requireUnchanged(before)

	// A newly started application reports UPDATED and knows its release.
	rr.svc = rr.newService()
	rr.serveConsole(rr.svc)
	st := rr.svc.Status()
	if st.Result == nil || st.Result.Outcome != update.OutcomeApplied || !st.Installed.Known || st.Installed.Version != tagNewer || st.Ready != nil {
		t.Errorf("status after the update: result %+v installed %+v ready %+v", st.Result, st.Installed, st.Ready)
	}
	page := rr.get(updatesPage)
	requireContains(t, "the Updates page after the update", page, "<strong>UPDATED</strong>", `id="updates-installed-version">`+tagNewer, `id="updates-not-ready"`)
	requireAbsent(t, "the Updates page after the update", page, "NOT UPDATED", "RESTART FAILED")
	rr.requireCandidateIntact()
	r.Logf("replaced the disposable copy %s -> %s, previous kept as .old, reopened with the update, state unchanged", shortSHA(rr.cand.SHA256), shortSHA(rr.newSHA))
}

// FileLockKeepsPrevious proves W18.5 against the real candidate: while another
// handle holds the executable, the helper cannot move it aside, so the update
// ends NOT UPDATED within its bounded retries, the previous executable is
// intact and still starts, the verified update stays ready, nothing is left
// half-staged and the persisted state is unchanged; releasing the lock and
// trying again then succeeds.
func FileLockKeepsPrevious(t *testing.T, r Reporter, cand Candidate) {
	t.Helper()
	rr := newReplaceRig(t, r, cand)
	// The previous executable is restarted by the helper and may rewrite the
	// rest of the settings file, so only the update-owned settings are compared.
	before := rr.snapshot(false)

	lock, err := holdOpen(rr.exe)
	if err != nil {
		t.Fatalf("hold the executable open: %v", err)
	}
	released := false
	release := func() {
		if !released {
			released = true
			_ = lock.Close()
		}
	}
	t.Cleanup(release)

	start := time.Now()
	res := rr.restartAndUpdate()
	took := time.Since(start)
	if res.Outcome != update.OutcomeFailed || !strings.Contains(res.Message, "could not be moved aside") || !strings.Contains(res.Message, "kept") {
		t.Fatalf("the helper reported %+v, want a failed replacement that kept the current executable", res)
	}
	if strings.Contains(res.Message, "restarting") {
		t.Errorf("the previous executable could not be restarted: %s", res.Message)
	}
	if sum, err := disposable.FileSHA256(rr.exe); err != nil || sum != rr.cand.SHA256 {
		t.Fatalf("the previous executable was damaged: %v %s", err, sum)
	}
	for _, leftover := range []string{rr.exe + ".new", rr.exe + ".old"} {
		if disposable.Exists(leftover) {
			t.Errorf("%s was left behind by a failed replacement", filepath.Base(leftover))
		}
	}
	// The verified update stays ready: the failure left a consistent state from
	// which the operator can simply try again.
	if st := rr.svc.Status(); st.Ready == nil || st.ReadyProblem != "" || st.Result == nil || st.Result.Outcome != update.OutcomeFailed {
		t.Errorf("status after the failed replacement: ready %+v (%s) result %+v", st.Ready, st.ReadyProblem, st.Result)
	}
	if !rr.readyRecord() || len(rr.stagedFiles()) != 1 {
		t.Errorf("the verified update must stay staged: ready record %v, staged %v", rr.readyRecord(), rr.stagedFiles())
	}
	rr.requireUnchanged(before)
	page := rr.get(updatesPage)
	requireContains(t, "the Updates page after a failed replacement", page, "<strong>NOT UPDATED</strong>", update.Hints[update.ClassReplace][:40])
	requireAbsent(t, "the Updates page after a failed replacement", page, "<strong>UPDATED</strong>")

	// The previous executable still starts: it is the candidate, and it runs.
	ctx, cancel := context.WithTimeout(context.Background(), shortWait)
	defer cancel()
	out, err := exec.CommandContext(ctx, rr.exe, "e2e-previous-executable-probe").CombinedOutput()
	if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 2 || !strings.Contains(string(out), "usage: hachidori") {
		t.Errorf("the previous executable did not start and print its usage: %v %.200s", err, out)
	}
	r.Logf("held lock: NOT UPDATED after %s; previous executable intact, starts, update still ready", took.Round(time.Second))

	// Release the lock, end whatever the helper restarted, and try again.
	release()
	_ = killImage(rr.exe)
	until(t, "the restarted previous executable to end", shortWait, func() bool {
		n, err := imageProcesses(rr.exe)
		return err == nil && n == 0
	})
	res = rr.restartAndUpdate()
	if res.Outcome != update.OutcomeApplied || res.SHA256 != rr.newSHA {
		t.Fatalf("the retry after releasing the lock reported %+v", res)
	}
	if sum, _ := disposable.FileSHA256(rr.exe); sum != rr.newSHA {
		t.Error("the retry did not put the update in place")
	}
	if sum, err := disposable.FileSHA256(rr.exe + ".old"); err != nil || sum != rr.cand.SHA256 {
		t.Errorf("the previous executable was not kept as .old after the retry: %v %s", err, sum)
	}
	rr.requireNoStaging()
	if m := rr.awaitReopened(); m.SHA256 != rr.newSHA {
		t.Errorf("the reopened executable: %+v", m)
	}
	rr.requireUnchanged(before)
	rr.requireCandidateIntact()
	r.Logf("after releasing the lock the retry replaced the executable and reopened it")
}

func shortSHA(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}
