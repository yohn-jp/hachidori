package lab

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/update"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/disposable"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/releasestub"
)

const (
	apiList  = "GET " + releasestub.APIHost + "/repos/yohn-jp/hachidori/releases?per_page=100&page=1"
	tagInst  = "0.2.1-dev"
	tagNewer = "0.2.2-dev"
)

func (rg *rig) stagedPath(tag string) string {
	v, err := update.ParseVersion(tag)
	if err != nil {
		rg.t.Fatal(err)
	}
	return update.StagedPath(rg.home, v)
}

func tagURL(host, tag, asset string) string {
	switch host {
	case releasestub.DownloadHost:
		return "GET " + host + "/yohn-jp/hachidori/releases/download/" + tag + "/" + asset
	}
	return "GET " + host + "/blob/" + tag + "/" + asset
}

// CheckDownloadProgress proves W18.2 and W29.3: Check for updates lists the
// releases, Download reports its phases and bytes through the same operation a
// reload shows, refuses every conflicting action meanwhile, and reaches a
// verified, ready update whose digest equals the release checksum. Every
// request goes to the local stub, in the documented order.
func CheckDownloadProgress(t *testing.T, r Reporter) {
	t.Helper()
	const size = 4 << 20
	const heldAt = 1 << 20
	rg := newRig(t, r, rigOptions{})
	rg.seedInstalled(tagInst)
	newExe := Exe(tagNewer, size)
	rg.stub.Publish("0.1.9-dev", Exe("0.1.9-dev", 8<<10))
	rg.stub.Publish(tagInst, Exe(tagInst, 8<<10))
	rg.stub.Publish(tagNewer, newExe, func(r *releasestub.Release) { r.MetadataDigest = true })
	rg.setChannel(update.Development)

	// Check for updates.
	rg.check()
	if got := rg.stub.Strings(); !reflect.DeepEqual(got, []string{apiList}) {
		t.Fatalf("a check must send exactly one release-list request, got %v", got)
	}
	st := rg.svc.Status()
	var tags, rels []string
	for _, c := range st.Check.Releases {
		tags, rels = append(tags, c.Tag), append(rels, c.Relation)
	}
	if !reflect.DeepEqual(tags, []string{tagNewer, tagInst, "0.1.9-dev"}) || !reflect.DeepEqual(rels, []string{update.RelNewer, update.RelCurrent, update.RelOlder}) {
		t.Fatalf("the check listed %v %v", tags, rels)
	}
	page := rg.get(updatesPage)
	requireContains(t, "the Updates page after a check", page, `id="updates-releases"`, tagNewer, tagInst, "0.1.9-dev")
	if n := strings.Count(page, `action="/settings/updates/download"`); n != 1 {
		t.Errorf("Download must be offered for the one newer release only, got %d forms", n)
	}
	r.Logf("check listed %v", tags)

	// Download, held part-way so its progress is observable.
	gate := rg.stub.HoldExe(tagNewer, heldAt)
	before := rg.stub.Count()
	rg.startDownload(tagNewer)
	receive(t, "the held download", gate.Reached())
	until(t, "the held download's bytes to be reported", shortWait, func() bool {
		b := rg.svc.Status().Busy
		return b != nil && b.Phase == update.PhaseDownload && b.Done == heldAt
	})
	busy := rg.svc.Status().Busy
	if busy.Kind != update.KindDownload || busy.Tag != tagNewer || busy.Total != size ||
		!reflect.DeepEqual(busy.Phases, []string{update.PhaseChecksum, update.PhaseDownload}) ||
		!reflect.DeepEqual(busy.Plan, []string{update.PhaseChecksum, update.PhaseDownload, update.PhaseVerify}) {
		t.Fatalf("operation in flight: %+v", busy)
	}
	part := filepath.Join(rg.updatesDir(), tagNewer, update.ExeAsset+".part")
	if fi, err := os.Stat(part); err != nil || fi.Size() != heldAt {
		t.Fatalf("the partial download should hold the %d bytes received: %v %v", heldAt, fi, err)
	}
	live := rg.get(updatesPage)
	requireContains(t, "the Updates page while downloading", live, `id="models-busy"`, `data-kind="update"`, `aria-valuenow="25"`, "RUNNING", `id="updates-busy-note"`)
	for _, label := range []string{"Check for updates", "Download", "Save channel"} {
		if !buttonDisabled(t, live, label) {
			t.Errorf("%q must be disabled while a download runs", label)
		}
	}
	// A reload shows the same operation, not a copy: the page is projected from
	// the service's one operation, so a second reader sees the same one.
	again := rg.get(updatesPage)
	requireContains(t, "a reloaded Updates page", again, `id="models-busy"`, `data-kind="update"`, `aria-valuenow="25"`)
	if other := rg.svc.Status().Busy; other == nil || !other.Started.Equal(busy.Started) || other.Tag != busy.Tag {
		t.Errorf("a reload observed another operation: %+v", other)
	}
	// Every conflicting action is refused and sends nothing.
	sent := rg.stub.Count()
	if err := rg.svc.StartCheck(); !isRefused(err, "another update action is in progress") {
		t.Errorf("Check while downloading: %v", err)
	}
	if err := rg.svc.StartDownload(tagNewer); !isRefused(err, "another update action is in progress") {
		t.Errorf("Download while downloading: %v", err)
	}
	if err := rg.svc.SetChannel(update.Stable); !isRefused(err, "another update action is in progress") {
		t.Errorf("Save channel while downloading: %v", err)
	}
	if err := rg.svc.Install(); !isRefused(err, "") {
		t.Errorf("Restart & update while downloading: %v", err)
	}
	if rg.stub.Count() != sent {
		t.Errorf("refused actions sent requests: %d -> %d", sent, rg.stub.Count())
	}
	if sent-before != 4 { // checksum (redirect + blob) and executable (redirect + blob)
		t.Errorf("the download sent %d requests so far, want 4", sent-before)
	}

	gate.Release()
	rg.waitIdle("the released download to finish")
	last := rg.svc.Status().Last
	if last == nil || last.Failure != nil || last.Tag != tagNewer {
		t.Fatalf("download outcome: %+v", last)
	}
	if want := []string{update.PhaseChecksum, update.PhaseDownload, update.PhaseVerify}; !reflect.DeepEqual(last.Phases, want) {
		t.Errorf("phases %v, want %v", last.Phases, want)
	}
	if last.Phase != update.PhaseVerify || last.Done != size || last.Total != size {
		t.Errorf("the verify phase should report the whole file: %+v", last)
	}
	st = rg.svc.Status()
	if st.Ready == nil || st.ReadyProblem != "" || st.Ready.Tag != tagNewer || st.Ready.SHA256 != shaOf(newExe) || st.Ready.Size != size || st.Ready.Target != rg.exe {
		t.Fatalf("ready update: %+v (%s)", st.Ready, st.ReadyProblem)
	}
	staged := rg.stagedPath(st.Ready.Tag)
	if sum, err := disposable.FileSHA256(staged); err != nil || sum != shaOf(newExe) {
		t.Errorf("the staged file differs from the release: %v %s", err, sum)
	}
	if disposable.Exists(part) || !rg.readyRecord() {
		t.Errorf("staging after success: partial exists=%v ready record=%v", disposable.Exists(part), rg.readyRecord())
	}
	done := rg.get(updatesPage)
	requireContains(t, "the Updates page after the download", done, "matches the release checksum", `id="updates-op-next"`, `id="updates-install"`)
	want := []string{
		apiList,
		tagURL(releasestub.DownloadHost, tagNewer, update.SumAsset), tagURL(releasestub.AssetHost, tagNewer, update.SumAsset),
		tagURL(releasestub.DownloadHost, tagNewer, update.ExeAsset), tagURL(releasestub.AssetHost, tagNewer, update.ExeAsset),
	}
	if got := rg.stub.Strings(); !reflect.DeepEqual(got, want) {
		t.Errorf("requests\n got %v\nwant %v", got, want)
	}
	if len(rg.started) != 0 {
		t.Errorf("downloading started the replacement helper: %v", rg.started)
	}
	r.Logf("download of %s (%d bytes) reported phases %v and reached ready; %d requests, all to the local stub", tagNewer, size, last.Phases, rg.stub.Count())
	r.Attach("update-download-requests.txt", []byte(strings.Join(rg.stub.Strings(), "\n")), rg.home)
}

// NoNetworkOnOpen proves W18.1: rendering any page, opening Settings and
// Updates, and switching the channel back and forth sends no request, and the
// choice is persisted. An explicit check afterwards does send one, so the
// request counter is known to work.
func NoNetworkOnOpen(t *testing.T, r Reporter) {
	t.Helper()
	rg := newRig(t, r, rigOptions{})
	rg.stub.Publish(tagInst, Exe(tagInst, 8<<10))
	for _, p := range []string{"/", "/settings", updatesPage, "/diagnostics", "/experiments", "/errors", updatesPage} {
		rg.get(p)
	}
	for _, c := range []update.Channel{update.Development, update.Stable, update.Development, update.Stable, update.Development} {
		rg.setChannel(c)
		rg.get(updatesPage)
	}
	// A newly started application also reads its state without asking anyone.
	reopened := rg.newService()
	st := reopened.Status()
	if st.Channel != update.Development || st.Check != nil || st.LastCheck != nil || st.Busy != nil {
		t.Errorf("reopened status: channel %s check %v last check %v busy %v", st.Channel, st.Check, st.LastCheck, st.Busy)
	}
	if n := rg.stub.Count(); n != 0 {
		t.Fatalf("opening pages and switching the channel sent %d requests: %v", n, rg.stub.Strings())
	}
	if saved, err := rg.store.UpdateSettings(); err != nil || saved.Channel != update.Development {
		t.Errorf("the channel was not persisted: %+v %v", saved, err)
	}
	r.Logf("12 page views and 5 channel switches sent 0 requests; the saved channel is development")

	rg.check()
	if n := rg.stub.Count(); n != 1 {
		t.Fatalf("an explicit check should send exactly one request, sent %d", n)
	}
}

// SingleCheckInFlight proves W29.2: while a check is running, a second Check
// for updates (and every other update action) is refused and sends no second
// request; once the first finishes the burst was exactly one request.
func SingleCheckInFlight(t *testing.T, r Reporter) {
	t.Helper()
	rg := newRig(t, r, rigOptions{})
	rg.stub.Publish(tagInst, Exe(tagInst, 8<<10))
	rg.stub.Publish(tagNewer, Exe(tagNewer, 8<<10))
	rg.setChannel(update.Development)

	gate := rg.stub.HoldList()
	rg.post(updatesPage+"/check", nil)
	receive(t, "the held release-list request", gate.Reached())
	if b := rg.svc.Status().Busy; b == nil || b.Kind != update.KindCheck || b.Phase != update.PhaseReleases {
		t.Fatalf("the check in flight: %+v", b)
	}
	if rg.stub.Count() != 1 {
		t.Fatalf("one check should have sent one request, sent %v", rg.stub.Strings())
	}
	for i := 0; i < 3; i++ {
		rg.post(updatesPage+"/check", nil) // a repeated click
	}
	page := rg.get(updatesPage)
	requireContains(t, "the Updates page during a check", page, "another update action is in progress", `id="updates-busy-note"`)
	if !buttonDisabled(t, page, "Check for updates") || !buttonDisabled(t, page, "Save channel") {
		t.Error("Check and Save channel must be disabled while a check runs")
	}
	if err := rg.svc.StartCheck(); !isRefused(err, "another update action is in progress") {
		t.Errorf("a second check: %v", err)
	}
	if err := rg.svc.SetChannel(update.Stable); !isRefused(err, "another update action is in progress") {
		t.Errorf("a channel change during a check: %v", err)
	}
	if n := rg.stub.Count(); n != 1 {
		t.Fatalf("repeated checks started a second request burst: %v", rg.stub.Strings())
	}

	gate.Release()
	rg.waitIdle("the first check to finish")
	if n := rg.stub.Count(); n != 1 {
		t.Fatalf("the check should have sent one request in total, sent %v", rg.stub.Strings())
	}
	st := rg.svc.Status()
	if st.Check == nil || len(st.Check.Releases) != 2 || st.LastCheck == nil || !st.LastCheck.OK {
		t.Fatalf("the single check did not complete: %+v %+v", st.Check, st.LastCheck)
	}
	// The refusal was about the running check, not a permanent state.
	rg.check()
	if n := rg.stub.Count(); n != 2 {
		t.Fatalf("a check after the first finished should send one more request, total %d", n)
	}
	r.Logf("1 held check + 3 repeated clicks + 2 refused calls produced 1 request; the next check produced 1 more")
}

// CorruptionRejected proves W18.3: a download whose bytes do not match the
// release checksum, a checksum file that does not match, and unusable checksum
// files are each rejected with the right class and phase, leave no partial or
// staged file and no ready record, and Restart & update stays unavailable.
func CorruptionRejected(t *testing.T, r Reporter) {
	t.Helper()
	good := Exe("good", 256<<10)
	flipped := bytes.Clone(good)
	flipped[len(flipped)/2] ^= 0xff
	cases := []struct {
		name    string
		publish func(*releasestub.Release)
		class   update.Class
		phase   string
		sent    bool   // whether the executable asset must have been requested
		outcome string // what the page says was left
	}{
		{"corrupted bytes", func(r *releasestub.Release) { r.Served = flipped }, update.ClassChecksumMismatch, update.PhaseVerify, true,
			"The downloaded file failed verification and was discarded."},
		{"checksum file for other bytes", func(r *releasestub.Release) { r.Sum = releasestub.ChecksumFile(flipped) }, update.ClassChecksumMismatch, update.PhaseVerify, true,
			"The downloaded file failed verification and was discarded."},
		{"checksum file with extra lines", func(r *releasestub.Release) { r.Sum = append(releasestub.ChecksumFile(good), "stray line\n"...) }, update.ClassVerification, update.PhaseChecksum, false,
			"Nothing was downloaded."},
		{"checksum file for another asset", func(r *releasestub.Release) {
			r.Sum = bytes.Replace(releasestub.ChecksumFile(good), []byte(update.ExeAsset), []byte("other.exe"), 1)
		}, update.ClassVerification, update.PhaseChecksum, false, "Nothing was downloaded."},
	}
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			rg := newRig(t, r, rigOptions{})
			rg.seedInstalled(tagInst)
			rg.stub.Publish(tagNewer, good, c.publish)
			rg.setChannel(update.Development)
			rg.check()

			op := rg.download(tagNewer)
			f := op.Failure
			if f == nil || f.Class != c.class || f.Phase != c.phase {
				t.Fatalf("failure %+v, want class %s in phase %s", f, c.class, c.phase)
			}
			if f.Message == "" {
				t.Error("a rejected download must say why")
			}
			exeRequested := false
			for _, q := range rg.stub.Requests() {
				exeRequested = exeRequested || (q.Host == releasestub.AssetHost && strings.HasSuffix(q.Path, "/"+update.ExeAsset))
			}
			if exeRequested != c.sent {
				t.Errorf("executable requested = %v, want %v (%v)", exeRequested, c.sent, rg.stub.Strings())
			}
			if left := rg.stagedFiles(); len(left) != 0 || rg.readyRecord() {
				t.Errorf("a rejected download left files %v (ready record %v)", left, rg.readyRecord())
			}
			st := rg.svc.Status()
			if st.Ready != nil {
				t.Errorf("a rejected download is ready: %+v", st.Ready)
			}
			page := rg.get(updatesPage)
			requireContains(t, "the Updates page after a rejection", page, c.outcome, "No update is ready to install.", `id="updates-not-ready"`, "FAILED")
			if installForm(page) {
				t.Error("Restart & update is offered for a rejected download")
			}
			// Restart & update stays unavailable, by the service and by the page.
			rg.post(updatesPage+"/install", nil)
			if err := rg.svc.Install(); !isRefused(err, "no update can be installed") {
				t.Errorf("Install after a rejection: %v", err)
			}
			requireContains(t, "the Updates page after a refused install", rg.get(updatesPage), "no update can be installed")
			if len(rg.started) != 0 {
				t.Errorf("the helper was started for a rejected update: %v", rg.started)
			}
			if sum, _ := disposable.FileSHA256(rg.exe); sum != rg.exeSHA {
				t.Error("the installed executable changed")
			}
			r.Logf("%s: %s in phase %s; nothing staged, no ready record, install refused", c.name, f.Class, f.Phase)
		})
	}
}

// NetworkFailureBounded proves W29.4 and the interruption contract: a network
// failure mid-download ends the operation with its phase and cause, discards
// the partial file and keeps an earlier verified update ready; a truncated
// transfer and a failing release list are bounded the same way; every failure
// is recoverable by simply retrying once the network is back.
func NetworkFailureBounded(t *testing.T, r Reporter) {
	t.Helper()
	const heldAt = 1 << 20
	rg := newRig(t, r, rigOptions{})
	rg.seedInstalled(tagInst)
	first, second := Exe("0.2.2-dev", 2<<20), Exe("0.2.3-dev", 3<<20)
	rg.stub.Publish(tagNewer, first)
	rg.setChannel(update.Development)
	rg.check()
	if op := rg.download(tagNewer); op.Failure != nil {
		t.Fatalf("the first download failed: %+v", op.Failure)
	}
	firstSum := shaOf(first)

	// The network drops while the next release is half downloaded.
	rg.stub.Publish("0.2.3-dev", second)
	rg.check()
	gate := rg.stub.HoldExe("0.2.3-dev", heldAt)
	rg.startDownload("0.2.3-dev")
	receive(t, "the held download", gate.Reached())
	part := filepath.Join(rg.updatesDir(), "0.2.3-dev", update.ExeAsset+".part")
	until(t, "the partial download to hold the received bytes", shortWait, func() bool {
		fi, err := os.Stat(part)
		return err == nil && fi.Size() == heldAt
	})
	gate.Abort()
	rg.waitIdle("the interrupted download to end")
	f := rg.svc.Status().Last.Failure
	if f == nil || f.Class != update.ClassDownload || f.Phase != update.PhaseDownload || f.Message == "" {
		t.Fatalf("interrupted download failure: %+v", f)
	}
	if disposable.Exists(part) {
		t.Error("the partial download was not discarded")
	}
	if left := rg.stagedFiles(); len(left) != 1 || left[0] != "0.2.2-dev/"+update.ExeAsset {
		t.Errorf("staged files after the interruption: %v", left)
	}
	st := rg.svc.Status()
	if st.Ready == nil || st.Ready.Tag != tagNewer || st.Ready.SHA256 != firstSum || st.ReadyProblem != "" {
		t.Errorf("the earlier verified update was not kept ready: %+v (%s)", st.Ready, st.ReadyProblem)
	}
	if sum, err := disposable.FileSHA256(rg.stagedPath(st.Ready.Tag)); err != nil || sum != firstSum {
		t.Errorf("the earlier staged file changed: %v %s", err, sum)
	}
	page := rg.get(updatesPage)
	requireContains(t, "the Updates page after an interruption", page, "FAILED", "Failed in phase", "The partial download was discarded.",
		"A previously verified update is still ready to install.", `id="updates-op-hint"`, update.Hints[update.ClassDownload][:40])
	r.Logf("interrupted download: class %s in phase %s, partial discarded, earlier update %s still ready", f.Class, f.Phase, st.Ready.Tag)

	// A transfer cut short by the server ends the same way.
	rg.stub.CutExe("0.2.3-dev", 300<<10)
	if op := rg.download("0.2.3-dev"); op.Failure == nil || op.Failure.Class != update.ClassDownload || op.Failure.Phase != update.PhaseDownload {
		t.Fatalf("truncated transfer: %+v", op.Failure)
	}
	if disposable.Exists(part) {
		t.Error("the truncated partial download was not discarded")
	}

	// Retrying after the network is back completes, and staging is cleaned.
	rg.stub.Heal()
	if op := rg.download("0.2.3-dev"); op.Failure != nil {
		t.Fatalf("the retry failed: %+v", op.Failure)
	}
	st = rg.svc.Status()
	if st.Ready == nil || st.Ready.Tag != "0.2.3-dev" || st.Ready.SHA256 != shaOf(second) {
		t.Fatalf("after the retry: %+v", st.Ready)
	}
	if left := rg.stagedFiles(); len(left) != 1 || left[0] != "0.2.3-dev/"+update.ExeAsset {
		t.Errorf("staging after the retry (the superseded version and partial files must be gone): %v", left)
	}

	// The release list failing is bounded and reported, and changes nothing.
	exeBefore, _ := disposable.FileSHA256(rg.exe)
	for _, inject := range []struct {
		name string
		do   func()
		want string
	}{
		{"connection dropped", rg.stub.DropList, ""},
		{"service unavailable", func() { rg.stub.FailList(503) }, "503"},
	} {
		rg.stub.Heal()
		inject.do()
		rg.post(updatesPage+"/check", nil)
		rg.waitIdle("the failing check to end")
		st := rg.svc.Status()
		if st.LastCheck == nil || st.LastCheck.OK || st.LastCheck.Class != update.ClassNetwork || !strings.Contains(st.LastCheck.Message, inject.want) || st.Check != nil {
			t.Fatalf("%s: last check %+v, results %v", inject.name, st.LastCheck, st.Check)
		}
		page := rg.get(updatesPage)
		requireContains(t, "the Updates page after a failed check ("+inject.name+")", page, `id="updates-check-failure"`, "Nothing was downloaded and the installed executable was not changed.",
			update.Hints[update.ClassNetwork][:40])
		r.Logf("release list %s: network class reported, no results shown", inject.name)
	}
	if sum, _ := disposable.FileSHA256(rg.exe); sum != exeBefore {
		t.Error("the installed executable changed")
	}
	rg.stub.Heal()
	rg.check()
	if rg.svc.Status().Check == nil {
		t.Error("a check after the network returned did not list releases")
	}
	r.Attach("update-network-requests.txt", []byte(strings.Join(rg.stub.Strings(), "\n")), rg.home)
}

// seedHome writes the persisted state an update promises to leave unchanged:
// a model, the activation record, history and evidence in the home, and
// settings in the profile.
func (rg *rig) seedHome() {
	rg.t.Helper()
	files := map[string]string{
		"models/example--fixture/rev1/model.safetensors":    "model-binary-sentinel",
		"models/example--fixture/rev1/hachidori-model.json": `{"id":"fixture"}`,
		"state/active-runtime.json":                         `{"runtime":"fixture","model":"example--fixture/rev1","device":"cpu"}`,
		"state/history/experiment-1.json":                   `{"schema":"fixture","name":"history sentinel"}`,
		"state/forge/runs/run-1.json":                       `{"schema":"fixture","evidence":"evidence sentinel"}`,
		"state/certifications/variant-1/certification.json": `{"schema":"fixture"}`,
		"variants/fixture/v1/hachidori-variant.json":        `{"schema":"fixture"}`,
		"logs/worker.log":                                   "[worker] fixture\n",
	}
	for rel, data := range files {
		p := filepath.Join(rg.home, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			rg.t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
			rg.t.Fatal(err)
		}
	}
	if err := rg.store.SetMemoryBudget(8 << 30); err != nil {
		rg.t.Fatal(err)
	}
	if err := rg.store.SetLocale("en"); err != nil {
		rg.t.Fatal(err)
	}
}

// promised is the part of the home an update must never change: models,
// variants, the activation record, history, evidence and certifications.
// Caches, logs, runtimes materialized later and the update area are expected to
// change.
func promised(rel string, _ bool) bool {
	for _, p := range []string{"models", "variants", "state/active-runtime.json", "state/history", "state/forge", "state/certifications"} {
		if rel == p || strings.HasPrefix(rel, p+"/") || strings.HasPrefix(p, rel+"/") {
			return true
		}
	}
	return false
}

// settingsFile keeps only the settings file of the profile folder; the rest of
// the folder belongs to whatever application the helper restarts.
func settingsFile(rel string, _ bool) bool {
	return rel == "Hachidori" || rel == "Hachidori/settings.json"
}

// persisted is a comparable snapshot of the promised state of the home and of
// the settings file (which lives in the profile folder, beside the desktop
// preferences, not in the home).
type persisted struct {
	home, profile map[string]string
	// updates is the update subsystem's own slice of settings.json, compared by
	// value when the application that is restarted afterwards may legitimately
	// rewrite the rest of the file.
	updates update.Settings
	exact   bool
}

// snapshot records the promised state. With exact the settings file is
// compared byte for byte; otherwise only its update-owned settings by value.
func (rg *rig) snapshot(exact bool) persisted {
	rg.t.Helper()
	h, err := disposable.Snapshot(rg.home, promised)
	if err != nil {
		rg.t.Fatal(err)
	}
	p, err := disposable.Snapshot(rg.profile, settingsFile)
	if err != nil {
		rg.t.Fatal(err)
	}
	if len(h) < 8 || len(p) != 2 {
		rg.t.Fatalf("the snapshot is nearly empty (%d home entries, %d profile entries): the seed did not run", len(h), len(p))
	}
	u, err := rg.store.UpdateSettings()
	if err != nil {
		rg.t.Fatal(err)
	}
	return persisted{home: h, profile: p, updates: u, exact: exact}
}

func (a persisted) diff(b persisted) []string {
	var out []string
	for _, d := range disposable.Diff(a.home, b.home) {
		out = append(out, "home "+d)
	}
	if a.exact {
		for _, d := range disposable.Diff(a.profile, b.profile) {
			out = append(out, "profile "+d)
		}
	} else {
		if _, ok := b.profile["Hachidori/settings.json"]; !ok {
			out = append(out, "profile removed: Hachidori/settings.json")
		}
		if !reflect.DeepEqual(a.updates, b.updates) {
			out = append(out, fmt.Sprintf("update settings changed: %+v -> %+v", a.updates, b.updates))
		}
	}
	return out
}
