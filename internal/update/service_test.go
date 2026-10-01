package update

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func t0() context.Context { return context.Background() }

func (e *env) catalog(tags ...string) {
	for _, t := range tags {
		e.g.publish(t, true)
	}
	var ents []string
	for _, t := range tags {
		ents = append(ents, e.g.entry(t, !!strings.Contains(t, "-")))
	}
	e.g.setList(ents...)
}

// The observable invariant: using Hachidori and its Updates surface, changing the
// channel included, makes zero update requests. Only Check and Download do.
func TestNoUpdateNetworkWithoutAnExplicitAction(t *testing.T) {
	e := newEnv(t)
	e.catalog("0.2.5-dev", "0.2.1-dev")
	for i := 0; i < 3; i++ {
		_ = e.svc.Status() // opening Settings / Updates
		if err := e.svc.SetChannel(Development); err != nil {
			t.Fatal(err)
		}
		_ = e.svc.Status()
		if err := e.svc.SetChannel(Stable); err != nil {
			t.Fatal(err)
		}
	}
	if n := e.g.count(); n != 0 {
		t.Fatalf("%d update requests without an explicit action: %v", n, e.g.requests())
	}
	if st := e.svc.Status(); st.Check != nil || st.Channel != Stable || st.Busy != nil {
		t.Fatalf("status %+v", st)
	}
	e.check()
	for _, r := range e.g.requests() {
		if !strings.HasPrefix(r, apiHost+"/repos/"+Owner+"/"+Repo+"/releases") {
			t.Errorf("Check requested %s: only release metadata may be retrieved", r)
		}
	}
	if e.g.count() == 0 {
		t.Fatal("Check made no request")
	}
}

func TestChannelChangeNeverContactsDownloadsOrInstalls(t *testing.T) {
	e := newEnv(t)
	e.catalog("0.2.5-dev", "0.2.1-dev")
	if err := e.svc.SetChannel(Development); err != nil {
		t.Fatal(err)
	}
	e.check()
	e.download("0.2.5-dev")
	before, _ := os.ReadFile(e.exe)
	n := e.g.count()
	if err := e.svc.SetChannel(Stable); err != nil {
		t.Fatal(err)
	}
	if e.g.count() != n || len(e.started) != 0 {
		t.Fatal("channel change used the network or started the helper")
	}
	if after, _ := os.ReadFile(e.exe); string(after) != string(before) {
		t.Fatal("channel change touched the executable")
	}
	st := e.svc.Status()
	if st.Check != nil {
		t.Error("results of a check made for another channel stay displayed")
	}
	if st.Ready == nil || st.ReadyProblem == "" {
		t.Errorf("the downloaded development update must remain but not be installable on Stable: %+v %q", st.Ready, st.ReadyProblem)
	}
	if err := e.svc.Install(); err == nil || len(e.started) != 0 {
		t.Fatalf("install of a development update on Stable: %v", err)
	}
	if err := e.svc.SetChannel("beta"); err == nil {
		t.Error("an unknown channel was accepted")
	}
}

func TestCheckPresentsOnlyEligibleReleasesAndRemembersTheCheck(t *testing.T) {
	e := newEnv(t)
	e.g.publish("0.2.5-dev", true)
	e.g.publish("0.2.0", false)
	e.g.setList(e.g.entry("0.2.5-dev", true), e.g.entry("0.2.0", false), `{"tag_name":"dev-24","prerelease":true,"assets":[]}`)
	e.check()
	st := e.svc.Status()
	if st.Check == nil || len(st.Check.Releases) != 1 || st.Check.Releases[0].Tag != "0.2.0" || st.Check.Channel != Stable {
		t.Fatalf("stable check: %+v", st.Check)
	}
	if st.LastCheck == nil || !st.LastCheck.OK || st.LastCheck.Latest != "0.2.0" || st.LastCheck.Time.IsZero() {
		t.Fatalf("last check not remembered: %+v", st.LastCheck)
	}
	e.svc.SetChannel(Development)
	e.check()
	if got := tags(e.svc.Status().Check.Releases); strings.Join(got, " ") != "0.2.5-dev 0.2.0" {
		t.Fatalf("development: %v", got)
	}
}

func TestCheckWithNoEligibleReleasesIsNotAFailure(t *testing.T) {
	e := newEnv(t)
	e.g.setList(`{"tag_name":"dev-24","prerelease":true,"assets":[]}`)
	e.check()
	st := e.svc.Status()
	if st.Check == nil || len(st.Check.Releases) != 0 || st.LastCheck == nil || !st.LastCheck.OK || !strings.Contains(st.LastCheck.Message, "no eligible") {
		t.Fatalf("%+v %+v", st.Check, st.LastCheck)
	}
}

func TestCheckFailuresAreClassifiedAndLeaveNothingDisplayed(t *testing.T) {
	for name, tc := range map[string]struct {
		setup func(*fakeGitHub)
		class Class
		text  string
	}{
		"server error": {func(g *fakeGitHub) { g.apiStatus = 500 }, ClassNetwork, "500"},
		"rate limit":   {func(g *fakeGitHub) { g.apiStatus = 403 }, ClassNetwork, "limit"},
		"api redirect": {func(g *fakeGitHub) { g.apiRedir = "https://evil.example/x" }, ClassNetwork, "302"},
		"not json":     {func(g *fakeGitHub) { g.pages[1] = "<html>" }, ClassMalformed, "malformed"},
		"not a list":   {func(g *fakeGitHub) { g.pages[1] = `{"message":"x"}` }, ClassMalformed, "malformed"},
		"duplicate tag": {func(g *fakeGitHub) {
			g.publish("0.2.5-dev", true)
			g.setList(g.entry("0.2.5-dev", true), g.entry("0.2.5-dev", true))
		}, ClassMalformed, "twice"},
	} {
		e := newEnv(t)
		e.catalog("0.2.1-dev")
		e.check() // earlier results exist
		tc.setup(e.g)
		err := e.svc.Check(t0())
		var ue *Error
		if err == nil || !asError(err, &ue) || ue.Class != tc.class || !strings.Contains(err.Error(), tc.text) {
			t.Errorf("%s: %v (want class %s containing %q)", name, err, tc.class, tc.text)
			continue
		}
		st := e.svc.Status()
		if st.Check != nil || st.LastCheck == nil || st.LastCheck.OK || st.LastCheck.Message == "" {
			t.Errorf("%s: stale results or no failure remembered: %+v %+v", name, st.Check, st.LastCheck)
		}
		if got, _ := os.ReadFile(e.exe); string(got) != string(e.exeData) {
			t.Errorf("%s: executable touched", name)
		}
	}
}

func asError(err error, target **Error) bool {
	for err != nil {
		if e, ok := err.(*Error); ok {
			*target = e
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

func TestDownloadVerifiesAndOnlyThenIsReady(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	if e.svc.Status().Ready != nil {
		t.Fatal("ready before any download")
	}
	checkReqs := e.g.count()
	op := e.download("0.2.5-dev")
	if op == nil || op.Failure != nil {
		t.Fatalf("download: %+v", op)
	}
	if strings.Join(op.Phases, " ") != "checksum download verify" || op.Total == 0 || op.Done != op.Total {
		t.Errorf("progress phases/bytes: %+v", op)
	}
	want := []string{
		"github.com/yohn-jp/hachidori/releases/download/0.2.5-dev/" + SumAsset, "objects.githubusercontent.com/blob/0.2.5-dev/" + SumAsset,
		"github.com/yohn-jp/hachidori/releases/download/0.2.5-dev/" + ExeAsset, "objects.githubusercontent.com/blob/0.2.5-dev/" + ExeAsset,
	}
	if got := e.g.requests()[checkReqs:]; strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("download requests:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
	st := e.svc.Status()
	r, v, err := LoadReady(e.root)
	if err != nil || st.Ready == nil || r.Tag != "0.2.5-dev" || r.SHA256 != hexSum(exeBytes("0.2.5-dev")) || r.Target != e.exe || v.String() != "0.2.5-dev" || st.ReadyProblem != "" {
		t.Fatalf("ready: %+v %v %+v", r, err, st)
	}
	staged, _ := os.ReadFile(StagedPath(e.root, v))
	if string(staged) != string(exeBytes("0.2.5-dev")) {
		t.Fatal("staged file differs from the release asset")
	}
	if _, err := os.Stat(StagedPath(e.root, v) + ".part"); err == nil {
		t.Error("partial file left behind")
	}
	if got, _ := os.ReadFile(e.exe); string(got) != string(e.exeData) {
		t.Fatal("the running executable was touched by a download")
	}
	if len(e.started) != 0 {
		t.Fatal("download started the helper")
	}
}

func TestVerificationFailureLeavesNothingReadyAndTheExecutableUntouched(t *testing.T) {
	for name, tc := range map[string]struct {
		mutate   func(g *fakeGitHub, tag string)
		class    Class
		exeFetch bool // whether the executable asset is requested at all
	}{
		"checksum mismatch": {func(g *fakeGitHub, tag string) {
			g.blobs[tag+"/"+SumAsset] = []byte(strings.Repeat("0", 64) + "  " + ExeAsset + "\n")
		}, ClassChecksumMismatch, true},
		"corrupted download": {func(g *fakeGitHub, tag string) {
			b := append([]byte(nil), g.blobs[tag+"/"+ExeAsset]...)
			b[100] ^= 0xff
			g.blobs[tag+"/"+ExeAsset] = b
		}, ClassChecksumMismatch, true},
		"checksum asset missing": {func(g *fakeGitHub, tag string) { g.status[tag+"/"+SumAsset] = 404 }, ClassVerification, false},
		"checksum for another asset": {func(g *fakeGitHub, tag string) {
			g.blobs[tag+"/"+SumAsset] = []byte(hexSum(g.blobs[tag+"/"+ExeAsset]) + "  other.exe\n")
		}, ClassVerification, false},
		"checksum malformed": {func(g *fakeGitHub, tag string) { g.blobs[tag+"/"+SumAsset] = []byte("<html>not found</html>") }, ClassVerification, false},
		"checksum ambiguous": {func(g *fakeGitHub, tag string) {
			h := hexSum(g.blobs[tag+"/"+ExeAsset])
			g.blobs[tag+"/"+SumAsset] = []byte(h + "  " + ExeAsset + "\n" + h + "  " + ExeAsset + "\n")
		}, ClassVerification, false},
		"executable missing": {func(g *fakeGitHub, tag string) { g.status[tag+"/"+ExeAsset] = 404 }, ClassDownload, true},
		"executable truncated": {func(g *fakeGitHub, tag string) {
			g.blobs[tag+"/"+ExeAsset] = g.blobs[tag+"/"+ExeAsset][:2000]
		}, ClassDownload, true},
		"executable too long": {func(g *fakeGitHub, tag string) {
			g.blobs[tag+"/"+ExeAsset] = append(g.blobs[tag+"/"+ExeAsset], 'x')
		}, ClassDownload, true},
		"not an executable": {func(g *fakeGitHub, tag string) {
			b := []byte("<html>" + strings.Repeat("x", 4000))
			g.blobs[tag+"/"+ExeAsset] = b
			g.blobs[tag+"/"+SumAsset] = []byte(hexSum(b) + "  " + ExeAsset + "\n")
		}, ClassAsset, true},
	} {
		e := newEnv(t)
		e.svc.SetChannel(Development)
		e.catalog("0.2.5-dev")
		e.check() // metadata sizes are taken from the pristine release
		tc.mutate(e.g, "0.2.5-dev")
		if name == "not an executable" {
			e.g.setList(e.g.entry("0.2.5-dev", true)) // the release's own metadata matches what it serves
			e.check()
		}
		op := e.download("0.2.5-dev")
		if op == nil || op.Failure == nil || op.Failure.Class != tc.class || op.Failure.Message == "" {
			t.Errorf("%s: %+v", name, op)
			continue
		}
		st := e.svc.Status()
		if st.Ready != nil {
			t.Errorf("%s: ready after a failed verification", name)
		}
		if _, err := os.Stat(StagedPath(e.root, mustVersion("0.2.5-dev"))); err == nil {
			t.Errorf("%s: unverified executable left at its final name", name)
		}
		if _, err := os.Stat(StagedPath(e.root, mustVersion("0.2.5-dev")) + ".part"); err == nil {
			t.Errorf("%s: partial file left behind", name)
		}
		if got, _ := os.ReadFile(e.exe); string(got) != string(e.exeData) {
			t.Errorf("%s: executable touched", name)
		}
		fetchedExe := false
		for _, r := range e.g.requests() {
			if strings.HasSuffix(r, "/"+ExeAsset) {
				fetchedExe = true
			}
		}
		if fetchedExe != tc.exeFetch {
			t.Errorf("%s: executable fetched=%v, want %v", name, fetchedExe, tc.exeFetch)
		}
		if len(e.started) != 0 || e.svc.Install() == nil {
			t.Errorf("%s: install possible after failure", name)
		}
	}
}

func mustVersion(s string) Version { v, _ := ParseVersion(s); return v }

func TestFailedDownloadKeepsAnEarlierVerifiedUpdate(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.6-dev", "0.2.5-dev")
	e.check()
	e.download("0.2.5-dev")
	e.g.blobs["0.2.6-dev/"+SumAsset] = []byte(strings.Repeat("1", 64) + "  " + ExeAsset)
	if op := e.download("0.2.6-dev"); op.Failure == nil {
		t.Fatal("expected failure")
	}
	if r, _, err := LoadReady(e.root); err != nil || r.Tag != "0.2.5-dev" {
		t.Fatalf("earlier ready update lost: %+v %v", r, err)
	}
	// A later successful download replaces the ready update and its staged file.
	e.g.blobs["0.2.6-dev/"+SumAsset] = []byte(hexSum(e.g.blobs["0.2.6-dev/"+ExeAsset]) + "  " + ExeAsset)
	e.download("0.2.6-dev")
	if r, _, _ := LoadReady(e.root); r.Tag != "0.2.6-dev" {
		t.Fatalf("ready %+v", r)
	}
	if _, err := os.Stat(StagedPath(e.root, mustVersion("0.2.5-dev"))); err == nil {
		t.Error("the superseded staged executable was kept")
	}
}

func TestDownloadRedirectsAreConfinedToGitHubAssetHosts(t *testing.T) {
	for name, to := range map[string]string{
		"foreign host":   "https://evil.example/hachidori.exe",
		"plain http":     "http://objects.githubusercontent.com/blob/x",
		"other port":     "https://objects.githubusercontent.com:8443/blob/x",
		"credentials":    "https://user:pw@objects.githubusercontent.com/blob/x",
		"github subhost": "https://uploads.github.com/x",
		"look-alike":     "https://objects.githubusercontent.com.evil.example/x",
		"api host":       "https://api.github.com/repos/yohn-jp/hachidori/releases",
	} {
		for _, asset := range []string{ExeAsset, SumAsset} {
			e := newEnv(t)
			e.svc.SetChannel(Development)
			e.catalog("0.2.5-dev")
			e.check()
			e.g.redirectTo["0.2.5-dev/"+asset] = to
			op := e.download("0.2.5-dev")
			if op.Failure == nil || !strings.Contains(op.Failure.Message, "refused") {
				t.Errorf("%s/%s: %+v", name, asset, op)
			}
			for _, r := range e.g.requests() {
				if strings.Contains(r, "evil.example") || strings.Contains(r, "uploads.github") || strings.Contains(r, ":8443") {
					t.Errorf("%s: followed the redirect: %s", name, r)
				}
			}
			if e.svc.Status().Ready != nil {
				t.Errorf("%s: ready", name)
			}
		}
	}
}

func TestChecksumDigestFromMetadataMustAgree(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.g.publish("0.2.5-dev", true)
	ent := e.g.entry("0.2.5-dev", true)
	// GitHub reports a digest for the executable that differs from the checksum file.
	ent = strings.Replace(ent, `"state"`, `"digest":"sha256:`+strings.Repeat("c", 64)+`","state"`, 1)
	e.g.setList(ent)
	e.check()
	op := e.download("0.2.5-dev")
	if op.Failure == nil || op.Failure.Class != ClassVerification {
		t.Fatalf("%+v", op)
	}
}

func TestDownloadIsBoundToTheMostRecentCheck(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	if err := e.svc.StartDownload("0.2.5-dev"); err == nil {
		t.Fatal("download without a check")
	}
	e.check()
	for _, tag := range []string{"0.2.9-dev", "dev-24", "../0.2.5-dev", "", "0.2.5-dev/../x"} {
		if err := e.svc.StartDownload(tag); err == nil {
			t.Errorf("download of %q accepted", tag)
		}
	}
	n := e.g.count()
	if e.download("0.2.5-dev").Failure != nil {
		t.Fatal("download failed")
	}
	if e.g.count() == n {
		t.Fatal("no request made")
	}
	if err := e.svc.StartDownload("0.2.5-dev"); err != nil { // retry is allowed
		t.Fatal(err)
	}
	if err := e.svc.StartDownload("0.2.5-dev"); err == nil {
		t.Error("two downloads at once")
	}
	e.wait()
}

func TestNoDowngradeAndReleasesWithProblemsAreNotDownloadable(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.g.publish("0.2.5-dev", true)
	e.g.publish("0.2.1-dev", true)
	e.g.publish("0.2.7-dev", true)
	broken := strings.Replace(e.g.entry("0.2.7-dev", true), SumAsset, "other.sha256", 1)
	e.g.setList(broken, e.g.entry("0.2.5-dev", true), e.g.entry("0.2.1-dev", true))
	// Tell the service which release this executable is (digest known to the check).
	e.set.s.Installed = &Installed{Version: "0.2.5-dev", SHA256: hexSum(e.exeData)}
	e.check()
	st := e.svc.Status()
	if !st.Installed.Known || st.Installed.Version != "0.2.5-dev" {
		t.Fatalf("installed %+v", st.Installed)
	}
	rel := map[string]Candidate{}
	for _, c := range st.Check.Releases {
		rel[c.Tag] = c
	}
	if rel["0.2.7-dev"].Problem == "" || rel["0.2.7-dev"].Downloadable() {
		t.Errorf("0.2.7-dev with a missing checksum asset: %+v", rel["0.2.7-dev"])
	}
	if rel["0.2.5-dev"].Relation != RelCurrent || rel["0.2.1-dev"].Relation != RelOlder {
		t.Errorf("relations %+v", rel)
	}
	for _, tag := range []string{"0.2.7-dev", "0.2.5-dev", "0.2.1-dev"} {
		if err := e.svc.StartDownload(tag); err == nil {
			t.Errorf("download of %s accepted", tag)
		}
	}
	if len(e.g.requests()) != e.g.count() || strings.Contains(strings.Join(e.g.requests(), " "), "/releases/download/") {
		t.Error("an artifact was requested for a release that must not be downloaded")
	}
}

func TestInstalledReleaseIsIdentifiedByDigestOnlyAfterAnExplicitCheck(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.g.publish("0.2.5-dev", true)
	e.g.publish("0.2.6-dev", true)
	// GitHub reports the executable's own digest for 0.2.5-dev.
	ent := strings.Replace(e.g.entry("0.2.5-dev", true), `"state"`, `"digest":"sha256:`+hexSum(e.exeData)+`","state"`, 1)
	e.g.setList(e.g.entry("0.2.6-dev", true), ent)
	if st := e.svc.Status(); st.Installed.Known || st.Installed.SHA256 != hexSum(e.exeData) {
		t.Fatalf("before a check the version must be unknown: %+v", st.Installed)
	}
	if e.g.count() != 0 {
		t.Fatal("status used the network")
	}
	e.check()
	st := e.svc.Status()
	if !st.Installed.Known || st.Installed.Version != "0.2.5-dev" {
		t.Fatalf("not identified: %+v", st.Installed)
	}
	if st.Check.Releases[0].Relation != RelNewer || st.Check.Releases[1].Relation != RelCurrent {
		t.Errorf("relations %+v", st.Check.Releases)
	}
	// Remembered for exactly this executable; a different executable does not inherit it.
	n := e.g.count()
	if st := e.svc.Status(); !st.Installed.Known || e.g.count() != n {
		t.Error("identity not remembered locally")
	}
	os.WriteFile(e.exe, exeBytes("another build"), 0o755)
	other := &Service{Home: e.svc.Home, Exe: e.exe, Settings: e.set, Transport: e.g}
	if other.Status().Installed.Known {
		t.Error("the version was believed for a different executable")
	}
}

func TestInstallHandsOverToAHelperWithoutTouchingTheExecutable(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.svc.RestartHome = filepath.Join(filepath.Dir(e.root), "elsewhere")
	e.catalog("0.2.5-dev")
	if err := e.svc.Install(); err == nil {
		t.Fatal("install without a ready update")
	}
	e.check()
	e.download("0.2.5-dev")
	n := e.g.count()
	if err := e.svc.Install(); err != nil {
		t.Fatal(err)
	}
	if e.g.count() != n {
		t.Fatal("install used the network")
	}
	if len(e.started) != 1 {
		t.Fatalf("helper starts: %v", e.started)
	}
	helper, args := e.started[0][0], e.started[0][1:]
	if filepath.Dir(helper) != filepath.Join(Dir(e.root), "helper") || helper == e.exe {
		t.Errorf("helper %s", helper)
	}
	if b, _ := os.ReadFile(helper); string(b) != string(e.exeData) {
		t.Error("the helper is not a copy of the running executable")
	}
	want := []string{"apply-update", "--home", e.root, "--pid", itoa(int64(os.Getpid())), "--target", e.exe, "--restart-home", e.svc.RestartHome}
	if strings.Join(args, "\x00") != strings.Join(want, "\x00") {
		t.Errorf("helper args %q, want %q", args, want)
	}
	if got, _ := os.ReadFile(e.exe); string(got) != string(e.exeData) {
		t.Fatal("Install overwrote the running executable")
	}
}

func TestInstallRefusesWhatIsNotAVerifiedNewerUpdateForThisExecutable(t *testing.T) {
	prepare := func() *env {
		e := newEnv(t)
		e.svc.SetChannel(Development)
		e.catalog("0.2.5-dev")
		e.check()
		e.download("0.2.5-dev")
		return e
	}
	for name, mutate := range map[string]func(e *env){
		"staged file tampered": func(e *env) {
			p := StagedPath(e.root, mustVersion("0.2.5-dev"))
			b, _ := os.ReadFile(p)
			b[50] ^= 1
			os.WriteFile(p, b, 0o755)
		},
		"staged file replaced by a different size": func(e *env) {
			os.WriteFile(StagedPath(e.root, mustVersion("0.2.5-dev")), []byte("MZ"), 0o755)
		},
		"staged file removed": func(e *env) { os.Remove(StagedPath(e.root, mustVersion("0.2.5-dev"))) },
		"prepared for another executable": func(e *env) {
			e.svc.Exe = filepath.Join(filepath.Dir(e.exe), "other.exe")
			os.WriteFile(e.svc.Exe, e.exeData, 0o755)
		},
		"not newer than installed": func(e *env) {
			e.set.s.Installed = &Installed{Version: "0.2.9-dev", SHA256: hexSum(e.exeData)}
		},
		"channel does not offer it": func(e *env) { e.svc.SetChannel(Stable) },
		"ready record forged": func(e *env) {
			var r Ready
			r, _, _ = LoadReady(e.root)
			r.Asset = "evil.exe"
			writeJSONFile(t, readyPath(e.root), r)
		},
		"no helper support": func(e *env) { e.svc.StartHelper = nil },
		"executable is not a PE": func(e *env) {
			os.WriteFile(e.exe, []byte("not an exe"), 0o755)
			os.WriteFile(e.exe, []byte("not an exe"), 0o755)
		},
	} {
		e := prepare()
		mutate(e)
		if err := e.svc.Install(); err == nil {
			t.Errorf("%s: install accepted", name)
		}
		if len(e.started) != 0 {
			t.Errorf("%s: helper started", name)
		}
	}
}

func TestInstallIsRefusedWhileAnotherUpdateActionRuns(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	e.download("0.2.5-dev")
	e.svc.mu.Lock()
	e.svc.op = &Operation{Kind: KindDownload}
	e.svc.mu.Unlock()
	if err := e.svc.Install(); err == nil || len(e.started) != 0 {
		t.Fatalf("install during an operation: %v", err)
	}
	if err := e.svc.Check(t0()); err == nil {
		t.Fatal("check during an operation")
	}
}

func TestResultOfAnAppliedUpdateIdentifiesTheInstalledRelease(t *testing.T) {
	e := newEnv(t)
	if e.svc.Status().Installed.Known {
		t.Fatal("known without evidence")
	}
	res := Result{Tag: "0.2.5-dev", SHA256: hexSum(e.exeData), Outcome: OutcomeApplied, Target: e.exe, Time: e.clock}
	os.MkdirAll(Dir(e.root), 0o755)
	if err := writeResult(e.root, res); err != nil {
		t.Fatal(err)
	}
	st := e.svc.Status()
	if !st.Installed.Known || st.Installed.Version != "0.2.5-dev" || st.Result == nil || st.Result.Outcome != OutcomeApplied {
		t.Fatalf("%+v", st)
	}
	res.SHA256 = strings.Repeat("0", 64) // a result for another executable
	writeResult(e.root, res)
	if e.svc.Status().Installed.Known {
		t.Error("a result for another executable identified this one")
	}
}

func TestStatusReadsNoNetworkEvenWithEveryStateOnDisk(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	e.download("0.2.5-dev")
	n := e.g.count()
	for i := 0; i < 5; i++ {
		_ = e.svc.Status()
	}
	if e.g.count() != n {
		t.Fatal("Status used the network")
	}
}

func TestSettingsValidation(t *testing.T) {
	good := Settings{Channel: Development, LastCheck: &LastCheck{Channel: Stable}, Installed: &Installed{Version: "0.2.5-dev", SHA256: strings.Repeat("a", 64)}}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, s := range map[string]Settings{
		"channel":      {Channel: "beta"},
		"last channel": {LastCheck: &LastCheck{Channel: "beta"}},
		"version":      {Installed: &Installed{Version: "dev-24", SHA256: strings.Repeat("a", 64)}},
		"digest":       {Installed: &Installed{Version: "0.2.5-dev", SHA256: "abc"}},
	} {
		if s.Validate() == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestRequestsAreRefusedOutsideTheOfficialAuthority(t *testing.T) {
	e := newEnv(t)
	for _, u := range []string{"https://evil.example/x", "http://api.github.com/x", "https://github.com.evil.example/x", "https://raw.githubusercontent.com/x"} {
		if resp, err := e.svc.get(t0(), u, "*/*", true); err == nil {
			resp.Body.Close()
			t.Errorf("requested %s", u)
		}
	}
	if e.g.count() != 0 {
		t.Fatal("a refused request reached the transport")
	}
}

func TestStalledOrSlowChecksAreBoundedByContext(t *testing.T) {
	e := newEnv(t)
	ctx, cancel := context.WithTimeout(t0(), time.Millisecond)
	defer cancel()
	e.svc.Transport = blockTransport{}
	if err := e.svc.Check(ctx); err == nil {
		t.Fatal("check did not fail on a stalled authority")
	}
	if st := e.svc.Status(); st.Checking || st.Check != nil {
		t.Fatalf("%+v", st)
	}
}
