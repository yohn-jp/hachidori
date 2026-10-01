package update

import (
	"errors"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// rig is one prepared update: a downloaded, verified 0.2.5-dev, the installed
// executable and a HACHIDORI_HOME with data an update must never touch.
type rig struct {
	*env
	newData   []byte
	restarts  [][]string
	waits     []int
	sleeps    int
	renames   []string
	sentinels map[string]string
}

func newRig(t *testing.T) *rig {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	e.download("0.2.5-dev")
	r := &rig{env: e, newData: exeBytes("0.2.5-dev"), sentinels: map[string]string{}}
	for rel, content := range map[string]string{
		"models/laya-base/model.safetensors": "weights", "runtime/cu128-x/manifest.json": "{}",
		"state/history/h1.json": "evidence", "state/active-runtime.json": "{}", "logs/worker.log": "log", "cache/x": "c",
	} {
		p := filepath.Join(e.root, rel)
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, []byte(content), 0o644)
		r.sentinels[p] = content
	}
	return r
}

func (r *rig) env_() ApplyEnv {
	return ApplyEnv{
		WaitExit: func(pid int, d time.Duration) error { r.waits = append(r.waits, pid); return nil },
		Restart: func(exe string, args []string) error {
			r.restarts = append(r.restarts, append([]string{exe}, args...))
			return nil
		},
		Rename: func(a, b string) error {
			r.renames = append(r.renames, filepath.Base(a)+">"+filepath.Base(b))
			return os.Rename(a, b)
		},
		Sleep:       func(time.Duration) { r.sleeps++ },
		Now:         func() time.Time { return r.clock },
		ExitTimeout: time.Second,
	}
}

// restarted is the service of the process the helper started: same home,
// executable path and settings, nothing remembered in memory.
func (r *rig) restarted() *Service {
	return &Service{Home: r.svc.Home, Exe: r.exe, Settings: r.set, Transport: r.g}
}

func (r *rig) plan() Plan { return Plan{Home: r.root, PID: os.Getpid() + 1, Target: r.exe} }

func (r *rig) assertHomeUntouched() {
	r.t.Helper()
	for p, want := range r.sentinels {
		if got, err := os.ReadFile(p); err != nil || string(got) != want {
			r.t.Errorf("%s changed: %q %v", p, got, err)
		}
	}
}

func (r *rig) targetIs(data []byte) bool {
	got, err := os.ReadFile(r.exe)
	return err == nil && string(got) == string(data)
}

func TestApplyReplacesAfterTheApplicationExitedAndRestartsTheSamePath(t *testing.T) {
	r := newRig(t)
	ae := r.env_()
	ae.WaitExit = func(pid int, d time.Duration) error {
		if !r.targetIs(r.exeData) {
			t.Error("the destination changed before the application exited")
		}
		if _, err := os.Stat(suffixNew(r.exe)); err == nil {
			t.Error("a copy was prepared before the application exited")
		}
		r.waits = append(r.waits, pid)
		return nil
	}
	p := r.plan()
	p.RestartHome = filepath.Join(r.root, "..", "explicit")
	p.RestartHome = filepath.Clean(p.RestartHome)
	res := Apply(p, ae)
	if res.Outcome != OutcomeApplied || res.Tag != "0.2.5-dev" {
		t.Fatalf("%+v", res)
	}
	if len(r.waits) != 1 || r.waits[0] != p.PID {
		t.Errorf("waits %v", r.waits)
	}
	if !r.targetIs(r.newData) {
		t.Fatal("the destination is not the update")
	}
	if old, _ := os.ReadFile(suffixOld(r.exe)); string(old) != string(r.exeData) {
		t.Error("the previous executable was not kept")
	}
	if _, err := os.Stat(suffixNew(r.exe)); err == nil {
		t.Error("temporary copy left behind")
	}
	if len(r.restarts) != 1 || r.restarts[0][0] != r.exe || strings.Join(r.restarts[0][1:], " ") != "desktop --home "+p.RestartHome {
		t.Errorf("restart %v", r.restarts)
	}
	if _, err := os.Stat(readyPath(r.root)); err == nil {
		t.Error("ready record kept after commit")
	}
	if _, err := os.Stat(StagedPath(r.root, mustVersion("0.2.5-dev"))); err == nil {
		t.Error("staged file kept after commit")
	}
	got, err := LoadResult(r.root)
	if err != nil || got.Outcome != OutcomeApplied || got.SHA256 != hexSum(r.newData) || got.Target != r.exe {
		t.Fatalf("result %+v %v", got, err)
	}
	r.assertHomeUntouched()
	// The updated executable recognizes itself from the result, with no network.
	n := r.g.count()
	if st := r.restarted().Status(); !st.Installed.Known || st.Installed.Version != "0.2.5-dev" || r.g.count() != n {
		t.Errorf("after update: %+v", st.Installed)
	}
}

func TestApplyWithoutExplicitHomeRestartsWithNoArguments(t *testing.T) {
	r := newRig(t)
	if res := Apply(r.plan(), r.env_()); res.Outcome != OutcomeApplied {
		t.Fatal(res)
	}
	if len(r.restarts) != 1 || len(r.restarts[0]) != 1 {
		t.Fatalf("restart %v", r.restarts)
	}
}

func TestApplyReplacesAStalePreviousCopy(t *testing.T) {
	r := newRig(t)
	os.WriteFile(suffixOld(r.exe), []byte("MZ stale"), 0o755)
	if res := Apply(r.plan(), r.env_()); res.Outcome != OutcomeApplied {
		t.Fatal(res)
	}
	if old, _ := os.ReadFile(suffixOld(r.exe)); string(old) != string(r.exeData) {
		t.Error("stale previous copy not replaced")
	}
}

func TestApplyKeepsTheCurrentExecutableWhenTheApplicationDoesNotExit(t *testing.T) {
	r := newRig(t)
	ae := r.env_()
	ae.WaitExit = func(int, time.Duration) error { return errors.New("still running") }
	res := Apply(r.plan(), ae)
	if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, "did not exit") {
		t.Fatalf("%+v", res)
	}
	if !r.targetIs(r.exeData) || len(r.restarts) != 0 || len(r.renames) != 0 {
		t.Fatalf("touched: restarts=%v renames=%v", r.restarts, r.renames)
	}
	if _, err := os.Stat(readyPath(r.root)); err != nil {
		t.Error("ready record lost; the update can be retried")
	}
	r.assertHomeUntouched()
}

func TestApplyRefusesWhatTheReadyRecordDoesNotAuthorize(t *testing.T) {
	for name, tc := range map[string]struct {
		mutate func(r *rig, p *Plan)
	}{
		"destination differs from the prepared one": {func(r *rig, p *Plan) {
			p.Target = filepath.Join(filepath.Dir(r.exe), "other.exe")
			os.WriteFile(p.Target, []byte("MZ other"), 0o755)
		}},
		"destination in the update area": {func(r *rig, p *Plan) {
			p.Target = StagedPath(r.root, mustVersion("0.2.5-dev"))
			rec, _, _ := LoadReady(r.root)
			rec.Target = p.Target
			writeJSONFile(t, readyPath(r.root), rec)
		}},
		"no process to wait for": {func(r *rig, p *Plan) { p.PID = 0 }},
		"waiting for itself":     {func(r *rig, p *Plan) { p.PID = os.Getpid() }},
		"relative home":          {func(r *rig, p *Plan) { p.Home = "home" }},
		"unclean home":           {func(r *rig, p *Plan) { p.Home = r.root + "/../home" }},
		"relative destination":   {func(r *rig, p *Plan) { p.Target = "hachidori.exe" }},
		"relative restart home":  {func(r *rig, p *Plan) { p.RestartHome = "x" }},
		"no ready record":        {func(r *rig, p *Plan) { os.Remove(readyPath(r.root)) }},
		"ready record wrong asset": {func(r *rig, p *Plan) {
			rec, _, _ := LoadReady(r.root)
			rec.Asset = "x.exe"
			writeJSONFile(t, readyPath(r.root), rec)
		}},
		"ready record traversal tag": {func(r *rig, p *Plan) {
			rec, _, _ := LoadReady(r.root)
			rec.Tag = "../../0.2.5-dev"
			writeJSONFile(t, readyPath(r.root), rec)
		}},
	} {
		r := newRig(t)
		p := r.plan()
		tc.mutate(r, &p)
		before, _ := os.ReadFile(r.exe)
		res := Apply(p, r.env_())
		if res.Outcome != OutcomeRefused {
			t.Errorf("%s: %+v", name, res)
		}
		if len(r.waits) != 0 || len(r.restarts) != 0 || len(r.renames) != 0 {
			t.Errorf("%s: acted: waits=%v restarts=%v renames=%v", name, r.waits, r.restarts, r.renames)
		}
		if after, _ := os.ReadFile(r.exe); string(after) != string(before) {
			t.Errorf("%s: the executable changed", name)
		}
		r.assertHomeUntouched()
	}
}

func TestApplyDoesNotReplaceWithAStagedFileThatNoLongerMatches(t *testing.T) {
	for name, mutate := range map[string]func(path string){
		"modified": func(p string) { b, _ := os.ReadFile(p); b[10] ^= 1; os.WriteFile(p, b, 0o755) },
		"swapped":  func(p string) { os.WriteFile(p, []byte("MZ evil"), 0o755) },
		"deleted":  func(p string) { os.Remove(p) },
		"symlink": func(p string) {
			os.Remove(p)
			os.Symlink(os.Args[0], p)
		},
	} {
		r := newRig(t)
		mutate(StagedPath(r.root, mustVersion("0.2.5-dev")))
		res := Apply(r.plan(), r.env_())
		if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, "not usable") {
			t.Errorf("%s: %+v", name, res)
		}
		if !r.targetIs(r.exeData) || len(r.renames) != 0 {
			t.Errorf("%s: the destination was touched", name)
		}
		if len(r.restarts) != 1 || r.restarts[0][0] != r.exe {
			t.Errorf("%s: the previous executable was not restarted: %v", name, r.restarts)
		}
	}
}

func TestApplyOnlyReplacesARegularWindowsExecutableFile(t *testing.T) {
	for name, mutate := range map[string]func(r *rig){
		"not a PE":  func(r *rig) { os.WriteFile(r.exe, []byte("#!/bin/sh"), 0o755) },
		"directory": func(r *rig) { os.Remove(r.exe); os.Mkdir(r.exe, 0o755) },
		"symlink": func(r *rig) {
			other := filepath.Join(filepath.Dir(r.exe), "real.exe")
			os.WriteFile(other, r.exeData, 0o755)
			os.Remove(r.exe)
			os.Symlink(other, r.exe)
		},
		"missing": func(r *rig) { os.Remove(r.exe) },
	} {
		r := newRig(t)
		mutate(r)
		res := Apply(r.plan(), r.env_())
		if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, "not a replaceable executable") {
			t.Errorf("%s: %+v", name, res)
		}
		if len(r.renames) != 0 || len(r.restarts) != 0 {
			t.Errorf("%s: acted: %v %v", name, r.renames, r.restarts)
		}
	}
	r := newRig(t)
	ren := filepath.Join(filepath.Dir(r.exe), "hachidori.bin")
	os.Rename(r.exe, ren)
	r.exe = ren
	rec, _, _ := LoadReady(r.root)
	rec.Target = ren
	writeJSONFile(t, readyPath(r.root), rec)
	if res := Apply(r.plan(), r.env_()); res.Outcome != OutcomeFailed {
		t.Errorf("a destination that is not named *.exe was replaced: %+v", res)
	}
}

func TestApplyFailureBeforeTheSwapKeepsTheExecutableAndRestartsIt(t *testing.T) {
	r := newRig(t)
	os.Mkdir(suffixNew(r.exe), 0o755) // the update cannot be prepared beside the destination
	res := Apply(r.plan(), r.env_())
	if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, "could not be prepared") {
		t.Fatalf("%+v", res)
	}
	if !r.targetIs(r.exeData) || len(r.restarts) != 1 || len(r.renames) != 0 {
		t.Fatalf("destination touched or not restarted: %v %v", r.restarts, r.renames)
	}
	if _, err := os.Stat(readyPath(r.root)); err != nil {
		t.Error("ready record lost")
	}
}

func TestApplyRenameFailuresAreRecoverable(t *testing.T) {
	// Moving the current executable aside fails: it stays, and is restarted.
	r := newRig(t)
	ae := r.env_()
	ae.Rename = func(a, b string) error {
		r.renames = append(r.renames, filepath.Base(a)+">"+filepath.Base(b))
		if a == r.exe {
			return errors.New("access denied")
		}
		return os.Rename(a, b)
	}
	res := Apply(r.plan(), ae)
	if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, "kept") || !r.targetIs(r.exeData) {
		t.Fatalf("%+v", res)
	}
	if r.sleeps != renameTries-1+1 && r.sleeps != renameTries {
		t.Errorf("retries: %d sleeps", r.sleeps)
	}
	if _, err := os.Stat(suffixNew(r.exe)); err == nil {
		t.Error("temporary copy left behind")
	}
	if len(r.restarts) != 1 || r.restarts[0][0] != r.exe {
		t.Errorf("restarts %v", r.restarts)
	}
	r.assertHomeUntouched()

	// Putting the update in place fails: the previous executable is restored.
	r = newRig(t)
	ae = r.env_()
	ae.Rename = func(a, b string) error {
		r.renames = append(r.renames, filepath.Base(a)+">"+filepath.Base(b))
		if b == r.exe && a == suffixNew(r.exe) {
			return errors.New("sharing violation")
		}
		return os.Rename(a, b)
	}
	res = Apply(r.plan(), ae)
	if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, "restored") || !r.targetIs(r.exeData) {
		t.Fatalf("%+v", res)
	}
	if len(r.restarts) != 1 {
		t.Errorf("restarts %v", r.restarts)
	}
	if _, err := os.Stat(readyPath(r.root)); err != nil {
		t.Error("ready record must stay so Restart & update can be retried")
	}
	if got, _ := LoadResult(r.root); got.Outcome != OutcomeFailed {
		t.Errorf("result %+v", got)
	}
	r.assertHomeUntouched()

	// Restoring fails too: the failure says where the previous executable is.
	r = newRig(t)
	ae = r.env_()
	ae.Rename = func(a, b string) error {
		if b == r.exe {
			return errors.New("locked")
		}
		return os.Rename(a, b)
	}
	res = Apply(r.plan(), ae)
	if res.Outcome != OutcomeFailed || !strings.Contains(res.Message, suffixOld(r.exe)) {
		t.Fatalf("%+v", res)
	}
	if old, _ := os.ReadFile(suffixOld(r.exe)); string(old) != string(r.exeData) {
		t.Error("the previous executable is not recoverable at target.old")
	}
}

func TestApplyReportsARestartFailureAfterACommittedReplacement(t *testing.T) {
	r := newRig(t)
	ae := r.env_()
	ae.Restart = func(string, []string) error { return errors.New("launch refused") }
	res := Apply(r.plan(), ae)
	if res.Outcome != OutcomeRestart || !strings.Contains(res.Message, "launch refused") || !r.targetIs(r.newData) {
		t.Fatalf("%+v", res)
	}
	if got, _ := LoadResult(r.root); got.Outcome != OutcomeRestart {
		t.Errorf("result %+v", got)
	}
	// The update is in place and recognized even though the restart failed.
	if st := r.restarted().Status(); !st.Installed.Known {
		t.Error("committed update not recognized")
	}
}

func TestApplyDoesNotRequireTheNetwork(t *testing.T) {
	r := newRig(t)
	n := r.g.count()
	r.svc.Transport = blockTransport{} // would hang if used
	Apply(r.plan(), r.env_())
	if r.g.count() != n {
		t.Fatal("Apply used the network")
	}
}

// The helper is bounded by construction: its code opens no network connection
// and starts nothing but the destination it replaced.
func TestHelperCodeHasNoNetworkOrGeneralDownloaderDependency(t *testing.T) {
	for _, file := range []string{"apply.go", "proc_other.go", "proc_windows.go"} {
		f, err := parser.ParseFile(token.NewFileSet(), file, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if strings.HasPrefix(path, "net") || strings.Contains(path, "http") {
				t.Errorf("%s imports %s", file, path)
			}
		}
	}
	src, _ := os.ReadFile("apply.go")
	if strings.Contains(string(src), "s.get(") || strings.Contains(string(src), "Service") {
		t.Error("the helper must not use the Service or its downloader")
	}
}

// Every request the subsystem can make goes through Service.get, which only
// builds requests for the two fixed authority hosts. Nothing else in the
// package may import net/http or open a connection.
func TestOneNetworkSeam(t *testing.T) {
	files, _ := filepath.Glob("*.go")
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), file, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range f.Imports {
			if path, _ := strconv.Unquote(imp.Path.Value); path == "net/http" && file != "service.go" {
				t.Errorf("%s imports net/http: requests belong in service.go", file)
			}
			if path, _ := strconv.Unquote(imp.Path.Value); path == "net" {
				t.Errorf("%s imports net", file)
			}
		}
		src, _ := os.ReadFile(file)
		if n := strings.Count(string(src), "http.NewRequest"); n != 0 && (file != "service.go" || n != 1) {
			t.Errorf("%s builds %d requests", file, n)
		}
	}
}

func TestInstallClearsEarlierHelperCopies(t *testing.T) {
	r := newRig(t)
	stale := filepath.Join(helperDir(r.root), "hachidori-update-1.exe")
	os.MkdirAll(helperDir(r.root), 0o755)
	os.WriteFile(stale, []byte("MZ old helper"), 0o755)
	if err := r.svc.Install(); err != nil {
		t.Fatal(err)
	}
	ents, _ := os.ReadDir(helperDir(r.root))
	if _, err := os.Stat(stale); err == nil || len(ents) != 1 {
		t.Errorf("helper dir: %v", ents)
	}
}

func TestRefusedPlansCreateNothing(t *testing.T) {
	r := newRig(t)
	missing := filepath.Join(t.TempDir(), "missing-home")
	for _, p := range []Plan{
		{Home: missing, PID: os.Getpid() + 1, Target: r.exe},
		{Home: "relative-home", PID: os.Getpid() + 1, Target: r.exe},
		{Home: missing, PID: 0, Target: "x.exe"},
	} {
		if res := Apply(p, r.env_()); res.Outcome != OutcomeRefused {
			t.Errorf("%+v: %+v", p, res)
		}
	}
	if _, err := os.Stat(missing); !os.IsNotExist(err) {
		t.Errorf("a refused plan created %s", missing)
	}
	if _, err := os.Stat("relative-home"); !os.IsNotExist(err) {
		t.Error("a refused plan created a relative directory")
	}
}
