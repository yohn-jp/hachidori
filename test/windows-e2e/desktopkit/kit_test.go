package desktopkit

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
)

func TestProfileEnvConfinesTheChild(t *testing.T) {
	root := t.TempDir()
	p, err := NewProfile(root)
	if err != nil {
		t.Fatal(err)
	}
	base := []string{
		"PATH=/bin", "LocalAppData=C:\\Real\\Local", "APPDATA=C:\\Real\\Roaming", "UserProfile=C:\\Real",
		"HACHIDORI_HOME=C:\\RealHome", "hachidori_windows_e2e=1", "TEMP=C:\\Real\\Temp", "KEEP=1",
	}
	env := p.Env(base, "HACHIDORI_HOME="+filepath.Join(root, "h"))
	got := map[string]string{}
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		if _, dup := got[strings.ToUpper(k)]; dup {
			t.Errorf("duplicate %s in the child environment", k)
		}
		got[strings.ToUpper(k)] = v
	}
	for k, want := range map[string]string{
		"LOCALAPPDATA": p.LocalAppData, "APPDATA": p.AppData, "USERPROFILE": p.UserProfile, "TEMP": p.Temp, "TMP": p.Temp,
		"HACHIDORI_HOME": filepath.Join(root, "h"), "PATH": "/bin", "KEEP": "1",
	} {
		if got[k] != want {
			t.Errorf("%s = %q, want %q", k, got[k], want)
		}
	}
	if _, ok := got["HACHIDORI_WINDOWS_E2E"]; ok {
		t.Error("a certification variable leaked into the child")
	}
	for _, d := range []string{p.LocalAppData, p.AppData, p.UserProfile, p.Temp} {
		if !strings.HasPrefix(d, root) {
			t.Errorf("%s is outside the disposable root", d)
		}
	}
	if want := filepath.Join(p.LocalAppData, "Hachidori", "bootstrap.json"); p.LocatorPath() != want {
		t.Errorf("locator path %s, want %s", p.LocatorPath(), want)
	}
	// Without an explicit home the child has none at all.
	for _, kv := range p.Env(base) {
		if strings.HasPrefix(strings.ToUpper(kv), "HACHIDORI_") {
			t.Errorf("unexpected %s", kv)
		}
	}
}

func TestPollObservesStateAndBoundsTheWait(t *testing.T) {
	n := 0
	if err := poll(time.Second, time.Millisecond, "counter", func() (bool, error) { n++; return n >= 3, errors.New("not yet") }); err != nil {
		t.Fatal(err)
	}
	err := poll(30*time.Millisecond, time.Millisecond, "never", func() (bool, error) { return false, errors.New("still closed") })
	if err == nil || !strings.Contains(err.Error(), "never") || !strings.Contains(err.Error(), "still closed") {
		t.Fatalf("timeout error %v", err)
	}
	boom := errors.New("process exited")
	start := time.Now()
	err = poll(time.Minute, time.Millisecond, "fatal", func() (bool, error) { return false, Fatal{boom} })
	if !errors.Is(err, boom) || time.Since(start) > time.Second {
		t.Fatalf("a fatal probe must end the wait at once, got %v after %s", err, time.Since(start))
	}
}

func TestProcBoundsOutputAndReportsExit(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	p, err := Start(self, []string{"-test.run=^TestHelperProcess$"}, append(os.Environ(), "DESKTOPKIT_HELPER=chatty"), "")
	if err != nil {
		t.Fatal(err)
	}
	code, err := p.WaitExit(30 * time.Second)
	if err != nil || code != 3 {
		t.Fatalf("exit %d, %v", code, err)
	}
	out := p.Output()
	if len(out) > TailBytes+64 || !strings.Contains(out, "truncated") || !strings.Contains(out, "last line") {
		t.Fatalf("output not bounded or lost its end: %d bytes", len(out))
	}
	if p.Alive() {
		t.Error("an exited process reports alive")
	}
	p.Kill() // a no-op after exit

	hang, err := Start(self, []string{"-test.run=^TestHelperProcess$"}, append(os.Environ(), "DESKTOPKIT_HELPER=hang"), "")
	if err != nil {
		t.Fatal(err)
	}
	if !hang.Alive() {
		t.Fatal("helper exited early")
	}
	if _, err := hang.WaitExit(50 * time.Millisecond); err == nil {
		t.Fatal("WaitExit must report a process that is still running")
	}
	hang.Kill()
	if hang.Alive() {
		t.Fatal("Kill left the process running")
	}
}

// TestHelperProcess is not a test: it is re-executed by TestProcBounds....
func TestHelperProcess(t *testing.T) {
	switch os.Getenv("DESKTOPKIT_HELPER") {
	case "chatty":
		line := strings.Repeat("x", 1023) + "\n"
		for i := 0; i < 200; i++ {
			_, _ = os.Stdout.WriteString(line)
		}
		_, _ = os.Stderr.WriteString("last line\n")
		os.Exit(3)
	case "hang":
		time.Sleep(time.Minute)
	}
}

func TestFreeAddrsAreDistinctAndFree(t *testing.T) {
	addrs, err := FreeAddrs(6)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, a := range addrs {
		if seen[a] {
			t.Errorf("duplicate address %s", a)
		}
		seen[a] = true
		if Dialable(a) {
			t.Errorf("%s answers although nothing listens", a)
		}
	}
}

func TestPathShapes(t *testing.T) {
	base := filepath.Join(t.TempDir(), "b")
	names := map[string]bool{}
	for _, s := range Shapes() {
		names[s.Name] = true
		p, err := s.Path(base)
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}
		if !filepath.IsAbs(p) || filepath.Clean(p) != p || !strings.HasPrefix(p, base) {
			t.Errorf("%s: %q is not a clean absolute path beneath the base", s.Name, p)
		}
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Errorf("%s: %v", s.Name, err)
		}
		switch s.Name {
		case "spaces":
			if !strings.Contains(p, "  (x86)") || !strings.Contains(p, "My Models") {
				t.Errorf("spaces shape %q lacks spaces", p)
			}
		case "unicode":
			if len(p) == len([]rune(p)) {
				t.Errorf("unicode shape %q is ASCII only", p)
			}
		case "deep":
			if n := len(p); n > DeepPathChars || n < DeepPathChars-12 {
				t.Errorf("deep shape has %d characters, want %d-%d", n, DeepPathChars-12, DeepPathChars)
			}
		}
	}
	if len(names) != 3 {
		t.Errorf("shapes %v, want spaces, unicode and deep", names)
	}
	if _, err := DeepPath(strings.Repeat("a", 200), 150); err == nil {
		t.Error("a base longer than the budget must be refused")
	}
}

func TestResidueAndTree(t *testing.T) {
	root := t.TempDir()
	for _, d := range []string{"runtime/.staging-cpu-abc/env", "models/org/repo/rev.staging", "state", "cache/x"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(d)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, f := range []string{"packages/m.safetensors.part", "packages/m.safetensors.part.json", "state/.bootstrap.json.123.tmp", "state/.write-probe",
		"state/active-runtime.json", "cache/.hachidori-probe-9"} {
		p := filepath.Join(root, filepath.FromSlash(f))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want := []string{".staging-cpu-abc", ".hachidori-probe-9", ".bootstrap.json.123.tmp", ".write-probe", "m.safetensors.part", "m.safetensors.part.json", "rev.staging"}
	var got []string
	for _, r := range Residue(root) {
		got = append(got, filepath.Base(r))
	}
	if len(got) != len(want) {
		t.Fatalf("residue %v, want %v", got, want)
	}
	for _, w := range want {
		found := false
		for _, g := range got {
			found = found || g == w
		}
		if !found {
			t.Errorf("residue %v lacks %s", got, w)
		}
	}
	tree := Tree(root, 0)
	if !Has(tree, "state/active-runtime.json") || !Has(tree, "state/") || !HasPrefix(tree, "runtime/") {
		t.Errorf("tree %v", tree)
	}
	if got := Below(tree, "runtime/"); len(got) != 2 {
		t.Errorf("below runtime/: %v", got)
	}
	if got := Tree(root, 3); len(got) != 4 || got[3] != "..." {
		t.Errorf("a limited tree must say it was cut: %v", got)
	}
	// WebView2's own profile is not Hachidori-owned staging.
	if err := os.MkdirAll(filepath.Join(root, "cache", "webview2", "EBWebView"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "cache", "webview2", "EBWebView", ".x.tmp"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, r := range Residue(root) {
		if strings.Contains(r, "webview2") {
			t.Errorf("residue scanned the WebView2 profile: %s", r)
		}
	}
	if m := InstallMarkers(root); len(m) != 1 || m[0] != "state/active-runtime.json" {
		t.Errorf("install markers %v", m)
	}
	if len(Residue(filepath.Join(root, "missing"))) != 0 {
		t.Error("residue of a missing directory must be empty")
	}
}

func TestLocatorFixturesMatchTheProductionParser(t *testing.T) {
	dir := t.TempDir()
	homeRoot := filepath.Join(dir, "My Home ハチドリ")
	path := filepath.Join(dir, "profile", "bootstrap.json")

	data, err := WriteLocator(path, homeRoot)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckLocatorShape(data, homeRoot); err != nil {
		t.Fatal(err)
	}
	if err := Unchanged(path, data); err != nil {
		t.Fatal(err)
	}
	if CheckLocatorShape(data, homeRoot+"x") == nil || CheckLocatorShape([]byte(`{"schema":"hachidori.bootstrap/1","home":"`+strings.ReplaceAll(homeRoot, `\`, `\\`)+`","extra":1}`), homeRoot) == nil {
		t.Error("a wrong home or an extra key must fail the shape check")
	}
	// A locator naming a folder that does not exist is a valid record whose home is
	// missing, not a first run and not malformed.
	if _, _, err := (home.Locator{Path: path}).Lookup(); !errors.Is(err, home.ErrStoredHomeMissing) {
		t.Fatalf("Lookup of a missing home: %v", err)
	}
	if err := os.MkdirAll(homeRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if h, found, err := (home.Locator{Path: path}).Lookup(); err != nil || !found || h.Root != homeRoot {
		t.Fatalf("Lookup = %v %v %v", h, found, err)
	}
	if err := WriteRaw(path, append(data, 'x')); err != nil {
		t.Fatal(err)
	}
	if Unchanged(path, data) == nil {
		t.Error("a modified locator must not count as unchanged")
	}

	for _, c := range Corruptions(homeRoot) {
		p := filepath.Join(dir, c.Name, "bootstrap.json")
		if err := WriteRaw(p, c.Data); err != nil {
			t.Fatal(err)
		}
		_, found, err := (home.Locator{Path: p}).Load()
		if found || err == nil {
			t.Errorf("%s: the production parser accepted it (found=%v, err=%v)", c.Name, found, err)
			continue
		}
		if !errors.Is(err, home.ErrBootstrapMalformed) && !errors.Is(err, home.ErrBootstrapUnknownSchema) {
			t.Errorf("%s: %v is neither malformed nor an unknown schema", c.Name, err)
		}
	}
}

func TestNetstatAndProcessHelpers(t *testing.T) {
	out := `
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1012
  TCP    127.0.0.1:7843         0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:7844         0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:50000        127.0.0.1:7844         ESTABLISHED     999
  TCP    [::1]:8080             [::]:0                 LISTENING       77
  TCP    garbage
`
	ls := ParseNetstat(out)
	if len(ls) != 4 {
		t.Fatalf("listeners %+v", ls)
	}
	if got := OnPort(ls, 7844); len(got) != 1 || got[0].PID != 4242 || got[0].Host != "127.0.0.1" {
		t.Errorf("port 7844: %+v", got)
	}
	if got := OnPort(ls, 8080); len(got) != 1 || got[0].Host != "::1" {
		t.Errorf("port 8080: %+v", got)
	}
	all := []ProcEntry{{1, 0, "System"}, {10, 1, "hachidori.exe"}, {11, 10, "python.exe"}, {12, 10, "msedgewebview2.exe"}, {13, 99, "Python.exe"}, {14, 10, "PYTHON.EXE"}}
	if got := PIDs(ChildrenNamed(all, 10, "python.exe")); !reflect.DeepEqual(got, []int{11, 14}) {
		t.Errorf("children %v", got)
	}
	if got := PIDs(Named(all, "python.exe")); !reflect.DeepEqual(got, []int{11, 13, 14}) {
		t.Errorf("named %v", got)
	}
}

func TestExtractToken(t *testing.T) {
	page := `<form method="post" action="/runtime/start"><input type="hidden" name="token" value="0123abcdef"><button>`
	if tok, ok := ExtractToken(page); !ok || tok != "0123abcdef" {
		t.Errorf("token %q %v", tok, ok)
	}
	if _, ok := ExtractToken("<html>no form</html>"); ok {
		t.Error("a page without a form has no token")
	}
}

func TestScenarioCoverageDetectsDrift(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a_test.go", "s := e2e.Begin(t, \"one-scenario\")\ne2e.Begin(\n\tt, \"two-scenario\")\ne2e.VerifyCandidateScenario(t)\n")
	write("required.json", `{"scenarios": ["candidate-identity", "one-scenario", "two-scenario"]}`)
	got, err := BeginIDs(dir)
	if err != nil || !reflect.DeepEqual(got, []string{"candidate-identity", "one-scenario", "two-scenario"}) {
		t.Fatalf("BeginIDs = %v, %v", got, err)
	}
	if p, err := CoverageProblem(dir); err != nil || p != "" {
		t.Fatalf("in-step contract reported %q, %v", p, err)
	}

	write("required.json", `{"scenarios": ["candidate-identity", "one-scenario"]}`)
	if p, err := CoverageProblem(dir); err != nil || !strings.Contains(p, "two-scenario") {
		t.Fatalf("a scenario that is begun but not required must be reported, got %q, %v", p, err)
	}
	write("required.json", `{"scenarios": ["candidate-identity", "one-scenario", "two-scenario", "ghost"]}`)
	if p, err := CoverageProblem(dir); err != nil || !strings.Contains(p, "ghost") {
		t.Fatalf("a required scenario that no test begins must be reported, got %q, %v", p, err)
	}
}

func TestLauncherConfinesAndObservesAnInstance(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	prof, err := NewProfile(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("HACHIDORI_HOME", "/real/home")
	l := Launcher{Exe: self, Profile: prof, Extra: []string{"DESKTOPKIT_FAKE_EXE=stay"}}
	i, err := l.Desktop()
	if err != nil {
		t.Fatal(err)
	}
	defer i.Kill()
	if err := i.WaitShell(30 * time.Second); err != nil {
		t.Fatal(err)
	}
	out := i.Output()
	for _, want := range []string{"args=desktop --listen " + i.API + " --addr " + i.Dash, "LOCALAPPDATA=" + prof.LocalAppData, `HACHIDORI_HOME=""`} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	if i.StartMode() != "recovery_invalid" || i.API == i.Dash || len(i.Endpoints()) != 2 {
		t.Errorf("mode %q endpoints %v", i.StartMode(), i.Endpoints())
	}
	i.Kill()
	if i.Alive() {
		t.Error("Kill left the instance running")
	}

	d, err := Launcher{Exe: self, Profile: prof, Extra: []string{"DESKTOPKIT_FAKE_EXE=die"}}.Desktop()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Kill()
	_, err = d.WaitWizard(30*time.Second, "a state", func(firstrun.View) bool { return true })
	if err == nil || !strings.Contains(err.Error(), "exited with code 1") || !strings.Contains(err.Error(), "the fake desktop failed") {
		t.Fatalf("a launch that exits must fail the wait at once with its output, got %v", err)
	}
	if err := d.WaitShell(5 * time.Second); err == nil {
		t.Error("WaitShell must fail for a process that exited")
	}

	if n, err := l.NoArg(); err != nil {
		t.Fatal(err)
	} else {
		defer n.Kill()
		if n.API != server.DefaultListen || n.Dash != dashboard.DefaultListen {
			t.Errorf("no-argument endpoints %v", n.Endpoints())
		}
		if err := n.WaitShell(30 * time.Second); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(n.Output(), "args=\n") {
			t.Errorf("the no-argument launch must pass no arguments:\n%s", n.Output())
		}
	}
}

func TestWebView2Present(t *testing.T) {
	for pv, want := range map[string]bool{"": false, "0.0.0.0": false, " 0.0.0.0\n": false, "126.0.2592.87": true, " 140.0.1.2 ": true} {
		if got := webView2Present(pv); got != want {
			t.Errorf("webView2Present(%q) = %v, want %v", pv, got, want)
		}
	}
}
