package harness

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
)

func TestStubAnswerIsTheDocumentedRule(t *testing.T) {
	choice, probs := StubAnswer("Alpha beta beta, BETA! alphabet", []string{"alpha", "beta"})
	// alpha: 1 whole word (alphabet does not count) -> weight 2; beta: 3 -> weight 4.
	if choice != "beta" || !near(probs["alpha"], 2.0/6) || !near(probs["beta"], 4.0/6) {
		t.Fatalf("answer %q %v", choice, probs)
	}
	choice, probs = StubAnswer("nothing relevant", []string{"yes", "no"})
	if choice != "yes" || !near(probs["yes"], 0.5) || !near(probs["no"], 0.5) {
		t.Fatalf("a tie takes the first choice: %q %v", choice, probs)
	}
}

func near(a, b float64) bool { d := a - b; return d < 1e-12 && d > -1e-12 }

func TestCheckAnswerRejectsACannedReply(t *testing.T) {
	choices := []string{"alpha", "beta"}
	state := "beta beta alpha"
	good := api.DecideResponse{Schema: api.SchemaV1, Results: []api.Result{{ID: "q", Type: "choice", Choice: "beta",
		Confidence: 3.0 / 5, Probabilities: map[string]float64{"alpha": 2.0 / 5, "beta": 3.0 / 5}}}}
	if err := CheckAnswer(good, "q", state, choices); err != nil {
		t.Fatal(err)
	}
	bad := good
	bad.Results = []api.Result{{ID: "q", Type: "choice", Choice: "alpha", Confidence: 0.5, Probabilities: map[string]float64{"alpha": 0.5, "beta": 0.5}}}
	if err := CheckAnswer(bad, "q", state, choices); err == nil {
		t.Fatal("a reply that is not the model's answer must be rejected")
	}
	flat := good
	flat.Results = []api.Result{{ID: "q", Type: "choice", Choice: "beta", Confidence: 0.5, Probabilities: map[string]float64{"alpha": 0.5, "beta": 0.5}}}
	if err := CheckAnswer(flat, "q", state, choices); err == nil {
		t.Fatal("probabilities that differ from the rule must be rejected")
	}
}

func TestProcessTableParsingAndWorkerOwnership(t *testing.T) {
	// A bare object (one process) and an array both parse; a null command line is empty.
	one, err := ParseProcessJSON([]byte("\xef\xbb\xbf" + `{"ProcessId":4,"ParentProcessId":0,"CommandLine":null}`))
	if err != nil || len(one) != 1 || one[0].PID != 4 || one[0].Command != "" {
		t.Fatalf("%v %+v", err, one)
	}
	if _, err := ParseProcessJSON([]byte("not json")); err == nil {
		t.Fatal("garbage must be an error")
	}
	if got, err := ParseProcessJSON(nil); err != nil || got != nil {
		t.Fatalf("empty output: %v %v", got, err)
	}
	home := filepath.Join(string(filepath.Separator), "tmp", "e2e home", "home")
	other := filepath.Join(string(filepath.Separator), "tmp", "other", "home")
	script := func(h string) string {
		return filepath.Join(h, "workers", "abc", "hachidori_worker.py") + " --model-dir " + filepath.Join(h, "models")
	}
	all := []Process{
		{PID: 10, PPID: 1, Command: "hachidori.exe desktop --home " + home},
		{PID: 20, PPID: 10, Command: "launcher " + script(home)},     // a venv launcher
		{PID: 21, PPID: 20, Command: "python " + script(home)},       // its interpreter
		{PID: 30, PPID: 10, Command: "python " + script(other)},      // another home's worker
		{PID: 40, PPID: 10, Command: "python -c probe " + home},      // not a worker
		{PID: 50, PPID: 1, Command: "python " + script(home) + " x"}, // a second owner/orphan
	}
	ws := WorkerProcesses(all, home)
	if !reflect.DeepEqual(PIDs(ws), []int{20, 21, 50}) {
		t.Fatalf("worker processes %v", PIDs(ws))
	}
	if r := Roots(ws); !reflect.DeepEqual(PIDs(r), []int{20, 50}) {
		t.Fatalf("roots %v: a launcher and its interpreter are one worker, the stray is another", PIDs(r))
	}
	if d := Descendants(all, 10); !reflect.DeepEqual(d, []int{21, 20, 30, 40}) {
		t.Fatalf("descendants %v", d)
	}
}

func TestParsePS(t *testing.T) {
	got := ParsePS([]byte("    1     0 /sbin/init splash\n 4242  1 python  -I a b\n\nbad line\n"))
	if len(got) != 2 || got[1].PID != 4242 || got[1].PPID != 1 || got[1].Command != "python  -I a b" {
		t.Fatalf("%+v", got)
	}
}

func TestProfileEnvIsADisposableAllowList(t *testing.T) {
	p, err := NewProfile(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	parent := map[string]string{"SystemRoot": `C:\Windows`, "PATH": "p", "HACHIDORI_HOME": "/real", "GITHUB_TOKEN": "secret",
		"LOCALAPPDATA": "/real/local", "USERPROFILE": "/real/user", "APPDATA": "/real/roaming", "HTTPS_PROXY": "proxy"}
	env := p.Env(func(k string) (string, bool) { v, ok := parent[k]; return v, ok }, "EXTRA=1")
	got := map[string]string{}
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		got[k] = v
	}
	for _, k := range []string{"HACHIDORI_HOME", "GITHUB_TOKEN", "HTTPS_PROXY"} {
		if _, ok := got[k]; ok {
			t.Errorf("%s leaked into the child environment", k)
		}
	}
	for k, want := range map[string]string{"LOCALAPPDATA": p.LocalAppData, "APPDATA": p.AppData, "USERPROFILE": p.UserProfile,
		"TEMP": p.Temp, "TMP": p.Temp, "SystemRoot": `C:\Windows`, "EXTRA": "1"} {
		if got[k] != want {
			t.Errorf("%s=%q, want %q", k, got[k], want)
		}
	}
	for _, v := range []string{p.LocalAppData, p.AppData, p.UserProfile, p.Temp} {
		if !strings.HasPrefix(v, p.Dir) {
			t.Errorf("%s is outside the disposable profile %s", v, p.Dir)
		}
	}
	if want := filepath.Join(p.LocalAppData, "Hachidori", "bootstrap.json"); p.Locator().Path != want {
		t.Errorf("locator %s", p.Locator().Path)
	}
}

func TestLaunchArgs(t *testing.T) {
	o := LaunchOptions{APIAddr: "127.0.0.1:1", Dash: "127.0.0.1:2"}
	if got := strings.Join(o.Args(), " "); got != "desktop --listen 127.0.0.1:1 --addr 127.0.0.1:2" {
		t.Fatalf("remembered-home launch: %s", got)
	}
	o.Home = "H"
	if got := strings.Join(o.Args(), " "); got != "desktop --listen 127.0.0.1:1 --addr 127.0.0.1:2 --home H" {
		t.Fatalf("explicit-home launch: %s", got)
	}
}

func TestRecoveryNotice(t *testing.T) {
	for page, want := range map[string]string{
		`<section data-recovery="recovering">`: "recovering",
		`<section data-recovery="gave_up">`:    "gave_up",
		`<p>healthy</p>`:                       "",
	} {
		if got := RecoveryNotice(page); got != want {
			t.Errorf("%q: %q, want %q", page, got, want)
		}
	}
}

func TestEventuallyIsBoundedAndReportsTheLastObservation(t *testing.T) {
	n := 0
	if err := Eventually(time.Second, time.Millisecond, "count", func() (bool, string, error) { n++; return n == 3, "", nil }); err != nil || n != 3 {
		t.Fatalf("n=%d err=%v", n, err)
	}
	err := Eventually(30*time.Millisecond, 5*time.Millisecond, "never", func() (bool, string, error) { return false, "state=starting", nil })
	if !errors.Is(err, ErrDeadline) || !strings.Contains(err.Error(), "state=starting") {
		t.Fatalf("deadline error %v", err)
	}
	boom := errors.New("boom")
	if err := Eventually(time.Second, time.Millisecond, "x", func() (bool, string, error) { return false, "", boom }); !errors.Is(err, boom) {
		t.Fatalf("a hard error ends the wait: %v", err)
	}
}

func TestPersistedStateDiff(t *testing.T) {
	a := PersistedState{"state/active-runtime.json": "1", "x": "2"}
	if d := a.Diff(PersistedState{"state/active-runtime.json": "1", "x": "2"}); d != "" {
		t.Fatal(d)
	}
	if d := a.Diff(PersistedState{"state/active-runtime.json": "9", "x": "2"}); !strings.Contains(d, "active-runtime.json") {
		t.Fatal(d)
	}
	if d := a.Diff(PersistedState{"x": "2"}); d == "" {
		t.Fatal("a missing file is a difference")
	}
}

// The client reads the dashboard's token from the page and posts the same form
// the page's buttons post; a missing token is refused.
func TestClientRuntimeActionCarriesTheFormToken(t *testing.T) {
	var gotOp, gotToken string
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`<form><input type="hidden" name="token" value="tok-1"></form>`))
	})
	mux.HandleFunc("POST /runtime/{op}", func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		gotOp, gotToken = r.PathValue("op"), r.PostFormValue("token")
		if gotToken != "tok-1" {
			http.Error(w, "stale", http.StatusForbidden)
			return
		}
		http.Redirect(w, r, "/", http.StatusSeeOther)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	c := NewClient("127.0.0.1:1", strings.TrimPrefix(srv.URL, "http://"))
	if err := c.RuntimeAction("restart"); err != nil || gotOp != "restart" || gotToken != "tok-1" {
		t.Fatalf("op %q token %q err %v", gotOp, gotToken, err)
	}
}

func TestFreeAddrsAreDistinctAndLoopback(t *testing.T) {
	a, err := FreeAddrs(2)
	if err != nil || len(a) != 2 || a[0] == a[1] || !strings.HasPrefix(a[0], "127.0.0.1:") {
		t.Fatalf("%v %v", a, err)
	}
	if Listening(a[0]) {
		t.Fatal("a freed address is not listening")
	}
}

func TestFixtureHomeIsAValidActiveRuntimeWithStubs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the Windows E2E builds its fixtures itself; this checks the portable construction")
	}
	py, err := HostPython()
	if err != nil {
		t.Skip(err)
	}
	for _, device := range []string{"cpu", "cuda"} {
		fix, err := NewHome(t.TempDir(), device, py)
		if err != nil {
			t.Fatalf("%s: %v", device, err)
		}
		st, err := fix.Persisted()
		if err != nil || len(st) != 3 {
			t.Fatalf("persisted %v %v", st, err)
		}
		if left, err := fix.Leftovers(); err != nil || len(left) != 0 {
			t.Fatalf("a fresh fixture home has no staging: %v %v", left, err)
		}
		if _, err := os.Stat(fix.Python); err != nil {
			t.Fatal(err)
		}
	}
}
