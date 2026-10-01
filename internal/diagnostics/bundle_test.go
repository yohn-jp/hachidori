package diagnostics

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const (
	secretEnv   = "SENTINEL-ENV-SECRET-7f3a"
	secretSSH   = "SENTINEL-SSH-KEY-1c9d"
	secretBody  = "SENTINEL-SEMANTIC-BODY-52be"
	secretModel = "SENTINEL-MODEL-BINARY-88aa"
	secretData  = "SENTINEL-DATASET-4d10"
)

var sentinels = []string{secretEnv, secretSSH, secretBody, secretModel, secretData}

// fixture builds a HACHIDORI_HOME full of things that must never be exported.
func fixture(t *testing.T) (Source, string) {
	t.Helper()
	root := t.TempDir()
	t.Setenv("HACHIDORI_SENTINEL_TOKEN", secretEnv)
	write := func(rel, content string) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(".ssh/id_ed25519", secretSSH)
	write(".ssh/known_hosts", secretSSH)
	write("models/laya/model.safetensors", secretModel)
	write("state/history/exp-1.json", secretData)
	write("state/dashboard.json", secretBody)
	write("logs/doctor-worker.log", "[worker] "+secretBody)
	write("logs/worker.log", strings.Join([]string{
		"[worker] loading model under " + root,
		"Traceback (most recent call last):",
		"  File \"x.py\", line 1: state=" + secretBody,
		"[worker] READY",
		"",
	}, "\n"))
	exe := filepath.Join(root, "hachidori.exe")
	write("hachidori.exe", "fake executable")

	st := server.Status{
		Runtime: server.Runtime{Home: root, Runtime: "0.1.0-cu128", ModelID: "laya-base", Model: "repo/rev", Device: "cuda"},
		Worker: worker.Snapshot{
			State: worker.StateFailed, Phase: "ready", Starts: 4, Restarts: 3, Requests: 9,
			Errors:      map[string]int64{"not_ready": 1},
			Info:        worker.Info{"provider": "laya", "torch_version": "2.11.0", "device_name": "RTX", "load_ms": 12.5, "prompt": secretBody, "env": secretEnv},
			Accelerator: map[string]any{"leak": secretBody},
			LastFailure: &worker.FailureView{Class: worker.ClassCrash, Message: "exit under " + root,
				Stderr: []string{"stderr " + secretBody}},
		},
	}
	return Source{Status: st, Home: root, WebView2: "154.0.4258.37", Executable: exe,
		Now: time.Date(2026, 9, 30, 1, 2, 3, 0, time.UTC)}, root
}

func readZip(t *testing.T, b []byte) map[string][]byte {
	t.Helper()
	zr, err := zip.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	out := map[string][]byte{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(rc)
		rc.Close()
		out[f.Name] = data
	}
	return out
}

func TestBundleContainsOnlyTheAllowlistedEntries(t *testing.T) {
	src, _ := fixture(t)
	var buf bytes.Buffer
	if err := Write(&buf, src); err != nil {
		t.Fatal(err)
	}
	files := readZip(t, buf.Bytes())
	var names []string
	for n := range files {
		names = append(names, n)
	}
	sort.Strings(names)
	if got, want := strings.Join(names, ","), "facts.json,manifest.json,worker-log-tail.txt"; got != want {
		t.Fatalf("entries = %s, want %s", got, want)
	}
}

func TestBundleExcludesSecretsSemanticsAndBinaries(t *testing.T) {
	src, root := fixture(t)
	var buf bytes.Buffer
	if err := Write(&buf, src); err != nil {
		t.Fatal(err)
	}
	for name, data := range readZip(t, buf.Bytes()) {
		for _, s := range sentinels {
			if bytes.Contains(data, []byte(s)) {
				t.Errorf("%s contains excluded content %q", name, s)
			}
		}
		if bytes.Contains(data, []byte(root)) {
			t.Errorf("%s contains the HACHIDORI_HOME path", name)
		}
		for _, k := range []string{"id_ed25519", "known_hosts", "safetensors", "HACHIDORI_SENTINEL_TOKEN"} {
			// The manifest's exclusion declaration legitimately names known_hosts.
			if name != ManifestFile && bytes.Contains(data, []byte(k)) {
				t.Errorf("%s names excluded material %q", name, k)
			}
		}
	}
}

func TestBundleFactsAndManifestGolden(t *testing.T) {
	src, _ := fixture(t)
	var buf bytes.Buffer
	if err := Write(&buf, src); err != nil {
		t.Fatal(err)
	}
	files := readZip(t, buf.Bytes())

	var m Manifest
	if err := json.Unmarshal(files[ManifestFile], &m); err != nil {
		t.Fatal(err)
	}
	if m.Schema != ManifestSchema || !m.LocalOnly || m.CreatedUTC != "2026-09-30T01:02:03Z" || len(m.Excluded) == 0 {
		t.Fatalf("manifest = %+v", m)
	}
	if len(m.Files) != 2 || m.Files[0].Name != FactsFile || m.Files[1].Name != LogFile {
		t.Fatalf("manifest files = %+v", m.Files)
	}
	for _, e := range m.Files {
		if e.Bytes != len(files[e.Name]) || e.SHA256 == "" {
			t.Errorf("manifest entry %+v does not describe the archive entry", e)
		}
	}

	var f Facts
	dec := json.NewDecoder(bytes.NewReader(files[FactsFile]))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		t.Fatal(err)
	}
	want := Facts{
		Schema:   FactsSchema,
		App:      App{Version: f.App.Version, Revision: f.App.Revision, GoVersion: f.App.GoVersion, Executable: f.App.Executable},
		Runtime:  Runtime{Runtime: "0.1.0-cu128", ModelID: "laya-base", Model: "repo/rev", Device: "cuda"},
		Provider: Provider{Provider: "laya", TorchVersion: "2.11.0", DeviceName: "RTX", LoadMS: 12.5},
		WebView2: WebView2{Version: "154.0.4258.37"},
		System:   f.System,
		Worker: Worker{State: "failed", Phase: "ready", Starts: 4, RestartsInWindow: 3, Recovery: "gave_up", Requests: 9,
			ErrorCounts: map[string]int64{"not_ready": 1},
			LastFailure: &Failure{Class: worker.ClassCrash, Message: "exit under <HACHIDORI_HOME>"}},
	}
	if !reflect.DeepEqual(f, want) {
		t.Fatalf("facts = %+v\nwant   %+v", f, want)
	}
	if f.System.OS == "" || f.System.Arch == "" || f.App.GoVersion == "" {
		t.Errorf("system/app identity missing: %+v %+v", f.System, f.App)
	}
	if f.App.Executable.Name != "hachidori.exe" || len(f.App.Executable.SHA256) != 64 || f.App.Executable.Bytes != int64(len("fake executable")) {
		t.Errorf("executable identity = %+v", f.App.Executable)
	}
}

// The fact document is typed: no free-form maps of values, interfaces or raw
// byte payloads can carry unlisted data into the bundle.
func TestFactsHaveNoFreeFormFields(t *testing.T) {
	var walk func(path string, ty reflect.Type)
	walk = func(path string, ty reflect.Type) {
		switch ty.Kind() {
		case reflect.Interface:
			t.Errorf("%s is an interface", path)
		case reflect.Map:
			if ty.Elem().Kind() != reflect.Int64 || ty.Key().Kind() != reflect.String {
				t.Errorf("%s is a free-form map %s", path, ty)
			}
		case reflect.Slice:
			t.Errorf("%s is a slice", path)
		case reflect.Ptr:
			walk(path, ty.Elem())
		case reflect.Struct:
			for i := 0; i < ty.NumField(); i++ {
				walk(path+"."+ty.Field(i).Name, ty.Field(i).Type)
			}
		}
	}
	walk("Facts", reflect.TypeOf(Facts{}))
}

func TestRecoveryStatesMirrorDiagnostics(t *testing.T) {
	for _, c := range []struct {
		state    string
		restarts int
		want     string
	}{
		{worker.StateReady, 0, ""},
		{worker.StateRestarting, 1, "recovering"},
		{worker.StateFailed, 0, ""},
		{worker.StateFailed, 3, "gave_up"},
		{worker.StateStopped, 3, ""},
	} {
		f, _ := Collect(Source{Status: server.Status{Worker: worker.Snapshot{State: c.state, Restarts: c.restarts}}})
		if f.Worker.Recovery != c.want {
			t.Errorf("%s/%d recovery = %q, want %q", c.state, c.restarts, f.Worker.Recovery, c.want)
		}
	}
}

func TestLogTailIsBoundedAndPrefixFiltered(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for i := 0; i < 5000; i++ {
		fmt.Fprintf(&b, "[worker] line %d %s\n", i, strings.Repeat("x", 2000))
		fmt.Fprintf(&b, "library noise %s %d\n", secretBody, i)
	}
	if err := os.WriteFile(filepath.Join(root, "logs", "worker.log"), []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	_, tail := Collect(Source{Home: root})
	if len(tail) == 0 || len(tail) > MaxLogLines {
		t.Fatalf("tail has %d lines, want 1..%d", len(tail), MaxLogLines)
	}
	for _, l := range tail {
		if len(l) > MaxLogLineBytes+len("...[truncated]") || strings.Contains(l, secretBody) {
			t.Fatalf("unbounded or unfiltered line: %.60q", l)
		}
	}
	if !strings.Contains(tail[len(tail)-1], "line 4999") {
		t.Errorf("tail is not the end of the log: %.40q", tail[len(tail)-1])
	}
}

func TestExportWritesOneLocalFileUnderTheGivenDirectory(t *testing.T) {
	src, root := fixture(t)
	dir := filepath.Join(root, "state", "diagnostics")
	path, err := Export(src, dir)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(path) != dir || filepath.Base(path) != "hachidori-diagnostics-20260930T010203Z.zip" {
		t.Fatalf("path = %s", path)
	}
	ents, _ := os.ReadDir(dir)
	if len(ents) != 1 {
		t.Fatalf("export wrote %d files, want 1", len(ents))
	}
	b, _ := os.ReadFile(path)
	if len(readZip(t, b)) != 3 {
		t.Fatal("archive does not hold the three allowlisted entries")
	}
}

// A Windows worker failure names the home the way Python quotes an OSError
// filename (doubled backslashes). The exported facts must carry neither the
// user name nor the home path.
func TestWorkerFailureMessageDoesNotCarryTheWindowsHome(t *testing.T) {
	const homeDir, profile = `C:\Users\alice\Hachidori`, `C:\Users\alice`
	t.Setenv("HOME", profile) // os.UserHomeDir on every platform under test
	t.Setenv("USERPROFILE", profile)
	f, _ := Collect(Source{Home: homeDir, Status: server.Status{Worker: worker.Snapshot{
		State: worker.StateFailed,
		LastFailure: &worker.FailureView{Class: worker.ClassModelLoad,
			Message: `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\alice\\Hachidori\\models\\m.safetensors'`},
	}}})
	msg := f.Worker.LastFailure.Message
	if strings.Contains(msg, "alice") || !strings.Contains(msg, "<HACHIDORI_HOME>") {
		t.Fatalf("failure message keeps the user name or loses the placeholder: %q", msg)
	}
}

func TestMissingLogAndHomeStillExport(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, Source{}); err != nil {
		t.Fatal(err)
	}
	if got := readZip(t, buf.Bytes()); len(got) != 3 || len(got[LogFile]) != 0 {
		t.Fatalf("entries = %d, log %q", len(got), got[LogFile])
	}
}
