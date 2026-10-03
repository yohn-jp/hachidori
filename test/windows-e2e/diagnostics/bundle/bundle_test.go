package bundle

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
	"github.com/yohn-jp/hachidori/test/windows-e2e/diagnostics/fixture"
)

// realBundle is a bundle written by the production exporter for the fixture
// home, with a failed worker whose stderr tail quotes a secret.
func realBundle(t *testing.T) (*Bundle, fixture.Canaries, fixture.Info, string) {
	t.Helper()
	root, profile := t.TempDir(), t.TempDir()
	info, err := fixture.Materialize(root)
	if err != nil {
		t.Fatal(err)
	}
	c, err := fixture.Plant(info, profile)
	if err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(t.TempDir(), "hachidori.exe")
	if err := os.WriteFile(exe, []byte("MZ executable under test"), 0o755); err != nil {
		t.Fatal(err)
	}
	st := server.Status{
		Runtime: server.Runtime{Home: root, Runtime: info.Runtime, ModelID: info.ModelID, Model: info.ModelDir, Device: "cpu"},
		Worker: worker.Snapshot{State: worker.StateFailed, Phase: "spawning", Starts: 1, Restarts: 2, QueueLimit: 64,
			LastFailure: &worker.FailureView{Class: worker.ClassStartup, Message: "cannot start " + filepath.Join(root, "x"),
				Stderr: []string{"stderr tail with " + c.Env[0]}}},
	}
	var buf bytes.Buffer
	if err := diagnostics.Write(&buf, diagnostics.Source{Status: st, Home: root, Executable: exe, Now: time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)}); err != nil {
		t.Fatal(err)
	}
	b, err := Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return b, c, info, exe
}

func forbidden(c fixture.Canaries, extra ...Forbidden) []Forbidden {
	var out []Forbidden
	for _, can := range c.List {
		out = append(out, Forbidden{can.Label, can.Needle})
	}
	return append(out, extra...)
}

func TestInspectorAcceptsTheProductionBundleAndFindsNoCanary(t *testing.T) {
	b, c, info, exe := realBundle(t)
	if p := b.Problems(); len(p) != 0 {
		t.Fatalf("the production bundle violates the contract: %v", p)
	}
	if got := b.Facts.App.Executable.Name; got != filepath.Base(exe) || len(b.Facts.App.Executable.SHA256) != 64 {
		t.Errorf("executable identity %+v", b.Facts.App.Executable)
	}
	if b.Facts.Runtime.Runtime != info.Runtime || b.Facts.Worker.State != worker.StateFailed || b.Facts.Worker.LastFailure == nil {
		t.Errorf("facts %+v", b.Facts)
	}
	if f := b.Scan(forbidden(c, Forbidden{"home path", info.Root})); len(f) != 0 {
		t.Fatalf("the production exporter leaked: %v", f)
	}
	if n := len(b.LogLines); n != MaxLogLines {
		t.Errorf("the planted log has more than %d prefixed lines, the tail must be capped at %d, got %d", MaxLogLines, MaxLogLines, n)
	}
	last := b.LogLines[len(b.LogLines)-1]
	if last != fixture.LogSeq(320) {
		t.Errorf("the tail must end at the newest line: %q", last)
	}
	for _, l := range b.LogLines {
		if l == fixture.LogSeq(100) || l == fixture.LogSeq(1) {
			t.Fatalf("a line older than the 200-line tail is present: %q", l)
		}
	}
	if !strings.Contains(string(b.Entries[LogFile]), "<HACHIDORI_HOME>") {
		t.Error("the home path in a kept log line should have been replaced by the placeholder")
	}
	if !strings.Contains(string(b.Entries[LogFile]), "[truncated]") {
		t.Error("the overlong log line should have been truncated")
	}
}

func TestScanFindsPlantedTextInEveryPlace(t *testing.T) {
	b, c, _, _ := realBundle(t)
	// Text that is in the bundle must be found, in entries and raw bytes alike.
	known := forbidden(c, Forbidden{"identity field", "hachidori.diagnostics.facts/v1"}, Forbidden{"a log line", fixture.LogSeq(320)},
		Forbidden{"backslash path", `C:\some\scratch`}, Forbidden{"empty", ""})
	got := b.Scan(known)
	var labels []string
	for _, f := range got {
		labels = append(labels, f.String())
	}
	joined := strings.Join(labels, "|")
	for _, want := range []string{"identity field found in facts.json", "a log line found in worker-log-tail.txt"} {
		if !strings.Contains(joined, want) {
			t.Errorf("scan missed %q (found %v)", want, labels)
		}
	}
	if strings.Contains(joined, "backslash path") || strings.Contains(joined, "empty") {
		t.Errorf("scan reported text that is not there: %v", labels)
	}
}

func TestScanFindsAJSONEscapedWindowsPath(t *testing.T) {
	m := mustManifest(t, map[string][]byte{FactsFile: []byte(`{"schema":"` + FactsSchema + `","note":"C:\\Users\\canary\\home"}`), LogFile: nil})
	b, err := Parse(craft(t, m))
	if err != nil {
		t.Fatal(err)
	}
	if f := b.Scan([]Forbidden{{"profile", `C:\Users\canary\home`}}); len(f) == 0 {
		t.Fatal("a JSON-escaped path was not found")
	}
}

type ent struct {
	name string
	data []byte
}

func craft(t *testing.T, entries []ent) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, e := range entries {
		name := e.name
		fw, err := zw.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Deflate})
		if err != nil {
			t.Fatal(err)
		}
		fw.Write(e.data)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// mustManifest builds a conforming set of entries around the given facts/log.
func mustManifest(t *testing.T, files map[string][]byte) []ent {
	t.Helper()
	m := Manifest{Schema: ManifestSchema, CreatedUTC: "2026-10-01T12:00:00Z", LocalOnly: true, Excluded: RequiredExclusions}
	for _, name := range []string{FactsFile, LogFile} {
		sum := sha256.Sum256(files[name])
		m.Files = append(m.Files, FileEntry{name, len(files[name]), hex.EncodeToString(sum[:])})
	}
	mj, _ := json.Marshal(m)
	return []ent{{ManifestFile, mj}, {FactsFile, files[FactsFile]}, {LogFile, files[LogFile]}}
}

func goodFacts() []byte {
	var keys []string
	for _, k := range FactsTopLevel {
		if k == "schema" {
			keys = append(keys, `"schema":"`+FactsSchema+`"`)
		} else {
			keys = append(keys, `"`+k+`":{}`)
		}
	}
	return []byte("{" + strings.Join(keys, ",") + "}")
}

func TestParseAndProblemsRejectEveryViolationOfTheContract(t *testing.T) {
	logOK := []byte(LogPrefix + "ok\n")
	good := func() []ent { return mustManifest(t, map[string][]byte{FactsFile: goodFacts(), LogFile: logOK}) }
	if b, err := Parse(craft(t, good())); err != nil || len(b.Problems()) != 0 {
		t.Fatalf("the reference bundle must conform: %v %v", err, b)
	}

	parseErr := map[string][]ent{
		"extra entry":   append(good(), ent{"models.bin", []byte("x")}),
		"missing entry": good()[:2],
		"directory":     append(good()[:2], ent{"worker-log-tail.txt/", nil}),
		"nested name":   {{"a/manifest.json", nil}, {FactsFile, nil}, {LogFile, nil}},
		"duplicate":     append(good(), ent{FactsFile, []byte("{}")}),
		"oversize":      {{ManifestFile, good()[0].data}, {FactsFile, good()[1].data}, {LogFile, bytes.Repeat([]byte("a"), MaxEntryBytes+1)}},
		"bad json":      {{ManifestFile, []byte("{")}, {FactsFile, good()[1].data}, {LogFile, nil}},
	}
	for name, es := range parseErr {
		if _, err := Parse(craft(t, es)); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
	if _, err := Parse([]byte("not a zip")); err == nil {
		t.Error("a non-zip was accepted")
	}
	if _, err := Parse(bytes.Repeat([]byte{0}, MaxZipBytes+1)); err == nil {
		t.Error("an oversize archive was accepted")
	}

	mutate := func(name string, fn func(m *Manifest, facts *[]byte, log *[]byte), wantSub string) {
		t.Helper()
		facts, log := goodFacts(), append([]byte(nil), logOK...)
		es := mustManifest(t, map[string][]byte{FactsFile: facts, LogFile: log})
		var m Manifest
		json.Unmarshal(es[0].data, &m)
		fn(&m, &facts, &log)
		es[0].data, _ = json.Marshal(m)
		es[1].data, es[2].data = facts, log
		b, err := Parse(craft(t, es))
		if err != nil {
			t.Errorf("%s: %v", name, err)
			return
		}
		if p := strings.Join(b.Problems(), "; "); !strings.Contains(p, wantSub) {
			t.Errorf("%s: problems %q do not mention %q", name, p, wantSub)
		}
	}
	mutate("wrong manifest schema", func(m *Manifest, _, _ *[]byte) { m.Schema = "x/v2" }, "manifest schema")
	mutate("bad time", func(m *Manifest, _, _ *[]byte) { m.CreatedUTC = "yesterday" }, "created_utc")
	mutate("not local only", func(m *Manifest, _, _ *[]byte) { m.LocalOnly = false }, "local-only")
	mutate("wrong digest", func(m *Manifest, _, _ *[]byte) { m.Files[0].SHA256 = strings.Repeat("0", 64) }, "SHA-256")
	mutate("wrong size", func(m *Manifest, _, _ *[]byte) { m.Files[1].Bytes++ }, "bytes")
	mutate("missing exclusion", func(m *Manifest, _, _ *[]byte) { m.Excluded = m.Excluded[1:] }, "exclusion")
	mutate("listed file absent", func(m *Manifest, _, _ *[]byte) { m.Files[0].Name = "ghost.txt" }, "ghost.txt")
	mutate("wrong facts schema", func(m *Manifest, f, _ *[]byte) { *f = bytes.Replace(*f, []byte(FactsSchema), []byte("facts/v9"), 1) }, "facts schema")
	mutate("unreviewed key", func(m *Manifest, f, _ *[]byte) {
		*f = bytes.Replace(*f, []byte(`{"schema"`), []byte(`{"env":{},"schema"`), 1)
	}, `"env"`)
	mutate("missing key", func(m *Manifest, f, _ *[]byte) { *f = bytes.Replace(*f, []byte(`,"worker":{}`), nil, 1) }, `lacks the top-level key "worker"`)
	mutate("unprefixed log line", func(m *Manifest, _, l *[]byte) { *l = []byte("Traceback: secret\n") }, "worker prefix")
	mutate("too many lines", func(m *Manifest, _, l *[]byte) {
		*l = []byte(strings.Repeat(LogPrefix+"x\n", MaxLogLines+1))
	}, "over the 200 line bound")
	mutate("overlong line", func(m *Manifest, _, l *[]byte) { *l = []byte(LogPrefix + strings.Repeat("y", 700) + "\n") }, "byte bound")
}
