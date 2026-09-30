// Package diagnostics builds the local, non-secret diagnostic bundle an
// operator can attach to an Issue.
//
// The bundle is assembled only when the operator asks for it, is written only
// to a local file, and is never uploaded. It is allowlist-based by
// construction: every value comes from a typed field of Facts (filled from the
// existing status authority, never from a free-form map or the environment)
// plus a bounded tail of Hachidori's own worker log lines. Nothing else is read.
//
// Excluded by construction: semantic request/state/question bodies, dataset
// and experiment contents, environment variables, SSH credentials and
// known_hosts, model binaries, raw worker stderr and doctor output.
package diagnostics

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Format identifiers. The manifest and facts are versioned independently of
// the application; a reader must reject a schema it does not know.
const (
	ManifestSchema = "hachidori.diagnostics.manifest/v1"
	FactsSchema    = "hachidori.diagnostics.facts/v1"
)

// The complete, fixed set of archive entries.
const (
	ManifestFile = "manifest.json"
	FactsFile    = "facts.json"
	LogFile      = "worker-log-tail.txt"
)

// Bounds. The log tail is the last MaxLogLines lines of at most MaxLogLineBytes
// each, taken from the last logReadWindow bytes of the worker log.
const (
	MaxLogLines     = 200
	MaxLogLineBytes = 512
	maxMessageBytes = 512
	logReadWindow   = 256 << 10
	// logPrefix marks lines written by Hachidori's own worker log function.
	// Library output and tracebacks (which may quote values) are not kept.
	logPrefix = "[worker] "
)

// Excluded declares in the manifest what the bundle never contains.
var Excluded = []string{
	"semantic request, state and question bodies",
	"dataset and experiment contents",
	"environment variables",
	"SSH credentials, keys and known_hosts",
	"model binaries",
	"raw worker stderr, doctor output and unprefixed log lines",
	"the HACHIDORI_HOME path (replaced by a placeholder)",
}

// Source is everything the exporter is given. Collect reads named fields only.
type Source struct {
	Status     server.Status // the /v1/status document
	Home       string        // HACHIDORI_HOME root; only logs/worker.log is read
	WebView2   string        // installed WebView2 Runtime version, "" when not hosted by the desktop shell
	Executable string        // path of the running executable, for its identity
	Now        time.Time
}

// Facts is the typed, allowlisted fact document (facts.json).
type Facts struct {
	Schema   string   `json:"schema"`
	App      App      `json:"app"`
	System   System   `json:"system"`
	Runtime  Runtime  `json:"runtime"`
	Provider Provider `json:"provider"`
	WebView2 WebView2 `json:"webview2"`
	Worker   Worker   `json:"worker"`
}

type App struct {
	Version    string     `json:"version"`
	Revision   string     `json:"revision"`
	GoVersion  string     `json:"go_version"`
	Executable Executable `json:"executable"`
}

type Executable struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type System struct {
	OS   string `json:"os"`
	Arch string `json:"arch"`
	CPUs int    `json:"cpus"`
}

type Runtime struct {
	Runtime string `json:"runtime"`
	ModelID string `json:"model_id"`
	Model   string `json:"model"`
	Device  string `json:"device"`
}

type Provider struct {
	Provider      string  `json:"provider,omitempty"`
	LayaVersion   string  `json:"laya_version,omitempty"`
	TorchVersion  string  `json:"torch_version,omitempty"`
	TorchCUDA     string  `json:"torch_cuda,omitempty"`
	PythonVersion string  `json:"python_version,omitempty"`
	Device        string  `json:"device,omitempty"`
	DeviceName    string  `json:"device_name,omitempty"`
	LoadMS        float64 `json:"load_ms,omitempty"`
	WarmupMS      float64 `json:"warmup_ms,omitempty"`
}

type WebView2 struct {
	Version string `json:"version,omitempty"`
}

type Worker struct {
	State            string           `json:"state"`
	Phase            string           `json:"phase"`
	Ready            bool             `json:"ready"`
	Starts           int              `json:"starts"`
	RestartsInWindow int              `json:"restarts_in_window"`
	Recovery         string           `json:"recovery"` // "", "recovering" or "gave_up" (the Diagnostics rule)
	LastFailure      *Failure         `json:"last_failure,omitempty"`
	Requests         int64            `json:"requests"`
	ErrorCounts      map[string]int64 `json:"error_counts,omitempty"` // by error class only
	QueueLimit       int              `json:"queue_limit"`
}

// Failure is the last worker failure without its stderr tail.
type Failure struct {
	Class   string `json:"class"`
	Message string `json:"message"`
}

// Manifest describes the archive.
type Manifest struct {
	Schema     string      `json:"schema"`
	CreatedUTC string      `json:"created_utc"`
	LocalOnly  bool        `json:"local_only"`
	Files      []FileEntry `json:"files"`
	Excluded   []string    `json:"excluded"`
}

type FileEntry struct {
	Name   string `json:"name"`
	Bytes  int    `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// Collect gathers the allowlisted facts and the bounded log tail.
func Collect(src Source) (Facts, []string) {
	s := newScrubber(src.Home)
	st := src.Status
	f := Facts{
		Schema: FactsSchema,
		App:    App{GoVersion: runtime.Version(), Executable: identify(src.Executable)},
		System: System{OS: runtime.GOOS, Arch: runtime.GOARCH, CPUs: runtime.NumCPU()},
		Runtime: Runtime{Runtime: st.Runtime.Runtime, ModelID: st.Runtime.ModelID,
			Model: st.Runtime.Model, Device: st.Runtime.Device},
		WebView2: WebView2{Version: src.WebView2},
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		f.App.Version = bi.Main.Version
		for _, kv := range bi.Settings {
			if kv.Key == "vcs.revision" {
				f.App.Revision = kv.Value
			}
		}
	}
	info := st.Worker.Info
	str := func(k string) string { v, _ := info[k].(string); return s.line(v, maxMessageBytes) }
	num := func(k string) float64 { v, _ := info[k].(float64); return v }
	f.Provider = Provider{Provider: str("provider"), LayaVersion: str("laya_version"), TorchVersion: str("torch_version"),
		TorchCUDA: str("torch_cuda"), PythonVersion: str("python_version"), Device: str("device"),
		DeviceName: str("device_name"), LoadMS: num("load_ms"), WarmupMS: num("warmup_ms")}

	w := st.Worker
	f.Worker = Worker{State: w.State, Phase: w.Phase, Ready: w.Ready, Starts: w.Starts, RestartsInWindow: w.Restarts,
		Requests: w.Requests, ErrorCounts: w.Errors, QueueLimit: w.QueueLimit}
	switch {
	case w.State == worker.StateRestarting:
		f.Worker.Recovery = "recovering"
	case w.State == worker.StateFailed && w.Restarts > 0:
		f.Worker.Recovery = "gave_up"
	}
	if lf := w.LastFailure; lf != nil {
		f.Worker.LastFailure = &Failure{Class: s.line(lf.Class, 64), Message: s.line(lf.Message, maxMessageBytes)}
	}
	return f, tailLog(src.Home, s)
}

type entry struct {
	name string
	data []byte
}

// Write writes the bundle archive to w.
func Write(w io.Writer, src Source) error {
	now := src.Now
	if now.IsZero() {
		now = time.Now()
	}
	facts, tail := Collect(src)
	factsJSON, err := json.MarshalIndent(facts, "", "  ")
	if err != nil {
		return err
	}
	logText := strings.Join(tail, "\n")
	if logText != "" {
		logText += "\n"
	}
	files := []entry{{FactsFile, append(factsJSON, '\n')}, {LogFile, []byte(logText)}}

	m := Manifest{Schema: ManifestSchema, CreatedUTC: now.UTC().Format(time.RFC3339), LocalOnly: true, Excluded: Excluded}
	for _, f := range files {
		sum := sha256.Sum256(f.data)
		m.Files = append(m.Files, FileEntry{Name: f.name, Bytes: len(f.data), SHA256: hex.EncodeToString(sum[:])})
	}
	manifestJSON, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	zw := zip.NewWriter(w)
	for _, f := range append([]entry{{ManifestFile, append(manifestJSON, '\n')}}, files...) {
		fw, err := zw.CreateHeader(&zip.FileHeader{Name: f.name, Method: zip.Deflate, Modified: now.UTC()})
		if err != nil {
			return err
		}
		if _, err := fw.Write(f.data); err != nil {
			return err
		}
	}
	return zw.Close()
}

// Export writes one bundle into dir (created if needed) and returns its path.
// It only writes locally; nothing is sent anywhere.
func Export(src Source, dir string) (string, error) {
	if src.Now.IsZero() {
		src.Now = time.Now()
	}
	var buf bytes.Buffer
	if err := Write(&buf, src); err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "hachidori-diagnostics-"+src.Now.UTC().Format("20060102T150405Z")+".zip")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// identify reports the running executable's name, size and SHA-256.
func identify(path string) Executable {
	if path == "" {
		return Executable{}
	}
	e := Executable{Name: filepath.Base(path)}
	f, err := os.Open(path)
	if err != nil {
		return e
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return e
	}
	e.Bytes, e.SHA256 = n, hex.EncodeToString(h.Sum(nil))
	return e
}

// tailLog returns the last MaxLogLines Hachidori-prefixed lines of
// HOME/logs/worker.log, read from at most the last logReadWindow bytes.
func tailLog(home string, s scrubber) []string {
	if home == "" {
		return nil
	}
	f, err := os.Open(filepath.Join(home, "logs", "worker.log"))
	if err != nil {
		return nil
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil
	}
	off := int64(0)
	if st.Size() > logReadWindow {
		off = st.Size() - logReadWindow
	}
	buf := make([]byte, st.Size()-off)
	if _, err := f.ReadAt(buf, off); err != nil && err != io.EOF {
		return nil
	}
	var out []string
	for _, l := range strings.Split(string(buf), "\n") {
		l = strings.TrimSuffix(l, "\r")
		if strings.HasPrefix(l, logPrefix) {
			out = append(out, s.line(l, MaxLogLineBytes))
		}
	}
	if len(out) > MaxLogLines {
		out = out[len(out)-MaxLogLines:]
	}
	return out
}

// scrubber replaces local paths and bounds free text.
type scrubber struct{ r *strings.Replacer }

func newScrubber(home string) scrubber {
	var pairs []string
	add := func(p, placeholder string) {
		if p = strings.TrimRight(p, `/\`); p == "" {
			return
		}
		pairs = append(pairs, p, placeholder, filepath.ToSlash(p), placeholder)
	}
	add(home, "<HACHIDORI_HOME>")
	if uh, err := os.UserHomeDir(); err == nil {
		add(uh, "<USERPROFILE>")
	}
	return scrubber{strings.NewReplacer(pairs...)}
}

func (s scrubber) line(v string, max int) string {
	v = s.r.Replace(v)
	v = strings.Map(func(r rune) rune {
		if r < 0x20 && r != '\t' || r == 0x7f {
			return -1
		}
		return r
	}, strings.ToValidUTF8(v, "?"))
	if len(v) > max {
		v = strings.ToValidUTF8(v[:max], "") + "...[truncated]"
	}
	return v
}
