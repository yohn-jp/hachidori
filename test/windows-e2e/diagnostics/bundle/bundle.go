// Package bundle inspects a Hachidori diagnostic bundle (the archive Export
// bundle writes under state/diagnostics) the way a reviewer or an Issue
// triager would: it enforces the documented shape and bounds, reads the
// versioned documents, and scans everything for text that must never be there.
//
// It deliberately re-derives the contract (exactly three entries, the two
// schema identifiers, the log bounds, the manifest's size and digest per
// entry) instead of trusting the exporter's own types for the shape, so a
// regression in the exporter fails here.
package bundle

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"
)

// Contract constants (docs/certification.md "Diagnostic bundle").
const (
	ManifestSchema = "hachidori.diagnostics.manifest/v1"
	FactsSchema    = "hachidori.diagnostics.facts/v1"

	ManifestFile = "manifest.json"
	FactsFile    = "facts.json"
	LogFile      = "worker-log-tail.txt"

	// The log tail is at most MaxLogLines lines of MaxLogLineBytes each (plus
	// the truncation marker the redactor appends).
	MaxLogLines     = 200
	MaxLogLineBytes = 512
	truncationMark  = "...[truncated]"
	LogPrefix       = "[worker] "

	// Bounds the inspector enforces on the archive itself.
	MaxEntryBytes = 160 << 10
	MaxZipBytes   = 256 << 10
)

// Names is the complete, fixed set of entries, in archive order.
var Names = []string{ManifestFile, FactsFile, LogFile}

// FactsTopLevel is the allowlist of top-level keys of facts.json. A new key is
// a new disclosure and must be added here deliberately, after review.
var FactsTopLevel = []string{"schema", "app", "system", "runtime", "provider", "webview2", "worker"}

// Exclusions the manifest must declare.
var RequiredExclusions = []string{
	"semantic request, state and question bodies",
	"dataset and experiment contents",
	"environment variables",
	"SSH credentials, keys and known_hosts",
	"model binaries",
	"raw worker stderr, doctor output and unprefixed log lines",
	"the HACHIDORI_HOME path",
}

// Manifest is manifest.json.
type Manifest struct {
	Schema     string      `json:"schema"`
	CreatedUTC string      `json:"created_utc"`
	LocalOnly  bool        `json:"local_only"`
	Files      []FileEntry `json:"files"`
	Excluded   []string    `json:"excluded"`
}

// FileEntry is one manifest file record.
type FileEntry struct {
	Name   string `json:"name"`
	Bytes  int    `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// Facts is the part of facts.json the E2E reads. Unknown fields are ignored
// here; the top-level key allowlist is checked separately.
type Facts struct {
	Schema string `json:"schema"`
	App    struct {
		Version    string `json:"version"`
		Revision   string `json:"revision"`
		GoVersion  string `json:"go_version"`
		Executable struct {
			Name   string `json:"name"`
			SHA256 string `json:"sha256"`
			Bytes  int64  `json:"bytes"`
		} `json:"executable"`
	} `json:"app"`
	System struct {
		OS   string `json:"os"`
		Arch string `json:"arch"`
		CPUs int    `json:"cpus"`
	} `json:"system"`
	Runtime struct {
		Runtime      string `json:"runtime"`
		Directory    string `json:"runtime_directory"`
		WorkerSHA256 string `json:"worker_sha256"`
		WorkerABI    string `json:"worker_abi"`
		ModelID      string `json:"model_id"`
		Model        string `json:"model"`
		Device       string `json:"device"`
	} `json:"runtime"`
	WebView2 struct {
		Version string `json:"version"`
	} `json:"webview2"`
	Worker struct {
		State       string `json:"state"`
		Phase       string `json:"phase"`
		Ready       bool   `json:"ready"`
		Starts      int    `json:"starts"`
		Restarts    int    `json:"restarts_in_window"`
		Recovery    string `json:"recovery"`
		LastFailure *struct {
			Class   string `json:"class"`
			Message string `json:"message"`
		} `json:"last_failure"`
		Requests    int64            `json:"requests"`
		ErrorCounts map[string]int64 `json:"error_counts"`
		QueueLimit  int              `json:"queue_limit"`
	} `json:"worker"`
}

// Bundle is a parsed bundle.
type Bundle struct {
	Raw      []byte
	Names    []string          // archive order
	Entries  map[string][]byte // decompressed
	Manifest Manifest
	Facts    Facts
	// FactsKeys are the top-level keys of facts.json.
	FactsKeys []string
	// LogLines are the lines of worker-log-tail.txt.
	LogLines []string
}

// Parse reads raw as a bundle. It fails on anything that is not exactly the
// documented shape or that exceeds the bounds; it does not check the content's
// consistency (see Problems).
func Parse(raw []byte) (*Bundle, error) {
	if len(raw) > MaxZipBytes {
		return nil, fmt.Errorf("the archive is %d bytes, over the %d byte bound", len(raw), MaxZipBytes)
	}
	zr, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return nil, fmt.Errorf("not a zip archive: %w", err)
	}
	b := &Bundle{Raw: raw, Entries: map[string][]byte{}}
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || strings.ContainsAny(f.Name, `/\`) {
			return nil, fmt.Errorf("entry %q is not a plain file name", f.Name)
		}
		if _, dup := b.Entries[f.Name]; dup {
			return nil, fmt.Errorf("entry %q appears twice", f.Name)
		}
		if f.UncompressedSize64 > MaxEntryBytes {
			return nil, fmt.Errorf("entry %q declares %d bytes, over the %d byte bound", f.Name, f.UncompressedSize64, MaxEntryBytes)
		}
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("entry %q: %w", f.Name, err)
		}
		data, err := io.ReadAll(io.LimitReader(rc, MaxEntryBytes+1))
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("entry %q: %w", f.Name, err)
		}
		if len(data) > MaxEntryBytes {
			return nil, fmt.Errorf("entry %q is over the %d byte bound", f.Name, MaxEntryBytes)
		}
		b.Names = append(b.Names, f.Name)
		b.Entries[f.Name] = data
	}
	if !slices.Equal(sorted(b.Names), sorted(Names)) {
		return nil, fmt.Errorf("the archive holds %v, want exactly %v", b.Names, Names)
	}
	if err := json.Unmarshal(b.Entries[ManifestFile], &b.Manifest); err != nil {
		return nil, fmt.Errorf("manifest.json: %w", err)
	}
	if err := json.Unmarshal(b.Entries[FactsFile], &b.Facts); err != nil {
		return nil, fmt.Errorf("facts.json: %w", err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(b.Entries[FactsFile], &top); err != nil {
		return nil, fmt.Errorf("facts.json: %w", err)
	}
	for k := range top {
		b.FactsKeys = append(b.FactsKeys, k)
	}
	slices.Sort(b.FactsKeys)
	if text := strings.TrimSuffix(string(b.Entries[LogFile]), "\n"); text != "" {
		b.LogLines = strings.Split(text, "\n")
	}
	return b, nil
}

func sorted(in []string) []string {
	out := slices.Clone(in)
	slices.Sort(out)
	return out
}

// Problems checks the parsed bundle against the contract and returns every
// violation found (an empty result means the bundle conforms).
func (b *Bundle) Problems() []string {
	var p []string
	add := func(format string, a ...any) { p = append(p, fmt.Sprintf(format, a...)) }

	m := b.Manifest
	if m.Schema != ManifestSchema {
		add("manifest schema %q, want %s", m.Schema, ManifestSchema)
	}
	if _, err := time.Parse(time.RFC3339, m.CreatedUTC); err != nil {
		add("manifest created_utc %q is not an RFC 3339 time", m.CreatedUTC)
	}
	if !m.LocalOnly {
		add("manifest does not declare the bundle local-only")
	}
	var listed []string
	for _, f := range m.Files {
		listed = append(listed, f.Name)
		data, ok := b.Entries[f.Name]
		switch {
		case !ok:
			add("manifest lists %q, which is not in the archive", f.Name)
			continue
		case f.Bytes != len(data):
			add("manifest says %s is %d bytes, it is %d", f.Name, f.Bytes, len(data))
		}
		sum := sha256.Sum256(data)
		if f.SHA256 != hex.EncodeToString(sum[:]) {
			add("manifest SHA-256 of %s does not match its content", f.Name)
		}
	}
	if !slices.Equal(sorted(listed), sorted([]string{FactsFile, LogFile})) {
		add("manifest lists %v, want facts.json and worker-log-tail.txt", listed)
	}
	for _, want := range RequiredExclusions {
		if !slices.ContainsFunc(m.Excluded, func(s string) bool { return strings.Contains(s, want) }) {
			add("manifest does not declare the exclusion %q", want)
		}
	}

	if b.Facts.Schema != FactsSchema {
		add("facts schema %q, want %s", b.Facts.Schema, FactsSchema)
	}
	for _, k := range b.FactsKeys {
		if !slices.Contains(FactsTopLevel, k) {
			add("facts.json has the unreviewed top-level key %q", k)
		}
	}
	for _, k := range FactsTopLevel {
		if !slices.Contains(b.FactsKeys, k) {
			add("facts.json lacks the top-level key %q", k)
		}
	}

	if len(b.LogLines) > MaxLogLines {
		add("the log tail has %d lines, over the %d line bound", len(b.LogLines), MaxLogLines)
	}
	for i, l := range b.LogLines {
		switch {
		case !strings.HasPrefix(l, LogPrefix):
			add("log line %d does not carry Hachidori's own worker prefix", i+1)
		case len(l) > MaxLogLineBytes+len(truncationMark):
			add("log line %d is %d bytes, over the %d byte bound", i+1, len(l), MaxLogLineBytes)
		}
	}
	return p
}

// Finding is one forbidden text found in a bundle.
type Finding struct {
	Label string
	Where string // an entry name, or "archive bytes"
}

func (f Finding) String() string { return f.Label + " found in " + f.Where }

// Forbidden is text that must not appear anywhere in the bundle.
type Forbidden struct {
	Label  string
	Needle string
}

// Scan searches every entry's decompressed content and the raw archive bytes
// for each forbidden text. Matching is exact and case-sensitive, and also
// tries the JSON-escaped spelling (backslashes doubled) so a Windows path in
// facts.json cannot slip through.
func (b *Bundle) Scan(forbidden []Forbidden) []Finding {
	var out []Finding
	for _, f := range forbidden {
		if f.Needle == "" {
			continue
		}
		spellings := []string{f.Needle}
		if esc := strings.ReplaceAll(f.Needle, `\`, `\\`); esc != f.Needle {
			spellings = append(spellings, esc)
		}
		if sl := strings.ReplaceAll(f.Needle, `\`, `/`); sl != f.Needle {
			spellings = append(spellings, sl)
		}
		for _, s := range spellings {
			for _, name := range b.Names {
				if bytes.Contains(b.Entries[name], []byte(s)) {
					out = append(out, Finding{f.Label, name})
				}
			}
			if bytes.Contains(b.Raw, []byte(s)) {
				out = append(out, Finding{f.Label, "archive bytes"})
			}
		}
	}
	return out
}
