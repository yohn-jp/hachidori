// Package history is the append-oriented store of saved experiment evidence
// under HACHIDORI_HOME.
//
// Each entry is a directory holding the canonical hachidori.evidence.v1
// document exactly as it was saved (evidence.json, never modified) and a
// separate operator metadata file (meta.json: label, note, save time). The
// index (index.json) is a rebuildable cache of per-entry summaries derived
// from the stored evidence; it is never the source of truth and is recreated
// from the entries whenever it is missing, unreadable or stale.
//
// Nothing here contacts an inference endpoint. Evidence is accepted and
// returned only through the strict hachidori.evidence.v1 decoding of
// internal/eval/explore, and deletion is bounded to one entry directory of
// this store, addressed by a validated entry id.
package history

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/eval/explore"
)

// Schemas of the store's own (non-evidence) documents.
const (
	EntrySchema = "hachidori.history-entry.v1"
	IndexSchema = "hachidori.history-index.v1"
)

// File names inside the store.
const (
	entriesDir   = "entries"
	evidenceFile = "evidence.json"
	metaFile     = "meta.json"
	indexFile    = "index.json"
)

// Bounds on operator metadata.
const (
	MaxLabelBytes = 200
	MaxNoteBytes  = 4000
)

const maxMetaBytes = 1 << 20

var idPattern = regexp.MustCompile(`^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$`)

// ErrNotFound is returned when no entry has the requested id.
var ErrNotFound = errors.New("history entry not found")

// Meta is the operator metadata of one entry. It lives beside, never inside,
// the canonical evidence.
type Meta struct {
	Schema   string `json:"schema"`
	ID       string `json:"id"`
	SavedAt  string `json:"saved_at"` // RFC 3339, UTC
	Label    string `json:"label,omitempty"`
	Note     string `json:"note,omitempty"`
	Evidence string `json:"evidence_sha256"`
}

// Summary is what listing shows: entry metadata plus the identity and core
// metrics read from the stored evidence.
type Summary struct {
	ID               string  `json:"id"`
	SavedAt          string  `json:"saved_at"`
	Label            string  `json:"label,omitempty"`
	Note             string  `json:"note,omitempty"`
	EvidenceSHA256   string  `json:"evidence_sha256"`
	Bytes            int     `json:"bytes"`
	StartedAt        string  `json:"started_at"`
	Dataset          string  `json:"dataset"`
	DatasetSHA256    string  `json:"dataset_sha256"`
	Endpoint         string  `json:"endpoint"`
	ServedModel      string  `json:"served_model,omitempty"`
	ServedIdentity   string  `json:"served_identity_sha256,omitempty"`
	ServedConsistent bool    `json:"served_consistent"`
	Cases            int     `json:"cases"`
	Observations     int     `json:"observations"`
	RequestErrors    int     `json:"request_errors"`
	Accuracy         float64 `json:"choice_accuracy"`
	MeanConfidence   float64 `json:"mean_confidence"`
	ECE              float64 `json:"ece"`
	RequestP50MS     float64 `json:"request_p50_ms"`
	RequestP95MS     float64 `json:"request_p95_ms"`
}

// Problem is an entry that is present in the store but cannot be listed
// (malformed or incompatible evidence, unexpected entry layout). It is
// reported, never repaired or deleted implicitly.
type Problem struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

type index struct {
	Schema   string    `json:"schema"`
	Entries  []Summary `json:"entries"`
	Problems []Problem `json:"problems,omitempty"`
}

// Store is a history root directory (normally HACHIDORI_HOME/state/history).
type Store struct {
	root string
	mu   sync.Mutex
	now  func() time.Time
}

// Open returns the store rooted at root. The directory is created on the
// first save; opening never writes.
func Open(root string) (*Store, error) {
	if root == "" || !filepath.IsAbs(root) {
		return nil, fmt.Errorf("history root must be an absolute path, got %q", root)
	}
	return &Store{root: filepath.Clean(root), now: time.Now}, nil
}

// Root is the history root directory.
func (s *Store) Root() string { return s.root }

func (s *Store) entries() string { return filepath.Join(s.root, entriesDir) }

// entryDir maps a validated id to its directory; anything else is refused, so
// no id can address a path outside the entries directory.
func (s *Store) entryDir(id string) (string, error) {
	if !idPattern.MatchString(id) {
		return "", fmt.Errorf("invalid history entry id %q", id)
	}
	return filepath.Join(s.entries(), id), nil
}

func validText(what, v string, max int, allowNewline bool) error {
	if len(v) > max {
		return fmt.Errorf("%s is longer than %d bytes", what, max)
	}
	if !utf8.ValidString(v) {
		return fmt.Errorf("%s is not valid UTF-8", what)
	}
	for _, r := range v {
		if r < 0x20 && !(allowNewline && (r == '\n' || r == '\r' || r == '\t')) || r == 0x7f {
			return fmt.Errorf("%s contains a control character", what)
		}
	}
	return nil
}

// Save stores data as a new entry. data must be a hachidori.evidence.v1
// document accepted by strict decoding; it is stored byte for byte. label and
// note are operator metadata kept in a separate file.
func (s *Store) Save(data []byte, label, note string) (Summary, error) {
	if err := validText("label", label, MaxLabelBytes, false); err != nil {
		return Summary{}, err
	}
	if err := validText("note", note, MaxNoteBytes, true); err != nil {
		return Summary{}, err
	}
	if len(data) > explore.MaxReportBytes {
		return Summary{}, fmt.Errorf("evidence is larger than %d bytes", explore.MaxReportBytes)
	}
	rep, err := explore.Decode(data)
	if err != nil {
		return Summary{}, err
	}
	sum := sha256.Sum256(data)
	digest := hex.EncodeToString(sum[:])
	saved := s.now().UTC()
	id := saved.Format("20060102T150405Z") + "-" + digest[:12]

	s.mu.Lock()
	defer s.mu.Unlock()
	dir, err := s.entryDir(id)
	if err != nil {
		return Summary{}, err
	}
	if err := os.MkdirAll(s.entries(), 0o755); err != nil {
		return Summary{}, err
	}
	if _, err := os.Lstat(dir); err == nil {
		// Same evidence saved within the same second: already stored.
		return s.summaryOf(id)
	}
	// Build the entry beside its final name and publish it with one rename so
	// an entry is never seen half written.
	tmp, err := os.MkdirTemp(s.entries(), ".saving-")
	if err != nil {
		return Summary{}, err
	}
	published := false
	defer func() {
		if !published {
			os.RemoveAll(tmp)
		}
	}()
	m := Meta{Schema: EntrySchema, ID: id, SavedAt: saved.Format(time.RFC3339), Label: label, Note: note, Evidence: digest}
	mb, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return Summary{}, err
	}
	if err := writeNew(filepath.Join(tmp, evidenceFile), data, 0o644); err != nil {
		return Summary{}, err
	}
	if err := writeNew(filepath.Join(tmp, metaFile), append(mb, '\n'), 0o644); err != nil {
		return Summary{}, err
	}
	if err := os.Rename(tmp, dir); err != nil {
		return Summary{}, err
	}
	published = true
	sm := summarize(m, rep, len(data))
	// The index is a cache: a failure to refresh it does not undo the save.
	_ = s.rebuildLocked()
	return sm, nil
}

func writeNew(path string, data []byte, perm os.FileMode) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func summarize(m Meta, r eval.Report, size int) Summary {
	return Summary{ID: m.ID, SavedAt: m.SavedAt, Label: m.Label, Note: m.Note, EvidenceSHA256: m.Evidence, Bytes: size,
		StartedAt: r.StartedAt, Dataset: r.Dataset, DatasetSHA256: r.DatasetSHA256, Endpoint: r.Endpoint,
		ServedModel: r.Served.Model(), ServedIdentity: servedDigest(r), ServedConsistent: r.ServedConsistent,
		Cases: r.Cases, Observations: r.Observations, RequestErrors: len(r.Errors),
		Accuracy: r.ChoiceAccuracy, MeanConfidence: r.MeanConfidence, ECE: r.ECE,
		RequestP50MS: r.RequestLatency.P50, RequestP95MS: r.RequestLatency.P95}
}

func servedDigest(r eval.Report) string {
	if r.Served == nil {
		return ""
	}
	return r.Served.Digest
}

// readEntry reads and strictly decodes one entry from disk. Symbolic links
// and non-regular files are refused.
func (s *Store) readEntry(id string) (Meta, eval.Report, []byte, error) {
	dir, err := s.entryDir(id)
	if err != nil {
		return Meta{}, eval.Report{}, nil, err
	}
	fi, err := os.Lstat(dir)
	if errors.Is(err, os.ErrNotExist) {
		return Meta{}, eval.Report{}, nil, ErrNotFound
	} else if err != nil {
		return Meta{}, eval.Report{}, nil, err
	}
	if !fi.IsDir() {
		return Meta{}, eval.Report{}, nil, fmt.Errorf("history entry %s is not a directory", id)
	}
	data, err := readRegular(filepath.Join(dir, evidenceFile), explore.MaxReportBytes)
	if err != nil {
		return Meta{}, eval.Report{}, nil, err
	}
	rep, err := explore.Decode(data)
	if err != nil {
		return Meta{}, eval.Report{}, nil, err
	}
	sum := sha256.Sum256(data)
	m := Meta{Schema: EntrySchema, ID: id, Evidence: hex.EncodeToString(sum[:])}
	// Metadata is operator data: missing or damaged metadata loses the label
	// and note, never the entry.
	if mb, err := readRegular(filepath.Join(dir, metaFile), maxMetaBytes); err == nil {
		var got Meta
		if json.Unmarshal(mb, &got) == nil && got.Schema == EntrySchema && got.ID == id &&
			validText("label", got.Label, MaxLabelBytes, false) == nil && validText("note", got.Note, MaxNoteBytes, true) == nil {
			m.SavedAt, m.Label, m.Note = got.SavedAt, got.Label, got.Note
		}
	}
	if _, err := time.Parse(time.RFC3339, m.SavedAt); err != nil {
		t, _ := time.Parse("20060102T150405Z", id[:16])
		m.SavedAt = t.UTC().Format(time.RFC3339)
	}
	return m, rep, data, nil
}

func readRegular(path string, limit int64) ([]byte, error) {
	fi, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !fi.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", filepath.Base(path))
	}
	if fi.Size() > limit {
		return nil, fmt.Errorf("%s is larger than %d bytes", filepath.Base(path), limit)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(io.LimitReader(f, limit+1))
}

func (s *Store) summaryOf(id string) (Summary, error) {
	m, rep, data, err := s.readEntry(id)
	if err != nil {
		return Summary{}, err
	}
	return summarize(m, rep, len(data)), nil
}

// Open returns the stored evidence of one entry, strictly decoded, with the
// SHA-256 of its stored bytes. It reads the store only.
func (s *Store) Open(id string) (eval.Report, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, rep, _, err := s.readEntry(id)
	if err != nil {
		return eval.Report{}, "", err
	}
	return rep, m.Evidence, nil
}

// Evidence returns the stored canonical bytes of one entry, validated.
func (s *Store) Evidence(id string) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, _, data, err := s.readEntry(id)
	return data, err
}

// List returns the entries newest first, plus the entries that could not be
// read. It uses the index when the index matches the entries on disk and
// rebuilds it from the stored evidence otherwise.
func (s *Store) List() ([]Summary, []Problem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids, err := s.entryIDs()
	if err != nil {
		return nil, nil, err
	}
	ix, ok := s.loadIndex()
	if ok && sameIDs(ix, ids) {
		return ix.Entries, ix.Problems, nil
	}
	if !ok && len(ids) == 0 {
		return nil, nil, nil // nothing stored: a read creates nothing
	}
	ix, err = s.scan(ids)
	if err != nil {
		return nil, nil, err
	}
	_ = s.writeIndex(ix)
	return ix.Entries, ix.Problems, nil
}

// Rebuild discards the index and recreates it from the stored entries.
func (s *Store) Rebuild() ([]Summary, []Problem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids, err := s.entryIDs()
	if err != nil {
		return nil, nil, err
	}
	ix, err := s.scan(ids)
	if err != nil {
		return nil, nil, err
	}
	if err := s.writeIndex(ix); err != nil {
		return nil, nil, err
	}
	return ix.Entries, ix.Problems, nil
}

func (s *Store) rebuildLocked() error {
	ids, err := s.entryIDs()
	if err != nil {
		return err
	}
	ix, err := s.scan(ids)
	if err != nil {
		return err
	}
	return s.writeIndex(ix)
}

// entryIDs lists the entry directory names that look like entries. Other
// names (in-progress saves, foreign files) are ignored and never touched.
func (s *Store) entryIDs() ([]string, error) {
	des, err := os.ReadDir(s.entries())
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var ids []string
	for _, de := range des {
		if idPattern.MatchString(de.Name()) {
			ids = append(ids, de.Name())
		}
	}
	return ids, nil
}

func (s *Store) scan(ids []string) (index, error) {
	ix := index{Schema: IndexSchema}
	for _, id := range ids {
		sm, err := s.summaryOf(id)
		if err != nil {
			ix.Problems = append(ix.Problems, Problem{ID: id, Reason: err.Error()})
			continue
		}
		ix.Entries = append(ix.Entries, sm)
	}
	sort.Slice(ix.Entries, func(i, j int) bool {
		a, b := ix.Entries[i], ix.Entries[j]
		if a.SavedAt != b.SavedAt {
			return a.SavedAt > b.SavedAt
		}
		return a.ID > b.ID
	})
	sort.Slice(ix.Problems, func(i, j int) bool { return ix.Problems[i].ID > ix.Problems[j].ID })
	return ix, nil
}

func (s *Store) loadIndex() (index, bool) {
	b, err := readRegular(filepath.Join(s.root, indexFile), 64<<20)
	if err != nil {
		return index{}, false
	}
	var ix index
	if json.Unmarshal(b, &ix) != nil || ix.Schema != IndexSchema {
		return index{}, false
	}
	return ix, true
}

func sameIDs(ix index, ids []string) bool {
	if len(ix.Entries)+len(ix.Problems) != len(ids) {
		return false
	}
	have := map[string]bool{}
	for _, e := range ix.Entries {
		have[e.ID] = true
	}
	for _, p := range ix.Problems {
		have[p.ID] = true
	}
	for _, id := range ids {
		if !have[id] {
			return false
		}
	}
	return true
}

func (s *Store) writeIndex(ix index) error {
	if err := os.MkdirAll(s.root, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(ix, "", "  ")
	if err != nil {
		return err
	}
	var r [6]byte
	if _, err := rand.Read(r[:]); err != nil {
		return err
	}
	tmp := filepath.Join(s.root, ".index-"+hex.EncodeToString(r[:])+".tmp")
	if err := writeNew(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, filepath.Join(s.root, indexFile)); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// Delete removes exactly one entry: the entry directory named by a validated
// id, its evidence and metadata files, and nothing else. An entry directory
// holding any other file, a symbolic link, or an id that does not name an
// entry of this store is refused and left untouched. Evidence that was
// exported from, or is the source of, an entry is never referenced here.
func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	dir, err := s.entryDir(id)
	if err != nil {
		return err
	}
	fi, err := os.Lstat(dir)
	if errors.Is(err, os.ErrNotExist) {
		return ErrNotFound
	} else if err != nil {
		return err
	}
	if !fi.IsDir() {
		return fmt.Errorf("history entry %s is not a directory; refusing to delete it", id)
	}
	des, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, de := range des {
		if n := de.Name(); n != evidenceFile && n != metaFile {
			return fmt.Errorf("history entry %s contains unexpected file %q; refusing to delete it", id, n)
		}
		if !de.Type().IsRegular() {
			return fmt.Errorf("history entry %s: %s is not a regular file; refusing to delete it", id, de.Name())
		}
	}
	for _, de := range des {
		p := filepath.Join(dir, de.Name())
		if err := os.Remove(p); err != nil {
			return err
		}
	}
	if err := os.Remove(dir); err != nil {
		return err
	}
	_ = s.rebuildLocked()
	return nil
}
