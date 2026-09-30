package history

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/eval"
)

type stub struct{}

func (stub) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	if r.State == "fail" {
		return api.DecideResponse{}, os.ErrDeadlineExceeded
	}
	var rs []api.Result
	for _, q := range r.Questions {
		rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: "yes", Confidence: 0.8,
			Probabilities: map[string]float64{"yes": 0.8, "no": 0.2}})
	}
	return api.DecideResponse{Schema: api.SchemaV1, Results: rs, Timing: &api.Timing{InferenceMS: 3}}, nil
}

func evidence(t *testing.T, dataset string) []byte {
	t.Helper()
	q := api.Question{ID: "x", Type: "choice", Instructions: "x?", Choices: []string{"yes", "no"}}
	cases := []eval.Case{
		{ID: "c1", State: "a", Questions: []api.Question{q}, Expected: map[string]string{"x": "yes"}},
		{ID: "c2", State: "b", Questions: []api.Question{q}, Expected: map[string]string{"x": "no"}},
		{ID: "c3", State: "fail", Questions: []api.Question{q}, Expected: map[string]string{"x": "no"}},
	}
	r := eval.Run(stub{}, cases, eval.Options{Passes: 1})
	r.Endpoint, r.Dataset, r.DatasetSHA256 = "http://127.0.0.1:1", dataset, strings.Repeat("ab", 32)
	r.Served = &eval.Served{Runtime: map[string]any{"model": "m-1"}, Provider: map[string]any{}, Digest: strings.Repeat("cd", 32)}
	r.ServedEnd, r.ServedConsistent = r.Served, true
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(b, '\n')
}

func newStore(t *testing.T) (*Store, string) {
	t.Helper()
	home := t.TempDir()
	s, err := Open(filepath.Join(home, "state", "history"))
	if err != nil {
		t.Fatal(err)
	}
	clock := time.Date(2026, 9, 30, 1, 2, 3, 0, time.UTC)
	s.now = func() time.Time { clock = clock.Add(time.Second); return clock }
	return s, home
}

func TestSaveStoresCanonicalBytesAndMetadataSeparately(t *testing.T) {
	s, _ := newStore(t)
	data := evidence(t, "/data/a.jsonl")
	sm, err := s.Save(data, "baseline", "first run\nsecond line")
	if err != nil {
		t.Fatal(err)
	}
	stored, err := os.ReadFile(filepath.Join(s.Root(), "entries", sm.ID, "evidence.json"))
	if err != nil || !bytes.Equal(stored, data) {
		t.Fatalf("stored evidence differs from saved bytes: %v", err)
	}
	if strings.Contains(string(stored), "baseline") || strings.Contains(string(stored), "first run") {
		t.Fatal("operator metadata leaked into canonical evidence")
	}
	meta, err := os.ReadFile(filepath.Join(s.Root(), "entries", sm.ID, "meta.json"))
	if err != nil || !strings.Contains(string(meta), "baseline") {
		t.Fatalf("metadata not stored separately: %v %s", err, meta)
	}
	if sm.Label != "baseline" || sm.Dataset != "/data/a.jsonl" || sm.ServedModel != "m-1" || sm.RequestErrors != 1 ||
		sm.Observations != 2 || sm.Cases != 3 || sm.ServedIdentity == "" || sm.SavedAt == "" || sm.Accuracy != 0.5 {
		t.Fatalf("summary %+v", sm)
	}
}

func TestListSurvivesRestartAndOpenReturnsStrictEvidence(t *testing.T) {
	s, _ := newStore(t)
	a, _ := s.Save(evidence(t, "/data/a.jsonl"), "a", "")
	b, err := s.Save(evidence(t, "/data/b.jsonl"), "b", "")
	if err != nil {
		t.Fatal(err)
	}
	s2, err := Open(s.Root()) // a new composition over the same root
	if err != nil {
		t.Fatal(err)
	}
	list, problems, err := s2.List()
	if err != nil || len(problems) != 0 || len(list) != 2 {
		t.Fatalf("list %v %v %v", list, problems, err)
	}
	if list[0].ID != b.ID || list[1].ID != a.ID {
		t.Fatalf("not newest first: %v", list)
	}
	rep, sum, err := s2.Open(a.ID)
	if err != nil || rep.Dataset != "/data/a.jsonl" || sum != a.EvidenceSHA256 {
		t.Fatalf("open %v %q %v", rep.Dataset, sum, err)
	}
}

func TestSaveRejectsMalformedAndIncompatibleEvidence(t *testing.T) {
	s, _ := newStore(t)
	good := evidence(t, "/d")
	var m map[string]any
	_ = json.Unmarshal(good, &m)
	withField, _ := json.Marshal(func() map[string]any { m["label"] = "x"; return m }())
	m["schema"] = "hachidori.evidence.v0"
	otherSchema, _ := json.Marshal(m)
	for name, data := range map[string][]byte{
		"empty":           nil,
		"not json":        []byte("hello"),
		"truncated":       good[:len(good)/2],
		"unknown field":   withField,
		"other schema":    otherSchema,
		"trailing object": append(append([]byte{}, good...), []byte(`{"schema":"x"}`)...),
	} {
		if _, err := s.Save(data, "", ""); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := os.Stat(filepath.Join(s.Root(), "entries")); err == nil {
		if ids, _ := s.entryIDs(); len(ids) != 0 {
			t.Fatalf("rejected saves left entries: %v", ids)
		}
	}
	if _, err := s.Save(good, strings.Repeat("x", MaxLabelBytes+1), ""); err == nil {
		t.Error("oversized label accepted")
	}
	if _, err := s.Save(good, "bad\x00label", ""); err == nil {
		t.Error("control character in label accepted")
	}
}

func TestMalformedStoredEntryIsReportedAndDoesNotBreakListing(t *testing.T) {
	s, _ := newStore(t)
	ok, _ := s.Save(evidence(t, "/d"), "ok", "")
	bad, _ := s.Save(evidence(t, "/e"), "bad", "")
	ev := filepath.Join(s.Root(), "entries", bad.ID, "evidence.json")
	if err := os.WriteFile(ev, []byte(`{"schema":"hachidori.evidence.v1","surprise":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	list, problems, err := s.Rebuild()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 || list[0].ID != ok.ID || len(problems) != 1 || problems[0].ID != bad.ID {
		t.Fatalf("list %v problems %v", list, problems)
	}
	if _, _, err := s.Open(bad.ID); err == nil {
		t.Fatal("malformed entry opened")
	}
	// A reported problem is stable across List calls and the entry is kept.
	if _, p, _ := s.List(); len(p) != 1 {
		t.Fatalf("problems %v", p)
	}
	if _, err := os.Stat(ev); err != nil {
		t.Fatal("malformed entry was deleted implicitly")
	}
}

func TestIndexIsRebuildableAndNotSourceOfTruth(t *testing.T) {
	s, _ := newStore(t)
	a, _ := s.Save(evidence(t, "/a"), "la", "na")
	if _, err := s.Save(evidence(t, "/b"), "lb", ""); err != nil {
		t.Fatal(err)
	}
	idx := filepath.Join(s.Root(), "index.json")
	want, _, _ := s.List()

	for name, content := range map[string]string{"deleted": "", "garbage": "{not json", "lying": `{"schema":"hachidori.history-index.v1","entries":[]}`} {
		if content == "" {
			os.Remove(idx)
		} else if err := os.WriteFile(idx, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		got, _, err := s.List()
		if err != nil || len(got) != len(want) {
			t.Fatalf("%s: %v %v", name, got, err)
		}
		if got[0] != want[0] || got[1] != want[1] {
			t.Fatalf("%s: rebuilt index differs", name)
		}
	}
	// Lost metadata keeps the entry, loses only the label.
	os.Remove(filepath.Join(s.Root(), "entries", a.ID, "meta.json"))
	got, _, _ := s.Rebuild()
	found := false
	for _, e := range got {
		if e.ID == a.ID {
			found = true
			if e.Label != "" || e.SavedAt == "" || e.EvidenceSHA256 != a.EvidenceSHA256 {
				t.Fatalf("entry without meta: %+v", e)
			}
		}
	}
	if !found {
		t.Fatal("entry without metadata dropped")
	}
}

func TestDeleteRemovesOnlyTheEntry(t *testing.T) {
	s, home := newStore(t)
	exported := filepath.Join(home, "export.json")
	data := evidence(t, "/a")
	if err := os.WriteFile(exported, data, 0o644); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(home, "dataset.jsonl")
	os.WriteFile(source, []byte("x"), 0o644)
	a, _ := s.Save(data, "", "")
	b, _ := s.Save(evidence(t, "/b"), "", "")
	if err := s.Delete(a.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(s.Root(), "entries", a.ID)); !os.IsNotExist(err) {
		t.Fatal("entry still present")
	}
	for _, p := range []string{exported, source, filepath.Join(s.Root(), "entries", b.ID, "evidence.json")} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s removed: %v", p, err)
		}
	}
	list, _, _ := s.List()
	if len(list) != 1 || list[0].ID != b.ID {
		t.Fatalf("list after delete: %v", list)
	}
	if err := s.Delete(a.ID); err != ErrNotFound {
		t.Fatalf("second delete: %v", err)
	}
}

func TestPathTraversalIsRefused(t *testing.T) {
	s, home := newStore(t)
	sm, _ := s.Save(evidence(t, "/a"), "", "")
	victim := filepath.Join(home, "victim")
	if err := os.MkdirAll(victim, 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(victim, "keep.txt"), []byte("keep"), 0o644)
	os.WriteFile(filepath.Join(home, "state", "dashboard.json"), []byte("{}"), 0o644)
	for _, id := range []string{
		"", ".", "..", "../..", "../../victim", "../entries/" + sm.ID, sm.ID + "/..", sm.ID + "/evidence.json",
		"entries/" + sm.ID, `..\..\victim`, filepath.Join(s.Root(), "entries", sm.ID), "/etc", "index.json",
		strings.ToUpper(sm.ID), sm.ID + " ", sm.ID + "\x00", sm.ID[:len(sm.ID)-1],
	} {
		if err := s.Delete(id); err == nil {
			t.Errorf("Delete(%q) accepted", id)
		}
		if _, _, err := s.Open(id); err == nil {
			t.Errorf("Open(%q) accepted", id)
		}
	}
	for _, p := range []string{filepath.Join(victim, "keep.txt"), filepath.Join(home, "state", "dashboard.json"),
		filepath.Join(s.Root(), "entries", sm.ID, "evidence.json"), filepath.Join(s.Root(), "index.json")} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s damaged: %v", p, err)
		}
	}
}

func TestDeleteRefusesForeignContentAndSymlinks(t *testing.T) {
	s, home := newStore(t)
	sm, _ := s.Save(evidence(t, "/a"), "", "")
	dir := filepath.Join(s.Root(), "entries", sm.ID)
	foreign := filepath.Join(dir, "notes.txt")
	os.WriteFile(foreign, []byte("mine"), 0o644)
	if err := s.Delete(sm.ID); err == nil {
		t.Fatal("deleted an entry holding a foreign file")
	}
	for _, p := range []string{foreign, filepath.Join(dir, "evidence.json"), filepath.Join(dir, "meta.json")} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s removed: %v", p, err)
		}
	}
	os.Remove(foreign)

	// An id-shaped symlink to a directory elsewhere is not an entry.
	outside := filepath.Join(home, "outside")
	os.MkdirAll(outside, 0o755)
	os.WriteFile(filepath.Join(outside, "evidence.json"), evidence(t, "/z"), 0o644)
	link := filepath.Join(s.Root(), "entries", "20260101T000000Z-aaaaaaaaaaaa")
	if err := os.Symlink(outside, link); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := s.Delete("20260101T000000Z-aaaaaaaaaaaa"); err == nil {
		t.Fatal("deleted through a symlink")
	}
	if _, _, err := s.Open("20260101T000000Z-aaaaaaaaaaaa"); err == nil {
		t.Fatal("opened through a symlink")
	}
	if _, err := os.Stat(filepath.Join(outside, "evidence.json")); err != nil {
		t.Fatal("outside file removed")
	}
}

func TestOpenRequiresAbsoluteRootAndReadsDoNotWrite(t *testing.T) {
	if _, err := Open("relative/history"); err == nil {
		t.Fatal("relative root accepted")
	}
	root := filepath.Join(t.TempDir(), "state", "history")
	s, _ := Open(root)
	if list, _, err := s.List(); err != nil || len(list) != 0 {
		t.Fatalf("empty list %v %v", list, err)
	}
	if _, err := os.Stat(root); err == nil {
		t.Fatal("listing an empty store created files")
	}
}

func TestSaveSameEvidenceTwiceInOneSecondIsIdempotent(t *testing.T) {
	s, _ := newStore(t)
	fixed := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	s.now = func() time.Time { return fixed }
	data := evidence(t, "/a")
	a, err := s.Save(data, "one", "")
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.Save(data, "two", "")
	if err != nil || a.ID != b.ID || b.Label != "one" {
		t.Fatalf("%v %+v", err, b)
	}
}
