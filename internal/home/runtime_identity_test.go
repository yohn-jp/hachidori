package home

import (
	"encoding/json"
	"strings"
	"testing"
)

func legacySpec(worker string) LegacyRuntimeSpec {
	s := testSpec()
	return LegacyRuntimeSpec{Schema: SchemaV1, Platform: s.Platform, Python: s.Python, Provider: s.Provider, Torch: s.Torch,
		Flavor: s.Flavor, UV: s.UV, UVSHA256: s.UVSHA256, Project: s.Project, Lock: s.Lock, Worker: worker}
}

// The serving identity encodes no worker digest: it cannot move with the worker.
func TestRuntimeSpecEncodesNoWorkerSource(t *testing.T) {
	b, _ := json.Marshal(testSpec())
	if strings.Contains(string(b), "worker_sha256") || !strings.Contains(string(b), `"worker_abi"`) {
		t.Fatalf("spec encoding %s", b)
	}
}

// Two schema-1 runtimes that differ only in the worker digest have different
// legacy identities but are the same dependency environment, whose identity is
// the one a current build derives.
func TestLegacyRuntimesOfDifferentWorkersAreOneEnvironment(t *testing.T) {
	a, b := legacySpec("worker-a"), legacySpec("worker-b")
	if a.ID() == b.ID() {
		t.Fatal("legacy identities should differ by worker digest")
	}
	if a.Environment().ID() != b.Environment().ID() {
		t.Fatal("the environment of a legacy runtime depends on its worker")
	}
	want := testSpec()
	want.Schema = SpecSchema
	if a.Environment() != want || a.Environment().ID() != want.ID() {
		t.Fatalf("legacy environment %+v, want %+v", a.Environment(), want)
	}
	other := legacySpec("worker-a")
	other.Lock = "another-lock"
	if other.Environment().ID() == want.ID() {
		t.Fatal("a different lock is the same environment")
	}
}

func decodeManifest(t *testing.T, v any) RuntimeManifest {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var m RuntimeManifest
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// A manifest of either spec schema is read; the legacy one keeps its own spec
// for the identity check and declares the environment it materially is.
func TestManifestOfBothSchemas(t *testing.T) {
	l := legacySpec("worker-a")
	m := decodeManifest(t, map[string]any{"identity": l.ID(), "spec": l, "python_version": "3.12.11",
		"worker": map[string]string{"worker/hachidori_worker.py": "worker-a"}})
	if m.Legacy == nil || *m.Legacy != l {
		t.Fatalf("legacy spec %+v", m.Legacy)
	}
	if err := m.CheckIdentity(l.ID()); err != nil {
		t.Fatal(err)
	}
	want := testSpec()
	want.Schema = SpecSchema
	if !m.Satisfies(want) || m.EnvironmentID() != want.ID() {
		t.Fatalf("environment %s, want %s", m.EnvironmentID(), want.ID())
	}
	// A manifest whose identity is not its own legacy spec's is corrupt.
	bad := decodeManifest(t, map[string]any{"identity": "cu128-0000000000000000", "spec": l})
	if err := bad.CheckIdentity("cu128-0000000000000000"); err == nil {
		t.Fatal("a legacy manifest with a foreign identity was accepted")
	}
	tampered := l
	tampered.Lock = "tampered"
	if err := decodeManifest(t, map[string]any{"identity": l.ID(), "spec": tampered}).CheckIdentity(l.ID()); err == nil {
		t.Fatal("a legacy manifest whose spec was edited was accepted")
	}

	cur := testSpec()
	cur.Schema = SpecSchema
	cm := decodeManifest(t, RuntimeManifest{Identity: cur.ID(), Spec: cur})
	if cm.Legacy != nil || cm.CheckIdentity(cur.ID()) != nil || cm.EnvironmentID() != cur.ID() || !cm.Satisfies(cur) {
		t.Fatalf("current manifest %+v", cm)
	}
	// Legacy specs are never written back.
	out, _ := json.Marshal(m)
	if strings.Contains(string(out), "worker_sha256") {
		t.Fatalf("a legacy spec is rewritten: %s", out)
	}
}
