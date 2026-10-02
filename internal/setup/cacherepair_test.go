package setup_test

import (
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// pycHeader is the 16-byte header of a CPython 3.12 bytecode cache file.
var pycHeader = append([]byte{0xcb, 0x0d, 0x0d, 0x0a}, make([]byte, 12)...)

// writeFile writes rel (slash separated) under dir.
func writeFile(t *testing.T, dir, rel string, body []byte) string {
	t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, body, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func writePyc(t *testing.T, dir, rel string) string {
	return writeFile(t, dir, rel, append(slices.Clone(pycHeader), "bytecode"...))
}

func exists(p string) bool { _, err := os.Lstat(p); return err == nil }

// pollutedVariant is a built, valid variant with the exact pollution of the
// physical defect added next to its files, outside the manifest.
func pollutedVariant(t *testing.T) (home.Home, home.VariantManifest, string) {
	t.Helper()
	h, _ := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	dir := h.VariantDir(setup.ClefFlash, v.ID)
	writePyc(t, dir, "__pycache__/joint_schema_model.cpython-312.pyc")
	return h, v, dir
}

func repair(h home.Home, v home.VariantManifest) (setup.CacheRepair, error) {
	return setup.RepairArtifactCache(h, setup.KindVariant, v.ID, nil)
}

// The verifier is correct and stays strict: a bytecode file outside the
// manifest fails, and so does any other unmanifested file. Verification never
// deletes anything.
func TestStrictVerificationRejectsUnmanifestedFiles(t *testing.T) {
	h, v, dir := pollutedVariant(t)
	pyc := filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc")
	err := setup.Verify(h, setup.KindVariant, v.ID, nil)
	if err == nil || !strings.Contains(err.Error(), "holds files its manifest does not list") || !strings.Contains(err.Error(), "__pycache__/joint_schema_model.cpython-312.pyc") {
		t.Fatalf("injected pyc: %v", err)
	}
	if !exists(pyc) {
		t.Fatal("verification deleted a file")
	}
	os.Remove(pyc)
	if err := setup.Verify(h, setup.KindVariant, v.ID, nil); err != nil {
		t.Fatalf("clean variant: %v", err)
	}
	writeFile(t, dir, "notes.txt", []byte("not an artifact"))
	if err := setup.Verify(h, setup.KindVariant, v.ID, nil); err == nil || !strings.Contains(err.Error(), "notes.txt") {
		t.Fatalf("unknown unmanifested file: %v", err)
	}
	if !exists(filepath.Join(dir, "notes.txt")) {
		t.Fatal("verification deleted a file")
	}
}

func TestRepairRemovesRecognizedCacheAndVerifies(t *testing.T) {
	h, v, dir := pollutedVariant(t)
	writePyc(t, dir, "__pycache__/other.cpython-312.opt-1.pyc")
	manifestBefore, _ := os.ReadFile(filepath.Join(dir, home.VariantManifestFile))
	filesBefore := map[string]string{}
	for _, rel := range v.VariantFileNames() {
		b, _ := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
		filesBefore[rel] = string(b)
	}

	rep, err := repair(h, v)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"__pycache__/joint_schema_model.cpython-312.pyc", "__pycache__/other.cpython-312.opt-1.pyc", "__pycache__/"}
	if rep.Kind != setup.KindVariant || rep.ID != v.ID || !rep.Found || !rep.Verified || !slices.Equal(rep.Removed, want) {
		t.Fatalf("report %+v, want removed %v", rep, want)
	}
	if exists(filepath.Join(dir, "__pycache__")) {
		t.Fatal("the now-empty __pycache__ directory was left behind")
	}
	if err := setup.Verify(h, setup.KindVariant, v.ID, nil); err != nil {
		t.Fatalf("strict verification after repair: %v", err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, home.VariantManifestFile)); string(got) != string(manifestBefore) {
		t.Fatal("the manifest was rewritten")
	}
	for rel, body := range filesBefore {
		if got, _ := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel))); string(got) != body {
			t.Fatalf("%s was modified", rel)
		}
	}
	// A second repair finds nothing.
	if rep, err := repair(h, v); err != nil || rep.Found || len(rep.Removed) != 0 || !rep.Verified {
		t.Fatalf("second repair: %+v %v", rep, err)
	}
}

func TestRepairOfCleanArtifactIsANoOp(t *testing.T) {
	h, _ := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	dir := h.VariantDir(setup.ClefFlash, v.ID)
	before := dirDigest(t, dir)
	mi, _ := os.Stat(filepath.Join(dir, home.VariantManifestFile))

	rep, err := repair(h, v)
	if err != nil || rep.Found || !rep.Verified || len(rep.Removed) != 0 {
		t.Fatalf("clean repair: %+v %v", rep, err)
	}
	if after := dirDigest(t, dir); after != before {
		t.Fatal("a clean artifact was modified")
	}
	if mj, _ := os.Stat(filepath.Join(dir, home.VariantManifestFile)); !mj.ModTime().Equal(mi.ModTime()) {
		t.Fatal("the manifest was rewritten")
	}
}

// The source model is repaired by the same operation, under its catalog ID.
func TestRepairSourceModel(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	writePyc(t, dir, "__pycache__/joint_schema_model.cpython-312.pyc")
	rep, err := setup.RepairArtifactCache(h, setup.KindModel, setup.ClefFlash, nil)
	if err != nil || !rep.Found || !rep.Verified || !slices.Equal(rep.Removed, []string{"__pycache__/joint_schema_model.cpython-312.pyc", "__pycache__/"}) {
		t.Fatalf("source repair: %+v %v", rep, err)
	}
	if exists(filepath.Join(dir, "__pycache__")) {
		t.Fatal("__pycache__ left behind")
	}
	writeFile(t, dir, "extra.bin", []byte("x"))
	writePyc(t, dir, "__pycache__/joint_schema_model.cpython-312.pyc")
	if _, err := setup.RepairArtifactCache(h, setup.KindModel, setup.ClefFlash, nil); !errors.Is(err, setup.ErrCacheRepairRefused) {
		t.Fatalf("source with an unknown file: %v", err)
	}
	if !exists(filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc")) {
		t.Fatal("a refused source repair deleted a file")
	}
}

// Anything that is not recognized cache pollution refuses the whole repair and
// nothing is deleted, not even the recognized cache beside it.
func TestRepairRefusesUnrecognizedEntries(t *testing.T) {
	for name, inject := range map[string]func(t *testing.T, dir string){
		"unknown file":            func(t *testing.T, dir string) { writeFile(t, dir, "notes.txt", []byte("x")) },
		"unknown file in a cache": func(t *testing.T, dir string) { writeFile(t, dir, "__pycache__/readme.txt", []byte("x")) },
		"pyc outside a cache":     func(t *testing.T, dir string) { writePyc(t, dir, "stray.cpython-312.pyc") },
		"pyc named wrongly":       func(t *testing.T, dir string) { writePyc(t, dir, "__pycache__/model.pyc") },
		"pyc with no bytecode": func(t *testing.T, dir string) {
			writeFile(t, dir, "__pycache__/m.cpython-312.pyc", []byte("MZ not python...."))
		},
		"unknown directory":       func(t *testing.T, dir string) { writePyc(t, dir, "extra/__pycache__/m.cpython-312.pyc") },
		"nested cache directory":  func(t *testing.T, dir string) { writePyc(t, dir, "__pycache__/__pycache__/m.cpython-312.pyc") },
		"empty unknown directory": func(t *testing.T, dir string) { os.MkdirAll(filepath.Join(dir, "scratch"), 0o755) },
	} {
		t.Run(name, func(t *testing.T) {
			h, v, dir := pollutedVariant(t)
			inject(t, dir)
			before := dirDigest(t, dir)
			rep, err := repair(h, v)
			if !errors.Is(err, setup.ErrCacheRepairRefused) || rep.Verified || len(rep.Removed) != 0 {
				t.Fatalf("repair = %+v, %v; want a refusal", rep, err)
			}
			if after := dirDigest(t, dir); after != before {
				t.Fatalf("a refused repair changed the artifact:\n%s\n--\n%s", before, after)
			}
			if !exists(filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc")) {
				t.Fatal("a refused repair deleted the recognized cache")
			}
		})
	}
}

// resealWith adds a manifested file to the variant and returns the variant
// under its new identity, directory renamed to match.
func resealWith(t *testing.T, h home.Home, v home.VariantManifest, rel string, body []byte) (home.VariantManifest, string) {
	t.Helper()
	old := h.VariantDir(setup.ClefFlash, v.ID)
	nv := v
	nv.Files = maps.Clone(v.Files)
	nv.Files[rel] = mustDigest(t, body)
	nv.Seal()
	dir := h.VariantDir(setup.ClefFlash, nv.ID)
	if err := os.Rename(old, dir); err != nil {
		t.Fatal(err)
	}
	writeFile(t, dir, rel, body)
	if err := home.WriteFileAtomic(filepath.Join(dir, home.VariantManifestFile), nv.Canonical(), 0o644); err != nil {
		t.Fatal(err)
	}
	return nv, dir
}

func mustDigest(t *testing.T, b []byte) string {
	t.Helper()
	d, err := setup.FileSHA256Bytes(b)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// A manifested file is never deleted, however much it looks like cache: it is
// part of the artifact, and an unmanifested sibling is still removed.
func TestRepairNeverDeletesManifestedCacheLikeFile(t *testing.T) {
	h, _ := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	rel := "__pycache__/kept.cpython-312.pyc"
	body := append(slices.Clone(pycHeader), "manifested"...)
	nv, dir := resealWith(t, h, v, rel, body)
	if err := setup.Verify(h, setup.KindVariant, nv.ID, nil); err != nil {
		t.Fatalf("fixture variant: %v", err)
	}
	kept := filepath.Join(dir, filepath.FromSlash(rel))

	rep, err := repair(h, nv)
	if err != nil || rep.Found || !rep.Verified || !exists(kept) {
		t.Fatalf("a manifested cache-like file: %+v %v (exists %v)", rep, err, exists(kept))
	}
	writePyc(t, dir, "__pycache__/polluting.cpython-312.pyc")
	rep, err = repair(h, nv)
	if err != nil || !rep.Found || !rep.Verified || !slices.Equal(rep.Removed, []string{"__pycache__/polluting.cpython-312.pyc"}) {
		t.Fatalf("repair beside a manifested cache file: %+v %v", rep, err)
	}
	if !exists(kept) || !exists(filepath.Join(dir, "__pycache__")) {
		t.Fatal("the manifested file or its directory was deleted")
	}
	// Differing only by letter case from a manifested file is ambiguous: refused
	// (on a case-insensitive filesystem the two names are one file, so skipped).
	alias := filepath.Join(dir, "__pycache__", "KEPT.cpython-312.pyc")
	if exists(alias) {
		return
	}
	writePyc(t, dir, "__pycache__/KEPT.cpython-312.pyc")
	if _, err := repair(h, nv); !errors.Is(err, setup.ErrCacheRepairRefused) || !exists(alias) || !exists(kept) {
		t.Fatalf("a case variant of a manifested file: %v", err)
	}
}

// The artifact is named by identity. A path-shaped or foreign identity never
// reaches the filesystem, and a file outside the artifact is never touched.
func TestRepairRefusesPathEscapeIdentities(t *testing.T) {
	h, v, dir := pollutedVariant(t)
	outside := writePyc(t, h.Root, "models/escape.cpython-312.pyc")
	for _, id := range []string{"../../models", "..", ".", "../" + v.ID, v.ID + "/..", filepath.Join(dir, ".."), dir, "clef-flash--../..--000000000000"} {
		if _, err := setup.RepairArtifactCache(h, setup.KindVariant, id, nil); err == nil {
			t.Errorf("variant id %q was accepted", id)
		}
	}
	for _, id := range []string{"../../variants", "..", "", filepath.Dir(dir), h.Root} {
		if _, err := setup.RepairArtifactCache(h, setup.KindModel, id, nil); err == nil {
			t.Errorf("model id %q was accepted", id)
		}
	}
	if _, err := setup.RepairArtifactCache(h, setup.KindRuntime, "any", nil); err == nil {
		t.Error("a runtime was accepted")
	}
	if !exists(outside) || !exists(filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc")) {
		t.Fatal("a refused repair deleted a file")
	}
}

// A symbolic link is never followed and never deleted through: a cache
// directory that points outside the artifact, a cache file that does, and an
// artifact root that is itself a link are all refused.
func TestRepairRefusesSymlinkEscape(t *testing.T) {
	trySymlink := func(t *testing.T, target, link string) {
		t.Helper()
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("symbolic links are not available here: %v", err)
		}
	}
	t.Run("cache directory", func(t *testing.T) {
		h, v, dir := pollutedVariant(t)
		os.RemoveAll(filepath.Join(dir, "__pycache__"))
		out := t.TempDir()
		victim := writePyc(t, out, "victim.cpython-312.pyc")
		trySymlink(t, out, filepath.Join(dir, "__pycache__"))
		if _, err := repair(h, v); !errors.Is(err, setup.ErrCacheRepairRefused) {
			t.Fatalf("repair = %v", err)
		}
		if !exists(victim) {
			t.Fatal("a file outside the artifact was deleted")
		}
	})
	t.Run("cache file", func(t *testing.T) {
		h, v, dir := pollutedVariant(t)
		out := t.TempDir()
		victim := writePyc(t, out, "victim.cpython-312.pyc")
		os.Remove(filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc"))
		trySymlink(t, victim, filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc"))
		if _, err := repair(h, v); !errors.Is(err, setup.ErrCacheRepairRefused) {
			t.Fatalf("repair = %v", err)
		}
		if !exists(victim) {
			t.Fatal("a file outside the artifact was deleted")
		}
	})
	t.Run("artifact root", func(t *testing.T) {
		h, v, dir := pollutedVariant(t)
		elsewhere := filepath.Join(t.TempDir(), "moved")
		if err := os.Rename(dir, elsewhere); err != nil {
			t.Fatal(err)
		}
		trySymlink(t, elsewhere, dir)
		if _, err := repair(h, v); !errors.Is(err, setup.ErrCacheRepairRefused) {
			t.Fatalf("repair = %v", err)
		}
		if !exists(filepath.Join(elsewhere, "__pycache__", "joint_schema_model.cpython-312.pyc")) {
			t.Fatal("a file behind a linked root was deleted")
		}
	})
}

// Success is reported only when strict verification passes afterwards: a
// variant that is also corrupt loses its cache pollution but is not reported
// as repaired.
func TestRepairReportsSuccessOnlyAfterStrictVerification(t *testing.T) {
	h, v, dir := pollutedVariant(t)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	rep, err := repair(h, v)
	if err == nil || rep.Verified || !rep.Found || errors.Is(err, setup.ErrCacheRepairRefused) || !strings.Contains(err.Error(), "strict verification failed after cache repair") {
		t.Fatalf("repair = %+v, %v", rep, err)
	}
	if exists(filepath.Join(dir, "__pycache__")) {
		t.Fatal("the cache was not removed")
	}
	if err := setup.Verify(h, setup.KindVariant, v.ID, nil); err == nil {
		t.Fatal("a corrupt variant verified")
	}
}
