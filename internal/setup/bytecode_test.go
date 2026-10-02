package setup_test

import (
	"io"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/setup/bytecodetest"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// requireNoBytecodeLaunch holds a worker launch to the guarantee that it
// cannot write Python bytecode into an immutable artifact tree: its
// interpreter flags are Hachidori's, and a real interpreter started with
// exactly those flags and that environment imports artifact-local modules
// without changing the tree.
func requireNoBytecodeLaunch(t *testing.T, name string, cfg worker.Config) {
	t.Helper()
	flags := setup.PythonArgs()
	if len(cfg.Args) <= len(flags) || !slices.Equal(cfg.Args[:len(flags)], flags) || filepath.Ext(cfg.Args[len(flags)]) != ".py" {
		t.Fatalf("%s: launch arguments %v do not start with %v and the worker script", name, cfg.Args, flags)
	}
	if !slices.Contains(cfg.Env, "PYTHONDONTWRITEBYTECODE=1") {
		t.Errorf("%s: launch environment lacks PYTHONDONTWRITEBYTECODE=1", name)
	}
	python := bytecodetest.HostPython(t)
	dir := bytecodetest.ArtifactFixture(t)
	before := bytecodetest.TreeState(t, dir)
	bytecodetest.Import(t, python, cfg.Args[:len(flags)], cfg.Env, dir)
	bytecodetest.RequireUnchanged(t, dir, before)
}

// Every route into a private worker is built by one launch authority, so the
// no-bytecode guarantee holds for serving the source, serving a variant, the
// resident set, the persisted-variant probe and the exact source and variant
// execution sessions of the Forge.
func TestEveryWorkerLaunchDisablesBytecodeWrites(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")

	cfg, _, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	requireNoBytecodeLaunch(t, "normal source worker", cfg)

	v := buildVariant(t, h)
	certify(t, h, m, v, true, time.Now())

	cfg, _, err = server.ProbeConfig(h, "cpu", v.ID, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	requireNoBytecodeLaunch(t, "probe / exact variant execution", cfg)

	cfg, _, err = server.SourceConfig(h, "cpu", setup.ClefFlash, "float32", io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	requireNoBytecodeLaunch(t, "exact source execution", cfg)

	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	cfg, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil || rt.Variant == nil {
		t.Fatalf("variant worker: %v %+v", err, rt)
	}
	requireNoBytecodeLaunch(t, "normal variant worker", cfg)

	cfg, _, err = server.ResidentConfig(h, setup.ClefFlash, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	requireNoBytecodeLaunch(t, "resident", cfg)
}

func TestOtherResidentLaunchDisablesBytecodeWrites(t *testing.T) {
	h, _, other := setup.MaterializeFakeResidents(t, "cpu")
	cfg, _, err := server.ResidentConfig(h, other, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	requireNoBytecodeLaunch(t, "alternate resident", cfg)
}
