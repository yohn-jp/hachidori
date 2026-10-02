package setup_test

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/setup/bytecodetest"
)

func TestPythonArgsDisableBytecodeWrites(t *testing.T) {
	args := setup.PythonArgs("worker.py", "--model-dir", "m")
	if !slices.Equal(args, []string{"-I", "-B", "-X", "utf8", "worker.py", "--model-dir", "m"}) {
		t.Fatalf("args %v", args)
	}
}

// An isolated interpreter ignores PYTHONDONTWRITEBYTECODE, so the environment
// alone does not protect an artifact: this is the defect, reproduced.
func TestIsolatedModeIgnoresTheBytecodeEnvironment(t *testing.T) {
	python := bytecodetest.HostPython(t)
	dir := bytecodetest.ArtifactFixture(t)
	h := home.Home{Root: t.TempDir()}
	env := h.Env(filepath.Dir(python), true)
	if !slices.Contains(env, "PYTHONDONTWRITEBYTECODE=1") {
		t.Fatal("the environment no longer disables bytecode writes")
	}
	bytecodetest.Import(t, python, []string{"-I", "-X", "utf8"}, env, dir)
	if _, err := os.Stat(filepath.Join(dir, "__pycache__")); err != nil {
		t.Skipf("this interpreter wrote no bytecode in isolated mode (%v)", err)
	}
}

// Importing artifact-local modules through Hachidori's launch arguments leaves
// the tree exactly as it was: same entries, same bytes, no __pycache__. The
// second launch has no environment at all: the flag alone is the guarantee.
func TestPythonArgsImportLeavesArtifactTreeUnchanged(t *testing.T) {
	python := bytecodetest.HostPython(t)
	dir := bytecodetest.ArtifactFixture(t)
	before := bytecodetest.TreeState(t, dir)
	h := home.Home{Root: t.TempDir()}
	bytecodetest.Import(t, python, setup.PythonArgs(), h.Env(filepath.Dir(python), true), dir)
	bytecodetest.Import(t, python, setup.PythonArgs(), nil, dir)
	bytecodetest.RequireUnchanged(t, dir, before)
}
