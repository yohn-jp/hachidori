// Package bytecodetest holds the fixtures of the Python bytecode-immutability
// proofs: a stand-in artifact tree, a real interpreter launch that imports out
// of it, and an exact picture of the tree to compare before and after.
package bytecodetest

import (
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// HostPython is a Python 3 interpreter of the machine running the tests, or the
// test is skipped: the proofs run a real interpreter against a fixture tree,
// never only inspect a command line.
func HostPython(t testing.TB) string {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		p, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		if out, err := exec.Command(p, "-c", "import sys; print(sys.version_info[0])").Output(); err == nil && strings.TrimSpace(string(out)) == "3" {
			return p
		}
	}
	t.Skip("no Python 3 interpreter on this host")
	return ""
}

// ArtifactFixture is a stand-in for an immutable artifact tree holding
// importable modules, like a source model's joint_schema_model.py.
func ArtifactFixture(t testing.TB) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "artifact")
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	for rel, body := range map[string]string{
		"joint_schema_model.py": "VALUE = 7\n",
		"config.json":           "{}\n",
		"sub/helper.py":         "HELPER = 1\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, filepath.FromSlash(rel)), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TreeState lists every entry (directories too) of dir with the content of
// each file, so any added, removed or changed entry shows.
func TreeState(t testing.TB, dir string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		if d.IsDir() {
			out = append(out, "d "+filepath.ToSlash(rel))
			return nil
		}
		b, err := os.ReadFile(p)
		out = append(out, "f "+filepath.ToSlash(rel)+" "+string(b))
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// LoadScript writes a script that imports the fixture modules out of the
// directory named by its first argument and returns its path.
func LoadScript(t testing.TB) string {
	t.Helper()
	script := filepath.Join(t.TempDir(), "load.py")
	src := "import sys\nsys.path.insert(0, sys.argv[1])\nimport joint_schema_model, sub.helper\nprint(joint_schema_model.VALUE, sub.helper.HELPER)\n"
	if err := os.WriteFile(script, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return script
}

// Env is env plus the one variable Windows needs to start an interpreter.
func Env(env []string) []string {
	return append(slices.Clone(env), "SystemRoot="+os.Getenv("SystemRoot"))
}

// Import runs one real interpreter launch with the given interpreter flags and
// environment, importing the fixture modules out of dir.
func Import(t testing.TB, python string, flags, env []string, dir string) {
	t.Helper()
	cmd := exec.Command(python, append(slices.Clone(flags), LoadScript(t), dir)...)
	cmd.Env = Env(env)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v\n%s", cmd.Args, err, out)
	}
}

// RequireUnchanged fails unless dir is exactly as before (a TreeState taken
// earlier): same entries, same bytes, and in particular no __pycache__.
func RequireUnchanged(t testing.TB, dir string, before []string) {
	t.Helper()
	if after := TreeState(t, dir); !slices.Equal(before, after) {
		t.Fatalf("artifact tree changed:\nbefore %q\nafter  %q", before, after)
	}
	if _, err := os.Stat(filepath.Join(dir, "__pycache__")); err == nil {
		t.Fatal("__pycache__ was created")
	}
}
