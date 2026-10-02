package py_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The private scripts are Python run only on a materialized runtime, which
// portable tests do not have; when any python3 is present they are at least
// parsed, so a syntax error cannot ship.
func TestPrivateScriptsParse(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3 to parse the private scripts")
	}
	for _, p := range []string{"hachidori_worker.py", filepath.Join("..", "..", "optimize", "py", "hachidori_optimizer.py"),
		filepath.Join("..", "..", "optimize", "testdata", "tinyclef.py")} {
		if _, err := os.Stat(p); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(python, "-c", "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])", p)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Errorf("%s does not parse: %v\n%s", p, err, out)
		}
	}
}
