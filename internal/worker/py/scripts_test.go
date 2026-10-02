package py_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

// The worker's command line is half of the launch contract with Hachidori. The
// arguments of the clef provider and its variants are checked by the script's
// own argument parser, which runs before anything heavy is imported, so this
// needs no torch: refusals exit 2 with the reason, and a well-formed clef launch
// gets past parsing and speaks the protocol (hello) before failing on the
// missing manifest.
func TestWorkerArgumentContract(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	run := func(args ...string) (string, int) {
		cmd := exec.Command(python, append([]string{"hachidori_worker.py"}, args...)...)
		out, err := cmd.CombinedOutput()
		code := 0
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		}
		return string(out), code
	}
	base := []string{"--model-dir", "m", "--device", "cpu", "--manifest", "missing.json"}
	for name, tc := range map[string]struct {
		args []string
		want string
	}{
		"variant dir without manifest": {append(append([]string{}, base...), "--provider", "clef", "--variant-dir", "v"), "go together"},
		"variant manifest without dir": {append(append([]string{}, base...), "--provider", "clef", "--variant-manifest", "v.json"), "go together"},
		"variant for another provider": {append(append([]string{}, base...), "--provider", "laya", "--variant-dir", "v", "--variant-manifest", "v.json"), "only supported by the clef provider"},
		"dtype for laya":               {append(append([]string{}, base...), "--provider", "laya", "--dtype", "bfloat16"), "only supported by the opendecider and clef providers"},
		"unsupported dtype":            {append(append([]string{}, base...), "--provider", "clef", "--dtype", "float16"), "invalid choice"},
		"unknown provider":             {append(append([]string{}, base...), "--provider", "gpt"), "invalid choice"},
	} {
		out, code := run(tc.args...)
		if code != 2 || !strings.Contains(out, tc.want) {
			t.Errorf("%s: exit %d, output %q, want exit 2 mentioning %q", name, code, out, tc.want)
		}
	}
	for _, args := range [][]string{
		append(append([]string{}, base...), "--provider", "clef"),
		append(append([]string{}, base...), "--provider", "clef", "--dtype", "float32"),
		append(append([]string{}, base...), "--provider", "clef", "--variant-dir", "v", "--variant-manifest", "v.json"),
	} {
		out, code := run(args...)
		if code == 2 || !strings.Contains(out, `"event":"hello"`) {
			t.Errorf("a well-formed clef launch %v: exit %d output %q", args, code, out)
		}
	}
}
