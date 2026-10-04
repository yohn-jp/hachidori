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
		filepath.Join("..", "..", "optimize", "testdata", "tinyclef.py"),
		filepath.Join("testdata", "profile_w4.py"), filepath.Join("testdata", "w4_linear.py"), filepath.Join("testdata", "clef_batch.py")} {
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
		"trial session for laya":       {append(append([]string{}, base...), "--provider", "laya", "--trial-session"), "only supported by the clef provider on the source model"},
		"trial session with a variant": {append(append([]string{}, base...), "--provider", "clef", "--trial-session", "--variant-dir", "v", "--variant-manifest", "v.json"), "only supported by the clef provider on the source model"},
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
		append(append([]string{}, base...), "--provider", "clef", "--trial-session"),
	} {
		out, code := run(args...)
		if code == 2 || !strings.Contains(out, `"event":"hello"`) {
			t.Errorf("a well-formed clef launch %v: exit %d output %q", args, code, out)
		}
	}
}

func TestClefBatchContract(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	cmd := exec.Command(python, "-B", filepath.Join("testdata", "clef_batch.py"), "hachidori_worker.py")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Clef batch contract: %v\n%s", err, out)
	}
}

// A missing account name is reported as itself before torch is imported, a
// one-shot import-time registration is never re-run by the worker's own import
// order, and a startup failure keeps a bounded traceback (#259).
func TestWorkerStartupDiagnostics(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	cmd := exec.Command(python, "-B", filepath.Join("testdata", "startup_diagnostics.py"), "hachidori_worker.py")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("worker startup diagnostics: %v\n%s", err, out)
	}
}

func TestClefKernelEvidence(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	cmd := exec.Command(python, "-B", filepath.Join("testdata", "clef_kernels.py"), "hachidori_worker.py")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Clef kernel dispatch evidence: %v\n%s", err, out)
	}
}

// The trial executor's transactional logic (validation, atomic replacement,
// rollback, accounting, protocol errors) runs against a fake torch adapter, so
// it needs no torch. These prove orchestration and invariants, not accelerator
// behavior.
func TestTrialExecutorContract(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	cmd := exec.Command(python, "-B", filepath.Join("testdata", "trial_executor.py"), "hachidori_worker.py")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("trial executor contract: %v\n%s", err, out)
	}
}

// With a python3 that has torch and compressed-tensors (the worker runtime's own
// dependencies) the replacement semantics are exercised on real torch modules,
// and, with llmcompressor as well, a trial component is compared bit for bit
// with what the Forge optimizer writes. Portable CI without them skips these
// scripts' bodies (they report SKIP and exit 0); set HACHIDORI_TEST_TORCH_PYTHON
// to run them against a prepared interpreter.
func TestTrialReplacementOnRealTorch(t *testing.T) {
	python := os.Getenv("HACHIDORI_TEST_TORCH_PYTHON")
	if python == "" {
		var err error
		if python, err = exec.LookPath("python3"); err != nil {
			t.Skip("no python3")
		}
	}
	for _, args := range [][]string{
		{filepath.Join("testdata", "trial_torch.py"), "hachidori_worker.py"},
		{filepath.Join("testdata", "w4_linear.py"), "hachidori_worker.py"},
		{filepath.Join("testdata", "trial_equivalence.py"), "hachidori_worker.py", filepath.Join("..", "..", "optimize", "testdata", "tinyclef.py")},
	} {
		cmd := exec.Command(python, append([]string{"-B"}, args...)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s: %v\n%s", args[0], err, out)
		}
		if strings.Contains(string(out), "SKIP:") {
			t.Logf("%s skipped: %s", args[0], strings.TrimSpace(string(out)))
		}
	}
}
