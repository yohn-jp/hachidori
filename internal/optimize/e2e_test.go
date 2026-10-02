package optimize_test

import (
	"context"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// The end-to-end test runs the real optimizer (LLM Compressor) and the real
// clef provider adapter on a tiny, randomly initialized model laid out like
// the pinned Clef-Flash release. It needs a prepared optimizer environment and
// the upstream joint_schema_model.py, so it is skipped unless both are named:
//
//	HACHIDORI_TEST_OPTIMIZER_ENV=<uv project environment with the optimizer lock>
//	HACHIDORI_TEST_JOINT_SCHEMA=<path to the release's joint_schema_model.py>
//
// Normal CI sets neither: it downloads nothing and runs no optimizer.
func TestTinyClefEndToEnd(t *testing.T) {
	envDir, joint := os.Getenv("HACHIDORI_TEST_OPTIMIZER_ENV"), os.Getenv("HACHIDORI_TEST_JOINT_SCHEMA")
	if envDir == "" || joint == "" {
		t.Skip("set HACHIDORI_TEST_OPTIMIZER_ENV and HACHIDORI_TEST_JOINT_SCHEMA to run the real optimizer on a tiny model")
	}
	python := filepath.Join(envDir, "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Skipf("no interpreter at %s", python)
	}

	// 1. A tiny source model, materialized under the catalog layout.
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	gen := exec.Command(python, "testdata/tinyclef.py", "--out", filepath.Join(tmp, "src"), "--joint-schema", joint)
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("tiny model: %v\n%s", err, out)
	}
	var m home.ModelManifest
	if err := home.ReadJSON(filepath.Join(tmp, "src", "hachidori-model.json"), &m); err != nil {
		t.Fatal(err)
	}
	srcDir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if err := os.MkdirAll(filepath.Dir(srcDir), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Join(tmp, "src"), srcDir); err != nil {
		t.Fatal(err)
	}
	old := setup.Models
	setup.Models = []home.ModelManifest{m}
	t.Cleanup(func() { setup.Models = old })

	// 2. The real optimizer, offline, through the real builder.
	script, _ := filepath.Abs("py/hachidori_optimizer.py")
	workerScript, _ := filepath.Abs("../worker/py/hachidori_worker.py")
	runner := optimize.NewProcessRunner(python, script, h.Env(filepath.Dir(python), true), h.Path("state"))
	req := optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16}
	build := func() optimize.Result {
		res, err := optimize.Build(context.Background(), h, req, optimize.Deps{Runner: runner, OptimizerRuntime: "optimizer-e2e"}, io.Discard, nil)
		if err != nil {
			t.Fatalf("real optimizer build: %v", err)
		}
		return res
	}
	res := build()
	v := res.Variant
	if v.Optimizer.Version != setup.OptimizerEngineVersion || v.Optimizer.Versions["compressed-tensors"] != "0.19.0" {
		t.Fatalf("engine facts %+v", v.Optimizer)
	}
	if err := setup.VerifyVariantArtifacts(h, m, v, nil); err != nil {
		t.Fatalf("the built variant does not verify: %v", err)
	}
	// Reproducible on the same machine: same contract, same bytes, same identity.
	req.Reproduce = true
	again := build()
	if !again.Reproduced || again.Variant.ID != v.ID {
		t.Fatalf("the real optimizer is not reproducible: %+v", again)
	}
	req.Reproduce = false

	// 3. The provider adapter: source on the CPU, then the variant, same inputs.
	items := []worker.Item{{State: "The invoice is overdue and the vendor called twice.", Questions: []api.Question{
		{ID: "status", Type: "choice", Instructions: "What is the invoice status?", Choices: []string{"paid", "overdue", "draft"},
			Descriptions: map[string]string{"paid": "Invoice is paid."}},
		{ID: "urgent", Type: "choice", Instructions: "Is it urgent?", Choices: []string{"yes", "no"}}}}}
	start := func(extra ...string) (*worker.Process, error) {
		args := setup.PythonArgs(append([]string{workerScript, "--model-dir", srcDir, "--device", "cpu",
			"--manifest", filepath.Join(srcDir, "hachidori-model.json"), "--provider", "clef"}, extra...)...)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		t.Cleanup(cancel)
		return worker.Start(ctx, worker.Config{Python: python, Args: args, Env: h.Env(filepath.Dir(python), true), Dir: h.Path("state"),
			Log: io.Discard, StartTimeout: 5 * time.Minute, RequestTimeout: time.Minute}, nil)
	}
	vdir := h.VariantDir(setup.ClefFlash, v.ID)
	variantArgs := []string{"--variant-dir", vdir, "--variant-manifest", filepath.Join(vdir, home.VariantManifestFile)}

	src, err := start()
	if err != nil {
		t.Fatalf("source worker: %v", err)
	}
	defer src.Close()
	if src.Info["variant_id"] != nil || src.Info["device"] != "cpu" || src.Info["dtype"] != "torch.bfloat16" || src.Info["provider"] != "clef" {
		t.Fatalf("source info %v", src.Info)
	}
	want, _, err := src.Decide(items)
	if err != nil {
		t.Fatal(err)
	}

	cand, err := start(variantArgs...)
	if err != nil {
		t.Fatalf("variant worker: %v", err)
	}
	defer cand.Close()
	if cand.Info["variant_id"] != v.ID || cand.Info["device"] != "cpu" || cand.Info["dtype"] != "torch.bfloat16" || cand.Info["model_id"] != setup.ClefFlash ||
		cand.Info["quantization_scheme"] != "W4A16" || cand.Info["execution"] != "variant" || cand.Info["weights_quantized_modules"] == float64(0) {
		t.Fatalf("variant info %v", cand.Info)
	}
	got, _, err := cand.Decide(items)
	if err != nil {
		t.Fatal(err)
	}
	again2, _, _ := cand.Decide(items)
	for qi, r := range got[0] {
		sum := 0.0
		for _, p := range r.Probabilities {
			if math.IsNaN(p) || p < 0 || p > 1 {
				t.Fatalf("invalid probability %v", r.Probabilities)
			}
			sum += p
		}
		if math.Abs(sum-1) > 1e-4 || r.Probabilities[r.Choice] != r.Confidence {
			t.Fatalf("result %+v", r)
		}
		if again2[0][qi].Confidence != r.Confidence {
			t.Fatal("the variant is not deterministic for identical input")
		}
		if want[0][qi].Probabilities[want[0][qi].Choice] == r.Confidence {
			t.Errorf("question %s: variant and source answered identically; the variant is not running quantized", r.ID)
		}
	}
	if st, err := cand.Stats(); err != nil || st["host_rss_bytes"] == nil {
		t.Fatalf("host RAM was not reported: %v %v", err, st)
	}

	// 4. No fallbacks. CUDA is not available here: it fails, it does not run on the CPU.
	for name, extra := range map[string][]string{"source": nil, "variant": variantArgs} {
		args := setup.PythonArgs(append([]string{workerScript, "--model-dir", srcDir, "--device", "cuda",
			"--manifest", filepath.Join(srcDir, "hachidori-model.json"), "--provider", "clef"}, extra...)...)
		_, err := worker.Start(context.Background(), worker.Config{Python: python, Args: args, Env: h.Env(filepath.Dir(python), true), Dir: h.Path("state"),
			Log: io.Discard, StartTimeout: time.Minute, RequestTimeout: time.Minute}, nil)
		var f *worker.Failure
		if err == nil {
			t.Fatalf("%s: CUDA requested without CUDA but a worker became ready", name)
		}
		if asFailure(err, &f) && f.Class != worker.ClassDevice {
			t.Fatalf("%s: failure class %q, want %q", name, f.Class, worker.ClassDevice)
		}
	}
	// The variant executes at its declared dtype only.
	if _, err := start(append([]string{"--dtype", "float32"}, variantArgs...)...); err == nil {
		t.Fatal("a variant was started at another dtype")
	}
	// The source reference can be requested at float32.
	f32, err := start("--dtype", "float32")
	if err != nil || f32.Info["dtype"] != "torch.float32" {
		t.Fatalf("float32 reference: %v", err)
	}
	f32.Close()

	// 5. A corrupt variant never becomes ready, and the source is not served in its place.
	file := filepath.Join(vdir, "model.safetensors")
	b, _ := os.ReadFile(file)
	os.WriteFile(file, append([]byte("X"), b[1:]...), 0o644)
	if p, err := start(variantArgs...); err == nil {
		p.Close()
		t.Fatal("a corrupt variant became ready")
	} else if f, ok := err.(*worker.Failure); !ok || f.Class != worker.ClassModelLoad || !strings.Contains(f.Message, "model.safetensors") {
		t.Fatalf("corrupt variant failure: %v", err)
	}
	os.WriteFile(file, b, 0o644)
}

func asFailure(err error, out **worker.Failure) bool {
	f, ok := err.(*worker.Failure)
	if ok {
		*out = f
	}
	return ok
}
