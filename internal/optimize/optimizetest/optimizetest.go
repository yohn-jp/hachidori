// Package optimizetest is a fake optimizer for tests: a Runner that behaves
// like the private optimizer process (same arguments, same protocol events,
// same output layout) on a tiny synthetic module graph, without Python, a
// model or a network. Production code does not import it.
package optimizetest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Modules are the Linear modules of the synthetic graph, named like the real
// Clef-Flash backbone so that the canonical recipe's preserved patterns match.
var Modules = []string{
	"lm_head",
	"model.visual.blocks.0.attn.qkv",
	"model.visual.merger.linear_fc1",
	"model.language_model.layers.0.linear_attn.in_proj_a",
	"model.language_model.layers.0.linear_attn.in_proj_b",
	"model.language_model.layers.0.linear_attn.in_proj_qkv",
	"model.language_model.layers.0.mlp.gate_proj",
	"model.language_model.layers.3.self_attn.q_proj",
}

// Runner is the fake optimizer.
type Runner struct {
	// Fail selects a failure: "fatal" emits a fatal event and exits 3;
	// "crash" writes partial output and exits 1 without a fatal event;
	// "incomplete" writes everything but exits 0 without the done event;
	// "noreport" completes without writing the optimizer report;
	// "unpreserved" completes with a saved config that ignores no module;
	// "wrong-scheme" saves an 8-bit config.
	Fail string
	// Salt changes the written model bytes, as a nondeterministic rebuild would.
	Salt string
	// Engine and Version override what the optimizer reports.
	Engine, Version string
	// OnStart, if set, is called after the staging directory has been
	// populated and before the process ends (cancellation tests block here).
	OnStart func(ctx context.Context, out string) error
	// Calls counts runs.
	Calls int
}

// Run implements optimize.Runner.
func (r *Runner) Run(ctx context.Context, args []string, events, log io.Writer) error {
	r.Calls++
	get := func(flag string) string {
		for i := 0; i+1 < len(args); i++ {
			if args[i] == flag {
				return args[i+1]
			}
		}
		return ""
	}
	src, recipePath, out := get("--source-dir"), get("--recipe"), get("--out")
	emit := func(v map[string]any) {
		b, _ := json.Marshal(v)
		fmt.Fprintln(events, string(b))
	}
	var recipe home.Recipe
	if err := home.ReadJSON(recipePath, &recipe); err != nil {
		return err
	}
	engine, version := r.Engine, r.Version
	if engine == "" {
		engine = setup.OptimizerEngine
	}
	if version == "" {
		version = setup.OptimizerEngineVersion
	}
	emit(map[string]any{"event": "hello", "protocol": "hachidori-optimizer.v1"})
	emit(map[string]any{"event": "phase", "phase": "importing"})
	emit(map[string]any{"event": "engine", "engine": engine, "version": version,
		"versions": map[string]string{"llmcompressor": version, "compressed-tensors": "0.19.0", "torch": "2.11.0+cpu", "transformers": "5.17.0"}})
	emit(map[string]any{"event": "phase", "phase": "loading_source", "detail": src})
	emit(map[string]any{"event": "phase", "phase": "preparing"})

	var patterns []string
	for _, p := range recipe.Preserved {
		if p.Scope == home.ScopeBackbone {
			patterns = append(patterns, p.Pattern)
		}
	}
	preserved := map[string][]string{}
	var quantized []string
	for _, m := range Modules {
		var hit []string
		for _, p := range patterns {
			if match(p, m) {
				hit = append(hit, p)
			}
		}
		if len(hit) > 0 {
			preserved[m] = hit
		} else {
			quantized = append(quantized, m)
		}
	}
	emit(map[string]any{"event": "plan", "linear_modules": len(Modules), "quantized": len(quantized), "preserved": len(preserved)})
	emit(map[string]any{"event": "phase", "phase": "quantizing", "detail": fmt.Sprintf("%d Linear modules", len(quantized))})
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	if r.Fail == "fatal" {
		emit(map[string]any{"event": "fatal", "class": "quantize", "message": "fake optimizer failed"})
		return errors.New("exit status 3")
	}
	emit(map[string]any{"event": "phase", "phase": "serializing"})
	bits := 4
	if r.Fail == "wrong-scheme" {
		bits = 8
	}
	ignore := make([]string, 0, len(preserved))
	for m := range preserved {
		ignore = append(ignore, m)
	}
	sort.Strings(ignore)
	if r.Fail == "unpreserved" {
		ignore = nil
	}
	cfg := map[string]any{"model_type": "qwen3_5", "quantization_config": map[string]any{
		"quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": ignore,
		"config_groups": map[string]any{"group_0": map[string]any{"targets": []string{"Linear"}, "input_activations": nil,
			"weights": map[string]any{"num_bits": bits, "type": "int", "group_size": 128, "symmetric": true}}}}}
	if err := writeJSON(filepath.Join(out, "config.json"), cfg); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "model.safetensors"), []byte("fake-quantized:"+r.Salt+":"+recipe.Name), 0o644); err != nil {
		return err
	}
	for _, rel := range recipe.Carry {
		b, err := os.ReadFile(filepath.Join(src, filepath.FromSlash(rel)))
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(filepath.Join(out, filepath.FromSlash(rel))), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(out, filepath.FromSlash(rel)), b, 0o644); err != nil {
			return err
		}
	}
	if r.Fail == "crash" {
		return errors.New("exit status 1")
	}
	if r.OnStart != nil {
		if err := r.OnStart(ctx, out); err != nil {
			return err
		}
	}
	emit(map[string]any{"event": "phase", "phase": "verifying"})
	if r.Fail != "noreport" {
		pm := map[string]any{}
		for m, pats := range preserved {
			pm[m] = map[string]any{"patterns": pats, "source_precision": "bfloat16", "written_dtype": "BF16"}
		}
		rep := map[string]any{"schema": "hachidori.optimizer-report/1", "engine": engine, "scheme": recipe.Scheme, "algorithm": recipe.Algorithm,
			"device": "cpu", "linear_modules": len(Modules), "quantized_modules": len(quantized), "preserved_modules": pm,
			"quantization_config_ignore": ignore}
		if err := writeJSON(filepath.Join(out, setup.OptimizerReportFile), rep); err != nil {
			return err
		}
	}
	if r.Fail != "incomplete" {
		emit(map[string]any{"event": "done", "report": setup.OptimizerReportFile})
	}
	return nil
}

// match is compressed-tensors' ignore semantics: a "re:" pattern is a regular
// expression matched from the start of the name, anything else is exact.
func match(pattern, name string) bool {
	if rest, ok := strings.CutPrefix(pattern, "re:"); ok {
		return regexp.MustCompile("^(?:" + rest + ")").MatchString(name)
	}
	return pattern == name
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
