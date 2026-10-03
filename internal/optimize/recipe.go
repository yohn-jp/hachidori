// Package optimize is the System One optimizer: it builds an immutable,
// source-linked, quantized variant of one pinned catalog model from a
// canonical Hachidori recipe, using an OSS compression backend running in the
// separate optimizer runtime. It performs no model discovery and no network
// access: the optimizer receives a materialized, digest-verified source and a
// recipe, and nothing it writes is visible until every output file has been
// digested and verified and the staged directory is published atomically.
package optimize

import (
	"fmt"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Recipe names.
const (
	// RecipeClefFlashW4A16 is the first real recipe: data-free, weight-only
	// 4-bit (symmetric int4, group size 128, round to nearest) on the Linear
	// modules of the Clef-Flash backbone, with the modules below preserved.
	// The preserved set is derived from the module graph of the pinned release
	// (the checkpoint's weight map and joint_schema_model.py at its revision),
	// not from the names one would expect.
	RecipeClefFlashW4A16 = "clef-flash-w4a16-rtn-g128"
)

// clefCarry are the source files copied unchanged into every Clef variant so
// that it is complete and loadable without the source: everything the release
// ships except the backbone files the backend writes itself (config.json,
// generation_config.json, the model-*.safetensors shards and their index).
var clefCarry = []string{"LICENSE", "chat_template.jinja", "joint_head.safetensors", "joint_head_config.json",
	"joint_schema_model.py", "processor_config.json", "tokenizer.json", "tokenizer_config.json"}

func clefFlashW4A16() home.Recipe {
	return home.Recipe{
		Schema: home.RecipeSchema, Name: RecipeClefFlashW4A16, Engine: setup.OptimizerEngine,
		Scheme: "W4A16", Algorithm: "rtn", Targets: []string{"Linear"},
		Preserved: []home.PreservedModule{
			{Pattern: "lm_head", Scope: home.ScopeBackbone, Precision: "bfloat16",
				Reason: "output embedding: the joint head reads its rows as lexical option vectors (ClefModel passes get_output_embeddings().weight to the head), so quantization error would enter the decision path directly"},
			{Pattern: `re:.*linear_attn\.in_proj_a$`, Scope: home.ScopeBackbone, Precision: "bfloat16",
				Reason: "gated delta-rule decay-gate projection: 32 outputs, negligible bytes, numerically sensitive recurrent state"},
			{Pattern: `re:.*linear_attn\.in_proj_b$`, Scope: home.ScopeBackbone, Precision: "bfloat16",
				Reason: "gated delta-rule beta-gate projection: 32 outputs, negligible bytes, numerically sensitive recurrent state"},
			{Pattern: `re:model\.visual.*`, Scope: home.ScopeBackbone, Precision: "bfloat16",
				Reason: "vision tower: unused by text-only typed decisions, kept at source precision rather than quantized without validation"},
			{Pattern: "joint_head.safetensors", Scope: home.ScopeCarried, Precision: "bfloat16",
				Reason: "joint schema head: the decision-specific scoring head is not part of the backbone checkpoint and is never quantized; carried byte for byte"},
		},
		Carry: append([]string(nil), clefCarry...),
	}
}

// recipes are the canonical recipes per catalog model. A recipe is chosen by
// name; there is no free-form recipe input.
var recipes = map[string][]func() home.Recipe{
	setup.ClefFlash: {clefFlashW4A16},
}

// RecipeNames lists the recipes available for a catalog model.
func RecipeNames(modelID string) []string {
	var out []string
	for _, f := range recipes[modelID] {
		out = append(out, f().Name)
	}
	sort.Strings(out)
	return out
}

// LookupRecipe resolves a recipe name for a catalog model.
func LookupRecipe(modelID, name string) (home.Recipe, error) {
	for _, f := range recipes[modelID] {
		if r := f(); r.Name == name {
			if err := r.Validate(); err != nil {
				return r, err
			}
			return r, nil
		}
	}
	return home.Recipe{}, fmt.Errorf("model %s has no recipe %q (available: %s)", modelID, name, strings.Join(RecipeNames(modelID), ", "))
}

// WeightsOf is the weight precision a recipe's scheme declares. It is checked
// against the saved config of every variant built (setup.CheckPreserved), so
// a scheme the backend did not really apply is refused, never recorded. Tuning
// trials read the same declaration, so a trial transformation and a Forge
// build cannot disagree about the scheme's parameters.
func WeightsOf(r home.Recipe) (home.WeightPrecision, error) { return weightsOf(r) }

func weightsOf(r home.Recipe) (home.WeightPrecision, error) {
	switch r.Scheme {
	case "W4A16":
		return home.WeightPrecision{Scheme: "W4A16", Bits: 4, GroupSize: 128, Symmetric: true,
			Format: "compressed-tensors/pack-quantized", DType: "bfloat16"}, nil
	}
	return home.WeightPrecision{}, fmt.Errorf("recipe scheme %q is not supported", r.Scheme)
}
