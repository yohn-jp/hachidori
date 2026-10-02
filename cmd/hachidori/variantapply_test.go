package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// `variant apply` refuses a bad invocation and an uncertified variant before
// anything is changed or started.
func TestVariantApplyRefusesBeforeChangingAnything(t *testing.T) {
	h, _ := forgeSource(t)
	res, err := optimize.Build(context.Background(), h, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-fake"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	v := res.Variant
	for name, args := range map[string][]string{
		"no variant":     {"-home", h.Root, "-device", "cuda"},
		"no device":      {"-home", h.Root, v.ID},
		"unknown device": {"-home", h.Root, "-device", "tpu", v.ID},
		"unknown flag":   {"-home", h.Root, "-device", "cuda", "-fallback", v.ID},
	} {
		var out, errb bytes.Buffer
		if err := runVariantApply(&out, &errb, args, nil); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if code := run([]string{"variant", "apply", "-home", h.Root, "-device", "cuda", v.ID}, nil); code != 1 {
		t.Fatalf("an uncertified variant exited %d, want 1", code)
	}
	var out, errb bytes.Buffer
	err = runVariantApply(&out, &errb, []string{"-home", h.Root, "-device", "cuda", v.ID}, nil)
	if err == nil || !strings.Contains(err.Error(), "no accepted certification record") || !strings.Contains(errb.String(), "nothing was changed") {
		t.Fatalf("err = %v, stderr %q", err, errb.String())
	}
	if _, err := os.Stat(h.Path("state", "active-runtime.json")); !os.IsNotExist(err) {
		t.Fatal("a refused apply wrote an activation record")
	}
	if _, ok := commands()["variant"]; !ok || !strings.Contains(forgeUsage, "apply") {
		t.Fatal("variant apply is not documented in the usage")
	}
}
