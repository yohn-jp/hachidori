package main

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// repair-cache is the explicit, identity-addressed repair of bytecode-cache
// pollution: it repairs a polluted variant and source, leaves everything else
// alone, and accepts no directory as deletion authority.
func TestRepairCacheCommand(t *testing.T) {
	h, m := forgeSource(t)
	res, err := optimize.Build(context.Background(), h, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-fake"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	v := res.Variant
	vdir := h.VariantDir(setup.ClefFlash, v.ID)
	sdir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	pyc := append([]byte{0xcb, 0x0d, 0x0d, 0x0a}, make([]byte, 12)...)
	inject := func(dir string) string {
		p := filepath.Join(dir, "__pycache__", "joint_schema_model.cpython-312.pyc")
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, pyc, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	polluted := inject(vdir)
	if code := run([]string{"variant", "verify", "-home", h.Root, v.ID}, nil); code != 0 {
		t.Fatalf("generated Python bytecode invalidated the variant (exit %d)", code)
	}
	// Bad invocations change nothing: no target, two targets, a directory
	// instead of an identity, an extra argument.
	for _, args := range [][]string{
		{"repair-cache", "-home", h.Root},
		{"repair-cache", "-home", h.Root, "-variant", v.ID, "-model", m.ID},
		{"repair-cache", "-home", h.Root, "-variant", vdir},
		{"repair-cache", "-home", h.Root, "-model", sdir},
		{"repair-cache", "-home", h.Root, "-variant", v.ID, vdir},
		{"repair-cache", "-home", h.Root, "-variant", "../.."},
	} {
		if code := run(args, nil); code == 0 {
			t.Errorf("%v succeeded", args)
		}
	}
	if _, err := os.Stat(polluted); err != nil {
		t.Fatal("an invalid invocation deleted the cache")
	}
	// An unknown unmanifested file refuses the repair and keeps the cache.
	stray := filepath.Join(vdir, "notes.txt")
	os.WriteFile(stray, []byte("x"), 0o644)
	if code := run([]string{"repair-cache", "-home", h.Root, "-variant", v.ID}, nil); code != 1 {
		t.Fatalf("repair beside an unknown file exited %d", code)
	}
	if _, err := os.Stat(polluted); err != nil {
		t.Fatal("a refused repair deleted the cache")
	}
	os.Remove(stray)

	if code := run([]string{"repair-cache", "-home", h.Root, "-variant", v.ID}, nil); code != 0 {
		t.Fatalf("repair exited %d", code)
	}
	if code := run([]string{"variant", "verify", "-home", h.Root, v.ID}, nil); code != 0 {
		t.Fatalf("variant does not verify after repair (exit %d)", code)
	}
	if _, err := os.Stat(filepath.Join(vdir, "__pycache__")); err == nil {
		t.Fatal("__pycache__ remains")
	}

	src := inject(sdir)
	if code := run([]string{"repair-cache", "-home", h.Root, "-model", m.ID}, nil); code != 0 {
		t.Fatalf("source repair exited %d", code)
	}
	if _, err := os.Stat(src); err == nil {
		t.Fatal("the source cache remains")
	}
	// Clean artifacts: a repeat is a successful no-op.
	for _, args := range [][]string{{"-variant", v.ID}, {"-model", m.ID}} {
		if code := run(append([]string{"repair-cache", "-home", h.Root}, args...), nil); code != 0 {
			t.Errorf("%v on a clean artifact exited %d", args, code)
		}
	}
}
