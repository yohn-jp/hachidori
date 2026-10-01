package main

import (
	"flag"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
)

// Issue #9 proof 11: setup selects a model only by catalog ID; there is no
// repository or revision input, and the default preserves today's behavior.
func TestSetupFlagsSelectCatalogModelOnly(t *testing.T) {
	fs, f := newSetupFlags()
	var names []string
	fs.VisitAll(func(fl *flag.Flag) { names = append(names, fl.Name) })
	if !slices.Equal(names, []string{"device", "home", "model"}) {
		t.Fatalf("setup flags %v", names)
	}
	if err := fs.Parse(nil); err != nil || f.model != setup.DefaultModel || f.device != "cuda" {
		t.Fatalf("defaults %+v %v", f, err)
	}
	fs, f = newSetupFlags()
	if err := fs.Parse([]string{"--device", "cpu", "--model", "laya-base"}); err != nil || f.model != "laya-base" || f.device != "cpu" {
		t.Fatalf("parsed %+v %v", f, err)
	}
}

// A home that was never set up must be answered with the setup instruction,
// and serving must not create anything under it.
func TestServeOnAnUnsetHomeSaysToRunSetup(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"serve", "dashboard"} {
		err := runHost(name, []string{"--home", root, "--listen", "127.0.0.1:0"})
		if err == nil || !strings.Contains(err.Error(), "hachidori setup") {
			t.Errorf("%s: err = %v, want the run-setup instruction", name, err)
		}
	}
	if ents, _ := os.ReadDir(root); len(ents) != 0 {
		t.Fatalf("serving an unset home created %v", ents)
	}
}
