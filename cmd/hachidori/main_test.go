package main

import (
	"flag"
	"slices"
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
