package main

import (
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
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

// `decide -model` targets one resident by catalog ID and overrides the
// request file; without it the request is sent exactly as written.
func TestDecideModelFlagSetsTheDirectTarget(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.DecideRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		got = append(got, api.ModelRef(req.Model))
		_ = json.NewEncoder(w).Encode(api.DecideResponse{Schema: api.SchemaV1, Served: &api.Served{Model: api.ModelRef(req.Model)}})
	}))
	defer srv.Close()
	file := filepath.Join(t.TempDir(), "req.json")
	const body = `{"schema":"hachidori.v1","state":"s","questions":[{"id":"q","type":"choice","instructions":"i","choices":["a","b"]}]`
	for _, tc := range []struct{ json, flag, want string }{
		{body + `}`, "", ""},
		{body + `}`, setup.OpenDeciderNano, setup.OpenDeciderNano},
		{body + `,"model":"` + setup.DefaultModel + `"}`, "", setup.DefaultModel},
		{body + `,"model":"` + setup.DefaultModel + `"}`, setup.OpenDeciderNano, setup.OpenDeciderNano},
	} {
		if err := os.WriteFile(file, []byte(tc.json), 0o644); err != nil {
			t.Fatal(err)
		}
		args := []string{"-endpoint", srv.URL}
		if tc.flag != "" {
			args = append(args, "-model", tc.flag)
		}
		if err := cmdDecide(append(args, file)); err != nil {
			t.Fatal(err)
		}
		if last := got[len(got)-1]; last != tc.want {
			t.Fatalf("flag %q file %q: sent model %q, want %q", tc.flag, tc.json, last, tc.want)
		}
	}
}
