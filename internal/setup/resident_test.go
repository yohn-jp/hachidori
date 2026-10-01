package setup_test

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func argValue(args []string, flag string) string {
	if i := slices.Index(args, flag); i >= 0 && i+1 < len(args) {
		return args[i+1]
	}
	return ""
}

// Issue #121: both catalog models are launched from the one active runtime on
// the one active device, each with its own model directory and provider; the
// activation record is only read.
func TestResidentConfigLaunchesEachModelFromTheActiveRuntime(t *testing.T) {
	h, def, other := setup.MaterializeFakeResidents(t, "cuda")
	var before home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &before); err != nil {
		t.Fatal(err)
	}

	want, wantRT, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"", def} { // the default route is WorkerConfig itself
		cfg, rt, err := server.ResidentConfig(h, id, io.Discard)
		if err != nil || rt != wantRT || !slices.Equal(cfg.Args, want.Args) || cfg.Python != want.Python {
			t.Fatalf("default resident %q: %+v %v", id, rt, err)
		}
	}

	cfg, rt, err := server.ResidentConfig(h, other, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	m, _ := setup.LookupModel(other)
	modelDir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if rt.ModelID != other || rt.Runtime != wantRT.Runtime || rt.Device != "cuda" || rt.Model != setup.ModelDirName(m) {
		t.Fatalf("resident runtime %+v", rt)
	}
	if cfg.Python != want.Python || argValue(cfg.Args, "--model-dir") != modelDir ||
		argValue(cfg.Args, "--provider") != m.Provider || argValue(cfg.Args, "--device") != "cuda" ||
		argValue(cfg.Args, "--manifest") != filepath.Join(modelDir, "hachidori-model.json") ||
		argValue(cfg.Args, "--provider") == argValue(want.Args, "--provider") {
		t.Fatalf("resident args %v (default %v)", cfg.Args, want.Args)
	}
	if cfg.Preflight == nil || cfg.Preflight() != nil {
		t.Fatal("resident launch contract not enforced")
	}
	var after home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &after); err != nil || after != before {
		t.Fatalf("activation record changed: %+v -> %+v (%v)", before, after, err)
	}
}

// Only catalog identities can be residents, and only when already
// materialized: nothing is downloaded and no repository can be named.
func TestResidentConfigRefusesUncatalogedAndUnmaterializedModels(t *testing.T) {
	h, _, other := setup.MaterializeFakeResidents(t, "cpu")
	if _, _, err := server.ResidentConfig(h, "someone/else", io.Discard); err == nil || !strings.Contains(err.Error(), "unsupported model") {
		t.Fatalf("arbitrary model: %v", err)
	}
	m, _ := setup.LookupModel(other)
	if err := os.RemoveAll(h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))); err != nil {
		t.Fatal(err)
	}
	if _, _, err := server.ResidentConfig(h, other, io.Discard); err == nil || !strings.Contains(err.Error(), "not materialized") {
		t.Fatalf("unmaterialized model: %v", err)
	}
}

// A resident set whose second model is gone still brings the default resident
// up; the missing one is failed with a cause that names its model and
// provider, never READY, and no process is started for it.
func TestOpenResidentsIsolatesAnUnmaterializedModel(t *testing.T) {
	h, def, other := setup.MaterializeFakeResidents(t, "cpu")
	m, _ := setup.LookupModel(other)
	if err := os.RemoveAll(h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s, err := app.OpenResidents(ctx, h, io.Discard, worker.Policy{QueueDepth: 4}, []string{other, def, other})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Stop()
	if got := s.Models(); !slices.Equal(got, []string{def, other}) || s.Default().Model != def {
		t.Fatalf("members %v default %s", got, s.Default().Model)
	}
	r, ok := s.Resident(other)
	if !ok {
		t.Fatal("missing member")
	}
	s.StartResident(other)
	for i := 0; r.Supervisor.State() != worker.StateFailed; i++ {
		if i > 2000 {
			t.Fatal("resident did not fail")
		}
		time.Sleep(5 * time.Millisecond)
	}
	f := r.Supervisor.LastFailure()
	if f.Class != worker.ClassPreflight || !strings.Contains(f.Message, other) || !strings.Contains(f.Message, m.Provider) ||
		!strings.Contains(f.Message, "not materialized") {
		t.Fatalf("failure %+v", f)
	}
	if snap := r.Supervisor.Snapshot(); snap.PID != 0 || snap.Ready {
		t.Fatalf("snapshot %+v", snap)
	}
	if st := s.ResidentStatuses(); !st[0].Default || st[1].Model != other || st[1].Provider != m.Provider {
		t.Fatalf("statuses %+v", st)
	}
}

// An arbitrary or unknown model is refused when the set is opened.
func TestOpenResidentsRefusesUnknownModel(t *testing.T) {
	h, _, _ := setup.MaterializeFakeResidents(t, "cpu")
	if _, err := app.OpenResidents(context.Background(), h, io.Discard, worker.DefaultPolicy, []string{"someone/else"}); err == nil ||
		!strings.Contains(err.Error(), "unsupported model") {
		t.Fatalf("unknown model: %v", err)
	}
	if _, err := app.OpenResidents(context.Background(), home.Home{Root: t.TempDir()}, io.Discard, worker.DefaultPolicy, nil); err == nil ||
		!strings.Contains(err.Error(), "no active runtime") {
		t.Fatalf("empty home: %v", err)
	}
}
