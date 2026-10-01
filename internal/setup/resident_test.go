package setup_test

import (
	"context"
	"errors"
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

// openConfigured opens the desktop's runtime for h with the desired additional
// residents and reports which binding it produced.
func openConfigured(t *testing.T, h home.Home, want []string, wantErr error) (rt app.Runtime, single, set bool, err error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	var gotSingle, gotSet bool
	open := app.ConfiguredRuntime(ctx, io.Discard, worker.Policy{QueueDepth: 4},
		func() ([]string, error) { return want, wantErr },
		func(*app.WorkerBinding) { gotSingle = true }, func(*app.ResidentSet) { gotSet = true })
	rt, err = open(h.Root)
	if rt != nil {
		t.Cleanup(rt.Stop)
	}
	return rt, gotSingle, gotSet, err
}

// Issue #130: the desktop's runtime follows the saved selection. With no extra
// resident it is exactly the one-worker binding; otherwise it is the resident
// set of the active model (default route) plus the selection, in both
// directions of the catalog pair. Opening only reads: no artifact is created
// and the activation record is unchanged.
func TestConfiguredRuntimeFollowsTheSavedSelection(t *testing.T) {
	h, def, other := setup.MaterializeFakeResidents(t, "cpu")
	var before home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &before); err != nil {
		t.Fatal(err)
	}

	for _, want := range [][]string{nil, {}, {def}, {"", def}} { // the active model alone, however it is spelled
		rt, single, set, err := openConfigured(t, h, want, nil)
		if err != nil {
			t.Fatalf("%v: %v", want, err)
		}
		if _, ok := rt.(*app.WorkerBinding); !ok || !single || set {
			t.Fatalf("selection %v is not the one-worker binding: %T", want, rt)
		}
	}

	for _, want := range [][]string{{other}, {other, def}, {def, other, other}} {
		rt, single, set, err := openConfigured(t, h, want, nil)
		if err != nil {
			t.Fatalf("%v: %v", want, err)
		}
		rs, ok := rt.(*app.ResidentSet)
		if !ok || single || !set || !slices.Equal(rs.Models(), []string{def, other}) || rs.Default().Model != def {
			t.Fatalf("selection %v: %T %v", want, rt, rs)
		}
		if st := rs.Status(); st.Runtime.ModelID != def || len(st.Residents) != 2 || !st.Residents[0].Default || st.Residents[1].Default || rs.Running() {
			t.Fatalf("selection %v: status %+v running %v", want, st, rs.Running())
		}
	}

	// The other direction: opendecider is active and default, laya is extra.
	if _, err := setup.Activate(h, "cpu", other, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	rt, _, set, err := openConfigured(t, h, []string{def}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rs, ok := rt.(*app.ResidentSet); !ok || !set || !slices.Equal(rs.Models(), []string{other, def}) || rs.Default().Model != other {
		t.Fatalf("active %s: %T", other, rt)
	}
	if _, single, _, _ := openConfigured(t, h, []string{other}, nil); !single {
		t.Fatal("selecting the active model added a resident")
	}

	var after home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &after); err != nil || after.ModelID != other {
		t.Fatalf("activation record: %+v %v", after, err)
	}
	if after.Runtime != before.Runtime || after.Device != before.Device {
		t.Fatalf("selection changed the runtime or device: %+v -> %+v", before, after)
	}
}

// An unreadable selection fails the open with its cause; it never silently
// starts fewer residents than were asked for.
func TestConfiguredRuntimeRefusesAnUnreadableSelection(t *testing.T) {
	h, _, _ := setup.MaterializeFakeResidents(t, "cpu")
	rt, single, set, err := openConfigured(t, h, nil, errors.New("settings.json is corrupt"))
	if err == nil || !strings.Contains(err.Error(), "resident models") || !strings.Contains(err.Error(), "settings.json is corrupt") || rt != nil || single || set {
		t.Fatalf("unreadable selection: %v %v %v %v", rt, err, single, set)
	}
}

// A selected resident whose model is not materialized comes up failed with a
// named cause; the default resident is unaffected, nothing is downloaded or
// materialized for it, and no other model stands in for it.
func TestConfiguredRuntimeReportsAMissingSelectedResident(t *testing.T) {
	h, def, other := setup.MaterializeFakeResidents(t, "cpu")
	m, _ := setup.LookupModel(other)
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	rt, _, set, err := openConfigured(t, h, []string{other}, nil)
	if err != nil || !set {
		t.Fatalf("open: %v %v", err, set)
	}
	rs := rt.(*app.ResidentSet)
	rs.StartResident(other)
	r, _ := rs.Resident(other)
	for i := 0; r.Supervisor.State() != worker.StateFailed; i++ {
		if i > 2000 {
			t.Fatal("resident did not fail")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if f := r.Supervisor.LastFailure(); f.Class != worker.ClassPreflight || !strings.Contains(f.Message, other) || !strings.Contains(f.Message, "not materialized") {
		t.Fatalf("failure %+v", f)
	}
	if snap := r.Supervisor.Snapshot(); snap.PID != 0 || snap.Ready {
		t.Fatalf("snapshot %+v", snap)
	}
	if got := rs.Models(); !slices.Equal(got, []string{def, other}) {
		t.Fatalf("members %v", got)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("selecting a resident materialized it: %v", err)
	}
}
