package setup

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

func runtimeEntry(t *testing.T, inv Inventory, device string) RuntimeEntry {
	t.Helper()
	for _, e := range inv.Runtimes {
		if e.Device == device {
			return e
		}
	}
	t.Fatalf("no %s runtime in inventory", device)
	return RuntimeEntry{}
}

func modelEntry(t *testing.T, inv Inventory, id string) ModelEntry {
	t.Helper()
	for _, e := range inv.Models {
		if e.ID == id {
			return e
		}
	}
	t.Fatalf("no model %s in inventory", id)
	return ModelEntry{}
}

// Inspect is read-only and reports supported, materialized, verified and
// active identities without any network access.
func TestInspectInventory(t *testing.T) {
	f := newFixture(t)
	empty := Inspect(f.H, true)
	if empty.Active != nil || empty.ActiveErr != "" || len(empty.Runtimes) != 2 || len(empty.Models) != len(Models) {
		t.Fatalf("empty inventory %+v", empty)
	}
	if _, err := os.Stat(f.H.Root + "/runtime"); !os.IsNotExist(err) {
		t.Fatal("Inspect created the layout")
	}
	for _, r := range empty.Runtimes {
		if !r.Supported || r.Materialized || r.Active || r.ID == "" {
			t.Fatalf("empty runtime %+v", r)
		}
	}

	f.mustRun("cpu")
	f.srvDown()
	before := snapshot(t, f.H.Root)
	inv := Inspect(f.H, false)
	cpu, cuda := runtimeEntry(t, inv, "cpu"), runtimeEntry(t, inv, "cuda")
	want, _ := Desired("cpu")
	if cpu.ID != want.ID() || !cpu.Materialized || !cpu.Active || cpu.Verified || cuda.Materialized || cuda.Active {
		t.Fatalf("runtimes cpu=%+v cuda=%+v", cpu, cuda)
	}
	def, tuned := modelEntry(t, inv, DefaultModel), modelEntry(t, inv, tunedModel)
	if !def.Materialized || !def.Active || def.Verified || def.Repo != "test/model" || def.Revision == "" || tuned.Materialized || tuned.Active {
		t.Fatalf("models %+v %+v", def, tuned)
	}
	if inv.Active == nil || inv.Active.Device != "cpu" {
		t.Fatalf("active %+v", inv.Active)
	}

	inv = Inspect(f.H, true)
	if !runtimeEntry(t, inv, "cpu").Verified || !modelEntry(t, inv, DefaultModel).Verified {
		t.Fatalf("full verification not reported: %+v", inv)
	}
	if after := snapshot(t, f.H.Root); !maps(before, after) {
		t.Fatal("Inspect changed HACHIDORI_HOME")
	}

	// A corrupted model file is materialized-looking but fails verification.
	mdir := f.H.Path("models", filepath.FromSlash(ModelDirName(f.model(""))))
	if err := os.WriteFile(filepath.Join(mdir, "config.json"), []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	if e := modelEntry(t, Inspect(f.H, true), DefaultModel); e.Verified || e.Problem == "" {
		t.Fatalf("tampered model verified: %+v", e)
	}
}

// srvDown makes any further artifact request fail the test: inspection,
// verification, activation and removal never resolve anything over the network.
func (f *fixture) srvDown() {
	modelBaseURL = "http://127.0.0.1:1/"
	a := uvArtifacts[platform()]
	a.URL = "http://127.0.0.1:1/uv" // the pinned digests, and so the runtime identity, are unchanged
	uvArtifacts[platform()] = a
}

func TestActivateExplicitAndOffline(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	f.mustRunModel("cpu", tunedModel) // activates tuned; make default the active again below
	var log strings.Builder
	if changed, err := Activate(f.H, "cpu", "", &log, nil); err != nil || !changed {
		t.Fatalf("activate default: %v %v", changed, err)
	}
	if changed, err := Activate(f.H, "cpu", "", &log, nil); err != nil || changed {
		t.Fatalf("repeat activate changed=%v err=%v", changed, err)
	}
	f.srvDown()
	if changed, err := Activate(f.H, "cpu", tunedModel, io.Discard, nil); err != nil || !changed {
		t.Fatalf("activate tuned offline: %v %v", changed, err)
	}
	var a home.Active
	home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a)
	if a.ModelID != tunedModel || a.Device != "cpu" {
		t.Fatalf("active %+v", a)
	}
}

// Requested CUDA never silently becomes CPU, and a failed activation leaves
// the current record byte for byte.
func TestActivateFailurePreservesActive(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	old, base := uvArtifacts[platform()], modelBaseURL
	f.srvDown()
	if _, err := Activate(f.H, "cuda", "", io.Discard, nil); err == nil || !strings.Contains(err.Error(), "not materialized") {
		t.Fatalf("cuda activation without a cuda runtime: %v", err)
	}
	if _, err := Activate(f.H, "gpu", "", io.Discard, nil); err == nil {
		t.Fatal("unknown device accepted")
	}
	if _, err := Activate(f.H, "cpu", "not-in-catalog", io.Discard, nil); err == nil {
		t.Fatal("non-catalog model accepted")
	}
	if _, err := Activate(f.H, "cpu", tunedModel, io.Discard, nil); err == nil || !strings.Contains(err.Error(), "not materialized") {
		t.Fatalf("unmaterialized model: %v", err)
	}
	// A corrupted target model is refused; nothing is activated.
	uvArtifacts[platform()] = old
	modelBaseURL = base
	f.mustRunModel("cpu", tunedModel)
	f.mustRunModel("cpu", "")
	prev = f.active()
	tdir := f.H.Path("models", filepath.FromSlash(ModelDirName(f.model(tunedModel))))
	os.WriteFile(filepath.Join(tdir, "config.json"), []byte("bad"), 0o644)
	if _, err := Activate(f.H, "cpu", tunedModel, io.Discard, nil); err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("corrupt model activated: %v", err)
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("failed activation changed the record")
	}
}

func TestMaterializeDoesNotActivate(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	var phases []Phase
	if err := Materialize(f.H, "cuda", tunedModel, io.Discard, &Observer{OnPhase: func(p Phase) { phases = append(phases, p) }}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("Materialize changed the activation record")
	}
	if got := phases[len(phases)-1]; got != PhasePublish {
		t.Fatalf("phases %v", phases)
	}
	inv := Inspect(f.H, false)
	if !runtimeEntry(t, inv, "cuda").Materialized || !modelEntry(t, inv, tunedModel).Materialized {
		t.Fatalf("not materialized: %+v", inv)
	}
}

func TestFailedMaterializePreservesActive(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	f.control(fakeControl{Fail: "sync"})
	if err := Materialize(f.H, "cuda", "", io.Discard, nil); err == nil {
		t.Fatal("injected failure not reported")
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("failed materialization changed the activation record")
	}
	if e := runtimeEntry(t, Inspect(f.H, false), "cuda"); e.Materialized {
		t.Fatalf("failed runtime published: %+v", e)
	}
}

func TestVerifyArtifacts(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	spec, _ := Desired("cpu")
	if err := Verify(f.H, KindRuntime, spec.ID(), nil); err != nil {
		t.Fatal(err)
	}
	if err := Verify(f.H, KindModel, DefaultModel, nil); err != nil {
		t.Fatal(err)
	}
	for _, c := range [][2]string{{KindRuntime, "cpu-deadbeef"}, {KindRuntime, ""}, {KindModel, ""}, {KindModel, "x/y"}, {"other", "a"}} {
		if err := Verify(f.H, c[0], c[1], nil); err == nil {
			t.Fatalf("Verify(%q,%q) accepted", c[0], c[1])
		}
	}
	if err := Verify(f.H, KindModel, tunedModel, nil); err == nil {
		t.Fatal("unmaterialized model verified")
	}
	os.WriteFile(filepath.Join(f.H.Path("models", filepath.FromSlash(ModelDirName(f.model("")))), "config.json"), []byte("bad"), 0o644)
	if err := Verify(f.H, KindModel, DefaultModel, nil); err == nil {
		t.Fatal("corrupt model verified")
	}
}

func TestRepairRebuildsAndRollsBack(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	mdir := f.H.Path("models", filepath.FromSlash(ModelDirName(f.model(""))))
	cfg := filepath.Join(mdir, "config.json")
	os.WriteFile(cfg, []byte("bad"), 0o644)

	// A failing rebuild puts the broken artifact back and changes nothing else.
	f.setDown("/test/model/resolve/"+f.model("").Revision+"/config.json", true)
	if err := Repair(f.H, "cpu", "", io.Discard, nil); err == nil {
		t.Fatal("repair succeeded with the source down")
	}
	if b, _ := os.ReadFile(cfg); string(b) != "bad" {
		t.Fatalf("broken artifact not restored: %q", b)
	}
	if _, err := os.Stat(mdir + ".repair-old"); !os.IsNotExist(err) {
		t.Fatal("aside directory left behind")
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("failed repair changed the activation record")
	}

	f.setDown("/test/model/resolve/"+f.model("").Revision+"/config.json", false)
	if err := Repair(f.H, "cpu", "", io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	if err := Verify(f.H, KindModel, DefaultModel, nil); err != nil {
		t.Fatalf("still broken after repair: %v", err)
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("repair changed the activation record")
	}
	// Repairing a healthy choice rebuilds nothing.
	hits := f.hitCount("/test/model/resolve/" + f.model("").Revision + "/config.json")
	if err := Repair(f.H, "cpu", "", io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	if f.hitCount("/test/model/resolve/"+f.model("").Revision+"/config.json") != hits {
		t.Fatal("healthy artifact was re-downloaded")
	}
}

func TestRemoveUnusedAndRefuseActive(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	f.mustRun("cuda")
	f.mustRunModel("cuda", tunedModel) // active: cuda + tuned
	cpu, _ := Desired("cpu")
	cuda, _ := Desired("cuda")
	prev := f.active()

	for _, c := range [][2]string{{KindRuntime, cuda.ID()}, {KindModel, tunedModel}} {
		if err := Remove(f.H, c[0], c[1], nil); !errors.Is(err, ErrActive) {
			t.Fatalf("Remove(%v) = %v, want ErrActive", c, err)
		}
	}
	if err := Remove(f.H, KindRuntime, cpu.ID(), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(f.H.Path("runtime", cpu.ID())); !os.IsNotExist(err) {
		t.Fatal("unused runtime not removed")
	}
	if err := Remove(f.H, KindModel, DefaultModel, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(f.H.Path("models", "test--model")); !os.IsNotExist(err) {
		t.Fatal("empty repository directory left behind")
	}
	if err := Remove(f.H, KindModel, DefaultModel, nil); err == nil {
		t.Fatal("removing an absent model succeeded")
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("removal changed the activation record")
	}
	if _, _, _, err := f.H.LoadActive(); err != nil {
		t.Fatalf("active state damaged: %v", err)
	}
}

// Removal is confined to catalog identities beneath HACHIDORI_HOME.
func TestRemoveConfinement(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	outside := t.TempDir()
	keep := filepath.Join(outside, "keep")
	os.WriteFile(keep, []byte("x"), 0o644)
	os.MkdirAll(f.H.Path("runtime", "cuda-notcatalog"), 0o755)

	for _, c := range [][2]string{
		{KindRuntime, "../state"}, {KindRuntime, "../../" + filepath.Base(outside)}, {KindRuntime, "."},
		{KindRuntime, ""}, {KindRuntime, "cuda-notcatalog"}, {KindRuntime, f.H.Path("runtime", "x")},
		{KindModel, "../tools"}, {KindModel, ""}, {KindModel, "test--model/" + f.model("").Revision},
		{KindModel, "test/model"}, {"tools", "uv"}, {"", ""},
	} {
		if err := Remove(f.H, c[0], c[1], nil); err == nil {
			t.Fatalf("Remove(%q,%q) accepted", c[0], c[1])
		}
	}
	if _, err := os.Stat(f.H.Path("runtime", "cuda-notcatalog")); err != nil {
		t.Fatal("non-catalog runtime removed")
	}

	// Wrong roots: relative, empty, filesystem root, and a directory that is
	// not a Hachidori home.
	for _, h := range []home.Home{{Root: ""}, {Root: "relative/home"}, {Root: string(filepath.Separator)}, {Root: t.TempDir()}} {
		if err := Remove(h, KindModel, DefaultModel, nil); err == nil {
			t.Fatalf("Remove under %q accepted", h.Root)
		}
	}
	if err := removeConfined(f.H, f.H.Path("runtime"), f.H.Path("runtime")); err == nil {
		t.Fatal("base itself removable")
	}
	if err := removeConfined(f.H, f.H.Path("runtime"), outside); err == nil {
		t.Fatal("target outside base removable")
	}
	if err := removeConfined(f.H, f.H.Path("runtime"), f.H.Path("runtime", "..", "state")); err == nil {
		t.Fatal("traversal removable")
	}
	if err := removeConfined(f.H, f.H.Path("models"), f.H.Path("runtime", "x")); err == nil {
		t.Fatal("target under a different base removable")
	}

	if runtime.GOOS == "windows" {
		return
	}
	// A catalog-named entry that is a symlink out of the home is never followed.
	spec, _ := Desired("cuda")
	if err := os.Symlink(outside, f.H.Path("runtime", spec.ID())); err != nil {
		t.Fatal(err)
	}
	if err := Remove(f.H, KindRuntime, spec.ID(), nil); err == nil {
		t.Fatal("symlinked runtime removed")
	}
	// A runtime/ directory that is itself a link elsewhere is refused too.
	f2 := newFixture(t)
	f2.mustRun("cpu")
	f2.mustRun("cuda")
	f2.mustRunModel("cuda", tunedModel)
	moved := filepath.Join(outside, "runtime")
	os.Rename(f2.H.Path("runtime"), moved)
	os.Symlink(moved, f2.H.Path("runtime"))
	cpu, _ := Desired("cpu")
	if err := Remove(f2.H, KindRuntime, cpu.ID(), nil); err == nil {
		t.Fatal("runtime/ symlinked out of the home was removed")
	}
	if _, err := os.Stat(filepath.Join(moved, cpu.ID())); err != nil {
		t.Fatal("artifact outside the home deleted")
	}
	if b, err := os.ReadFile(keep); err != nil || string(b) != "x" {
		t.Fatal("outside file damaged")
	}
}

// An unreadable activation record cannot prove an artifact is unused.
func TestRemoveRefusedWithoutReadableActivation(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	f.mustRunModel("cpu", tunedModel)
	os.WriteFile(f.H.Path("state", "active-runtime.json"), []byte("{not json"), 0o644)
	if err := Remove(f.H, KindModel, DefaultModel, nil); err == nil {
		t.Fatal("removed with an unreadable activation record")
	}
	if inv := Inspect(f.H, false); inv.ActiveErr == "" || inv.Active != nil {
		t.Fatalf("inventory hides the unreadable record: %+v", inv)
	}
}
