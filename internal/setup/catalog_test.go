package setup

import (
	"bytes"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Issue #9 proof 1: the current upstream Laya checkpoint is the default
// catalog identity, unchanged from the former singleton pin.
func TestCatalogDefaultIsUpstreamLaya(t *testing.T) {
	m, err := LookupModel("")
	if err != nil {
		t.Fatal(err)
	}
	byID, _ := LookupModel(DefaultModel)
	if !reflect.DeepEqual(m, byID) || m.ID != "laya-base" || m.Provider != "laya" {
		t.Fatalf("default %+v", m)
	}
	if m.Repo != "convaiinnovations/laya" || m.Revision != "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851" {
		t.Fatalf("default is not the pinned upstream checkpoint: %s@%s", m.Repo, m.Revision)
	}
	want := map[string]string{
		"rl_agent_config.json":            "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
		"model.safetensors":               "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
		"encoder/config.json":             "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
		"tokenizer/tokenizer.json":        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
		"tokenizer/tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
	}
	if !reflect.DeepEqual(m.Files, want) {
		t.Fatalf("default files %v", m.Files)
	}
	// The activation directory is the one the singleton pin used, so existing
	// homes keep their materialized model.
	if ModelDirName(m) != "convaiinnovations--laya/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851" {
		t.Fatalf("dir %s", ModelDirName(m))
	}
}

// Every catalog entry is an immutable, uniquely addressed identity loadable
// by the runtime's provider.
func TestCatalogEntriesAreImmutableIdentities(t *testing.T) {
	commit := regexp.MustCompile(`^[0-9a-f]{40}$`)
	idRe := regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*$`)
	ids, dirs := map[string]bool{}, map[string]bool{}
	for _, m := range Models {
		if !idRe.MatchString(m.ID) || ids[m.ID] {
			t.Errorf("model ID %q invalid or duplicated", m.ID)
		}
		ids[m.ID] = true
		if m.Provider != providerName {
			t.Errorf("%s: provider %q is not the runtime provider %s", m.ID, m.Provider, providerName)
		}
		if !commit.MatchString(m.Revision) {
			t.Errorf("%s: revision %q is not an immutable commit", m.ID, m.Revision)
		}
		if d := ModelDirName(m); dirs[d] || strings.Count(m.Repo, "/") != 1 {
			t.Errorf("%s: directory %s not unique or repo %q malformed", m.ID, d, m.Repo)
		}
		dirs[ModelDirName(m)] = true
		if len(m.Files) == 0 {
			t.Errorf("%s: no pinned files", m.ID)
		}
		for rel, sum := range m.Files {
			if !hex64.MatchString(sum) || path.Clean(rel) != rel || path.IsAbs(rel) || strings.HasPrefix(rel, "..") {
				t.Errorf("%s: bad pin %s=%s", m.ID, rel, sum)
			}
		}
	}
	if !ids[DefaultModel] {
		t.Error("default model not in catalog")
	}
}

// Proofs 3 and 4: known IDs resolve deterministically; anything else,
// including repository or revision spellings, is rejected before setup
// touches the home.
func TestModelSelection(t *testing.T) {
	f := newFixture(t)
	for _, id := range []string{DefaultModel, tunedModel} {
		a, err := LookupModel(id)
		b, _ := LookupModel(id)
		if err != nil || a.ID != id || !reflect.DeepEqual(a, b) {
			t.Fatalf("%s: %+v %v", id, a, err)
		}
	}
	for _, id := range []string{"laya", "LAYA-BASE", "laya-base ", "test/model", "test/model@" + strings.Repeat("ab", 20),
		"convaiinnovations/laya", "https://huggingface.co/convaiinnovations/laya", strings.Repeat("ab", 20)} {
		if _, err := LookupModel(id); err == nil || !strings.Contains(err.Error(), "unsupported model") ||
			!strings.Contains(err.Error(), DefaultModel+", "+tunedModel) {
			t.Errorf("%q: %v", id, err)
		}
		if _, err := f.runModel("cpu", id); err == nil || !strings.Contains(err.Error(), "unsupported model") {
			t.Errorf("setup accepted %q: %v", id, err)
		}
	}
	if _, err := os.Stat(f.H.Path("state")); !os.IsNotExist(err) || len(f.calls()) != 0 || f.hitCount("/uv/uv-fake.tar.gz") != 0 {
		t.Fatal("setup with an unsupported model changed the home")
	}
	// A catalog entry for a provider the runtime cannot load is refused.
	Models = append(Models, home.ModelManifest{ID: "other", Provider: "openjev", Repo: "x/y", Revision: strings.Repeat("ef", 20)})
	if _, err := f.runModel("cpu", "other"); err == nil || !strings.Contains(err.Error(), "not supported by runtime provider") {
		t.Fatalf("foreign provider: %v", err)
	}
}

func (f *fixture) activeRecord() home.Active {
	f.t.Helper()
	var a home.Active
	if err := home.ReadJSON(f.H.Path("state", "active-runtime.json"), &a); err != nil {
		f.t.Fatal(err)
	}
	return a
}

// Proof 2: omitting the selection is exactly selecting the default.
func TestOmittedModelIsDefault(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	omitted := f.active()
	f.mustRunModel("cpu", DefaultModel)
	if !bytes.Equal(f.active(), omitted) {
		t.Fatalf("explicit default differs from omitted selection:\n%s\n%s", omitted, f.active())
	}
	spec, _ := Desired("cpu")
	if a := f.activeRecord(); a != (home.Active{Runtime: spec.ID(), ModelID: DefaultModel, Model: ModelDirName(f.model("")), Device: "cpu"}) {
		t.Fatalf("active %+v", a)
	}
}

// Proofs 5 and 6: selecting another compatible checkpoint reuses the same
// runtime identity, and the activation is derived from the selected manifest.
func TestModelSelectionIsIndependentOfRuntime(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	base := f.activeRecord()
	uvCalls := len(f.calls())
	runtimes, _ := os.ReadDir(f.H.Path("runtime"))

	f.mustRunModel("cpu", tunedModel)
	tuned := f.model(tunedModel)
	a := f.activeRecord()
	if a.Runtime != base.Runtime || len(f.calls()) != uvCalls {
		t.Fatalf("model selection changed/rematerialized the runtime: %s -> %s, uv calls %d -> %d", base.Runtime, a.Runtime, uvCalls, len(f.calls()))
	}
	if after, _ := os.ReadDir(f.H.Path("runtime")); len(after) != len(runtimes) {
		t.Fatal("runtime duplicated for a different model")
	}
	if a.ModelID != tunedModel || a.Model != "test--tuned/"+tuned.Revision || a.Model == base.Model {
		t.Fatalf("active %+v", a)
	}
	var mm home.ModelManifest
	if err := home.ReadJSON(filepath.Join(f.H.ModelDir(a), "hachidori-model.json"), &mm); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mm, tuned) {
		t.Fatalf("materialized manifest %+v, want catalog entry %+v", mm, tuned)
	}
	if err := VerifyModel(f.H.ModelDir(a), tuned); err != nil {
		t.Fatal(err)
	}
	if err := VerifyModel(f.H.ModelDir(a), f.model("")); err == nil {
		t.Fatal("tuned directory verified as the default model")
	}
	if got, err := ActiveModel(a); err != nil || !reflect.DeepEqual(got, tuned) {
		t.Fatalf("active model %+v %v", got, err)
	}
}

// Proof 7: verified model artifacts are reused, including a model
// materialized before the catalog existed (manifest without ID).
func TestModelArtifactsReused(t *testing.T) {
	f := newFixture(t)
	def, tuned := f.model(""), f.model(tunedModel)
	defPath := "/test/model/resolve/" + def.Revision + "/config.json"
	tunedPath := "/test/tuned/resolve/" + tuned.Revision + "/config.json"
	f.mustRun("cpu")
	f.mustRunModel("cpu", tunedModel)
	defDir := f.H.Path("models", filepath.FromSlash(ModelDirName(def)))
	before := snapshot(t, defDir)
	log, err := f.run("cpu")
	if err != nil || !strings.Contains(log, "model "+DefaultModel+" (test/model@abababababab) verified, reusing") {
		t.Fatalf("%v\n%s", err, log)
	}
	if log, err := f.runModel("cpu", tunedModel); err != nil || !strings.Contains(log, "model "+tunedModel+" (") {
		t.Fatalf("%v\n%s", err, log)
	}
	if f.hitCount(defPath) != 1 || f.hitCount(tunedPath) != 1 {
		t.Fatalf("models re-downloaded: %d %d", f.hitCount(defPath), f.hitCount(tunedPath))
	}
	if !maps(before, snapshot(t, defDir)) {
		t.Fatal("reused model modified")
	}

	// Legacy manifest (repo, revision, files only) is reused after verification.
	legacy := home.ModelManifest{Repo: def.Repo, Revision: def.Revision, Files: def.Files}
	if err := home.WriteJSON(filepath.Join(defDir, "hachidori-model.json"), legacy); err != nil {
		t.Fatal(err)
	}
	f.mustRun("cpu")
	if f.hitCount(defPath) != 1 || f.activeRecord().ModelID != DefaultModel {
		t.Fatal("legacy model not reused")
	}
	// A manifest claiming another identity is never accepted.
	home.WriteJSON(filepath.Join(defDir, "hachidori-model.json"), home.ModelManifest{ID: tunedModel, Repo: def.Repo, Revision: def.Revision, Files: def.Files})
	if _, err := f.run("cpu"); err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("mislabelled model accepted: %v", err)
	}
}

// Proof 8: a failed materialization of a newly selected model leaves the
// previously active, valid model active and untouched.
func TestFailedModelSelectionPreservesActiveModel(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cpu")
	prev := f.active()
	a := f.activeRecord()
	before := snapshot(t, f.H.ModelDir(a))
	tuned := f.model(tunedModel)
	tunedPath := "/test/tuned/resolve/" + tuned.Revision + "/config.json"

	f.setDown(tunedPath, true)
	if _, err := f.runModel("cuda", tunedModel); err == nil || !strings.Contains(err.Error(), "model "+tunedModel) {
		t.Fatalf("failed materialization not reported: %v", err)
	}
	if !bytes.Equal(f.active(), prev) {
		t.Fatal("active changed by failed model materialization")
	}
	if !maps(before, snapshot(t, f.H.ModelDir(a))) {
		t.Fatal("active model modified")
	}
	if _, err := os.Stat(f.H.Path("models", filepath.FromSlash(ModelDirName(tuned)))); !os.IsNotExist(err) {
		t.Fatal("failed model published")
	}
	cuda, _ := Desired("cuda")
	if _, err := os.Stat(f.H.Path("runtime", cuda.ID())); !os.IsNotExist(err) {
		t.Fatal("runtime published although the model failed")
	}
	if _, _, _, err := f.H.LoadActive(); err != nil {
		t.Fatal(err)
	}
	if m, err := ActiveModel(a); err != nil || VerifyModel(f.H.ModelDir(a), m) != nil {
		t.Fatalf("previous model no longer valid: %v", err)
	}

	f.setDown(tunedPath, false)
	f.mustRunModel("cuda", tunedModel)
	if got := f.activeRecord(); got.ModelID != tunedModel || got.Runtime != cuda.ID() {
		t.Fatalf("rerun did not converge: %+v", got)
	}
}

func TestActiveModelResolution(t *testing.T) {
	newFixture(t)
	def := Models[0]
	for _, tc := range []struct {
		a  home.Active
		ok bool
	}{
		{home.Active{ModelID: DefaultModel, Model: ModelDirName(def)}, true},
		{home.Active{Model: ModelDirName(def)}, true}, // record from before model selection
		{home.Active{Model: "some--repo/" + strings.Repeat("0", 40)}, false},
		{home.Active{ModelID: "unknown", Model: ModelDirName(def)}, false},
		{home.Active{ModelID: tunedModel, Model: ModelDirName(def)}, false},
	} {
		m, err := ActiveModel(tc.a)
		if (err == nil) != tc.ok || (tc.ok && m.ID != DefaultModel) {
			t.Errorf("%+v: %+v %v", tc.a, m.ID, err)
		}
	}
}
