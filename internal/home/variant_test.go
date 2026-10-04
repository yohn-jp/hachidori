package home

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func testRecipe() Recipe {
	return Recipe{Schema: RecipeSchema, Name: "test-w4a16", Engine: "llmcompressor", Scheme: "W4A16", Algorithm: "rtn", Targets: []string{"Linear"},
		Preserved: []PreservedModule{
			{Pattern: "lm_head", Scope: ScopeBackbone, Precision: "bfloat16", Reason: "output embedding"},
			{Pattern: "head.safetensors", Scope: ScopeCarried, Precision: "bfloat16", Reason: "decision head"},
		},
		Carry: []string{"head.safetensors"}}
}

func testSource() ModelManifest {
	return ModelManifest{ID: "clef-flash", Provider: ProviderClef, Repo: "o/clef", Revision: strings.Repeat("ab", 20),
		Files: map[string]string{"config.json": strings.Repeat("1", 64), "head.safetensors": strings.Repeat("2", 64)}}
}

func testVariant(files map[string]string) VariantManifest {
	if files == nil {
		files = map[string]string{"config.json": strings.Repeat("3", 64), "model.safetensors": strings.Repeat("4", 64), "head.safetensors": strings.Repeat("2", 64)}
	}
	v := VariantManifest{Source: SourceOf(testSource()), Provider: ProviderClef,
		Optimizer: Optimizer{Engine: "llmcompressor", Version: "0.14.0", Runtime: "optimizer-cpu-x", Device: "cpu"},
		Recipe:    testRecipe(), Weights: WeightPrecision{Scheme: "W4A16", Bits: 4, GroupSize: 128, Symmetric: true, Format: "compressed-tensors/pack-quantized", DType: "bfloat16"},
		Files: files, Creation: Creation{CreatedAt: "2026-01-01T00:00:00Z", Platform: "linux/amd64", Command: "hachidori variant optimize"}}
	v.Seal()
	return v
}

func testTuning() *TuningProvenance {
	return &TuningProvenance{
		Schema: TuningProvenanceSchemaV1, Source: SourceOf(testSource()),
		ProfileID: strings.Repeat("a", 64), ProfileSHA256: strings.Repeat("a", 64),
		AnalysisID: strings.Repeat("b", 64), AnalysisSHA256: strings.Repeat("b", 64),
		CompilerVersion: "home-recipe/1",
	}
}

func priorRecipeOnlyBuildID(v VariantManifest) string {
	b, err := json.Marshal(struct {
		Schema      string        `json:"schema"`
		Source      VariantSource `json:"source"`
		Provider    string        `json:"provider"`
		Engine      string        `json:"engine"`
		Version     string        `json:"version"`
		Recipe      string        `json:"recipe_sha256"`
		Calibration *Calibration  `json:"calibration,omitempty"`
	}{VariantSchema, v.Source, v.Provider, v.Optimizer.Engine, v.Optimizer.Version, v.RecipeSHA256, v.Calibration})
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Identity is deterministic, and changes with exactly the inputs it is derived
// from: source, engine version, recipe, calibration and artifact bytes.
func TestVariantIdentityIsDeterministicAndSensitive(t *testing.T) {
	a, b := testVariant(nil), testVariant(nil)
	if a.ID != b.ID || a.BuildID != b.BuildID || string(a.Canonical()) != string(b.Canonical()) || a.ManifestSHA256() != b.ManifestSHA256() {
		t.Fatal("identical inputs produced different identities or manifests")
	}
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(a.ID, "clef-flash--test-w4a16--") {
		t.Fatalf("id %q", a.ID)
	}
	// BuildID is the contract only: artifact bytes change the ID, not the BuildID.
	files := map[string]string{"config.json": strings.Repeat("3", 64), "model.safetensors": strings.Repeat("5", 64), "head.safetensors": strings.Repeat("2", 64)}
	drift := testVariant(files)
	if drift.BuildID != a.BuildID || drift.ID == a.ID {
		t.Fatalf("artifact bytes: build %v id %v", drift.BuildID == a.BuildID, drift.ID == a.ID)
	}
	mutate := map[string]func(*VariantManifest){
		"source revision": func(v *VariantManifest) { v.Source.Revision = strings.Repeat("cd", 20) },
		"source files":    func(v *VariantManifest) { v.Source.FilesSHA256 = strings.Repeat("9", 64) },
		"engine version":  func(v *VariantManifest) { v.Optimizer.Version = "0.15.0" },
		"recipe":          func(v *VariantManifest) { v.Recipe.Preserved[0].Reason = "another reason" },
		"calibration":     func(v *VariantManifest) { v.Calibration = &Calibration{ID: "c", SHA256: strings.Repeat("7", 64)} },
	}
	for name, f := range mutate {
		v := testVariant(nil)
		v.Recipe.Preserved = append([]PreservedModule(nil), v.Recipe.Preserved...)
		f(&v)
		v.Seal()
		if v.ID == a.ID || v.BuildID == a.BuildID {
			t.Errorf("a changed %s did not change the variant identity", name)
		}
		if err := v.Validate(); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	// The optimizer runtime and creation time are provenance, not identity.
	v := testVariant(nil)
	v.Optimizer.Runtime, v.Creation.CreatedAt = "optimizer-cpu-y", "2027-01-01T00:00:00Z"
	v.Seal()
	if v.ID != a.ID {
		t.Error("provenance changed the variant identity")
	}
}

func TestTuningProvenanceBindsBuildIdentityAndLegacyVariantsRemainValid(t *testing.T) {
	legacy := testVariant(nil)
	if err := legacy.Validate(); err != nil {
		t.Fatalf("recipe-only legacy variant is invalid: %v", err)
	}
	if priorRecipeOnlyBuildID(legacy) != legacy.BuildID || DeriveBuildIDWithTuning(legacy.Source, legacy.Provider, legacy.Optimizer, legacy.RecipeSHA256, legacy.Calibration, nil) != legacy.BuildID {
		t.Fatal("an absent tuning record changed the legacy build identity")
	}

	tuned := testVariant(nil)
	tuned.Tuning = testTuning()
	tuned.Seal()
	if err := tuned.Validate(); err != nil {
		t.Fatal(err)
	}
	if tuned.RecipeSHA256 != legacy.RecipeSHA256 || tuned.BuildID == legacy.BuildID || tuned.ID == legacy.ID {
		t.Fatal("tuning provenance did not add its identity while keeping the canonical recipe identity")
	}
	otherProfile := testVariant(nil)
	otherProfile.Tuning = testTuning()
	otherProfile.Tuning.ProfileID = strings.Repeat("c", 64)
	otherProfile.Tuning.ProfileSHA256 = strings.Repeat("c", 64)
	otherProfile.Seal()
	if otherProfile.BuildID == tuned.BuildID {
		t.Fatal("a different exact tuning profile did not change BuildID")
	}

	bad := tuned
	bad.Tuning = testTuning()
	bad.Tuning.Source.Revision = strings.Repeat("f", 40)
	if err := bad.Validate(); err == nil {
		t.Fatal("variant accepted tuning provenance for another source")
	}
	bad = tuned
	bad.Tuning = testTuning()
	bad.Tuning.AnalysisSHA256 = strings.Repeat("c", 64)
	if err := bad.Validate(); err == nil {
		t.Fatal("variant accepted an analysis identity that differs from its digest")
	}
}

// A manifest that was edited, truncated or assembled by hand is rejected.
func TestVariantValidateRejectsInconsistentManifests(t *testing.T) {
	base := testVariant(nil)
	cases := map[string]func(*VariantManifest){
		"schema":              func(v *VariantManifest) { v.Schema = "x/1" },
		"id":                  func(v *VariantManifest) { v.ID = "clef-flash--test-w4a16--000000000000" },
		"build id":            func(v *VariantManifest) { v.BuildID = strings.Repeat("0", 64) },
		"recipe digest":       func(v *VariantManifest) { v.RecipeSHA256 = strings.Repeat("0", 64) },
		"recipe edited":       func(v *VariantManifest) { v.Recipe.Scheme = "W8A16" },
		"preserved edited":    func(v *VariantManifest) { v.Preserved = v.Preserved[:1] },
		"no source":           func(v *VariantManifest) { v.Source = VariantSource{} },
		"provider":            func(v *VariantManifest) { v.Provider = "laya" },
		"engine mismatch":     func(v *VariantManifest) { v.Optimizer.Engine = "bitsandbytes" },
		"scheme mismatch":     func(v *VariantManifest) { v.Weights.Scheme = "W8A16" },
		"file added":          func(v *VariantManifest) { v.Files["extra"] = strings.Repeat("1", 64) },
		"file path escapes":   func(v *VariantManifest) { v.Files["../x"] = strings.Repeat("1", 64) },
		"manifest as file":    func(v *VariantManifest) { v.Files[VariantManifestFile] = strings.Repeat("1", 64) },
		"short digest":        func(v *VariantManifest) { v.Files["config.json"] = "abc" },
		"no files":            func(v *VariantManifest) { v.Files = nil },
		"certification link":  func(v *VariantManifest) { v.Certification.Records = "state/other" },
		"certification state": func(v *VariantManifest) { v.Certification.Status = "accepted" },
	}
	for name, f := range cases {
		v := base
		v.Files = map[string]string{}
		for k, d := range base.Files {
			v.Files[k] = d
		}
		v.Preserved = append([]PreservedModule(nil), base.Preserved...)
		f(&v)
		if err := v.Validate(); err == nil {
			t.Errorf("%s: an inconsistent manifest validated", name)
		}
	}
	if err := base.Validate(); err != nil {
		t.Fatal(err)
	}
}

// A recipe that preserves a carried file it does not carry, or carries an
// escaping path, is invalid.
func TestRecipeValidate(t *testing.T) {
	r := testRecipe()
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	r.Carry = nil
	if err := r.Validate(); err == nil {
		t.Error("preserved carried file that is not carried")
	}
	r = testRecipe()
	r.Carry = []string{"../x", "head.safetensors"}
	if err := r.Validate(); err == nil {
		t.Error("escaping carry path")
	}
	r = testRecipe()
	r.Preserved[0].Pattern = "re:("
	if err := r.Validate(); err == nil {
		t.Error("an invalid regular expression")
	}
}

// A variant is linked to exactly one source identity: repository, revision and
// the digest of every pinned source file.
func TestVariantCheckSource(t *testing.T) {
	v, src := testVariant(nil), testSource()
	if err := v.CheckSource(src); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*ModelManifest){
		"id":       func(m *ModelManifest) { m.ID = "clef" },
		"repo":     func(m *ModelManifest) { m.Repo = "evil/clef" },
		"revision": func(m *ModelManifest) { m.Revision = strings.Repeat("cd", 20) },
		"file": func(m *ModelManifest) {
			m.Files = map[string]string{"config.json": strings.Repeat("8", 64), "head.safetensors": strings.Repeat("2", 64)}
		},
		"provider": func(m *ModelManifest) { m.Provider = "laya" },
	} {
		s := src
		s.Files = map[string]string{}
		for k, d := range src.Files {
			s.Files[k] = d
		}
		mutate(&s)
		if err := v.CheckSource(s); err == nil {
			t.Errorf("a variant matched a source whose %s differs", name)
		}
	}
}

// Activation records written before variants existed decode as source-only and
// encode byte for byte as before; the variant fields appear only when set.
func TestActiveRecordCompatibility(t *testing.T) {
	legacy := `{"runtime":"cpu-1","model_id":"laya-base","model":"a--b/rev","device":"cpu"}`
	var a Active
	if err := json.Unmarshal([]byte(legacy), &a); err != nil {
		t.Fatal(err)
	}
	if a.Variant != "" || a.Experimental {
		t.Fatalf("legacy record carries a variant: %+v", a)
	}
	b, _ := json.Marshal(a)
	if string(b) != legacy {
		t.Fatalf("a source-only record changed shape:\n%s\n%s", b, legacy)
	}
	a.Variant, a.Experimental = "clef-flash--x--000000000000", true
	b, _ = json.Marshal(a)
	if !strings.Contains(string(b), `"variant":"clef-flash--x--000000000000"`) || !strings.Contains(string(b), `"experimental":true`) {
		t.Fatalf("variant fields missing: %s", b)
	}
	var back Active
	json.Unmarshal(b, &back)
	if back != a {
		t.Fatalf("round trip %+v", back)
	}
}

// Existing runtime identities are unchanged by the optimizer role field: only
// a spec that names the role derives a role-prefixed identity.
func TestRuntimeSpecRoleDoesNotChangeServingIdentity(t *testing.T) {
	s := RuntimeSpec{Schema: "s", Platform: "linux/amd64", Python: "3.12", Provider: "laya==1", Torch: "2+cpu", Flavor: "cpu", UV: "1", UVSHA256: "a", Project: "p", Lock: "l", WorkerABI: "w"}
	b, _ := json.Marshal(s)
	if strings.Contains(string(b), "role") {
		t.Fatalf("a serving spec encodes a role: %s", b)
	}
	if !strings.HasPrefix(s.ID(), "cpu-") {
		t.Fatalf("serving id %s", s.ID())
	}
	o := s
	o.Role = RoleOptimizer
	if !strings.HasPrefix(o.ID(), "optimizer-cpu-") || o.ID() == s.ID() {
		t.Fatalf("optimizer id %s", o.ID())
	}
}

func TestLoadVariantResolvesRecords(t *testing.T) {
	h := Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := h.LoadVariant(Active{ModelID: "clef-flash"}); ok || err != nil {
		t.Fatalf("a source-only record has a variant: %v %v", ok, err)
	}
	v := testVariant(nil)
	if err := os.MkdirAll(h.VariantDir("clef-flash", v.ID), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSON(h.VariantDir("clef-flash", v.ID)+"/"+VariantManifestFile, v); err != nil {
		t.Fatal(err)
	}
	got, ok, err := h.LoadVariant(Active{ModelID: "clef-flash", Variant: v.ID})
	if err != nil || !ok || got.ID != v.ID {
		t.Fatalf("LoadVariant: %v %v %v", got.ID, ok, err)
	}
	// A variant directory is addressed by its source model and identity; another model ID does not find it.
	if _, _, err := h.LoadVariant(Active{ModelID: "laya-base", Variant: v.ID}); err == nil {
		t.Fatal("a variant resolved under another model")
	}
}

// The concrete optimizer device is part of the build contract unless it is the
// cpu: a cpu build keeps the identity existing variants have, a cuda build is a
// different contract and never shares a build or variant identity with a cpu
// build, even when both produce the same bytes. The optimizer runtime stays
// provenance. A policy such as "auto" is never a recordable device.
func TestOptimizerDeviceSeparatesBuildIdentity(t *testing.T) {
	cpu := testVariant(nil)
	if priorRecipeOnlyBuildID(cpu) != cpu.BuildID {
		t.Fatal("a cpu build no longer derives the identity existing variants were built with")
	}
	unrecorded := testVariant(nil)
	unrecorded.Optimizer.Device = ""
	unrecorded.Seal()
	if unrecorded.BuildID != cpu.BuildID {
		t.Fatal("a manifest that predates the device record no longer derives as the cpu")
	}

	cuda := testVariant(nil)
	cuda.Optimizer.Device, cuda.Optimizer.Runtime = OptimizerDeviceCUDA, "optimizer-cu128-x"
	cuda.Seal()
	if cuda.BuildID == cpu.BuildID || cuda.ID == cpu.ID {
		t.Fatalf("a cuda and a cpu build of the same contract and bytes share an identity: build %v id %v", cuda.BuildID == cpu.BuildID, cuda.ID == cpu.ID)
	}
	if err := cuda.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := cuda.Optimizer; got.Device != "cuda" || got.Runtime != "optimizer-cu128-x" {
		t.Fatalf("the manifest does not record the concrete device and runtime: %+v", got)
	}
	// Only the device is identity: the same device on another runtime is the same contract.
	again := cuda
	again.Optimizer.Runtime = "optimizer-cu128-y"
	again.Seal()
	if again.ID != cuda.ID {
		t.Fatal("the optimizer runtime changed the variant identity")
	}
	// An edited device is rejected: the derived identity no longer recomputes.
	forged := cuda
	forged.Optimizer.Device = OptimizerDeviceCPU
	if err := forged.Validate(); err == nil {
		t.Fatal("a cuda variant relabelled as cpu was accepted")
	}
	for _, device := range []string{"auto", "gpu", "CUDA"} {
		bad := testVariant(nil)
		bad.Optimizer.Device = device
		bad.Seal()
		if err := bad.Validate(); err == nil {
			t.Errorf("optimizer device %q was accepted as a recorded device", device)
		}
	}
}
