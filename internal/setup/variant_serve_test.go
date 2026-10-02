package setup_test

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// activeVariantHome is a home whose active record names a certified variant.
func activeVariantHome(t *testing.T) (home.Home, home.ModelManifest, home.VariantManifest) {
	t.Helper()
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	certify(t, h, m, v, true, time.Now())
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	return h, m, v
}

// A source-only activation launches the source with no variant; a variant
// activation launches the variant's own directory and reports both the
// semantic source model and the variant as execution provenance.
func TestWorkerConfigReportsSourceAndVariantProvenance(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	cfg, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if rt.Variant != nil || slices.Contains(cfg.Args, "--variant-dir") || argValue(cfg.Args, "--provider") != "clef" {
		t.Fatalf("source launch: %+v %v", rt, cfg.Args)
	}

	v := buildVariant(t, h)
	certify(t, h, m, v, true, time.Now())
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	cfg, rt, err = server.WorkerConfig(h, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	vdir := h.VariantDir(setup.ClefFlash, v.ID)
	if argValue(cfg.Args, "--variant-dir") != vdir || argValue(cfg.Args, "--variant-manifest") != filepath.Join(vdir, home.VariantManifestFile) ||
		argValue(cfg.Args, "--model-dir") != h.ModelDir(activeRecord(t, h)) {
		t.Fatalf("variant launch args %v", cfg.Args)
	}
	got := rt.Variant
	if rt.ModelID != setup.ClefFlash || got == nil || got.ID != v.ID || got.Scheme != "W4A16" || got.Bits != 4 || got.DType != "bfloat16" ||
		got.Engine != "llmcompressor" || got.Certification != "accepted" || got.ManifestSHA256 != v.ManifestSHA256() ||
		got.Source.ID != setup.ClefFlash || got.Source.Revision != m.Revision || got.Source.Repo != m.Repo {
		t.Fatalf("provenance %+v (model %s)", got, rt.ModelID)
	}
	// The status document carries it, additively under runtime.variant.
	b, _ := json.Marshal(server.Status{Runtime: rt})
	if !strings.Contains(string(b), `"variant":{"id":"`+v.ID+`"`) || !strings.Contains(string(b), `"model_id":"clef-flash"`) {
		t.Fatalf("status runtime %s", b)
	}
	b, _ = json.Marshal(server.Runtime{ModelID: "laya-base"})
	if strings.Contains(string(b), "variant") {
		t.Fatalf("a source-only runtime encodes a variant: %s", b)
	}
}

// Nothing falls back: a variant that fails any launch check fails the launch,
// and the source is never started in its place.
func TestVariantLaunchNeverFallsBackToSource(t *testing.T) {
	h, m, v := activeVariantHome(t)
	if _, _, err := server.WorkerConfig(h, io.Discard); err != nil {
		t.Fatal(err)
	}
	certDir := filepath.Join(h.Root, filepath.FromSlash(home.CertificationDir(v.ID)))
	moved := certDir + ".away"
	if err := os.Rename(certDir, moved); err != nil {
		t.Fatal(err)
	}
	if cfg, _, err := server.WorkerConfig(h, io.Discard); err == nil || !strings.Contains(err.Error(), "accepted certification record") || cfg.Python != "" {
		t.Fatalf("an uncertified variant launched (or fell back): %v %v", err, cfg.Args)
	}
	os.Rename(moved, certDir)

	// A later rejecting record withdraws acceptance, even for an experimental activation.
	certify(t, h, m, v, false, time.Now().Add(time.Hour))
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("a variant with a rejecting latest record launched")
	}
	os.RemoveAll(certDir)

	// Missing or replaced variant directory.
	os.RemoveAll(h.VariantDir(setup.ClefFlash, v.ID))
	cfg, _, err := server.WorkerConfig(h, io.Discard)
	if err == nil || cfg.Python != "" {
		t.Fatalf("a missing variant launched: %v", err)
	}
}

// The experimental launch is explicit in the record and reported as such; it
// is not a certification.
func TestExperimentalVariantLaunchIsReported(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID, AllowUncertified: true}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	_, rt, err := server.WorkerConfig(h, io.Discard)
	if err != nil || rt.Variant == nil || rt.Variant.Certification != eval.StateExperimental {
		t.Fatalf("experimental launch: %+v %v", rt.Variant, err)
	}
	// Once the variant has an accepted record, the same activation reports it as accepted.
	certify(t, h, m, v, true, time.Now())
	if _, rt, err = server.WorkerConfig(h, io.Discard); err != nil || rt.Variant.Certification != "accepted" {
		t.Fatalf("after certification: %+v %v", rt.Variant, err)
	}
	// A hand-edited record that marks a source-only activation experimental is refused.
	a := activeRecord(t, h)
	a.Variant, a.Experimental = "", true
	home.WriteJSON(h.Path("state", "active-runtime.json"), a)
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("an experimental mark without a variant launched")
	}
}

// The dtype control applies to the source reference only; a variant executes
// at the dtype it declares, and unsupported values are launch errors.
func TestClefDTypeControl(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	t.Setenv(server.EnvClefDType, "float32")
	cfg, _, err := server.WorkerConfig(h, io.Discard)
	if err != nil || argValue(cfg.Args, "--dtype") != "float32" {
		t.Fatalf("reference dtype: %v %v", err, cfg.Args)
	}
	t.Setenv(server.EnvClefDType, "float16")
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("an unsupported dtype was accepted")
	}
	t.Setenv(server.EnvClefDType, "")
	cfg, _, err = server.WorkerConfig(h, io.Discard)
	if err != nil || slices.Contains(cfg.Args, "--dtype") {
		t.Fatalf("default dtype: %v %v", err, cfg.Args)
	}
	v := buildVariant(t, h)
	certify(t, h, m, v, true, time.Now())
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	t.Setenv(server.EnvClefDType, "float32")
	if _, _, err := server.WorkerConfig(h, io.Discard); err == nil {
		t.Fatal("a dtype override was applied to a variant")
	}
}

// The resident set keeps the semantic model identity: the variant is execution
// provenance of the default resident and the source-only members are unchanged.
func TestResidentSetCarriesVariantProvenance(t *testing.T) {
	h, m, v := activeVariantHome(t)
	set, err := app.OpenResidents(context.Background(), h, io.Discard, worker.DefaultPolicy, nil)
	if err != nil {
		t.Fatal(err)
	}
	st := set.Status()
	if st.Runtime.ModelID != setup.ClefFlash || st.Runtime.Variant == nil || st.Runtime.Variant.ID != v.ID {
		t.Fatalf("status runtime %+v", st.Runtime)
	}
	if len(st.Residents) != 1 || st.Residents[0].Model != setup.ClefFlash || st.Residents[0].Status.Runtime.Variant == nil {
		t.Fatalf("residents %+v", st.Residents)
	}
	if id, ok := set.Identity(setup.ClefFlash); !ok || id.Model != setup.ClefFlash || id.Provider != "clef" {
		t.Fatalf("routing identity %+v: a variant must not become a different model identity", id)
	}
	_ = m
}

// A probe launches the persisted variant by the same worker script, arguments
// and verification as serving it, on an explicit device and whatever the
// variant's certification state, and never reads or changes the activation
// record. Nothing falls back to the source or to another device.
func TestProbeConfigLaunchesThePersistedVariantWithoutActivation(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	activeBefore, _ := os.ReadFile(h.Path("state", "active-runtime.json"))

	cfg, rt, err := server.ProbeConfig(h, "cpu", v.ID, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	vdir := h.VariantDir(setup.ClefFlash, v.ID)
	if argValue(cfg.Args, "--variant-dir") != vdir || argValue(cfg.Args, "--variant-manifest") != filepath.Join(vdir, home.VariantManifestFile) ||
		argValue(cfg.Args, "--device") != "cpu" || argValue(cfg.Args, "--provider") != "clef" || argValue(cfg.Args, "--model-dir") == "" {
		t.Fatalf("probe launch args %v", cfg.Args)
	}
	if rt.Variant == nil || rt.Variant.ID != v.ID || rt.Variant.Certification != eval.StateUncertified || rt.ModelID != setup.ClefFlash {
		t.Fatalf("probe provenance %+v", rt.Variant)
	}
	// The same variant is not launchable for serving until it is certified;
	// certification state is reported by the probe, never required by it.
	if _, _, err := server.WorkerConfig(h, io.Discard); err != nil {
		t.Fatalf("source launch: %v", err)
	}
	certify(t, h, m, v, false, time.Now())
	if _, rt, err = server.ProbeConfig(h, "cpu", v.ID, io.Discard); err != nil || rt.Variant.Certification != eval.StateRejected {
		t.Fatalf("a rejected variant must still be probeable and reported as rejected: %+v %v", rt.Variant, err)
	}
	if after, _ := os.ReadFile(h.Path("state", "active-runtime.json")); string(after) != string(activeBefore) {
		t.Fatal("ProbeConfig changed the activation record")
	}

	// No fallback: an unknown variant, an unmaterialized device runtime and a
	// missing variant directory are launch errors.
	if _, _, err := server.ProbeConfig(h, "cpu", "clef-flash--none--000000000000", io.Discard); err == nil {
		t.Fatal("an unknown variant launched")
	}
	if cfg, _, err := server.ProbeConfig(h, "cuda", v.ID, io.Discard); err == nil || cfg.Python != "" || !strings.Contains(err.Error(), "not materialized") {
		t.Fatalf("a device without a runtime launched or fell back: %v", err)
	}
	if _, _, err := server.ProbeConfig(h, "tpu", v.ID, io.Discard); err == nil {
		t.Fatal("an unsupported device was accepted")
	}
	os.RemoveAll(vdir)
	if cfg, _, err := server.ProbeConfig(h, "cpu", v.ID, io.Discard); err == nil || cfg.Python != "" {
		t.Fatalf("a missing variant launched: %v", err)
	}
}

// Experimental is only for a variant with no certification record at all. A
// record that exists but no longer verifies (here a rejecting record whose
// report is gone) is not "no record": neither activation nor launch admits
// the variant as experimental.
func TestExperimentalNeverAdmitsAVariantWithUntrustedRecords(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID, AllowUncertified: true}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	rec := certify(t, h, m, v, false, time.Now())
	dir := filepath.Join(h.Root, filepath.FromSlash(home.CertificationDir(v.ID)))
	if err := os.Remove(filepath.Join(dir, rec.Report)); err != nil {
		t.Fatal(err)
	}
	if st := eval.ResolveCertification(h, v); st.State != eval.StateUncertified || len(st.Problems) == 0 {
		t.Fatalf("precondition: an untrusted record resolves to %+v", st)
	}
	before := rawActive(h)
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID, AllowUncertified: true}, io.Discard, nil); err == nil || !strings.Contains(err.Error(), "not trusted") {
		t.Fatalf("a variant with an untrusted rejecting record was activated as experimental: %v", err)
	}
	if rawActive(h) != before {
		t.Fatal("a refused activation changed the activation record")
	}
	// The earlier experimental activation does not launch it either.
	if cfg, _, err := server.WorkerConfig(h, io.Discard); err == nil || !strings.Contains(err.Error(), "not trusted") || cfg.Python != "" {
		t.Fatalf("a variant with an untrusted rejecting record launched as experimental: %v", err)
	}
}
