package setup_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// buildVariant builds a variant of the fixture System One model with the
// canonical recipe and the fake optimizer, through the real builder.
func buildVariant(t *testing.T, h home.Home) home.VariantManifest {
	t.Helper()
	res, err := optimize.Build(context.Background(), h, optimize.Request{Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		optimize.Deps{Runner: &optimizetest.Runner{}, OptimizerRuntime: "optimizer-fake"}, io.Discard, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res.Variant
}

// certify records a certification of v with the given verdict. The report is
// a real eval.Certification: its verdict follows from the built-in policy and
// the evidence it carries, which is what the record reader recomputes.
func certify(t *testing.T, h home.Home, m home.ModelManifest, v home.VariantManifest, accepted bool, at time.Time) eval.CertificationRecord {
	t.Helper()
	pol := eval.DefaultPolicy()
	c := eval.Certification{
		Schema: eval.CertificationSchema, CreatedAt: at.UTC().Format(time.RFC3339), Source: v.Source,
		Variant: eval.CertifiedVariant{ID: v.ID, ManifestSHA256: v.ManifestSHA256(), BuildID: v.BuildID, Recipe: v.Recipe.Name,
			RecipeSHA256: v.RecipeSHA256, Scheme: v.Weights.Scheme, Engine: v.Optimizer.Engine, EngineVersion: v.Optimizer.Version},
		Reference: eval.Execution{Role: "reference", ModelID: m.ID, Provider: m.Provider, Revision: m.Revision, Device: "cpu", DType: "torch.bfloat16",
			IdentitySHA256: "ref-identity", ResidentStable: true},
		Candidate: eval.Execution{Role: "candidate", ModelID: m.ID, Provider: m.Provider, Revision: m.Revision, VariantID: v.ID, Device: "cuda",
			DType: "torch.bfloat16", IdentitySHA256: "cand-identity", ResidentStable: true},
		Dataset:  eval.CertifiedDataset{SHA256: "dataset", Cases: 100, Observations: 100, QuestionsSHA256: "questions", InputSHA256: "inputs", SentSHA256: "sent"},
		Fidelity: eval.Fidelity{Paired: 100}, Policy: pol, PolicySHA256: pol.SHA256(),
	}
	if !accepted {
		c.Fidelity.ChoiceFlips, c.Fidelity.FlipRate = 50, 0.5
	}
	c.Verdict = pol.Evaluate(c)
	if got := c.Verdict.Status == eval.VerdictAccepted; got != accepted {
		t.Fatalf("fixture certification verdict %s, wanted accepted=%v", c.Verdict.Status, accepted)
	}
	rec, err := eval.SaveCertification(h, c)
	if err != nil {
		t.Fatal(err)
	}
	return rec
}

func activeRecord(t *testing.T, h home.Home) home.Active {
	t.Helper()
	var a home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); err != nil {
		t.Fatal(err)
	}
	return a
}

func rawActive(h home.Home) string {
	b, _ := os.ReadFile(h.Path("state", "active-runtime.json"))
	return string(b)
}

// A certified variant activates as source plus variant; the source model
// directory and its manifest are untouched, and a record without a variant
// keeps meaning the source artifact.
func TestActivateCertifiedVariantLeavesSourceIntact(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	if a := activeRecord(t, h); a.Variant != "" || a.Experimental {
		t.Fatalf("source-only activation carries a variant: %+v", a)
	}
	srcBefore := dirDigest(t, h.Path("models"))
	v := buildVariant(t, h)
	if v.Source.ID != setup.ClefFlash || v.Source.Revision != m.Revision || !strings.HasPrefix(v.ID, "clef-flash--"+optimize.RecipeClefFlashW4A16+"--") {
		t.Fatalf("variant identity %+v", v)
	}
	certify(t, h, m, v, true, time.Now())
	changed, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil)
	if err != nil || !changed {
		t.Fatalf("activate variant: %v %v", changed, err)
	}
	a := activeRecord(t, h)
	if a.Variant != v.ID || a.ModelID != setup.ClefFlash || a.Experimental {
		t.Fatalf("active %+v", a)
	}
	if got, ok, err := h.LoadVariant(a); err != nil || !ok || got.ID != v.ID {
		t.Fatalf("LoadVariant: %v %v %v", got.ID, ok, err)
	}
	if srcBefore != dirDigest(t, h.Path("models")) {
		t.Fatal("activating a variant changed the source model directory")
	}
	// Source-only activation afterwards is an explicit choice and drops the variant.
	if changed, err := setup.Activate(h, "cpu", setup.ClefFlash, io.Discard, nil); err != nil || !changed || activeRecord(t, h).Variant != "" {
		t.Fatalf("source activation: %v %v %+v", changed, err, activeRecord(t, h))
	}
	// Repeating the same variant activation is idempotent.
	setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil)
	if changed, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil || changed {
		t.Fatalf("repeat: %v %v", changed, err)
	}
}

func dirDigest(t *testing.T, dir string) string {
	t.Helper()
	files, err := setup.DigestTree(dir, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, rel := range sortedKeys(files) {
		b.WriteString(rel + "=" + files[rel] + "\n")
	}
	return b.String()
}

func sortedKeys(m map[string]string) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	for i := range ks {
		for j := i + 1; j < len(ks); j++ {
			if ks[j] < ks[i] {
				ks[i], ks[j] = ks[j], ks[i]
			}
		}
	}
	return ks
}

// Without an accepted certification record a variant is refused, the record
// is untouched, and nothing falls back to anything.
func TestActivateUncertifiedVariantIsRefused(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	before := rawActive(h)
	_, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil)
	if !errors.Is(err, setup.ErrVariantNotCertified) {
		t.Fatalf("uncertified variant activated: %v", err)
	}
	if rawActive(h) != before {
		t.Fatal("a refused activation changed the activation record")
	}
	// A rejecting record is a judgement: refused, with or without the experimental escape.
	certify(t, h, m, v, false, time.Now())
	for _, allow := range []bool{false, true} {
		_, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID, AllowUncertified: allow}, io.Discard, nil)
		if !errors.Is(err, setup.ErrVariantNotCertified) || !strings.Contains(err.Error(), "rejecting") {
			t.Fatalf("rejected variant (experimental=%v): %v", allow, err)
		}
	}
	if rawActive(h) != before {
		t.Fatal("a refused activation changed the activation record")
	}
}

// The experimental launch is opt-in, visibly marked in the activation record,
// and only for a variant nothing has judged.
func TestActivateExperimentalVariantIsMarked(t *testing.T) {
	h, _ := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID, AllowUncertified: true}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	if a := activeRecord(t, h); a.Variant != v.ID || !a.Experimental {
		t.Fatalf("experimental activation not marked: %+v", a)
	}
	// Never implicit: activating the same variant normally is refused, and
	// activating the source clears the mark.
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); !errors.Is(err, setup.ErrVariantNotCertified) {
		t.Fatalf("normal activation of an uncertified variant: %v", err)
	}
	if _, err := setup.Activate(h, "cpu", setup.ClefFlash, io.Discard, nil); err != nil || activeRecord(t, h).Experimental {
		t.Fatalf("source activation kept the experimental mark: %v %+v", err, activeRecord(t, h))
	}
}

// Corrupt, extended, wrong-source and sourceless variants never activate, and
// each failure leaves the activation record exactly as it was.
func TestActivateRejectsInvalidVariants(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	certify(t, h, m, v, true, time.Now())
	before := rawActive(h)
	dir := h.VariantDir(setup.ClefFlash, v.ID)
	try := func(name, want string) {
		t.Helper()
		_, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s: err %v, want %q", name, err, want)
		}
		if rawActive(h) != before {
			t.Fatalf("%s: the activation record changed", name)
		}
	}

	// A flipped byte in an artifact.
	file := filepath.Join(dir, "model.safetensors")
	orig, _ := os.ReadFile(file)
	os.WriteFile(file, append([]byte("x"), orig[1:]...), 0o644)
	try("corrupt artifact", "model.safetensors")
	os.WriteFile(file, orig, 0o644)

	// A file the manifest does not list.
	os.WriteFile(filepath.Join(dir, "extra.bin"), []byte("x"), 0o644)
	try("extra file", "extra.bin")
	os.Remove(filepath.Join(dir, "extra.bin"))

	// A missing artifact.
	os.Remove(file)
	try("missing artifact", "model.safetensors")
	os.WriteFile(file, orig, 0o644)

	// A manifest edited after publication no longer derives its own identity.
	manifest := filepath.Join(dir, home.VariantManifestFile)
	mb, _ := os.ReadFile(manifest)
	os.WriteFile(manifest, []byte(strings.Replace(string(mb), `"scheme": "W4A16"`, `"scheme": "W8A16"`, 1)), 0o644)
	try("edited manifest", "variant")
	os.WriteFile(manifest, mb, 0o644)

	// A variant of another source revision: its source link does not match the catalog.
	other := v
	other.Source.Revision = strings.Repeat("00", 20)
	other.Seal()
	if err := os.MkdirAll(h.VariantDir(setup.ClefFlash, other.ID), 0o755); err != nil {
		t.Fatal(err)
	}
	copyTree(t, dir, h.VariantDir(setup.ClefFlash, other.ID))
	os.WriteFile(filepath.Join(h.VariantDir(setup.ClefFlash, other.ID), home.VariantManifestFile), other.Canonical(), 0o644)
	_, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: other.ID}, io.Discard, nil)
	if err == nil || !strings.Contains(err.Error(), "derives from") {
		t.Fatalf("wrong-source variant: %v", err)
	}
	if rawActive(h) != before {
		t.Fatal("wrong-source activation changed the record")
	}

	// Sanity: the intact variant still activates, so each refusal above was about the fault.
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatalf("intact variant: %v", err)
	}
}

func copyTree(t *testing.T, src, dst string) {
	t.Helper()
	filepath.Walk(src, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		b, _ := os.ReadFile(p)
		os.MkdirAll(filepath.Dir(filepath.Join(dst, rel)), 0o755)
		return os.WriteFile(filepath.Join(dst, rel), b, 0o644)
	})
}

// A variant only exists for System One models, needs its source materialized,
// and activation of a missing variant fails explicitly.
func TestActivateVariantPreconditions(t *testing.T) {
	h, _ := setup.MaterializeFakeClef(t, "cpu")
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: "clef-flash--nope--000000000000"}, io.Discard, nil); err == nil {
		t.Fatal("a variant that does not exist was activated")
	}
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: "../../models"}, io.Discard, nil); err == nil {
		t.Fatal("a path was accepted as a variant")
	}
}

// Certification state is resolved from verified records only: a forged
// verdict, a tampered report and a record about another variant never certify.
func TestCertificationRecordsAreVerified(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	if st := eval.ResolveCertification(h, v); st.State != eval.StateUncertified {
		t.Fatalf("fresh variant is %s", st.State)
	}
	rec := certify(t, h, m, v, false, time.Now().Add(-time.Hour))
	if st := eval.ResolveCertification(h, v); st.State != eval.StateRejected {
		t.Fatalf("state %s", st.State)
	}
	// The latest valid record decides: a later accepted certification supersedes.
	acc := certify(t, h, m, v, true, time.Now())
	if st := eval.ResolveCertification(h, v); st.State != eval.StateAccepted || st.Record == nil || st.Record.ReportSHA256 != acc.ReportSHA256 {
		t.Fatalf("state %+v", st)
	}
	dir := filepath.Join(h.Root, filepath.FromSlash(home.CertificationDir(v.ID)))

	// A hand-edited verdict does not certify.
	recFile := filepath.Join(dir, acc.ReportSHA256[:16]+".record.json")
	b, _ := os.ReadFile(recFile)
	os.WriteFile(recFile, []byte(strings.Replace(string(b), `"verdict": "accepted"`, `"verdict": "rejected"`, 1)), 0o644)
	if st := eval.ResolveCertification(h, v); st.State != eval.StateRejected {
		t.Fatalf("a forged record changed the state to %s (older valid record was rejected)", st.State)
	}
	os.WriteFile(recFile, b, 0o644)

	// A tampered report fails its recorded digest, so the record is not trusted.
	rep := filepath.Join(dir, acc.Report)
	rb, _ := os.ReadFile(rep)
	os.WriteFile(rep, []byte(strings.Replace(string(rb), `"cases": 100`, `"cases": 101`, 1)), 0o644)
	st := eval.ResolveCertification(h, v)
	if st.State != eval.StateRejected || len(st.Problems) == 0 {
		t.Fatalf("tampered report: %+v", st)
	}
	os.WriteFile(rep, rb, 0o644)

	// A record about another manifest digest of the same ID never counts: change the
	// variant manifest identity input (calibration) and nothing carries over.
	other := v
	other.Calibration = &home.Calibration{ID: "other", SHA256: strings.Repeat("a", 64)}
	other.Seal()
	if st := eval.ResolveCertification(h, other); st.State != eval.StateUncertified {
		t.Fatalf("another variant inherited certification: %+v", st)
	}
	_ = rec
}

// The inventory projects variants with their certification state, and Remove
// refuses the active variant and removes an inactive one without touching the source.
func TestInventoryAndRemoveVariants(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	inv := setup.Inspect(h, true)
	if len(inv.Variants) != 1 || inv.Variants[0].ID != v.ID || inv.Variants[0].Certification != eval.StateUncertified || !inv.Variants[0].Verified ||
		!inv.Variants[0].SourceMaterialized || inv.Variants[0].Scheme != "W4A16" || len(inv.Variants[0].Preserved) == 0 {
		t.Fatalf("inventory %+v", inv.Variants)
	}
	if inv.Optimizer == nil || inv.Optimizer.Materialized {
		t.Fatalf("optimizer entry %+v", inv.Optimizer)
	}
	certify(t, h, m, v, true, time.Now())
	if _, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID}, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	inv = setup.Inspect(h, false)
	if e := inv.Variants[0]; !e.Active || e.Certification != eval.StateAccepted {
		t.Fatalf("active variant entry %+v", e)
	}
	if err := setup.Remove(h, setup.KindVariant, v.ID, nil); !errors.Is(err, setup.ErrActive) {
		t.Fatalf("removing the active variant: %v", err)
	}
	if _, err := setup.Activate(h, "cpu", setup.ClefFlash, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	before := dirDigest(t, h.Path("models"))
	if err := setup.Remove(h, setup.KindVariant, v.ID, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(h.VariantDir(setup.ClefFlash, v.ID)); !os.IsNotExist(err) {
		t.Fatal("variant not removed")
	}
	if dirDigest(t, h.Path("models")) != before {
		t.Fatal("removing a variant touched the source")
	}
	if err := setup.Remove(h, setup.KindVariant, "../models", nil); err == nil {
		t.Fatal("a path was accepted as a variant ID")
	}
}

// Verify checks the preserved-module metadata, not just digests: an artifact
// set whose output no longer shows the declared preservation is refused even
// if its manifest was regenerated to match its bytes.
func TestVerifyVariantChecksPreservedModules(t *testing.T) {
	h, _ := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	if err := setup.Verify(h, setup.KindVariant, v.ID, nil); err != nil {
		t.Fatal(err)
	}
	if err := setup.Verify(h, setup.KindVariant, "clef-flash--none--000000000000", nil); err == nil {
		t.Fatal("verified a variant that does not exist")
	}
}

// A variant carries source files unchanged. A swapped carried file is refused
// even when the manifest was regenerated to match the new bytes: the carried
// file must still be the source's file, byte for byte.
func TestCarriedFilesMustBeSourceBytes(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	forged := v
	forged.Files = map[string]string{}
	for k, d := range v.Files {
		forged.Files[k] = d
	}
	dir := h.VariantDir(setup.ClefFlash, v.ID)
	evil := []byte("a different head")
	forged.Files["joint_head.safetensors"] = sha256Hex(evil)
	forged.Seal()
	fdir := h.VariantDir(setup.ClefFlash, forged.ID)
	copyTree(t, dir, fdir)
	os.WriteFile(filepath.Join(fdir, "joint_head.safetensors"), evil, 0o644)
	os.WriteFile(filepath.Join(fdir, home.VariantManifestFile), forged.Canonical(), 0o644)
	err := setup.VerifyVariantArtifacts(h, m, forged, nil)
	if err == nil || !strings.Contains(err.Error(), "joint_head.safetensors") {
		t.Fatalf("a swapped carried file was accepted: %v", err)
	}
}

func sha256Hex(b []byte) string {
	d, _ := setup.FileSHA256Bytes(b)
	return d
}

// Legacy certification records created within the same second have no order:
// the variant is neither accepted nor uncertified, so it is refused even on an
// explicit experimental request, and an accepted record among them never
// certifies it.
func TestAmbiguousLegacyCertificationIsNeverActivated(t *testing.T) {
	h, m := setup.MaterializeFakeClef(t, "cpu")
	v := buildVariant(t, h)
	at := time.Now()
	dir := filepath.Join(h.Root, filepath.FromSlash(home.CertificationDir(v.ID)))
	for _, accepted := range []bool{true, false} {
		rec := certify(t, h, m, v, accepted, at)
		path := filepath.Join(dir, rec.ReportSHA256[:16]+".record.json")
		b, _ := os.ReadFile(path)
		s := strings.Replace(string(b), eval.RecordSchema, eval.LegacyRecordSchema, 1)
		s = strings.Replace(s, fmt.Sprintf("  \"sequence\": %d,\n", rec.Sequence), "", 1)
		os.WriteFile(path, []byte(s), 0o644)
	}
	claims, _ := filepath.Glob(filepath.Join(dir, "*.order"))
	for _, c := range claims {
		os.Remove(c)
	}
	if st := eval.ResolveCertification(h, v); st.State != eval.StateAmbiguous {
		t.Fatalf("state %+v", st)
	}
	for _, allow := range []bool{false, true} {
		_, err := setup.ActivateTarget(h, "cpu", setup.ClefFlash, setup.ActivateOptions{Variant: v.ID, AllowUncertified: allow}, io.Discard, nil)
		if !errors.Is(err, setup.ErrVariantNotCertified) || !strings.Contains(err.Error(), "order cannot be determined") {
			t.Fatalf("ambiguous variant (experimental=%v): %v", allow, err)
		}
	}
}
