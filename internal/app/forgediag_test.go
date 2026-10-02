package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/optimize/optimizetest"
	"github.com/yohn-jp/hachidori/internal/setup"
)

const fakeToken = "hf_AbCdEf0123456789secrettoken"

func controllerFor(h home.Home, m Maintenance) *Controller {
	return New(Config{Home: h.Root, Installed: func(string) bool { return true },
		Open:        func(string) (Runtime, error) { return newFakeRT(), nil },
		Maintenance: m})
}

func diagOf(t *testing.T, h home.Home, id string) diagnostics.ForgeDiagnostic {
	t.Helper()
	d, err := diagnostics.LoadForge(h.Root, id)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := diagnostics.FormatForge(d)
	if strings.Contains(string(b), fakeToken) || strings.Contains(string(b), h.Root) {
		t.Fatalf("secret or home path in the diagnostic:\n%s", b)
	}
	return d
}

// A failed materialization of a System One source leaves a diagnostic with the
// pinned identity, the phase and the redacted error; the operation's own
// message is exactly the original error.
func TestMaterializeFailureLeavesADiagnostic(t *testing.T) {
	h, _ := forgeHome(t)
	cause := fmt.Errorf("model clef-flash: GET https://huggingface.co/x?token=%s: connection reset by peer", fakeToken)
	c := controllerFor(h, Maintenance{Materialize: func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		obs.OnPhase(setup.PhaseModel)
		obs.OnProgress(setup.Progress{Step: setup.StepDownload, Detail: "model-00001.safetensors", Done: 5, Total: 10, Resumed: 2})
		fmt.Fprintf(log, "downloading with HF_TOKEN=%s under %s\n", fakeToken, root)
		return cause
	}})
	if err := c.Materialize(SetupParams{Device: "cuda", Model: setup.ClefFlash}); err != nil {
		t.Fatal(err)
	}
	f := waitIdle(t, c).Maintenance.Failure
	if f == nil || f.Message != cause.Error() || f.Phase != "model" || f.Step != "downloading" || f.Diagnostic == "" {
		t.Fatalf("failure %+v", f)
	}
	d := diagOf(t, h, f.Diagnostic)
	if d.Operation.Kind != OpMaterialize || d.Operation.Phase != "model" || d.Operation.ID == "" || d.Identity.Model != setup.ClefFlash ||
		d.Identity.SourceRevision != strings.Repeat("ef", 20) || d.Identity.SourceFilesSHA256 == "" || d.Failure.Phase != "model" {
		t.Fatalf("diagnostic %+v", d)
	}
	if len(d.Failure.Chain) == 0 || !strings.Contains(d.Failure.Error, "connection reset by peer") {
		t.Fatalf("failure %+v", d.Failure)
	}
	if len(d.Failure.LogTail) == 0 || d.Runtime.Device != "cuda" || d.Identity.Runtime == "" {
		t.Fatalf("log tail %v runtime %+v", d.Failure.LogTail, d.Runtime)
	}
}

// Only System One operations leave a diagnostic: a failed Laya materialization
// does not.
func TestNonForgeFailuresLeaveNoDiagnostic(t *testing.T) {
	h, _ := forgeHome(t)
	c := controllerFor(h, Maintenance{Materialize: func(root, device, model string, log io.Writer, obs *setup.Observer) error { return errors.New("boom") }})
	if err := c.Materialize(SetupParams{Device: "cpu", Model: setup.DefaultModel}); err != nil {
		t.Fatal(err)
	}
	f := waitIdle(t, c).Maintenance.Failure
	if f == nil || f.Diagnostic != "" || len(diagnostics.ListForge(h.Root, diagnostics.ForgeFilter{})) != 0 {
		t.Fatalf("a Laya failure left a Forge diagnostic: %+v", f)
	}
}

func TestOptimizeFailureDiagnosticCarriesRecipeAndOptimizerEvidence(t *testing.T) {
	h, _ := forgeHome(t)
	// Another source revision's variant exists; this build fails in the optimizer.
	c := controllerFor(h, Maintenance{Optimize: func(ctx context.Context, root, model, recipe string, log io.Writer, obs *setup.Observer) error {
		fmt.Fprintln(log, "optimizer stderr: CUDA out of memory, retry with "+fakeToken)
		_, err := optimize.Build(ctx, h, optimize.Request{Model: model, Recipe: recipe, Reproduce: false},
			optimize.Deps{Runner: &optimizetest.Runner{Fail: "fatal", Salt: "x"}, OptimizerRuntime: "optimizer-test", Preflight: optimize.PreflightDeps{Host: nil}}, log, obs)
		return err
	}})
	// A different build contract (salted runner changes bytes only, so force a
	// new contract by removing the published variant first).
	entries, _ := os.ReadDir(h.VariantsDir(setup.ClefFlash))
	for _, e := range entries {
		os.RemoveAll(filepath.Join(h.VariantsDir(setup.ClefFlash), e.Name()))
	}
	if err := c.Optimize(setup.ClefFlash, optimize.RecipeClefFlashW4A16); err != nil {
		t.Fatal(err)
	}
	f := waitIdle(t, c).Maintenance.Failure
	if f == nil || !strings.Contains(f.Message, "fake optimizer failed") || f.Diagnostic == "" {
		t.Fatalf("failure %+v", f)
	}
	d := diagOf(t, h, f.Diagnostic)
	o := d.Optimization
	if o == nil || o.Recipe != optimize.RecipeClefFlashW4A16 || o.RecipeSHA256 == "" || o.Backend != "llmcompressor" || o.Scheme != "W4A16" ||
		len(o.Preserved) == 0 || !contains(o.Preserved, "lm_head") {
		t.Fatalf("optimization %+v", o)
	}
	if d.Operation.Kind != OpOptimize || d.Identity.Model != setup.ClefFlash || d.Runtime.CompressionBackend != "llmcompressor" {
		t.Fatalf("diagnostic %+v", d)
	}
	// The optimizer's stderr reached the setup log, and so the bounded log tail.
	if !containsLine(d.Failure.LogTail, "CUDA out of memory") {
		t.Fatalf("log tail %v", d.Failure.LogTail)
	}
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

func containsLine(lines []string, sub string) bool {
	for _, l := range lines {
		if strings.Contains(l, sub) {
			return true
		}
	}
	return false
}

func TestCertifyFailureDiagnosticHasIdentitiesNotDatasetContents(t *testing.T) {
	h, v := forgeHome(t)
	dir := t.TempDir()
	ref := filepath.Join(dir, "reference.json")
	os.WriteFile(ref, []byte(`{"schema":"hachidori.resident-run.v1","dataset":"SECRET-DATASET-CONTENT","observations":"SECRET-STATE-BODY"}`), 0o644)
	c := controllerFor(h, Maintenance{})
	if err := c.Certify(CertifyParams{Variant: v.ID, Reference: ref, Candidate: filepath.Join(dir, "missing.json")}); err != nil {
		t.Fatal(err)
	}
	f := waitIdle(t, c).Maintenance.Failure
	if f == nil || f.Diagnostic == "" {
		t.Fatalf("failure %+v", f)
	}
	d := diagOf(t, h, f.Diagnostic)
	b, _ := diagnostics.FormatForge(d)
	if strings.Contains(string(b), "SECRET-DATASET-CONTENT") || strings.Contains(string(b), "SECRET-STATE-BODY") {
		t.Fatalf("dataset or state content copied into the diagnostic:\n%s", b)
	}
	if d.Operation.Kind != OpCertify || d.Identity.Variant != v.ID || d.Identity.VariantManifestSHA256 != v.ManifestSHA256() ||
		d.Certification == nil || d.Certification.PolicyID == "" || d.Certification.PolicySHA256 == "" || d.Optimization == nil {
		t.Fatalf("diagnostic %+v", d)
	}
	// What could not be collected is secondary evidence, not the failure.
	if len(d.Secondary) == 0 || !strings.Contains(strings.Join(d.Secondary, " "), "reference run identity") {
		t.Fatalf("secondary %v", d.Secondary)
	}
	if !strings.Contains(d.Failure.Error, "missing.json") && !strings.Contains(d.Failure.Error, "reference") {
		t.Fatalf("error %q", d.Failure.Error)
	}
}

func TestProbeFailureDiagnosticCarriesRuntimeAndResourceFacts(t *testing.T) {
	h, v := forgeHome(t)
	rec, err := Probe(context.Background(), h, ProbeParams{Variant: v.ID, Device: "cuda"}, probeDeps(t, "crash_on_decide", t.TempDir(), "uncertified"), io.Discard, nil)
	if err == nil {
		t.Fatal("expected a probe failure")
	}
	id := DiagnosticID(WithForgeDiagnostic(h.Root, ForgeFailure{OperationID: "op-1", Kind: OpProbe, Phase: "probing", Variant: v.ID, Device: "cuda", Probe: &rec}, err))
	if id == "" {
		t.Fatal("no diagnostic")
	}
	d := diagOf(t, h, id)
	if d.Runtime.Device != "cuda" || d.Runtime.DType != "bfloat16" || d.Runtime.Torch != "2.11.0" || d.Runtime.Python != "3.12.0" || d.Runtime.Quantization != "W4A16" {
		t.Fatalf("runtime %+v", d.Runtime)
	}
	if d.Resources.LoadMS != 1234.5 || d.Resources.WorkerClass == "" || d.Resources.RAMTotal == 0 && d.Resources.DiskFree == 0 {
		t.Fatalf("resources %+v", d.Resources)
	}
	if !containsLine(d.Failure.StderrTail, "CUDA out of memory") || containsLine(d.Failure.StderrTail, "eyJhbGciOiJIUzI1NiJ9") {
		t.Fatalf("stderr tail %v", d.Failure.StderrTail)
	}
	if d.Preflight == nil || d.Preflight.Kind != setup.PreflightProbe {
		t.Fatalf("preflight %+v", d.Preflight)
	}
	if d.Identity.VariantBuildID == "" || d.Optimization == nil || d.Optimization.RecipeSHA256 == "" {
		t.Fatalf("identity %+v optimization %+v", d.Identity, d.Optimization)
	}
}

// Diagnostic collection can fail in any way; the operation's own error is
// what is reported, and the failure to collect is attached evidence only.
func TestDiagnosticCollectionFailureNeverReplacesThePrimaryError(t *testing.T) {
	h, _ := forgeHome(t)
	primary := errors.New("optimizer exited: killed")
	// 1. The store cannot be written (state/forge is a file).
	os.MkdirAll(h.Path("state"), 0o755)
	os.WriteFile(h.Path("state", "forge"), []byte("x"), 0o644)
	err := WithForgeDiagnostic(h.Root, ForgeFailure{Kind: OpOptimize, Model: setup.ClefFlash}, primary)
	var de *DiagnosticError
	if !errors.As(err, &de) || de.ID != "" || de.CollectErr == nil || !errors.Is(err, primary) || err.Error() != primary.Error() {
		t.Fatalf("unwritable store: %v / %+v", err, de)
	}
	// 2. Collection itself blows up: an error whose text panics.
	err = WithForgeDiagnostic(h.Root, ForgeFailure{Kind: OpOptimize, Model: setup.ClefFlash}, panicky{})
	if !errors.As(err, &de) || de.ID != "" || de.CollectErr == nil || !strings.Contains(de.CollectErr.Error(), "panicked") {
		t.Fatalf("a panic while collecting escaped or was lost: %v", err)
	}
	if _, ok := err.(*DiagnosticError).Err.(panicky); !ok {
		t.Fatal("the original error was replaced")
	}
	// 3. No home: nothing to write into.
	if _, cerr := RecordForgeFailure("", ForgeFailure{Kind: OpOptimize}, primary); cerr == nil {
		t.Fatal("recording without a home succeeded")
	}
	// A nil error stays nil, and an already recorded one is not recorded twice.
	if WithForgeDiagnostic(h.Root, ForgeFailure{}, nil) != nil {
		t.Fatal("nil became an error")
	}
	if again := WithForgeDiagnostic(h.Root, ForgeFailure{Kind: OpOptimize}, err); again != err {
		t.Fatal("a recorded error was wrapped again")
	}
}

type panicky struct{}

func (panicky) Error() string { panic("Error() blew up") }

// Through the controller: a diagnostic that cannot be written leaves the
// operation's message exactly the original error and no diagnostic identity.
func TestControllerFailureKeepsThePrimaryErrorWhenTheDiagnosticCannotBeWritten(t *testing.T) {
	h, _ := forgeHome(t)
	os.MkdirAll(h.Path("state"), 0o755)
	os.WriteFile(h.Path("state", "forge"), []byte("x"), 0o644)
	cause := errors.New("optimizer ended without completing")
	c := controllerFor(h, Maintenance{Optimize: func(context.Context, string, string, string, io.Writer, *setup.Observer) error { return cause }})
	if err := c.Optimize(setup.ClefFlash, optimize.RecipeClefFlashW4A16); err != nil {
		t.Fatal(err)
	}
	f := waitIdle(t, c).Maintenance.Failure
	if f == nil || f.Message != cause.Error() || f.Diagnostic != "" {
		t.Fatalf("failure %+v", f)
	}
}

// A resumed download's progress carries the bytes it already held, through the
// controller's operation and its JSON form; no total is invented.
func TestMaterializeProgressCarriesResumedBytes(t *testing.T) {
	h, _ := forgeHome(t)
	gate := make(chan struct{})
	c := controllerFor(h, Maintenance{Materialize: func(root, device, model string, log io.Writer, obs *setup.Observer) error {
		obs.OnPhase(setup.PhaseModel)
		obs.OnProgress(setup.Progress{Step: setup.StepDownload, Detail: "model-00001.safetensors", Done: 4 << 30, Resumed: 4 << 30, Item: 1, Items: 4})
		<-gate
		return nil
	}})
	if err := c.Materialize(SetupParams{Device: "cuda", Model: setup.ClefFlash}); err != nil {
		t.Fatal(err)
	}
	var p *setup.Progress
	waitFor(t, "progress", func() bool {
		if op := c.Snapshot().Operation; op != nil && op.Progress != nil {
			p = op.Progress
			return true
		}
		return false
	})
	if p.Resumed != 4<<30 || p.Done != 4<<30 || p.Determinate() {
		t.Fatalf("progress %+v: held bytes are reported, and an unknown total stays indeterminate", p)
	}
	if b, _ := json.Marshal(p); !strings.Contains(string(b), `"resumed":4294967296`) || strings.Contains(string(b), `"total"`) {
		t.Fatalf("progress JSON %s", b)
	}
	close(gate)
	if s := waitIdle(t, c); s.Maintenance.Failure != nil {
		t.Fatalf("failure %+v", s.Maintenance.Failure)
	}
}

// recordFailure records the diagnostic of a failed operation f and loads it.
func recordFailure(t *testing.T, h home.Home, f ForgeFailure) diagnostics.ForgeDiagnostic {
	t.Helper()
	id, err := RecordForgeFailure(h.Root, f, errors.New("the operation failed"))
	if err != nil || id == "" {
		t.Fatalf("recording: %v", err)
	}
	return diagOf(t, h, id)
}

// A failed operation's diagnostic carries the preflight of exactly its target
// or none: a report of another device, recipe, variant, model or operation
// kind is never attached, nor a stale or legacy one of the same target.
func TestFailureDiagnosticAttachesOnlyTheExactTargetPreflight(t *testing.T) {
	h, v := forgeHome(t)
	at := time.Unix(100, 0)
	// Evidence on record: a cpu probe of v, a cpu certify preflight of v, an
	// optimize preflight of an unknown recipe, a cpu materialize preflight.
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightProbe, Variant: v.ID, Device: "cpu", Quick: true}, at)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightCertify, Variant: v.ID, Device: "cpu", Quick: true}, at)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: "clef-flash-other-recipe", Quick: true}, at)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightMaterialize, Model: setup.ClefFlash, Device: "cpu", Quick: true}, at)

	for name, f := range map[string]ForgeFailure{
		"cuda probe failure, cpu probe preflight":       {Kind: OpProbe, Variant: v.ID, Device: "cuda"},
		"probe of another variant":                      {Kind: OpProbe, Variant: "clef-flash--other--000000000000", Device: "cpu"},
		"optimize with recipe A, preflight of recipe B": {Kind: OpOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16},
		"optimize with an unresolvable recipe":          {Kind: OpOptimize, Model: setup.ClefFlash, Recipe: "clef-flash-other-recipe"},
		"materialize of another model":                  {Kind: OpMaterialize, Model: setup.DefaultModel, Device: "cpu"},
		"cuda materialize, cpu materialize preflight":   {Kind: OpMaterialize, Model: setup.ClefFlash, Device: "cuda"},
		"repair attaches no preflight":                  {Kind: OpRepair, Model: setup.ClefFlash, Device: "cpu"},
		"certify without a device":                      {Kind: OpCertify, Variant: v.ID},
	} {
		if d := recordFailure(t, h, f); d.Preflight != nil {
			t.Errorf("%s: attached %+v", name, d.Preflight)
		}
	}

	// The exact targets get exactly their report.
	for name, tc := range map[string]struct {
		f    ForgeFailure
		kind string
	}{
		"cpu probe":                       {ForgeFailure{Kind: OpProbe, Variant: v.ID, Device: "cpu"}, setup.PreflightProbe},
		"cpu materialize":                 {ForgeFailure{Kind: OpMaterialize, Model: setup.ClefFlash, Device: "cpu"}, setup.PreflightMaterialize},
		"cpu setup maps to materialize":   {ForgeFailure{Kind: OpSetup, Model: setup.ClefFlash, Device: "cpu"}, setup.PreflightMaterialize},
		"probe and certify stay separate": {ForgeFailure{Kind: OpProbe, Variant: v.ID, Device: "cpu"}, setup.PreflightProbe},
	} {
		d := recordFailure(t, h, tc.f)
		p := d.Preflight
		if p == nil || p.Kind != tc.kind || p.Evidence != setup.EvidenceCurrent || p.Device != tc.f.Device || p.Model != setup.ClefFlash || p.Variant != tc.f.Variant {
			t.Errorf("%s: preflight %+v", name, p)
		}
	}

	// The optimize report of the exact recipe is attached; another one's is not.
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16, Quick: true}, at)
	d := recordFailure(t, h, ForgeFailure{Kind: OpOptimize, Model: setup.ClefFlash, Recipe: optimize.RecipeClefFlashW4A16})
	if p := d.Preflight; p == nil || p.Kind != setup.PreflightOptimize || p.Recipe != optimize.RecipeClefFlashW4A16 || p.Device != "" {
		t.Fatalf("optimize preflight %+v", p)
	}

	// A stale report of the exact target is not attached: the runtime it was
	// bound to has changed.
	spec, _ := setup.Desired("cpu")
	path := h.Path("runtime", spec.ID(), "manifest.json")
	var rm home.RuntimeManifest
	home.ReadJSON(path, &rm)
	rm.PythonVersion += "+rebuilt"
	home.WriteJSON(path, rm)
	if d := recordFailure(t, h, ForgeFailure{Kind: OpProbe, Variant: v.ID, Device: "cpu"}); d.Preflight != nil {
		t.Fatalf("a stale preflight was attached: %+v", d.Preflight)
	}
}

// A variant that does not resolve names no source: the diagnostic does not
// assume the default model and attaches no preflight.
func TestUnresolvedVariantDiagnosticAssumesNoModel(t *testing.T) {
	h, _ := forgeHome(t)
	recordPreflight(t, h, optimize.PreflightRequest{Kind: setup.PreflightMaterialize, Model: setup.ClefFlash, Device: "cpu", Quick: true}, time.Unix(100, 0))
	d := recordFailure(t, h, ForgeFailure{Kind: OpProbe, Variant: "clef-flash--other--000000000000", Device: "cpu"})
	if d.Identity.Model != "" || d.Identity.Variant != "clef-flash--other--000000000000" || d.Preflight != nil ||
		!strings.Contains(strings.Join(d.Secondary, " "), "variant") {
		t.Fatalf("diagnostic identity %+v preflight %+v secondary %v", d.Identity, d.Preflight, d.Secondary)
	}
}
