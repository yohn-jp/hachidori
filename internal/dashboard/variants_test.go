package dashboard

import (
	"errors"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

type fakeVariants struct {
	mu    sync.Mutex
	calls []string
	err   error
	diag  map[string][]byte
}

func (f *fakeVariants) rec(s string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, s)
	return f.err
}
func (f *fakeVariants) ActivateVariant(device, model, variant string, experimental bool) error {
	s := "activate-variant " + device + " " + model + " " + variant
	if experimental {
		s += " experimental"
	}
	return f.rec(s)
}
func (f *fakeVariants) Optimize(model, recipe string) error {
	return f.rec("optimize " + model + " " + recipe)
}
func (f *fakeVariants) Certify(variant, reference, candidate, policy string) error {
	return f.rec("certify " + variant + " " + reference + " " + candidate + " [" + policy + "]")
}

func (f *fakeVariants) Preflight(kind, model, recipe, variant, device string) error {
	return f.rec("preflight " + kind + " [" + model + "] [" + recipe + "] [" + variant + "] [" + device + "]")
}
func (f *fakeVariants) Probe(variant, device string) error {
	return f.rec("probe " + variant + " " + device)
}
func (f *fakeVariants) Diagnostic(id string) ([]byte, error) {
	f.rec("diagnostic " + id)
	if f.diag == nil {
		return nil, errors.New("no such diagnostic")
	}
	b, ok := f.diag[id]
	if !ok {
		return nil, errors.New("no such diagnostic")
	}
	return b, f.err
}

func withVariants(e *env, fv VariantActions) {
	cfg := e.d.cfg
	cfg.Variants = fv
	e.d = New(cfg)
}

func variantInventory() setup.Inventory {
	inv := modelsInventory()
	inv.Models = append(inv.Models, setup.ModelEntry{ID: "clef-flash", Provider: "clef", Repo: "Cloudflare/clef-flash",
		Revision: "17f0b0ad64efb65d273590632833508766b2aae6", Files: 15, Materialized: true})
	row := func(id, cert string) setup.VariantEntry {
		return setup.VariantEntry{ID: id, SourceID: "clef-flash", SourceRevision: "17f0b0ad64efb65d273590632833508766b2aae6", Recipe: "clef-flash-w4a16-rtn-g128",
			Scheme: "W4A16", Bits: 4, GroupSize: 128, DType: "bfloat16", Engine: "llmcompressor", EngineVersion: "0.14.0", Files: 12,
			Preserved: []string{"lm_head", "joint_head.safetensors"}, Certification: cert, SourceMaterialized: true}
	}
	accepted, uncertified, rejected, broken := row("clef-flash--r--aaaaaaaaaaaa", "accepted"), row("clef-flash--r--bbbbbbbbbbbb", "uncertified"),
		row("clef-flash--r--cccccccccccc", "rejected"), row("clef-flash--r--dddddddddddd", "uncertified")
	accepted.Active = true
	broken.Problem = "model.safetensors: sha256 mismatch"
	inv.Variants = []setup.VariantEntry{accepted, uncertified, rejected, broken}
	inv.Optimizer = &setup.RuntimeEntry{ID: "optimizer-cpu-1234", Device: "cpu", Supported: true}
	return inv
}

// Variants, their recipe and precision, certification state, selected/pending
// state and the operator actions they allow are a projection of the inventory.
func TestVariantsProjection(t *testing.T) {
	e := newEnv(t)
	fm := &fakeModels{state: ModelsState{Inventory: variantInventory(), RestartRequired: true}}
	withModels(e, fm)
	withVariants(e, &fakeVariants{})
	body := e.get(t, "/settings").Body.String()
	for _, want := range []string{`id="variant-inventory"`, `data-variant="clef-flash--r--aaaaaaaaaaaa"`, "W4A16 · 4-bit g128 · compute bfloat16",
		"llmcompressor 0.14.0", `<span class="badge tone-ok">accepted</span>`, `<span class="badge tone-bad">rejected</span>`, `<span class="badge">uncertified</span>`,
		"active · applies on restart", "model.safetensors: sha256 mismatch", `id="optimizer-runtime"`, "optimizer-cpu-1234",
		`id="optimize-clef-flash"`, `action="/settings/variants/optimize"`, `value="clef-flash-w4a16-rtn-g128"`,
		`id="certify-variant"`, `action="/settings/variants/certify"`, `name="reference"`, `name="candidate"`} {
		if !strings.Contains(body, want) {
			t.Errorf("variants section lacks %q", want)
		}
	}
	// Only a certified variant offers Activate; only an unjudged one offers the
	// explicit experimental launch; a rejected or broken one offers neither.
	row := func(id string) string {
		i := strings.Index(body, `data-variant="`+id+`"`)
		if i < 0 {
			t.Fatalf("no row for %s", id)
		}
		end := strings.Index(body[i:], "</tr>")
		return body[i : i+end]
	}
	has := func(r, s string) bool { return strings.Contains(r, s) }
	if r := row("clef-flash--r--aaaaaaaaaaaa"); !has(r, `action="/settings/variants/activate"`) || has(r, `name="experimental"`) || has(r, `action="/settings/models/remove"`) {
		t.Errorf("accepted active variant row:\n%s", r)
	}
	if r := row("clef-flash--r--bbbbbbbbbbbb"); has(r, `class="btn primary">Activate`) || !has(r, `name="experimental" value="1"`) || !has(r, "Activate as experimental") ||
		!has(r, "experimental/uncertified? It has no certification record") {
		t.Errorf("uncertified variant row:\n%s", r)
	}
	if r := row("clef-flash--r--cccccccccccc"); has(r, `/settings/variants/activate`) || !has(r, `action="/settings/models/remove"`) || !has(r, `name="kind" value="variant"`) {
		t.Errorf("rejected variant row:\n%s", r)
	}
	if r := row("clef-flash--r--dddddddddddd"); has(r, `/settings/variants/activate`) {
		t.Errorf("a variant with a problem can be activated:\n%s", r)
	}
	// Every variant can be verified through the existing maintenance path.
	if n := strings.Count(body, `name="kind" value="variant"><input type="hidden" name="id"`); n < 4 {
		t.Errorf("%d variant forms", n)
	}
}

// The experimental mark and the running variant are projections of the
// activation record and the status document, never of dashboard state.
func TestVariantRunningAndExperimentalBadges(t *testing.T) {
	e := newEnv(t)
	inv := variantInventory()
	inv.Variants[0].Active, inv.Variants[0].Experimental = false, false
	inv.Variants[1].Active, inv.Variants[1].Experimental = true, true
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "clef-flash", Device: "cuda", Variant: "clef-flash--r--bbbbbbbbbbbb", Experimental: true}
	fm := &fakeModels{state: ModelsState{Inventory: inv}}
	cfg := e.d.cfg
	cfg.Models = fm
	cfg.Variants = &fakeVariants{}
	started := time.Now()
	rt := server.Runtime{Home: e.home, Runtime: "cu128-aaaa", ModelID: "clef-flash", Model: "Cloudflare--clef-flash/17f0b0ad", Device: "cuda",
		Variant: &server.Variant{ID: "clef-flash--r--bbbbbbbbbbbb", Recipe: "clef-flash-w4a16-rtn-g128", Scheme: "W4A16", Bits: 4, DType: "bfloat16",
			Format: "compressed-tensors/pack-quantized", Engine: "llmcompressor", EngineVersion: "0.14.0", Certification: "experimental/uncertified",
			Source: server.VariantSource{ID: "clef-flash", Provider: "clef", Repo: "Cloudflare/clef-flash", Revision: "17f0b0ad"}}}
	cfg.Status = func() server.Status { return server.StatusBody(e.rt, rt, started) }
	e.d = New(cfg)
	settings := e.get(t, "/settings").Body.String()
	for _, want := range []string{`<span class="badge tone-ok">running</span>`, `<span class="badge tone-bad">experimental/uncertified</span>`,
		"clef-flash · cuda · variant clef-flash--r--bbbbbbbbbbbb"} {
		if !strings.Contains(settings, want) {
			t.Errorf("settings lacks %q", want)
		}
	}
	// The Runtime workspace states exactly what executes: the semantic model and the variant.
	runtime := e.get(t, "/").Body.String()
	for _, want := range []string{`id="runtime-variant"`, "variant clef-flash--r--bbbbbbbbbbbb · W4A16 · experimental/uncertified"} {
		if !strings.Contains(runtime, want) {
			t.Errorf("runtime page lacks %q", want)
		}
	}
	if st := e.get(t, "/api/status").Body.String(); !strings.Contains(st, `"variant"`) || !strings.Contains(st, `"model_id":"clef-flash"`) {
		t.Errorf("status document %s", st)
	}
}

// Without the variant actions the variants are still listed and verifiable but
// no lifecycle action is offered, and the routes do not exist.
func TestVariantControlsAbsentWithoutAuthority(t *testing.T) {
	e := newEnv(t)
	withModels(e, &fakeModels{state: ModelsState{Inventory: variantInventory()}})
	body := e.get(t, "/settings").Body.String()
	if !strings.Contains(body, `id="variant-inventory"`) || strings.Contains(body, "/settings/variants/") {
		t.Error("variant controls rendered without the authority")
	}
	if rec := e.post(t, "/settings/variants/optimize", url.Values{"model": {"clef-flash"}}); rec.Code != http.StatusNotFound {
		t.Fatalf("variants route without authority: %d", rec.Code)
	}
}

// Variant actions are forwarded with explicit arguments, return to Settings,
// show refusals and need the form token; they never touch the lifecycle.
func TestVariantActionsForwarded(t *testing.T) {
	e := newEnv(t)
	fm := &fakeModels{state: ModelsState{Inventory: variantInventory()}}
	fv := &fakeVariants{}
	withModels(e, fm)
	withVariants(e, fv)
	for _, c := range []struct {
		op   string
		form url.Values
		want string
	}{
		{"activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v1"}}, "activate-variant cuda clef-flash v1"},
		{"activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v2"}, "experimental": {"1"}}, "activate-variant cuda clef-flash v2 experimental"},
		{"optimize", url.Values{"model": {"clef-flash"}, "recipe": {"clef-flash-w4a16-rtn-g128"}}, "optimize clef-flash clef-flash-w4a16-rtn-g128"},
		{"certify", url.Values{"variant": {"v1"}, "reference": {" C:\\runs\\ref.json "}, "candidate": {"C:\\runs\\cand.json"}}, "certify v1 C:\\runs\\ref.json C:\\runs\\cand.json []"},
	} {
		c.form.Set("return", "settings")
		rec := e.post(t, "/settings/variants/"+c.op, c.form)
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/settings" {
			t.Fatalf("%s: %d %s", c.op, rec.Code, rec.Header().Get("Location"))
		}
		if a := e.lastAction(t); !a.OK {
			t.Fatalf("%s: %+v", c.op, a)
		}
		if got := fv.calls[len(fv.calls)-1]; got != c.want {
			t.Fatalf("%s forwarded %q, want %q", c.op, got, c.want)
		}
	}
	// Experimental is only ever the explicit value 1.
	e.post(t, "/settings/variants/activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v3"}, "experimental": {"true"}})
	if got := fv.calls[len(fv.calls)-1]; strings.Contains(got, "experimental") {
		t.Fatalf("a loose truthy value requested the experimental launch: %q", got)
	}
	if rec := e.post(t, "/settings/variants/format-disk", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("unknown op: %d", rec.Code)
	}
	if rec := e.post(t, "/settings/variants/optimize", url.Values{"token": {"forged"}, "model": {"clef-flash"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
	n := len(fv.calls)
	fv.err = errors.New("variant is not certified")
	e.post(t, "/settings/variants/activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v1"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "not certified") || len(fv.calls) != n+1 {
		t.Fatalf("refusal not shown: %+v", a)
	}
	// Verify and remove of a variant use the existing maintenance path.
	e.post(t, "/settings/models/verify", url.Values{"kind": {"variant"}, "id": {"v1"}})
	e.post(t, "/settings/models/remove", url.Values{"kind": {"variant"}, "id": {"v1"}})
	if got := fm.calls; len(got) < 2 || got[len(got)-2] != "verify variant v1" || got[len(got)-1] != "remove variant v1" {
		t.Fatalf("maintenance calls %v", got)
	}
	e.rt.mu.Lock()
	calls := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(calls) != 0 {
		t.Fatalf("lifecycle touched: %v", calls)
	}
}

// Optimization and certification progress use the same operation view: real
// phases of the plan, an indeterminate step without a percentage, and failure
// evidence with the phase it failed in. Every label has a Japanese entry
// (TestOperationMessagesAreCatalogued covers the catalog).
func TestOptimizeOperationProgressIsTruthful(t *testing.T) {
	e := newEnv(t)
	plan := []string{"model", "preparing", "runtime", "starting", "loading_source", "resolving", "quantizing", "serializing", "verifying", "publish"}
	fm := &fakeModels{state: ModelsState{Inventory: variantInventory(),
		Busy: &ModelOp{Kind: "optimize", Model: "clef-flash", Target: "recipe clef-flash-w4a16-rtn-g128", Plan: plan,
			Phases: plan[:7], Phase: "quantizing", Step: "materializing", Detail: "25 Linear modules", Started: time.Now().Add(-90 * time.Second)}}}
	withModels(e, fm)
	withVariants(e, &fakeVariants{})
	body := e.get(t, "/settings").Body.String()
	for _, want := range []string{`id="models-busy"`, "Quantizing", "Loading source model", "Resolving modules", "25 Linear modules"} {
		if !strings.Contains(body, want) {
			t.Errorf("busy view lacks %q", want)
		}
	}
	busy := body[strings.Index(body, `id="models-busy"`):]
	busy = busy[:strings.Index(busy, "</div>")+6]
	if strings.Contains(busy, "aria-valuenow") || strings.Contains(busy, "%)") {
		t.Errorf("a percentage was shown for an indeterminate step:\n%s", busy)
	}
	fm.state.Busy = nil
	fm.state.Last = &ModelOp{Kind: "optimize", Model: "clef-flash", Plan: plan, Phases: plan[:5], Phase: "quantizing", Failure: "optimizer quantize: out of memory",
		FailurePhase: "quantizing", FailureStep: "materializing", Started: time.Now().Add(-time.Minute), Finished: time.Now()}
	body = e.get(t, "/settings").Body.String()
	for _, want := range []string{`id="models-last"`, "optimizer quantize: out of memory", "Failed in phase <strong>Quantizing</strong>"} {
		if !strings.Contains(body, want) {
			t.Errorf("failure view lacks %q", want)
		}
	}
	_ = worker.StateReady
}

// Forge readiness is a projection of the recorded reports: blockers, warnings
// and unknowns are shown as such, an unknown is never turned into READY, the
// probe of each variant is shown with its outcome (and marked stale when it was
// recorded for another manifest), and nothing is derived from text.
func TestForgeReadinessProjection(t *testing.T) {
	e := newEnv(t)
	inv := variantInventory()
	inv.Variants[0].ManifestSHA256, inv.Variants[1].ManifestSHA256 = "sha-a", "sha-b"
	inv.Models[len(inv.Models)-1].PartialBytes = 3 << 30
	inv.Models[len(inv.Models)-1].Materialized = false
	fm := &fakeModels{state: ModelsState{Inventory: inv, Forge: ForgeState{
		Preflights: []PreflightRow{
			{Kind: "optimize", Model: "clef-flash", Recipe: "clef-flash-w4a16-rtn-g128", At: "2026-10-02T01:00:00Z", Outcome: "blocked", Pass: 5, Blocker: 1, Unknown: 1,
				Findings: []FindingRow{{ID: "storage.capacity", Status: "blocker", Summary: "1.0 GiB is free but even the smallest possible variant needs 4.6 GiB"},
					{ID: "memory.fit", Status: "unknown", Summary: "host RAM is 64.0 GiB; whether it fits is not known"}}},
			{Kind: "probe", Model: "clef-flash", Variant: "clef-flash--r--aaaaaaaaaaaa", Device: "cuda", At: "2026-10-02T00:00:00Z", Outcome: "attention", Pass: 8, Unknown: 1,
				Findings: []FindingRow{{ID: "accelerator.vram", Status: "unknown", Summary: "the weights are below the observed VRAM but fit is not known until it loads"}}},
		},
		Probes: []ProbeRow{
			{Variant: "clef-flash--r--aaaaaaaaaaaa", Device: "cuda", Result: "passed", StartedAt: "2026-10-02T00:00:00Z", ManifestSHA256: "sha-a", LoadMS: 1234.5, WarmupMS: 67.8, RequestMS: 12.5, DType: "bfloat16", DeviceName: "RTX 3060"},
			{Variant: "clef-flash--r--bbbbbbbbbbbb", Device: "cuda", Result: "passed", ManifestSHA256: "sha-OLD"},
			{Variant: "clef-flash--r--cccccccccccc", Device: "cpu", Result: "failed", Phase: "provenance", Error: "device cuda was requested but the probe worker is on cpu", ManifestSHA256: ""},
		},
		Diagnostics: []DiagnosticRow{{ID: "probe-20261002T000000Z-0123abcd", Kind: "probe", Phase: "probing", Model: "clef-flash", Created: "2026-10-02T00:00:00Z", Error: "boom"}},
	}}}
	withModels(e, fm)
	withVariants(e, &fakeVariants{})
	body := e.get(t, "/settings").Body.String()
	for _, want := range []string{`id="forge-readiness"`, `data-outcome="blocked"`, `data-outcome="attention"`, "storage.capacity", "memory.fit", "BLOCKED", "ATTENTION",
		"Not READY: this is not a promise", "interrupted download kept for resume", "3.0 GiB", `id="forge-diagnostics"`, `href="/settings/forge/diagnostics/probe-20261002T000000Z-0123abcd"`} {
		if !strings.Contains(body, want) {
			t.Errorf("settings lacks %q", want)
		}
	}
	forge := body[strings.Index(body, `id="forge-readiness"`):]
	if strings.Contains(forge, ">READY<") {
		t.Error("a report with an unknown or a blocker was shown as READY")
	}
	row := func(id string) string {
		i := strings.Index(body, `data-variant="`+id+`"`)
		end := strings.Index(body[i:], "</tr>")
		return body[i : i+end]
	}
	if r := row("clef-flash--r--aaaaaaaaaaaa"); !strings.Contains(r, `data-probe="passed"`) || !strings.Contains(r, "probe passed") || !strings.Contains(r, "1234.5 ms") || strings.Contains(r, "stale") ||
		!strings.Contains(r, `action="/settings/variants/probe"`) || !strings.Contains(r, `formaction="/settings/variants/preflight"`) {
		t.Errorf("probed variant row:\n%s", r)
	}
	if r := row("clef-flash--r--bbbbbbbbbbbb"); !strings.Contains(r, "stale") || strings.Contains(r, `class="badge tone-ok">probe passed`) {
		t.Errorf("stale probe row:\n%s", r)
	}
	if r := row("clef-flash--r--cccccccccccc"); !strings.Contains(r, "probe failed") || !strings.Contains(r, "phase provenance") || !strings.Contains(r, "no fallback") && !strings.Contains(r, "the probe worker is on cpu") {
		t.Errorf("failed probe row:\n%s", r)
	}
	if r := row("clef-flash--r--dddddddddddd"); !strings.Contains(r, "not probed") || strings.Contains(r, `action="/settings/variants/probe"`) {
		t.Errorf("a variant with a problem can be probed, or is shown as probed:\n%s", r)
	}
	// Probe is not certification: a passing probe never offers Activate.
	if r := row("clef-flash--r--bbbbbbbbbbbb"); strings.Contains(r, `class="btn primary">Activate`) {
		t.Errorf("a probe made a variant activatable:\n%s", r)
	}
}

func TestForgeActionsForwardedAndDiagnosticDownload(t *testing.T) {
	e := newEnv(t)
	withModels(e, &fakeModels{state: ModelsState{Inventory: variantInventory()}})
	fv := &fakeVariants{diag: map[string][]byte{"probe-20261002T000000Z-0123abcd": []byte(`{"schema":"hachidori.forge-diagnostic/v1"}`)}}
	withVariants(e, fv)
	for _, c := range []struct {
		op   string
		form url.Values
		want string
	}{
		{"preflight", url.Values{"kind": {"optimize"}, "model": {"clef-flash"}, "recipe": {"r1"}}, "preflight optimize [clef-flash] [r1] [] []"},
		{"preflight", url.Values{"kind": {"probe"}, "variant": {"v1"}, "device": {"cuda"}}, "preflight probe [] [] [v1] [cuda]"},
		{"probe", url.Values{"variant": {"v1"}, "device": {"cpu"}}, "probe v1 cpu"},
	} {
		c.form.Set("return", "settings")
		rec := e.post(t, "/settings/variants/"+c.op, c.form)
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/settings" {
			t.Fatalf("%s: %d", c.op, rec.Code)
		}
		if got := fv.calls[len(fv.calls)-1]; got != c.want {
			t.Fatalf("%s forwarded %q, want %q", c.op, got, c.want)
		}
	}
	if rec := e.post(t, "/settings/variants/probe", url.Values{"token": {"forged"}, "variant": {"v1"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
	// The stored diagnostic is served as a download of exactly that document.
	rec := e.get(t, "/settings/forge/diagnostics/probe-20261002T000000Z-0123abcd")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Header().Get("Content-Disposition"), "attachment") || !strings.Contains(rec.Body.String(), "forge-diagnostic") ||
		rec.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("diagnostic download: %d %v %s", rec.Code, rec.Header(), rec.Body.String())
	}
	n := len(fv.calls)
	for _, id := range []string{"..%2F..%2Fetc%2Fpasswd", "x", "probe-1-2", "optimize-20261002T000000Z-ffffffff.json"} {
		if rec := e.get(t, "/settings/forge/diagnostics/"+id); rec.Code != http.StatusNotFound {
			t.Errorf("%q: %d", id, rec.Code)
		}
	}
	if rec := e.get(t, "/settings/forge/diagnostics/optimize-20261002T000000Z-ffffffff"); rec.Code != http.StatusNotFound {
		t.Errorf("an unknown diagnostic: %d", rec.Code)
	}
	if len(fv.calls) != n+1 {
		t.Fatalf("malformed identities reached the authority: %v", fv.calls[n:])
	}
}

// The failure state shows the exact failing phase, where the diagnostic is and
// how to inspect it; a resumed download reports the bytes it already held and
// the real transferred bytes, and an unknown total is not turned into a percentage.
func TestForgeFailureAndResumeAreShownTruthfully(t *testing.T) {
	e := newEnv(t)
	plan := []string{"preparing", "runtime", "model", "publish"}
	fm := &fakeModels{state: ModelsState{Inventory: variantInventory(),
		Busy: &ModelOp{Kind: "materialize", Device: "cpu", Model: "clef-flash", Plan: plan, Phases: plan[:3], Phase: "model", Step: "downloading", Detail: "model-00002-of-00004.safetensors",
			Item: 2, Items: 15, Done: 5 << 30, Total: 0, Resumed: 4 << 30, Started: time.Now().Add(-time.Minute)}}}
	withModels(e, fm)
	withVariants(e, &fakeVariants{})
	body := e.get(t, "/settings").Body.String()
	busy := body[strings.Index(body, `id="models-busy"`):]
	busy = busy[strings.Index(busy, `<p class="op-line">`):]
	busy = busy[:strings.Index(busy, "</p>")+4]
	if !strings.Contains(busy, "resuming") || !strings.Contains(busy, "4.0 GiB") || !strings.Contains(busy, "5.0 GiB received") || strings.Contains(busy, "aria-valuenow") || strings.Contains(busy, "%)") {
		t.Errorf("resuming download:\n%s", busy)
	}
	fm.state.Busy = nil
	fm.state.Last = &ModelOp{Kind: "optimize", Model: "clef-flash", Plan: plan, Phases: plan[:2], Phase: "quantizing", Failure: "optimizer quantize: out of memory",
		FailurePhase: "quantizing", FailureStep: "materializing", Diagnostic: "optimize-20261002T000000Z-0123abcd", Started: time.Now().Add(-time.Minute), Finished: time.Now()}
	body = e.get(t, "/settings").Body.String()
	for _, want := range []string{`id="forge-diagnostic-last"`, `href="/settings/forge/diagnostics/optimize-20261002T000000Z-0123abcd"`,
		"hachidori forge diagnostics show optimize-20261002T000000Z-0123abcd", "hachidori forge diagnostics export -out DIR", "Failed in phase"} {
		if !strings.Contains(body, want) {
			t.Errorf("failure state lacks %q", want)
		}
	}
	// A failure without a diagnostic does not invent one.
	fm.state.Last.Diagnostic = ""
	if body = e.get(t, "/settings").Body.String(); strings.Contains(body, `id="forge-diagnostic-last"`) {
		t.Error("a diagnostic link was shown for a failure that has none")
	}
}
