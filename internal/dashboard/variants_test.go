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
