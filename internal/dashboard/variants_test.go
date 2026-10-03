package dashboard

import (
	"errors"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// hostPath names an absolute fixture path in the host's own syntax: the
// dashboard requires paths that are absolute for the platform it runs on, so
// POSIX-rooted literals are not absolute on Windows.
func hostPath(elem ...string) string {
	root := filepath.VolumeName(os.TempDir()) + string(filepath.Separator)
	return filepath.Join(append([]string{root}, elem...)...)
}

type fakeVariants struct {
	mu            sync.Mutex
	calls         []string
	buildRequests []ForgeBuildEvaluateRequest
	err           error
	diag          map[string][]byte
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
func (f *fakeVariants) BuildAndEvaluate(r ForgeBuildEvaluateRequest) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	r.Questions = append([]string(nil), r.Questions...)
	f.buildRequests = append(f.buildRequests, r)
	f.calls = append(f.calls, "build-evaluate")
	return f.err
}
func (f *fakeVariants) CertifyVariant(r CertifyRequest) error {
	return f.rec("certify " + r.Variant + " device=" + r.Device + " reference=" + r.ReferenceDevice + "/" + r.ReferenceDType + " dataset=" + r.Dataset +
		" questions=[" + strings.Join(r.Questions, ",") + "] policy=[" + r.Policy + "] materialize=" + strconv.FormatBool(r.Materialize))
}
func (f *fakeVariants) Apply(device, model, variant string, materialize bool) error {
	return f.rec("apply " + device + " " + model + " " + variant + " materialize=" + strconv.FormatBool(materialize))
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

// card is the Forge card of one variant.
func card(t *testing.T, body, id string) string {
	t.Helper()
	i := strings.Index(body, `data-variant="`+id+`" id="variant-`)
	if i < 0 {
		t.Fatalf("no Forge card for %s", id)
	}
	end := strings.Index(body[i:], "</article>")
	return body[i : i+end]
}

func has(s string, wants ...string) bool {
	for _, w := range wants {
		if !strings.Contains(s, w) {
			return false
		}
	}
	return true
}

// forgeEnv hosts the Models and Forge authorities over an inventory.
func forgeEnv(t *testing.T, inv setup.Inventory) (*env, *fakeModels, *fakeVariants) {
	e := newEnv(t)
	fm, fv := &fakeModels{state: ModelsState{Inventory: inv}}, &fakeVariants{}
	withModels(e, fm)
	withVariants(e, fv)
	return e, fm, fv
}

// Variants, their recipe and precision, certification state, lifecycle and
// the operator actions they allow are a projection of the inventory.
func TestForgeVariantsProjection(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.RestartRequired = false
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="forge-sources"`, `data-source="clef-flash"`, `data-variant="clef-flash--r--aaaaaaaaaaaa"`, "W4A16 · 4-bit g128 · compute bfloat16",
		"llmcompressor 0.14.0", `<span class="badge tone-ok">accepted</span>`, `<span class="badge tone-bad">rejected</span>`, `<span class="badge">uncertified</span>`,
		"model.safetensors: sha256 mismatch", `id="optimizer-runtime"`, "optimizer-cpu-1234",
		`id="forge-intent-form"`, `action="/forge/build-evaluate"`, `name="profile"`, `value="clef-flash-w4a16-rtn-g128"`} {
		if !strings.Contains(body, want) {
			t.Errorf("Forge lacks %q", want)
		}
	}
	// Apply only for an accepted variant; the explicit experimental launch
	// only in the advanced section; a rejected or broken one offers neither.
	if c := card(t, body, "clef-flash--r--aaaaaaaaaaaa"); !has(c, `action="/forge/apply"`, `>Apply certified variant<`, `name="device"`) || has(c, `name="experimental"`) || has(c, `action="/models/remove"`) {
		t.Errorf("accepted active variant card:\n%s", c)
	}
	if c := card(t, body, "clef-flash--r--bbbbbbbbbbbb"); has(c, `action="/forge/apply"`, `name="experimental"`) || has(c, `action="/forge/apply"`, `action="/forge/certify"`) || !has(c, "Use Build &amp; evaluate to produce a candidate with evaluation evidence.") {
		t.Errorf("uncertified variant card:\n%s", c)
	}
	if c := card(t, body, "clef-flash--r--cccccccccccc"); has(c, `action="/forge/apply"`) || !has(c, `action="/models/remove"`, `name="kind" value="variant"`) || !has(c, "Rejected: use Build &amp; evaluate to produce another candidate.") {
		t.Errorf("rejected variant card:\n%s", c)
	}
	if c := card(t, body, "clef-flash--r--dddddddddddd"); has(c, `action="/forge/apply"`, `action="/forge/certify"`, `action="/forge/probe"`) || !has(c, "Verify the variant: its artifact reports a problem.") {
		t.Errorf("broken variant card:\n%s", c)
	}
	// Every variant can be verified through the existing maintenance path.
	if n := strings.Count(body, `name="kind" value="variant"><input type="hidden" name="id"`); n < 4 {
		t.Errorf("%d variant forms", n)
	}
	// Restarting is required before an apply can be safe: the page says so
	// from the authority's flag and offers no apply form.
	fm.state.RestartRequired = true
	body = e.get(t, "/forge").Body.String()
	if !has(body, `id="restart-required"`, "Restart the runtime in Models before applying a variant.") || has(body, `action="/forge/apply"`) {
		t.Error("a pending restart does not stop the apply offer")
	}
}

// The normal intent form asks for source/profile and semantic evaluation
// inputs only: never a reference or candidate run file.
func TestForgeIntentFormHasNoResidentRunInputs(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	body := e.get(t, "/forge").Body.String()
	form := section(body, `<form method="post" action="/forge/build-evaluate"`, `</form>`)
	for _, want := range []string{`name="source"`, `name="profile"`, `name="dataset"`, `name="questions"`, `name="policy"`, `name="device"`, `name="reference_device"`, `name="reference_dtype"`, `name="provisioning"`, "Evaluation suite", "Build &amp; evaluate"} {
		if !strings.Contains(form, want) {
			t.Errorf("intent form lacks %q:\n%s", want, form)
		}
	}
	for _, banned := range []string{`name="reference_run"`, `name="candidate_run"`, `name="reference"`, `name="candidate"`, "ResidentRun", "run file"} {
		if strings.Contains(body, banned) {
			t.Errorf("the Forge page mentions %q", banned)
		}
	}
	// The same holds in every other workspace that can start a certification.
	for _, p := range []string{"/models", "/settings", "/"} {
		if b := e.get(t, p).Body.String(); has(b, `name="reference"`) || has(b, `name="candidate"`) {
			t.Errorf("%s carries run-path inputs", p)
		}
	}
}

// A variant's stages are derived from the records on every render; a stage the
// records do not show is "not yet", never a failure.
func TestLifecycleOfProjectsBackendRecords(t *testing.T) {
	built := func(r VariantRow) VariantRow { r.SourceMaterialized = true; return r }
	state := func(r VariantRow) map[string]string {
		m := map[string]string{}
		for _, st := range lifecycleOf(r) {
			m[st.Name] = st.State
		}
		return m
	}
	row := func(cert string) VariantRow {
		return built(VariantRow{VariantEntry: setup.VariantEntry{ID: "v", Certification: cert}})
	}
	for name, tc := range map[string]struct {
		r    VariantRow
		want map[string]string
		next string
	}{
		"built only": {row("uncertified"), map[string]string{"built": StageDone, "probed": StagePending, "certified": StagePending, "active": StagePending},
			"Use Build & evaluate to produce a candidate with evaluation evidence."},
		"probed": {func() VariantRow {
			r := row("uncertified")
			r.Probe = &ProbeRow{Result: "passed", Device: "cuda"}
			return r
		}(),
			map[string]string{"built": StageDone, "probed": StageDone, "certified": StagePending, "active": StagePending}, "Use Build & evaluate to produce a candidate with evaluation evidence."},
		"stale probe": {func() VariantRow {
			r := row("uncertified")
			r.Probe, r.ProbeStale = &ProbeRow{Result: "passed"}, true
			return r
		}(),
			map[string]string{"built": StageDone, "probed": StageStale, "certified": StagePending, "active": StagePending}, "Use Build & evaluate to produce a candidate with evaluation evidence."},
		"failed probe": {func() VariantRow {
			r := row("uncertified")
			r.Probe = &ProbeRow{Result: "failed", Phase: "provenance"}
			return r
		}(),
			map[string]string{"built": StageDone, "probed": StageBad, "certified": StagePending, "active": StagePending}, "Use Build & evaluate to produce a candidate with evaluation evidence."},
		"accepted": {row("accepted"), map[string]string{"built": StageDone, "probed": StagePending, "certified": StageDone, "active": StagePending}, "Next: apply the certified variant."},
		"rejected": {row("rejected"), map[string]string{"built": StageDone, "probed": StagePending, "certified": StageBad, "active": StagePending},
			"Rejected: use Build & evaluate to produce another candidate. A rejected variant is never applied."},
		"active, applies on restart": {func() VariantRow { r := row("accepted"); r.Active, r.Pending = true, true; return r }(),
			map[string]string{"built": StageDone, "probed": StagePending, "certified": StageDone, "active": StageCurrent}, "Applies on restart: restart the runtime in Models."},
		"serving": {func() VariantRow { r := row("accepted"); r.Active, r.Running = true, true; return r }(),
			map[string]string{"built": StageDone, "probed": StagePending, "certified": StageDone, "active": StageDone}, "Serving: this certified variant is the execution artifact."},
		"problem": {func() VariantRow { r := row("accepted"); r.Problem = "sha256 mismatch"; return r }(),
			map[string]string{"built": StageBad, "probed": StagePending, "certified": StageDone, "active": StagePending}, "Verify the variant: its artifact reports a problem."},
	} {
		got := state(tc.r)
		for k, w := range tc.want {
			if got[k] != w {
				t.Errorf("%s: stage %s = %q, want %q", name, k, got[k], w)
			}
		}
		if n := nextStepOf(tc.r); n != tc.next {
			t.Errorf("%s: next step %q, want %q", name, n, tc.next)
		}
		// Every label, detail and next step has a Japanese entry.
		for _, st := range lifecycleOf(tc.r) {
			for _, m := range []string{st.Label, st.Detail} {
				if !i18n.Japanese.Has(m) {
					t.Errorf("%q has no Japanese entry", m)
				}
			}
		}
		if !i18n.Japanese.Has(tc.next) {
			t.Errorf("%q has no Japanese entry", tc.next)
		}
	}
	// Not yet probed or certified is not rendered as a failure.
	e, _, _ := forgeEnv(t, variantInventory())
	c := card(t, e.get(t, "/forge").Body.String(), "clef-flash--r--bbbbbbbbbbbb")
	if !has(c, `data-stage="probed" data-state="pending"`, `data-stage="certified" data-state="pending"`, `data-stage="active" data-state="pending"`, "not yet probed", "not yet certified") ||
		has(c, `class="failed"`) {
		t.Errorf("a built-only variant shows missing stages as failures:\n%s", c)
	}
}

// The Forge lifecycle and the execution artifact are re-derived from the
// authorities on every request. A dashboard keeps nothing: a replacement
// dashboard over the same authority renders the same page, and the same
// request over changed records renders the changed state.
func TestForgeKeepsNoLifecycleStateOfItsOwn(t *testing.T) {
	e, fm, fv := forgeEnv(t, variantInventory())
	fm.state.RestartRequired = false
	id := "clef-flash--r--bbbbbbbbbbbb"
	stage := func(name string) string {
		c := card(t, e.get(t, "/forge").Body.String(), id)
		i := strings.Index(c, `data-stage="`+name+`" data-state="`)
		if i < 0 {
			t.Fatalf("no %s stage", name)
		}
		c = c[i+len(`data-stage="`+name+`" data-state="`):]
		return c[:strings.Index(c, `"`)]
	}
	if stage("certified") != StagePending || stage("active") != StagePending {
		t.Fatal("start state")
	}
	// Acting does not move any stage: only the records do.
	e.post(t, "/forge/certify", url.Values{"variant": {id}, "dataset": {hostPath("d.jsonl")}, "device": {"cuda"}})
	e.post(t, "/forge/apply", url.Values{"variant": {id}, "device": {"cuda"}})
	if stage("certified") != StagePending || stage("active") != StagePending {
		t.Fatal("a posted action changed the projected lifecycle")
	}
	fm.state.Inventory.Variants[1].Certification = "accepted"
	if stage("certified") != StageDone {
		t.Fatal("an accepted record is not projected")
	}
	fm.state.Inventory.Variants[1].Active = true
	if stage("active") != StageDone {
		t.Fatal("the activation record is not projected")
	}
	fm.state.Inventory.Variants[1].Certification, fm.state.Inventory.Variants[1].Active = "uncertified", false
	if stage("certified") != StagePending || stage("active") != StagePending {
		t.Fatal("the page remembered a previous state")
	}
	// A replaced dashboard (the desktop builds one on every runtime rebind) over
	// the same authorities and token renders the same lifecycle.
	// (The outcome banner of the last request is the one thing a dashboard
	// holds for the operator; it is not lifecycle state.)
	banner := regexp.MustCompile(`(?s)<div class="last tone-(?:ok|bad)" role="status">.*?</div>`)
	before := banner.ReplaceAllString(e.get(t, "/forge").Body.String(), "")
	cfg := e.d.cfg
	cfg.Token = e.d.token
	e.d = New(cfg)
	if after := banner.ReplaceAllString(e.get(t, "/forge").Body.String(), ""); after != before {
		t.Error("a replacement dashboard renders another Forge page")
	}
	if len(fv.calls) != 2 {
		t.Fatalf("authority calls %v", fv.calls)
	}
}

// Apply is one backend call. The dashboard never chains activation, restart,
// materialization or verification itself, and never touches the lifecycle.
func TestForgeApplyIsOneBackendAction(t *testing.T) {
	e, fm, fv := forgeEnv(t, variantInventory())
	id := "clef-flash--r--aaaaaaaaaaaa"
	rec := e.post(t, "/forge/apply", url.Values{"return": {"forge"}, "variant": {id}, "model": {"clef-flash"}, "device": {"cuda"}})
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/forge" {
		t.Fatalf("apply: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if a := e.lastAction(t); !a.OK || a.Name != "apply "+id || !strings.Contains(a.Message, "one transaction") {
		t.Fatalf("action %+v", a)
	}
	e.post(t, "/forge/apply", url.Values{"variant": {id}, "model": {"clef-flash"}, "device": {"cpu"}, "materialize": {"1"}})
	e.post(t, "/forge/apply", url.Values{"variant": {id}, "model": {"clef-flash"}, "device": {"cuda"}, "materialize": {"true"}})
	if got := fv.calls; len(got) != 3 || got[0] != "apply cuda clef-flash "+id+" materialize=false" || got[1] != "apply cpu clef-flash "+id+" materialize=true" ||
		got[2] != "apply cuda clef-flash "+id+" materialize=false" {
		t.Fatalf("apply calls %v", got)
	}
	if len(fm.calls) != 0 {
		t.Fatalf("the dashboard also called the maintenance authority: %v", fm.calls)
	}
	e.rt.mu.Lock()
	lc := append([]string(nil), e.rt.calls...)
	e.rt.mu.Unlock()
	if len(lc) != 0 {
		t.Fatalf("lifecycle touched: %v", lc)
	}
	// A refusal is the backend's and is shown as it is.
	fv.err = errors.New("variant is not certified")
	e.post(t, "/forge/apply", url.Values{"variant": {id}, "device": {"cuda"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "not certified") {
		t.Fatalf("refusal not shown: %+v", a)
	}
	// The form posts one action and carries no activate, restart or verify.
	e2, fm2, _ := forgeEnv(t, variantInventory())
	fm2.state.RestartRequired = false
	form := section(card(t, e2.get(t, "/forge").Body.String(), id), `<form method="post" action="/forge/apply"`, `</form>`)
	if !has(form, `name="variant" value="`+id+`"`, `name="device"`, `name="materialize"`) || has(form, "/activate", "/restart", "/verify", "/materialize") {
		t.Errorf("apply form:\n%s", form)
	}
	if rec := e.post(t, "/forge/apply", url.Values{"token": {"forged"}, "variant": {id}, "device": {"cuda"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
}

// Experimental activation is an advanced escape hatch, outside the certified
// workflow: it is never inside a variant's lifecycle steps and cannot be taken
// for the next step.
func TestForgeExperimentalActivationIsSeparated(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.RestartRequired = false
	body := e.get(t, "/forge").Body.String()
	adv := section(body, `id="forge-advanced"`, `id="forge-readiness-region"`)
	if !has(adv, `<details data-keep="forge-advanced">`, `name="experimental" value="1"`, "Activate as experimental", "Activate without applying", "not steps of the lifecycle above") {
		t.Fatalf("advanced section:\n%s", adv)
	}
	if strings.Contains(adv, "open>") || strings.Contains(adv, "<details open") {
		t.Error("the advanced section is open by default")
	}
	rest := strings.Replace(body, adv, "", 1)
	if has(rest, `name="experimental"`, "Activate as experimental") {
		t.Error("experimental activation is outside the advanced section")
	}
	for _, c := range []string{"clef-flash--r--aaaaaaaaaaaa", "clef-flash--r--bbbbbbbbbbbb"} {
		cd := card(t, body, c)
		if has(cd, "/forge/activate", "xperimental") && !has(cd, "certified variant") {
			t.Errorf("a variant card offers activation inside the workflow:\n%s", cd)
		}
		if has(cd, `class="btn primary">Activate`) {
			t.Errorf("a variant card has a primary Activate button:\n%s", cd)
		}
	}
	if !has(card(t, body, "clef-flash--r--bbbbbbbbbbbb"), "Use Build &amp; evaluate to produce a candidate with evaluation evidence.") {
		t.Error("the next step of an uncertified variant does not use Build & evaluate")
	}
	// An accepted variant's activation without apply is also only advanced.
	if !has(adv, `data-variant="clef-flash--r--aaaaaaaaaaaa"`) || !has(section(adv, `data-variant="clef-flash--r--aaaaaaaaaaaa"`, `</tr>`), "Activate without applying") {
		t.Error("low-level activation of a certified variant is not in the advanced section")
	}
	// A rejected or broken variant offers no activation at all.
	for _, id := range []string{"clef-flash--r--cccccccccccc", "clef-flash--r--dddddddddddd"} {
		if strings.Contains(adv, `value="`+id+`"`) {
			t.Errorf("%s is offered for activation", id)
		}
	}
}

// The execution artifact is always SOURCE or an exact VARIANT, on Models,
// Forge and Runtime, with the variant's provenance when one serves.
func TestExecutionArtifactIsAlwaysExplicit(t *testing.T) {
	e := newEnv(t)
	inv := variantInventory()
	fm := &fakeModels{state: ModelsState{Inventory: inv}}
	withModels(e, fm)
	withVariants(e, &fakeVariants{})
	// A source serves; the next start selects the same source.
	for _, p := range []string{"/models", "/forge"} {
		b := e.get(t, p).Body.String()
		run := section(b, `id="artifact-running"`, `id="artifact-next"`)
		if !has(run, `data-artifact="SOURCE"`, `<span class="badge">SOURCE</span>`, "the upstream source artifact (no variant)") || has(run, "VARIANT") {
			t.Errorf("%s running artifact:\n%s", p, run)
		}
	}
	if b := e.get(t, "/").Body.String(); !has(b, `id="runtime-source"`, "SOURCE ·") || has(b, `id="runtime-variant"`) {
		t.Error("the Runtime page does not say SOURCE")
	}
	if b := e.get(t, "/live").Body.String(); !has(b, `id="statusbar-artifact">SOURCE<`) {
		t.Error("the status bar does not say SOURCE")
	}

	// A variant serves: the exact identity, scheme, bits, certification and the
	// requested and reported device and dtype.
	id := "clef-flash--r--aaaaaaaaaaaa"
	inv.Variants[0].Active, inv.Variants[0].Experimental = true, false
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "clef-flash", Device: "cuda", Variant: id}
	fm.state.Inventory = inv
	cfg := e.d.cfg
	started := time.Now()
	rt := server.Runtime{Home: e.home, Runtime: "cu128-aaaa", ModelID: "clef-flash", Model: "Cloudflare--clef-flash/17f0b0ad", Device: "cuda",
		Variant: &server.Variant{ID: id, Recipe: "clef-flash-w4a16-rtn-g128", Scheme: "W4A16", Bits: 4, DType: "bfloat16", Format: "compressed-tensors/pack-quantized",
			Engine: "llmcompressor", EngineVersion: "0.14.0", Certification: "accepted", Source: server.VariantSource{ID: "clef-flash"}}}
	e.rt.mu.Lock()
	e.rt.snap.Info["device"], e.rt.snap.Info["dtype"] = "cuda:0", "torch.bfloat16"
	e.rt.mu.Unlock()
	cfg.Status = func() server.Status { return server.StatusBody(e.rt, rt, started) }
	e.d = New(cfg)
	for _, p := range []string{"/models", "/forge"} {
		b := e.get(t, p).Body.String()
		run := section(b, `id="artifact-running"`, `id="artifact-next"`)
		for _, want := range []string{`data-artifact="VARIANT"`, `<span class="badge tone-ok">VARIANT</span>`, `<dd class="mono">` + id + `</dd>`, "W4A16 · 4-bit · compute bfloat16",
			"<dd>accepted</dd>", `<dd class="mono">cuda</dd>`, "cuda:0 · dtype torch.bfloat16"} {
			if !strings.Contains(run, want) {
				t.Errorf("%s running variant lacks %q:\n%s", p, want, run)
			}
		}
		if has(run, "the upstream source artifact") {
			t.Errorf("%s shows a variant as the source", p)
		}
		next := section(b, `id="artifact-next"`, `</section>`)
		if !has(next, `data-artifact="VARIANT"`, id, "W4A16") {
			t.Errorf("%s next-start artifact:\n%s", p, next)
		}
	}
	if b := e.get(t, "/").Body.String(); !has(b, `id="runtime-variant"`, "VARIANT "+id+" · W4A16 · accepted") || has(b, `id="runtime-source"`) {
		t.Error("the Runtime page does not say VARIANT with its exact ID")
	}
	if b := e.get(t, "/live").Body.String(); !has(b, `id="statusbar-artifact">VARIANT `+id+`<`) {
		t.Error("the status bar does not say VARIANT")
	}

	// An experimental activation is marked as such, from the activation record.
	inv.Variants[0].Active = false
	inv.Variants[1].Active, inv.Variants[1].Experimental = true, true
	inv.Active = &home.Active{Runtime: "cu128-aaaa", ModelID: "clef-flash", Device: "cuda", Variant: "clef-flash--r--bbbbbbbbbbbb", Experimental: true}
	fm.state.Inventory = inv
	if next := section(e.get(t, "/models").Body.String(), `id="artifact-next"`, `</section>`); !has(next, `<span class="badge tone-bad">experimental/uncertified</span>`, "clef-flash--r--bbbbbbbbbbbb") {
		t.Errorf("experimental next start:\n%s", next)
	}
	// Nothing running and nothing activated is stated, not left blank.
	fm.state.Inventory.Active = nil
	if b := e.get(t, "/models").Body.String(); !has(b, `id="artifact-next" data-artifact="none"`) {
		t.Error("an empty next start is not stated")
	}
}

// Without the variant actions the Forge still lists and verifies the variants
// but offers no lifecycle action, and the routes do not exist.
func TestForgeControlsAbsentWithoutAuthority(t *testing.T) {
	e := newEnv(t)
	withModels(e, &fakeModels{state: ModelsState{Inventory: variantInventory()}})
	body := e.get(t, "/forge").Body.String()
	if !strings.Contains(body, `id="forge-variants"`) || has(body, `action="/forge/build-evaluate"`, "/forge/optimize", "/forge/probe", "/forge/certify", "/forge/apply", "/forge/activate", "/settings/variants/") {
		t.Error("variant controls rendered without the authority")
	}
	for _, p := range []string{"/forge/optimize", "/settings/variants/optimize", "/forge/apply", "/forge/certify"} {
		if rec := e.post(t, p, url.Values{"model": {"clef-flash"}}); rec.Code != http.StatusNotFound {
			t.Fatalf("%s without authority: %d", p, rec.Code)
		}
	}
	// Without the maintenance authority neither workspace exists.
	bare := newEnv(t)
	for _, p := range []string{"/models", "/forge"} {
		if rec := bare.get(t, p); rec.Code != http.StatusNotFound {
			t.Errorf("GET %s without authority: %d", p, rec.Code)
		}
	}
	if has(navRe.FindString(bare.get(t, "/").Body.String()), "/models", "/forge") {
		t.Error("navigation offers Models or Forge without the authority")
	}
}

// Variant actions are forwarded with explicit arguments, return to Forge, show
// refusals and need the form token; they never touch the lifecycle.
func TestForgeActionsForwarded(t *testing.T) {
	e, fm, fv := forgeEnv(t, variantInventory())
	dataset, shortDataset := hostPath("data", "eval.jsonl"), hostPath("d.jsonl")
	qFile, qDir, policy := hostPath("q", "a.json"), hostPath("q", "dir"), hostPath("p.json")
	for _, c := range []struct {
		op   string
		form url.Values
		want string
	}{
		{"activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v1"}}, "activate-variant cuda clef-flash v1"},
		{"activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v2"}, "experimental": {"1"}}, "activate-variant cuda clef-flash v2 experimental"},
		{"optimize", url.Values{"model": {"clef-flash"}, "recipe": {"clef-flash-w4a16-rtn-g128"}}, "optimize clef-flash clef-flash-w4a16-rtn-g128"},
		{"certify", url.Values{"variant": {"v1"}, "device": {"cuda"}, "dataset": {" " + dataset + " "}, "questions": {qFile + "\r\n\r\n " + qDir + " \n"}, "policy": {" " + policy + " "},
			"reference_device": {"cpu"}, "reference_dtype": {"float32"}, "materialize": {"1"}},
			"certify v1 device=cuda reference=cpu/float32 dataset=" + dataset + " questions=[" + qFile + "," + qDir + "] policy=[" + policy + "] materialize=true"},
		{"certify", url.Values{"variant": {"v1"}, "device": {"cpu"}, "dataset": {shortDataset}}, "certify v1 device=cpu reference=/ dataset=" + shortDataset + " questions=[] policy=[] materialize=false"},
		{"preflight", url.Values{"kind": {"optimize"}, "model": {"clef-flash"}, "recipe": {"r1"}}, "preflight optimize [clef-flash] [r1] [] []"},
		{"preflight", url.Values{"kind": {"probe"}, "variant": {"v1"}, "device": {"cuda"}}, "preflight probe [] [] [v1] [cuda]"},
		{"probe", url.Values{"variant": {"v1"}, "device": {"cpu"}}, "probe v1 cpu"},
	} {
		c.form.Set("return", "forge")
		rec := e.post(t, "/forge/"+c.op, c.form)
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/forge" {
			t.Fatalf("%s: %d %s", c.op, rec.Code, rec.Header().Get("Location"))
		}
		if a := e.lastAction(t); !a.OK {
			t.Fatalf("%s: %+v", c.op, a)
		}
		if got := fv.calls[len(fv.calls)-1]; got != c.want {
			t.Fatalf("%s forwarded %q, want %q", c.op, got, c.want)
		}
	}
	// Only explicit absolute local paths are forwarded: a relative or ".."
	// path, or a missing dataset, is refused before the authority is called.
	n0 := len(fv.calls)
	for name, form := range map[string]url.Values{
		"relative dataset":  {"variant": {"v1"}, "device": {"cuda"}, "dataset": {"eval.jsonl"}},
		"missing dataset":   {"variant": {"v1"}, "device": {"cuda"}},
		"relative question": {"variant": {"v1"}, "device": {"cuda"}, "dataset": {shortDataset}, "questions": {qFile + "\nq/b.json"}},
		"relative policy":   {"variant": {"v1"}, "device": {"cuda"}, "dataset": {shortDataset}, "policy": {"p.json"}},
	} {
		e.post(t, "/forge/certify", form)
		if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "path") {
			t.Errorf("%s: not refused: %+v", name, a)
		}
	}
	if len(fv.calls) != n0 {
		t.Fatalf("an unchecked path reached the authority: %v", fv.calls[n0:])
	}
	// A path is forwarded in its cleaned form, so it names exactly the
	// location that is shown.
	e.post(t, "/forge/certify", url.Values{"variant": {"v1"}, "device": {"cuda"}, "dataset": {hostPath("data") + string(filepath.Separator) + ".." + string(filepath.Separator) + "etc" + string(filepath.Separator) + "eval.jsonl"}})
	if got := fv.calls[len(fv.calls)-1]; !strings.Contains(got, "dataset="+hostPath("etc", "eval.jsonl")+" ") {
		t.Fatalf("path not cleaned: %q", got)
	}
	// A run-file certification is no longer a dashboard input: the old
	// fields are ignored, never forwarded.
	e.post(t, "/forge/certify", url.Values{"variant": {"v1"}, "device": {"cuda"}, "dataset": {shortDataset}, "reference": {"C:\\runs\\ref.json"}, "candidate": {"C:\\runs\\cand.json"}})
	if got := fv.calls[len(fv.calls)-1]; strings.Contains(got, "ref.json") || strings.Contains(got, "cand.json") {
		t.Fatalf("a run path reached the authority: %q", got)
	}
	// Experimental is only ever the explicit value 1.
	e.post(t, "/forge/activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v3"}, "experimental": {"true"}})
	if got := fv.calls[len(fv.calls)-1]; strings.Contains(got, "experimental") {
		t.Fatalf("a loose truthy value requested the experimental launch: %q", got)
	}
	if rec := e.post(t, "/forge/format-disk", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("unknown op: %d", rec.Code)
	}
	if rec := e.post(t, "/forge/optimize", url.Values{"token": {"forged"}, "model": {"clef-flash"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
	n := len(fv.calls)
	fv.err = errors.New("variant is not certified")
	e.post(t, "/forge/activate", url.Values{"device": {"cuda"}, "model": {"clef-flash"}, "variant": {"v1"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "not certified") || len(fv.calls) != n+1 {
		t.Fatalf("refusal not shown: %+v", a)
	}
	// Verify and remove of a variant use the existing maintenance path.
	e.post(t, "/models/verify", url.Values{"kind": {"variant"}, "id": {"v1"}})
	e.post(t, "/models/remove", url.Values{"kind": {"variant"}, "id": {"v1"}})
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

// The /settings/... routes that predate the workspaces still work: they reach
// the same handlers and show their outcome where the action now lives.
func TestSettingsCompatibilityRoutesStillWork(t *testing.T) {
	e, fm, fv := forgeEnv(t, variantInventory())
	withResidency(e, fm, &fakeResidency{})
	for _, c := range []struct {
		path string
		form url.Values
		dest string
	}{
		{"/settings/models/verify", url.Values{"kind": {"model"}, "id": {"laya-base"}}, "/models"},
		{"/settings/models/restart", nil, "/models"},
		{"/settings/residents", url.Values{"resident": {"laya-base"}}, "/models"},
		{"/settings/variants/probe", url.Values{"variant": {"v1"}, "device": {"cpu"}}, "/forge"},
		{"/settings/variants/certify", url.Values{"variant": {"v1"}, "device": {"cpu"}, "dataset": {"/d"}}, "/forge"},
		{"/models/verify", url.Values{"kind": {"model"}, "id": {"laya-base"}}, "/models"},
		{"/forge/probe", url.Values{"variant": {"v1"}, "device": {"cpu"}}, "/forge"},
		// the forms say where to come back to
		{"/forge/preflight", url.Values{"kind": {"materialize"}, "return": {"models"}}, "/models"},
		{"/models/verify", url.Values{"kind": {"variant"}, "id": {"v1"}, "return": {"forge"}}, "/forge"},
	} {
		rec := e.post(t, c.path, c.form)
		if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != c.dest {
			t.Errorf("%s: %d -> %q, want %q", c.path, rec.Code, rec.Header().Get("Location"), c.dest)
		}
	}
	fv.diag = map[string][]byte{"probe-20261002T000000Z-0123abcd": []byte(`{"schema":"hachidori.forge-diagnostic/v1"}`)}
	for _, p := range []string{"/settings/forge/diagnostics/probe-20261002T000000Z-0123abcd", "/forge/diagnostics/probe-20261002T000000Z-0123abcd"} {
		if rec := e.get(t, p); rec.Code != http.StatusOK {
			t.Errorf("GET %s: %d", p, rec.Code)
		}
	}
}

// Optimization and certification progress use the same operation view: real
// phases of the plan, an indeterminate step without a percentage, and failure
// evidence with the phase it failed in. Every label has a Japanese entry
// (TestOperationMessagesAreCatalogued covers the catalog).
func TestOptimizeOperationProgressIsTruthful(t *testing.T) {
	plan := []string{"model", "preparing", "runtime", "starting", "loading_source", "resolving", "quantizing", "serializing", "verifying", "publish"}
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Busy = &ModelOp{Kind: "optimize", Model: "clef-flash", Target: "recipe clef-flash-w4a16-rtn-g128", Plan: plan,
		Phases: plan[:7], Phase: "quantizing", Step: "materializing", Detail: "25 Linear modules", Started: time.Now().Add(-90 * time.Second)}
	for _, p := range []string{"/forge", "/models"} {
		body := e.get(t, p).Body.String()
		for _, want := range []string{`id="models-busy"`, "Quantizing", "Loading source model", "Resolving modules", "25 Linear modules"} {
			if !strings.Contains(body, want) {
				t.Errorf("%s busy view lacks %q", p, want)
			}
		}
		busy := body[strings.Index(body, `id="models-busy"`):]
		busy = busy[:strings.Index(busy, "</div>")+6]
		if strings.Contains(busy, "aria-valuenow") || strings.Contains(busy, "%)") {
			t.Errorf("a percentage was shown for an indeterminate step:\n%s", busy)
		}
	}
	fm.state.Busy = nil
	fm.state.Last = &ModelOp{Kind: "optimize", Model: "clef-flash", Plan: plan, Phases: plan[:5], Phase: "quantizing", Failure: "optimizer quantize: out of memory",
		FailurePhase: "quantizing", FailureStep: "materializing", Started: time.Now().Add(-time.Minute), Finished: time.Now()}
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="models-last"`, "optimizer quantize: out of memory", "Failed in phase <strong>Quantizing</strong>"} {
		if !strings.Contains(body, want) {
			t.Errorf("failure view lacks %q", want)
		}
	}
	_ = worker.StateReady
}

// A certification and an apply are projected from the controller's operation
// state like every other action: the same plan and phases, the failure's phase
// and, for a failed apply, its rollback outcome.
func TestCertifyAndApplyOperationsAreProjected(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	certPlan := []string{"resolving", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
	fm.state.Busy = &ModelOp{Kind: "forge_certify", Device: "cuda", Target: "variant clef-flash--r--bbbbbbbbbbbb", Plan: certPlan, Phases: certPlan[:4], Phase: "reference_run",
		Started: time.Now().Add(-20 * time.Second)}
	body := e.get(t, "/forge").Body.String()
	if !has(body, `id="models-busy"`, `aria-current="step"`) || strings.Contains(body, "aria-valuenow") {
		t.Errorf("certification in flight:\n%s", section(body, `id="models-busy"`, `</div>`))
	}
	applyPlan := []string{"validating", "snapshotting", "runtime", "model", "variant", "activation", "rebinding", "awaiting_ready", "proving", "smoke", "finalizing"}
	fm.state.Busy = nil
	fm.state.Last = &ModelOp{Kind: "apply", Device: "cuda", Target: "variant clef-flash--r--aaaaaaaaaaaa", Plan: applyPlan, Phases: applyPlan[:8], Phase: "awaiting_ready",
		Failure: "worker did not become READY; the previous serving target was restored and verified", FailurePhase: "awaiting_ready", Diagnostic: "apply-20261002T000000Z-0123abcd",
		Started: time.Now().Add(-time.Minute), Finished: time.Now()}
	body = e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="models-last"`, "the previous serving target was restored and verified", "Failed in phase", `href="/forge/diagnostics/apply-20261002T000000Z-0123abcd"`} {
		if !strings.Contains(body, want) {
			t.Errorf("failed apply lacks %q", want)
		}
	}
	for _, p := range append(append([]string{}, certPlan...), applyPlan...) {
		if got := phaseLabel(p); got == "" || !i18n.Japanese.Has(got) {
			t.Errorf("phase %q label %q has no Japanese entry", p, got)
		}
	}
	for _, p := range []string{"resolve_inputs", "provision", "build", "resolving"} {
		label := opPhaseLabel("forge_build_evaluate", p)
		if label == "" || !i18n.Japanese.Has(label) {
			t.Errorf("Build & evaluate phase %q label %q has no Japanese entry", p, label)
		}
	}
}

func TestForgeBuildEvaluateProgressUsesBackendPhases(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	plan := []string{"resolve_inputs", "preflight", "build", "resolving", "preflight", "probe", "reference_run", "candidate_run", "aligning", "certifying", "persisting"}
	fm.state.Busy = &ModelOp{Kind: "forge_build_evaluate", Model: "clef-flash", Target: "source clef-flash recipe clef-flash-w4a16-rtn-g128",
		Plan: plan, Phases: plan[:4], Phase: "resolving", Started: time.Now().Add(-time.Minute)}
	fm.state.Forge.Resolution = &ForgeResolution{Source: "clef-flash", Recipe: "clef-flash-w4a16-rtn-g128",
		CandidateDevice: ForgeResolvedValue{Mode: "Auto", Value: "cuda"}, ReferenceDevice: ForgeResolvedValue{Mode: "Auto", Value: "cpu"},
		ReferenceDType: ForgeResolvedValue{Mode: "Auto", Value: "bfloat16"}}
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="models-busy"`, "Resolving build intent", "Building candidate", "Resolving evaluation inputs", `aria-current="step"`, `id="forge-resolution"`, `Auto → <span class="mono">cuda</span>`} {
		if !strings.Contains(body, want) {
			t.Errorf("Build & evaluate progress lacks %q", want)
		}
	}
}

func TestForgeBuildEvaluateUsesOneBackendOperation(t *testing.T) {
	e, _, fv := forgeEnv(t, variantInventory())
	dataset, shortDataset := hostPath("data", "eval.jsonl"), hostPath("d.jsonl")
	qFile, qDir, policy := hostPath("defs", "a.json"), hostPath("defs", "questions"), hostPath("policy", "p.json")
	form := url.Values{
		"return": {"forge"}, "source": {"clef-flash"}, "profile": {"clef-flash-w4a16-rtn-g128"},
		"dataset": {" " + dataset + " "}, "questions": {qFile + "\r\n " + qDir + " "}, "policy": {" " + policy + " "},
		"reference_device": {""}, "reference_dtype": {""}, "provisioning": {"auto"},
	}
	rec := e.post(t, "/forge/build-evaluate", form)
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/forge" {
		t.Fatalf("Build & evaluate: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	if a := e.lastAction(t); !a.OK || a.Name != "build and evaluate clef-flash" {
		t.Fatalf("action %+v", a)
	}
	if len(fv.calls) != 1 || fv.calls[0] != "build-evaluate" || len(fv.buildRequests) != 1 {
		t.Fatalf("Build & evaluate made %d backend calls: %v", len(fv.calls), fv.calls)
	}
	r := fv.buildRequests[0]
	if r.Source != "clef-flash" || r.Profile != "clef-flash-w4a16-rtn-g128" || r.Dataset != dataset ||
		strings.Join(r.Questions, ",") != qFile+","+qDir || r.Policy != policy || !r.Materialize ||
		r.Device != "" || r.ReferenceDevice != "" || r.ReferenceDType != "" {
		t.Fatalf("semantic request %+v", r)
	}

	// Overrides are forwarded exactly and provisioning can be explicitly disabled.
	e.post(t, "/forge/build-evaluate", url.Values{
		"return": {"forge"}, "source": {"clef-flash"}, "profile": {"clef-flash-w4a16-rtn-g128"},
		"dataset": {dataset}, "device": {"cuda"}, "reference_device": {"cpu"}, "reference_dtype": {"float32"}, "provisioning": {"never"},
	})
	if len(fv.calls) != 2 || len(fv.buildRequests) != 2 {
		t.Fatalf("override operation count calls=%v requests=%d", fv.calls, len(fv.buildRequests))
	}
	r = fv.buildRequests[1]
	if r.Device != "cuda" || r.ReferenceDevice != "cpu" || r.ReferenceDType != "float32" || r.Materialize {
		t.Fatalf("override request %+v", r)
	}
	if strings.Contains(strings.Join(fv.calls, " "), "apply") {
		t.Fatalf("Build & evaluate applied a candidate: %v", fv.calls)
	}

	// Dataset, Question Definition and policy inputs must be absolute before the
	// composed authority is called; the UI exposes no ResidentRun paths.
	n := len(fv.calls)
	for name, bad := range map[string]url.Values{
		"relative dataset":  {"source": {"clef-flash"}, "profile": {"clef-flash-w4a16-rtn-g128"}, "dataset": {"eval.jsonl"}, "provisioning": {"auto"}},
		"relative question": {"source": {"clef-flash"}, "profile": {"clef-flash-w4a16-rtn-g128"}, "dataset": {shortDataset}, "questions": {"q.json"}, "provisioning": {"auto"}},
		"relative policy":   {"source": {"clef-flash"}, "profile": {"clef-flash-w4a16-rtn-g128"}, "dataset": {shortDataset}, "policy": {"p.json"}, "provisioning": {"auto"}},
	} {
		e.post(t, "/forge/build-evaluate", bad)
		if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "path") {
			t.Errorf("%s was not refused: %+v", name, a)
		}
	}
	if len(fv.calls) != n {
		t.Fatalf("invalid path reached the backend: %v", fv.calls[n:])
	}
}

func TestForgeBuildEvaluateResolutionIsVisible(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Forge.Resolution = &ForgeResolution{
		Source: "clef-flash", Recipe: "clef-flash-w4a16-rtn-g128", Variant: "clef-flash--r--eeeeeeeeeeee",
		CandidateDevice: ForgeResolvedValue{Mode: "Auto", Value: "cuda"},
		ReferenceDevice: ForgeResolvedValue{Mode: "Override", Value: "cpu"},
		ReferenceDType:  ForgeResolvedValue{Mode: "Auto", Value: "bfloat16"},
		CandidateDType:  "bfloat16",
	}
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="forge-resolution"`, `Auto → <span class="mono">cuda</span>`, `Override → <span class="mono">cpu</span>`, `Auto → <span class="mono">bfloat16</span>`} {
		if !strings.Contains(body, want) {
			t.Errorf("resolved plan lacks %q", want)
		}
	}
}

func TestForgeIntentPickerKeepsResourcesAndBrowserFallback(t *testing.T) {
	e, _, _ := forgeEnv(t, variantInventory())
	body := e.get(t, "/forge").Body.String()
	if has(body, `action="/forge/pick"`) || !has(body, `name="dataset"`, "absolute path") {
		t.Error("browser Forge lacks path fallback or exposes native picker controls")
	}
	p := &fakePathPicker{path: "/chosen/resource.jsonl"}
	withPathPicker(e, p)
	form := url.Values{
		"source": {"clef-flash"}, "profile": {"clef-flash-w4a16-rtn-g128"}, "dataset": {"/typed/dataset.jsonl"},
		"questions": {"/defs/old.json"}, "policy": {"/policy/old.json"}, "device": {"cuda"}, "reference_device": {"cpu"}, "reference_dtype": {"float32"}, "provisioning": {"never"},
	}
	postPick := func(kind string) string {
		v := url.Values{"pick": {kind}}
		for k, values := range form {
			v[k] = values
		}
		return e.post(t, "/forge/pick", v).Body.String()
	}
	body = postPick("dataset")
	for _, want := range []string{`name="dataset" value="/chosen/resource.jsonl"`, `name="source" required`, `value="cuda" selected`, `value="never" selected`} {
		if !strings.Contains(body, want) {
			t.Errorf("dataset picker lost intent field %q", want)
		}
	}
	body = postPick("question-file")
	if !strings.Contains(body, "/defs/old.json\n/chosen/resource.jsonl</textarea>") {
		t.Error("Question Definition picker did not append the selected path")
	}
	body = postPick("question-folder")
	if p.calls[len(p.calls)-1] != "folder" || !strings.Contains(body, "/defs/old.json\n/chosen/resource.jsonl</textarea>") {
		t.Error("Question Definition folder picker did not reuse the native folder dialog")
	}
	body = postPick("policy")
	if !strings.Contains(body, `name="policy" value="/chosen/resource.jsonl"`) {
		t.Error("policy picker lost the selected path")
	}
}

// Forge readiness is a projection of the recorded reports: blockers, warnings
// and unknowns are shown as such, an unknown is never turned into READY, the
// probe of each variant is shown with its outcome (and marked stale when it was
// recorded for another manifest), and nothing is derived from text.
func TestForgeReadinessProjection(t *testing.T) {
	inv := variantInventory()
	inv.Variants[0].ManifestSHA256, inv.Variants[1].ManifestSHA256 = "sha-a", "sha-b"
	inv.Models[len(inv.Models)-1].PartialBytes = 3 << 30
	inv.Models[len(inv.Models)-1].Materialized = false
	e, fm, _ := forgeEnv(t, inv)
	fm.state.Forge = ForgeState{
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
		Diagnostics: []DiagnosticRow{
			{ID: "probe-20261002T000000Z-0123abcd", Kind: "probe", Phase: "probing", Model: "clef-flash", Variant: "clef-flash--r--aaaaaaaaaaaa", Created: "2026-10-02T00:00:00Z", Error: "boom"},
			{ID: "optimize-20261002T000000Z-0123abcd", Kind: "optimize", Phase: "quantizing", Model: "clef-flash", Created: "2026-10-02T00:00:00Z", Error: "out of memory"},
		},
	}
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="forge-readiness"`, `data-outcome="blocked"`, `data-outcome="attention"`, "storage.capacity", "memory.fit", "BLOCKED", "ATTENTION",
		"Not READY: this is not a promise", `id="forge-diagnostics"`, `href="/forge/diagnostics/optimize-20261002T000000Z-0123abcd"`} {
		if !strings.Contains(body, want) {
			t.Errorf("Forge lacks %q", want)
		}
	}
	// A report that belongs to a variant is on that variant's card, not twice.
	global := section(body, `id="forge-readiness"`, `</section>`)
	if strings.Contains(global, "accelerator.vram") || strings.Contains(global, "probe-20261002T000000Z-0123abcd") || !strings.Contains(global, "storage.capacity") {
		t.Errorf("global readiness region:\n%s", global)
	}
	if c := card(t, body, "clef-flash--r--aaaaaaaaaaaa"); !has(c, "accelerator.vram", `href="/forge/diagnostics/probe-20261002T000000Z-0123abcd"`, "Failure diagnostics", "hachidori certify show clef-flash--r--aaaaaaaaaaaa") {
		t.Errorf("the variant's reports and diagnostics are not reachable from its card:\n%s", c)
	}
	if strings.Contains(global, ">READY<") {
		t.Error("a report with an unknown or a blocker was shown as READY")
	}
	stage := func(id, name string) string {
		c := card(t, body, id)
		return section(c, `data-stage="`+name+`"`, `</li>`)
	}
	if r := stage("clef-flash--r--aaaaaaaaaaaa", "probed"); !has(r, `data-state="done"`, "cuda") {
		t.Errorf("probed variant:\n%s", r)
	}
	if r := stage("clef-flash--r--bbbbbbbbbbbb", "probed"); !has(r, `data-state="stale"`, "another manifest") {
		t.Errorf("stale probe:\n%s", r)
	}
	if r := stage("clef-flash--r--cccccccccccc", "probed"); !has(r, `data-state="bad"`, "the probe failed", "provenance") {
		t.Errorf("failed probe:\n%s", r)
	}
	if r := stage("clef-flash--r--dddddddddddd", "probed"); !has(r, `data-state="pending"`, "not yet probed") {
		t.Errorf("unprobed variant:\n%s", r)
	}
	// Probe is not certification: a passing probe never offers Apply.
	if c := card(t, body, "clef-flash--r--bbbbbbbbbbbb"); has(c, `action="/forge/apply"`) {
		t.Errorf("a probe made a variant appliable:\n%s", c)
	}
	// Mechanical probe/certification controls are no longer part of candidate
	// cards; the composed operation owns those phases.
	for _, id := range []string{"clef-flash--r--aaaaaaaaaaaa", "clef-flash--r--dddddddddddd"} {
		if c := card(t, body, id); has(c, `action="/forge/probe"`, `action="/forge/certify"`, `formaction="/forge/preflight"`) {
			t.Errorf("candidate %s exposes internal execution controls:\n%s", id, c)
		}
	}
	if mb := e.get(t, "/models").Body.String(); !has(mb, "interrupted download kept for resume", "3.0 GiB") {
		t.Error("the partial download is not shown in Models")
	}
}

func TestForgeActionsForwardedAndDiagnosticDownload(t *testing.T) {
	e, _, fv := forgeEnv(t, variantInventory())
	fv.diag = map[string][]byte{"probe-20261002T000000Z-0123abcd": []byte(`{"schema":"hachidori.forge-diagnostic/v1"}`)}
	if rec := e.post(t, "/forge/probe", url.Values{"token": {"forged"}, "variant": {"v1"}}); rec.Code != http.StatusForbidden {
		t.Fatalf("forged token: %d", rec.Code)
	}
	// The stored diagnostic is served as a download of exactly that document.
	rec := e.get(t, "/forge/diagnostics/probe-20261002T000000Z-0123abcd")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Header().Get("Content-Disposition"), "attachment") || !strings.Contains(rec.Body.String(), "forge-diagnostic") ||
		rec.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("diagnostic download: %d %v %s", rec.Code, rec.Header(), rec.Body.String())
	}
	n := len(fv.calls)
	for _, id := range []string{"..%2F..%2Fetc%2Fpasswd", "x", "probe-1-2", "optimize-20261002T000000Z-ffffffff.json"} {
		if rec := e.get(t, "/forge/diagnostics/"+id); rec.Code != http.StatusNotFound {
			t.Errorf("%q: %d", id, rec.Code)
		}
	}
	if rec := e.get(t, "/forge/diagnostics/optimize-20261002T000000Z-ffffffff"); rec.Code != http.StatusNotFound {
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
	e, fm, _ := forgeEnv(t, variantInventory())
	plan := []string{"preparing", "runtime", "model", "publish"}
	fm.state.Busy = &ModelOp{Kind: "materialize", Device: "cpu", Model: "clef-flash", Plan: plan, Phases: plan[:3], Phase: "model", Step: "downloading", Detail: "model-00002-of-00004.safetensors",
		Item: 2, Items: 15, Done: 5 << 30, Total: 0, Resumed: 4 << 30, Started: time.Now().Add(-time.Minute)}
	body := e.get(t, "/models").Body.String()
	busy := body[strings.Index(body, `id="models-busy"`):]
	busy = busy[strings.Index(busy, `<p class="op-line">`):]
	busy = busy[:strings.Index(busy, "</p>")+4]
	if !strings.Contains(busy, "resuming") || !strings.Contains(busy, "4.0 GiB") || !strings.Contains(busy, "5.0 GiB received") || strings.Contains(busy, "aria-valuenow") || strings.Contains(busy, "%)") {
		t.Errorf("resuming download:\n%s", busy)
	}
	fm.state.Busy = nil
	fm.state.Last = &ModelOp{Kind: "optimize", Model: "clef-flash", Plan: plan, Phases: plan[:2], Phase: "quantizing", Failure: "optimizer quantize: out of memory",
		FailurePhase: "quantizing", FailureStep: "materializing", Diagnostic: "optimize-20261002T000000Z-0123abcd", Started: time.Now().Add(-time.Minute), Finished: time.Now()}
	for _, p := range []string{"/models", "/forge"} {
		body = e.get(t, p).Body.String()
		for _, want := range []string{`id="forge-diagnostic-last"`, `href="/forge/diagnostics/optimize-20261002T000000Z-0123abcd"`,
			"hachidori forge diagnostics show optimize-20261002T000000Z-0123abcd", "hachidori forge diagnostics export -out DIR", "Failed in phase"} {
			if !strings.Contains(body, want) {
				t.Errorf("%s failure state lacks %q", p, want)
			}
		}
	}
	// A failure without a diagnostic does not invent one.
	fm.state.Last.Diagnostic = ""
	if body = e.get(t, "/forge").Body.String(); strings.Contains(body, `id="forge-diagnostic-last"`) {
		t.Error("a diagnostic link was shown for a failure that has none")
	}
}
