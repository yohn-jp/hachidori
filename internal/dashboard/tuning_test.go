package dashboard

import (
	"encoding/json"
	"errors"
	"html"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// fakeTuning is a tuning authority over the real internal/tuning analysis,
// profile validation and compiler, with an in-memory store.
type fakeTuning struct {
	mu          sync.Mutex
	analysis    tuning.Analysis
	analysisErr error
	profiles    map[string]tuning.Profile
	order       []string // newest last
	impact      TuningImpact
	impactErr   error
}

func (f *fakeTuning) Analysis(source string) (tuning.Analysis, error) {
	if f.analysisErr != nil {
		return tuning.Analysis{}, f.analysisErr
	}
	if source != f.analysis.Source.ID {
		return tuning.Analysis{}, errors.New("no analysis for " + source)
	}
	return f.analysis, nil
}

func (f *fakeTuning) Profiles(source string) ([]tuning.Profile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []tuning.Profile
	for i := len(f.order) - 1; i >= 0; i-- {
		if p := f.profiles[f.order[i]]; p.Source.ID == source {
			out = append(out, p)
		}
	}
	return out, nil
}

func (f *fakeTuning) LoadProfile(id string) (tuning.Profile, tuning.Analysis, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p, ok := f.profiles[id]
	if !ok {
		return tuning.Profile{}, tuning.Analysis{}, errors.New("no such profile")
	}
	return p, f.analysis, nil
}

func (f *fakeTuning) SaveProfile(p tuning.Profile, a tuning.Analysis) error {
	if _, err := tuning.Compile(p, a); err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.profiles == nil {
		f.profiles = map[string]tuning.Profile{}
	}
	if _, ok := f.profiles[p.ID()]; !ok {
		f.profiles[p.ID()] = p
		f.order = append(f.order, p.ID())
	}
	return nil
}

func (f *fakeTuning) Impact(tuning.Profile, tuning.Analysis, tuning.Compilation) (TuningImpact, error) {
	return f.impact, f.impactErr
}

func tuningAnalysis(t testing.TB) tuning.Analysis {
	t.Helper()
	source, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	analysis, err := tuning.Analyze(source, tuning.DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text",
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes:    []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{
			"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj",
		},
		CarriedFiles: []string{"joint_head.safetensors"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return analysis
}

func withTuning(e *env, ft Tuning) {
	cfg := e.d.cfg
	cfg.Tuning = ft
	e.d = New(cfg)
}

func tuningEnv(t *testing.T) (*env, *fakeTuning, *fakeVariants) {
	t.Helper()
	e, _, fv := forgeEnv(t, variantInventory())
	ft := &fakeTuning{analysis: tuningAnalysis(t)}
	withTuning(e, ft)
	return e, ft, fv
}

// expectedProfile is the profile the form of a test pins, built independently
// of the handler from the backend analysis.
func expectedProfile(t *testing.T, a tuning.Analysis, objective string, pinned ...string) tuning.Profile {
	t.Helper()
	p, err := tuning.NewDefaultProfile(a, objective)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range pinned {
		p.Preservation[id] = tuning.PreservationChoice{Mode: tuning.PreservationPinned, Precision: tuning.PreservedPrecision}
	}
	return p
}

// workspace is the Tuning page content between the shell's header and footer.
func workspace(body string) string {
	i := strings.Index(body, `id="tuning-workspace"`)
	if i < 0 {
		return ""
	}
	return body[i : i+strings.Index(body[i:], "</main>")]
}

func tuningRow(t *testing.T, body, region string) string {
	t.Helper()
	i := strings.Index(body, `data-region="`+region+`"`)
	if i < 0 {
		t.Fatalf("no row for region %s", region)
	}
	return body[i : i+strings.Index(body[i:], "</tr>")]
}

func impactRow(t *testing.T, body, key string) string {
	t.Helper()
	i := strings.Index(body, `data-impact="`+key+`"`)
	if i < 0 {
		t.Fatalf("no impact row %s", key)
	}
	end := strings.Index(body[i:], "</tr>")
	if dd := strings.Index(body[i:], "</dd>"); dd >= 0 && (end < 0 || dd < end) {
		end = dd
	}
	return body[i : i+end]
}

// regionEvidence is the Evidence line of one region: its member counts and
// what the canonical policy does with it.
func regionEvidence(t *testing.T, body, region string) string {
	t.Helper()
	i := strings.Index(body, `data-evidence-region-counts="`+region+`"`)
	if i < 0 {
		t.Fatalf("no Evidence line for region %s", region)
	}
	return body[i : i+strings.Index(body[i:], "</li>")]
}

func TestTuningIsAFirstClassWorkspaceInTheNavigation(t *testing.T) {
	e, _, _ := tuningEnv(t)
	withSettings(e, &fakeSettings{}, nil)
	for _, p := range []string{"/", "/models", "/forge", "/tuning", "/workbench", "/experiments", "/errors", "/diagnostics", "/settings"} {
		rec := e.get(t, p)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s: %d", p, rec.Code)
		}
		nav := navRe.FindString(rec.Body.String())
		at := -1
		for _, want := range []string{`>Runtime</a>`, `href="/models"`, `href="/forge"`, `href="/tuning"`, `href="/workbench"`} {
			i := strings.Index(nav, want)
			if i <= at {
				t.Errorf("%s: navigation lacks %q in order:\n%s", p, want, nav)
			}
			at = i
		}
		if n := strings.Count(nav, `aria-current="page"`); n != 1 {
			t.Errorf("%s marks %d current pages", p, n)
		}
	}
	body := e.get(t, "/tuning").Body.String()
	if !strings.Contains(navRe.FindString(body), `<a href="/tuning" aria-current="page">Tuning</a>`) ||
		!strings.Contains(body, "<title>Tuning · Hachidori</title>") || strings.Count(body, "<h1>") != 1 {
		t.Error("Tuning is not the marked current page with one title and heading")
	}
}

func TestTuningExistsOnlyWhereItsAuthorityAndModelsAreHosted(t *testing.T) {
	e := newEnv(t)
	withModels(e, &fakeModels{state: ModelsState{Inventory: variantInventory()}})
	if rec := e.get(t, "/tuning"); rec.Code != http.StatusNotFound {
		t.Errorf("Tuning without its authority: %d", rec.Code)
	}
	if strings.Contains(navRe.FindString(e.get(t, "/").Body.String()), "/tuning") {
		t.Error("navigation offers Tuning without its authority")
	}
	// Existing workspaces are unchanged without it.
	for _, p := range []string{"/models", "/forge"} {
		if rec := e.get(t, p); rec.Code != http.StatusOK {
			t.Errorf("GET %s: %d", p, rec.Code)
		}
	}
	e = newEnv(t)
	withTuning(e, &fakeTuning{analysis: tuningAnalysis(t)})
	if rec := e.get(t, "/tuning"); rec.Code != http.StatusNotFound {
		t.Errorf("Tuning without Models: %d", rec.Code)
	}
}

func TestTuningInitialStateIsTruthfulAutoNotCheckedAndUnsaved(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	// exact source and analysis identity; the profile is shown as unsaved
	for _, want := range []string{"clef-flash", "Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6", tuning.ClefAnalyzerVersion,
		`data-tuning-analysis="` + ft.analysis.SHA256() + `"`, tuning.RecipeCompilerVersion, tuning.ProfileSchema, "Hachidori&#39;s analyzed default; it is saved when you build.", `data-unsaved="true"`} {
		if !strings.Contains(body, want) {
			t.Errorf("identity lacks %q", want)
		}
	}
	initial := expectedProfile(t, ft.analysis, "balanced")
	if !strings.Contains(body, `data-tuning-profile="`+initial.ID()+`"`) {
		t.Error("the initial profile does not show the identity it would be saved under")
	}
	// every region is Auto and nothing is invented as evidence
	for _, r := range ft.analysis.Regions {
		row := tuningRow(t, body, r.ID)
		if !strings.Contains(row, `<option value="auto" selected>`) || strings.Contains(row, `<option value="pinned" selected>`) {
			t.Errorf("region %s is not Auto: %s", r.ID, row)
		}
	}
	for _, key := range []string{"size", "memory", "latency", "fidelity"} {
		row := impactRow(t, body, key)
		if !strings.Contains(row, `data-state="NOT_CHECKED"`) || !strings.Contains(row, "No valid evidence") || strings.Contains(row, `data-state="MEASURED"`) {
			t.Errorf("%s without evidence is not NOT_CHECKED: %s", key, row)
		}
	}
	if len(ft.profiles) != 0 {
		t.Error("rendering saved or built something")
	}
}

func TestTuningRegionsComeFromBackendAnalysisAndRawMappingsAreEvidenceOnly(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	for _, r := range ft.analysis.Regions {
		row := tuningRow(t, body, r.ID)
		if r.ID == tuning.RegionJointSchemaHead && !strings.Contains(regionEvidence(t, body, r.ID), "1 files") {
			t.Errorf("file count missing: %s", row)
		}
		if r.ID == tuning.RegionVision && !strings.Contains(regionEvidence(t, body, r.ID), "2 modules") {
			t.Errorf("module count missing: %s", row)
		}
		if !strings.Contains(row, `name="region.`+r.ID+`"`) {
			t.Errorf("region %s has no control", r.ID)
		}
	}
	if n := strings.Count(body, "data-region="); n != len(ft.analysis.Regions) {
		t.Errorf("%d region rows, analysis has %d", n, len(ft.analysis.Regions))
	}
	// The canonical policy is stated per region: the gates stay preserved
	// under Auto, the backbone projections are quantized.
	if row := tuningRow(t, body, tuning.RegionLinearAttentionDecay); !strings.Contains(row, `data-effective="PRESERVED"`) || !strings.Contains(regionEvidence(t, body, tuning.RegionLinearAttentionDecay), "already preserves") {
		t.Errorf("decay gate under Auto: %s", row)
	}
	if row := tuningRow(t, body, tuning.RegionFeedForward); !strings.Contains(row, `data-effective="AUTO"`) || !strings.Contains(regionEvidence(t, body, tuning.RegionFeedForward), "quantizes") {
		t.Errorf("feed-forward under Auto: %s", row)
	}
	// Raw module names, recipe patterns and the recipe digest exist only in
	// the Evidence disclosure, after every operator control and impact value.
	ev := strings.Index(body, `data-disclosure="evidence"`)
	if ev < 0 {
		t.Fatal("no Evidence disclosure")
	}
	normal := body[:ev]
	for _, raw := range []string{"model.language_model.layers", "model.visual.blocks", `re:.*linear_attn`, `re:model\.visual`, "lm_head", "in_proj_a"} {
		if strings.Contains(normal, raw) {
			t.Errorf("the normal UI exposes the raw mapping %q", raw)
		}
	}
	evidence := body[ev:]
	for _, want := range []string{"model.language_model.layers.0.linear_attn.in_proj_a", "model.visual.blocks.0.attn.qkv", "re:.*linear_attn", "clef-flash-w4a16-rtn-g128", "Recipe SHA-256", `data-evidence-region="` + tuning.RegionFeedForward + `"`} {
		if !strings.Contains(evidence, want) {
			t.Errorf("Evidence lacks %q", want)
		}
	}
	// No control accepts a pattern: the only editable region input is the
	// choice between Auto and Preserved.
	if strings.Contains(normal, `name="pattern"`) || strings.Contains(normal, `name="regex"`) || strings.Contains(normal, `type="text"`) {
		t.Error("a free-form module/pattern control is offered")
	}
}

func TestTuningSavesAVersionedProfileWithoutBuilding(t *testing.T) {
	e, ft, fv := tuningEnv(t)
	form := url.Values{"source": {"clef-flash"}, "objective": {"maximum-fidelity"},
		"region." + tuning.RegionFeedForward: {"pinned"}, "region." + tuning.RegionFullAttention: {"pinned"}, "region." + tuning.RegionVision: {"auto"}}
	rec := e.post(t, "/tuning/save", form)
	want := expectedProfile(t, ft.analysis, "maximum-fidelity", tuning.RegionFeedForward, tuning.RegionFullAttention)
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/tuning?profile="+want.ID()+"&source=clef-flash" {
		t.Fatalf("save: %d %q", rec.Code, rec.Header().Get("Location"))
	}
	saved, ok := ft.profiles[want.ID()]
	if !ok || string(saved.Canonical()) != string(want.Canonical()) {
		t.Fatalf("saved profile differs from the described one: %+v", saved)
	}
	if saved.Schema != tuning.ProfileSchema || saved.CompilerVersion != tuning.RecipeCompilerVersion || saved.AnalysisSHA256 != ft.analysis.SHA256() {
		t.Errorf("profile is not versioned and bound: %+v", saved)
	}
	if len(fv.calls) != 0 {
		t.Errorf("saving ran a Forge action: %v", fv.calls)
	}

	body := e.get(t, rec.Header().Get("Location")).Body.String()
	for _, want := range []string{`data-tuning-profile="` + want.ID() + `"`, "Saved profile", `<option value="maximum-fidelity" selected>Maximum fidelity</option>`} {
		if !strings.Contains(body, want) {
			t.Errorf("saved profile page lacks %q", want)
		}
	}
	if strings.Contains(body, `data-unsaved="true"`) {
		t.Error("a saved profile is shown as unsaved")
	}
	for _, region := range []string{tuning.RegionFeedForward, tuning.RegionFullAttention} {
		row := tuningRow(t, body, region)
		if !strings.Contains(row, `<option value="pinned" selected>`) || !strings.Contains(row, `data-effective="PRESERVED"`) || !strings.Contains(regionEvidence(t, body, region), "Pinned by this profile") {
			t.Errorf("%s is not pinned/preserved: %s", region, row)
		}
	}
	if row := tuningRow(t, body, tuning.RegionLinearAttention); !strings.Contains(row, `<option value="auto" selected>`) {
		t.Errorf("an unpinned region is not Auto: %s", row)
	}
	// without a profile in the URL the newest saved profile of the source is shown
	if latest := e.get(t, "/tuning").Body.String(); !strings.Contains(latest, `data-tuning-profile="`+want.ID()+`"`) {
		t.Error("the saved profile is not the one shown on return")
	}
	if !strings.Contains(body, `data-profile="`+want.ID()+`"`) || !strings.Contains(body, `id="tuning-profiles"`) {
		t.Error("saved profile is not listed")
	}
	// A different intent is a distinct profile identity; the first is kept.
	e.post(t, "/tuning/save", url.Values{"source": {"clef-flash"}, "objective": {"balanced"}, "region." + tuning.RegionFeedForward: {"pinned"}})
	if len(ft.profiles) != 2 {
		t.Errorf("a changed profile did not create a new version: %d", len(ft.profiles))
	}
}

func TestTuningRefusesIntentThatIsNotAnObjectiveAndRegionChoice(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	for name, form := range map[string]url.Values{
		"unknown objective": {"source": {"clef-flash"}, "objective": {"fastest"}},
		"unknown region":    {"source": {"clef-flash"}, "objective": {"balanced"}, "region.model.language_model.layers.0.mlp.gate_proj": {"pinned"}},
		"raw pattern":       {"source": {"clef-flash"}, "objective": {"balanced"}, "region." + `re:.*`: {"pinned"}},
		"unknown level":     {"source": {"clef-flash"}, "objective": {"balanced"}, "region." + tuning.RegionFeedForward: {"quantize-more"}},
		"no source":         {"objective": {"balanced"}},
	} {
		rec := e.post(t, "/tuning/save", form)
		if rec.Code != http.StatusSeeOther {
			t.Errorf("%s: %d", name, rec.Code)
		}
		if a := e.lastAction(t); a.OK {
			t.Errorf("%s was accepted: %+v", name, a)
		}
		rec = e.post(t, "/tuning/build", form)
		if a := e.lastAction(t); a.OK || rec.Code != http.StatusSeeOther {
			t.Errorf("%s: build was accepted: %+v", name, a)
		}
	}
	if len(ft.profiles) != 0 {
		t.Errorf("a refused intent was saved: %d", len(ft.profiles))
	}
	if rec := e.post(t, "/tuning/save", url.Values{"token": {"forged"}, "source": {"clef-flash"}, "objective": {"balanced"}}); rec.Code != http.StatusForbidden {
		t.Errorf("forged token: %d", rec.Code)
	}
}

func TestTuningHandsTheExactSavedProfileToForgesOneOperation(t *testing.T) {
	e, ft, fv := tuningEnv(t)
	form := url.Values{"source": {"clef-flash"}, "objective": {"minimum-size"}, "region." + tuning.RegionOutputEmbeddings: {"pinned"}, "region." + tuning.RegionFeedForward: {"pinned"}}
	rec := e.post(t, "/tuning/build", form)
	want := expectedProfile(t, ft.analysis, "minimum-size", tuning.RegionOutputEmbeddings, tuning.RegionFeedForward)
	loc := "/forge?profile=tuning%3A" + want.ID() + "&source=clef-flash#forge-intent-form"
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != loc {
		t.Fatalf("handoff: %d %q", rec.Code, rec.Header().Get("Location"))
	}
	if _, ok := ft.profiles[want.ID()]; !ok {
		t.Fatal("the profile was not saved before the handoff")
	}
	// Tuning starts nothing: no build-only path and no Forge action.
	if len(fv.calls) != 0 || len(fv.buildRequests) != 0 {
		t.Errorf("Tuning started a build itself: %v", fv.calls)
	}
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, want.ID()) || !strings.Contains(a.Message, "nothing was built") {
		t.Errorf("handoff outcome: %+v", a)
	}
	// Forge shows exactly that saved profile selected for its one operation.
	body := e.get(t, strings.TrimSuffix(loc, "#forge-intent-form")).Body.String()
	if !strings.Contains(body, `<option value="tuning:`+want.ID()+`" selected>`) {
		t.Error("Forge does not preselect the handed profile")
	}
	e.post(t, "/forge/build-evaluate", url.Values{"source": {"clef-flash"}, "profile": {"tuning:" + want.ID()}, "dataset": {hostPath("data", "eval.jsonl")}, "provisioning": {"auto"}})
	if len(fv.buildRequests) != 1 || fv.buildRequests[0].TuningProfile != want.ID() {
		t.Fatalf("Forge build/evaluate request %+v", fv.buildRequests)
	}
	// A handoff naming a profile Forge does not offer selects nothing.
	if body := e.get(t, "/forge?source=clef-flash&profile=tuning%3A"+strings.Repeat("a", 64)).Body.String(); strings.Contains(body, "tuning:"+strings.Repeat("a", 64)) {
		t.Error("Forge selected an unknown profile")
	}
}

func TestTuningPageOffersNoOptimizerLifecycleAndRunsNoClientCode(t *testing.T) {
	e, _, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	main := workspace(body)
	for _, banned := range []string{"/forge/optimize", "/forge/preflight", "/forge/probe", "/forge/certify", "/forge/apply", "/forge/build-evaluate", "/models/materialize", "<script", "fetch(", "XMLHttpRequest"} {
		if strings.Contains(main, banned) {
			t.Errorf("Tuning offers %q", banned)
		}
	}
	actions := regexp.MustCompile(`(?:form)?action="([^"]+)"`).FindAllStringSubmatch(main, -1)
	for _, a := range actions {
		if a[1] != "/tuning" && a[1] != "/tuning/save" && a[1] != "/tuning/build" && a[1] != "/tuning/budget" {
			t.Errorf("Tuning posts to %s", a[1])
		}
	}
	if !strings.Contains(main, `formaction="/tuning/build"`) || !strings.Contains(main, ">Continue in Forge<") || strings.Contains(main, ">Build candidate<") {
		t.Error("the primary action is not the handoff to Forge")
	}
}

func TestTuningImpactNeverConflatesMeasuredEstimatedAndNotChecked(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	ft.impact = TuningImpact{
		Size:     ImpactValue{State: Measured, Value: "5.1 GiB", Basis: "variant clef-flash--r--aaaaaaaaaaaa artifact files"},
		Memory:   ImpactValue{State: Estimated, Value: "about 5.4 GiB", Basis: "weights-only estimate from source tensor sizes"},
		Latency:  ImpactValue{State: Measured, Value: "", Basis: "probe"},              // no value: cannot be measured
		Fidelity: ImpactValue{State: Measured, Value: "0.40% choice flips", Basis: ""}, // no evidence named: cannot be measured
	}
	body := e.get(t, "/tuning").Body.String()
	size, memory := impactRow(t, body, "size"), impactRow(t, body, "memory")
	if !strings.Contains(size, `data-state="MEASURED"`) || !strings.Contains(size, "5.1 GiB") || !strings.Contains(size, "artifact files") || !strings.Contains(size, `tone-ok`) {
		t.Errorf("measured size: %s", size)
	}
	if !strings.Contains(memory, `data-state="ESTIMATED"`) || !strings.Contains(memory, "about 5.4 GiB") || strings.Contains(memory, `MEASURED`) || strings.Contains(memory, `tone-ok`) || !strings.Contains(memory, "tone-warn") {
		t.Errorf("estimated memory is styled or worded as measured: %s", memory)
	}
	for _, key := range []string{"latency", "fidelity"} {
		row := impactRow(t, body, key)
		if !strings.Contains(row, `data-state="NOT_CHECKED"`) || strings.Contains(row, "choice flips") || strings.Contains(row, "tone-ok") {
			t.Errorf("%s with incomplete evidence is not NOT_CHECKED: %s", key, row)
		}
	}
	// an unknown state or an unreadable evidence store is never a number
	ft.impact = TuningImpact{Size: ImpactValue{State: "VERIFIED", Value: "1 GiB", Basis: "x"}}
	if row := impactRow(t, e.get(t, "/tuning").Body.String(), "size"); !strings.Contains(row, `data-state="NOT_CHECKED"`) || strings.Contains(row, "1 GiB") {
		t.Errorf("unknown state: %s", row)
	}
	ft.impact, ft.impactErr = TuningImpact{Size: ImpactValue{State: Measured, Value: "5 GiB", Basis: "x"}}, errors.New("evidence store unreadable")
	body = e.get(t, "/tuning").Body.String()
	for _, key := range []string{"size", "memory", "latency", "fidelity"} {
		if row := impactRow(t, body, key); !strings.Contains(row, `data-state="NOT_CHECKED"`) || !strings.Contains(row, "evidence store unreadable") || strings.Contains(row, "5 GiB") {
			t.Errorf("%s after an evidence error: %s", key, row)
		}
	}
}

func TestTuningUnavailableAnalysisIsNotCheckedAndOffersNothingToBuild(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	ft.analysisErr = errors.New("model clef-flash is not materialized")
	rec := e.get(t, "/tuning")
	body := rec.Body.String()
	if rec.Code != http.StatusOK || !strings.Contains(body, `id="tuning-unavailable"`) || !strings.Contains(body, "Semantic analysis is unavailable: model clef-flash is not materialized") {
		t.Fatalf("unavailable analysis: %d", rec.Code)
	}
	for _, banned := range []string{`id="tuning-form"`, `data-region=`, `/tuning/build`, `id="tuning-impact"`, `data-tuning-profile`} {
		if strings.Contains(body, banned) {
			t.Errorf("an unanalyzed source shows %q", banned)
		}
	}
	// a stale or foreign profile identity is refused, not displayed
	ft.analysisErr = nil
	body = e.get(t, "/tuning?source=clef-flash&profile="+strings.Repeat("a", 64)).Body.String()
	if !strings.Contains(body, "cannot be used") || !strings.Contains(body, `data-unsaved="true"`) {
		t.Error("an unknown profile identity is not refused")
	}
	if body := e.get(t, "/tuning?source=laya-base").Body.String(); !strings.Contains(body, "cannot be tuned") || strings.Contains(body, `id="tuning-form"`) {
		t.Error("a model without semantic regions is offered tuning")
	}
}

func TestTuningResponsiveAccessibleAndLocalized(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	fs := &fakeSettings{}
	withSettings(e, fs, nil)
	ft.impact = TuningImpact{Size: ImpactValue{State: Estimated, Value: "about 5 GiB", Basis: "estimator"}}
	en := e.get(t, "/tuning").Body.String()
	// responsive: tables scroll in their own container, the shared narrow-width rules are present
	if strings.Count(en, `<div class="table-wrap"><table class="data"`) < 3 || !strings.Contains(en, "@media (max-width: 42rem)") {
		t.Error("tables are not contained for narrow widths")
	}
	// accessible: every region control is labelled, tables have captions and row headers, the error is an alert
	for _, r := range ft.analysis.Regions {
		if !strings.Contains(en, `<label for="region-`+r.ID+`">`) || !strings.Contains(en, `id="region-`+r.ID+`"`) {
			t.Errorf("region %s control is not labelled", r.ID)
		}
	}
	if strings.Count(en, "<caption") < 3 || !strings.Contains(en, `<th scope="row" class="strong"><label`) || !strings.Contains(en, `<th scope="col">Preservation</th>`) {
		t.Error("tables lack captions or headers")
	}
	if w := workspace(en); strings.Contains(w, "tabindex=") || strings.Contains(w, `onclick=`) || strings.Contains(w, "<script") {
		t.Error("Tuning adds custom interaction instead of native controls")
	}

	if rec := e.post(t, "/settings/locale", url.Values{"locale": {"ja"}, "return": {"settings"}}); rec.Code != http.StatusSeeOther {
		t.Fatalf("select ja: %d", rec.Code)
	}
	ja := e.get(t, "/tuning").Body.String()
	for _, want := range []string{`<html lang="ja">`, `<h1>チューニング</h1>`, "Forge で続ける", "プロファイルのみを保存", "意味的リージョン", "推定", "未確認", "保持", "最大の忠実度", `<a href="/tuning" aria-current="page">チューニング</a>`} {
		if !strings.Contains(ja, want) {
			t.Errorf("Japanese Tuning lacks %q", want)
		}
	}
	// machine identities, forms and routes do not change with the language
	if a, b := machineAttrRe.FindAllString(en, -1), machineAttrRe.FindAllString(ja, -1); strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Error("forms, identifiers or routes differ between locales")
	}
	for _, id := range []string{ft.analysis.SHA256(), "model.language_model.layers.0.linear_attn.in_proj_a", "clef-flash-w4a16-rtn-g128"} {
		if !strings.Contains(ja, id) {
			t.Errorf("Japanese Tuning lacks machine identity %q", id)
		}
	}
}

// Every operator-facing message Tuning composes in Go has a Japanese entry.
func TestTuningCopyIsInTheJapaneseCatalog(t *testing.T) {
	for id, c := range regionCopy {
		for _, m := range []string{c.Label, c.Purpose} {
			if !i18n.Japanese.Has(m) {
				t.Errorf("region %s: %q has no Japanese entry", id, m)
			}
		}
	}
	for _, o := range TuningObjectives {
		if !i18n.Japanese.Has(objectiveLabel(o)) {
			t.Errorf("objective %s has no Japanese entry", o)
		}
	}
	for _, m := range []string{"Model size", "Memory (VRAM/RAM)", "Latency", "Fidelity", "Generated mappings and recipe",
		"Pinned by this profile at source precision.", "Auto: the canonical policy already preserves this region.", "Auto: the canonical policy quantizes this region."} {
		if !i18n.Japanese.Has(m) {
			t.Errorf("%q has no Japanese entry", m)
		}
	}
	b, err := pageFS.ReadFile("tuning.html")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, m := range catalogRefRe.FindAllStringSubmatch(string(b), -1) {
		msg, err := strconv.Unquote(`"` + m[1] + `"`)
		if err != nil {
			t.Fatal(err)
		}
		n++
		if !i18n.Japanese.Has(msg) {
			t.Errorf("tuning.html: %q has no Japanese entry", msg)
		}
	}
	if n < 50 {
		t.Errorf("only %d catalogued Tuning messages", n)
	}
	// Every region the backend can name has operator copy.
	a := tuningAnalysis(t)
	for _, r := range a.Regions {
		if _, ok := regionCopy[r.ID]; !ok {
			t.Errorf("analysis region %s has no operator label", r.ID)
		}
	}
}

func TestTuningSourceSelectionUsesTheModelsInventory(t *testing.T) {
	e, _, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	if !strings.Contains(body, `<option value="clef-flash" selected>clef-flash</option>`) || strings.Contains(body, `value="laya-base"`) {
		t.Error("sources are not the System One models of the inventory")
	}
}

// ---- Experiment/Evidence context handed to Tuning (#188) ----

// feedbackAnalysis has three linear-attention projection modules against one
// full-attention module, so the smallest unpreserved supported region is
// unique: full-attention-projections.
func feedbackAnalysis(t testing.TB) tuning.Analysis {
	t.Helper()
	source, err := setup.LookupModel(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	analysis, err := tuning.Analyze(source, tuning.DeclaredLayout{
		ModelType: "qwen3_5", TextModelType: "qwen3_5_text",
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		LayerTypes:    []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearModules: []string{
			"lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
			"model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
			"model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_z",
			"model.language_model.layers.0.linear_attn.out_proj", "model.language_model.layers.0.mlp.gate_proj",
			"model.language_model.layers.3.self_attn.q_proj",
		},
		CarriedFiles: []string{"joint_head.safetensors"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return analysis
}

// writeTunedVariant stores a sealed variant manifest in the dashboard's home
// whose provenance names the profile (or none, when tuned is false).
func writeTunedVariant(t *testing.T, root string, p tuning.Profile, a tuning.Analysis, tuned bool) string {
	t.Helper()
	c, err := tuning.Compile(p, a)
	if err != nil {
		t.Fatal(err)
	}
	v := home.VariantManifest{Source: p.Source, Provider: p.Source.Provider,
		Optimizer: home.Optimizer{Engine: c.Recipe.Engine, Version: "0.14.0", Runtime: "optimizer-cpu-x", Device: "cpu"}, Recipe: c.Recipe,
		Weights:  home.WeightPrecision{Scheme: "W4A16", Bits: 4, GroupSize: 128, Symmetric: true, Format: "compressed-tensors/pack-quantized", DType: "bfloat16"},
		Files:    map[string]string{"model.safetensors": strings.Repeat("4", 64)},
		Creation: home.Creation{CreatedAt: "2026-01-01T00:00:00Z", Platform: "linux/amd64", Command: "hachidori variant optimize"}}
	if tuned {
		v.Tuning = &home.TuningProvenance{Schema: home.TuningProvenanceSchema, Source: p.Source, ProfileID: p.ID(), ProfileSHA256: p.ID(),
			AnalysisID: a.ID(), AnalysisSHA256: a.SHA256(), CompilerVersion: tuning.RecipeCompilerVersion}
	}
	v.Seal()
	h := home.Home{Root: root}
	if err := os.MkdirAll(h.VariantDir(p.Source.ID, v.ID), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := home.WriteJSON(filepath.Join(h.VariantDir(p.Source.ID, v.ID), home.VariantManifestFile), v); err != nil {
		t.Fatal(err)
	}
	return v.ID
}

// evidenceOf is a strictly valid hachidori.evidence.v1 report of one served
// model/variant: n observations of question q1, correct of them right.
func evidenceOf(model, revision, variant, dataset string, n, correct int) eval.Report {
	prov := map[string]any{"model_revision": revision}
	if variant != "" {
		prov["variant_id"] = variant
	}
	r := eval.Report{Schema: eval.EvidenceSchema, Endpoint: "http://127.0.0.1:7843", Dataset: "d.jsonl", DatasetSHA256: dataset,
		StartedAt: "2026-01-01T00:00:00Z", ServedConsistent: true, Cases: n, Observations: n, Errors: []eval.RequestError{}, Passes: 1,
		ChoiceAccuracy: float64(correct) / float64(n),
		Served:         &eval.Served{StatusSchema: "hachidori.v1", Runtime: map[string]any{"model_id": model}, Provider: prov, Digest: "id-" + model + variant},
		PerQuestion:    map[string]eval.QuestionStats{"q1": {N: n, Accuracy: float64(correct) / float64(n), MeanConfidence: 0.9}}}
	for i := 0; i < n; i++ {
		choice := "yes"
		if i >= correct {
			choice = "no"
		}
		r.Results = append(r.Results, eval.Observation{CaseID: "c" + strconv.Itoa(i), QuestionID: "q1", QuestionSHA256: strings.Repeat("5", 64),
			Expected: "yes", Choice: choice, Confidence: 0.9, Probabilities: map[string]float64{"yes": 0.5, "no": 0.5}, Correct: choice == "yes"})
	}
	return r
}

type feedback struct {
	e       *env
	ft      *fakeTuning
	profile tuning.Profile
	variant string // candidate variant built from profile
	dataset string
}

func (f *feedback) save(t *testing.T, r eval.Report, label string) string {
	t.Helper()
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	sm, err := f.e.d.hist.Save(append(b, '\n'), label, "")
	if err != nil {
		t.Fatal(err)
	}
	return sm.ID
}

func feedbackEnv(t *testing.T) *feedback {
	t.Helper()
	e, ft, _ := tuningEnv(t)
	cfg := e.d.cfg
	cfg.HistoryDir = filepath.Join(e.home, "state", "history")
	e.d = New(cfg)
	ft.analysis = feedbackAnalysis(t)
	p := expectedProfile(t, ft.analysis, "balanced")
	if err := ft.SaveProfile(p, ft.analysis); err != nil {
		t.Fatal(err)
	}
	return &feedback{e: e, ft: ft, profile: p, variant: writeTunedVariant(t, e.home, p, ft.analysis, true), dataset: strings.Repeat("d", 64)}
}

func (f *feedback) model() (string, string) { return f.profile.Source.ID, f.profile.Source.Revision }

// pair stores a source baseline and a candidate of the profile's variant.
func (f *feedback) pair(t *testing.T, candidateCorrect int) (base, cand string) {
	t.Helper()
	m, rev := f.model()
	base = f.save(t, evidenceOf(m, rev, "", f.dataset, 20, 19), "source")
	cand = f.save(t, evidenceOf(m, rev, f.variant, f.dataset, 20, candidateCorrect), "candidate")
	return base, cand
}

var handoffRe = regexp.MustCompile(`href="(/tuning\?[^"]+)" data-tuning-handoff="true"`)

func (f *feedback) compare(t *testing.T, a, b string) string {
	t.Helper()
	return html.UnescapeString(f.e.post(t, "/history/compare", url.Values{"a": {a}, "b": {b}}).Body.String())
}

// handoff is the Tuning link of Experiments' comparison.
func (f *feedback) handoff(t *testing.T, a, b string) string {
	t.Helper()
	body := f.compare(t, a, b)
	m := handoffRe.FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("comparison offers no Tuning handoff:\n%s", body)
	}
	return m[1]
}

func (f *feedback) page(t *testing.T, link string) string {
	t.Helper()
	rec := f.e.get(t, link)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET %s: %d", link, rec.Code)
	}
	return html.UnescapeString(workspace(rec.Body.String()))
}

// historyBytes is every stored evidence and meta file: evidence must be
// immutable.
func historyBytes(t *testing.T, e *env) string {
	t.Helper()
	var s string
	root := filepath.Join(e.home, "state", "history", "entries")
	dirs, _ := os.ReadDir(root)
	for _, d := range dirs {
		files, _ := os.ReadDir(filepath.Join(root, d.Name()))
		for _, f := range files {
			b, _ := os.ReadFile(filepath.Join(root, d.Name(), f.Name()))
			s += d.Name() + "/" + f.Name() + "\n" + string(b)
		}
	}
	if s == "" {
		t.Fatal("no stored evidence")
	}
	return s
}

func TestTuningReceivesTheExactExperimentContext(t *testing.T) {
	f := feedbackEnv(t)
	base, cand := f.pair(t, 15)
	link := f.handoff(t, base, cand)
	body := f.page(t, link)
	br, _, _ := f.e.d.hist.Open(base)
	cr, _, _ := f.e.d.hist.Open(cand)
	_, baseSHA, _ := f.e.d.hist.Open(base)
	_, candSHA, _ := f.e.d.hist.Open(cand)
	m, rev := f.model()
	for _, want := range []string{`data-context="bound"`, m + "@" + rev, `data-context-candidate="` + f.variant + `"`, `data-context-profile="` + f.profile.ID() + `"`,
		`data-context-dataset="` + f.dataset + `"`, `data-context-questions="` + eval.QuestionIdentitiesSHA256(cr) + `"`,
		`data-context-baseline="` + baseSHA + `"`, `data-context-evidence="` + candSHA + `"`} {
		if !strings.Contains(body, want) {
			t.Errorf("exact context lacks %q", want)
		}
	}
	if eval.QuestionIdentitiesSHA256(br) != eval.QuestionIdentitiesSHA256(cr) {
		t.Fatal("fixture questions differ")
	}
	// The page shows that exact profile, not whichever one is newest.
	if !strings.Contains(body, `data-tuning-profile="`+f.profile.ID()+`"`) {
		t.Error("the page does not show the profile of the candidate")
	}
	// The measured regression is projected beside the preservation controls.
	reg := body[strings.Index(body, `id="tuning-regression-table"`):]
	reg = reg[:strings.Index(reg, "</table>")]
	for _, want := range []string{`data-regression="all"`, "0.9500", "0.7500", "20 cases", `data-regression="q1"`} {
		if !strings.Contains(reg, want) {
			t.Errorf("regression lacks %q:\n%s", want, reg)
		}
	}
	if strings.Index(body, `id="tuning-regression-table"`) > strings.Index(body, `id="tuning-region-table"`) {
		t.Error("the regression is not projected before the preservation controls")
	}
}

func TestTuningRecommendationStatesEvidenceBasisAndRegionChange(t *testing.T) {
	f := feedbackEnv(t)
	base, cand := f.pair(t, 15)
	body := f.page(t, f.handoff(t, base, cand))
	i := strings.Index(body, `id="tuning-recommendation"`)
	if i < 0 {
		t.Fatalf("no recommendation:\n%s", body)
	}
	rec := body[i : i+strings.Index(body[i:], `id="tuning-accept-form"`)]
	for _, want := range []string{`data-recommendation="full-attention-projections"`, "auto → pinned", "Evidence basis", "0.9500 to 0.7500 over 20 cases",
		"Question q1", f.variant, f.dataset[:12], "does not show that this region caused it"} {
		if !strings.Contains(rec, want) {
			t.Errorf("recommendation lacks %q:\n%s", want, rec)
		}
	}
	if strings.Contains(strings.ToLower(rec), "confidence") {
		t.Error("the recommendation invents a confidence")
	}
	if row := tuningRow(t, body, tuning.RegionFullAttention); !strings.Contains(row, `data-recommended="true"`) {
		t.Errorf("the recommended region is not marked beside its control:\n%s", row)
	}
	if row := tuningRow(t, body, tuning.RegionFeedForward); strings.Contains(row, "data-recommended") {
		t.Error("an unrelated region is marked recommended")
	}
}

func TestTuningRefusesAndSeparatesIncompatibleContext(t *testing.T) {
	f := feedbackEnv(t)
	m, rev := f.model()
	base, cand := f.pair(t, 15)

	// Experiments does not hand over pairs that are not like for like.
	otherDS := f.save(t, evidenceOf(m, rev, f.variant, strings.Repeat("e", 64), 20, 15), "other dataset")
	otherModel := f.save(t, evidenceOf("laya-base", rev, f.variant, f.dataset, 20, 15), "other model")
	otherRev := f.save(t, evidenceOf(m, "otherrev", f.variant, f.dataset, 20, 15), "other revision")
	variantBase := f.save(t, evidenceOf(m, rev, f.variant, f.dataset, 20, 19), "variant as baseline")
	untuned := writeTunedVariant(t, f.e.home, f.profile, f.ft.analysis, false)
	noProvenance := f.save(t, evidenceOf(m, rev, untuned, f.dataset, 20, 15), "variant without a profile")
	for name, pair := range map[string][2]string{"dataset": {base, otherDS}, "model": {base, otherModel}, "revision": {base, otherRev},
		"baseline is a variant": {variantBase, cand}, "candidate is the source": {cand, base}, "no profile provenance": {base, noProvenance}} {
		body := f.compare(t, pair[0], pair[1])
		if handoffRe.MatchString(body) || !strings.Contains(body, `data-tuning-handoff-refused="true"`) {
			t.Errorf("%s: the pair was offered to Tuning or not explained:\n%s", name, body)
		}
	}

	// A link whose identity does not match the stored evidence is refused,
	// and the page is the plain profile with no regression or recommendation.
	link := f.handoff(t, base, cand)
	u, _ := url.Parse(link)
	tamper := map[string]string{"profile": strings.Repeat("0", 64), "cand_sha": strings.Repeat("0", 64), "base_sha": strings.Repeat("0", 64),
		"ds": strings.Repeat("0", 64), "qs": strings.Repeat("0", 64), "variant": "clef-flash--r--ffffffffffff", "source": "laya-base", "base": cand}
	for key, value := range tamper {
		q := u.Query()
		q.Set(key, value)
		body := f.page(t, "/tuning?"+q.Encode())
		if !strings.Contains(body, `id="tuning-context-refused"`) || strings.Contains(body, `id="tuning-recommendation"`) ||
			strings.Contains(body, `id="tuning-regression-table"`) || strings.Contains(body, `data-context="bound"`) {
			t.Errorf("tampered %s was not refused:\n%s", key, body)
		}
	}
	// Candidate-less and unknown evidence ids are refused too.
	for _, q := range []string{"cand=20260101T000000Z-aaaaaaaaaaaa&source=clef-flash", "cand=../x"} {
		if body := f.page(t, "/tuning?"+q); !strings.Contains(body, `id="tuning-context-refused"`) {
			t.Errorf("%s: not refused", q)
		}
	}
}

func TestTuningNoRecommendationIsANormalState(t *testing.T) {
	f := feedbackEnv(t)
	m, rev := f.model()
	base := f.save(t, evidenceOf(m, rev, "", f.dataset, 20, 19), "source")
	same := f.save(t, evidenceOf(m, rev, f.variant, f.dataset, 20, 19), "candidate, no regression")
	body := f.page(t, f.handoff(t, base, same))
	for _, want := range []string{`data-context="bound"`, `id="tuning-no-recommendation"`, "No recommendation.", "no accuracy regression"} {
		if !strings.Contains(body, want) {
			t.Errorf("no-regression context lacks %q", want)
		}
	}
	if strings.Contains(body, `id="tuning-recommendation"`) || strings.Contains(body, `id="tuning-regression-table"`) || strings.Contains(body, "data-recommended") {
		t.Error("a recommendation or regression was projected without a measured regression")
	}
	// A single report is a context but not a comparison.
	_, cand := f.pair(t, 15)
	f.e.post(t, "/history/open", url.Values{"id": {cand}})
	ev := html.UnescapeString(f.e.get(t, "/errors").Body.String())
	m2 := handoffRe.FindStringSubmatch(ev)
	if m2 == nil {
		t.Fatal("Evidence offers no Tuning handoff for a stored variant report")
	}
	one := f.page(t, m2[1])
	if !strings.Contains(one, `data-context="bound"`) || !strings.Contains(one, `data-context-candidate="`+f.variant+`"`) ||
		!strings.Contains(one, "needs a compatible baseline") || strings.Contains(one, `id="tuning-recommendation"`) {
		t.Errorf("single-report context:\n%s", one)
	}
	// Unsupported: when the supported regions are already pinned, nothing is recommended.
	pinned := expectedProfile(t, f.ft.analysis, "balanced", tuning.RegionFullAttention, tuning.RegionLinearAttention)
	if err := f.ft.SaveProfile(pinned, f.ft.analysis); err != nil {
		t.Fatal(err)
	}
	v2 := writeTunedVariant(t, f.e.home, pinned, f.ft.analysis, true)
	cand2 := f.save(t, evidenceOf(m, rev, v2, f.dataset, 20, 15), "candidate of the pinned profile")
	got := f.page(t, f.handoff(t, base, cand2))
	if !strings.Contains(got, "already applied") || strings.Contains(got, `id="tuning-recommendation"`) {
		t.Errorf("pinned profile still received a recommendation:\n%s", got)
	}
}

func TestTuningAcceptanceIsExplicitAndCreatesADistinctProfile(t *testing.T) {
	f := feedbackEnv(t)
	base, cand := f.pair(t, 15)
	link := f.handoff(t, base, cand)
	u, _ := url.Parse(link)
	evidenceBefore := historyBytes(t, f.e)

	// Viewing the context changes nothing.
	f.page(t, link)
	if len(f.ft.order) != 1 {
		t.Fatalf("viewing saved a profile: %v", f.ft.order)
	}
	form := func(region string) url.Values {
		v := url.Values{"accept_recommendation": {region}}
		for k := range u.Query() {
			v.Set(k, u.Query().Get(k))
		}
		return v
	}

	// A recommendation other than the one shown, or a stale identity, creates nothing.
	if rec := f.e.post(t, "/tuning/save", form(tuning.RegionFeedForward)); rec.Code != http.StatusSeeOther || len(f.ft.order) != 1 {
		t.Fatalf("a different region was accepted: %d %v", rec.Code, f.ft.order)
	}
	stale := form(tuning.RegionFullAttention)
	stale.Set("cand_sha", strings.Repeat("0", 64))
	f.e.post(t, "/tuning/save", stale)
	if len(f.ft.order) != 1 {
		t.Fatal("a stale evidence identity was accepted")
	}
	noToken := form(tuning.RegionFullAttention)
	noToken.Set("token", "")
	if rec := f.e.post(t, "/tuning/save", noToken); rec.Code != http.StatusForbidden || len(f.ft.order) != 1 {
		t.Fatalf("acceptance without the form token: %d", rec.Code)
	}

	// The explicit action saves exactly the new profile and goes to it.
	rec := f.e.post(t, "/tuning/save", form(tuning.RegionFullAttention))
	if rec.Code != http.StatusSeeOther {
		t.Fatalf("accept: %d %s", rec.Code, rec.Body.String())
	}
	if len(f.ft.order) != 2 {
		t.Fatalf("saved profiles %v", f.ft.order)
	}
	next := f.ft.profiles[f.ft.order[1]]
	want := expectedProfile(t, f.ft.analysis, "balanced", tuning.RegionFullAttention)
	if next.ID() == f.profile.ID() || next.ID() != want.ID() {
		t.Fatalf("accepted profile %s, want %s distinct from %s", next.ID(), want.ID(), f.profile.ID())
	}
	if loc := rec.Header().Get("Location"); loc != tuningLocation(f.profile.Source.ID, next.ID()) {
		t.Errorf("redirect %q", loc)
	}
	if got := f.ft.profiles[f.profile.ID()]; got.ID() != f.profile.ID() || got.Preservation[tuning.RegionFullAttention].Mode != tuning.PreservationAuto {
		t.Error("the original profile changed")
	}
	// Evidence is byte-for-byte immutable across viewing, refusals and acceptance.
	if historyBytes(t, f.e) != evidenceBefore {
		t.Error("stored evidence changed")
	}
	shown := f.page(t, rec.Header().Get("Location"))
	if !strings.Contains(shown, `data-tuning-profile="`+next.ID()+`"`) || !strings.Contains(tuningRow(t, shown, tuning.RegionFullAttention), `data-effective="PRESERVED"`) {
		t.Errorf("the new profile is not shown pinned:\n%s", shown)
	}
	// Accepting the same recommendation again names the same distinct profile:
	// the store holds it once.
	f.e.post(t, "/tuning/save", form(tuning.RegionFullAttention))
	if len(f.ft.order) != 2 {
		t.Errorf("repeat acceptance produced %d profiles", len(f.ft.order))
	}
}

// Existing deep links, filters and history behaviour keep working without a context.
func TestTuningWithoutContextAndExistingDeepLinksAreUnchanged(t *testing.T) {
	f := feedbackEnv(t)
	for _, link := range []string{"/tuning", "/tuning?source=clef-flash", "/tuning?source=clef-flash&profile=" + f.profile.ID()} {
		body := f.page(t, link)
		if strings.Contains(body, `id="tuning-context"`) || !strings.Contains(body, `id="tuning-regions"`) {
			t.Errorf("%s changed without a context", link)
		}
	}
	base, cand := f.pair(t, 15)
	// A comparison that is refused for tuning is still a normal comparison.
	if body := f.compare(t, cand, base); !strings.Contains(body, `data-compare="compatible"`) {
		t.Errorf("comparison changed:\n%s", body)
	}
	f.e.post(t, "/history/open", url.Values{"id": {cand}})
	if rec := f.e.get(t, "/errors?outcome=wrong&th=0.9&sort=confidence&desc=1"); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Failures") {
		t.Errorf("Evidence filters: %d", rec.Code)
	}
}

var candidateCompareRe = regexp.MustCompile(`href="(/tuning\?[^"]+)" data-tuning-candidate-compare="true"`)

// candidates is a second saved profile with its own tuned variant, beside the
// feedback fixture's profile and variant.
type candidates struct {
	*feedback
	profileB tuning.Profile
	variantB string
}

func candidateEnv(t *testing.T) *candidates {
	t.Helper()
	f := feedbackEnv(t)
	pb := expectedProfile(t, f.ft.analysis, "maximum-fidelity", tuning.RegionFullAttention)
	if err := f.ft.SaveProfile(pb, f.ft.analysis); err != nil {
		t.Fatal(err)
	}
	return &candidates{feedback: f, profileB: pb, variantB: writeTunedVariant(t, f.e.home, pb, f.ft.analysis, true)}
}

// withLatency records request latency samples in evidence.
func withLatency(r eval.Report, p50, p95, mean float64) eval.Report {
	r.RequestLatency = eval.Latency{N: r.Observations, P50: p50, P95: p95, Mean: mean}
	return r
}

// pair stores evidence of candidate A (the fixture's variant) and of candidate B.
func (c *candidates) pair(t *testing.T, aCorrect, bCorrect int) (a, b string) {
	t.Helper()
	m, rev := c.model()
	a = c.save(t, withLatency(evidenceOf(m, rev, c.variant, c.dataset, 20, aCorrect), 10, 20, 12), "candidate A")
	b = c.save(t, withLatency(evidenceOf(m, rev, c.variantB, c.dataset, 20, bCorrect), 12.5, 19, 13), "candidate B")
	return a, b
}

func (c *candidates) link(t *testing.T, a, b string) string {
	t.Helper()
	body := c.compare(t, a, b)
	m := candidateCompareRe.FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("comparison offers no candidate comparison in Tuning:\n%s", body)
	}
	if handoffRe.MatchString(body) {
		t.Error("a candidate pair is also offered as a source-baseline handoff")
	}
	return m[1]
}

func compareRow(t *testing.T, body, attr, key string) string {
	t.Helper()
	i := strings.Index(body, attr+`="`+key+`"`)
	if i < 0 {
		t.Fatalf("no %s row %s", attr, key)
	}
	return body[i : i+strings.Index(body[i:], "</tr>")]
}

func TestTuningComparesTwoTunedCandidatesWithExactIdentities(t *testing.T) {
	c := candidateEnv(t)
	a, b := c.pair(t, 15, 18)
	before := historyBytes(t, c.e)
	body := c.page(t, c.link(t, a, b))
	ra, aSHA, _ := c.e.d.hist.Open(a)
	_, bSHA, _ := c.e.d.hist.Open(b)
	m, rev := c.model()
	for _, want := range []string{`data-compare-candidates="bound"`, m + "@" + rev,
		`data-compare-variant-a="` + c.variant + `"`, `data-compare-variant-b="` + c.variantB + `"`,
		`data-compare-profile-a="` + c.profile.ID() + `"`, `data-compare-profile-b="` + c.profileB.ID() + `"`,
		`data-compare-evidence-a="` + aSHA + `"`, `data-compare-evidence-b="` + bSHA + `"`,
		`data-compare-dataset="` + c.dataset + `"`, `data-compare-questions="` + eval.QuestionIdentitiesSHA256(ra) + `"`,
		`data-tuning-profile="` + c.profileB.ID() + `"`} {
		if !strings.Contains(body, want) {
			t.Errorf("exact comparison lacks %q", want)
		}
	}

	// The deterministic semantic profile difference.
	for _, want := range []string{`data-delta="objective">Balanced → Maximum fidelity`, `data-delta="analysis">unchanged`, `data-delta="compiler">unchanged`} {
		if !strings.Contains(body, want) {
			t.Errorf("profile difference lacks %q", want)
		}
	}
	row := compareRow(t, body, "data-delta-region", tuning.RegionFullAttention)
	if !strings.Contains(row, "Auto") || !strings.Contains(row, "Preserved") {
		t.Errorf("region difference row: %s", row)
	}
	if strings.Contains(body, `data-delta-region="`+tuning.RegionLinearAttention+`"`) {
		t.Error("an unchanged region is listed as a difference")
	}

	// Measured quality, identity-aligned question slice and resource deltas.
	if row := compareRow(t, body, "data-compare-quality", "accuracy"); !strings.Contains(row, `data-state="MEASURED"`) ||
		!strings.Contains(row, "0.7500") || !strings.Contains(row, "0.9000") || !strings.Contains(row, "+0.1500") {
		t.Errorf("accuracy row: %s", row)
	}
	if row := compareRow(t, body, "data-compare-question", "q1"); !strings.Contains(row, "+0.1500") || !strings.Contains(row, "20 / 20") {
		t.Errorf("question row: %s", row)
	}
	if row := compareRow(t, body, "data-compare-resource", "request-p50"); !strings.Contains(row, `data-state="MEASURED"`) ||
		!strings.Contains(row, "10.0 ms") || !strings.Contains(row, "12.5 ms") || !strings.Contains(row, "+2.5 ms") {
		t.Errorf("request p50 row: %s", row)
	}
	if row := compareRow(t, body, "data-compare-resource", "request-p95"); !strings.Contains(row, "-1.0 ms") {
		t.Errorf("request p95 row: %s", row)
	}
	// Figures the evidence does not record stay NOT_CHECKED with no number.
	for _, key := range []string{"size", "memory", "inference-p50", "inference-p95", "inference-mean"} {
		row := compareRow(t, body, "data-compare-resource", key)
		if !strings.Contains(row, `data-state="NOT_CHECKED"`) || strings.Contains(row, "MEASURED") || strings.Contains(row, " ms") || !strings.Contains(row, "No valid evidence") {
			t.Errorf("%s was not left NOT_CHECKED: %s", key, row)
		}
	}

	// No recommendation, causal attribution or source-baseline context.
	for _, banned := range []string{`id="tuning-recommendation"`, `accept_recommendation`, `id="tuning-context"`, `id="tuning-no-recommendation"`, `data-recommended`} {
		if strings.Contains(body, banned) {
			t.Errorf("candidate comparison renders %q", banned)
		}
	}
	if !strings.Contains(body, `id="tuning-compare-no-attribution"`) {
		t.Error("the comparison does not state that it makes no causal attribution")
	}
	if after := historyBytes(t, c.e); after != before {
		t.Error("stored evidence changed")
	}
	if len(c.ft.order) != 2 {
		t.Errorf("comparison saved profiles: %d profiles", len(c.ft.order))
	}
}

func TestTuningCandidateComparisonResourcesAreMeasuredOnlyWhereBothSidesRecordThem(t *testing.T) {
	c := candidateEnv(t)
	m, rev := c.model()
	a := c.save(t, withLatency(evidenceOf(m, rev, c.variant, c.dataset, 20, 15), 10, 20, 12), "A has latency")
	b := c.save(t, evidenceOf(m, rev, c.variantB, c.dataset, 20, 18), "B has none")
	body := c.page(t, c.link(t, a, b))
	for _, key := range []string{"request-p50", "request-p95", "request-mean"} {
		if row := compareRow(t, body, "data-compare-resource", key); !strings.Contains(row, `data-state="NOT_CHECKED"`) || strings.Contains(row, " ms") {
			t.Errorf("%s was estimated from one side: %s", key, row)
		}
	}
}

func TestTuningRefusesIncompatibleCandidateComparison(t *testing.T) {
	c := candidateEnv(t)
	m, rev := c.model()
	a, b := c.pair(t, 15, 18)
	ev := func(model, revision, variant, dataset string) string {
		return c.save(t, evidenceOf(model, revision, variant, dataset, 20, 18), "mismatch")
	}
	untuned := writeTunedVariant(t, c.e.home, c.profile, c.ft.analysis, false)
	otherQuestion := evidenceOf(m, rev, c.variantB, c.dataset, 20, 18)
	for i := range otherQuestion.Results {
		otherQuestion.Results[i].QuestionSHA256 = strings.Repeat("6", 64)
	}
	source := c.save(t, evidenceOf(m, rev, "", c.dataset, 20, 19), "source")
	for name, pair := range map[string][2]string{
		"different dataset":           {a, ev(m, rev, c.variantB, strings.Repeat("e", 64))},
		"different model":             {a, ev("laya-base", rev, c.variantB, c.dataset)},
		"different revision":          {a, ev(m, "otherrev", c.variantB, c.dataset)},
		"different question identity": {a, c.save(t, otherQuestion, "other question")},
		"no tuning provenance":        {a, ev(m, rev, untuned, c.dataset)},
		"candidate B is the source":   {a, source},
		"same variant twice":          {a, ev(m, rev, c.variant, c.dataset)},
	} {
		body := c.compare(t, pair[0], pair[1])
		if candidateCompareRe.MatchString(body) || handoffRe.MatchString(body) || !strings.Contains(body, `data-tuning-handoff-refused="true"`) {
			t.Errorf("%s: the pair was offered to Tuning or not explained:\n%s", name, body)
		}
	}

	// A link whose identity does not match the stored evidence is refused and
	// shows the plain profile with no comparison.
	u, _ := url.Parse(c.link(t, a, b))
	zero := strings.Repeat("0", 64)
	tamper := map[string]string{"profile": zero, "cmp_a_profile": zero, "cmp_b_profile": zero, "cmp_a_sha": zero, "cmp_b_sha": zero, "ds": zero, "qs": zero,
		"cmp_a_variant": "clef-flash--r--ffffffffffff", "cmp_b_variant": c.variant, "source": "laya-base", "cmp_a": b, "cmp_b": a}
	for key, value := range tamper {
		q := u.Query()
		q.Set(key, value)
		body := c.page(t, "/tuning?"+q.Encode())
		if !strings.Contains(body, `id="tuning-compare-refused"`) || strings.Contains(body, `data-compare-candidates="bound"`) ||
			strings.Contains(body, `id="tuning-compare-quality-table"`) || strings.Contains(body, `id="tuning-recommendation"`) {
			t.Errorf("tampered %s was not refused:\n%s", key, body)
		}
	}
	for _, q := range []string{"cmp_a=20260101T000000Z-aaaaaaaaaaaa&cmp_b=20260101T000000Z-bbbbbbbbbbbb&source=clef-flash", "cmp_a=" + a, "cmp_b=../x"} {
		if body := c.page(t, "/tuning?"+q); !strings.Contains(body, `id="tuning-compare-refused"`) {
			t.Errorf("%s: not refused", q)
		}
	}
}

// A comparison of two profiles cannot be shown when one of them is not saved.
func TestTuningRefusesCandidateComparisonWhoseProfileIsNotSaved(t *testing.T) {
	c := candidateEnv(t)
	a, b := c.pair(t, 15, 18)
	link := c.link(t, a, b)
	delete(c.ft.profiles, c.profile.ID())
	if body := c.page(t, link); !strings.Contains(body, `id="tuning-compare-refused"`) || strings.Contains(body, `id="tuning-compare-quality-table"`) {
		t.Errorf("a comparison with an unavailable profile was shown:\n%s", body)
	}
}

// The source-baseline path of the recommendation flow does not change when
// candidates can be compared.
func TestTuningCandidateComparisonLeavesTheSourceBaselineFlowUnchanged(t *testing.T) {
	c := candidateEnv(t)
	base, cand := c.feedback.pair(t, 15)
	body := c.compare(t, base, cand)
	if !handoffRe.MatchString(body) || candidateCompareRe.MatchString(body) {
		t.Fatalf("source-baseline pair no longer offers only its handoff:\n%s", body)
	}
	page := c.page(t, c.handoff(t, base, cand))
	for _, want := range []string{`data-context="bound"`, `id="tuning-recommendation"`, `data-recommendation="` + tuning.RegionFullAttention + `"`} {
		if !strings.Contains(page, want) {
			t.Errorf("recommendation flow lacks %q", want)
		}
	}
	if strings.Contains(page, `id="tuning-compare"`) {
		t.Error("the source-baseline page shows a candidate comparison")
	}
}
