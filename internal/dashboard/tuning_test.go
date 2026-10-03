package dashboard

import (
	"errors"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

type tuningBuild struct{ Source, Profile string }

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
	builds      []tuningBuild
	buildErr    error
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

func (f *fakeTuning) BuildCandidate(source, profileID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.builds = append(f.builds, tuningBuild{source, profileID})
	return f.buildErr
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
	return body[i : i+strings.Index(body[i:], "</tr>")]
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
		`data-tuning-analysis="` + ft.analysis.SHA256() + `"`, tuning.RecipeCompilerVersion, tuning.ProfileSchema, "initial all-Auto profile · not saved", `data-unsaved="true"`} {
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
	if len(ft.profiles) != 0 || len(ft.builds) != 0 {
		t.Error("rendering saved or built something")
	}
}

func TestTuningRegionsComeFromBackendAnalysisAndRawMappingsAreEvidenceOnly(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	body := e.get(t, "/tuning").Body.String()
	for _, r := range ft.analysis.Regions {
		row := tuningRow(t, body, r.ID)
		if r.ID == tuning.RegionJointSchemaHead && !strings.Contains(row, "1 files") {
			t.Errorf("file count missing: %s", row)
		}
		if r.ID == tuning.RegionVision && !strings.Contains(row, "2 modules") {
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
	if row := tuningRow(t, body, tuning.RegionLinearAttentionDecay); !strings.Contains(row, `data-effective="PRESERVED"`) || !strings.Contains(row, "already preserves") {
		t.Errorf("decay gate under Auto: %s", row)
	}
	if row := tuningRow(t, body, tuning.RegionFeedForward); !strings.Contains(row, `data-effective="AUTO"`) || !strings.Contains(row, "quantizes") {
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
	if len(ft.builds) != 0 || len(fv.calls) != 0 {
		t.Errorf("saving ran a build or Forge action: %v %v", ft.builds, fv.calls)
	}

	body := e.get(t, rec.Header().Get("Location")).Body.String()
	for _, want := range []string{`data-tuning-profile="` + want.ID() + `"`, "saved, immutable profile", `<option value="maximum-fidelity" selected>Maximum fidelity</option>`} {
		if !strings.Contains(body, want) {
			t.Errorf("saved profile page lacks %q", want)
		}
	}
	if strings.Contains(body, `data-unsaved="true"`) {
		t.Error("a saved profile is shown as unsaved")
	}
	for _, region := range []string{tuning.RegionFeedForward, tuning.RegionFullAttention} {
		row := tuningRow(t, body, region)
		if !strings.Contains(row, `<option value="pinned" selected>`) || !strings.Contains(row, `data-effective="PRESERVED"`) || !strings.Contains(row, "Pinned by this profile") {
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
	if len(ft.profiles) != 0 || len(ft.builds) != 0 {
		t.Errorf("a refused intent was saved or built: %d %v", len(ft.profiles), ft.builds)
	}
	if rec := e.post(t, "/tuning/save", url.Values{"token": {"forged"}, "source": {"clef-flash"}, "objective": {"balanced"}}); rec.Code != http.StatusForbidden {
		t.Errorf("forged token: %d", rec.Code)
	}
}

func TestTuningBuildCandidateHandsTheExactSavedProfileToForge(t *testing.T) {
	e, ft, fv := tuningEnv(t)
	form := url.Values{"source": {"clef-flash"}, "objective": {"minimum-size"}, "region." + tuning.RegionOutputEmbeddings: {"pinned"}, "region." + tuning.RegionFeedForward: {"pinned"}}
	rec := e.post(t, "/tuning/build", form)
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != "/forge" {
		t.Fatalf("build: %d %q", rec.Code, rec.Header().Get("Location"))
	}
	want := expectedProfile(t, ft.analysis, "minimum-size", tuning.RegionOutputEmbeddings, tuning.RegionFeedForward)
	if _, ok := ft.profiles[want.ID()]; !ok {
		t.Fatal("the profile was not saved before the handoff")
	}
	if len(ft.builds) != 1 || ft.builds[0] != (tuningBuild{"clef-flash", want.ID()}) {
		t.Fatalf("Forge received %+v, want the saved profile %s", ft.builds, want.ID())
	}
	// the handed identity is the stored document's own digest
	stored := ft.profiles[ft.builds[0].Profile]
	if stored.ID() != ft.builds[0].Profile || stored.SHA256() != ft.builds[0].Profile {
		t.Error("the handed identity is not the saved profile's digest")
	}
	// Tuning ran no optimizer lifecycle of its own: the Forge actions are untouched
	if len(fv.calls) != 0 {
		t.Errorf("Tuning invoked Forge actions itself: %v", fv.calls)
	}
	if a := e.lastAction(t); !a.OK || !strings.Contains(a.Message, want.ID()) || !strings.Contains(a.Message, "nothing is applied") {
		t.Errorf("handoff outcome: %+v", a)
	}

	// a refusal is shown on Tuning, with the saved profile still selected
	ft.buildErr = errors.New("another operation is running")
	rec = e.post(t, "/tuning/build", form)
	if rec.Header().Get("Location") != "/tuning?profile="+want.ID()+"&source=clef-flash" {
		t.Errorf("failed handoff location %q", rec.Header().Get("Location"))
	}
	body := e.get(t, rec.Header().Get("Location")).Body.String()
	if !strings.Contains(body, "another operation is running") || !strings.Contains(body, "FAILED") {
		t.Error("a failed handoff is not reported")
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
		if a[1] != "/tuning" && a[1] != "/tuning/save" && a[1] != "/tuning/build" {
			t.Errorf("Tuning posts to %s", a[1])
		}
	}
	if !strings.Contains(main, `formaction="/tuning/build"`) || !strings.Contains(main, ">Build candidate<") {
		t.Error("the primary action is not Build candidate")
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
	for _, want := range []string{`<html lang="ja">`, `<h1>チューニング</h1>`, "候補を構築", "プロファイルを保存", "意味的リージョン", "推定", "未確認", "保持", "最大の忠実度", `<a href="/tuning" aria-current="page">チューニング</a>`} {
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
