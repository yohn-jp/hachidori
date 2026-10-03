package dashboard

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// editForm describes the editor form of the fixture: the source, the objective
// and the per-group policies, as the browser would post them.
func editForm(op string, groups map[string]string, extra url.Values) url.Values {
	f := url.Values{"source": {"clef-flash"}, "objective": {"balanced"}, "op": {op}}
	for id, policy := range groups {
		f.Set(groupForm(id), policy)
	}
	for k, v := range extra {
		f[k] = v
	}
	return f
}

func selectedPolicy(t *testing.T, body, group string) string {
	t.Helper()
	row := tuningGroup(t, body, group)
	m := regexp.MustCompile(`<option value="([^"]+)"[^>]*selected>`).FindStringSubmatch(row)
	if m == nil {
		t.Fatalf("no selected policy in %s", row)
	}
	return m[1]
}

// Bulk override applies a policy to a family and block range and shows the
// result as an unsaved draft: nothing is saved, nothing is built.
func TestTuningBulkOverrideAppliesToAFamilyAndBlockRangeAsAnUnsavedDraft(t *testing.T) {
	e, ft, fv := tuningEnv(t)
	rec := e.post(t, "/tuning/edit", editForm("bulk", map[string]string{"block.03.full-attn": "source-precision"}, url.Values{
		"bulk_region": {tuning.RegionFeedForward}, "bulk_from": {"0"}, "bulk_to": {"1"}, "bulk_policy": {"source-precision"}}))
	if rec.Code != http.StatusOK {
		t.Fatalf("edit: %d %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, group := range []string{"block.00.mlp", "block.01.mlp"} {
		if got := selectedPolicy(t, body, group); got != "source-precision" || !strings.Contains(tuningGroup(t, body, group), `data-state="OVERRIDDEN"`) {
			t.Errorf("%s was not overridden by the range: %s", group, got)
		}
	}
	// outside the range and outside the family, and the operator's earlier
	// per-group choice from the same form, are as they were
	if got := selectedPolicy(t, body, "block.03.mlp"); got != "auto" {
		t.Errorf("a block outside the range changed: %s", got)
	}
	if got := selectedPolicy(t, body, "block.00.linear-attn"); got != "auto" {
		t.Errorf("another family changed: %s", got)
	}
	if got := selectedPolicy(t, body, "block.03.full-attn"); got != "source-precision" {
		t.Errorf("the operator's earlier choice was lost by the bulk edit: %s", got)
	}
	want := expectedProfile(t, ft.analysis, "balanced", "block.00.mlp", "block.01.mlp", "block.03.full-attn")
	for _, w := range []string{`data-tuning-profile="` + want.ID() + `"`, `data-unsaved="true"`, `id="tuning-notice"`, "unsaved draft", `data-overridden="3"`, `data-disclosure="advanced" id="tuning-advanced" open`} {
		if !strings.Contains(body, w) {
			t.Errorf("the draft page lacks %q", w)
		}
	}
	if len(ft.profiles) != 0 || len(fv.calls) != 0 {
		t.Errorf("editing saved or built something: %d profiles, %v", len(ft.profiles), fv.calls)
	}
	// saving the draft saves exactly that profile
	save := editForm("", map[string]string{"block.00.mlp": "source-precision", "block.01.mlp": "source-precision", "block.03.full-attn": "source-precision"}, nil)
	save.Del("op")
	if rec := e.post(t, "/tuning/save", save); rec.Code != http.StatusSeeOther {
		t.Fatalf("save: %d", rec.Code)
	}
	if _, ok := ft.profiles[want.ID()]; !ok || len(ft.profiles) != 1 {
		t.Errorf("the saved profile is not the draft: %v", ft.order)
	}
}

func TestTuningBulkOverrideRangesAreOpenEndedAndAddressEveryFamilyWhenNoneIsNamed(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	body := e.post(t, "/tuning/edit", editForm("bulk", nil, url.Values{"bulk_from": {"1"}, "bulk_policy": {"source-precision"}})).Body.String()
	// blocks 1 and later, every tunable family: not block 0
	for group, want := range map[string]string{"block.00.mlp": "auto", "block.00.linear-attn": "auto", "block.01.mlp": "source-precision", "block.01.linear-attn": "source-precision",
		"block.03.mlp": "source-precision", "block.03.full-attn": "source-precision"} {
		if got := selectedPolicy(t, body, group); got != want {
			t.Errorf("%s = %s, want %s", group, got, want)
		}
	}
	_ = ft
	// a block-less selection of a bounded range addresses nothing
	rec := e.post(t, "/tuning/edit", editForm("bulk", nil, url.Values{"bulk_region": {tuning.RegionFeedForward}, "bulk_from": {"7"}, "bulk_to": {"9"}, "bulk_policy": {"source-precision"}}))
	if b := rec.Body.String(); !strings.Contains(b, `id="tuning-notice"`) || !strings.Contains(b, "addresses no tunable group") || !strings.Contains(b, `role="alert"`) {
		t.Errorf("an empty selection was not refused:\n%s", workspace(b))
	}
}

func TestTuningResetToAutoByFamilyAndForEveryGroup(t *testing.T) {
	e, _, _ := tuningEnv(t)
	groups := map[string]string{"block.00.mlp": "source-precision", "block.01.mlp": "w4a16", "block.03.full-attn": "source-precision", "block.00.linear-attn": "source-precision"}

	body := e.post(t, "/tuning/edit", editForm("reset:"+tuning.RegionFeedForward, groups, nil)).Body.String()
	for group, want := range map[string]string{"block.00.mlp": "auto", "block.01.mlp": "auto", "block.03.full-attn": "source-precision", "block.00.linear-attn": "source-precision"} {
		if got := selectedPolicy(t, body, group); got != want {
			t.Errorf("after resetting the family, %s = %s, want %s", group, got, want)
		}
	}
	if !strings.Contains(body, "this family") || !strings.Contains(body, `data-overridden="2"`) {
		t.Error("the family reset is not reported or counted")
	}

	body = e.post(t, "/tuning/edit", editForm("reset_all", groups, nil)).Body.String()
	tunable, _ := tuning.Tunable(tuningAnalysis(t))
	for _, id := range tunable {
		if got := selectedPolicy(t, body, id); got != "auto" || !strings.Contains(tuningGroup(t, body, id), `data-state="AUTO"`) {
			t.Errorf("after reset-all, %s = %s", id, got)
		}
	}
	initial := expectedProfile(t, tuningAnalysis(t), "balanced")
	if !strings.Contains(body, `data-overridden="0"`) || !strings.Contains(body, `data-tuning-profile="`+initial.ID()+`"`) {
		t.Error("resetting every group did not return the all-AUTO profile identity")
	}
}

// A refused edit never loses the operator's draft and says why.
func TestTuningRefusedEditsKeepTheDraftAndExplainWhy(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	draft := map[string]string{"block.00.mlp": "source-precision"}
	tunable, _ := tuning.Tunable(tuningAnalysis(t))
	allSource := map[string]string{}
	for _, id := range tunable {
		allSource[id] = "source-precision"
	}
	allButOne := map[string]string{}
	for _, id := range tunable[1:] {
		allButOne[id] = "source-precision"
	}
	for name, c := range map[string]struct {
		form url.Values
		want string
	}{
		"unsupported policy": {editForm("bulk", draft, url.Values{"bulk_policy": {"w8a16"}}), "unsupported policy"},
		"reversed range":     {editForm("bulk", draft, url.Values{"bulk_from": {"3"}, "bulk_to": {"1"}, "bulk_policy": {"w4a16"}}), "is empty"},
		"not a number":       {editForm("bulk", draft, url.Values{"bulk_from": {"first"}, "bulk_policy": {"w4a16"}}), "whole block numbers"},
		"negative":           {editForm("bulk", draft, url.Values{"bulk_to": {"-2"}, "bulk_policy": {"w4a16"}}), "whole block numbers"},
		"required family":    {editForm("bulk", draft, url.Values{"bulk_region": {tuning.RegionVision}, "bulk_policy": {"w4a16"}}), "addresses no tunable group"},
		"unknown operation":  {editForm("explode", draft, nil), "unknown editor operation"},
		"nothing quantized":  {editForm("bulk", allButOne, url.Values{"bulk_region": {tuning.RegionLinearAttention}, "bulk_from": {"0"}, "bulk_to": {"0"}, "bulk_policy": {"source-precision"}}), "nothing would be W4A16"},
		"unknown group":      {editForm("bulk", map[string]string{"not.a.group": "w4a16"}, url.Values{"bulk_policy": {"w4a16"}}), "unknown group"},
		"unknown policy":     {editForm("refresh", map[string]string{"block.00.mlp": "quantize-more"}, nil), "unsupported policy"},
	} {
		rec := e.post(t, "/tuning/edit", c.form)
		if rec.Code != http.StatusOK {
			t.Errorf("%s: %d", name, rec.Code)
			continue
		}
		b := rec.Body.String()
		if !strings.Contains(b, `id="tuning-notice"`) || !strings.Contains(b, `role="alert"`) || !strings.Contains(b, c.want) {
			t.Errorf("%s: not explained (%q):\n%s", name, c.want, workspace(b))
		}
		if name != "unknown group" && name != "unknown policy" && selectedPolicy(t, b, "block.00.mlp") != "source-precision" {
			t.Errorf("%s: the operator's draft was lost", name)
		}
	}
	if len(ft.profiles) != 0 {
		t.Error("a refused edit saved a profile")
	}
	_ = allSource
}

// Every legacy coarse profile stays readable: it is shown as its exact
// layer-wise equivalent, stays buildable as it is, and is upgraded by saving.
func TestTuningShowsALegacyProfileAsItsExactLayerwiseEquivalentWithAnUpgradePath(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	legacy, la := legacyProfile(t, ft, "maximum-fidelity", tuning.RegionFullAttention, tuning.RegionOutputEmbeddings)
	ft.legacyAnalysis = la
	ft.profiles = map[string]tuning.Profile{legacy.ID(): legacy}
	ft.order = []string{legacy.ID()}
	legacyID := legacy.ID()

	body := e.get(t, "/tuning?source=clef-flash&profile="+legacyID).Body.String()
	upgraded, err := tuning.MigrateProfile(legacy, la, ft.analysis)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`id="tuning-legacy"`, `data-legacy="` + legacyID + `"`, `data-legacy-profile="` + legacyID + `"`, `data-upgraded-profile="` + upgraded.ID() + `"`,
		"not saved yet", "Pins already required", `name="legacy_profile" value="` + legacyID + `"`, `data-tuning-profile="` + upgraded.ID() + `"`, `data-unsaved="true"`,
		`<option value="maximum-fidelity" selected>`} {
		if !strings.Contains(body, want) {
			t.Errorf("the legacy page lacks %q", want)
		}
	}
	if got := selectedPolicy(t, body, "block.03.full-attn"); got != "source-precision" || !strings.Contains(tuningGroup(t, body, "block.03.full-attn"), `data-state="OVERRIDDEN"`) {
		t.Errorf("the legacy pin is not the layer-wise override: %s", got)
	}
	if got := selectedPolicy(t, body, "block.03.mlp"); got != "auto" {
		t.Errorf("an unpinned region became %s", got)
	}
	// the legacy profile is listed as such and is the current one
	row := body[strings.Index(body, `data-profile="`+legacyID+`"`):]
	row = row[:strings.Index(row, "</tr>")]
	if !strings.Contains(row, "legacy coarse") || !strings.Contains(row, `aria-current="true"`) {
		t.Errorf("legacy profile row: %s", row)
	}
	// Viewing saved nothing and changed nothing.
	if len(ft.profiles) != 1 || ft.profiles[legacyID].ID() != legacyID {
		t.Fatal("viewing a legacy profile saved or changed profiles")
	}

	// Saving from the legacy page is the explicit upgrade: a new layer-wise
	// profile that names its lineage; the legacy profile is untouched.
	form := editForm("", map[string]string{"block.03.full-attn": "source-precision"}, url.Values{"legacy_profile": {legacyID}, "objective": {"maximum-fidelity"}})
	form.Del("op")
	rec := e.post(t, "/tuning/save", form)
	if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != tuningLocation("clef-flash", upgraded.ID()) {
		t.Fatalf("upgrade: %d %q, want %s", rec.Code, rec.Header().Get("Location"), upgraded.ID())
	}
	saved := ft.profiles[upgraded.ID()]
	if saved.Schema != tuning.ProfileSchema || saved.MigratedFrom == nil || saved.MigratedFrom.FromProfileID != legacyID {
		t.Fatalf("upgraded profile %+v", saved)
	}
	if ft.profiles[legacyID].ID() != legacyID || len(ft.profiles) != 2 {
		t.Error("the upgrade replaced the legacy profile")
	}
	if page := e.get(t, rec.Header().Get("Location")).Body.String(); strings.Contains(page, `id="tuning-legacy"`) || !strings.Contains(page, "Saved profile") {
		t.Error("the upgraded profile is still shown as a legacy upgrade")
	}
	// Forge still offers the legacy profile and builds it exactly as before.
	var labels []string
	for _, p := range e.d.forgeView().Forge.Profiles {
		if p.Tuned {
			labels = append(labels, p.Label)
		}
	}
	joined := strings.Join(labels, "\n")
	if len(labels) != 2 || !strings.Contains(joined, short12(legacyID)) || !strings.Contains(joined, "legacy ·") || !strings.Contains(joined, "groups at source precision") {
		t.Errorf("Forge profile labels %q", labels)
	}
}

func TestTuningRefusesToUpgradeALegacyProfileOfAnotherStructure(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	legacy, la := legacyProfile(t, ft, "balanced", tuning.RegionFullAttention)
	// the legacy analysis describes a smaller model than the current one
	source, _ := tuningSourceManifest(t)
	layout := tuningLayout()
	layout.LinearModules = layout.LinearModules[:len(layout.LinearModules)-1]
	small, err := tuning.AnalyzeLegacy(source, layout)
	if err != nil {
		t.Fatal(err)
	}
	if small.SHA256() == la.SHA256() {
		t.Fatal("fixture: the analyses are identical")
	}
	legacy2, err := tuning.NewLegacyProfile(small, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	ft.legacyAnalysis = small
	ft.profiles = map[string]tuning.Profile{legacy2.ID(): legacy2}
	ft.order = []string{legacy2.ID()}
	_ = legacy
	body := e.get(t, "/tuning?source=clef-flash&profile="+legacy2.ID()).Body.String()
	if !strings.Contains(body, "cannot be upgraded to a layer-wise profile") || !strings.Contains(body, "different model structure") {
		t.Errorf("a legacy profile of another structure was not refused:\n%s", workspace(body))
	}
	if strings.Contains(body, `data-legacy-profile=`) {
		t.Error("an upgrade was offered for an incompatible legacy profile")
	}
	form := editForm("", nil, url.Values{"legacy_profile": {legacy2.ID()}})
	form.Del("op")
	e.post(t, "/tuning/save", form)
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "cannot be upgraded") {
		t.Errorf("saving an upgrade of an incompatible profile: %+v", a)
	}
	if len(ft.profiles) != 1 {
		t.Error("an incompatible upgrade saved a profile")
	}
}

func TestTuningShowsWhatChangedFromTheBaselineAndSaysWhenItCannotKnow(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	p := expectedProfile(t, ft.analysis, "balanced", "block.00.mlp", "block.01.mlp", "block.03.mlp")
	if err := ft.SaveProfile(p, ft.analysis); err != nil {
		t.Fatal(err)
	}
	page := "/tuning?source=clef-flash&profile=" + p.ID()

	// No accepted baseline variant: the canonical recipe is the baseline.
	ft.candidate = TuningCandidate{BaselineWhy: "no runtime is active, so there is no accepted baseline"}
	body := e.get(t, page).Body.String()
	row := body[strings.Index(body, `data-change="Feed-forward projections"`):]
	row = row[:strings.Index(row, "</tr>")]
	for _, want := range []string{"0–1, 3", "W4A16 (4-bit weights)", "Source precision (bfloat16)", `<td class="num">3</td>`} {
		if !strings.Contains(row, want) {
			t.Errorf("change row lacks %q: %s", want, row)
		}
	}
	for _, want := range []string{"the canonical recipe, the policy of every untuned variant", "no runtime is active", `data-changes="1"`} {
		if !strings.Contains(body, want) {
			t.Errorf("baseline basis lacks %q", want)
		}
	}

	// A baseline built by the canonical recipe is the same all-AUTO resolution.
	ft.candidate = TuningCandidate{Baseline: "clef-flash--r--aaaaaaaaaaaa", BaselineCanonical: true}
	if body := e.get(t, page).Body.String(); !strings.Contains(body, "accepted baseline clef-flash--r--aaaaaaaaaaaa (built with the canonical recipe)") || !strings.Contains(body, `data-changes="1"`) {
		t.Error("a canonical baseline is not stated or does not give the changes")
	}

	// A baseline with applied per-group evidence is compared group by group.
	ev := &home.TuningEvidence{Schema: home.TuningEvidenceSchema, PlanSHA256: strings.Repeat("a", 64), AutoPolicy: tuning.AutoPolicyVersion}
	canonical := mustCompileDash(t, expectedProfile(t, ft.analysis, "balanced"), ft.analysis)
	for _, g := range canonical.Plan.Groups {
		applied := g.Effective
		if g.ID == "block.00.mlp" {
			applied = home.PolicySourcePrecision // the baseline already preserved this one
		}
		ev.Groups = append(ev.Groups, home.TuningGroupApplied{ID: g.ID, Selection: g.Selection, Requested: g.Requested, Effective: applied, Applied: applied, Preserved: applied == home.PolicySourcePrecision})
	}
	ft.candidate = TuningCandidate{Baseline: "clef-flash--r--bbbbbbbbbbbb", BaselineEvidence: ev}
	body = e.get(t, page).Body.String()
	row = body[strings.Index(body, `data-change="Feed-forward projections"`):]
	row = row[:strings.Index(row, "</tr>")]
	if !strings.Contains(row, "1, 3") || strings.Contains(row, "0–1") || !strings.Contains(body, "its applied evidence") {
		t.Errorf("per-group baseline evidence was not used: %s", row)
	}

	// A baseline whose policy is unknown is NOT_CHECKED, never guessed.
	ft.candidate = TuningCandidate{Baseline: "clef-flash--r--cccccccccccc"}
	body = e.get(t, page).Body.String()
	if !strings.Contains(body, "records no per-group evidence") || !strings.Contains(body, "is NOT_CHECKED") || strings.Contains(body, `id="tuning-changes-table"`) {
		t.Errorf("an unknown baseline policy was compared:\n%s", workspace(body))
	}

	// Unchanged against the baseline says so.
	ft.candidate = TuningCandidate{BaselineWhy: "none"}
	auto := expectedProfile(t, ft.analysis, "balanced")
	if err := ft.SaveProfile(auto, ft.analysis); err != nil {
		t.Fatal(err)
	}
	body = e.get(t, "/tuning?source=clef-flash&profile="+auto.ID()).Body.String()
	if !strings.Contains(body, `id="tuning-no-changes"`) || !strings.Contains(body, `data-changes="0"`) {
		t.Error("an unchanged profile does not say it matches the baseline")
	}
	// An unreadable evidence store is reported, not hidden.
	ft.candidate, ft.candidateErr = TuningCandidate{}, http.ErrAbortHandler
	if body := e.get(t, page).Body.String(); !strings.Contains(body, "could not be read") {
		t.Error("an unreadable candidate record is not reported")
	}
}

func mustCompileDash(t *testing.T, p tuning.Profile, a tuning.Analysis) tuning.Compilation {
	t.Helper()
	c, err := tuning.Compile(p, a)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// The candidate built from a profile is compared with the accepted baseline
// on accuracy, fidelity, latency, VRAM and artifact size, each with its state;
// a figure with no record is NOT_CHECKED and carries no number.
func TestTuningComparesTheCandidateWithTheAcceptedBaselineWithoutInventingFigures(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	p := expectedProfile(t, ft.analysis, "balanced", "block.03.full-attn")
	if err := ft.SaveProfile(p, ft.analysis); err != nil {
		t.Fatal(err)
	}
	c := mustCompileDash(t, p, ft.analysis)
	measured := func(v, basis string) ImpactValue { return ImpactValue{State: Measured, Value: v, Basis: basis} }
	none := ImpactValue{State: NotChecked, Basis: "no certification"}
	ft.candidate = TuningCandidate{Variant: "clef-flash--r--dddddddddddd", Baseline: "clef-flash--r--aaaaaaaaaaaa", Comparable: true,
		Figures: []TuningFigure{
			{Key: "accuracy", Label: "Choice accuracy", Evaluated: true, Baseline: measured("0.9500 accuracy over 20 labelled observations", "certification of baseline"), Candidate: measured("0.9000 accuracy over 20 labelled observations", "certification of candidate")},
			{Key: "fidelity", Label: "Fidelity to the source model", Evaluated: true, Baseline: measured("0.50% choice flips over 200 paired observations", "certification"), Candidate: measured("1.00% choice flips over 200 paired observations", "certification")},
			{Key: "latency", Label: "Request latency", Evaluated: true, Baseline: measured("p50 11.0 ms", "certification"), Candidate: none},
			{Key: "vram", Label: "VRAM", Evaluated: true, Baseline: none, Candidate: ImpactValue{State: Measured, Value: "", Basis: "probe"}},
			{Key: "size", Label: "Artifact size", Baseline: measured("5.00 GiB", "bytes of its artifact files"), Candidate: measured("4.80 GiB", "bytes of its artifact files")},
		},
		Evidence: &home.TuningEvidence{Schema: home.TuningEvidenceSchema, PlanSHA256: c.Plan.SHA256(), AutoPolicy: tuning.AutoPolicyVersion}}
	for _, g := range c.Plan.Groups {
		ft.candidate.Evidence.Groups = append(ft.candidate.Evidence.Groups, home.TuningGroupApplied{ID: g.ID, Selection: g.Selection, Requested: g.Requested, Effective: g.Effective, Required: g.Required,
			Applied: g.Effective, Preserved: g.Effective == home.PolicySourcePrecision})
	}
	body := e.get(t, "/tuning?source=clef-flash&profile="+p.ID()).Body.String()
	fig := func(key string) string {
		i := strings.Index(body, `data-figure="`+key+`"`)
		if i < 0 {
			t.Fatalf("no figure %s", key)
		}
		return body[i : i+strings.Index(body[i:], "</tr>")]
	}
	if r := fig("accuracy"); !strings.Contains(r, `data-baseline-state="MEASURED"`) || !strings.Contains(r, `data-candidate-state="MEASURED"`) || !strings.Contains(r, "0.9500") || !strings.Contains(r, "0.9000") {
		t.Errorf("accuracy: %s", r)
	}
	if r := fig("latency"); !strings.Contains(r, `data-candidate-state="NOT_CHECKED"`) || !strings.Contains(r, "No valid evidence") || !strings.Contains(r, "p50 11.0 ms") {
		t.Errorf("a missing candidate latency was not NOT_CHECKED: %s", r)
	}
	if r := fig("vram"); !strings.Contains(r, `data-candidate-state="NOT_CHECKED"`) || !strings.Contains(r, `data-baseline-state="NOT_CHECKED"`) {
		t.Errorf("a measured state without a value was not demoted: %s", r)
	}
	if r := fig("size"); !strings.Contains(r, "5.00 GiB") || !strings.Contains(r, "4.80 GiB") || strings.Contains(r, "NOT_CHECKED") {
		t.Errorf("size: %s", r)
	}
	if strings.Contains(body, `id="tuning-not-comparable"`) || !strings.Contains(body, `data-candidate="clef-flash--r--dddddddddddd"`) {
		t.Error("a comparable pair is flagged or the candidate is not named")
	}
	// The applied evidence names every group: requested, effective, AUTO vs OVERRIDDEN.
	if n := strings.Count(body, "data-applied-group="); n != len(c.Plan.Groups) {
		t.Errorf("%d applied rows, %d groups", n, len(c.Plan.Groups))
	}
	i := strings.Index(body, `data-applied-group="block.03.full-attn"`)
	row := body[i : i+strings.Index(body[i:], "</tr>")]
	if !strings.Contains(row, `data-selection="OVERRIDDEN"`) || !strings.Contains(row, `data-applied="source-precision"`) || !strings.Contains(row, "Source precision (bfloat16)") {
		t.Errorf("applied evidence of an override: %s", row)
	}
	i = strings.Index(body, `data-applied-group="block.03.mlp"`)
	if row := body[i : i+strings.Index(body[i:], "</tr>")]; !strings.Contains(row, `data-selection="AUTO"`) || !strings.Contains(row, `data-applied="w4a16"`) {
		t.Errorf("applied evidence of an AUTO group: %s", row)
	}

	// Evaluated figures of different evaluations are flagged, not silently compared.
	ft.candidate.Comparable, ft.candidate.ComparableWhy = false, "the two variants were certified on different datasets or questions, so their evaluated figures are not comparable"
	if b := e.get(t, "/tuning?source=clef-flash&profile="+p.ID()).Body.String(); !strings.Contains(b, `id="tuning-not-comparable"`) || !strings.Contains(b, "certified on different datasets") {
		t.Error("figures of different evaluations are not flagged")
	}
	// Evidence of another resolution of the profile is not shown against this plan.
	ft.candidate.Evidence.PlanSHA256 = strings.Repeat("9", 64)
	b := e.get(t, "/tuning?source=clef-flash&profile="+p.ID()).Body.String()
	if !strings.Contains(b, `id="tuning-applied-note"`) || !strings.Contains(b, "another resolution of this profile") || strings.Contains(b, "data-applied-group=") {
		t.Error("evidence bound to another plan was shown against the current one")
	}
	// A candidate built from a legacy profile has no per-group evidence, and says so.
	ft.candidate.Evidence = nil
	if b := e.get(t, "/tuning?source=clef-flash&profile="+p.ID()).Body.String(); !strings.Contains(b, "no per-group transformation is recorded") {
		t.Error("a candidate without per-group evidence does not say so")
	}
	// No candidate built yet: nothing is shown or invented.
	ft.candidate = TuningCandidate{}
	if b := e.get(t, "/tuning?source=clef-flash&profile="+p.ID()).Body.String(); strings.Contains(b, `id="tuning-baseline-compare"`) || strings.Contains(b, `id="tuning-applied"`) {
		t.Error("a comparison was shown without a candidate")
	}
}

func TestTuningCompareShowsGroupPolicyDifferencesAndRefusesAMixedSchemaDifference(t *testing.T) {
	c := candidateEnv(t)
	a, b := c.pair(t, 15, 18)
	body := c.page(t, c.link(t, a, b))
	if !strings.Contains(body, `id="tuning-compare-group-table"`) || strings.Contains(body, `id="tuning-compare-region-table"`) {
		t.Error("a layer-wise difference is not shown by group")
	}
	// candidate A was built from a legacy coarse profile: the difference by group is undefined
	legacy, la := legacyProfile(t, c.ft, "balanced")
	c.ft.legacyAnalysis = la
	c.ft.profiles[legacy.ID()] = legacy
	c.ft.order = append(c.ft.order, legacy.ID())
	delete(c.ft.profiles, c.profile.ID())
	vLegacy := writeTunedVariant(t, c.e.home, legacy, la, true)
	m, rev := c.model()
	la2 := c.save(t, withLatency(evidenceOf(m, rev, vLegacy, c.dataset, 20, 15), 10, 20, 12), "legacy candidate")
	body = c.page(t, c.link(t, la2, b))
	if !strings.Contains(body, `id="tuning-compare-schema-changed"`) || strings.Contains(body, `id="tuning-compare-group-table"`) || strings.Contains(body, `id="tuning-compare-region-table"`) {
		t.Errorf("a mixed-schema pair invented a difference:\n%s", body)
	}
	if !strings.Contains(body, `data-compare-candidates="bound"`) || !strings.Contains(body, "data-compare-profile-a=\""+legacy.ID()+"\"") {
		t.Error("the mixed-schema comparison is not bound to its exact profiles")
	}
}

// A recommendation accepted for a candidate built from a legacy profile is
// saved as a layer-wise profile: no new legacy profile is ever created.
func TestTuningAcceptingARecommendationOfALegacyProfileSavesALayerwiseProfile(t *testing.T) {
	f := feedbackEnv(t)
	f.ft.analysis = feedbackAnalysis(t)
	source, _ := tuningSourceManifest(t)
	legacyA, err := tuning.AnalyzeLegacy(source, feedbackLayout())
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := tuning.NewLegacyProfile(legacyA, "balanced")
	if err != nil {
		t.Fatal(err)
	}
	f.ft.legacyAnalysis = legacyA
	f.ft.profiles[legacy.ID()] = legacy
	f.ft.order = append(f.ft.order, legacy.ID())
	f.profile = legacy
	f.variant = writeTunedVariant(t, f.e.home, legacy, legacyA, true)
	base, cand := f.pair(t, 15)
	link := f.handoff(t, base, cand)
	body := f.page(t, link)
	if !strings.Contains(body, `data-context="bound"`) || !strings.Contains(body, `data-recommendation="full-attention-projections"`) || !strings.Contains(body, `id="tuning-legacy"`) {
		t.Fatalf("a legacy candidate's context was not bound:\n%s", body)
	}
	if fam := tuningFamily(t, body, tuning.RegionFullAttention); !strings.Contains(fam, `data-recommended="true"`) {
		t.Error("the recommended family is not marked on the legacy upgrade view")
	}
	u, _ := url.Parse(link)
	form := url.Values{"accept_recommendation": {tuning.RegionFullAttention}}
	for k := range u.Query() {
		form.Set(k, u.Query().Get(k))
	}
	before := len(f.ft.order)
	rec := f.e.post(t, "/tuning/save", form)
	if rec.Code != http.StatusSeeOther || len(f.ft.order) != before+1 {
		t.Fatalf("accept: %d, profiles %d -> %d (%s)", rec.Code, before, len(f.ft.order), f.e.lastAction(t).Message)
	}
	next := f.ft.profiles[f.ft.order[len(f.ft.order)-1]]
	if next.Schema != tuning.ProfileSchema || next.MigratedFrom == nil || next.MigratedFrom.FromProfileID != legacy.ID() ||
		next.Groups["block.03.full-attn"] != (tuning.GroupChoice{Mode: tuning.GroupOverride, Policy: home.PolicySourcePrecision}) {
		t.Errorf("accepted profile %+v", next)
	}
	if f.ft.profiles[legacy.ID()].ID() != legacy.ID() {
		t.Error("the legacy profile changed")
	}
}

func TestCompactBlocksNamesRangesAndSingleBlocks(t *testing.T) {
	for in, want := range map[string]string{"": "", "0": "0", "0 1 2": "0–2", "7 3 4 5 9": "3–5, 7, 9", "1 3 5": "1, 3, 5", "31 30": "30–31"} {
		var blocks []int
		for _, f := range strings.Fields(in) {
			n := 0
			for _, r := range f {
				n = n*10 + int(r-'0')
			}
			blocks = append(blocks, n)
		}
		if got := compactBlocks(blocks); got != want {
			t.Errorf("compactBlocks(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPolicyControlIsOrderedAndLabelledByNamedPolicies(t *testing.T) {
	policies := tuningPolicies()
	if len(policies) != 2 || policies[0].ID != home.PolicySourcePrecision || policies[1].ID != home.PolicyW4A16 || policies[0].Rank >= policies[1].Rank {
		t.Fatalf("policies %+v", policies)
	}
	for _, p := range policies {
		if p.Label == "" || p.Label == p.ID || p.Transformation == "" {
			t.Errorf("policy %+v is not a named, described transformation", p)
		}
	}
	if policyLabel(home.PolicyAuto) != "AUTO" || policyLabel("") != "AUTO" || policyLabel("future") != "future" {
		t.Error("policy labels")
	}
}
