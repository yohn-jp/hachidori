package dashboard

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// failingLoad is a tuning authority whose saved profile cannot be loaded.
type failingLoad struct{ *fakeTuning }

func (failingLoad) LoadProfile(string) (tuning.Profile, tuning.Analysis, error) {
	return tuning.Profile{}, tuning.Analysis{}, errors.New("store unavailable")
}

func forgeProfileLabels(t *testing.T, e *env) []string {
	t.Helper()
	var labels []string
	for _, p := range e.d.forgeView().Forge.Profiles {
		if p.Tuned {
			labels = append(labels, p.Label)
		}
	}
	return labels
}

// An all-AUTO profile overrides nothing yet the canonical compiler keeps the
// required groups at source precision: Forge shows the resolved plan's count,
// the one Tuning shows, never the number of explicit overrides.
func TestForgeProfileLabelIsTheEffectiveGroupCount(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	auto := expectedProfile(t, ft.analysis, "balanced")
	if err := ft.SaveProfile(auto, ft.analysis); err != nil {
		t.Fatal(err)
	}
	compiled, err := tuning.Compile(auto, ft.analysis)
	if err != nil {
		t.Fatal(err)
	}
	want := 0
	for _, g := range compiled.Plan.Groups {
		if g.Effective == home.PolicySourcePrecision {
			want++
		}
	}
	if overrideCount(auto) != 0 || want == 0 {
		t.Fatalf("fixture: %d overrides, %d groups at source precision; the profile must preserve groups with no override", overrideCount(auto), want)
	}
	labels := forgeProfileLabels(t, e)
	wantText := strconv.Itoa(want) + " / " + strconv.Itoa(len(compiled.Plan.Groups)) + " groups at source precision · 0 overridden"
	if len(labels) != 1 || !strings.HasSuffix(labels[0], wantText) {
		t.Fatalf("labels %q, want one ending %q", labels, wantText)
	}
	// Tuning shows the same count for the same profile.
	tv := e.d.tuningView(e.d.modelsView(e.d.view("Tuning", "tuning")), ft.analysis.Source.ID, auto.ID(), nil)
	if tv.Summary.AtSource != want || tv.Summary.Groups != len(compiled.Plan.Groups) {
		t.Fatalf("Tuning shows %d / %d groups at source precision, Forge %d / %d", tv.Summary.AtSource, tv.Summary.Groups, want, len(compiled.Plan.Groups))
	}
}

// The profile of the physical run: all AUTO over the Clef-Flash layout, whose
// compiled recipe preserves lm_head, both linear-attention gates, the vision
// tower and the carried joint head: five selectors, five required groups.
func TestForgeProfileLabelOfTheAllAutoClefProfileIsNotZero(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	auto := expectedProfile(t, ft.analysis, "balanced")
	if err := ft.SaveProfile(auto, ft.analysis); err != nil {
		t.Fatal(err)
	}
	compiled, err := tuning.Compile(auto, ft.analysis)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.Recipe.Preserved) != 5 {
		t.Fatalf("recipe preserves %d selectors, want the canonical five", len(compiled.Recipe.Preserved))
	}
	labels := forgeProfileLabels(t, e)
	if len(labels) != 1 || !strings.Contains(labels[0], "5 / "+strconv.Itoa(len(ft.analysis.Groups))+" groups at source precision") {
		t.Fatalf("labels %q, want 5 required groups at source precision", labels)
	}
}

// A saved legacy coarse profile is labelled as such with its own regional
// count, never as a layer-wise plan.
func TestForgeProfileLabelOfALegacyProfileSaysLegacy(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	legacy, la := legacyProfile(t, ft, "balanced")
	ft.legacyAnalysis = la
	ft.profiles = map[string]tuning.Profile{legacy.ID(): legacy}
	ft.order = []string{legacy.ID()}
	labels := forgeProfileLabels(t, e)
	if len(labels) != 1 || !strings.Contains(labels[0], "legacy · 5 / "+strconv.Itoa(len(la.Regions))+" regions preserved") {
		t.Fatalf("labels %q, want the legacy regional count", labels)
	}
}

// A profile that cannot be compiled or loaded gets no invented count.
func TestForgeProfileLabelNeverInventsACount(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	auto := expectedProfile(t, ft.analysis, "balanced")
	if err := ft.SaveProfile(auto, ft.analysis); err != nil {
		t.Fatal(err)
	}
	withTuning(e, failingLoad{ft})
	labels := forgeProfileLabels(t, e)
	if len(labels) != 1 || !strings.HasSuffix(labels[0], "preservation unavailable") || strings.Contains(labels[0], "preserved") {
		t.Fatalf("labels %q, want the unavailable state", labels)
	}

	// Stored but no longer valid against its analysis: Compile refuses it.
	bad := auto
	bad.CompilerVersion = "obsolete"
	ft.profiles[auto.ID()], ft.profiles[bad.ID()] = bad, bad
	withTuning(e, ft)
	if labels := forgeProfileLabels(t, e); len(labels) != 1 || !strings.HasSuffix(labels[0], "preservation unavailable") {
		t.Fatalf("labels %q for an uncompilable profile, want the unavailable state", labels)
	}
}
