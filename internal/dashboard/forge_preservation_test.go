package dashboard

import (
	"errors"
	"strconv"
	"strings"
	"testing"

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

// An all-Auto profile pins nothing yet the canonical compiler preserves
// regions: Forge shows the compiled effective count, the one Tuning shows,
// never the number of explicit pins.
func TestForgeProfileLabelIsTheEffectivePreservedCount(t *testing.T) {
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
	for _, m := range compiled.Evidence.Regions {
		if m.Preserved {
			want++
		}
	}
	if pinnedCount(auto) != 0 || want == 0 {
		t.Fatalf("fixture: %d pins, %d compiled preserved regions; the profile must preserve regions with no pin", pinnedCount(auto), want)
	}
	labels := forgeProfileLabels(t, e)
	wantText := strconv.Itoa(want) + " / " + strconv.Itoa(len(ft.analysis.Regions)) + " preserved"
	if len(labels) != 1 || !strings.HasSuffix(labels[0], wantText) || strings.Contains(labels[0], " 0 / ") {
		t.Fatalf("labels %q, want one ending %q", labels, wantText)
	}
	// Tuning shows the same count for the same profile.
	tv := e.d.tuningView(e.d.modelsView(e.d.view("Tuning", "tuning")), ft.analysis.Source.ID, auto.ID())
	if tv.PreservedRegions != want {
		t.Fatalf("Tuning shows %d preserved regions, Forge %d", tv.PreservedRegions, want)
	}
}

// The profile of the physical run: all Auto over the Clef-Flash layout, whose
// compiled recipe preserves lm_head, both linear-attention gates, the vision
// tower and the carried joint head — five selectors — in five regions.
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
	if len(labels) != 1 || !strings.HasSuffix(labels[0], "5 / "+strconv.Itoa(len(ft.analysis.Regions))+" preserved") {
		t.Fatalf("labels %q, want 5 effective preserved regions", labels)
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
