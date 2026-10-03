package tuning

import (
	"fmt"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// BuildRequest is the Forge build request that reproduces one persisted
// profile exactly: the compiled canonical recipe, the profile's provenance and,
// for a layer-wise profile, its resolved plan. Every Forge build of a profile
// (the Tuning build, the composed build and evaluate, a finalist built from a
// trial Candidate) goes through it, so none can resolve a different plan.
func BuildRequest(h home.Home, source home.ModelManifest, profileID string) (optimize.Request, error) {
	if !setup.SupportsVariants(source) {
		return optimize.Request{}, fmt.Errorf("model %s has no variants (only System One models are optimized)", source.ID)
	}
	profile, analysis, err := LoadProfile(h, profileID)
	if err != nil {
		return optimize.Request{}, err
	}
	if profile.Source != home.SourceOf(source) {
		return optimize.Request{}, fmt.Errorf("tuning profile %s is bound to a different source model", profileID)
	}
	compiled, err := Compile(profile, analysis)
	if err != nil {
		return optimize.Request{}, err
	}
	provenance := Provenance(profile, analysis, compiled)
	return optimize.Request{Model: source.ID, Recipe: compiled.Recipe.Name, CompiledRecipe: &compiled.Recipe, Tuning: &provenance, Plan: compiled.Plan}, nil
}
