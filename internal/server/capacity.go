package server

import (
	"fmt"

	"github.com/yohn-jp/hachidori/internal/home"
)

// resolveCapacityProfile selects only the exact profile for the active Clef
// execution target. A missing CUDA profile is carried to the worker as a typed
// readiness failure so the resident can report its capacity state.
func resolveCapacityProfile(h home.Home, a home.Active, rm home.RuntimeManifest,
	model home.ModelManifest, mm home.ModelManifest, variant *Variant, dtype string) (*home.CapacityProfile, string) {
	if dtype == "" {
		dtype = "bfloat16"
	}
	if variant != nil {
		dtype = variant.DType
	}
	target := home.CapacityTarget{
		Runtime: rm.EnvironmentID(), ModelID: model.ID, Provider: model.Provider,
		Repo: mm.Repo, Revision: mm.Revision,
		SourceFilesSHA256: home.SourceOf(mm).FilesSHA256, Device: a.Device, DType: dtype,
	}
	if variant != nil {
		target.VariantID = variant.ID
	}
	profiles, err := h.LoadCapacityProfiles()
	if err != nil {
		if a.Device == "cuda" {
			return nil, fmt.Sprintf("capacity profile unavailable: %v", err)
		}
		return nil, ""
	}
	profile, ok := profiles.Find(target)
	if !ok {
		if a.Device == "cuda" {
			return nil, fmt.Sprintf("no capacity profile matches runtime %s, model %s, provider %s, device %s, variant %s",
				target.Runtime, target.ModelID, target.Provider, target.Device, target.VariantID)
		}
		return nil, ""
	}
	return &profile, ""
}
