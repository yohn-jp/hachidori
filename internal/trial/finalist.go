package trial

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// Finalist is a Candidate prepared for Forge materialization: the build
// request that reproduces the candidate's exact resolved plan.
type Finalist struct {
	Candidate Candidate
	Request   optimize.Request
}

// PrepareFinalist resolves the Forge build request of a stored candidate. It
// never builds from whatever profile is currently open: the request is derived
// from the profile the candidate names, and it is refused unless that profile
// still resolves to byte-for-byte the plan the candidate recorded (a change of
// the AUTO rules or of the canonical recipe since the trial would otherwise
// materialize a different plan than the one that was measured).
func PrepareFinalist(h home.Home, source home.ModelManifest, candidateID string) (Finalist, error) {
	c, err := LoadCandidate(h, candidateID)
	if err != nil {
		return Finalist{}, err
	}
	if c.Source != home.SourceOf(source) {
		return Finalist{}, fmt.Errorf("candidate %s was measured on a different source model than %s", short(c.ID), source.ID)
	}
	req, err := tuning.BuildRequest(h, source, c.Profile.ID)
	if err != nil {
		return Finalist{}, fmt.Errorf("candidate %s: %w", short(c.ID), err)
	}
	if req.Plan == nil || req.Tuning == nil {
		return Finalist{}, errors.New("the candidate's profile is not a layer-wise profile; it has no resolved plan to reproduce")
	}
	if !bytes.Equal(req.Plan.Canonical(), c.Plan.Canonical()) || req.Tuning.PlanSHA256 != c.PlanSHA256 {
		return Finalist{}, fmt.Errorf("the profile of candidate %s no longer resolves to the plan that was measured (plan %s, now %s): the candidate cannot be materialized as measured",
			short(c.ID), short(c.PlanSHA256), short(req.Plan.SHA256()))
	}
	return Finalist{Candidate: c, Request: req}, nil
}

// RecordMaterialization records that a Variant is the Forge build of a
// candidate. The variant must carry the candidate's source, profile and exact
// plan in its own provenance: the record only links two things that are already
// bound to the same plan, and the Variant itself is never modified.
func RecordMaterialization(h home.Home, candidateID string, v home.VariantManifest, now time.Time) error {
	c, err := LoadCandidate(h, candidateID)
	if err != nil {
		return err
	}
	switch {
	case v.Source != c.Source:
		return errors.New("the variant is not a variant of the candidate's source")
	case v.Tuning == nil || v.Tuning.PlanSHA256 != c.PlanSHA256 || v.Tuning.ProfileID != c.Profile.ID:
		return errors.New("the variant was not built from the candidate's profile and resolved plan")
	}
	m := Materialization{Schema: MaterializationSchema, CandidateID: c.ID, PlanSHA256: c.PlanSHA256, VariantID: v.ID,
		VariantManifest: v.ManifestSHA256(), RecordedAt: now.UTC().Format(time.RFC3339)}
	dir := filepath.Join(Dir(h, c.ID), "materialized")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, v.ID+".json")
	var existing Materialization
	switch err := home.ReadJSON(path, &existing); {
	case err == nil:
		if existing.VariantManifest != m.VariantManifest || existing.PlanSHA256 != m.PlanSHA256 {
			return fmt.Errorf("variant %s is already recorded for candidate %s with a different manifest", v.ID, short(c.ID))
		}
		return nil
	case !errors.Is(err, os.ErrNotExist):
		return err
	}
	return home.WriteJSON(path, m)
}

func short(id string) string { return id[:min(len(id), 12)] }

// LinkVariant records a freshly built Variant as the materialization of every
// stored candidate that names the same source, profile and exact resolved plan.
// It is how a Forge build, started from the Tuning page or the command line,
// closes the loop without Forge knowing about trials: the link is made only
// where the Variant's own provenance already carries the candidate's plan. A
// Variant without tuning provenance links nothing.
func LinkVariant(h home.Home, v home.VariantManifest, now time.Time) ([]string, error) {
	if v.Tuning == nil || v.Tuning.PlanSHA256 == "" {
		return nil, nil
	}
	candidates, err := List(h, v.Source.ID)
	if err != nil {
		return nil, err
	}
	var linked []string
	for _, c := range candidates {
		if c.PlanSHA256 != v.Tuning.PlanSHA256 || c.Profile.ID != v.Tuning.ProfileID || c.Source != v.Source {
			continue
		}
		if err := RecordMaterialization(h, c.ID, v, now); err != nil {
			return linked, err
		}
		linked = append(linked, c.ID)
	}
	return linked, nil
}
