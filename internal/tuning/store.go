package tuning

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/yohn-jp/hachidori/internal/home"
)

// SaveAnalysis stores one immutable, source-bound analysis document under
// HACHIDORI_HOME/state/model-analysis/<analysis ID>.json.
func SaveAnalysis(h home.Home, analysis Analysis) error {
	if err := validateAnalysis(analysis); err != nil {
		return err
	}
	id := analysis.ID()
	path := analysisPath(h, id)
	var existing Analysis
	if err := home.ReadJSON(path, &existing); err == nil {
		if err := validateAnalysis(existing); err != nil {
			return fmt.Errorf("stored tuning analysis %s is invalid: %w", id, err)
		}
		if existing.ID() != id || !bytes.Equal(existing.Canonical(), analysis.Canonical()) {
			return fmt.Errorf("stored tuning analysis %s does not match its identity", id)
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("reading stored tuning analysis %s: %w", id, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return home.WriteJSON(path, analysis)
}

// LoadAnalysis reads and verifies an analysis by its SHA-256 identity.
func LoadAnalysis(h home.Home, id string) (Analysis, error) {
	if !isDigest(id) {
		return Analysis{}, errors.New("tuning analysis ID must be a SHA-256 digest")
	}
	var analysis Analysis
	if err := home.ReadJSON(analysisPath(h, id), &analysis); err != nil {
		return Analysis{}, err
	}
	if err := validateAnalysis(analysis); err != nil {
		return Analysis{}, fmt.Errorf("stored tuning analysis %s is invalid: %w", id, err)
	}
	if analysis.ID() != id {
		return Analysis{}, fmt.Errorf("stored tuning analysis does not match ID %s", id)
	}
	return analysis, nil
}

// SaveProfile stores the analysis and one immutable, source-bound profile.
// Each document is atomically written with the repository's home-state helper.
func SaveProfile(h home.Home, profile Profile, analysis Analysis) error {
	if err := validateProfile(profile, analysis); err != nil {
		return err
	}
	if _, err := Compile(profile, analysis); err != nil {
		return err
	}
	if err := SaveAnalysis(h, analysis); err != nil {
		return err
	}
	id := profile.ID()
	path := profilePath(h, id)
	var existing Profile
	if err := home.ReadJSON(path, &existing); err == nil {
		if existing.ID() != id || !bytes.Equal(existing.Canonical(), profile.Canonical()) {
			return fmt.Errorf("stored tuning profile %s does not match its identity", id)
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("reading stored tuning profile %s: %w", id, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return home.WriteJSON(path, profile)
}

// LoadProfile returns a profile and its exact analysis after verifying both
// content identities and compiling the pair against the canonical recipe.
func LoadProfile(h home.Home, id string) (Profile, Analysis, error) {
	if !isDigest(id) {
		return Profile{}, Analysis{}, errors.New("tuning profile ID must be a SHA-256 digest")
	}
	var profile Profile
	if err := home.ReadJSON(profilePath(h, id), &profile); err != nil {
		return Profile{}, Analysis{}, err
	}
	if profile.ID() != id {
		return Profile{}, Analysis{}, fmt.Errorf("stored tuning profile does not match ID %s", id)
	}
	analysis, err := LoadAnalysis(h, profile.AnalysisSHA256)
	if err != nil {
		return Profile{}, Analysis{}, fmt.Errorf("tuning profile %s analysis: %w", id, err)
	}
	if _, err := Compile(profile, analysis); err != nil {
		return Profile{}, Analysis{}, fmt.Errorf("stored tuning profile %s is invalid: %w", id, err)
	}
	return profile, analysis, nil
}

func analysisPath(h home.Home, id string) string {
	return h.Path("state", "model-analysis", id+".json")
}

func profilePath(h home.Home, id string) string {
	return h.Path("state", "tuning", "profiles", id+".json")
}
