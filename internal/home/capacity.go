package home

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

// CapacityProfilesFile is the operator-owned input and device-headroom
// contract for resident workers under HACHIDORI_HOME/state.
const CapacityProfilesFile = "capacity-profiles.json"

const CapacityProfilesSchema = "hachidori.capacity-profiles.v1"

// CapacityTarget identifies the exact execution contract a capacity profile
// was measured/configured for. SourceFilesSHA256 pins the catalog artifact;
// VariantID pins the derived artifact when one is active.
type CapacityTarget struct {
	Runtime           string `json:"runtime"`
	ModelID           string `json:"model_id"`
	Provider          string `json:"provider"`
	Repo              string `json:"repo"`
	Revision          string `json:"revision"`
	SourceFilesSHA256 string `json:"source_files_sha256"`
	Device            string `json:"device"`
	VariantID         string `json:"variant_id,omitempty"`
}

// CapacityProfile declares an operator-validated input shape and the free GPU
// memory required after the selected model and provider runtime are resident.
// Token counts refer to the provider's complete encoded input, including the
// state and question schema. Padded tokens are rows multiplied by the padded
// sequence length of one provider forward.
type CapacityProfile struct {
	CapacityTarget
	MaxStateTokens           int    `json:"max_state_tokens"`
	MaxInputTokens           int    `json:"max_input_tokens"`
	MaxBatchItems            int    `json:"max_batch_items"`
	MaxBatchPaddedTokens     int    `json:"max_batch_padded_tokens"`
	RequiredGPUHeadroomBytes uint64 `json:"required_gpu_headroom_bytes,omitempty"`
}

// CapacityProfiles is state/capacity-profiles.json. Profiles are keyed by
// their full target tuple so a source model, variant, runtime or device cannot
// inherit another target's capacity.
type CapacityProfiles struct {
	Schema   string            `json:"schema"`
	Profiles []CapacityProfile `json:"profiles"`
}

// Validate checks a profile's target and declared limits without deriving or
// guessing any capacity values.
func (p CapacityProfile) Validate() error {
	t := p.CapacityTarget
	if t.Runtime == "" || t.ModelID == "" || t.Provider == "" || t.Repo == "" || t.Revision == "" ||
		t.SourceFilesSHA256 == "" || (t.Device != "cpu" && t.Device != "cuda") {
		return errors.New("capacity profile target must name runtime, model, provider, source revision, source digest and cpu or cuda device")
	}
	if p.MaxStateTokens <= 0 || p.MaxInputTokens <= 0 || p.MaxBatchItems <= 0 || p.MaxBatchPaddedTokens <= 0 {
		return errors.New("capacity profile token and batch limits must be positive")
	}
	if p.MaxInputTokens < p.MaxStateTokens {
		return errors.New("capacity profile max_input_tokens must be at least max_state_tokens")
	}
	if p.MaxBatchPaddedTokens < p.MaxInputTokens {
		return errors.New("capacity profile max_batch_padded_tokens must admit one max_input_tokens item")
	}
	if t.Device == "cuda" && p.RequiredGPUHeadroomBytes == 0 {
		return errors.New("CUDA capacity profile must declare required_gpu_headroom_bytes")
	}
	if t.Device == "cpu" && p.RequiredGPUHeadroomBytes != 0 {
		return errors.New("CPU capacity profile cannot declare GPU headroom")
	}
	return nil
}

// Validate checks the versioned profile document and rejects duplicate target
// identities. An empty profile list is valid; a worker with no matching entry
// reports its capacity as unconfigured and stays not ready on Clef CUDA.
// CPU launches do not require a GPU capacity profile.
func (ps CapacityProfiles) Validate() error {
	if ps.Schema != CapacityProfilesSchema {
		return fmt.Errorf("schema must be %q", CapacityProfilesSchema)
	}
	seen := map[CapacityTarget]bool{}
	for i, p := range ps.Profiles {
		if err := p.Validate(); err != nil {
			return fmt.Errorf("profiles[%d]: %w", i, err)
		}
		if seen[p.CapacityTarget] {
			return fmt.Errorf("profiles[%d]: duplicate capacity target", i)
		}
		seen[p.CapacityTarget] = true
	}
	return nil
}

// Find returns the exact matching profile. It never substitutes another
// runtime, model revision, device or variant.
func (ps CapacityProfiles) Find(target CapacityTarget) (CapacityProfile, bool) {
	for _, p := range ps.Profiles {
		if p.CapacityTarget == target {
			return p, true
		}
	}
	return CapacityProfile{}, false
}

// LoadCapacityProfiles reads the operator-visible profile document strictly.
// Unknown fields and trailing JSON are rejected so a misspelled limit cannot
// silently become an unbounded/default capacity.
func (h Home) LoadCapacityProfiles() (CapacityProfiles, error) {
	path := h.Path("state", CapacityProfilesFile)
	b, err := os.ReadFile(path)
	if err != nil {
		return CapacityProfiles{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var ps CapacityProfiles
	if err := dec.Decode(&ps); err != nil {
		return CapacityProfiles{}, fmt.Errorf("capacity profiles: %w", err)
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		if err == nil {
			return CapacityProfiles{}, errors.New("capacity profiles: trailing JSON value")
		}
		return CapacityProfiles{}, fmt.Errorf("capacity profiles: %w", err)
	}
	if err := ps.Validate(); err != nil {
		return CapacityProfiles{}, err
	}
	return ps, nil
}
