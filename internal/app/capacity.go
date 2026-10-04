package app

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// capacityCalibrationResult is the measured safe envelope returned by the
// isolated Clef calibration worker. It intentionally contains no execution
// identity: the host binds the measurement to the exact immutable target it
// launched and verified.
type capacityCalibrationResult struct {
	MaxStateTokens           int    `json:"max_state_tokens"`
	MaxInputTokens           int    `json:"max_input_tokens"`
	MaxBatchItems            int    `json:"max_batch_items"`
	MaxBatchPaddedTokens     int    `json:"max_batch_padded_tokens"`
	RequiredGPUHeadroomBytes uint64 `json:"required_gpu_headroom_bytes"`
}

// ensureVariantCapacity makes the #264 CUDA capacity contract durable before
// an accepted Clef variant is activated. Existing exact profiles are reused.
// A missing profile is measured on the exact persisted variant in an isolated
// worker, validated, and atomically upserted. Nothing is written until the
// measurement completes successfully.
func ensureVariantCapacity(ctx context.Context, h home.Home, variantID, device string, log io.Writer) error {
	if device != "cuda" {
		return nil
	}
	model, variant, err := setup.FindVariant(h, variantID)
	if err != nil {
		return err
	}
	if model.Provider != home.ProviderClef {
		return nil
	}

	cfg, rt, err := server.CapacityCalibrationConfig(h, device, variantID, log)
	if err != nil {
		return err
	}
	if rt.Variant == nil || rt.Variant.ID != variant.ID {
		return fmt.Errorf("capacity calibration resolved variant %q, want %q", func() string {
			if rt.Variant == nil {
				return ""
			}
			return rt.Variant.ID
		}(), variant.ID)
	}
	target := home.CapacityTarget{
		Runtime:           rt.Runtime,
		ModelID:           model.ID,
		Provider:          model.Provider,
		Repo:              model.Repo,
		Revision:          model.Revision,
		SourceFilesSHA256: home.SourceOf(model).FilesSHA256,
		Device:            device,
		DType:             rt.Variant.DType,
		VariantID:         variant.ID,
	}
	if profiles, err := h.LoadCapacityProfiles(); err == nil {
		if _, ok := profiles.Find(target); ok {
			return nil
		}
	}

	fmt.Fprintf(log, "capacity: calibrating exact target %s on %s before activation\n", variant.ID, device)
	proc, err := worker.Start(ctx, cfg, nil)
	if err != nil {
		return fmt.Errorf("capacity calibration worker: %w", err)
	}
	defer proc.Close()
	raw, err := proc.Call("capacity_calibrate", nil)
	if err != nil {
		return fmt.Errorf("capacity calibration: %w", err)
	}
	var measured capacityCalibrationResult
	if err := json.Unmarshal(raw, &measured); err != nil {
		return fmt.Errorf("capacity calibration result: %w", err)
	}
	profile := home.CapacityProfile{
		CapacityTarget:           target,
		MaxStateTokens:           measured.MaxStateTokens,
		MaxInputTokens:           measured.MaxInputTokens,
		MaxBatchItems:            measured.MaxBatchItems,
		MaxBatchPaddedTokens:     measured.MaxBatchPaddedTokens,
		RequiredGPUHeadroomBytes: measured.RequiredGPUHeadroomBytes,
	}
	if err := profile.Validate(); err != nil {
		return fmt.Errorf("capacity calibration produced an invalid profile: %w", err)
	}
	if err := h.SaveCapacityProfile(profile); err != nil {
		return fmt.Errorf("persist capacity profile: %w", err)
	}
	fmt.Fprintf(log, "capacity: recorded state=%d input=%d batch=%d padded=%d headroom=%d bytes for %s\n",
		profile.MaxStateTokens, profile.MaxInputTokens, profile.MaxBatchItems,
		profile.MaxBatchPaddedTokens, profile.RequiredGPUHeadroomBytes, variant.ID)
	return nil
}
