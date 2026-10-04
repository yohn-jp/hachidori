package home

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func capacityTarget(device, variant string) CapacityTarget {
	return CapacityTarget{Runtime: "cu128-test", ModelID: "clef-flash", Provider: ProviderClef,
		Repo: "Cloudflare/clef-flash", Revision: "17f0b0ad", SourceFilesSHA256: strings.Repeat("a", 64),
		Device: device, DType: "bfloat16", VariantID: variant}
}

func capacityProfile(device, variant string) CapacityProfile {
	profile := CapacityProfile{CapacityTarget: capacityTarget(device, variant), MaxStateTokens: 200,
		MaxInputTokens: 240, MaxBatchItems: 4, MaxBatchPaddedTokens: 960}
	if device == "cuda" {
		profile.RequiredGPUHeadroomBytes = 1
	}
	return profile
}

func TestCapacityProfilesRequireExactTargetAndPositiveDeclaredLimits(t *testing.T) {
	profiles := CapacityProfiles{Schema: CapacityProfilesSchema,
		Profiles: []CapacityProfile{capacityProfile("cuda", "variant-a"), capacityProfile("cpu", "")}}
	if err := profiles.Validate(); err != nil {
		t.Fatal(err)
	}
	if got, ok := profiles.Find(capacityTarget("cuda", "variant-a")); !ok || got.MaxStateTokens != 200 {
		t.Fatalf("exact profile = %+v, %v", got, ok)
	}
	for name, target := range map[string]CapacityTarget{
		"dtype":    func() CapacityTarget { x := capacityTarget("cuda", "variant-a"); x.DType = "float32"; return x }(),
		"runtime":  func() CapacityTarget { x := capacityTarget("cuda", "variant-a"); x.Runtime += "-other"; return x }(),
		"model":    func() CapacityTarget { x := capacityTarget("cuda", "variant-a"); x.ModelID += "-other"; return x }(),
		"provider": func() CapacityTarget { x := capacityTarget("cuda", "variant-a"); x.Provider = "laya"; return x }(),
		"repo":     func() CapacityTarget { x := capacityTarget("cuda", "variant-a"); x.Repo += "-other"; return x }(),
		"revision": func() CapacityTarget { x := capacityTarget("cuda", "variant-a"); x.Revision += "-other"; return x }(),
		"source files": func() CapacityTarget {
			x := capacityTarget("cuda", "variant-a")
			x.SourceFilesSHA256 = strings.Repeat("b", 64)
			return x
		}(),
		"device":  capacityTarget("cpu", "variant-a"),
		"variant": capacityTarget("cuda", "variant-b"),
	} {
		if _, ok := profiles.Find(target); ok {
			t.Errorf("%s mismatch selected a profile", name)
		}
	}
}

func TestCapacityProfilesRejectUnusableAndDuplicateEntries(t *testing.T) {
	cases := map[string]func(*CapacityProfile){
		"missing state limit":   func(p *CapacityProfile) { p.MaxStateTokens = 0 },
		"input below state":     func(p *CapacityProfile) { p.MaxInputTokens = p.MaxStateTokens - 1 },
		"padded below input":    func(p *CapacityProfile) { p.MaxBatchPaddedTokens = p.MaxInputTokens - 1 },
		"missing cuda headroom": func(p *CapacityProfile) { p.RequiredGPUHeadroomBytes = 0 },
		"cpu headroom":          func(p *CapacityProfile) { p.RequiredGPUHeadroomBytes = 1 },
		"invalid device":        func(p *CapacityProfile) { p.Device = "auto" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			p := capacityProfile("cuda", "")
			if name == "cpu headroom" {
				p = capacityProfile("cpu", "")
			}
			mutate(&p)
			if err := p.Validate(); err == nil {
				t.Fatal("invalid profile accepted")
			}
		})
	}
	duplicate := capacityProfile("cpu", "")
	if err := (CapacityProfiles{Schema: CapacityProfilesSchema, Profiles: []CapacityProfile{duplicate, duplicate}}).Validate(); err == nil {
		t.Fatal("duplicate target accepted")
	}
	if err := (CapacityProfiles{Schema: "unknown", Profiles: []CapacityProfile{}}).Validate(); err == nil {
		t.Fatal("unknown schema accepted")
	}
}

func TestLoadCapacityProfilesRejectsUnknownAndTrailingFields(t *testing.T) {
	h := Home{Root: t.TempDir()}
	if err := os.MkdirAll(h.Path("state"), 0o755); err != nil {
		t.Fatal(err)
	}
	valid := `{"schema":"` + CapacityProfilesSchema + `","profiles":[]}`
	for name, data := range map[string]string{
		"valid":         valid,
		"unknown field": strings.TrimSuffix(valid, "}") + `,"max_tokens":100}`,
		"trailing JSON": valid + ` {}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(filepath.Join(h.Root, "state", CapacityProfilesFile), []byte(data), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := h.LoadCapacityProfiles()
			if name == "valid" && err != nil {
				t.Fatal(err)
			}
			if name != "valid" && err == nil {
				t.Fatal("invalid document accepted")
			}
		})
	}
}
