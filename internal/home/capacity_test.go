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
	cuda := capacityProfile("cuda", "variant-a")
	// A request above the measured bound remains valid configuration; the worker
	// clamps its effective admission limit to the measured value.
	cuda.RequestedMaxInputTokens = 300
	profiles := CapacityProfiles{Schema: CapacityProfilesSchema,
		Profiles: []CapacityProfile{cuda, capacityProfile("cpu", "")}}
	if err := profiles.Validate(); err != nil {
		t.Fatal(err)
	}
	if got, ok := profiles.Find(capacityTarget("cuda", "variant-a")); !ok || got.MaxStateTokens != 200 || got.RequestedMaxInputTokens != 300 {
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
		"missing state limit":      func(p *CapacityProfile) { p.MaxStateTokens = 0 },
		"input below state":        func(p *CapacityProfile) { p.MaxInputTokens = p.MaxStateTokens - 1 },
		"padded below input":       func(p *CapacityProfile) { p.MaxBatchPaddedTokens = p.MaxInputTokens - 1 },
		"negative requested input": func(p *CapacityProfile) { p.RequestedMaxInputTokens = -1 },
		"missing cuda headroom":    func(p *CapacityProfile) { p.RequiredGPUHeadroomBytes = 0 },
		"cpu headroom":             func(p *CapacityProfile) { p.RequiredGPUHeadroomBytes = 1 },
		"invalid device":           func(p *CapacityProfile) { p.Device = "auto" },
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

func TestSaveCapacityProfileInitializesUpsertsAndPreservesOtherTargets(t *testing.T) {
	h := Home{Root: t.TempDir()}
	first := capacityProfile("cuda", "variant-a")
	if err := h.SaveCapacityProfile(first); err != nil {
		t.Fatal(err)
	}
	got, err := h.LoadCapacityProfiles()
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Profiles) != 1 || got.Profiles[0] != first {
		t.Fatalf("profiles = %+v", got.Profiles)
	}

	other := capacityProfile("cuda", "variant-b")
	other.MaxInputTokens = 300
	other.MaxBatchPaddedTokens = 1200
	if err := h.SaveCapacityProfile(other); err != nil {
		t.Fatal(err)
	}
	replacement := first
	replacement.MaxStateTokens = 180
	replacement.MaxInputTokens = 220
	replacement.MaxBatchPaddedTokens = 880
	replacement.RequiredGPUHeadroomBytes = 1234
	replacement.RequestedMaxInputTokens = 120
	if err := h.SaveCapacityProfile(replacement); err != nil {
		t.Fatal(err)
	}
	got, err = h.LoadCapacityProfiles()
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Profiles) != 2 {
		t.Fatalf("profiles = %+v", got.Profiles)
	}
	if p, ok := got.Find(first.CapacityTarget); !ok || p != replacement {
		t.Fatalf("replacement = %+v, %v", p, ok)
	}
	if p, ok := got.Find(other.CapacityTarget); !ok || p != other {
		t.Fatalf("other = %+v, %v", p, ok)
	}
	fi, err := os.Stat(h.Path("state", CapacityProfilesFile))
	if err != nil {
		t.Fatal(err)
	}
	// Native Windows does not preserve POSIX permission bits. On platforms
	// that do, the persisted operator contract is private to the account.
	if os.PathSeparator != '\\' && fi.Mode().Perm()&0o077 != 0 {
		t.Fatalf("capacity profile permissions: %v", fi.Mode().Perm())
	}
}
