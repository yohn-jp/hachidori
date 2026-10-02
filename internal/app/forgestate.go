package app

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// The records of the Forge readiness operations live under
// HACHIDORI_HOME/state/forge: the latest preflight per target and the latest
// probe per variant and device. They are evidence about an operation, never
// inputs to activation or certification, and are read back by the CLI and the
// desktop through this one authority.

func forgeDir(h home.Home, kind string) string { return h.Path("state", "forge", kind) }

// preflightName names the latest-preflight record of a target.
func preflightName(r setup.PreflightReport) string {
	parts := []string{r.Kind, r.Model}
	if r.Variant != "" {
		parts = append(parts, r.Variant)
	}
	if r.Device != "" {
		parts = append(parts, r.Device)
	}
	return safeName(strings.Join(parts, "--")) + ".json"
}

// safeName keeps a record name to plain identifier characters.
func safeName(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '.', r == '_':
			return r
		}
		return '_'
	}, s)
}

// SavePreflight records a preflight report as the latest of its target.
func SavePreflight(h home.Home, r setup.PreflightReport) error {
	dir := forgeDir(h, "preflight")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return home.WriteFileAtomic(filepath.Join(dir, preflightName(r)), append(b, '\n'), 0o644)
}

// LatestPreflight returns the most recently recorded preflight report that
// matches kind (and model, variant when given), or false.
func LatestPreflight(h home.Home, kind, model, variant string) (setup.PreflightReport, bool) {
	var best setup.PreflightReport
	found := false
	for _, f := range readDirJSON(forgeDir(h, "preflight")) {
		var r setup.PreflightReport
		if json.Unmarshal(f, &r) != nil || r.Schema != setup.PreflightSchema {
			continue
		}
		if (kind != "" && r.Kind != kind) || (model != "" && r.Model != model) || (variant != "" && r.Variant != variant) {
			continue
		}
		if !found || r.CreatedAt > best.CreatedAt {
			best, found = r, true
		}
	}
	return best, found
}

// SaveProbe records a probe as the latest of its variant and device.
func SaveProbe(h home.Home, r ProbeRecord) error {
	dir := forgeDir(h, "probe")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return home.WriteFileAtomic(filepath.Join(dir, safeName(r.Variant+"--"+r.Device)+".json"), append(b, '\n'), 0o644)
}

// LatestProbe returns the most recent probe record of variantID (on any
// device), or false.
func LatestProbe(h home.Home, variantID string) (ProbeRecord, bool) {
	var best ProbeRecord
	found := false
	for _, f := range readDirJSON(forgeDir(h, "probe")) {
		var r ProbeRecord
		if json.Unmarshal(f, &r) != nil || r.Schema != ProbeSchema || r.Variant != variantID {
			continue
		}
		if !found || r.StartedAt > best.StartedAt {
			best, found = r, true
		}
	}
	return best, found
}

// readDirJSON reads every regular .json file of dir, in name order. A missing
// directory is empty.
func readDirJSON(dir string) [][]byte {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	var out [][]byte
	for _, e := range entries {
		if !e.Type().IsRegular() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		if b, err := os.ReadFile(filepath.Join(dir, e.Name())); err == nil {
			out = append(out, b)
		}
	}
	return out
}

// statSize is the size of the regular file rel under dir.
func statSize(dir, rel string) (uint64, error) {
	fi, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil || !fi.Mode().IsRegular() {
		return 0, errors.New("not a regular file")
	}
	return uint64(fi.Size()), nil
}

// ForgeState is what the home records about the Forge readiness operations:
// the latest preflight per target, the latest probe per variant and device,
// and the newest failure diagnostics. The desktop and the CLI read it from
// here; nothing in it is kept by a front end.
type ForgeState struct {
	Preflights  []setup.PreflightReport    `json:"preflights"`
	Probes      []ProbeRecord              `json:"probes"`
	Diagnostics []diagnostics.ForgeSummary `json:"diagnostics"`
}

// maxForgeListed bounds each list of ForgeState.
const maxForgeListed = 8

// ReadForgeState reads the Forge records of the home. It only reads.
func ReadForgeState(h home.Home) ForgeState {
	st := ForgeState{Preflights: []setup.PreflightReport{}, Probes: []ProbeRecord{}, Diagnostics: []diagnostics.ForgeSummary{}}
	for _, b := range readDirJSON(forgeDir(h, "preflight")) {
		var r setup.PreflightReport
		if json.Unmarshal(b, &r) == nil && r.Schema == setup.PreflightSchema {
			st.Preflights = append(st.Preflights, r)
		}
	}
	sort.SliceStable(st.Preflights, func(i, j int) bool { return st.Preflights[i].CreatedAt > st.Preflights[j].CreatedAt })
	for _, b := range readDirJSON(forgeDir(h, "probe")) {
		var r ProbeRecord
		if json.Unmarshal(b, &r) == nil && r.Schema == ProbeSchema {
			st.Probes = append(st.Probes, r)
		}
	}
	sort.SliceStable(st.Probes, func(i, j int) bool { return st.Probes[i].StartedAt > st.Probes[j].StartedAt })
	st.Diagnostics = append(st.Diagnostics, diagnostics.ListForge(h.Root, diagnostics.ForgeFilter{})...)
	if len(st.Preflights) > maxForgeListed {
		st.Preflights = st.Preflights[:maxForgeListed]
	}
	if len(st.Probes) > maxForgeListed*2 {
		st.Probes = st.Probes[:maxForgeListed*2]
	}
	if len(st.Diagnostics) > maxForgeListed {
		st.Diagnostics = st.Diagnostics[:maxForgeListed]
	}
	return st
}

// Forge reads the Forge records of the selected home; it is empty while no
// home is selected.
func (c *Controller) Forge() ForgeState {
	c.mu.Lock()
	root := c.home
	c.mu.Unlock()
	if root == "" {
		return ForgeState{Preflights: []setup.PreflightReport{}, Probes: []ProbeRecord{}, Diagnostics: []diagnostics.ForgeSummary{}}
	}
	return ReadForgeState(home.Home{Root: root})
}
