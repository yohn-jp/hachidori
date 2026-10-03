package tuning

import (
	"errors"
	"fmt"
	"sort"
)

// Group is one stable, addressable part of the source model: the Linear
// modules (or carried files) of one component of one transformer block, or of
// one block-less part such as the output embeddings. Its ID is a semantic name
// derived only from the declared layout, never from tensor names, and is what
// profiles, plans and evidence refer to. A model whose structure differs has
// different groups and therefore a different analysis digest: a saved profile
// cannot silently apply to it.
type Group struct {
	ID     string `json:"id"`
	Region string `json:"region"`
	// Layer is the transformer block index, or -1 for a group that belongs to
	// no block.
	Layer int `json:"layer"`
	// LayerType is the declared type of the block ("linear_attention" or
	// "full_attention"); empty when Layer is -1.
	LayerType string   `json:"layer_type,omitempty"`
	Modules   []string `json:"modules,omitempty"`
	Files     []string `json:"files,omitempty"`
}

// maxBlocks bounds the block index width of group IDs.
const maxBlocks = 100

// blockComponent is the component name of a region inside a transformer block.
var blockComponent = map[string]string{
	RegionLinearAttention:      "linear-attn",
	RegionLinearAttentionDecay: "linear-attn.decay-gate",
	RegionLinearAttentionBeta:  "linear-attn.beta-gate",
	RegionFullAttention:        "full-attn",
	RegionFeedForward:          "mlp",
}

// groupID is the stable identifier of the group of a region and block.
func groupID(region string, layer int) (string, error) {
	if layer < 0 {
		if _, inBlock := blockComponent[region]; inBlock {
			return "", fmt.Errorf("region %q belongs to a transformer block", region)
		}
		return region, nil
	}
	component, ok := blockComponent[region]
	if !ok {
		return "", fmt.Errorf("region %q has no transformer block", region)
	}
	if layer >= maxBlocks {
		return "", fmt.Errorf("transformer block %d is beyond the supported %d", layer, maxBlocks)
	}
	return fmt.Sprintf("block.%02d.%s", layer, component), nil
}

// deriveGroups partitions a declared layout into stable groups.
func deriveGroups(layout DeclaredLayout) ([]Group, error) {
	byID := map[string]*Group{}
	for _, module := range layout.LinearModules {
		region, layer, err := classifyModule(layout, module)
		if err != nil {
			return nil, err
		}
		id, err := groupID(region, layer)
		if err != nil {
			return nil, err
		}
		g, ok := byID[id]
		if !ok {
			g = &Group{ID: id, Region: region, Layer: layer}
			if layer >= 0 {
				g.LayerType = layout.LayerTypes[layer]
			}
			byID[id] = g
		}
		g.Modules = append(g.Modules, module)
	}
	byID[RegionJointSchemaHead] = &Group{ID: RegionJointSchemaHead, Region: RegionJointSchemaHead, Layer: -1, Files: []string{"joint_head.safetensors"}}
	groups := make([]Group, 0, len(byID))
	for _, g := range byID {
		sort.Strings(g.Modules)
		groups = append(groups, *g)
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].ID < groups[j].ID })
	return groups, nil
}

// validateGroups checks that an analysis' groups are a canonical, exact
// partition of its regions' members, with IDs derived as deriveGroups derives
// them.
func validateGroups(a Analysis) error {
	if len(a.Groups) == 0 {
		return errors.New("tuning analysis has no groups")
	}
	regions := make(map[string]SemanticRegion, len(a.Regions))
	for _, r := range a.Regions {
		regions[r.ID] = r
	}
	members := map[string][]string{}
	files := map[string][]string{}
	last := ""
	for _, g := range a.Groups {
		if g.ID <= last {
			return errors.New("tuning analysis groups are not in canonical order")
		}
		last = g.ID
		if _, ok := regions[g.Region]; !ok {
			return fmt.Errorf("tuning analysis group %q names unknown region %q", g.ID, g.Region)
		}
		want, err := groupID(g.Region, g.Layer)
		if err != nil || want != g.ID {
			return fmt.Errorf("tuning analysis group %q is not the identifier of region %q block %d", g.ID, g.Region, g.Layer)
		}
		if g.Layer >= 0 && g.LayerType == "" || g.Layer < 0 && g.LayerType != "" {
			return fmt.Errorf("tuning analysis group %q has an inconsistent layer type", g.ID)
		}
		if (g.Region == RegionLinearAttention || g.Region == RegionLinearAttentionDecay || g.Region == RegionLinearAttentionBeta) && g.LayerType != "linear_attention" ||
			g.Region == RegionFullAttention && g.LayerType != "full_attention" {
			return fmt.Errorf("tuning analysis group %q conflicts with layer type %q", g.ID, g.LayerType)
		}
		if len(g.Modules)+len(g.Files) == 0 || !isSortedUnique(g.Modules) || !isSortedUnique(g.Files) {
			return fmt.Errorf("tuning analysis group %q members are empty, or not unique and sorted", g.ID)
		}
		members[g.Region] = append(members[g.Region], g.Modules...)
		files[g.Region] = append(files[g.Region], g.Files...)
	}
	for id, r := range regions {
		m, f := append([]string(nil), members[id]...), append([]string(nil), files[id]...)
		sort.Strings(m)
		sort.Strings(f)
		if !equalSorted(m, r.Modules) || !equalSorted(f, r.Files) {
			return fmt.Errorf("tuning analysis groups do not partition region %q", id)
		}
	}
	return nil
}

func equalSorted(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// GroupByID returns the group of an analysis with the given ID.
func (a Analysis) GroupByID(id string) (Group, bool) {
	i := sort.Search(len(a.Groups), func(i int) bool { return a.Groups[i].ID >= id })
	if i < len(a.Groups) && a.Groups[i].ID == id {
		return a.Groups[i], true
	}
	return Group{}, false
}
