package trial

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"github.com/yohn-jp/hachidori/internal/home"
)

// State is the effective policy of every group of one model structure: the
// resolved plan reduced to what a Trial's GPU composition depends on. It is
// derived only from a home.TuningPlan; there is no second policy
// representation.
type State struct {
	// Structure is the digest of the plan's schema, recipe lineage and group
	// layout. Two states are comparable only when it is equal: a delta never
	// crosses model structures.
	Structure string
	Policies  map[string]string // group ID -> effective policy
	groups    map[string]home.TuningGroupPlan
}

// StateOf is the State of a resolved plan.
func StateOf(plan home.TuningPlan) (State, error) {
	if err := plan.Validate(); err != nil {
		return State{}, fmt.Errorf("tuning plan: %w", err)
	}
	h := sha256.New()
	fmt.Fprintf(h, "%s\n%s\n", plan.Schema, plan.Recipe)
	s := State{Policies: make(map[string]string, len(plan.Groups)), groups: make(map[string]home.TuningGroupPlan, len(plan.Groups))}
	for _, g := range plan.Groups {
		fmt.Fprintf(h, "group %s %s %d\n", g.ID, g.Region, g.Layer)
		for _, m := range g.Modules {
			fmt.Fprintf(h, " module %s\n", m)
		}
		for _, f := range g.Files {
			fmt.Fprintf(h, " file %s\n", f)
		}
		s.Policies[g.ID] = g.Effective
		s.groups[g.ID] = g
	}
	s.Structure = hex.EncodeToString(h.Sum(nil))
	return s, nil
}

// Baseline is the state with every group at its source precision: what the
// resident worker holds before any trial.
func (s State) Baseline() State {
	b := State{Structure: s.Structure, Policies: make(map[string]string, len(s.Policies)), groups: s.groups}
	for id := range s.Policies {
		b.Policies[id] = home.PolicySourcePrecision
	}
	return b
}

// Group returns the plan group with the given ID.
func (s State) Group(id string) (home.TuningGroupPlan, bool) {
	g, ok := s.groups[id]
	return g, ok
}

// Change is one group whose policy differs between two states.
type Change struct {
	Group string `json:"group"`
	From  string `json:"from"`
	To    string `json:"to"`
}

// Delta is the exact difference between two states of one structure. Unchanged
// groups keep their GPU representation untouched.
type Delta struct {
	Changed   []Change `json:"changed"`
	Unchanged []string `json:"unchanged"`
}

// ErrStructure is a delta across different model structures or plan schemas.
var ErrStructure = errors.New("the plans address different model structures")

// Diff computes the delta from one state to another, by group ID order.
func Diff(from, to State) (Delta, error) {
	if from.Structure != to.Structure {
		return Delta{}, ErrStructure
	}
	var d Delta
	ids := make([]string, 0, len(to.Policies))
	for id := range to.Policies {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		was, ok := from.Policies[id]
		if !ok {
			return Delta{}, fmt.Errorf("group %s is not in the state being left: %w", id, ErrStructure)
		}
		if was == to.Policies[id] {
			d.Unchanged = append(d.Unchanged, id)
			continue
		}
		d.Changed = append(d.Changed, Change{Group: id, From: was, To: to.Policies[id]})
	}
	return d, nil
}
