package trial

import (
	"errors"
	"reflect"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

func TestDiffIsTheExactBoundedDeltaBetweenResolvedPlans(t *testing.T) {
	w := newWorld(t)
	a, err := StateOf(w.planW4(t, mlp0, mlp1))
	if err != nil {
		t.Fatal(err)
	}
	b, err := StateOf(w.planW4(t, mlp0))
	if err != nil {
		t.Fatal(err)
	}
	d, err := Diff(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(d.Changed, []Change{{Group: mlp1, From: home.PolicyW4A16, To: home.PolicySourcePrecision}}) {
		t.Fatalf("changed %+v", d.Changed)
	}
	total := len(w.plan(t).Groups)
	if len(d.Changed)+len(d.Unchanged) != total || len(d.Unchanged) != total-1 {
		t.Fatalf("%d changed, %d unchanged of %d groups", len(d.Changed), len(d.Unchanged), total)
	}
	same, _ := Diff(a, a)
	if len(same.Changed) != 0 {
		t.Fatalf("a state differs from itself: %+v", same)
	}
}

func TestDiffFromTheBaselineIsEveryQuantizedGroup(t *testing.T) {
	w := newWorld(t)
	target, _ := StateOf(w.planW4(t, mlp0, mlp2))
	d, err := Diff(target.Baseline(), target)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Changed) != 2 || d.Changed[0].Group != mlp0 || d.Changed[1].Group != mlp2 {
		t.Fatalf("changed %+v", d.Changed)
	}
}

func TestDiffRefusesToCrossModelStructures(t *testing.T) {
	w := newWorld(t)
	a, _ := StateOf(w.plan(t))
	plan := w.plan(t)
	plan.Groups = append([]home.TuningGroupPlan(nil), plan.Groups...)
	plan.Groups[0].Modules = append(append([]string(nil), plan.Groups[0].Modules...), "model.extra")
	b, err := StateOf(plan)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Diff(a, b); !errors.Is(err, ErrStructure) {
		t.Fatalf("err = %v", err)
	}
	other := w.plan(t)
	other.Recipe = "another-recipe"
	c, _ := StateOf(other)
	if _, err := Diff(a, c); !errors.Is(err, ErrStructure) {
		t.Fatalf("a delta crossed recipe lineages: %v", err)
	}
	if _, err := StateOf(home.TuningPlan{}); err == nil {
		t.Fatal("an invalid plan has a state")
	}
}
