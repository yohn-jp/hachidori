package worker

import (
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
)

func TestCoalesceStates(t *testing.T) {
	q := func(id string) api.Question {
		return api.Question{ID: id, Type: "choice", Instructions: "answer", Choices: []string{"yes", "no"}}
	}
	a := Item{State: "policy", StateRef: "sha256:one", Questions: []api.Question{q("a")}}
	b := Item{State: "policy", StateRef: "sha256:one", Questions: []api.Question{q("b")}}
	pending := []batchRequest{{items: []Item{a}}, {items: []Item{b}}}
	keyA, _, _ := batchMetadata([]Item{a})
	keyB, _, _ := batchMetadata([]Item{b})
	if keyA != keyB {
		t.Fatal("same State has different scheduler keys")
	}
	out, counts, ok := coalesceStates(pending, []Item{a, b})
	if !ok || len(out) != 1 || len(out[0].Questions) != 2 || out[0].Questions[0].ID != "a" || out[0].Questions[1].ID != "b" || counts[0] != 1 || counts[1] != 1 {
		t.Fatalf("coalescing: %#v %#v %v", out, counts, ok)
	}
	other := b
	other.StateRef = "sha256:two"
	otherKey, _, _ := batchMetadata([]Item{other})
	if otherKey == keyA {
		t.Fatal("different States share scheduler key")
	}
	if _, _, ok = coalesceStates(pending, []Item{a, other}); ok {
		t.Fatal("cross-State coalescing")
	}
	dup := b
	dup.Questions = []api.Question{q("a")}
	if _, _, ok = coalesceStates(pending, []Item{a, dup}); ok {
		t.Fatal("duplicate IDs must not coalesce")
	}
	inline := a
	inline.StateRef = ""
	if _, _, ok = coalesceStates(pending, []Item{inline, b}); ok {
		t.Fatal("inline path changed")
	}
}
