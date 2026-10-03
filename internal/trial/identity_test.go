package trial

import (
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

func observed(f *fakeBackend) map[string]Member {
	out := map[string]Member{}
	for _, m := range f.modules {
		out[m.Module] = m
	}
	return out
}

func identityOf(t *testing.T, w world, plan home.TuningPlan, group, policy, backend string, obs map[string]Member) ComponentIdentity {
	t.Helper()
	for _, g := range plan.Groups {
		if g.ID == group {
			id, err := Identity(w.source, plan, g, policy, backend, obs)
			if err != nil {
				t.Fatal(err)
			}
			return id
		}
	}
	t.Fatalf("no group %s", group)
	return ComponentIdentity{}
}

func TestComponentIdentityIsStableAndContentDerived(t *testing.T) {
	w := newWorld(t)
	obs := observed(newFake(moduleNames(layout())))
	a := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)
	// The same transformation reached through a different profile (another plan
	// that quantizes this group) is the same component: nothing about the profile
	// or any display state is part of the identity.
	b := identityOf(t, w, w.planW4(t, mlp0, mlp1, mlp2), mlp0, home.PolicyW4A16, "b 1", obs)
	if a.ID() != b.ID() || len(a.ID()) != 64 {
		t.Fatalf("component identity depends on the plan: %s vs %s", a.ID(), b.ID())
	}
	for i := 0; i < 5; i++ {
		if again := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs); again.ID() != a.ID() {
			t.Fatal("component identity is not stable")
		}
	}
	if a.Transformation.Scheme != "W4A16" || a.Transformation.Bits != 4 || a.Transformation.GroupSize != 128 || !a.Transformation.Symmetric || a.Transformation.Algorithm != "rtn" {
		t.Fatalf("transformation %+v is not read from the recipe's declaration", a.Transformation)
	}
	if raw := string(a.Canonical()); strings.Contains(raw, "profile") {
		t.Fatalf("identity names a profile: %s", raw)
	}
}

func TestComponentIdentityChangesWithEveryMaterialInput(t *testing.T) {
	w := newWorld(t)
	obs := observed(newFake(moduleNames(layout())))
	base := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)
	differs := map[string]ComponentIdentity{}

	differs["backend"] = identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 2", obs)
	differs["policy"] = identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicySourcePrecision, "b 1", obs)
	differs["group"] = identityOf(t, w, w.planW4(t, mlp0), mlp1, home.PolicyW4A16, "b 1", obs)

	other := w
	other.source.Revision = strings.Repeat("ab", 20)
	differs["source revision"] = identityOf(t, other, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)
	other = w
	other.source.FilesSHA256 = strings64("f")
	differs["source files"] = identityOf(t, other, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)

	shaped := observed(newFake(moduleNames(layout())))
	m := shaped[mod0]
	m.Shape = []int64{256, 512}
	shaped[mod0] = m
	differs["tensor shape"] = identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", shaped)
	shaped = observed(newFake(moduleNames(layout())))
	m = shaped[mod0]
	m.DType = "float16"
	shaped[mod0] = m
	differs["tensor dtype"] = identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", shaped)

	schema := w.planW4(t, mlp0)
	schema.Schema = "hachidori.tuning-plan/2"
	schemaID := base
	schemaID.TuningSchema = schema.Schema
	differs["tuning schema"] = schemaID

	impl := base
	impl.Transformation.Implementation = "hachidori.trial-transform/2"
	differs["transform implementation"] = impl
	params := base
	params.Transformation.GroupSize = 64
	differs["transformation parameter"] = params

	seen := map[string]string{base.ID(): "base"}
	for name, id := range differs {
		if id.ID() == base.ID() {
			t.Errorf("%s does not change the component identity", name)
		}
		if prev, dup := seen[id.ID()]; dup {
			t.Errorf("%s and %s share an identity", name, prev)
		}
		seen[id.ID()] = name
	}
}

func TestACacheKeyedByIdentityNeverCrossesIncompatibleInputs(t *testing.T) {
	w := newWorld(t)
	obs := observed(newFake(moduleNames(layout())))
	c, _ := NewCache(1 << 20)
	a := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)
	if _, err := c.Reserve(a.ID(), a.Group, 100); err != nil || c.Commit(a.ID(), 100) != nil {
		t.Fatal("setup")
	}
	if !c.Lookup(a.ID()) {
		t.Fatal("the same component must hit")
	}
	moved := w
	moved.source.Revision = strings.Repeat("cd", 20)
	if c.Lookup(identityOf(t, moved, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs).ID()) {
		t.Fatal("a component crossed a source revision")
	}
	if c.Lookup(identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 2", obs).ID()) {
		t.Fatal("a component crossed a transformation backend")
	}
}

func TestContractRejectsPoliciesTheOptimizerDoesNotExecute(t *testing.T) {
	w := newWorld(t)
	plan := w.plan(t)
	for _, policy := range []string{"w8a16", "w3a16", "prune", ""} {
		if _, err := Contract(w.source, plan, policy, "b"); err == nil {
			t.Errorf("policy %q has a contract", policy)
		}
	}
	src, err := Contract(w.source, plan, home.PolicySourcePrecision, "b")
	if err != nil || src.Representation != RepresentationDense || src.Bits != 0 {
		t.Fatalf("source-precision contract %+v (%v)", src, err)
	}
	plan.Recipe = "no-such-recipe"
	if _, err := Contract(w.source, plan, home.PolicyW4A16, "b"); err == nil {
		t.Fatal("a plan for an unknown recipe has a contract")
	}
}

func TestComponentIdentityValidation(t *testing.T) {
	w := newWorld(t)
	obs := observed(newFake(moduleNames(layout())))
	good := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)
	for name, mutate := range map[string]func(*ComponentIdentity){
		"schema":     func(c *ComponentIdentity) { c.Schema = "x" },
		"source":     func(c *ComponentIdentity) { c.Source.Revision = "" },
		"group":      func(c *ComponentIdentity) { c.Group = "" },
		"members":    func(c *ComponentIdentity) { c.Members = nil },
		"shape":      func(c *ComponentIdentity) { c.Members = []Member{{Module: "m", DType: "bfloat16"}} },
		"order":      func(c *ComponentIdentity) { c.Members = []Member{good.Members[0], good.Members[0]} },
		"transform":  func(c *ComponentIdentity) { c.Transformation.Implementation = "" },
		"files only": func(c *ComponentIdentity) { c.Members, c.Files = nil, []string{"joint_head.safetensors"} },
	} {
		c := good
		mutate(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: invalid identity accepted", name)
		}
	}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestComponentBytesAreExactFromTensorGeometry(t *testing.T) {
	w := newWorld(t)
	obs := observed(newFake(moduleNames(layout())))
	id := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicyW4A16, "b 1", obs)
	got, err := ComponentBytes(id)
	if err != nil || got != 256*32*4+256*2*2+16 {
		t.Fatalf("component bytes = %d (%v)", got, err)
	}
	dense := identityOf(t, w, w.planW4(t, mlp0), mlp0, home.PolicySourcePrecision, "b 1", obs)
	if n, _ := ComponentBytes(dense); n != 0 {
		t.Fatalf("source-precision material is the canonical source, not a copy: %d", n)
	}
	src, err := SourceBytes(id.Members)
	if err != nil || src != 256*256*2 {
		t.Fatalf("source bytes = %d (%v)", src, err)
	}
	bad := id
	bad.Members = []Member{{Module: "m", Shape: []int64{256, 100}, DType: "bfloat16"}}
	if _, err := ComponentBytes(bad); err == nil {
		t.Fatal("a width that is not a multiple of the group size has bytes")
	}
	if !reflect.DeepEqual(PolicyOfRepresentation(RepresentationPacked), home.PolicyW4A16) || PolicyOfRepresentation("mixed") == home.PolicyW4A16 {
		t.Fatal("representation to policy mapping")
	}
}
