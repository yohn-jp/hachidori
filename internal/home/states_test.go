package home

import (
	"os"
	"strings"
	"sync"
	"testing"
)

func TestStateRegistration(t *testing.T) {
	h := Home{Root: t.TempDir()}
	state := "policy\n  keep whitespace\n"
	ref, err := h.RegisterState(state)
	if err != nil {
		t.Fatal(err)
	}
	if ref != StateRef(state) || ref == StateRef(strings.TrimSpace(state)) {
		t.Fatalf("unexpected identity %s", ref)
	}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, e := h.RegisterState(state)
			if e != nil || got != ref {
				t.Errorf("duplicate registration: %s %v", got, e)
			}
		}()
	}
	wg.Wait()
	got, err := h.ResolveState(ref)
	if err != nil || got != state {
		t.Fatalf("resolve: %q %v", got, err)
	}
	if _, err = h.ResolveState(StateRef("other")); err == nil {
		t.Fatal("unknown reference accepted")
	}
	if _, err = h.ResolveState("../active-runtime.json"); err == nil {
		t.Fatal("invalid reference accepted")
	}
	path := h.Path("state", "registered", strings.TrimPrefix(ref, "sha256:"))
	if err = os.WriteFile(path, []byte("other"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = h.ResolveState(ref); err == nil {
		t.Fatal("corrupt reference accepted")
	}
}
