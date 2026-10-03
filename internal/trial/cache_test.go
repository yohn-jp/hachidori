package trial

import (
	"errors"
	"reflect"
	"testing"
)

func commit(t *testing.T, c *Cache, id string, bytes int64) []string {
	t.Helper()
	ev, err := c.Reserve(id, "g", bytes)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Commit(id, bytes); err != nil {
		t.Fatal(err)
	}
	return ev
}

func TestCacheHitAndMissAreCounted(t *testing.T) {
	c, _ := NewCache(1000)
	commit(t, c, "a", 100)
	if !c.Lookup("a") {
		t.Fatal("a resident component must hit")
	}
	if c.Lookup("b") {
		t.Fatal("an absent component must miss")
	}
	if !c.Lookup("a") {
		t.Fatal("a resident component must hit again")
	}
	if st := c.Stats(); st.Hits != 2 || st.Misses != 1 || st.TransformedBytes != 100 || st.Entries != 1 {
		t.Fatalf("stats %+v", st)
	}
	if c.Has("b") || !c.Has("a") {
		t.Fatal("Has")
	}
}

func TestEvictionIsLeastRecentlyUsedThenByIdentity(t *testing.T) {
	c, _ := NewCache(300)
	for _, id := range []string{"c", "a", "b"} {
		commit(t, c, id, 100)
		c.Unpin(id)
	}
	c.Lookup("c") // recency order, oldest first: a, b, c
	if ev := commit(t, c, "d", 100); !reflect.DeepEqual(ev, []string{"a"}) {
		t.Fatalf("evicted %v, want [a]", ev)
	}
	c.Unpin("d")
	if ev := commit(t, c, "e", 200); !reflect.DeepEqual(ev, []string{"b", "c"}) {
		t.Fatalf("evicted %v, want [b c]", ev)
	}
	st := c.Stats()
	if st.Evictions != 3 || st.EvictedBytes != 300 || st.CurrentBytes != 300 {
		t.Fatalf("stats %+v", st)
	}
	// The same history evicts the same components, every time.
	for run := 0; run < 5; run++ {
		d, _ := NewCache(300)
		for _, id := range []string{"c", "a", "b"} {
			commit(t, d, id, 100)
			d.Unpin(id)
		}
		d.Lookup("c")
		if ev := commit(t, d, "d", 100); !reflect.DeepEqual(ev, []string{"a"}) {
			t.Fatalf("run %d evicted %v", run, ev)
		}
	}
}

func TestAdmissionOrderIsTheRecencyOrder(t *testing.T) {
	// Recency is a logical clock, not a name or a wall time: of two untouched
	// components the one admitted first is evicted first, whatever it is called.
	c, _ := NewCache(200)
	commit(t, c, "z", 100)
	c.Unpin("z")
	commit(t, c, "a", 100)
	c.Unpin("a")
	if ev := commit(t, c, "m", 100); !reflect.DeepEqual(ev, []string{"z"}) {
		t.Fatalf("evicted %v: the older admission goes first", ev)
	}
}

func TestPinnedAndInFlightComponentsAreNeverEvicted(t *testing.T) {
	c, _ := NewCache(300)
	commit(t, c, "pinned", 100) // still holds the pin of its commit
	commit(t, c, "free", 100)
	c.Unpin("free")
	if _, err := c.Reserve("inflight", "g", 100); err != nil { // pinned and in flight
		t.Fatal(err)
	}
	// Room for 100 more is not there: only "free" may go, and 200 are needed.
	_, err := c.Reserve("big", "g", 200)
	var be *BudgetError
	if !errors.As(err, &be) || be.Evictable != 100 {
		t.Fatalf("err = %v", err)
	}
	// A failed reservation evicted nothing.
	if !c.Has("free") || !c.Pinned("pinned") || c.Stats().Evictions != 0 {
		t.Fatalf("failed reservation changed the cache: %+v", c.Stats())
	}
	if ev, err := c.Reserve("fits", "g", 100); err != nil || !reflect.DeepEqual(ev, []string{"free"}) {
		t.Fatalf("evicted %v (%v)", ev, err)
	}
	if !c.Has("pinned") {
		t.Fatal("a pinned component was evicted")
	}
}

func TestCommitMustMatchTheReservationExactly(t *testing.T) {
	c, _ := NewCache(1000)
	if _, err := c.Reserve("a", "g", 100); err != nil {
		t.Fatal(err)
	}
	if err := c.Commit("a", 99); err == nil {
		t.Fatal("a mismatching commit was accepted")
	}
	c.Abort("a")
	if st := c.Stats(); st.Entries != 0 || st.CurrentBytes != 0 {
		t.Fatalf("abort left %+v", st)
	}
	if err := c.Commit("a", 100); err == nil {
		t.Fatal("commit without reservation")
	}
	if _, err := c.Reserve("a", "g", 100); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Reserve("a", "g", 100); err == nil {
		t.Fatal("a second reservation of one identity was accepted")
	}
}

func TestBudgetCountsTheCanonicalSourceAndRefusesToOvercommit(t *testing.T) {
	c, _ := NewCache(1000)
	if err := c.SetCanonical(900); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Reserve("a", "g", 101); err == nil {
		t.Fatal("the canonical source is not counted against the budget")
	}
	if _, err := c.Reserve("a", "g", 100); err != nil {
		t.Fatal(err)
	}
	st := c.Stats()
	if st.CanonicalBytes != 900 || st.TransformedBytes != 100 || st.CurrentBytes != 1000 || st.BudgetBytes != 1000 {
		t.Fatalf("stats %+v", st)
	}
	d, _ := NewCache(1000)
	if err := d.SetCanonical(1001); err == nil {
		t.Fatal("a canonical source larger than the budget was accepted")
	}
	if _, err := NewCache(0); err == nil {
		t.Fatal("a cache without a budget")
	}
}

func TestPinsAreCounted(t *testing.T) {
	c, _ := NewCache(1000)
	commit(t, c, "a", 10)
	if err := c.Pin("a"); err != nil {
		t.Fatal(err)
	}
	c.Unpin("a")
	if !c.Pinned("a") {
		t.Fatal("one pin must remain")
	}
	c.Unpin("a")
	c.Unpin("a") // never below zero
	if c.Pinned("a") {
		t.Fatal("still pinned")
	}
	if err := c.Pin("missing"); err == nil {
		t.Fatal("pinned a component that is not resident")
	}
	if !reflect.DeepEqual(c.IDs(), []string{"a"}) {
		t.Fatalf("ids %v", c.IDs())
	}
}
