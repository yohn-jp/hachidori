package trial

import (
	"errors"
	"fmt"
	"sort"
)

// Cache is the bookkeeping and eviction authority of the system-RAM tier. The
// bytes themselves live in the resident worker; the cache decides what may be
// admitted, what must be kept and what is evicted, and the session carries
// those decisions out through the Backend. It never guesses sizes: callers
// give exact byte counts.
//
// The policy is deliberately simple and fully deterministic: transformed
// components are evicted least-recently-used first, ties broken by identity.
// Nothing is evicted that is pinned (required by the current Trial, or part of
// the known-good state a rollback needs) or still being transformed, and the
// canonical source bytes are never evictable. A reservation that cannot fit
// even after evicting everything evictable fails without evicting anything.
type Cache struct {
	budget    int64
	canonical int64
	entries   map[string]*entry
	tick      uint64
	st        Stats
}

type entry struct {
	id       string
	group    string
	bytes    int64
	pins     int
	last     uint64
	inflight bool
}

// Stats is the accounting of the RAM tier. Hits and Misses count lookups of
// transformed components by Trials; Evictions and EvictedBytes what the budget
// forced out.
type Stats struct {
	BudgetBytes      int64 `json:"budget_bytes"`
	CurrentBytes     int64 `json:"current_bytes"`
	CanonicalBytes   int64 `json:"canonical_bytes"`
	TransformedBytes int64 `json:"transformed_bytes"`
	Entries          int   `json:"entries"`
	Pinned           int   `json:"pinned_entries"`
	Hits             int64 `json:"hits"`
	Misses           int64 `json:"misses"`
	Evictions        int64 `json:"evictions"`
	EvictedBytes     int64 `json:"evicted_bytes"`
}

// BudgetError is a reservation the budget cannot satisfy: what was needed, what
// the budget leaves, and how much of the rest is pinned or in flight.
type BudgetError struct {
	Need      int64
	Available int64
	Evictable int64
	Budget    int64
}

func (e *BudgetError) Error() string {
	return fmt.Sprintf("the RAM cache budget of %d bytes cannot hold a %d byte component: %d bytes are free and %d more can be evicted; the rest is required by the current or the known-good trial (raise the budget or choose a smaller delta)",
		e.Budget, e.Need, e.Available, e.Evictable)
}

// NewCache creates a cache with an explicit, non-negative byte budget.
func NewCache(budget int64) (*Cache, error) {
	if budget <= 0 {
		return nil, errors.New("the RAM cache budget must be a positive number of bytes")
	}
	return &Cache{budget: budget, entries: map[string]*entry{}}, nil
}

// SetCanonical accounts the canonical source bytes the session keeps resident.
// They count against the budget and are never evictable; a source that does not
// fit is refused rather than silently over-committing RAM.
func (c *Cache) SetCanonical(bytes int64) error {
	if bytes < 0 {
		return errors.New("canonical bytes must not be negative")
	}
	if bytes+c.transformed() > c.budget {
		return &BudgetError{Need: bytes, Available: c.budget - c.transformed(), Budget: c.budget}
	}
	c.canonical = bytes
	return nil
}

func (c *Cache) transformed() int64 {
	var n int64
	for _, e := range c.entries {
		n += e.bytes
	}
	return n
}

// Has reports whether a committed component is resident, without counting a
// lookup or touching its recency.
func (c *Cache) Has(id string) bool {
	e, ok := c.entries[id]
	return ok && !e.inflight
}

// Lookup is one Trial's lookup of a transformed component: a committed entry is
// a hit and becomes the most recently used; anything else is a miss.
func (c *Cache) Lookup(id string) bool {
	e, ok := c.entries[id]
	if !ok || e.inflight {
		c.st.Misses++
		return false
	}
	c.st.Hits++
	c.touch(e)
	return true
}

func (c *Cache) touch(e *entry) {
	c.tick++
	e.last = c.tick
}

// Reserve admits a component that is about to be transformed. The entry is
// pinned and in flight until Commit or Abort, so nothing can evict it. Space is
// made by evicting unpinned committed entries, least recently used first (ties
// by identity); the evicted identities are returned for the caller to release
// from the worker. Nothing is evicted when the reservation cannot succeed.
func (c *Cache) Reserve(id, group string, bytes int64) (evicted []string, err error) {
	if _, ok := c.entries[id]; ok {
		return nil, fmt.Errorf("component %s is already resident or in flight", id)
	}
	if bytes < 0 {
		return nil, errors.New("component bytes must not be negative")
	}
	used := c.canonical + c.transformed()
	free := c.budget - used
	if bytes > free {
		var evictable []*entry
		var evictableBytes int64
		for _, e := range c.entries {
			if e.pins == 0 && !e.inflight {
				evictable = append(evictable, e)
				evictableBytes += e.bytes
			}
		}
		if bytes > free+evictableBytes {
			return nil, &BudgetError{Need: bytes, Available: max(free, 0), Evictable: evictableBytes, Budget: c.budget}
		}
		sort.Slice(evictable, func(i, j int) bool {
			if evictable[i].last != evictable[j].last {
				return evictable[i].last < evictable[j].last
			}
			return evictable[i].id < evictable[j].id
		})
		for _, e := range evictable {
			if bytes <= free {
				break
			}
			free += e.bytes
			delete(c.entries, e.id)
			c.st.Evictions++
			c.st.EvictedBytes += e.bytes
			evicted = append(evicted, e.id)
		}
	}
	e := &entry{id: id, group: group, bytes: bytes, pins: 1, inflight: true}
	c.touch(e)
	c.entries[id] = e
	return evicted, nil
}

// Commit makes a transformed component resident. The bytes the worker actually
// built must equal what was reserved: the accounting is exact or the trial
// fails.
func (c *Cache) Commit(id string, actual int64) error {
	e, ok := c.entries[id]
	if !ok || !e.inflight {
		return fmt.Errorf("component %s has no reservation to commit", id)
	}
	if actual != e.bytes {
		return fmt.Errorf("component %s was built with %d bytes but %d were reserved", id, actual, e.bytes)
	}
	e.inflight = false
	// The reservation's pin becomes the caller's pin; the session unpins it
	// when the trial no longer requires the component.
	return nil
}

// Abort drops an in-flight reservation after a failed transformation. The
// worker discards partial material, so nothing else is released.
func (c *Cache) Abort(id string) {
	if e, ok := c.entries[id]; ok && e.inflight {
		delete(c.entries, id)
	}
}

// Pin protects a committed component from eviction; every Pin is matched by one
// Unpin.
func (c *Cache) Pin(id string) error {
	e, ok := c.entries[id]
	if !ok {
		return fmt.Errorf("component %s is not resident", id)
	}
	e.pins++
	return nil
}

// Unpin releases one pin.
func (c *Cache) Unpin(id string) {
	if e, ok := c.entries[id]; ok && e.pins > 0 {
		e.pins--
	}
}

// Pinned reports whether a resident component is protected from eviction.
func (c *Cache) Pinned(id string) bool {
	e, ok := c.entries[id]
	return ok && e.pins > 0
}

// IDs lists the resident component identities, sorted.
func (c *Cache) IDs() []string {
	ids := make([]string, 0, len(c.entries))
	for id := range c.entries {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Stats is a snapshot of the accounting.
func (c *Cache) Stats() Stats {
	s := c.st
	s.BudgetBytes, s.CanonicalBytes = c.budget, c.canonical
	s.TransformedBytes = c.transformed()
	s.CurrentBytes = s.CanonicalBytes + s.TransformedBytes
	s.Entries = len(c.entries)
	for _, e := range c.entries {
		if e.pins > 0 {
			s.Pinned++
		}
	}
	return s
}
