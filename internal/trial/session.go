package trial

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Assembly modes.
const (
	AssemblyDifferential = "differential"
	AssemblyReconstruct  = "reconstruct"
)

// Options configure a session. The RAM budget is explicit: nothing is assumed
// about the host.
type Options struct {
	// BudgetBytes bounds the system RAM the session may use for the canonical
	// source and the transformed components together.
	BudgetBytes int64
	// AllowReconstruct permits the explicit slower path when a delta cannot be
	// applied in place. It is recorded in the evidence of every trial that takes
	// it. Without it such a delta is refused.
	AllowReconstruct bool
	// Clock overrides the time source (tests).
	Clock func() time.Time
}

// Session is one tuning session: the resident source and transformed
// components in system RAM, and the model on the accelerator that is changed in
// place from one trial's plan to the next. It runs one trial at a time.
type Session struct {
	mu sync.Mutex

	be       Backend
	source   home.VariantSource
	opt      Options
	cache    *Cache
	opened   Opened
	observed map[string]Member
	groups   map[string][]string

	structure State
	current   State
	// plan is the resolved plan of the last committed trial (nil before the
	// first); components and active describe the committed state.
	plan       *home.TuningPlan
	components map[string]ComponentIdentity // group -> identity of its committed component
	built      map[string]Built             // component ID -> what the worker built
	active     map[string]bool              // component IDs the committed state pins

	trials int
	broken error
	closed bool
	now    func() time.Time
}

// Open starts a session over a resident backend. plan fixes the model structure
// the session's trials address; the resident model starts at its source
// precision and the session verifies that before it accepts a trial.
func Open(ctx context.Context, be Backend, source home.VariantSource, plan home.TuningPlan, opt Options) (*Session, error) {
	state, err := StateOf(plan)
	if err != nil {
		return nil, err
	}
	cache, err := NewCache(opt.BudgetBytes)
	if err != nil {
		return nil, err
	}
	groups := map[string][]string{}
	covered := map[string]bool{}
	for _, g := range plan.Groups {
		groups[g.ID] = append([]string(nil), g.Modules...)
		for _, m := range g.Modules {
			covered[m] = true
		}
	}
	opened, err := be.Open(ctx, groups)
	if err != nil {
		return nil, fmt.Errorf("opening the resident source: %w", err)
	}
	if opened.Device == "" || opened.DType == "" || opened.Backend == "" {
		return nil, errors.New("the resident source did not report its device, dtype and transformation backend")
	}
	s := &Session{be: be, source: source, opt: opt, cache: cache, opened: opened, observed: map[string]Member{}, groups: groups,
		structure: state.Baseline(), components: map[string]ComponentIdentity{}, built: map[string]Built{}, active: map[string]bool{},
		now: opt.Clock}
	if s.now == nil {
		s.now = time.Now
	}
	var unmapped []string
	for _, m := range opened.Modules {
		s.observed[m.Module] = m
		if !covered[m.Module] {
			unmapped = append(unmapped, m.Module)
		}
	}
	sort.Strings(unmapped)
	for _, g := range plan.Groups {
		for _, m := range g.Modules {
			if _, ok := s.observed[m]; !ok {
				return nil, s.openFailed(ctx, fmt.Errorf("group %s names module %s, which the resident source does not have: the plan is not for this model", g.ID, m))
			}
		}
	}
	if len(unmapped) > 0 {
		// A Forge build quantizes every Linear module the recipe does not
		// preserve; a trial changes only plan groups. Modules no group addresses
		// would make the two compute different models, so the session refuses.
		return nil, s.openFailed(ctx, &UnmappedError{Modules: unmapped})
	}
	if err := cache.SetCanonical(opened.CanonicalBytes); err != nil {
		return nil, s.openFailed(ctx, err)
	}
	s.current = s.structure
	obs, err := be.State(ctx)
	if err != nil {
		return nil, s.openFailed(ctx, fmt.Errorf("reading the resident model state: %w", err))
	}
	if err := matches(obs, s.current); err != nil {
		return nil, s.openFailed(ctx, fmt.Errorf("the resident model is not at its source precision: %w", err))
	}
	for _, g := range plan.Groups {
		id, err := s.identity(plan, g, home.PolicySourcePrecision)
		if err != nil {
			return nil, s.openFailed(ctx, err)
		}
		s.components[g.ID] = id
	}
	return s, nil
}

func (s *Session) openFailed(ctx context.Context, err error) error {
	_ = s.be.Close(context.WithoutCancel(ctx))
	return err
}

// UnmappedError lists Linear modules of the resident source that no plan group
// addresses.
type UnmappedError struct{ Modules []string }

func (e *UnmappedError) Error() string {
	shown := e.Modules
	if len(shown) > 3 {
		shown = shown[:3]
	}
	return fmt.Sprintf("%d Linear module(s) of the resident source belong to no tuning group (for example %v): a trial would not compute the model a Forge build of this plan produces", len(e.Modules), shown)
}

func (s *Session) identity(plan home.TuningPlan, g home.TuningGroupPlan, policy string) (ComponentIdentity, error) {
	return Identity(s.source, plan, g, policy, s.opened.Backend, s.observed)
}

// Assembly is the evidence of how a trial's model was put together.
type Assembly struct {
	Mode   string `json:"mode"`
	Reason string `json:"reason,omitempty"`
	// Changed are the groups whose representation was replaced; Reused counts
	// the groups whose accelerator state was left exactly as it was.
	Changed []Change `json:"changed"`
	Reused  int      `json:"reused_groups"`
	// CacheHits and CacheMisses count the transformed components the changed
	// groups needed; Transformed lists the components actually built.
	CacheHits   int      `json:"cache_hits"`
	CacheMisses int      `json:"cache_misses"`
	Transformed []string `json:"transformed_components,omitempty"`
	Evicted     []string `json:"evicted_components,omitempty"`
	// BytesToGPU and BytesReleased are what the assembly moved from system RAM
	// to the accelerator and freed there, as the backend reported them.
	BytesToGPU    int64   `json:"bytes_ram_to_gpu"`
	BytesReleased int64   `json:"bytes_released"`
	AssemblyMS    float64 `json:"assembly_ms"`
}

// ComponentRef names the component a group's modules were composed from.
type ComponentRef struct {
	Group     string `json:"group"`
	Policy    string `json:"policy"`
	Component string `json:"component"`
	Bytes     int64  `json:"bytes"`
	Digest    string `json:"digest,omitempty"`
}

// Resource is the memory the worker itself reported after the trial.
type Resource struct {
	GPUAllocatedBytes *int64 `json:"gpu_allocated_bytes,omitempty"`
	HostRSSBytes      *int64 `json:"host_rss_bytes,omitempty"`
}

// Result is one measured trial.
type Result struct {
	PlanSHA256  string          `json:"plan_sha256"`
	Plan        home.TuningPlan `json:"plan"`
	Assembly    Assembly        `json:"assembly"`
	Components  []ComponentRef  `json:"components"`
	Measurement Measurement     `json:"measurement"`
	Cache       Stats           `json:"cache"`
	Resource    Resource        `json:"resource"`
	// EvaluationMS and TotalMS are the wall-clock time of the evaluation and of
	// the whole trial (preparation, assembly and evaluation): what a physical
	// comparison with a full build-serialize-reload cycle reads.
	EvaluationMS float64 `json:"evaluation_ms"`
	TotalMS      float64 `json:"total_ms"`
}

// Error is a failed or cancelled trial. Restored says the model is back in the
// known-good state of the previous trial; when it is false the session is
// broken. Nothing about a failed trial is ever recorded as Evidence.
type Error struct {
	Phase    string
	Err      error
	Restored bool
	Recovery string // how the model was restored: "unchanged", "reverse-delta" or "reconstruct"
	Lost     error  // why the model could not be restored
}

func (e *Error) Error() string {
	msg := fmt.Sprintf("trial failed in %s: %v", e.Phase, e.Err)
	switch {
	case e.Restored:
		msg += " (the previous model state was restored: " + e.Recovery + ")"
	case e.Lost != nil:
		msg += "; the previous model state could not be restored: " + e.Lost.Error()
	}
	return msg
}

func (e *Error) Unwrap() error { return e.Err }

// RunTrial assembles plan on the resident model, measures it with ev and
// commits it as the new known-good state. It is transactional: on a failure
// to prepare, validate or apply, on an evaluation failure and on cancellation
// the model is returned to the previous trial's state, cached components stay,
// and no Result exists. The canonical source and every Variant are never
// touched.
func (s *Session) RunTrial(ctx context.Context, plan home.TuningPlan, ev Evaluator) (Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Result{}, errors.New("the tuning session is closed")
	}
	if s.broken != nil {
		return Result{}, fmt.Errorf("%w: %v", ErrBroken, s.broken)
	}
	if err := ctx.Err(); err != nil {
		return Result{}, &Error{Phase: "start", Err: err, Restored: true, Recovery: "unchanged"}
	}
	target, err := StateOf(plan)
	if err != nil {
		return Result{}, &Error{Phase: "plan", Err: err, Restored: true, Recovery: "unchanged"}
	}
	delta, err := Diff(s.current, target)
	if err != nil {
		return Result{}, &Error{Phase: "plan", Err: err, Restored: true, Recovery: "unchanged"}
	}
	for _, c := range delta.Changed {
		if g, _ := target.Group(c.Group); len(g.Modules) == 0 {
			return Result{}, &Error{Phase: "plan", Err: fmt.Errorf("group %s carries no modules and cannot change policy", c.Group), Restored: true, Recovery: "unchanged"}
		}
	}

	begin := s.now()
	asm := Assembly{Mode: AssemblyDifferential, Changed: delta.Changed, Reused: len(delta.Unchanged)}
	noCancel := context.WithoutCancel(ctx)

	// held are the pins this trial added; they become the committed state's
	// pins or are released on failure.
	var held []string
	release := func() {
		for _, id := range held {
			s.cache.Unpin(id)
		}
	}

	// 1. Prepare: every changed group's target component, from cache or built.
	comps := make(map[string]ComponentIdentity, len(plan.Groups))
	for _, g := range plan.Groups {
		comps[g.ID] = s.components[g.ID]
	}
	for _, c := range delta.Changed {
		g, _ := target.Group(c.Group)
		id, err := s.identity(plan, g, c.To)
		if err != nil {
			release()
			return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
		}
		comps[c.Group] = id
		if c.To == home.PolicySourcePrecision {
			continue // the canonical source is the material
		}
		cid := id.ID()
		if s.cache.Lookup(cid) {
			asm.CacheHits++
			if err := s.cache.Pin(cid); err != nil {
				release()
				return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
			}
			held = append(held, cid)
			continue
		}
		asm.CacheMisses++
		if err := ctx.Err(); err != nil {
			release()
			return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
		}
		want, err := ComponentBytes(id)
		if err != nil {
			release()
			return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
		}
		evicted, err := s.cache.Reserve(cid, c.Group, want)
		if err != nil {
			release()
			return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
		}
		for _, e := range evicted {
			asm.Evicted = append(asm.Evicted, e)
			delete(s.built, e)
			if err := s.be.Release(noCancel, e); err != nil {
				// The worker may still hold bytes the cache no longer
				// accounts: the RAM accounting is no longer trustworthy.
				s.cache.Abort(cid)
				release()
				s.broken = fmt.Errorf("releasing evicted component %s: %w", e, err)
				return Result{}, &Error{Phase: "prepare", Err: s.broken, Restored: true, Recovery: "unchanged", Lost: s.broken}
			}
		}
		built, err := s.be.Transform(noCancel, id)
		if err == nil && built.Bytes != want {
			err = fmt.Errorf("the worker built %d bytes for component %s, the geometry says %d", built.Bytes, cid, want)
			_ = s.be.Release(noCancel, cid)
		}
		if err != nil {
			s.cache.Abort(cid)
			release()
			return Result{}, &Error{Phase: "prepare", Err: fmt.Errorf("transforming group %s: %w", c.Group, err), Restored: true, Recovery: "unchanged"}
		}
		if err := s.cache.Commit(cid, built.Bytes); err != nil {
			s.cache.Abort(cid)
			_ = s.be.Release(noCancel, cid)
			release()
			return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
		}
		s.built[cid] = built
		held = append(held, cid)
		asm.Transformed = append(asm.Transformed, cid)
	}
	if err := ctx.Err(); err != nil {
		release()
		return Result{}, &Error{Phase: "prepare", Err: err, Restored: true, Recovery: "unchanged"}
	}

	// 2. Validate: the in-place replacement must be provably the requested
	// representation, or the explicit reconstruction is chosen, or the trial is
	// refused. Nothing has changed on the accelerator yet.
	reps := s.replacements(delta.Changed, target, comps)
	verr := s.be.Validate(noCancel, reps)
	apply := s.be.Apply
	switch {
	case verr == nil:
	case isUnsupported(verr) && s.opt.AllowReconstruct:
		asm.Mode, asm.Reason = AssemblyReconstruct, verr.Error()
		reps = s.replacements(allChanges(target), target, comps)
		apply = s.be.Reconstruct
	default:
		release()
		return Result{}, &Error{Phase: "validate", Err: verr, Restored: true, Recovery: "unchanged"}
	}

	// 3. Apply the delta (or the reconstruction) and read back what the model
	// actually is.
	applied, aerr := apply(noCancel, reps)
	if aerr == nil {
		var obs Observed
		if obs, aerr = s.be.State(noCancel); aerr == nil {
			if aerr = matches(obs, target); aerr != nil {
				aerr = fmt.Errorf("the applied model is not the requested plan: %w", aerr)
			}
		}
	}
	if aerr != nil {
		release()
		return Result{}, s.recover(noCancel, "apply", aerr)
	}
	asm.BytesToGPU, asm.BytesReleased = applied.BytesToGPU, applied.BytesReleased
	asm.AssemblyMS = float64(s.now().Sub(begin)) / float64(time.Millisecond)

	// 4. Evaluate the assembled trial.
	res := Result{PlanSHA256: plan.SHA256(), Plan: plan, Assembly: asm}
	evalStart := s.now()
	m, eerr := ev.Evaluate(ctx, Assembled{PlanSHA256: res.PlanSHA256, Policies: clonePolicies(target.Policies)})
	if eerr == nil {
		eerr = ctx.Err()
	}
	if eerr == nil {
		eerr = m.Validate()
	}
	if eerr != nil {
		release()
		return Result{}, s.restore(noCancel, "evaluate", eerr, delta.Changed)
	}
	res.Measurement = m
	end := s.now()
	res.EvaluationMS = float64(end.Sub(evalStart)) / float64(time.Millisecond)
	res.TotalMS = float64(end.Sub(begin)) / float64(time.Millisecond)

	// 5. Commit: the trial becomes the known-good state.
	obs, serr := s.be.State(noCancel)
	if serr == nil {
		res.Resource = Resource{GPUAllocatedBytes: obs.GPUAllocated, HostRSSBytes: obs.HostRSS}
	}
	newActive := map[string]bool{}
	for _, g := range plan.Groups {
		c := comps[g.ID]
		if c.Transformation.Representation == RepresentationPacked {
			newActive[c.ID()] = true
		}
	}
	for id := range s.active {
		if !newActive[id] {
			s.cache.Unpin(id)
		}
	}
	s.active, s.current, s.components = newActive, target, comps
	p := plan
	s.plan = &p
	s.trials++
	res.Components = s.refs(plan)
	res.Cache = s.cache.Stats()
	return res, nil
}

// replacements are the backend requests for the changed groups.
func (s *Session) replacements(changes []Change, target State, comps map[string]ComponentIdentity) []Replacement {
	reps := make([]Replacement, 0, len(changes))
	for _, c := range changes {
		g, _ := target.Group(c.Group)
		id := comps[c.Group]
		r := Replacement{Group: c.Group, Policy: c.To, Modules: append([]string(nil), g.Modules...), Transformation: id.Transformation}
		if id.Transformation.Representation == RepresentationPacked {
			r.Component = id.ID()
		}
		reps = append(reps, r)
	}
	return reps
}

// allChanges is every group with modules, for a full reconstruction.
func allChanges(target State) []Change {
	ids := make([]string, 0, len(target.Policies))
	for id := range target.Policies {
		if g, _ := target.Group(id); len(g.Modules) > 0 {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	out := make([]Change, 0, len(ids))
	for _, id := range ids {
		out = append(out, Change{Group: id, To: target.Policies[id]})
	}
	return out
}

func isUnsupported(err error) bool {
	var u *UnsupportedError
	return errors.As(err, &u)
}

// matches compares the observed model with a state: every group with modules
// must observably realize exactly its policy.
func matches(obs Observed, want State) error {
	for id, policy := range want.Policies {
		g, _ := want.Group(id)
		if len(g.Modules) == 0 {
			continue
		}
		if got := obs.Policies[id]; got != policy {
			return fmt.Errorf("group %s is %q, not %q", id, got, policy)
		}
	}
	return nil
}

func clonePolicies(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// refs lists the component of every group of the committed state.
func (s *Session) refs(plan home.TuningPlan) []ComponentRef {
	refs := make([]ComponentRef, 0, len(plan.Groups))
	for _, g := range plan.Groups {
		id := s.components[g.ID]
		cid := id.ID()
		r := ComponentRef{Group: g.ID, Policy: id.Transformation.Policy, Component: cid}
		if b, ok := s.built[cid]; ok {
			r.Bytes, r.Digest = b.Bytes, b.Digest
		}
		refs = append(refs, r)
	}
	return refs
}

// recover handles a failed Apply: the model is either unchanged (a clean
// failure), or its state is unknown and is rebuilt from RAM.
func (s *Session) recover(ctx context.Context, phase string, cause error) error {
	obs, err := s.be.State(ctx)
	var lost *StateLostError
	if err == nil && !errors.As(cause, &lost) && matches(obs, s.current) == nil {
		return &Error{Phase: phase, Err: cause, Restored: true, Recovery: "unchanged"}
	}
	return s.reconstructKnownGood(ctx, phase, cause)
}

// restore returns the model to the known-good state after an evaluation
// failure or cancellation, by the reverse delta from RAM, and by full
// reconstruction when that fails.
func (s *Session) restore(ctx context.Context, phase string, cause error, changed []Change) error {
	back := make([]Change, 0, len(changed))
	for _, c := range changed {
		back = append(back, Change{Group: c.Group, From: c.To, To: c.From})
	}
	reps := s.replacements(back, s.current, s.components)
	if _, err := s.be.Apply(ctx, reps); err == nil {
		if obs, err := s.be.State(ctx); err == nil && matches(obs, s.current) == nil {
			return &Error{Phase: phase, Err: cause, Restored: true, Recovery: "reverse-delta"}
		}
	}
	return s.reconstructKnownGood(ctx, phase, cause)
}

func (s *Session) reconstructKnownGood(ctx context.Context, phase string, cause error) error {
	lose := func(err error) error {
		s.broken = err
		return &Error{Phase: phase, Err: cause, Lost: err}
	}
	reps := s.replacements(allChanges(s.current), s.current, s.components)
	if _, err := s.be.Reconstruct(ctx, reps); err != nil {
		return lose(err)
	}
	obs, err := s.be.State(ctx)
	if err != nil {
		return lose(err)
	}
	if err := matches(obs, s.current); err != nil {
		return lose(err)
	}
	return &Error{Phase: phase, Err: cause, Restored: true, Recovery: "reconstruct"}
}

// SessionStats is the session's accounting.
type SessionStats struct {
	Cache  Stats `json:"cache"`
	Trials int   `json:"trials"`
	Broken bool  `json:"broken"`
}

// Stats reports the cache accounting and the number of committed trials.
func (s *Session) Stats() SessionStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return SessionStats{Cache: s.cache.Stats(), Trials: s.trials, Broken: s.broken != nil}
}

// Opened reports what the resident source declared at the start.
func (s *Session) Opened() Opened { return s.opened }

// Current is the effective policy of every group of the committed state.
func (s *Session) Current() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return clonePolicies(s.current.Policies)
}

// Close ends the session: the worker's hold on RAM and accelerator is
// released. It is safe to call more than once.
func (s *Session) Close(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.be.Close(context.WithoutCancel(ctx))
}
