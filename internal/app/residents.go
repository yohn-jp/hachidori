package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Rejections of the per-resident actions and routes.
var (
	ErrNoResidents     = errors.New("the bound runtime is not a resident set")
	ErrUnknownResident = errors.New("model is not a member of the resident set")
)

// Resident is one member of a ResidentSet: a catalog model with its own
// supervisor and therefore its own worker process, PID, lifecycle, failure
// boundary, restart budget, counters, latency state and accelerator state.
// Nothing is shared with the other members except the home and the private
// Python runtime directory they were all launched from. Requests to a
// resident are serialized by its own worker.Process exactly as for a single
// resident worker.
type Resident struct {
	Model      string // stable Hachidori model identity (catalog ID)
	Provider   string // provider kind that loads it
	Info       server.Runtime
	Supervisor *worker.Supervisor
	lc         *worker.Lifecycle
}

// Decide runs items on this resident's worker.
func (r *Resident) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	return r.Supervisor.Decide(items)
}

// ResidentMember declares one member of a set: its identity and how to launch
// its worker. Config is not shared between members.
type ResidentMember struct {
	Model    string
	Provider string
	Info     server.Runtime
	Config   worker.Config
}

// ResidentStatus is the per-resident view in Snapshot.Residents and in the
// /v1/status document: the status of that resident's own supervisor.
type ResidentStatus = server.ResidentStatus

// ResidentRuntime is a Runtime made of independently supervised residents.
// Start, Stop and Restart act on the whole set; the per-resident methods act
// on exactly one member and never touch another.
type ResidentRuntime interface {
	Runtime
	ResidentStatuses() []ResidentStatus
	StartResident(model string) (started bool, err error)
	StopResident(model string) error
	RestartResident(model string) error
}

// ResidentSet is the keyed resident-worker set: one Resident per model, each
// with an independent Supervisor and worker.Lifecycle. It is the production
// ResidentRuntime. The default resident is the compatibility route: callers
// that name no model (Decide, the server.Decider methods, Status) get it,
// which is the active model of the home.
//
// A resident that fails, crashes or is stopped does not change any other
// resident: failure, restart and stop are per member, and the set never
// restarts, evicts or reloads a member on its own.
type ResidentSet struct {
	def     *Resident
	order   []*Resident
	byModel map[string]*Resident
	started time.Time

	mu     sync.RWMutex
	router *route.Router
}

var (
	_ ResidentRuntime = (*ResidentSet)(nil)
	_ server.Decider  = (*ResidentSet)(nil)
	_ server.Router   = (*ResidentSet)(nil)

	_ server.AutoRouting     = (*ResidentSet)(nil)
	_ server.RoutingReporter = (*ResidentSet)(nil)
	_ route.Backend          = (*ResidentSet)(nil)
)

// NewResidentSet builds the set; every worker it starts ends when parent
// does. def names the default resident and must be a member. Nothing starts
// until Start.
func NewResidentSet(parent context.Context, policy worker.Policy, def string, members []ResidentMember) (*ResidentSet, error) {
	if len(members) == 0 {
		return nil, errors.New("resident set has no members")
	}
	s := &ResidentSet{byModel: map[string]*Resident{}, started: time.Now()}
	for _, m := range members {
		if m.Model == "" {
			return nil, errors.New("resident set member has no model identity")
		}
		if _, dup := s.byModel[m.Model]; dup {
			return nil, fmt.Errorf("model %s is listed twice in the resident set", m.Model)
		}
		sup := worker.NewSupervisor(m.Config, policy)
		r := &Resident{Model: m.Model, Provider: m.Provider, Info: m.Info, Supervisor: sup, lc: worker.NewLifecycle(parent, sup)}
		s.byModel[m.Model] = r
		s.order = append(s.order, r)
	}
	s.def = s.byModel[def]
	if s.def == nil {
		return nil, fmt.Errorf("default model %s is not a member of the resident set", def)
	}
	return s, nil
}

// Models lists the member model IDs, default first.
func (s *ResidentSet) Models() []string {
	out := make([]string, len(s.order))
	for i, r := range s.order {
		out[i] = r.Model
	}
	return out
}

// Default is the compatibility route.
func (s *ResidentSet) Default() *Resident { return s.def }

// Resident returns the member serving model.
func (s *ResidentSet) Resident(model string) (*Resident, bool) {
	r, ok := s.byModel[model]
	return r, ok
}

// Runtime describes the default resident's runtime, for the API handler.
func (s *ResidentSet) Runtime() server.Runtime { return s.def.Info }

// Started is when the set was built.
func (s *ResidentSet) Started() time.Time { return s.started }

// DecideOn runs items on exactly the named resident ("" is the default). It
// is a direct call: it never starts, restarts or reloads any worker, never
// answers from another resident, and a resident that is not READY answers
// not_ready for itself only (naming the model). A model that is not a member
// is a request_invalid error.
func (s *ResidentSet) DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error) {
	if model == "" {
		return s.def.Decide(items)
	}
	r, ok := s.byModel[model]
	if !ok {
		return nil, 0, &worker.RequestError{Class: api.ErrRequestInvalid, Message: "model " + model + " is not resident"}
	}
	res, ms, err := r.Decide(items)
	var re *worker.RequestError
	if errors.As(err, &re) && re.Class == api.ErrNotReady {
		return nil, 0, &worker.RequestError{Class: api.ErrNotReady, Message: "model " + model + ": " + re.Message}
	}
	return res, ms, err
}

// Identity is the catalog identity of a member, for response provenance.
func (s *ResidentSet) Identity(model string) (api.Served, bool) {
	r, ok := s.byModel[model]
	if !ok {
		return api.Served{}, false
	}
	return api.Served{Model: r.Model, Provider: r.Provider}, true
}

// SetRouting binds a routing policy over the set's residents: requests that
// ask for route "auto" are then routed by it. Every model the policy names
// must be a member. The policy changes nothing about the members or the
// direct and default routes, and a set without a policy refuses routed
// requests. SetRouting replaces a previously bound policy.
func (s *ResidentSet) SetRouting(p route.Policy) (*route.Router, error) {
	r, err := route.New(p, s)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.router = r
	s.mu.Unlock()
	return r, nil
}

// AutoRouter is the bound router, nil when no policy is bound.
func (s *ResidentSet) AutoRouter() *route.Router {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.router
}

// RoutingStatus is the counters of the bound policy, nil when none is bound.
func (s *ResidentSet) RoutingStatus() *route.Status {
	r := s.AutoRouter()
	if r == nil {
		return nil
	}
	st := r.Status()
	return &st
}

// Decide, Ready, State and Snapshot are the default resident's: the existing
// single-model callers (the HTTP API) keep working unchanged.
func (s *ResidentSet) Decide(items []worker.Item) ([][]api.Result, float64, error) {
	return s.def.Decide(items)
}
func (s *ResidentSet) Ready() bool               { return s.def.Supervisor.Ready() }
func (s *ResidentSet) State() string             { return s.def.Supervisor.State() }
func (s *ResidentSet) Snapshot() worker.Snapshot { return s.def.Supervisor.Snapshot() }

// Status is the /v1/status document: the default resident's own fields plus
// every resident in Residents.
func (s *ResidentSet) Status() server.Status {
	st := server.StatusBody(s.def.Supervisor, s.def.Info, s.started)
	st.Residents = s.ResidentStatuses()
	st.Routing = s.RoutingStatus()
	return st
}

// ResidentStatuses is the status of every member, default first.
func (s *ResidentSet) ResidentStatuses() []ResidentStatus {
	out := make([]ResidentStatus, len(s.order))
	for i, r := range s.order {
		out[i] = ResidentStatus{Model: r.Model, Provider: r.Provider, Default: r == s.def,
			Running: r.lc.Running(), Status: server.StatusBody(r.Supervisor, r.Info, s.started)}
	}
	return out
}

// Start starts every member that is not running and reports whether it
// started any. A running member is never started twice.
func (s *ResidentSet) Start() bool {
	started := false
	for _, r := range s.order {
		if r.lc.Start() {
			started = true
		}
	}
	return started
}

// Stop stops every member and waits for all of them.
func (s *ResidentSet) Stop() { s.each((*worker.Lifecycle).Stop) }

// Restart restarts every member.
func (s *ResidentSet) Restart() { s.each((*worker.Lifecycle).Restart) }

// each runs op on every member concurrently (a worker's bounded shutdown
// must not add up across members) and waits for all.
func (s *ResidentSet) each(op func(*worker.Lifecycle)) {
	var wg sync.WaitGroup
	for _, r := range s.order {
		wg.Add(1)
		go func() { defer wg.Done(); op(r.lc) }()
	}
	wg.Wait()
}

// Running reports whether any member's supervisor is running.
func (s *ResidentSet) Running() bool {
	for _, r := range s.order {
		if r.lc.Running() {
			return true
		}
	}
	return false
}

func (s *ResidentSet) member(model string) (*Resident, error) {
	r, ok := s.byModel[model]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownResident, model)
	}
	return r, nil
}

// StartResident starts one member; false when it was already running.
func (s *ResidentSet) StartResident(model string) (bool, error) {
	r, err := s.member(model)
	if err != nil {
		return false, err
	}
	return r.lc.Start(), nil
}

// StopResident stops one member and waits for it. The others are untouched.
func (s *ResidentSet) StopResident(model string) error {
	r, err := s.member(model)
	if err != nil {
		return err
	}
	r.lc.Stop()
	return nil
}

// RestartResident restarts one member. A member the supervisor gave up on
// begins with a fresh restart budget, as for an operator restart of a single
// worker. The others are untouched.
func (s *ResidentSet) RestartResident(model string) error {
	r, err := s.member(model)
	if err != nil {
		return err
	}
	r.lc.Restart()
	return nil
}

// OpenResidents builds the resident set of the home: the active model (the
// default resident) plus the extra catalog models, each launched from the
// active runtime on the active device through server.ResidentConfig. Extra
// models must be catalog IDs; an arbitrary repository or revision cannot be
// named.
//
// A default model that cannot be launched fails the open, as it does for a
// single worker. An extra model that is not materialized or fails preflight
// does not fail the others: its member comes up failed with the cause, naming
// the model and provider, and never reports READY.
func OpenResidents(parent context.Context, h home.Home, log io.Writer, policy worker.Policy, extra []string) (*ResidentSet, error) {
	cfg, info, err := server.ResidentConfig(h, "", nil)
	if err != nil {
		return nil, err
	}
	def := info.ModelID
	ids := []string{def}
	for _, id := range extra {
		if id == "" || id == def {
			continue
		}
		dup := false
		for _, have := range ids {
			dup = dup || have == id
		}
		if !dup {
			ids = append(ids, id)
		}
	}
	var mu sync.Mutex // one worker's stderr line must not interleave with another's
	members := make([]ResidentMember, 0, len(ids))
	for i, id := range ids {
		model, err := setup.LookupModel(id)
		if err != nil {
			return nil, err
		}
		m := ResidentMember{Model: id, Provider: model.Provider}
		if i == 0 {
			m.Config, m.Info = cfg, info
		} else if m.Config, m.Info, err = server.ResidentConfig(h, id, nil); err != nil {
			cause := fmt.Errorf("model %s (provider %s): %w", id, model.Provider, err)
			m.Info = server.Runtime{Home: h.Root, ModelID: id, Device: info.Device, Runtime: info.Runtime}
			m.Config = worker.Config{Preflight: func() error { return cause }}
		}
		if log != nil {
			m.Config.Log = &prefixWriter{mu: &mu, w: log, prefix: "[" + id + "] "}
		}
		members = append(members, m)
	}
	return NewResidentSet(parent, policy, def, members)
}

// ResidentWorkerRuntime is WorkerRuntime for a resident set: the OpenFunc
// binds a ResidentSet of the active model plus extra. onOpen, if non-nil,
// receives each new set.
func ResidentWorkerRuntime(parent context.Context, log io.Writer, policy worker.Policy, extra []string, onOpen func(*ResidentSet)) OpenFunc {
	return func(root string) (Runtime, error) {
		s, err := OpenResidents(parent, home.Home{Root: root}, log, policy, extra)
		if err != nil {
			return nil, err
		}
		if onOpen != nil {
			onOpen(s)
		}
		return s, nil
	}
}

// additionalResidents is the extra residents a start would add beside the
// default model def: the requested catalog IDs without blanks, duplicates and
// def itself, in request order. Empty means a single resident.
func additionalResidents(def string, requested []string) []string {
	var out []string
	for _, id := range requested {
		if id == "" || id == def || slices.Contains(out, id) {
			continue
		}
		out = append(out, id)
	}
	return out
}

// boundExtras are the non-default members of a bound resident set; a single
// worker has none.
func boundExtras(rt Runtime) []string {
	if m, ok := rt.(interface{ Models() []string }); ok {
		if ms := m.Models(); len(ms) > 1 {
			return ms[1:]
		}
	}
	return nil
}

// residencyDrift reports whether the desired additional residents differ
// from the members rt was opened with, given its default model def. It reads
// the one desired selection (Config.Residents) and the one bound set; there
// is no copy of either in the controller.
func (c *Controller) residencyDrift(def string, rt Runtime) bool {
	if c.cfg.Residents == nil || rt == nil {
		return false
	}
	want, have := additionalResidents(def, c.cfg.Residents()), boundExtras(rt)
	if len(want) != len(have) {
		return true
	}
	for _, id := range want {
		if !slices.Contains(have, id) {
			return true
		}
	}
	return false
}

// ConfiguredRuntime is the OpenFunc of the desktop composition. When the
// requested selection adds no resident beside the active model it is exactly
// WorkerRuntime (one worker, onWorker); otherwise it is ResidentWorkerRuntime
// for the active model plus the selection (onSet). The selection is read at
// every open, so a restart that rebinds picks up the current one; an
// unreadable selection fails the open with the cause instead of silently
// starting fewer residents.
func ConfiguredRuntime(parent context.Context, log io.Writer, policy worker.Policy, requested func() ([]string, error),
	onWorker func(*WorkerBinding), onSet func(*ResidentSet)) OpenFunc {
	single := WorkerRuntime(parent, log, policy, onWorker)
	return func(root string) (Runtime, error) {
		want, err := requested()
		if err != nil {
			return nil, fmt.Errorf("resident models: %w", err)
		}
		if len(want) == 0 {
			return single(root)
		}
		_, info, err := server.ResidentConfig(home.Home{Root: root}, "", nil)
		if err != nil {
			return nil, err
		}
		extra := additionalResidents(info.ModelID, want)
		if len(extra) == 0 {
			return single(root)
		}
		return ResidentWorkerRuntime(parent, log, policy, extra, onSet)(root)
	}
}

// prefixWriter tags every write (the worker log writes one line at a time)
// with the resident it came from.
type prefixWriter struct {
	mu     *sync.Mutex
	w      io.Writer
	prefix string
}

func (p *prefixWriter) Write(b []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, err := p.w.Write(append([]byte(p.prefix), b...)); err != nil {
		return 0, err
	}
	return len(b), nil
}

// aggregateResidents is the one-worker view of a resident set that the
// application state is projected from. The default resident's status is the
// base; its worker state, phase, restart count and failure are replaced by
// those of the least healthy member that is not operator-stopped (failed,
// then restarting, then starting, then ready), so the application is Ready
// only when every resident that is meant to be up is READY, and a failed
// resident is never reported as ready. culprit is that member when it is not
// ready, so a failure can name the model and provider. When every member is
// stopped the base is returned unchanged.
func aggregateResidents(base server.Status, rs []ResidentStatus) (server.Status, *ResidentStatus) {
	rank := func(state string) int {
		switch state {
		case worker.StateFailed:
			return 4
		case worker.StateRestarting:
			return 3
		case worker.StateStarting:
			return 2
		case worker.StateReady:
			return 1
		}
		return 0 // stopped by the operator: not part of the aggregate
	}
	var worst *ResidentStatus
	for i := range rs {
		if rank(rs[i].Status.Worker.State) > 0 && (worst == nil || rank(rs[i].Status.Worker.State) > rank(worst.Status.Worker.State)) {
			worst = &rs[i]
		}
	}
	if worst == nil {
		return base, nil
	}
	w := worst.Status.Worker
	out := base
	out.Worker.State, out.Worker.Phase, out.Worker.Ready = w.State, w.Phase, w.Ready
	out.Worker.Restarts, out.Worker.LastFailure = w.Restarts, w.LastFailure
	if w.State == worker.StateReady {
		return out, nil
	}
	return out, worst
}
