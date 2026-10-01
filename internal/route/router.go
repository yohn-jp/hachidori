package route

import (
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Reason codes are the stable api constants; the short names keep the policy
// code readable.
const (
	reasonAlways        = api.ReasonHandoffAlways
	reasonChoice        = api.ReasonHandoffChoice
	reasonLowConfidence = api.ReasonHandoffLowConf
	reasonLowMargin     = api.ReasonHandoffLowMargin
)

// Codes of a routing failure, the prefix of its message. They are stable.
const (
	CodeFirstPathFailed        = "first_path_failed"
	CodeRequiredHandoffFailed  = "required_handoff_failed"
	CodeNoPolicy               = "no_routing_policy"
	CodeResidentMissing        = "resident_missing"
	CodeMalformedResidentReply = "malformed_resident_reply"
)

// Backend is the resident set the router sits above. DecideOn serves items on
// exactly the named resident (a direct call that never starts, reloads or
// redirects anything) and Identity reports a resident's catalog identity.
// app.ResidentSet implements it.
type Backend interface {
	DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error)
	Identity(model string) (api.Served, bool)
}

// Router applies one Policy over a Backend. It is safe for concurrent use;
// each resident still serializes its own requests.
type Router struct {
	policy Policy
	ref    api.PolicyRef
	be     Backend
	served map[string]api.Served

	mu    sync.Mutex
	stats stats
}

// New builds the router. Every model the policy names must be a resident of
// the backend: a policy that routes to a model that is not resident is a
// configuration error, found here and not on the first request.
func New(p Policy, be Backend) (*Router, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	r := &Router{policy: p, ref: api.PolicyRef{ID: p.ID, SHA256: p.Digest()}, be: be, served: map[string]api.Served{}}
	for _, m := range p.Models() {
		id, ok := be.Identity(m)
		if !ok {
			return nil, fmt.Errorf("routing policy %s names model %s, which is not a resident (resident models are chosen with --resident)", p.ID, m)
		}
		r.served[m] = id
	}
	r.stats = newStats(p.Models(), r.served)
	return r, nil
}

// Policy is the policy the router applies.
func (r *Router) Policy() Policy { return r.policy }

// Ref identifies the policy in responses and status.
func (r *Router) Ref() api.PolicyRef { return r.ref }

// Outcome is the result of one routed Decide: one ItemOutcome per input item
// in order, the per-resident latency contribution, and the handoff count.
// InferenceMS is the sum of the worker calls it made.
type Outcome struct {
	Items       []ItemOutcome
	Providers   []api.ProviderTiming
	Handoffs    int
	InferenceMS float64
}

// ItemOutcome is the final results of one item and their routing provenance,
// both in question order.
type ItemOutcome struct {
	Results []api.Result
	Routed  []api.RoutedResult
}

// at addresses one question of one item.
type at struct{ item, q int }

// Decide routes every question of every item by the policy. It is
// deterministic: for the same items, policy and resident answers it makes
// the same calls in the same order and returns the same results.
//
//  1. Each question goes to its rule's first-path resident, one worker call
//     per resident, residents in policy order. No other resident is asked.
//  2. The handoff condition of the rule is evaluated on each first-path
//     result. Only the questions whose condition holds are sent, in one call
//     per target resident.
//  3. Each handed-off result replaces exactly its first-path result; every
//     other result is kept exactly as its first-path resident returned it.
//
// A first-path failure, and a failure of a required handoff, return a
// routing_failed error and no results. A failure of an optional handoff keeps
// the first-path results and says so in the reason code.
func (r *Router) Decide(items []worker.Item) (Outcome, error) {
	out, err := r.decide(items)
	r.record(out, err)
	return out, err
}

func (r *Router) decide(items []worker.Item) (Outcome, error) {
	n := len(items)
	res := make([][]api.Result, n)
	routed := make([][]api.RoutedResult, n)
	rules := make([][]Rule, n)
	for i, it := range items {
		res[i] = make([]api.Result, len(it.Questions))
		routed[i] = make([]api.RoutedResult, len(it.Questions))
		rules[i] = make([]Rule, len(it.Questions))
	}
	calls := &callLog{}

	// First path.
	first := map[string][]at{}
	for i, it := range items {
		for j, q := range it.Questions {
			rule, family := r.policy.rule(q.ID)
			rules[i][j] = rule
			routed[i][j].Profile = family
			first[rule.First] = append(first[rule.First], at{i, j})
		}
	}
	for _, m := range r.policy.Models() {
		refs := first[m]
		if len(refs) == 0 {
			continue
		}
		got, err := r.call(calls, m, items, refs)
		if err != nil {
			return Outcome{}, fail(CodeFirstPathFailed, m, err)
		}
		for k, a := range refs {
			res[a.item][a.q] = got[k]
			routed[a.item][a.q].ID = got[k].ID
			routed[a.item][a.q].Served = r.served[m]
		}
	}

	// Handoff decision, in item and question order.
	type want struct {
		at
		reason string
	}
	handoff := map[string][]want{}
	for i, it := range items {
		for j := range it.Questions {
			h := rules[i][j].Handoff
			switch {
			case h == nil:
				routed[i][j].Reason = api.ReasonFirstPathOnly
			default:
				if reason := h.When.fires(res[i][j].Choice, res[i][j].Confidence, res[i][j].Probabilities); reason != "" {
					handoff[h.To] = append(handoff[h.To], want{at{i, j}, reason})
				} else {
					routed[i][j].Reason = api.ReasonFirstPathKept
				}
			}
		}
	}

	// Handoff: replace only the selected results.
	handoffs := 0
	for _, m := range r.policy.Models() {
		ws := handoff[m]
		if len(ws) == 0 {
			continue
		}
		refs := make([]at, len(ws))
		required := false
		for k, w := range ws {
			refs[k] = w.at
			required = required || !rules[w.item][w.q].Handoff.Optional
		}
		got, err := r.call(calls, m, items, refs)
		if err != nil {
			if required {
				return Outcome{}, fail(CodeRequiredHandoffFailed, m, err)
			}
			for _, w := range ws {
				routed[w.item][w.q].Reason = api.ReasonHandoffFailed
			}
			r.noteHandoffFailure(len(ws))
			continue
		}
		for k, w := range ws {
			prev := routed[w.item][w.q]
			old := res[w.item][w.q]
			res[w.item][w.q] = got[k]
			routed[w.item][w.q] = api.RoutedResult{ID: got[k].ID, Served: r.served[m], Reason: w.reason, Profile: prev.Profile,
				FirstPath: &api.FirstPath{Model: prev.Served.Model, Provider: prev.Served.Provider, Choice: old.Choice, Confidence: old.Confidence}}
			handoffs++
		}
	}

	out := Outcome{Items: make([]ItemOutcome, n), Providers: calls.timings(r.policy.Models(), r.served), Handoffs: handoffs, InferenceMS: calls.total}
	for i := range items {
		out.Items[i] = ItemOutcome{Results: res[i], Routed: routed[i]}
	}
	return out, nil
}

// call runs the questions at refs on resident m as one worker call. Items
// keep their order and each carries only its selected questions, so a state
// is sent only where a question of it was selected. The returned results are
// flattened in refs order. A reply that does not answer exactly the selected
// questions is a failure of that resident, never a partial answer.
func (r *Router) call(log *callLog, m string, items []worker.Item, refs []at) ([]api.Result, error) {
	var sub []worker.Item
	last := -1
	for _, a := range refs {
		if a.item != last {
			sub = append(sub, worker.Item{State: items[a.item].State})
			last = a.item
		}
		sub[len(sub)-1].Questions = append(sub[len(sub)-1].Questions, items[a.item].Questions[a.q])
	}
	got, ms, err := r.be.DecideOn(m, sub)
	if err != nil {
		return nil, err
	}
	if len(got) != len(sub) {
		return nil, &worker.RequestError{Class: api.ErrInferenceFailed, Message: fmt.Sprintf("%s: resident answered %d items for %d", CodeMalformedResidentReply, len(got), len(sub))}
	}
	flat := make([]api.Result, 0, len(refs))
	for k, it := range sub {
		if len(got[k]) != len(it.Questions) {
			return nil, &worker.RequestError{Class: api.ErrInferenceFailed, Message: fmt.Sprintf("%s: resident answered %d questions for %d", CodeMalformedResidentReply, len(got[k]), len(it.Questions))}
		}
		for q := range it.Questions {
			if got[k][q].ID != it.Questions[q].ID {
				return nil, &worker.RequestError{Class: api.ErrInferenceFailed, Message: fmt.Sprintf("%s: resident answered question %q for %q", CodeMalformedResidentReply, got[k][q].ID, it.Questions[q].ID)}
			}
			flat = append(flat, got[k][q])
		}
	}
	log.add(m, len(refs), ms)
	return flat, nil
}

// fail is the routing_failed error of a stage: code, the resident and the
// resident's own error class and message.
func fail(code, model string, err error) error {
	cause := err.Error()
	var re *worker.RequestError
	var f *worker.Failure
	switch {
	case errors.As(err, &re):
		cause = joinNonEmpty(re.Class, re.Message)
	case errors.As(err, &f):
		cause = joinNonEmpty(f.Class, f.Message)
	}
	return &worker.RequestError{Class: api.ErrRoutingFailed, Message: fmt.Sprintf("%s: model %s: %s", code, model, cause)}
}

// callLog is the worker calls of one routed Decide.
type callLog struct {
	by    map[string]*api.ProviderTiming
	total float64
}

func (l *callLog) add(model string, questions int, ms float64) {
	if l.by == nil {
		l.by = map[string]*api.ProviderTiming{}
	}
	t := l.by[model]
	if t == nil {
		t = &api.ProviderTiming{Model: model}
		l.by[model] = t
	}
	t.Calls++
	t.Questions += questions
	t.InferenceMS += ms
	l.total += ms
}

// timings lists the residents that served a call, in policy order.
func (l *callLog) timings(order []string, served map[string]api.Served) []api.ProviderTiming {
	var out []api.ProviderTiming
	for _, m := range order {
		if t := l.by[m]; t != nil {
			t.Provider = served[m].Provider
			out = append(out, *t)
		}
	}
	return out
}

// Routing is the response provenance of one routed item; providers is the
// latency contribution to report with it (nil inside a batch entry).
func (r *Router) Routing(item ItemOutcome, providers []api.ProviderTiming) *api.Routing {
	h := 0
	for _, rr := range item.Routed {
		if rr.FirstPath != nil {
			h++
		}
	}
	return &api.Routing{Mode: api.RouteAuto, Policy: r.ref, Results: slices.Clone(item.Routed), Handoffs: h, Providers: providers}
}
