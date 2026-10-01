package route

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const (
	laya = "laya-base"
	nano = "opendecider-nano"
)

// call is one worker call the fake backend served.
type call struct {
	model string
	items []worker.Item
}

// fakeBackend is a resident set of scripted residents. It records every call,
// so a test can prove which residents were asked and with what.
type fakeBackend struct {
	calls []call
	// answer scripts a resident's reply per question: choice, confidence.
	answer map[string]func(q api.Question) (string, float64)
	// fail makes a resident return that error.
	fail map[string]error
	// ms is the reported inference time per call.
	ms float64
	// mangle post-processes a reply, to simulate a misbehaving resident.
	mangle func(model string, res [][]api.Result) [][]api.Result
}

func (b *fakeBackend) Identity(model string) (api.Served, bool) {
	switch model {
	case laya:
		return api.Served{Model: laya, Provider: "laya"}, true
	case nano:
		return api.Served{Model: nano, Provider: "opendecider"}, true
	}
	return api.Served{}, false
}

func (b *fakeBackend) DecideOn(model string, items []worker.Item) ([][]api.Result, float64, error) {
	b.calls = append(b.calls, call{model, items})
	if err := b.fail[model]; err != nil {
		return nil, 0, err
	}
	var out [][]api.Result
	for _, it := range items {
		var rs []api.Result
		for _, q := range it.Questions {
			choice, conf := q.Choices[0], 0.9
			if f := b.answer[model]; f != nil {
				choice, conf = f(q)
			}
			probs := map[string]float64{}
			for _, c := range q.Choices {
				probs[c] = (1 - conf) / float64(len(q.Choices)-1)
			}
			probs[choice] = conf
			rs = append(rs, api.Result{ID: q.ID, Type: "choice", Choice: choice, Confidence: conf, Probabilities: probs})
		}
		out = append(out, rs)
	}
	if b.mangle != nil {
		out = b.mangle(model, out)
	}
	return out, b.ms, nil
}

func (b *fakeBackend) calledModels() []string {
	var out []string
	for _, c := range b.calls {
		out = append(out, c.model)
	}
	return out
}

func yn(id string) api.Question {
	return api.Question{ID: id, Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}
}

func ptr(v float64) *float64 { return &v }

// policy: family "ready" starts on laya and escalates to nano on low
// confidence or on the choice "no"; family "scope" always uses nano; family
// "plain" is laya only; everything else follows the default (laya, no handoff).
func testPolicy() Policy {
	return Policy{
		Schema:   PolicySchema,
		ID:       "p1",
		Families: map[string]string{"r1": "ready", "r2": "ready", "s1": "scope", "p1": "plain"},
		Rules: []Rule{
			{Family: "ready", First: laya, Handoff: &Handoff{To: nano, When: When{ConfidenceBelow: ptr(0.7), ChoiceIn: []string{"no"}}}},
			{Family: "scope", First: nano},
			{Family: "plain", First: laya},
		},
		Default: Rule{First: laya},
	}
}

func newRouter(t *testing.T, p Policy, b *fakeBackend) *Router {
	t.Helper()
	r, err := New(p, b)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func item(state string, ids ...string) worker.Item {
	it := worker.Item{State: state}
	for _, id := range ids {
		it.Questions = append(it.Questions, yn(id))
	}
	return it
}

func TestKeepsFirstPathWithoutInvokingTheAlternate(t *testing.T) {
	b := &fakeBackend{ms: 4}
	r := newRouter(t, testPolicy(), b)
	out, err := r.Decide([]worker.Item{item("s", "r1", "r2", "p1", "other")})
	if err != nil {
		t.Fatal(err)
	}
	if got := b.calledModels(); !reflect.DeepEqual(got, []string{laya}) {
		t.Fatalf("calls %v: the alternate must not run when no handoff is required", got)
	}
	reasons := map[string]string{}
	for _, rr := range out.Items[0].Routed {
		reasons[rr.ID] = rr.Reason
		if rr.Served.Model != laya || rr.FirstPath != nil {
			t.Fatalf("%+v", rr)
		}
	}
	want := map[string]string{"r1": api.ReasonFirstPathKept, "r2": api.ReasonFirstPathKept, "p1": api.ReasonFirstPathOnly, "other": api.ReasonFirstPathOnly}
	if !reflect.DeepEqual(reasons, want) {
		t.Fatalf("reasons %v", reasons)
	}
	if out.Handoffs != 0 || len(out.Providers) != 1 || out.Providers[0].Model != laya || out.Providers[0].Questions != 4 {
		t.Fatalf("%+v", out)
	}
}

func TestSelectiveHandoffReplacesOnlyTheSelectedResults(t *testing.T) {
	b := &fakeBackend{ms: 4, answer: map[string]func(api.Question) (string, float64){
		laya: func(q api.Question) (string, float64) {
			if q.ID == "r2" {
				return "yes", 0.55 // low confidence: handed off
			}
			return "yes", 0.95
		},
		nano: func(q api.Question) (string, float64) { return "no", 0.99 },
	}}
	r := newRouter(t, testPolicy(), b)
	items := []worker.Item{item("a", "r1", "r2", "p1"), item("b", "r1", "s1")}
	out, err := r.Decide(items)
	if err != nil {
		t.Fatal(err)
	}

	// Calls: laya first path for everything not on nano, nano first path for
	// s1, then nano for exactly the handed-off r2 of item a. Policy order.
	if got := b.calledModels(); !reflect.DeepEqual(got, []string{laya, nano, nano}) {
		t.Fatalf("calls %v", got)
	}
	handoffCall := b.calls[2]
	if len(handoffCall.items) != 1 || handoffCall.items[0].State != "a" ||
		len(handoffCall.items[0].Questions) != 1 || handoffCall.items[0].Questions[0].ID != "r2" {
		t.Fatalf("the handoff must carry only the selected question: %+v", handoffCall.items)
	}

	a, bb := out.Items[0], out.Items[1]
	// Unaffected results are exactly what the first-path resident returned.
	exact := func(model, id string, conf float64, choice string, got api.Result) {
		t.Helper()
		probs := map[string]float64{"yes": 0, "no": 0}
		for k := range probs {
			probs[k] = (1 - conf)
		}
		probs[choice] = conf
		want := api.Result{ID: id, Type: "choice", Choice: choice, Confidence: conf, Probabilities: probs}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s %s: %+v, want %+v", model, id, got, want)
		}
	}
	exact(laya, "r1", 0.95, "yes", a.Results[0])
	exact(laya, "p1", 0.95, "yes", a.Results[2])
	exact(laya, "r1", 0.95, "yes", bb.Results[0])
	exact(nano, "s1", 0.99, "no", bb.Results[1])
	// The handed-off result is the nano answer, in the first-path position.
	exact(nano, "r2", 0.99, "no", a.Results[1])

	rr := a.Routed[1]
	if rr.ID != "r2" || rr.Served.Model != nano || rr.Served.Provider != "opendecider" || rr.Reason != api.ReasonHandoffLowConf || rr.Profile != "ready" ||
		rr.FirstPath == nil || rr.FirstPath.Model != laya || rr.FirstPath.Choice != "yes" || rr.FirstPath.Confidence != 0.55 {
		t.Fatalf("handed-off provenance %+v", rr)
	}
	for _, i := range []int{0, 2} {
		if a.Routed[i].Served.Model != laya || a.Routed[i].FirstPath != nil {
			t.Fatalf("unaffected %+v", a.Routed[i])
		}
	}
	if bb.Routed[1].Served.Model != nano || bb.Routed[1].Reason != api.ReasonFirstPathOnly || bb.Routed[1].Profile != "scope" {
		t.Fatalf("always-nano family: %+v", bb.Routed[1])
	}
	if out.Handoffs != 1 {
		t.Fatalf("handoffs %d", out.Handoffs)
	}
	// Per-provider contribution: laya 1 call (4 questions), nano 2 calls (1 first-path + 1 handoff).
	want := []api.ProviderTiming{{Model: laya, Provider: "laya", Calls: 1, Questions: 4, InferenceMS: 4}, {Model: nano, Provider: "opendecider", Calls: 2, Questions: 2, InferenceMS: 8}}
	if !reflect.DeepEqual(out.Providers, want) || out.InferenceMS != 12 {
		t.Fatalf("providers %+v total %v", out.Providers, out.InferenceMS)
	}
}

func TestEveryHandoffDimensionHasItsOwnReason(t *testing.T) {
	mk := func(w When) Policy {
		return Policy{Schema: PolicySchema, ID: "d", Default: Rule{First: laya, Handoff: &Handoff{To: nano, When: w}}}
	}
	cases := []struct {
		name   string
		when   When
		choice string
		conf   float64
		reason string
	}{
		{"always", When{Always: true}, "yes", 0.99, api.ReasonHandoffAlways},
		{"choice", When{ChoiceIn: []string{"no"}}, "no", 0.99, api.ReasonHandoffChoice},
		{"choice not in set", When{ChoiceIn: []string{"no"}}, "yes", 0.2, api.ReasonFirstPathKept},
		{"confidence", When{ConfidenceBelow: ptr(0.8)}, "yes", 0.6, api.ReasonHandoffLowConf},
		{"confidence at threshold keeps", When{ConfidenceBelow: ptr(0.8)}, "yes", 0.8, api.ReasonFirstPathKept},
		{"margin", When{MarginBelow: ptr(0.3)}, "yes", 0.6, api.ReasonHandoffLowMargin}, // 0.6 vs 0.4: margin 0.2
		{"margin enough", When{MarginBelow: ptr(0.3)}, "yes", 0.9, api.ReasonFirstPathKept},
		// A margin policy without a confidence threshold: confidence alone is
		// not the universal criterion, and a confident-looking answer with a thin
		// margin is still escalated.
		{"margin without confidence", When{MarginBelow: ptr(0.5)}, "yes", 0.7, api.ReasonHandoffLowMargin},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := &fakeBackend{answer: map[string]func(api.Question) (string, float64){
				laya: func(api.Question) (string, float64) { return tc.choice, tc.conf }}}
			out, err := newRouter(t, mk(tc.when), b).Decide([]worker.Item{item("s", "q")})
			if err != nil {
				t.Fatal(err)
			}
			if got := out.Items[0].Routed[0].Reason; got != tc.reason {
				t.Fatalf("reason %s, want %s", got, tc.reason)
			}
			handed := tc.reason != api.ReasonFirstPathKept
			if (len(b.calls) == 2) != handed {
				t.Fatalf("calls %v for handoff=%v", b.calledModels(), handed)
			}
		})
	}
}

func TestAlwaysChecksAreEvaluatedInAFixedOrder(t *testing.T) {
	// Every check holds; the first in the documented order is the reason.
	p := Policy{Schema: PolicySchema, ID: "o", Default: Rule{First: laya, Handoff: &Handoff{To: nano,
		When: When{Always: true, ChoiceIn: []string{"yes"}, ConfidenceBelow: ptr(1), MarginBelow: ptr(1)}}}}
	out, err := newRouter(t, p, &fakeBackend{}).Decide([]worker.Item{item("s", "q")})
	if err != nil || out.Items[0].Routed[0].Reason != api.ReasonHandoffAlways {
		t.Fatalf("%v %+v", err, out.Items)
	}
	p.Default.Handoff.When.Always = false
	out, _ = newRouter(t, p, &fakeBackend{}).Decide([]worker.Item{item("s", "q")})
	if out.Items[0].Routed[0].Reason != api.ReasonHandoffChoice {
		t.Fatalf("%+v", out.Items[0].Routed[0])
	}
}

func TestRoutingIsDeterministic(t *testing.T) {
	run := func() (Outcome, []call) {
		b := &fakeBackend{ms: 2, answer: map[string]func(api.Question) (string, float64){
			laya: func(q api.Question) (string, float64) {
				if q.ID == "r2" {
					return "no", 0.9
				}
				return "yes", 0.5
			}}}
		out, err := newRouter(t, testPolicy(), b).Decide([]worker.Item{item("a", "r1", "r2", "s1", "p1"), item("b", "r2", "r1")})
		if err != nil {
			t.Fatal(err)
		}
		return out, b.calls
	}
	o1, c1 := run()
	for i := 0; i < 20; i++ {
		o2, c2 := run()
		if !reflect.DeepEqual(o1, o2) || !reflect.DeepEqual(c1, c2) {
			t.Fatalf("run %d differs:\n%+v\n%+v", i, o1, o2)
		}
	}
}

func TestRequiredHandoffFailureIsExplicitAndNeverAcceptsTheWeakerResult(t *testing.T) {
	for name, err := range map[string]error{
		"not_ready": &worker.RequestError{Class: api.ErrNotReady, Message: "model opendecider-nano: worker starting"},
		"inference": &worker.RequestError{Class: api.ErrInferenceFailed, Message: "boom"},
		"worker":    &worker.Failure{Class: worker.ClassCrash, Message: "exited"},
		"plain":     errors.New("pipe closed"),
	} {
		t.Run(name, func(t *testing.T) {
			b := &fakeBackend{fail: map[string]error{nano: err}, answer: map[string]func(api.Question) (string, float64){
				laya: func(api.Question) (string, float64) { return "yes", 0.2 }}}
			r := newRouter(t, testPolicy(), b)
			out, got := r.Decide([]worker.Item{item("s", "r1", "p1")})
			var re *worker.RequestError
			if !errors.As(got, &re) || re.Class != api.ErrRoutingFailed || !strings.HasPrefix(re.Message, CodeRequiredHandoffFailed+": model "+nano) {
				t.Fatalf("want a routing_failed %s error, got %v", CodeRequiredHandoffFailed, got)
			}
			if len(out.Items) != 0 {
				t.Fatalf("a failed routed request returns no results: %+v", out)
			}
			if st := r.Status(); st.Failures != 1 || st.Requests != 0 {
				t.Fatalf("failure not counted: %+v", st)
			}
		})
	}
}

func TestOptionalHandoffFailureKeepsFirstPathAndSaysSo(t *testing.T) {
	p := testPolicy()
	p.Rules[0].Handoff.Optional = true
	b := &fakeBackend{fail: map[string]error{nano: &worker.RequestError{Class: api.ErrNotReady, Message: "starting"}},
		answer: map[string]func(api.Question) (string, float64){laya: func(api.Question) (string, float64) { return "yes", 0.2 }}}
	r := newRouter(t, p, b)
	out, err := r.Decide([]worker.Item{item("s", "r1", "p1")})
	if err != nil {
		t.Fatal(err)
	}
	if rr := out.Items[0].Routed[0]; rr.Reason != api.ReasonHandoffFailed || rr.Served.Model != laya || rr.FirstPath != nil {
		t.Fatalf("%+v", rr)
	}
	if out.Handoffs != 0 || r.Status().HandoffFailures != 1 {
		t.Fatalf("handoffs %d status %+v", out.Handoffs, r.Status())
	}
}

func TestRequiredFailureWinsOverAnOptionalRuleOnTheSameTarget(t *testing.T) {
	p := testPolicy()
	p.Families["o1"] = "opt"
	p.Rules = append(p.Rules, Rule{Family: "opt", First: laya, Handoff: &Handoff{To: nano, Optional: true, When: When{Always: true}}})
	b := &fakeBackend{fail: map[string]error{nano: errors.New("down")},
		answer: map[string]func(api.Question) (string, float64){laya: func(api.Question) (string, float64) { return "yes", 0.2 }}}
	if _, err := newRouter(t, p, b).Decide([]worker.Item{item("s", "r1", "o1")}); err == nil {
		t.Fatal("a required handoff in the same call must fail the request")
	}
}

func TestFirstPathFailureIsExplicitAndTheAlternateIsNotAsked(t *testing.T) {
	b := &fakeBackend{fail: map[string]error{laya: &worker.RequestError{Class: api.ErrNotReady, Message: "model laya-base: failed"}}}
	_, err := newRouter(t, testPolicy(), b).Decide([]worker.Item{item("s", "r1")})
	var re *worker.RequestError
	if !errors.As(err, &re) || re.Class != api.ErrRoutingFailed || !strings.HasPrefix(re.Message, CodeFirstPathFailed+": model "+laya+": not_ready") {
		t.Fatalf("%v", err)
	}
	if got := b.calledModels(); !reflect.DeepEqual(got, []string{laya}) {
		t.Fatalf("calls %v: a failed first path is never answered by the alternate", got)
	}
}

func TestMalformedResidentReplyIsAFailureNotAPartialAnswer(t *testing.T) {
	for name, mangle := range map[string]func(string, [][]api.Result) [][]api.Result{
		"short":    func(_ string, r [][]api.Result) [][]api.Result { return r[:0] },
		"fewer":    func(_ string, r [][]api.Result) [][]api.Result { r[0] = r[0][:1]; return r },
		"wrong id": func(_ string, r [][]api.Result) [][]api.Result { r[0][0].ID = "zzz"; return r },
	} {
		t.Run(name, func(t *testing.T) {
			b := &fakeBackend{mangle: mangle}
			_, err := newRouter(t, testPolicy(), b).Decide([]worker.Item{item("s", "r1", "p1")})
			var re *worker.RequestError
			if !errors.As(err, &re) || re.Class != api.ErrRoutingFailed || !strings.Contains(re.Message, CodeMalformedResidentReply) {
				t.Fatalf("%v", err)
			}
		})
	}
}

func TestStatusCountsReasonsHandoffsAndPerProviderContribution(t *testing.T) {
	b := &fakeBackend{ms: 3, answer: map[string]func(api.Question) (string, float64){
		laya: func(q api.Question) (string, float64) {
			if q.ID == "r2" {
				return "no", 0.9
			}
			return "yes", 0.9
		}}}
	r := newRouter(t, testPolicy(), b)
	for i := 0; i < 2; i++ {
		if _, err := r.Decide([]worker.Item{item("s", "r1", "r2", "s1")}); err != nil {
			t.Fatal(err)
		}
	}
	st := r.Status()
	if st.Requests != 2 || st.Questions != 6 || st.Handoffs != 2 || st.Failures != 0 || st.Policy != r.Ref() {
		t.Fatalf("%+v", st)
	}
	if want := map[string]int64{api.ReasonFirstPathKept: 2, api.ReasonHandoffChoice: 2, api.ReasonFirstPathOnly: 2}; !reflect.DeepEqual(st.Reasons, want) {
		t.Fatalf("reasons %v", st.Reasons)
	}
	l, n := st.Providers[0], st.Providers[1]
	if l.Model != laya || n.Model != nano {
		t.Fatalf("%+v", st.Providers)
	}
	// laya: first path for r1 and r2 (twice); nano: first path for s1, handoff target for r2.
	if l.FirstPathQuestions != 4 || l.HandoffQuestions != 0 || l.FinalResults != 2 || l.Calls != 2 || l.InferenceMSTotal != 6 {
		t.Fatalf("laya %+v", l)
	}
	if n.FirstPathQuestions != 2 || n.HandoffQuestions != 2 || n.FinalResults != 4 || n.Calls != 4 || n.InferenceMSTotal != 12 {
		t.Fatalf("nano %+v", n)
	}
	// The status is a copy: mutating it does not change the router's.
	st.Reasons["x"] = 1
	st.Providers[0].Calls = 99
	if again := r.Status(); again.Reasons["x"] != 0 || again.Providers[0].Calls == 99 {
		t.Fatal("status aliases router state")
	}
}

func TestNewRejectsAPolicyThatNamesAModelThatIsNotResident(t *testing.T) {
	p := testPolicy()
	p.Rules[2].First = "openjev"
	if _, err := New(p, &fakeBackend{}); err == nil || !strings.Contains(err.Error(), "openjev") {
		t.Fatalf("%v", err)
	}
}

func TestParseValidatesTheClosedPolicyVocabulary(t *testing.T) {
	good := `{"schema":"hachidori.routing-policy.v1","id":"p","families":{"a":"f"},
		"rules":[{"family":"f","first":"laya-base","handoff":{"to":"opendecider-nano","when":{"confidence_below":0.7}}}],
		"default":{"first":"laya-base"}}`
	p, err := Parse([]byte(good))
	if err != nil {
		t.Fatal(err)
	}
	// Same policy, different formatting and field order: same digest.
	reordered := `{"default":{"first":"laya-base"},"id":"p","rules":[{"handoff":{"when":{"confidence_below":0.70},"to":"opendecider-nano"},"first":"laya-base","family":"f"}],"families":{"a":"f"},"schema":"hachidori.routing-policy.v1"}`
	q, err := Parse([]byte(reordered))
	if err != nil || p.Digest() != q.Digest() || !strings.HasPrefix(p.Digest(), "sha256:") {
		t.Fatalf("%v %s %s", err, p.Digest(), q.Digest())
	}
	if want := []string{laya, nano}; !reflect.DeepEqual(p.Models(), want) {
		t.Fatalf("models %v", p.Models())
	}
	// A change in a threshold changes the digest.
	if r, _ := Parse([]byte(strings.Replace(good, "0.7", "0.8", 1))); r.Digest() == p.Digest() {
		t.Fatal("digest ignores the threshold")
	}

	for name, bad := range map[string]string{
		"unknown field":         strings.Replace(good, `"id":"p"`, `"id":"p","extra":1`, 1),
		"wrong schema":          strings.Replace(good, "routing-policy.v1", "routing-policy.v9", 1),
		"bad id":                strings.Replace(good, `"id":"p"`, `"id":"a b"`, 1),
		"family without rule":   strings.Replace(good, `"a":"f"`, `"a":"g"`, 1),
		"handoff to self":       strings.Replace(good, `"to":"opendecider-nano"`, `"to":"laya-base"`, 1),
		"empty condition":       strings.Replace(good, `"when":{"confidence_below":0.7}`, `"when":{}`, 1),
		"threshold above one":   strings.Replace(good, "0.7", "1.5", 1),
		"threshold zero":        strings.Replace(good, "0.7", "0", 1),
		"default names family":  strings.Replace(good, `"default":{"first"`, `"default":{"family":"f","first"`, 1),
		"duplicate rule family": strings.Replace(good, `"rules":[`, `"rules":[{"family":"f","first":"laya-base"},`, 1),
		"missing first":         strings.Replace(good, `"default":{"first":"laya-base"}`, `"default":{}`, 1),
		"trailing data":         good + `{}`,
		"model is a repository": strings.Replace(good, `"first":"laya-base","handoff"`, `"first":"org/repo","handoff"`, 1),
	} {
		if _, err := Parse([]byte(bad)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestMarginIsTheTopTwoGapAndUndefinedEscalates(t *testing.T) {
	if got := margin(map[string]float64{"a": 0.5, "b": 0.3, "c": 0.2}); got < 0.199 || got > 0.201 {
		t.Fatalf("margin %v", got)
	}
	if got := margin(map[string]float64{"a": 0.5, "b": 0.5}); got != 0 {
		t.Fatalf("tie %v", got)
	}
	if got := margin(nil); got != 0 {
		t.Fatalf("undefined %v", got)
	}
	w := When{MarginBelow: ptr(0.1)}
	if w.fires("a", 1, nil) != api.ReasonHandoffLowMargin {
		t.Fatal("an undefined margin must escalate, not trust")
	}
}

func TestRoutedResultJSONHasNoProviderInternals(t *testing.T) {
	b := &fakeBackend{answer: map[string]func(api.Question) (string, float64){laya: func(api.Question) (string, float64) { return "no", 0.9 }}}
	r := newRouter(t, testPolicy(), b)
	out, err := r.Decide([]worker.Item{item("s", "r1")})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(r.Routing(out.Items[0], out.Providers))
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	for k := range m {
		switch k {
		case "mode", "policy", "results", "handoffs", "providers":
		default:
			t.Fatalf("unexpected routing field %q", k)
		}
	}
	if strings.Contains(string(raw), "prompt") || strings.Contains(string(raw), "token") {
		t.Fatalf("provider internals in %s", raw)
	}
}
