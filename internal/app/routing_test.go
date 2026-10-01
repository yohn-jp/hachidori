package app

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/route"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func routedItem(state string) []worker.Item {
	return []worker.Item{{State: state, Questions: []api.Question{
		{ID: "q", Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}}}}
}

// answerOf splits the fake worker's answer: model, pid, request sequence.
func answerOf(t *testing.T, r api.Result) (model string, pid int) {
	t.Helper()
	parts := strings.SplitN(r.Choice, "|", 4)
	pid, _ = strconv.Atoi(parts[1])
	return parts[0], pid
}

func keepPolicy() route.Policy {
	return route.Policy{Schema: route.PolicySchema, ID: "keep", Default: route.Rule{First: modelA}}
}

func handoffPolicy() route.Policy {
	return route.Policy{Schema: route.PolicySchema, ID: "handoff", Families: map[string]string{"q": "f"},
		Rules:   []route.Rule{{Family: "f", First: modelA, Handoff: &route.Handoff{To: modelB, When: route.When{Always: true}}}},
		Default: route.Rule{First: modelA}}
}

// Both residents are separate worker processes. Keep, handoff, direct and
// failed-handoff routes run over them without a reload, restart or eviction of
// either, and one resident going down does not change the other.
func TestRoutingOverRealResidentWorkersNeverReloads(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	s.Start()
	bothReady(t, s)
	if s.AutoRouter() != nil || s.RoutingStatus() != nil {
		t.Fatal("a set without a policy reports routing")
	}

	_, pidA := answerOf(t, mustDecide(t, s, modelA, "probe").Results)
	_, pidB := answerOf(t, mustDecide(t, s, modelB, "probe").Results)
	startsA, startsB := residentState(s, modelA).Starts, residentState(s, modelB).Starts

	// Keep: the policy keeps laya's answer and never asks the alternate.
	if _, err := s.SetRouting(keepPolicy()); err != nil {
		t.Fatal(err)
	}
	before := residentState(s, modelB).Requests
	out, err := s.AutoRouter().Decide(routedItem("keep"))
	if err != nil {
		t.Fatal(err)
	}
	if m, pid := answerOf(t, out.Items[0].Results[0]); m != modelA || pid != pidA ||
		out.Items[0].Routed[0].Served.Model != modelA || out.Items[0].Routed[0].Reason != api.ReasonFirstPathOnly {
		t.Fatalf("%+v", out.Items[0])
	}
	if got := residentState(s, modelB).Requests; got != before {
		t.Fatalf("the alternate resident served %d request(s) for a kept first-path result", got-before)
	}

	// Handoff: the final result is the alternate's, named as such.
	if _, err := s.SetRouting(handoffPolicy()); err != nil {
		t.Fatal(err)
	}
	out, err = s.AutoRouter().Decide(routedItem("hand"))
	if err != nil {
		t.Fatal(err)
	}
	rr := out.Items[0].Routed[0]
	if m, pid := answerOf(t, out.Items[0].Results[0]); m != modelB || pid != pidB ||
		rr.Served.Model != modelB || rr.Served.Provider != "fake-"+modelB || rr.Reason != api.ReasonHandoffAlways ||
		rr.FirstPath == nil || rr.FirstPath.Model != modelA {
		t.Fatalf("%+v", rr)
	}
	st := s.Status()
	if st.Routing == nil || st.Routing.Handoffs != 1 || st.Routing.Policy.ID != "handoff" || len(st.Routing.Providers) != 2 {
		t.Fatalf("%+v", st.Routing)
	}

	// Direct calls after routing still name their own resident only.
	if m, pid := answerOf(t, mustDecide(t, s, modelA, "after").Results); m != modelA || pid != pidA {
		t.Fatalf("direct laya after routing: %s %d", m, pid)
	}
	if m, pid := answerOf(t, mustDecide(t, s, modelB, "after").Results); m != modelB || pid != pidB {
		t.Fatalf("direct nano after routing: %s %d", m, pid)
	}

	// The alternate goes down. A required handoff fails explicitly, nothing
	// is answered by the weaker first-path resident, and laya still serves.
	if err := s.StopResident(modelB); err != nil {
		t.Fatal(err)
	}
	if _, err := s.AutoRouter().Decide(routedItem("down")); err == nil {
		t.Fatal("a required handoff to a stopped resident must fail")
	} else {
		var re *worker.RequestError
		if !errors.As(err, &re) || re.Class != api.ErrRoutingFailed || !strings.HasPrefix(re.Message, route.CodeRequiredHandoffFailed+": model "+modelB) {
			t.Fatalf("%v", err)
		}
	}
	if m, pid := answerOf(t, mustDecide(t, s, modelA, "still").Results); m != modelA || pid != pidA {
		t.Fatalf("laya after nano stopped: %s %d", m, pid)
	}
	if got := s.RoutingStatus(); got.Failures != 1 || got.Requests != 1 {
		t.Fatalf("%+v", got)
	}

	// Neither resident was reloaded or restarted by any route.
	if residentState(s, modelA).Starts != startsA || residentState(s, modelB).Starts != startsB {
		t.Fatalf("a resident was started again: laya %d->%d, nano %d->%d", startsA, residentState(s, modelA).Starts, startsB, residentState(s, modelB).Starts)
	}
}

func TestSetRoutingRefusesAPolicyNamingANonResident(t *testing.T) {
	s := newResidentSet(t, noRestart, residentMember(t, modelA, "ok"), residentMember(t, modelB, "ok"))
	p := handoffPolicy()
	p.Rules[0].Handoff.To = "openjev"
	if _, err := s.SetRouting(p); err == nil || !strings.Contains(err.Error(), "openjev") {
		t.Fatalf("%v", err)
	}
	if s.AutoRouter() != nil {
		t.Fatal("a refused policy was bound")
	}
}

type decided struct{ Results api.Result }

func mustDecide(t *testing.T, s *ResidentSet, model, state string) decided {
	t.Helper()
	res, _, err := s.DecideOn(model, routedItem(state))
	if err != nil {
		t.Fatalf("decide on %s: %v", model, err)
	}
	return decided{res[0][0]}
}
