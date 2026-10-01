package route

import (
	"maps"

	"github.com/yohn-jp/hachidori/internal/api"
)

// Status is the observable routing state: counters since the router was
// built, per reason code and per resident. It is the routing part of the
// /v1/status document; the residents' own status stays where it is.
type Status struct {
	Policy          api.PolicyRef    `json:"policy"`
	Requests        int64            `json:"requests"`
	Failures        int64            `json:"failures"`
	Questions       int64            `json:"questions"`
	Handoffs        int64            `json:"handoffs"`
	HandoffFailures int64            `json:"handoff_failures"`
	Reasons         map[string]int64 `json:"reasons"`
	Providers       []ProviderStatus `json:"providers"`
	Calibration     *Calibration     `json:"calibration,omitempty"`
}

// ProviderStatus is one resident's contribution to routed requests.
// FirstPathQuestions counts the questions it answered first, HandoffQuestions
// the questions it received by handoff, FinalResults the results it produced
// that were returned. Calls and InferenceMSTotal are its worker calls made
// for answered routed requests and their inference time.
type ProviderStatus struct {
	Model              string  `json:"model"`
	Provider           string  `json:"provider"`
	FirstPathQuestions int64   `json:"first_path_questions"`
	HandoffQuestions   int64   `json:"handoff_questions"`
	FinalResults       int64   `json:"final_results"`
	Calls              int64   `json:"calls"`
	InferenceMSTotal   float64 `json:"inference_ms_total"`
}

type stats struct {
	s     Status
	byMod map[string]*ProviderStatus
}

func newStats(models []string, served map[string]api.Served) stats {
	st := stats{byMod: map[string]*ProviderStatus{}}
	st.s.Reasons = map[string]int64{}
	for _, m := range models {
		st.s.Providers = append(st.s.Providers, ProviderStatus{Model: m, Provider: served[m].Provider})
	}
	for i := range st.s.Providers {
		st.byMod[st.s.Providers[i].Model] = &st.s.Providers[i]
	}
	return st
}

// record counts one routed request.
func (r *Router) record(out Outcome, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err != nil {
		r.stats.s.Failures++
		return
	}
	s := &r.stats
	s.s.Requests++
	s.s.Handoffs += int64(out.Handoffs)
	for _, t := range out.Providers {
		if p := s.byMod[t.Model]; p != nil {
			p.Calls += int64(t.Calls)
			p.InferenceMSTotal += t.InferenceMS
		}
	}
	for _, it := range out.Items {
		for _, rr := range it.Routed {
			s.s.Questions++
			s.s.Reasons[rr.Reason]++
			first := rr.Served.Model
			if rr.FirstPath != nil {
				first = rr.FirstPath.Model
				s.byMod[rr.Served.Model].HandoffQuestions++
			}
			s.byMod[first].FirstPathQuestions++
			s.byMod[rr.Served.Model].FinalResults++
		}
	}
}

// noteHandoffFailure counts questions whose optional handoff failed.
func (r *Router) noteHandoffFailure(questions int) {
	r.mu.Lock()
	r.stats.s.HandoffFailures += int64(questions)
	r.mu.Unlock()
}

// SetCalibration records the evidence the policy was verified against (see
// Verify) so status shows it.
func (r *Router) SetCalibration(c Calibration) {
	r.mu.Lock()
	r.stats.s.Calibration = &c
	r.mu.Unlock()
}

// Status is a copy of the routing counters.
func (r *Router) Status() Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.stats.s
	s.Policy = r.ref
	s.Reasons = maps.Clone(s.Reasons)
	s.Providers = append([]ProviderStatus(nil), s.Providers...)
	return s
}
