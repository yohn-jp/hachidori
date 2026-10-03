package dashboard

import (
	"fmt"
	"net/url"
	"slices"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/trial"
)

// TuningTrials is the optional trial projection of the tuning authority: the
// RAM-resident trial Candidates recorded for one source, with their Evidence
// and how far each progressed. The dashboard only reads it; trials are run by
// `hachidori forge trial` or the controller operation, never by rendering.
type TuningTrials interface {
	Trials(source string) ([]TuningTrialRecord, error)
}

// TuningTrialRecord is one recorded Candidate with its measurements.
type TuningTrialRecord struct {
	Candidate trial.Candidate
	Status    trial.Status
	Evidence  []trial.Evidence
}

// maxTrialRows bounds the recorded candidates listed on the page.
const maxTrialRows = 12

// TuningTrialRow is one recorded trial candidate. Everything it shows was
// measured on an ephemeral trial of the source model: it is never certification
// and never a measurement of an artifact.
type TuningTrialRow struct {
	CandidateID string
	ProfileID   string
	// Current is set for the candidate of exactly the plan shown in the editor.
	Current bool
	// Stage is "measured" (recorded, no artifact) or "materialized" (built by
	// Forge as an immutable Variant, which still needs its own certification).
	Stage    string
	Variants []string
	Accuracy string
	Latency  string
	Assembly string
	Total    string
	Dataset  string
	Mode     string
	// FinalistHref selects this candidate's exact profile in Forge, which
	// materializes it as an immutable Variant. It builds nothing by itself.
	FinalistHref string
	Detail       *TuningTrialDetail
}

// TuningTrialDetail is the Evidence-level account of how the latest measurement
// of a candidate was assembled. It is shown only under Details/Evidence.
type TuningTrialDetail struct {
	Assembly           string
	Changed, Reused    int
	CacheHits, Misses  int
	Transformed        int
	Evicted            int
	ToGPU, Released    string
	RAMUsed, RAMBudget string
	RAMCanonical       string
	RAMTransformed     string
	GPUAllocated       string
	HostRSS            string
	Evaluation         string
	Components         int
	ComponentsSHA256   string
	PlanSHA256         string
	EvidenceID         string
	Measurements       int
}

// TuningTrialsView is the Trials region of the Tuning page.
type TuningTrialsView struct {
	Rows []TuningTrialRow
	// Current is the row of the shown plan, nil when no trial recorded it.
	Current *TuningTrialRow
	// More counts recorded candidates not listed.
	More int
	Err  string
}

func (d *Dashboard) trialsView(source, planSHA string) *TuningTrialsView {
	src, ok := d.cfg.Tuning.(TuningTrials)
	if !ok {
		return nil
	}
	v := &TuningTrialsView{}
	records, err := src.Trials(source)
	if err != nil {
		v.Err = "Recorded trials could not be read: " + err.Error()
		return v
	}
	for _, r := range records {
		row := trialRow(r)
		row.FinalistHref = "/forge?" + url.Values{"source": {source}, "profile": {tuningProfilePrefix + r.Candidate.Profile.ID}}.Encode() + "#forge-intent-form"
		row.Current = r.Candidate.PlanSHA256 == planSHA
		if row.Current && v.Current == nil {
			cur := row
			v.Current = &cur
		}
		if len(v.Rows) == maxTrialRows {
			v.More++
			continue
		}
		v.Rows = append(v.Rows, row)
	}
	return v
}

func trialRow(r TuningTrialRecord) TuningTrialRow {
	c := r.Candidate
	row := TuningTrialRow{CandidateID: c.ID, ProfileID: c.Profile.ID, Stage: r.Status.Stage(), Variants: slices.Clone(r.Status.Variants), Mode: trial.EvalModeTrial}
	if len(r.Evidence) == 0 {
		return row
	}
	e := r.Evidence[len(r.Evidence)-1] // oldest first: the last is the latest
	m, x := e.Measurement, e.Execution
	row.Dataset = short12(m.DatasetSHA256)
	row.Mode = m.Mode
	if m.Accuracy != nil {
		row.Accuracy = fmt.Sprintf("%.4f", *m.Accuracy)
	}
	if m.LatencyP50MS != nil {
		row.Latency = fmt.Sprintf("%.1f ms", *m.LatencyP50MS)
	}
	row.Assembly = fmt.Sprintf("%.0f ms", x.Assembly.AssemblyMS)
	row.Total = fmt.Sprintf("%.0f ms", x.TotalMS)
	d := &TuningTrialDetail{Assembly: x.Assembly.Mode, Changed: len(x.Assembly.Changed), Reused: x.Assembly.Reused, CacheHits: x.Assembly.CacheHits,
		Misses: x.Assembly.CacheMisses, Transformed: len(x.Assembly.Transformed), Evicted: len(x.Assembly.Evicted),
		ToGPU: bytesIn(x.Assembly.BytesToGPU), Released: bytesIn(x.Assembly.BytesReleased),
		RAMUsed: bytesIn(x.Cache.CurrentBytes), RAMBudget: bytesIn(x.Cache.BudgetBytes),
		RAMCanonical: bytesIn(x.Cache.CanonicalBytes), RAMTransformed: bytesIn(x.Cache.TransformedBytes),
		Evaluation: fmt.Sprintf("%.0f ms", x.EvaluationMS), Components: len(c.Components), ComponentsSHA256: c.ComponentsSHA256,
		PlanSHA256: c.PlanSHA256, EvidenceID: e.ID(), Measurements: len(r.Evidence)}
	if x.Resource.GPUAllocatedBytes != nil {
		d.GPUAllocated = bytesIn(*x.Resource.GPUAllocatedBytes)
	}
	if x.Resource.HostRSSBytes != nil {
		d.HostRSS = bytesIn(*x.Resource.HostRSSBytes)
	}
	row.Detail = d
	return row
}

// planSHAOf is the digest of the resolved plan shown in the editor.
func planSHAOf(plan home.TuningPlan) string { return plan.SHA256() }
