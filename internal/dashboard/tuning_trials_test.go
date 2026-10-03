package dashboard

import (
	"errors"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/trial"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// trialTuning adds the optional trial projection to the fake tuning authority.
type trialTuning struct {
	*fakeTuning
	records []TuningTrialRecord
	err     error
}

func (f *trialTuning) Trials(source string) ([]TuningTrialRecord, error) { return f.records, f.err }

func trialRecord(planSHA, profileID string, variants ...string) TuningTrialRecord {
	acc, p50 := 0.9125, 41.5
	gpu, rss := int64(7<<30), int64(9<<30)
	cand := trial.Candidate{ID: strings.Repeat("c", 64), Profile: trial.ProfileRef{ID: profileID}, PlanSHA256: planSHA, ComponentsSHA256: strings.Repeat("d", 64),
		Components: make([]trial.ComponentRef, 5)}
	ev := trial.Evidence{CandidateID: cand.ID, Certification: trial.CertificationNone, Ephemeral: true,
		Measurement: trial.Measurement{Mode: trial.EvalModeTrial, Status: trial.MeasurementMeasured, DatasetSHA256: strings.Repeat("e", 64), Accuracy: &acc, LatencyP50MS: &p50},
		Execution: trial.Execution{EvaluationMS: 1200, TotalMS: 1450,
			Assembly: trial.Assembly{Mode: trial.AssemblyDifferential, Changed: make([]trial.Change, 2), Reused: 12, CacheHits: 1, CacheMisses: 1, Transformed: []string{"x"},
				BytesToGPU: 3 << 20, BytesReleased: 12 << 20, AssemblyMS: 250},
			Cache:    trial.Stats{BudgetBytes: 40 << 30, CurrentBytes: 11 << 30, CanonicalBytes: 9 << 30, TransformedBytes: 2 << 30},
			Resource: trial.Resource{GPUAllocatedBytes: &gpu, HostRSSBytes: &rss}}}
	st := trial.Status{Measurements: 1, Variants: variants}
	return TuningTrialRecord{Candidate: cand, Status: st, Evidence: []trial.Evidence{ev}}
}

func shownPlanSHA(t *testing.T, ft *fakeTuning) (string, string) {
	t.Helper()
	p := expectedProfile(t, ft.analysis, "balanced")
	c, err := tuning.Compile(p, ft.analysis)
	if err != nil {
		t.Fatal(err)
	}
	return c.Plan.SHA256(), p.ID()
}

func TestTuningShowsTheEphemeralTrialOfTheShownPlanWithoutCertificationClaims(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	sha, id := shownPlanSHA(t, ft)
	withTuning(e, &trialTuning{fakeTuning: ft, records: []TuningTrialRecord{trialRecord(sha, id), trialRecord(strings.Repeat("9", 64), strings.Repeat("8", 64), "variant-1")}})
	body := e.get(t, "/tuning").Body.String()
	ws := workspace(body)
	cur := between(t, ws, `id="tuning-trial-current"`, `</span>`)
	for _, want := range []string{"EPHEMERAL TRIAL", `data-trial-stage="measured"`, `data-evaluation="trial-fast"`, "recorded as a candidate, no artifact"} {
		if !strings.Contains(cur, want) {
			t.Errorf("current trial lacks %q: %s", want, cur)
		}
	}
	for _, want := range []string{"0.9125", "41.5 ms", "250 ms", "not certification", "A clean-loaded variant is evaluated separately in Forge."} {
		if !strings.Contains(ws, want) {
			t.Errorf("trial region lacks %q", want)
		}
	}
	if !strings.Contains(ws, `data-trials="2"`) {
		t.Error("the recorded candidates are not listed")
	}
	// A trial is never presented as MEASURED evidence of an artifact or as certified.
	region := between(t, ws, `id="tuning-trials"`, `</dd></div>`)
	for _, forbidden := range []string{`data-state="MEASURED"`, "CERTIFIED", "Certified", "accepted"} {
		if strings.Contains(region, forbidden) {
			t.Errorf("the trial region says %q", forbidden)
		}
	}
	// The materialized candidate shows its variant and offers no second build.
	rows := between(t, ws, `id="tuning-trials-table"`, `</table>`)
	mat := between(t, rows, `data-trial-stage="materialized"`, `</tr>`)
	if !strings.Contains(mat, "Variant built") || strings.Contains(mat, "Forge as finalist") {
		t.Errorf("materialized row: %s", mat)
	}
	meas := between(t, rows, `data-trial-stage="measured"`, `</tr>`)
	if !strings.Contains(meas, "Forge as finalist") || !strings.Contains(meas, "/forge?") || !strings.Contains(meas, "profile="+tuningProfilePrefixQuery(id)) {
		t.Errorf("measured row does not hand its exact profile to Forge: %s", meas)
	}
}

func tuningProfilePrefixQuery(id string) string {
	return strings.NewReplacer(":", "%3A").Replace(tuningProfilePrefix) + id
}

func between(t *testing.T, s, from, to string) string {
	t.Helper()
	i := strings.Index(s, from)
	if i < 0 {
		t.Fatalf("%q not found", from)
	}
	j := strings.Index(s[i:], to)
	if j < 0 {
		t.Fatalf("%q not found after %q", to, from)
	}
	return s[i : i+j]
}

func TestTuningTrialExecutionDetailsAreEvidenceOnly(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	sha, id := shownPlanSHA(t, ft)
	withTuning(e, &trialTuning{fakeTuning: ft, records: []TuningTrialRecord{trialRecord(sha, id)}})
	ws := workspace(e.get(t, "/tuning").Body.String())
	// The normal view states resource pressure only through the trial's facts;
	// cache mechanics live in the Evidence disclosure.
	evidenceAt := strings.Index(ws, `id="tuning-evidence-details"`)
	detail := strings.Index(ws, `id="tuning-trial-evidence"`)
	if detail < 0 || evidenceAt < 0 || detail < evidenceAt {
		t.Fatalf("trial execution evidence is not inside the Details/Evidence disclosure (%d, %d)", detail, evidenceAt)
	}
	normal := ws[:evidenceAt]
	for _, mechanic := range []string{"misses", "evicted", "RAM to GPU", "components transformed"} {
		if strings.Contains(normal, mechanic) {
			t.Errorf("the normal view shows cache mechanics: %q", mechanic)
		}
	}
	section := between(t, ws, `id="tuning-trial-evidence"`, `</section>`)
	for _, want := range []string{"differential", "2 groups replaced", "12 groups reused", "1 hits", "1 misses", "1 components transformed", "3.0 MiB", "12.0 MiB",
		"11.0 GiB / 40.0 GiB", "9.0 GiB canonical source", "2.0 GiB transformed components", "7.0 GiB", "1200 ms", "never certification"} {
		if !strings.Contains(section, want) {
			t.Errorf("trial evidence lacks %q", want)
		}
	}
}

func TestTuningWithoutATrialForTheShownPlanSaysSoAndStartsNothing(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	withTuning(e, &trialTuning{fakeTuning: ft, records: []TuningTrialRecord{trialRecord(strings.Repeat("9", 64), strings.Repeat("8", 64))}})
	ws := workspace(e.get(t, "/tuning").Body.String())
	if !strings.Contains(ws, `id="tuning-trial-none"`) || strings.Contains(ws, `id="tuning-trial-current"`) || !strings.Contains(ws, "No trial recorded for this exact plan.") {
		t.Error("a plan without a trial is not stated as such")
	}
	if strings.Contains(ws, `id="tuning-trial-evidence"`) {
		t.Error("evidence of another plan is shown for this one")
	}
	if strings.Contains(ws, "/tuning/trial") || strings.Contains(ws, `action="/tuning/run`) {
		t.Error("the page offers to run a trial")
	}
}

func TestTuningTrialsAreAbsentWhereTheApplicationProvidesNone(t *testing.T) {
	e, _, _ := tuningEnv(t)
	ws := workspace(e.get(t, "/tuning").Body.String())
	if strings.Contains(ws, `id="tuning-trials"`) || strings.Contains(ws, "EPHEMERAL TRIAL") {
		t.Error("a trial region without a trial authority")
	}
}

func TestTuningReportsUnreadableTrialsInsteadOfHidingThem(t *testing.T) {
	e, ft, _ := tuningEnv(t)
	withTuning(e, &trialTuning{fakeTuning: ft, err: errors.New("disk unreadable")})
	ws := workspace(e.get(t, "/tuning").Body.String())
	if !strings.Contains(ws, `id="tuning-trials-unavailable"`) || !strings.Contains(ws, "disk unreadable") {
		t.Error("an unreadable trial record is not reported")
	}
	_ = home.PolicyW4A16
}
