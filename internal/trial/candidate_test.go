package trial

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// measuredCandidate runs one trial and derives its candidate and evidence.
func measuredCandidate(t *testing.T, w world, quantize ...string) (Candidate, Evidence, Result) {
	t.Helper()
	s, _ := openSession(t, w, Options{})
	prof := w.profileW4(t, quantize...)
	c0, err := tuningCompile(w, prof)
	if err != nil {
		t.Fatal(err)
	}
	r := mustRun(t, s, c0, &evaluator{})
	c, err := NewCandidate(w.source, w.ref(t, prof), r, fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	e, err := NewEvidence(c, r, fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	return c, e, r
}

func TestCandidateBindsSourceProfilePlanAndTheExactComponentSet(t *testing.T) {
	w := newWorld(t)
	c, e, r := measuredCandidate(t, w, mlp0, mlp1)
	if c.Source != w.source || c.Profile.ID != w.profileW4(t, mlp0, mlp1).ID() || c.PlanSHA256 != r.Plan.SHA256() {
		t.Fatalf("candidate %+v", c)
	}
	if len(c.Components) != len(r.Plan.Groups) || c.ComponentsSHA256 == "" {
		t.Fatalf("components %d", len(c.Components))
	}
	if e.CandidateID != c.ID || e.ComponentsSHA256 != c.ComponentsSHA256 || e.PlanSHA256 != c.PlanSHA256 || e.ProfileID != c.Profile.ID || e.Source != c.Source {
		t.Fatalf("evidence does not bind the candidate: %+v", e)
	}
	// The evidence records how the trial was assembled and the cache it used.
	if e.Execution.Assembly.CacheMisses != 2 || len(e.Execution.Assembly.Transformed) != 2 || e.Execution.Cache.BudgetBytes == 0 || e.Execution.Resource.GPUAllocatedBytes == nil {
		t.Fatalf("execution evidence %+v", e.Execution)
	}
	if e.Measurement.DatasetSHA256 == "" || e.Measurement.InputSHA256 == "" || len(e.Measurement.Questions) == 0 || e.Measurement.Status != MeasurementMeasured {
		t.Fatalf("measurement does not bind the evaluation: %+v", e.Measurement)
	}
	if e.Measurement.Mode != EvalModeTrial {
		t.Fatalf("evaluation mode %q is not recorded as a trial", e.Measurement.Mode)
	}
}

func TestCandidateIdentityFollowsItsInputsOnly(t *testing.T) {
	w := newWorld(t)
	a, _, _ := measuredCandidate(t, w, mlp0)
	again, _, _ := measuredCandidate(t, w, mlp0)
	if a.ID != again.ID {
		t.Fatal("the same source, profile, plan and components give different candidates")
	}
	other, _, _ := measuredCandidate(t, w, mlp0, mlp1)
	if other.ID == a.ID {
		t.Fatal("a different plan has the same candidate identity")
	}
	moved := w
	moved.source.Revision = strings.Repeat("cd", 20)
	c := a
	c.Source = moved.source
	if err := c.Validate(); err == nil {
		t.Fatal("a candidate whose source was changed kept its identity")
	}
	d := a
	d.Components = append([]ComponentRef(nil), a.Components...)
	for i := range d.Components {
		if d.Components[i].Digest != "" {
			d.Components[i].Digest = strings.Repeat("0", 64)
			break
		}
	}
	if err := d.Validate(); err == nil {
		t.Fatal("a candidate whose component digest was changed kept its identity")
	}
	// A different transformed component set is a different candidate.
	_, _, r := measuredCandidate(t, w, mlp0)
	r2 := r
	r2.Components = append([]ComponentRef(nil), r.Components...)
	for i := range r2.Components {
		if r2.Components[i].Policy == home.PolicyW4A16 {
			r2.Components[i].Digest = "different-content"
		}
	}
	changed, err := NewCandidate(w.source, a.Profile, r2, fixedNow)
	if err != nil || changed.ID == a.ID {
		t.Fatalf("component digest is not part of the identity (%v)", err)
	}
}

func TestEvidenceIsNeverCertificationAndNeverAnArtifactMeasurement(t *testing.T) {
	w := newWorld(t)
	_, e, _ := measuredCandidate(t, w, mlp0)
	if !e.Ephemeral || e.Certification != CertificationNone || e.Schema != EvidenceSchema {
		t.Fatalf("evidence %+v", e)
	}
	raw := string(e.Canonical())
	for _, forbidden := range []string{"hachidori.certification", "hachidori.evidence.v1", "hachidori.resident-run"} {
		if strings.Contains(raw, forbidden) {
			t.Fatalf("trial evidence names %q", forbidden)
		}
	}
	for name, mutate := range map[string]func(*Evidence){
		"certified":     func(e *Evidence) { e.Certification = "CERTIFIED" },
		"not ephemeral": func(e *Evidence) { e.Ephemeral = false },
		"mode":          func(e *Evidence) { e.Measurement.Mode = "certification" },
		"variant":       func(e *Evidence) { e.Measurement.Served.VariantID = "abc" },
		"schema":        func(e *Evidence) { e.Schema = "hachidori.certification/1" },
		"unbound":       func(e *Evidence) { e.CandidateID = "" },
		"served":        func(e *Evidence) { e.Measurement.Served.Execution = "variant" },
	} {
		bad := e
		mutate(&bad)
		if err := bad.Validate(); err == nil {
			t.Errorf("%s: evidence accepted", name)
		}
	}
}

func TestEvidenceRefusesAResultThatIsNotTheCandidates(t *testing.T) {
	w := newWorld(t)
	c, _, r := measuredCandidate(t, w, mlp0)
	_, _, other := measuredCandidate(t, w, mlp0, mlp1)
	if _, err := NewEvidence(c, other, fixedNow); err == nil {
		t.Fatal("evidence of one trial was bound to another trial's candidate")
	}
	if _, err := NewEvidence(c, r, fixedNow); err != nil {
		t.Fatal(err)
	}
}

func TestStoreKeepsImmutableCandidateAndAppendsEvidence(t *testing.T) {
	w := newWorld(t)
	c, e, r := measuredCandidate(t, w, mlp0, mlp1)
	if err := Save(w.h, c, e); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCandidate(w.h, c.ID)
	if err != nil || loaded.ID != c.ID || loaded.PlanSHA256 != c.PlanSHA256 {
		t.Fatalf("load: %+v (%v)", loaded, err)
	}
	st, err := StatusOf(w.h, c.ID)
	if err != nil || st.Measurements != 1 || st.Stage() != "measured" || len(st.Variants) != 0 {
		t.Fatalf("status %+v (%v)", st, err)
	}
	// A second measurement of the same candidate is a second evidence document;
	// the candidate file is not rewritten.
	before, _ := os.ReadFile(filepath.Join(Dir(w.h, c.ID), "candidate.json"))
	r.Measurement.Cases = 8
	e2, err := NewEvidence(c, r, fixedNow.Add(1))
	if err != nil {
		t.Fatal(err)
	}
	if err := Save(w.h, c, e2); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(filepath.Join(Dir(w.h, c.ID), "candidate.json"))
	if string(before) != string(after) {
		t.Fatal("saving evidence rewrote the candidate")
	}
	ev, err := LoadEvidence(w.h, c.ID)
	if err != nil || len(ev) != 2 {
		t.Fatalf("evidence %d (%v)", len(ev), err)
	}
	list, err := List(w.h, w.model.ID)
	if err != nil || len(list) != 1 || list[0].ID != c.ID {
		t.Fatalf("list %v (%v)", list, err)
	}
	if other, _ := List(w.h, "another-model"); len(other) != 0 {
		t.Fatal("list crossed models")
	}
}

func TestStoreRefusesTamperedDocumentsAndMismatchedEvidence(t *testing.T) {
	w := newWorld(t)
	c, e, _ := measuredCandidate(t, w, mlp0)
	other, oe, _ := measuredCandidate(t, w, mlp0, mlp1)
	if err := Save(w.h, c, oe); err == nil {
		t.Fatal("evidence of another candidate was saved")
	}
	_ = other
	if err := Save(w.h, c, e); err != nil {
		t.Fatal(err)
	}
	// Tamper with the stored evidence: it no longer matches its identity.
	dir := filepath.Join(Dir(w.h, c.ID), "evidence")
	entries, _ := os.ReadDir(dir)
	path := filepath.Join(dir, entries[0].Name())
	raw, _ := os.ReadFile(path)
	if err := os.WriteFile(path, []byte(strings.Replace(string(raw), `"cases": 4`, `"cases": 400`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEvidence(w.h, c.ID); err == nil {
		t.Fatal("tampered evidence loaded")
	}
	// A stored candidate whose plan was edited is invalid.
	cpath := filepath.Join(Dir(w.h, c.ID), "candidate.json")
	var doc map[string]any
	b, _ := os.ReadFile(cpath)
	_ = json.Unmarshal(b, &doc)
	doc["plan_sha256"] = strings.Repeat("0", 64)
	b, _ = json.Marshal(doc)
	_ = os.WriteFile(cpath, b, 0o644)
	if _, err := LoadCandidate(w.h, c.ID); err == nil {
		t.Fatal("a candidate with an edited plan digest loaded")
	}
	if _, err := LoadCandidate(w.h, "../x"); err == nil {
		t.Fatal("a non-digest candidate ID was accepted")
	}
}

func TestMeasurementOfRefusesRunsThatAreNotMeasurements(t *testing.T) {
	m := measured()
	m.Mode = "certification"
	if err := m.Validate(); err == nil {
		t.Fatal("a certification measurement validated as trial evidence")
	}
	m = measured()
	m.Questions = nil
	if err := m.Validate(); err == nil {
		t.Fatal("a measurement that binds no questions validated")
	}
	m = measured()
	m.Status = "certified"
	if err := m.Validate(); err == nil {
		t.Fatal("an unknown status validated")
	}
	nc := Measurement{Mode: EvalModeTrial, Status: MeasurementNotChecked}
	if err := nc.Validate(); err != nil {
		t.Fatalf("a not-checked measurement is an honest record: %v", err)
	}
	_ = context.Background
}
