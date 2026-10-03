package home

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testPlan() TuningPlan {
	return TuningPlan{Schema: TuningPlanSchema, AutoPolicy: "clef-auto/1", Recipe: "clef-flash-w4a16-rtn-g128", RecipeSHA256: strings.Repeat("a", 64),
		Groups: []TuningGroupPlan{
			{ID: "block.00.mlp", Region: "feed-forward-projections", Layer: 0, Modules: []string{"m0"}, Selection: SelectionAuto, Requested: PolicyAuto, Effective: PolicyW4A16, Basis: "AUTO"},
			{ID: "block.01.mlp", Region: "feed-forward-projections", Layer: 1, Modules: []string{"m1"}, Selection: SelectionOverridden, Requested: PolicySourcePrecision, Effective: PolicySourcePrecision, Basis: "operator override"},
			{ID: "joint-schema-head", Region: "joint-schema-head", Layer: -1, Files: []string{"joint_head.safetensors"}, Selection: SelectionAuto, Requested: PolicyAuto, Effective: PolicySourcePrecision, Required: true, Basis: "required"},
		}}
}

func TestTuningPlanIdentityIsExactAndValidated(t *testing.T) {
	p := testPlan()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if p.SHA256() != testPlan().SHA256() {
		t.Fatal("identical plans have different identities")
	}
	bad := map[string]func(*TuningPlan){
		"schema":               func(p *TuningPlan) { p.Schema = "x" },
		"no groups":            func(p *TuningPlan) { p.Groups = nil },
		"unsorted":             func(p *TuningPlan) { p.Groups[0], p.Groups[1] = p.Groups[1], p.Groups[0] },
		"unsupported policy":   func(p *TuningPlan) { p.Groups[0].Effective = "w8a16" },
		"AUTO requests policy": func(p *TuningPlan) { p.Groups[0].Requested = PolicyW4A16 },
		"override differs":     func(p *TuningPlan) { p.Groups[1].Effective = PolicyW4A16 },
		"override of required": func(p *TuningPlan) {
			p.Groups[2].Selection, p.Groups[2].Requested = SelectionOverridden, PolicySourcePrecision
		},
		"required is quantized": func(p *TuningPlan) { p.Groups[2].Effective = PolicyW4A16 },
		"unknown selection":     func(p *TuningPlan) { p.Groups[0].Selection = "MANUAL" },
		"no members":            func(p *TuningPlan) { p.Groups[0].Modules = nil },
	}
	for name, mutate := range bad {
		q := testPlan()
		mutate(&q)
		if err := q.Validate(); err == nil {
			t.Errorf("%s: invalid plan accepted", name)
		}
	}
	changed := testPlan()
	changed.Groups[0].Effective, changed.Groups[0].Selection, changed.Groups[0].Requested = PolicySourcePrecision, SelectionOverridden, PolicySourcePrecision
	if changed.SHA256() == p.SHA256() {
		t.Fatal("a changed effective policy did not change the plan identity")
	}
}

func TestLayerwiseProvenanceNeedsAPlanAndLegacyCannotHaveOne(t *testing.T) {
	v1 := testTuning()
	if err := v1.Validate(SourceOf(testSource())); err != nil {
		t.Fatal(err)
	}
	withPlan := *v1
	withPlan.PlanSHA256 = strings.Repeat("c", 64)
	if err := withPlan.Validate(SourceOf(testSource())); err == nil {
		t.Error("legacy provenance with a plan accepted")
	}
	v2 := withPlan
	v2.Schema = TuningProvenanceSchema
	if err := v2.Validate(SourceOf(testSource())); err != nil {
		t.Fatal(err)
	}
	noDigest := v2
	noDigest.PlanSHA256 = ""
	if err := noDigest.Validate(SourceOf(testSource())); err == nil {
		t.Error("layer-wise provenance without a plan digest accepted")
	}

	// the effective plan is part of the build contract; a legacy record is not changed
	a, b := testVariant(nil), testVariant(nil)
	pa, pb := v2, v2
	pa.PlanSHA256, pb.PlanSHA256 = strings.Repeat("c", 64), strings.Repeat("d", 64)
	a.Tuning, b.Tuning = &pa, &pb
	a.Seal()
	b.Seal()
	if a.BuildID == b.BuildID || a.ID == b.ID {
		t.Fatal("a different effective plan did not change the build and variant identity")
	}
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestTuningEvidenceMustShowEveryGroupApplyingItsPlan(t *testing.T) {
	dir := t.TempDir()
	ev := TuningEvidence{Schema: TuningEvidenceSchema, PlanSHA256: strings.Repeat("a", 64), AutoPolicy: "clef-auto/1", Groups: []TuningGroupApplied{
		{ID: "a", Selection: SelectionAuto, Requested: PolicyAuto, Effective: PolicyW4A16, Applied: PolicyW4A16, Modules: 3},
		{ID: "b", Selection: SelectionOverridden, Requested: PolicySourcePrecision, Effective: PolicySourcePrecision, Applied: PolicySourcePrecision, Preserved: true, Modules: 1, WrittenDType: "BF16"},
	}}
	write := func(e TuningEvidence) {
		if err := os.WriteFile(filepath.Join(dir, TuningEvidenceFile), e.Canonical(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(ev)
	got, err := ReadTuningEvidence(dir)
	if err != nil || len(got.Groups) != 2 || got.Groups[1].WrittenDType != "BF16" {
		t.Fatalf("evidence %+v, %v", got, err)
	}
	ev.Groups[0].Applied = PolicySourcePrecision
	write(ev)
	if _, err := ReadTuningEvidence(dir); err == nil {
		t.Error("evidence whose applied transformation differs from the resolved policy was accepted")
	}
	if _, err := ReadTuningEvidence(t.TempDir()); !os.IsNotExist(unwrap(err)) {
		t.Errorf("missing evidence is not reported as absent: %v", err)
	}
}

func unwrap(err error) error {
	for err != nil {
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return err
		}
		err = u.Unwrap()
	}
	return err
}
