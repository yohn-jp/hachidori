package e2e

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixture struct {
	evidence, tests string
	cand            Candidate
}

func newFixture(t *testing.T) fixture {
	t.Helper()
	f := fixture{evidence: t.TempDir(), tests: t.TempDir()}
	f.cand = Candidate{Manifest: Manifest{SourceCommit: commitA, File: "hachidori.exe", SHA256: strings.Repeat("a", 64), Size: 7}}
	for _, shard := range Shards {
		if err := os.MkdirAll(filepath.Join(f.tests, shard), 0o755); err != nil {
			t.Fatal(err)
		}
		f.writeRequired(t, shard, `{"scenarios":["candidate-identity","extra-`+shard+`"]}`)
		f.writeResult(t, shard, func(r *Result) {})
	}
	return f
}

func (f fixture) writeRequired(t *testing.T, shard, doc string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(f.tests, shard, "required.json"), []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
}

func (f fixture) writeResult(t *testing.T, shard string, mutate func(*Result)) {
	t.Helper()
	res := Result{
		Schema:        ResultSchema,
		Shard:         shard,
		Status:        OutcomePass,
		EvidenceClass: EvidenceClass,
		Candidate:     CandidateRef{SourceCommit: f.cand.SourceCommit, File: f.cand.File, SHA256: f.cand.SHA256, Size: f.cand.Size},
		Run:           Run{ID: "42"},
		Required:      []string{"candidate-identity", "extra-" + shard},
		Scenarios: []ScenarioResult{
			{ID: "candidate-identity", Outcome: OutcomePass},
			{ID: "extra-" + shard, Outcome: OutcomePass},
		},
	}
	mutate(&res)
	dir := filepath.Join(f.evidence, EvidenceDirPrefix+shard)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(res)
	if err := os.WriteFile(filepath.Join(dir, "result.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func (f fixture) input() AggregateInput {
	return AggregateInput{
		EvidenceDir: f.evidence, TestsDir: f.tests, Candidate: f.cand, RunID: "42",
		JobResults: map[string]string{"candidate": "success", "shard": "success"},
	}
}

func TestAggregatePassesOnlyWhenEverythingAgrees(t *testing.T) {
	f := newFixture(t)
	cert := Aggregate(f.input())
	if cert.Status != OutcomePass || len(cert.Problems) != 0 || len(cert.Shards) != len(Shards) {
		t.Fatalf("complete evidence must PASS: %+v", cert)
	}
	if cert.PhysicalPass || cert.PhysicalChecks != "NOT_CHECKED" {
		t.Fatalf("hosted evidence must never be a physical PASS: %+v", cert)
	}
}

func TestAggregateFailsClosed(t *testing.T) {
	cases := map[string]func(t *testing.T, f fixture, in *AggregateInput){
		"missing shard result": func(t *testing.T, f fixture, in *AggregateInput) {
			if err := os.RemoveAll(filepath.Join(f.evidence, EvidenceDirPrefix+ShardUpdate)); err != nil {
				t.Fatal(err)
			}
		},
		"failed shard": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardRecovery, func(r *Result) { r.Status = OutcomeFail })
		},
		"shard job failed": func(t *testing.T, f fixture, in *AggregateInput) { in.JobResults["shard"] = "failure" },
		"candidate job cancelled": func(t *testing.T, f fixture, in *AggregateInput) {
			in.JobResults["candidate"] = "cancelled"
		},
		"missing job result": func(t *testing.T, f fixture, in *AggregateInput) { delete(in.JobResults, "shard") },
		"different candidate": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardRuntime, func(r *Result) { r.Candidate.SHA256 = strings.Repeat("b", 64) })
		},
		"different commit": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardRuntime, func(r *Result) { r.Candidate.SourceCommit = strings.Repeat("c", 40) })
		},
		"result from another run": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardBootstrap, func(r *Result) { r.Run.ID = "41" })
		},
		"physical pass claimed": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardBootstrap, func(r *Result) { r.PhysicalPass = true })
		},
		"required scenario skipped": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardDiagnostics, func(r *Result) { r.Scenarios[1].Outcome = OutcomeSkip })
		},
		"required scenario absent": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardDiagnostics, func(r *Result) { r.Scenarios = r.Scenarios[:1] })
		},
		"shard shrank its required set": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardUpdate, func(r *Result) {
				r.Required = []string{"candidate-identity"}
				r.Scenarios = r.Scenarios[:1]
			})
		},
		"repository requires more": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeRequired(t, ShardUpdate, `{"scenarios":["candidate-identity","extra-update","new-one"]}`)
		},
		"required.json drops candidate identity": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeRequired(t, ShardUpdate, `{"scenarios":["extra-update"]}`)
			f.writeResult(t, ShardUpdate, func(r *Result) {
				r.Required = []string{"extra-update"}
				r.Scenarios = r.Scenarios[1:]
			})
		},
		"corrupt result": func(t *testing.T, f fixture, in *AggregateInput) {
			if err := os.WriteFile(filepath.Join(f.evidence, EvidenceDirPrefix+ShardRuntime, "result.json"), []byte("{"), 0o644); err != nil {
				t.Fatal(err)
			}
		},
		"shard names another shard": func(t *testing.T, f fixture, in *AggregateInput) {
			f.writeResult(t, ShardRuntime, func(r *Result) { r.Shard = ShardUpdate })
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t)
			in := f.input()
			mutate(t, f, &in)
			cert := Aggregate(in)
			if cert.Status != OutcomeFail || len(cert.Problems) == 0 {
				t.Fatalf("%s must FAIL the certification: %+v", name, cert)
			}
		})
	}
}

func TestWriteCertificationAndReleaseGate(t *testing.T) {
	f := newFixture(t)
	cert := Aggregate(f.input())
	dir := t.TempDir()
	if err := WriteCertification(dir, cert); err != nil {
		t.Fatal(err)
	}
	md, _ := os.ReadFile(filepath.Join(dir, "certification.md"))
	if !strings.Contains(string(md), "certification: PASS") || !strings.Contains(string(md), f.cand.SHA256) {
		t.Fatalf("summary lacks verdict or identity: %s", md)
	}
	data, err := os.ReadFile(filepath.Join(dir, "certification.json"))
	if err != nil {
		t.Fatal(err)
	}
	expect := Expect{SourceCommit: f.cand.SourceCommit, SHA256: f.cand.SHA256}
	if err := CheckRelease(data, expect, "hachidori.exe"); err != nil {
		t.Fatalf("a PASS for the expected candidate must open the gate: %v", err)
	}

	for name, tc := range map[string]struct {
		expect Expect
		file   string
		data   func([]byte) []byte
	}{
		"other sha":    {Expect{SourceCommit: f.cand.SourceCommit, SHA256: strings.Repeat("d", 64)}, "", nil},
		"other commit": {Expect{SourceCommit: strings.Repeat("e", 40), SHA256: f.cand.SHA256}, "", nil},
		"no expect":    {Expect{}, "", nil},
		"other file":   {expect, "other.exe", nil},
		"failed status": {expect, "", func(b []byte) []byte {
			return []byte(strings.Replace(string(b), `"status": "PASS"`, `"status": "FAIL"`, 1))
		}},
		"missing shard": {expect, "", func(b []byte) []byte {
			var c Certification
			_ = json.Unmarshal(b, &c)
			c.Shards = c.Shards[1:]
			out, _ := json.Marshal(c)
			return out
		}},
		"physical": {expect, "", func(b []byte) []byte {
			return []byte(strings.Replace(string(b), `"physical_pass": false`, `"physical_pass": true`, 1))
		}},
		"garbage": {expect, "", func([]byte) []byte { return []byte("not json") }},
	} {
		t.Run(name, func(t *testing.T) {
			d := data
			if tc.data != nil {
				d = tc.data(data)
			}
			if err := CheckRelease(d, tc.expect, tc.file); err == nil {
				t.Fatalf("%s must keep the release gate closed", name)
			}
		})
	}

	failed := Aggregate(AggregateInput{EvidenceDir: t.TempDir(), TestsDir: f.tests, Candidate: f.cand, JobResults: map[string]string{}})
	out, _ := json.Marshal(failed)
	if err := CheckRelease(out, expect, ""); err == nil {
		t.Fatal("a FAIL certification must keep the release gate closed")
	}
}
