// Command e2e-aggregate decides the certification of one Windows E2E run and
// gates the development release on it.
//
//	e2e-aggregate aggregate -evidence DIR -tests DIR -candidate DIR -commit SHA -sha256 HEX \
//	    -jobs candidate=success,shard=success -out DIR
//	e2e-aggregate release-check -certification FILE -commit SHA -sha256 HEX [-file NAME]
//
// aggregate always writes certification.json and certification.md and exits
// non-zero unless every required shard passed for exactly the candidate.
// release-check exits non-zero unless certification.json is a PASS for exactly
// the expected candidate.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: e2e-aggregate aggregate|release-check [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "aggregate":
		os.Exit(aggregate(os.Args[2:]))
	case "release-check":
		os.Exit(releaseCheck(os.Args[2:]))
	default:
		fmt.Fprintf(os.Stderr, "e2e-aggregate: unknown command %q\n", os.Args[1])
		os.Exit(2)
	}
}

func aggregate(args []string) int {
	fs := flag.NewFlagSet("aggregate", flag.ContinueOnError)
	evidence := fs.String("evidence", "evidence", "directory holding the downloaded shard evidence artifacts")
	tests := fs.String("tests", "test/windows-e2e", "directory holding the shard packages")
	candidate := fs.String("candidate", "candidate", "directory holding the downloaded candidate")
	commit := fs.String("commit", "", "expected source commit SHA")
	sum := fs.String("sha256", "", "expected candidate SHA-256 (the candidate job's output)")
	jobs := fs.String("jobs", "", "comma-separated job=result pairs for candidate and shard")
	runID := fs.String("run-id", os.Getenv("GITHUB_RUN_ID"), "workflow run the results must belong to")
	out := fs.String("out", "certification", "output directory")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	results := map[string]string{}
	for _, kv := range strings.Split(*jobs, ",") {
		if k, v, ok := strings.Cut(kv, "="); ok {
			results[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
	}

	var in e2e.AggregateInput
	in.EvidenceDir, in.TestsDir, in.RunID, in.JobResults = *evidence, *tests, *runID, results

	var preflight []string
	if *commit == "" || *sum == "" {
		preflight = append(preflight, "the expected source commit and candidate SHA-256 are required")
	} else if c, err := e2e.LoadCandidate(*candidate, e2e.Expect{SourceCommit: *commit, SHA256: *sum}); err != nil {
		preflight = append(preflight, "candidate identity: "+err.Error())
	} else {
		in.Candidate = c
	}

	cert := e2e.Aggregate(in)
	if len(preflight) > 0 {
		cert.Status = e2e.OutcomeFail
		cert.Problems = append(preflight, cert.Problems...)
	}
	if err := e2e.WriteCertification(*out, cert); err != nil {
		fmt.Fprintln(os.Stderr, "e2e-aggregate:", err)
		return 2
	}
	fmt.Printf("status=%s\n", cert.Status)
	for _, p := range cert.Problems {
		fmt.Fprintln(os.Stderr, "e2e-aggregate:", p)
	}
	if cert.Status != e2e.OutcomePass {
		return 1
	}
	return 0
}

func releaseCheck(args []string) int {
	fs := flag.NewFlagSet("release-check", flag.ContinueOnError)
	cert := fs.String("certification", "certification/certification.json", "certification.json to check")
	commit := fs.String("commit", "", "expected source commit SHA")
	sum := fs.String("sha256", "", "expected candidate SHA-256")
	file := fs.String("file", "", "expected candidate file name")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	data, err := os.ReadFile(*cert)
	if err != nil {
		fmt.Fprintln(os.Stderr, "e2e-aggregate release-check:", err)
		return 1
	}
	if err := e2e.CheckRelease(data, e2e.Expect{SourceCommit: *commit, SHA256: *sum}, *file); err != nil {
		fmt.Fprintln(os.Stderr, "e2e-aggregate release-check:", err)
		return 1
	}
	fmt.Println("release gate: certification PASS for the expected candidate")
	return 0
}
