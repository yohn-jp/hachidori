# Windows appliance E2E certification

Main-push certification of the Windows `hachidori.exe` on GitHub-hosted
runners. It runs from `.github/workflows/windows-e2e.yml` on `push` to `main`
and `workflow_dispatch`, never on `pull_request`.

```
main push -> build candidate once -> parallel shards -> (aggregate -> development release)
```

Everything proven here is **CI evidence**. It never becomes a physical Windows
`PASS`; the assertions that need a human, a real desktop, a production model or
NVIDIA hardware stay `NOT_CHECKED` in
[docs/windows-certification-checklist.md](../../docs/windows-certification-checklist.md),
which classifies every assertion as `CI_AUTOMATED`, `PHYSICAL_REQUIRED` or
`OPTIONAL_HARDWARE`. #66 remains the authority for physical certification.

## Candidate contract

The `candidate` job builds `hachidori.exe` exactly once and uploads the
directory `candidate/` as the artifact `windows-candidate`:

- `hachidori.exe`
- `candidate.json` (`hachidori.windows-e2e.candidate/v1`): source commit,
  file name, SHA-256, size, GOOS/GOARCH, Go version, runner and run identity.

The job also publishes the SHA-256 and source commit as job outputs. Every shard
downloads the artifact and verifies it two ways before running anything: with
`e2e-candidate verify` and, inside the test binary, with `e2e.Main`, which
checks the executable bytes against the manifest **and** against the commit and
SHA-256 the workflow passes outside the artifact
(`HACHIDORI_E2E_SOURCE_SHA`, `HACHIDORI_E2E_CANDIDATE_SHA256`). A mismatch, a
missing manifest or a missing expectation is a failure, not a skip. A shard
never builds `hachidori.exe`.

## Shard contract

One shard is one proof boundary: `bootstrap`, `runtime`, `recovery`,
`single-instance`, `update`, `diagnostics`. Each is

- a matrix entry of the `shard` job (an independent parallel job, so a failing
  shard is diagnosable on its own),
- a Go package `test/windows-e2e/<shard>`,
- an evidence directory `evidence/<shard>/`.

A shard package contains:

- `required.json`: `{"scenarios": ["candidate-identity", ...]}`, the scenario
  IDs (lower-case kebab-case) that must pass for the shard to pass. A required
  scenario that did not run, was skipped or failed fails the shard.
- `e2e_test.go` with `func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.Shard...)) }`
  and tests that start with `s := e2e.Begin(t, "<scenario-id>")`.

`e2e.Main` is the only place that decides certification mode
(`HACHIDORI_WINDOWS_E2E=1`). Outside it (a plain `go test ./...` in normal CI)
the scenarios skip, so normal CI runs only the portable helper tests. In
certification mode it requires Windows, verifies the candidate, runs the tests
and always writes `result.json`, then exits non-zero unless the shard passed.

Scenarios synchronize on observable state (ready/endpoint/pid/persisted-file
conditions with a bounded deadline), never on a fixed sleep. They use
disposable homes and never touch a real Hachidori home.

`candidate-identity` is the shared scenario every shard runs
(`e2e.VerifyCandidateScenario`).

## Evidence

Each shard retains, under `evidence/<shard>/`, and uploads as the artifact
`windows-e2e-evidence-<shard>` also when the shard fails:

- `result.json` (`hachidori.windows-e2e.result/v1`): shard, `PASS`/`FAIL`
  status, `evidence_class` `CI_HOSTED`, `physical_pass` always `false`, the
  candidate identity (commit, file, SHA-256, size), runner and run identity,
  the required scenario IDs, every scenario outcome (`PASS`, `FAIL`, `SKIP`,
  `MISSING`) with a bounded log, problems and attachment names;
- `logs/*`: scenario attachments from `Scenario.Attach`;
- `go-test.log`: the last 2000 lines of the shard's `go test -v` output.

Bounds: 200 log lines of 512 bytes per scenario, 256 KiB per attachment, 4 MiB
of attachments per shard. Attachments and log lines pass through
`internal/redact` (credentials in URLs, token-like values and the home and
profile paths are replaced), and the runner identity is read only from a fixed
allow-list of GitHub Actions variables. Scenarios must not log request,
question or state text, credentials, environment dumps or model contents; this
keeps the diagnostic secrecy contract.

## Helpers

- `e2e`: candidate manifest/verification, evidence recorder, `Main`, `Begin`.
- `cmd/e2e-candidate`: `write` and `verify`, used by the workflow.
- `matrix`: validates that the checklist IDs are unique and every assertion has
  exactly one classification.
- `workflow`: validates the workflow contract (triggers, single build, shard
  consumption of the candidate, pinned actions).

Portable checks:

```sh
go test ./test/windows-e2e/...
```
