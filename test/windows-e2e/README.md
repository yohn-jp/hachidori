# Windows appliance E2E certification

Main-push certification of the Windows `hachidori.exe` on GitHub-hosted
runners. It runs from `.github/workflows/windows-e2e.yml` on `push` to `main`
and `workflow_dispatch`, never on `pull_request`.

```
main push -> build candidate once -> parallel shards -> aggregate certification -> development release (on PASS)
```

A `workflow_dispatch` run certifies and aggregates but never publishes a release.

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
never builds `hachidori.exe`. The build recipe is the one the previous
development release used (`go build -o hachidori.exe ./cmd/hachidori` for
`windows/amd64`; no cgo is linked), preceded by the Windows `go test ./...` that
release gate ran.

## Shard contract

One shard is one proof boundary: `bootstrap`, `runtime`, `recovery`,
`single-instance`, `update`, `diagnostics`. Each is

- a matrix entry of the `shard` job (an independent parallel job, so a failing
  shard is diagnosable on its own),
- a Go package `test/windows-e2e/<shard>`,
- an evidence directory `evidence/<shard>/`.

A shard package contains:

- `required.json`: `{"scenarios": ["candidate-identity", ...]}` lists the
  required scenario IDs (lower-case kebab-case). Its optional `dependencies`
  object maps a scenario ID to required prerequisite IDs; it does not add
  scenarios. A dependent scenario runs only after every prerequisite records
  `PASS`. A prerequisite that failed, was blocked, skipped or is missing makes
  the dependent scenario `BLOCKED` without executing its body.
- `e2e_test.go` with `func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.Shard...)) }`
  and tests that start with `s := e2e.Begin(t, "<scenario-id>")`.

`e2e.Main` is the only place that decides certification mode
(`HACHIDORI_WINDOWS_E2E=1`). Outside it (a plain `go test ./...` in normal CI)
the scenarios skip, so normal CI runs only the portable helper tests. In
certification mode it requires Windows, verifies the candidate, runs every
scenario whose prerequisites passed, and always writes `result.json`. A test
failure does not stop unrelated Go tests; declared dependents are recorded as
`BLOCKED`. A shard-wide preflight failure records required scenarios as
`BLOCKED` (or candidate identity as `FAIL` when candidate verification itself
failed) where the required plan can be read. The shard exits non-zero unless
every required scenario passed.

Scenarios synchronize on observable state (ready/endpoint/pid/persisted-file
conditions with a bounded deadline), never on a fixed sleep. They use
disposable homes and never touch a real Hachidori home.

The `runtime` and `recovery` shards build a fixture runtime from a host Python
3.9 or newer (the workflow resolves it; nothing is downloaded).

`candidate-identity` is the shared scenario every shard runs
(`e2e.VerifyCandidateScenario`).

## Aggregate certification and the release gate

The `aggregate` job runs after the candidate and every shard, also when one of
them failed or was cancelled (`if: always()`), and calls
`e2e-aggregate aggregate`. It fails closed: the verdict is `PASS` only when

- the `candidate` and `shard` jobs both report `success` (a missing or
  cancelled result is a failure),
- the candidate downloaded for the aggregate matches the candidate job's SHA-256
  output and the commit,
- every one of the six shards produced `evidence/windows-e2e-evidence-<shard>/result.json`,
- each result is `PASS`, certifies exactly that commit/file/SHA-256/size, belongs
  to this workflow run, is `CI_HOSTED` with `physical_pass: false`, and lists as
  required exactly the scenarios in the repository's `required.json` (which must
  contain `candidate-identity`),
- every required scenario has a valid `PASS`, `FAIL`, `BLOCKED`, `SKIP`, or
  `MISSING` outcome, and only `PASS` satisfies certification.

It always writes `certification.json` (`hachidori.windows-e2e.certification/v2`:
status, candidate identity, per-shard scenario outcomes, stable outcome counts,
problems, `physical_checks: NOT_CHECKED`) and `certification.md` (appended to
the job summary) and uploads them as the artifact `windows-e2e-certification`,
for passing and failing runs. The summary counts `PASS`, `FAIL`, `BLOCKED`,
`SKIP`, and `MISSING`, and groups every non-PASS scenario by shard.

The `release` job needs `candidate` and `aggregate` with the implicit
`success()` and runs only for a `push` to `main`. It downloads the certified
candidate and `certification.json`, re-verifies the candidate identity,
and runs `e2e-aggregate release-check` (PASS, same commit and SHA-256, all six
shards). It then publishes **the certified candidate bytes** as
`hachidori-windows-amd64.exe` (+ `.sha256`) with the existing development
prerelease tag allocation, and finally downloads the published asset and checks
its SHA-256 against the certified one. Nothing in the release job builds. A
failed or missing required shard fails `aggregate`, which skips `release`: that
commit gets no development release. `release.yml` keeps only the
`pull_request` build/test path.

## Evidence

Each shard retains, under `evidence/<shard>/`, and uploads as the artifact
`windows-e2e-evidence-<shard>` also when the shard fails:

- `result.json` (`hachidori.windows-e2e.result/v2`): shard, `PASS`/`FAIL`
  status, `evidence_class` `CI_HOSTED`, `physical_pass` always `false`, the
  candidate identity (commit, file, SHA-256, size), runner and run identity,
  the required scenario IDs, every scenario outcome (`PASS`, `FAIL`, `BLOCKED`,
  `SKIP`, `MISSING`) with a bounded log, structured bounded prerequisite
  identities/reason codes for `BLOCKED`, problems and attachment names;
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
- `e2e` also holds the aggregate and release-gate logic; `cmd/e2e-aggregate`
  is its CLI.
- `desktopkit`, `runtime/harness`, `update/*` and `diagnostics/*`: shard-owned
  fixtures and inspectors with portable unit tests.
- `matrix`: validates that the checklist IDs are unique, every assertion has
  exactly one classification, every `CI_AUTOMATED` assertion cites scenarios that
  exist in a shard's `required.json`, and every scenario is cited.
- `workflow`: validates the workflow contract (triggers, single build, shard
  consumption of the candidate, aggregate fail-closed wiring, release gating
  and ordering, pinned actions, no `continue-on-error`).

Portable checks:

```sh
go test ./test/windows-e2e/...
```
