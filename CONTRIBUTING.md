# Contributing to Hachidori

## Start with the contract

Read `AGENTS.md`, the organization governance contract, and the applicable purpose-specific Skill before making changes. `docs/architecture.md` defines the accepted product architecture; the governing Issue or latest explicit owner instruction defines the requested work.

Hachidori owns local semantic inference. Do not move orchestration policy, repository governance, workspace authority, or agent lifecycle into this repository.

## Environment

Hachidori is a Go module. Use the Go version declared in `go.mod`.

Fetch the declared dependency graph with standard Go tooling. Do not add dependencies unless the accepted change requires them.

## Isolated work

Do not implement directly on `main`. Use a governed branch/worktree and preserve unrelated work. Do not reset, force-push, merge, close, or alter live repository settings without explicit authority.

## Implementation boundaries

Reuse the existing runtime, API, worker, home, dashboard, desktop, setup, and evaluation primitives before adding new machinery.

`HACHIDORI_HOME` is the single mutable runtime root. Keep transport and UI surfaces thin over the same runtime semantics. The Windows desktop shell must not become a second runtime or policy authority.

## Validation

Run focused tests while editing. Before delivery run:

```sh
gofmt -l .
go vet ./...
go test -race ./...
```

A clean portable test run does not prove Windows WebView2 behavior, GPU inference, model materialization, or end-to-end certification. Use `docs/certification.md` for those evidence levels and report what was actually executed.

## Pull requests

Use the repository's current branch, title, template, and governance contracts. Keep PRs bounded to the accepted change and describe actual validation and limitations.

A PR being open or mergeable is not approval or merge authority.

## Security and disclosure

Follow [SECURITY.md](SECURITY.md). Never publish credentials, private tokens, sensitive runtime data, or unsafe reproduction material in Issues, PRs, logs, or retained evidence.
