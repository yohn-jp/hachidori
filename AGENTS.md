# Hachidori

Read `.github/agent-governance/AGENTS.md` and the applicable organization Skill before work. The organization execution contract remains in force.

Hachidori is a local semantic inference runtime. It keeps an inference worker resident on the accelerator host and exposes small typed semantic observations to higher-level products.

## Authority

- `docs/architecture.md` is the product architecture authority.
- `docs/runtime.md` defines the runtime, CLI, HTTP API, worker lifecycle, and mutable-home contract.
- `docs/certification.md` defines certification levels and evidence.
- Architecture changes require explicit product-owner/design-review approval.
- Hachidori owns semantic inference, not orchestration policy, repository governance, workspace authority, or agent lifecycle.
- Keep `HACHIDORI_HOME` as the single mutable runtime root. Do not introduce hidden mutable state elsewhere.
- Windows desktop behavior is a host shell over the same runtime/dashboard composition; do not create a second semantic or lifecycle authority in the GUI.

## Validation

Canonical verification:

```sh
gofmt -l .
go vet ./...
go test -race ./...
```

Run focused Go tests while editing, then the canonical verification before delivery. Platform-specific Windows/WebView2 behavior and GPU/runtime certification are separate evidence boundaries; do not claim them from portable unit tests alone.
