# Hachidori

Hachidori is a local semantic inference runtime for fast, low-cost, typed semantic observations.

It keeps the machine-learning runtime resident on an accelerator host, exposes a small transport-neutral HTTP contract, and lets higher-level products decide what to do with the returned observations.

Hachidori does **not** own orchestration policy, repository governance, workspace authority, or agent lifecycle. Its job is narrower: turn text or state into small typed semantic signals quickly and reproducibly.

See [docs/architecture.md](docs/architecture.md) for the target architecture.
