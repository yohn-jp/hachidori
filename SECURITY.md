# Security Policy

## Supported versions

Hachidori is pre-1.0. Security fixes target `main` and the latest 0.x release; there is no long-term-support branch.

## Reporting a vulnerability

Report suspected vulnerabilities privately through [GitHub Security Advisories](../../security/advisories/new), not a public Issue. If that channel is unavailable, open an Issue containing only minimal nonsensitive detail and ask a maintainer to establish a private channel.

Include the affected version or commit, impact, and a minimal safe reproduction. Do not publish credentials, tokens, private data, sensitive model inputs, or host-specific secrets.

The project aims to acknowledge reports within five business days. Response is best-effort for an independently maintained project without a dedicated security team.

## Security boundaries

Hachidori is a local semantic inference runtime. Its HTTP runtime and dashboard are host-local surfaces unless an explicitly governed transport is configured. Keep listen addresses, endpoint validation, runtime materialization, process execution, and filesystem paths within the boundaries defined by `docs/architecture.md` and `docs/runtime.md`.

`HACHIDORI_HOME` is the single mutable runtime root. Do not introduce credential stores, arbitrary host filesystem access, or hidden mutable state outside it.

The Windows desktop shell uses WebView2 as a presentation surface over the host-local dashboard. Preserve its navigation restrictions, permission denial, disabled host objects/web messaging/devtools, and single-instance/lifecycle boundaries.

Model/runtime downloads, subprocess execution, SSH tunnel launching, and endpoint exposure cross trust boundaries. Changes to those paths require focused security review and evidence appropriate to the affected platform.

## Evidence

Portable Go tests, Windows desktop tests, GPU inference, model materialization, and live endpoint behavior prove different boundaries. Follow `docs/certification.md` and report the exact environment and checks actually executed.
