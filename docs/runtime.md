# Hachidori runtime (milestone 1 vertical slice)

This document describes what is implemented. `docs/architecture.md` remains the
authority for product boundaries.

```text
client CLI / any HTTP caller
    -> HTTP (127.0.0.1:7843, schema hachidori.v1)
    -> Go runtime (hachidori serve: HTTP + supervisor)
    -> private Python worker (stdin/stdout NDJSON, no network listener)
    -> Laya 0.3.21 (laya.load / Agent.predict_batch)
    -> PyTorch 2.11.0 (+cu128 or +cpu)
    -> GPU / CPU
```

## Commands

| command | plane | purpose |
|---|---|---|
| `hachidori setup [--home H] [--device cuda\|cpu]` | host | materialize pinned Python, packages, worker and model under `HACHIDORI_HOME`, then activate |
| `hachidori serve [--home H] [--listen 127.0.0.1:7843]` | host | run HTTP + one resident worker; non-loopback binds are refused |
| `hachidori dashboard [--home H] [--listen 127.0.0.1:7843] [--addr 127.0.0.1:7844] [--ssh ssh]` | host | `serve` plus the host-local dashboard (see below) |
| `hachidori doctor [--home H]` | host | verify the installation including a real HTTP→worker→model smoke inference |
| `hachidori status [--endpoint URL]` | client | print `/v1/status` |
| `hachidori decide [--endpoint URL] <request.json\|->` | client | send one v1 decide request |
| `hachidori eval [--endpoint URL] [--out report.json] <dataset.jsonl>` | client | caller-side evaluation |
| `hachidori benchmark [--endpoint URL] [--warmup N] [--passes N] [--out report.json] <dataset.jsonl>` | client | eval plus warmup and repeated passes for latency |

`--home` defaults to `HACHIDORI_HOME`; there is no implicit home. `--endpoint`
defaults to `HACHIDORI_ENDPOINT`, then `http://127.0.0.1:7843`.

## HTTP API (`hachidori.v1`)

| endpoint | success | notes |
|---|---|---|
| `GET /health` | `200 {"ready":true,"state":"ready"}` | `503` with the same body while starting, restarting or failed |
| `GET /v1/status` | `200` | runtime, provider (versions, device, GPU name, load/warmup ms), accelerator memory, worker pid/state/starts/restarts, last failure with stderr tail, request/error counters, queue depth, inference p50/p95 |
| `POST /v1/decide` | `200` | one state, 1–32 `choice` questions |
| `POST /v1/decide/batch` | `200` | 1–64 decide requests; requests sharing a question set share forward passes |

Request:

```json
{
  "schema": "hachidori.v1",
  "state": "…",
  "questions": [
    {"id": "scope_expansion", "type": "choice",
     "instructions": "Did the agent modify files outside the requested scope?",
     "choices": ["yes", "no"],
     "descriptions": {"yes": "optional per-choice description"}}
  ],
  "options": {}
}
```

Response (results in question order, question ids preserved):

```json
{
  "schema": "hachidori.v1",
  "results": [
    {"id": "scope_expansion", "type": "choice", "choice": "yes",
     "confidence": 0.672, "probabilities": {"yes": 0.672, "no": 0.328}}
  ],
  "timing": {"inference_ms": 21.4, "total_ms": 21.9}
}
```

`confidence` is Laya's `answer_confidence` (max p, the temperature-calibrated
quantity ECE measures), not its entropy-based `confidence`.

Errors are structured and never look like a semantic answer:

| class | HTTP | meaning |
|---|---|---|
| `request_invalid` | 400 | schema/limit violation, unknown fields |
| `not_ready` | 503 | worker starting, restarting or failed |
| `capacity` | 429 | more than 64 requests queued or in flight |
| `inference_failed` | 500 | the healthy worker failed this request |
| `worker_failure` | 502 | the worker crashed, hung, or violated the protocol |

Only `choice` questions exist in v1. Limits: state 64 KiB, 32 questions,
64 choices, 64 batch entries.

## Worker lifecycle

```text
hachidori serve
  -> resolve state/active-runtime.json, verify worker digest
  -> spawn <home>/runtime/<ver>/python/... -I -X utf8 worker.py (explicit env)
  -> hello -> importing (torch, laya)
  -> device check (cuda requested and unavailable => device_unavailable)
  -> loading: laya.load(<model dir>, device, expected_sha256=<pinned digests>)
     (laya silently falls back to CPU; the worker treats that as device_unavailable)
  -> warming: 3 real predictions (+ cuda synchronize)
  -> ready  => /health 200
  -> serve requests on the same process and model until shutdown
```

- A request never starts Python, imports the ML stack, loads the model or moves it to the device.
- Startup failures (`provider_import`, `device_unavailable`, `model_load`, `warmup`, `startup_timeout`) are deterministic and are not retried; the runtime stays `failed` and reports the class.
- A worker that dies after READY is restarted (≤3 restarts per 10 minutes, 2 s backoff); readiness is false until the new worker has warmed up.
- A request that gets no response within 2 minutes marks the worker unresponsive; it is killed and restarted.
- `SIGINT`/`SIGTERM` → shutdown message, stdin closed, kill after 10 s. The worker also exits on stdin EOF, so it cannot outlive a killed server.

### Private protocol

Newline-delimited JSON. Go owns all stdio streams:

- stdin: requests `{"id":N,"op":"decide","items":[{"state":…,"questions":[…]}]}`, `{"id":N,"op":"stats"}`, `{"op":"shutdown"}`.
- stdout: protocol only. The worker duplicates fd 1 for the protocol and points fd 1 / `sys.stdout` at stderr, because Laya prints warnings to stdout. Any non-JSON line is a protocol violation.
- stderr: logs, appended to `<home>/logs/worker.log`; the last 64 lines are attached to failures.

## Host dashboard

`hachidori dashboard` runs exactly what `serve` runs (same API, same resident
worker) and adds a small server-rendered page on `http://127.0.0.1:7844/`
(`--addr`; non-loopback addresses are refused, and requests whose `Host` is not
loopback are rejected to defeat DNS rebinding).

```text
Development machine                    GPU host
-------------------                    ----------------
dataset / eval / benchmark             hachidori dashboard (127.0.0.1:7844)
        |                                      +-- runtime status / start / stop / restart
127.0.0.1:7843                                 +-- doctor
        ^                                      +-- SSH tunnel launcher
        +========== ssh -R ====================+
                                         Hachidori API 127.0.0.1:7843 -> resident worker
```

It holds no runtime state of its own:

| surface | authority |
|---|---|
| status (page, `GET /api/status`) | the `/v1/status` document (`server.StatusBody`) |
| Start / Stop / Restart | `worker.Lifecycle`, which `serve` also uses; Stop leaves the API bound and reporting not ready |
| Run doctor | `doctor.Run` on the same `HACHIDORI_HOME` (it starts its own temporary worker, as the CLI does) |
| tunnel | `tunnel.Manager`, `GET /api/tunnel` |

Every state-changing action is a same-origin `POST` carrying a per-process form
token; `GET` never changes state. All rendered values go through `html/template`
escaping. The page polls `/live` every 3 s; there is no frontend build.

### SSH reverse-tunnel launcher

Connect runs the host's existing `ssh` client directly (argument vector, no
shell) with a fixed option set:

```text
ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 -R <remote-bind>:<remote-port>:127.0.0.1:<local-port> -- <destination>
```

- Inputs: destination `[user@]host` or an ssh_config `Host` alias (letters,
  digits, `.`, `-`, `_`; no leading `-`, no spaces, ports or URIs), remote bind
  (loopback only: `127.0.0.1`, `::1`, `localhost`), remote port and local
  Hachidori port (1–65535). Anything else is rejected before a process starts.
- SSH authority stays outside Hachidori: keys, agent, identity selection,
  `known_hosts` and `ssh_config` are the host's. `BatchMode=yes` means ssh never
  prompts; authentication or host-key failures exit and are shown with the ssh
  stderr tail. Hachidori never generates, uploads, stores or reads keys or
  passwords and never edits SSH configuration.
- One managed tunnel: Connect with the identical spec while running is a no-op;
  a different spec is refused until Disconnect. Disconnect sends SIGTERM (kill
  on Windows), kills after 3 s, and returns once the child is reaped. An
  unexpected exit becomes state `exited` with the exit status and stderr tail.
  Tunnel and worker failures are separate states.
- Shutdown (Ctrl+C / SIGTERM, or a listener error) terminates the managed ssh
  child before the process exits. A hard kill of `hachidori` itself (task
  manager, SIGKILL) cannot run that cleanup: the `ssh` child then remains and
  must be ended by the operator (its PID is shown on the dashboard).
- Persisted: only the last successful form values in
  `state/dashboard.json` (`destination`, `remote_bind`, `remote_port`,
  `local_port`). Nothing secret exists to persist.
- The caller then uses `HACHIDORI_ENDPOINT=http://<remote-bind>:<remote-port>`
  (default `http://127.0.0.1:7843`) on the SSH destination host and runs
  `status`, `decide`, `eval` or `benchmark` there. Datasets and labels never
  reach the GPU host; the dashboard has no upload surface.

## HACHIDORI_HOME

```text
HACHIDORI_HOME/
  runtime/0.1.0-cu128/         immutable once materialized (0.1.0-cpu for CPU)
    python/                    python-build-standalone CPython 3.12.11 (sha256 pinned)
    worker/hachidori_worker.py (sha256 recorded in manifest)
    requirements.txt
    manifest.json              pins, indexes, pip freeze, worker digest
  packages/                    downloaded runtime archives
  models/convaiinnovations--laya/<revision>/
    model.safetensors …        every file sha256 pinned
    hachidori-model.json
  cache/{huggingface,torch,pip,xdg,nv,tmp,home}
  logs/worker.log, logs/doctor-worker.log
  state/active-runtime.json
  state/dashboard.json         last tunnel form values (non-secret), dashboard only
```

The worker environment is constructed, not inherited: `PYTHONNOUSERSITE=1`,
`python -I` (no user site, no `PYTHON*` variables, no script dir on `sys.path`),
`PATH` = private interpreter directory only (plus `System32` on Windows),
`HOME`/`USERPROFILE`/`APPDATA`/`LOCALAPPDATA`/`TMP`/`TEMP`/`XDG_CACHE_HOME`,
`HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` and `PIP_CACHE_DIR` under `cache/`,
`HF_HUB_OFFLINE=1` while serving. Host variables passed through: `CUDA_VISIBLE_DEVICES`,
proxy/CA settings (setup downloads), OS essentials on Windows, and `LD_LIBRARY_PATH`
on Linux (host GPU driver location, e.g. NixOS `/run/opengl-driver/lib`).

Pins (in `internal/setup/spec.go`): CPython 3.12.11 (python-build-standalone
20250902, linux-amd64 and windows-amd64), `torch==2.11.0+cu128` (NVIDIA driver
≥ 570, RTX 20xx–50xx) or `torch==2.11.0+cpu`, `laya==0.3.21`,
`transformers==5.17.0` and the full resolved package set, model
`convaiinnovations/laya@55cf4c4e…` (Laya's own reviewed revision).

Removing the executable and `HACHIDORI_HOME` removes everything Hachidori owns.

## Evaluation (caller side)

Dataset: JSONL, one case per line; `expected` never leaves the caller.

```json
{"id": "case-001", "state": "…", "questions": [{"id": "scope_expansion", "type": "choice", "instructions": "…", "choices": ["yes", "no"]}], "expected": {"scope_expansion": "yes"}}
```

Reported: choice accuracy, per-question accuracy / mean confidence / ECE,
overall mean confidence, ECE (15 equal-width bins, same binning as
`laya.common.ece_score`), client round-trip p50/p95 and server inference p50/p95,
per-observation results keyed by case id and question id, dataset SHA-256.
Request errors are listed and never scored.

`testdata/eval/contract-example.jsonl` is a three-case format example, **not**
benchmark evidence. The coding-agent benchmark from architecture §12 is not in
this repository and must be supplied as a local JSONL file in this format.
