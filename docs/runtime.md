# Hachidori runtime (milestone 1 vertical slice)

This document describes what is implemented. `docs/architecture.md` remains the
authority for product boundaries.

```text
client CLI / any HTTP caller
    -> HTTP (127.0.0.1:7843, schema hachidori.v1)
    -> Go runtime (hachidori serve: HTTP + supervisor)
    -> private Python worker (stdin/stdout NDJSON, no network listener)
    -> the active model's provider adapter, one at a time:
         Laya 0.3.21 (laya.load / Agent.predict_batch) or
         OpenDecider 0.3.0 (opendecider.load / system_one_batch)
    -> PyTorch 2.11.0 (+cu128 or +cpu)
    -> GPU / CPU
```

## Commands

| command | plane | purpose |
|---|---|---|
| `hachidori setup [--home H] [--device cuda\|cpu] [--model ID]` | host | reconcile `HACHIDORI_HOME` with the Runtime Spec: private uv materializes the locked Python environment, the selected catalog model (default `laya-base`) is materialized separately, then activate |
| `hachidori serve [--home H] [--listen 127.0.0.1:7843] [--resident ID]…` | host | run HTTP + one resident worker (plus one more worker process per `--resident` catalog model ID; see Multi-resident serving); non-loopback binds are refused |
| `hachidori dashboard [--home H] [--listen 127.0.0.1:7843] [--addr 127.0.0.1:7844] [--ssh ssh] [--resident ID]…` | host | `serve` plus the host-local dashboard (see below) |
| `hachidori desktop [--home H] [--listen …] [--addr …] [--ssh ssh] [--background]` | host (Windows) | the same desktop composition as a no-argument `hachidori.exe`: first run/recovery or normal start in a resident WebView2 window with a tray icon (see below); fails with a clear error on other systems |
| `hachidori doctor [--home H]` | host | verify the installation including a real HTTP→worker→model smoke inference |
| `hachidori status [--endpoint URL]` | client | print `/v1/status` (every resident under `residents`) |
| `hachidori decide [--endpoint URL] <request.json\|->` | client | send one v1 decide request |
| `hachidori eval [--endpoint URL] [--questions PATH]... [--out report.json] <dataset.jsonl>` | client | caller-side evaluation |
| `hachidori benchmark [--endpoint URL] [--questions PATH]... [--warmup N] [--passes N] [--out report.json] <dataset.jsonl>` | client | eval plus warmup and repeated passes for latency |
| `hachidori question <definition.json\|dir>...` | client | validate Question Definitions locally; print id, version, digest and the compiled v1 question (no endpoint) |
| `hachidori replay [--questions PATH]... [--dataset D] [--case ID]... [--question ID]... [--print] [--out replay.json] <report.json>` | client | reconstruct and optionally re-send decisions recorded by eval/benchmark |
| `hachidori precision [-model ID] [-out report.json] <baseline-comparison.json> <candidate-comparison.json>` | files only | pair one model's runs at two dtypes from two saved resident comparisons: choice flips, probability and confidence deltas, quality, latency, memory (certification.md) |

`--home` defaults to `HACHIDORI_HOME`; the CLI has no implicit home (the Windows
desktop bootstrap locator below is never consulted by CLI commands). `--endpoint`
defaults to `HACHIDORI_ENDPOINT`, then `http://127.0.0.1:7843`.

## HTTP API (`hachidori.v1`)

| endpoint | success | notes |
|---|---|---|
| `GET /health` | `200 {"ready":true,"state":"ready"}` | `503` with the same body while starting, restarting or failed |
| `GET /v1/status` | `200` | runtime (model ID, device), provider (name, version, the loaded model's ID and revision, device, dtype, GPU name, load/warmup ms), accelerator memory, worker pid/state/starts/restarts, last failure with stderr tail, request/error counters, queue depth, inference p50/p95; with several residents, `residents` lists each one's own such document |
| `POST /v1/decide` | `200` | one state, 1–32 `choice` questions; optional `model` targets one resident (below) |
| `POST /v1/decide/batch` | `200` | 1–64 decide requests; requests sharing a question set share forward passes; optional `model` targets one resident (below) |
| `GET /openapi.json` | `200` | the OpenAPI 3.1 description of this API (below) |

The API is host-local like the dashboard: a request whose `Host` is not a
loopback address is refused with `403` (DNS rebinding), and so is a `POST` that
carries a cross-origin `Origin` or `Sec-Fetch-Site: cross-site` (a web page in
the operator's browser). Non-browser callers, including SSH-forwarded ones that
use `127.0.0.1` or `localhost`, send neither and are unaffected.

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

`confidence` is the probability mass on the reported choice (max p). For Laya it
is `answer_confidence` (the temperature-calibrated quantity ECE measures), not
its entropy-based `confidence`; for OpenDecider-nano it is the softmax
probability of the reported option. The request and result shapes are the same
for every model; which model answered is reported by `/v1/status` (and, for a
directly targeted request, by `served` in the response, below).

### Multi-resident serving and direct selection

`serve --resident ID` (repeatable; also `dashboard`) keeps further catalog
models resident beside the active one, each as its own supervised worker
process with its own PID, lifecycle, failure boundary, counters and
accelerator state (`ResidentSet`; the lifecycle authority). The active model
is the default resident. Every resident uses the active runtime and the
active device: there is no per-resident device and no CUDA to CPU fallback.
A non-default model that is not materialized comes up `failed` with its
cause, without affecting the others.

Routing is explicit and additive. A request with no `model` is answered by the
default resident exactly as before, and its response is unchanged. Omitting
`model` is the only way to ask for the default route: an explicitly empty
`"model": ""` is `request_invalid` (`400`), never an omitted selector. A request
with `model` (a stable Hachidori catalog model ID, for example `laya-base` or
`opendecider-nano`; never a repository or revision, and any other shape,
including a blank or whitespace-only value, is `request_invalid`) is answered by
that resident only:

- the response carries `served: {"model", "provider"}`, the catalog identity of
  the resident that answered (no provider prompt or tokenization detail);
- a model that is not resident is `request_invalid` (`400`);
- a resident that is stopped, starting, restarting or failed is `not_ready`
  (`503`, the message names the model) and counts only against that resident;
- a request is never redirected to another resident, and selecting a target
  never starts, restarts or reloads a worker: alternating between residents
  leaves each one's PID, `starts` and loaded model unchanged;
- a batch is served by one resident: `model` on the batch, or the same `model`
  on its requests; different models in one batch are `request_invalid`, and an
  explicitly empty `model` on the batch or on any request is `request_invalid`
  exactly as for a single request;
- a single-worker `serve` accepts `model` only for the model it runs.

`GET /v1/status` keeps `runtime` and `worker` as the default resident's and adds
`residents`, default first: for each resident its `model`, `provider`,
`default`, `running` (false after an operator stop) and its own `status`
(runtime identity with requested device, worker state, phase, PID, restarts,
provider details with device, dtype and load/warmup ms, accelerator memory,
last failure, request/error counters, queue depth and limit, p50/p95). It is
the same projection the controller and the dashboard consume
(`ResidentSet.Status`); a failed resident does not change another's entry.
`hachidori decide -model ID request.json` sets the target from the command line.
The dashboard's Runtime page lists every resident in its own row (state, PID,
device, load/warmup, counters, queue, latency, GPU memory) and names a failed
non-default resident in the attention list; it restates the status document and
keeps no lifecycle state of its own.

#### Desktop resident models

The Windows desktop needs no `--resident` argument. Settings > Models &
runtimes lists the catalog models with a resident selection: the active model is
always the default resident (not selectable off), and any other materialized
catalog model can be selected as an additional resident (only stable catalog
IDs; no repository or revision input). The selection is stored in
`settings.json` (`resident_models`, additive to schema `hachidori.settings/1`)
and is next-start intent only: saving it downloads, materializes, activates,
starts, stops and restarts nothing, and never changes which model is active.
A normal no-argument launch opens the active model plus the saved selection
through the same `ResidentSet`; with no additional resident the desktop binds the
one worker exactly as before. `serve --resident` and `dashboard --resident` are
unchanged.

A selection that differs from the running residents sets `restart_required`
(and `residency_changed`) in the application snapshot, the Models & runtimes
page shows the restart-required banner and each model's row reads "selected ·
applies on next start" or "resident · removed on restart", and nothing running
is mutated. An explicit Restart (Settings, Runtime or the tray) stops the old
binding and opens the saved set; a Start of a stopped runtime does the same.
The selection is read at every open, and an unreadable `settings.json` fails the
open with its cause instead of starting fewer residents. A selected model that is
not materialized comes up as a failed resident naming its model and provider
(preflight), never READY and never replaced by another model; the other
residents keep serving. Runtime and status then project every resident, with its
own PID and readiness, from the `ResidentSet` status authority; the desktop keeps
no second resident-state model. Physical Windows dual-residency is certified
separately.

Errors are structured and never look like a semantic answer:

| class | HTTP | meaning |
|---|---|---|
| `request_invalid` | 400 | schema/limit violation, unknown fields |
| `not_ready` | 503 | worker starting, restarting or failed |
| `capacity` | 429 | more than 64 requests queued or in flight |
| `inference_failed` | 500 | the healthy worker failed this request |
| `worker_failure` | 502 | the worker crashed, hung, or violated the protocol |

The class is the stable part of an error. For `inference_failed` and
`worker_failure` the message is the worker's own text, passed through the one
redaction policy (`internal/redact`, also used by the diagnostic bundle): the
`HACHIDORI_HOME` and user-profile paths become `<HACHIDORI_HOME>` and
`<USERPROFILE>`, credentials in URLs, token-like `key=value` pairs and
bearer/cookie values become `<redacted>`, and the text is at most 1 KiB. The
operator reads the full text in the worker log and the dashboard.

Only `choice` questions exist in v1. Limits: state 64 KiB, 32 questions,
64 choices, 64 batch entries.

### OpenAPI description

`GET /openapi.json` serves the machine-readable contract of the endpoints above
as OpenAPI 3.1, from the binary and without network access. It is the one
public description of the API (there is no separate Hachidori schema endpoint).
`info.version` is the contract identifier `hachidori.v1`. The document covers
request and response bodies, the `choice` question rules, limits, structured
errors and their statuses, health and status. It is independent of the active
model and provider: those are reported by `GET /v1/status`. It is built in
`internal/server/openapi.go` from the limits in `internal/api` and the error
status map the handlers use, and `internal/server/openapi_test.go` checks it
against the Go types and the handlers, so a change to either that is not
reflected in the document fails the tests. Some rules cannot be expressed in the
schema and are stated in its descriptions only: byte (not character) limits on
`state` and `instructions`, blank-text rejection, unique question ids, and
`descriptions` keys naming existing choices. Like the rest of the API it is
subject to the host-local checks above.

## Worker lifecycle

```text
hachidori serve
  -> resolve state/active-runtime.json, verify worker digest
  -> preflight: the runtime's worker script must be the one this build embeds
     (older runtime => `preflight` failure, nothing is spawned)
  -> spawn <home>/runtime/<ver>/python/... -I -X utf8 worker.py (explicit env)
  -> hello -> importing (torch, the provider of the active model: laya | opendecider)
  -> device check (cuda requested and unavailable => device_unavailable)
  -> loading, per provider:
       laya:        laya.load(<model dir>, device, expected_sha256=<pinned digests>)
       opendecider: verify every pinned file digest, then opendecider.load(<model dir>, device, dtype=float32|bfloat16)
                    (a local directory; nothing is resolved from the Hub); the encoder must then
                    be in exactly the requested dtype, else `model_load`
  -> the worker checks where the model actually is: anything but the requested
     device is device_unavailable (Laya silently falls back to CPU; there is no
     CUDA -> CPU fallback for any provider)
  -> warming: 3 real predictions (+ cuda synchronize)
  -> ready  => /health 200
  -> serve requests on the same process and model until shutdown
```

- A request never starts Python, imports the ML stack, loads the model or moves it to the device.
- Startup failures (`preflight`, `worker_startup`, `provider_import`, `device_unavailable`, `model_load`, `warmup`, `startup_timeout`) are deterministic and are not retried; the runtime stays `failed` and reports the class.
- A worker that dies after READY is restarted (≤3 restarts per 10 minutes, 2 s backoff); readiness is false until the new worker has warmed up.
- A request that gets no response within 2 minutes marks the worker unresponsive; it is killed and restarted.
- A protocol line the runtime cannot read (over 16 MiB) is a `protocol` failure at once: the worker is killed and the failure is reported as such, not as an unresponsive worker.
- `/v1/status` never waits for an inference. While one is in flight the accelerator memory is the last value taken while the worker was idle and the snapshot carries `accelerator_stale: true` (the dashboard labels it "last known"); worker state, readiness and counters are always current.
- `SIGINT`/`SIGTERM` → shutdown message, stdin closed, kill after 10 s. The worker also exits on stdin EOF, so it cannot outlive a killed server.

### Private protocol

Newline-delimited JSON. Go owns all stdio streams:

- stdin: requests `{"id":N,"op":"decide","items":[{"state":…,"questions":[…]}]}`, `{"id":N,"op":"stats"}`, `{"op":"shutdown"}`.
- stdout: protocol only. The worker duplicates fd 1 for the protocol and points fd 1 / `sys.stdout` at stderr, because Laya prints warnings to stdout. Any non-JSON line is a protocol violation.
- The worker is started with `--provider <laya|opendecider>`, taken from the active catalog model (never from operator input). The protocol above it is identical for both providers; tokenization, the OpenDecider `[MASK]` option markers and result translation stay inside the worker's provider adapters.

### Decision models

Both supported models answer the same typed `choice` contract by closed-option
scoring: each question's choices are scored together and a probability
distribution over exactly those choices comes back. Nothing is generated,
parsed or retried.

- **Laya** (`laya-base`, the default): ModernBERT-large, scored by `laya`.
- **OpenDecider-nano** (`opendecider-nano`, a candidate): an Ettin-encoder-400m
  backbone with an MLP decision head. The `opendecider` package places one
  `[MASK]` marker before each option in a single sequence (`question`, options,
  `input: <state>`), reads the encoder state at each marker through the head to
  one logit per option and softmaxes over the question's options: all options of
  a question cost one encoder pass. The adapter maps Hachidori's choice question
  (instructions, choices, optional descriptions) onto that input and the
  probabilities back onto the v1 result. It runs in `float32` (what upstream
  evaluated) unless `HACHIDORI_OPENDECIDER_DTYPE=bfloat16` is set for the launch
  (an evaluation control, see certification.md; the dtype actually in use is
  `provider.dtype` in `/v1/status`, and a dtype the package did not honour fails
  the load instead of falling back) and scores at most 16 sequences (states × questions) per padded
  batch.
  Upstream facts that apply to the candidate and are not Hachidori certification
  evidence: a 2,048-token context shared by question, options and state (the
  state is truncated first so every option survives; Hachidori logs each
  truncation in the worker log and does not change the result shape), English
  evaluation only, roughly 2 GiB of runtime memory.

Which model is the default is decided by recorded evidence, not by size or
upstream claims; see "Decision-model comparison" in `certification.md`.
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
| Experiment Runner (`/experiments`) | `internal/eval` (`question.Load`, `eval.Load`, `eval.RunEvidence`) run caller-side in the dashboard process against the same inference API address |
| Evidence / Error Explorer (`/errors`) | `internal/eval/explore`, a pure read-only consumer of `eval.Report` (`hachidori.evidence.v1`); no endpoint access |
| Question Workbench (`/workbench`) | a caller of the existing `POST /v1/decide` on the dashboard's inference API address; `internal/question` / `internal/api` validation and compilation |

The page is one workstation shell: a persistent navigation for **Runtime**
(`/`, readiness, model/accelerator identity and lifecycle actions),
**Workbench** (`/workbench`), **Experiments** (`/experiments`), **Evidence**
(`/errors`, the Error Explorer) and, separated from them, **Diagnostics**
(`/diagnostics`: doctor, the last worker failure, the SSH tunnel launcher and
the Desktop panel). A compact readiness indicator and the runtime identity
(provider, model, device, inference API address) are restated from the
`/v1/status` document on every workspace. Lifecycle actions return to Runtime;
doctor, tunnel and desktop actions return to Diagnostics. `/#diagnostics` (the
tray's attention target) keeps an anchor on Runtime and forwards to
`/diagnostics`.

Every state-changing action is a same-origin `POST` carrying a per-process form
token; `GET` never changes state. All rendered values go through `html/template`
escaping. Every workspace polls `/live` every 3 s for the shell status (Runtime
and Diagnostics also for their detail); there is no frontend build.

### Question Workbench

`/workbench` is an interactive caller surface for one bounded state and one or
more editable v1 choice questions (id, instructions, choices, optional choice
descriptions). **Run** compiles every question with `internal/question`,
validates the request with the v1 contract and sends it as one
`POST /v1/decide` to the resident runtime; an invalid request shows the v1
validation error and is not sent. The page shows each result's choice,
confidence and per-choice probabilities, and the exact request JSON body.
There are no expected labels and no scoring here.

The workbench keeps no state: the editor travels in the page's form (bounded
by the v1 limits) and nothing is stored. Question Definition files are read or
written only at an absolute local path the operator types, one file per
action: **Load** validates one `hachidori.question.v1` file with the same rules
as `hachidori question` and projects it into the editor (showing whether the
edited question still matches the loaded identity); **Export** writes one
edited question as a new `hachidori.question.v1` file (id, explicit version,
content) and reports its identity and digest. Exports never replace an existing
file. The workbench POST has a larger body limit than the other dashboard
actions; its token, same-origin and loopback checks are the same.

### Experiment Runner

`/experiments` wraps the caller-side evaluation authority for an operator on
the same machine. The operator enters an absolute local JSONL dataset path,
optional Question Definition files/directories, warmup and passes
(`eval.Options`). **Preflight** resolves definitions (`question.Load`) and
loads and validates the whole dataset (`eval.Load`) without inference. **Run**
repeats that preflight, checks `/health`, and runs `eval.RunEvidence` in the
background against the dashboard's inference API address, exactly as
`hachidori benchmark` does: expected labels stay in the dashboard process and
only `Case.Request()` projections are sent. The page shows request progress,
a terminal state (`succeeded`; `failed` for request errors, an inconsistent
served identity, or no served identity; `aborted`), the report's summary and
per-question metrics, dataset SHA-256, definition identities and served
identity.

One experiment runs at a time per dashboard; a second run is refused. The
report is kept in memory only until the next run; **Export** writes it as
canonical `hachidori.evidence.v1` JSON (the `eval -out` encoding) to a new file
at an absolute path, never overwriting, and it can be replayed with
`hachidori replay`. Dashboard shutdown, and the desktop replacing or closing
its runtime, abort a running experiment; an aborted run keeps no report.

### Evidence (Error Explorer)

The Evidence workspace (`/errors`) analyses one `hachidori.evidence.v1` report: the current
experiment's report, or a report file opened by absolute path. Opening never
runs inference. Reports are decoded strictly (`internal/eval/explore`): another
schema, unknown fields, trailing data, or an internally inconsistent report
(observation count, `correct` versus choice/expected, confidence outside
[0, 1], missing probabilities, per-question counts) is rejected and the
previously open report stays.

All counts and selections are deterministic functions of the report:
correct, wrong, request errors, high-confidence wrong (wrong with confidence
>= the threshold), per-question correct/wrong/high-confidence-wrong/missing
results beside the report's own accuracy, mean confidence and ECE, and
request errors by class. Observations filter by question, expected choice,
predicted choice, outcome and confidence range, request errors by class, and
sort by confidence, request latency or inference latency (ties and missing
values in report order). The detail view shows case, question id and wire
digest, definition identity, expected and predicted choice, confidence,
probabilities, latencies and served identity.

The high-confidence threshold (initially 0.9) is an analysis control only;
Hachidori makes no pass/fail or fitness judgment. **Export selection** writes a
separate `hachidori.evidence-analysis.v1` document (source report digest and
identity, threshold, filter, counts, and the selected observations and request
errors copied verbatim) to a new file; it never modifies or replaces the
source report.

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

## Windows desktop shell

The Windows desktop is one composition, `desktopApp` (`cmd/hachidori/desktopapp.go`),
entered two ways: `hachidori.exe` with no arguments (the product path) and
`hachidori desktop [--home H] [--listen …] [--addr …] [--ssh …] [--background]`
(the same composition with explicit flags; start at sign-in uses it). There is
no second lifecycle composition: both entries run the single-instance guard,
home discovery, first-run/recovery or normal start, and the resident tray
window described below. It serves the API and the dashboard handler
(Host/Origin/form-token checks) and one native top-level window that shows
`http://<--addr>/` through Microsoft WebView2. The window has no runtime,
lifecycle, status or doctor logic of its own; it owns only its own lifetime.
The runtime is owned by the application controller (`internal/app`, see
below): setup, start, restart and quit all act on the one `app.Controller`,
which binds exactly one supervised worker, and the tray, dashboard and API all
observe that one supervisor. The home is `--home`, then `HACHIDORI_HOME`, then
the bootstrap locator; with none of them the desktop enters first-run setup.

Order of a launch:

1. the loopback listen addresses are checked;
2. the installed WebView2 Runtime is detected (registry and client DLL lookup).
   If it is missing the launch fails before anything starts, with a
   diagnostic that names the Evergreen WebView2 Runtime download page.
   Hachidori never downloads or installs WebView2;
3. the per-user single-instance guard is taken: a named mutex
   `Local\Hachidori.Desktop.<digest of the user SID>` (session-local
   namespace, no elevation). A second `desktop` launch by the same user
   activates the running instance instead: it posts a per-user registered
   window message to the running shell's window (found by a per-user window
   class, hidden windows included), which shows and focuses it, and exits 0
   before starting any runtime component. If that window cannot be found
   within ~10 s the second launch fails with "already running". The kernel
   drops the mutex when the process ends, so a crash never blocks a relaunch;
4. the home is discovered and the first-run/recovery or normal start plan is
   decided (see first run below); the controller starts a configured, installed
   home;
5. the API and dashboard listeners are bound before the window is created, then
   the window opens on the dashboard (or the first-run screen).

Window security:

- Top-level and frame navigations are allowed only to the exact dashboard
  origin (`http://127.0.0.1:7844` by default); everything else, including the
  inference API port, `localhost` aliases, `https:`, `file:`, `data:`,
  `javascript:` and `about:` URLs, is cancelled and logged. Popups /
  `target=_blank` / `window.open` never create a window and are not handed to
  another application (`internal/desktop` `Policy`).
- WebView2 web messaging and host objects are disabled, so there is no
  JS-to-Go bridge at all; DevTools, the default context menu and the status
  bar are off; every permission request (camera, clipboard, notifications, …)
  is denied.
- The page is the same server-rendered dashboard; its loopback `Host` check,
  same-origin check and per-process form token are unchanged.

Close behaviour: closing the window hides it to the tray (next section); the
runtime keeps running. **Quit Hachidori** (tray menu) ends the desktop session:
the WebView2 controller is closed (its browser processes exit), the tray icon
is removed, the window is destroyed, and then the process shuts down exactly as
`dashboard` does on Ctrl+C: the dashboard and API stop listening, the worker is
stopped through the controller, and a managed `ssh` child is terminated.
Ctrl+C in the console quits the same way.

### Tray and start at sign-in

While the desktop runs (either entry, including during first run and recovery) it shows one per-user notification-area icon
(`internal/desktop`, `tray_windows.go`). Its tooltip and the first menu line
show one concise state, derived only from `app.Controller.Snapshot()`: **Ready**,
**Starting** (starting/warming), **Setting up**, **Stopping**, **Stopped**, or
**Needs attention** (failed, no home, not installed). The tray has no status
calculation of its own. Menu: state line, **Open Hachidori**, **Restart
Runtime** (the controller's `Restart`; disabled when nothing is installed),
**Diagnostics**, the two preferences below, **Quit Hachidori**.

- Closing the window hides it to the tray and leaves the worker untouched. The
  first time, a tray notice says Hachidori keeps running; the dashboard's
  Desktop panel repeats this. If the tray icon could not be created (for
  example explorer is not up yet at sign-in) closing minimizes to the taskbar
  instead, and the icon is retried (`TaskbarCreated` and a periodic refresh).
- Open (or a tray click, or a second launch) shows the one existing window. When
  the state is Needs attention it opens on the dashboard's `#diagnostics`
  section, so a failed background start is visible. A worker that fails to reach
  READY is shown as Needs attention with a one-time notice; the application never
  restarts itself in a loop (worker restarts stay with the supervisor's policy;
  only the user's Restart Runtime asks for another one).
- **Start Hachidori when I sign in** (opt-in, off by default; tray menu and the
  dashboard's Desktop panel) writes one value to the current user's Run key:
  `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` value `Hachidori` =
  `"<exe>" desktop --background [--home "<home>"]`. It needs no elevation, is
  idempotent (enabling twice writes one identical value; a moved executable
  rewrites it), is removed by unchecking it (disabling an absent value
  succeeds), starts this same executable, and creates no Windows Service,
  scheduled task or machine-wide entry. `--home` is embedded unless the home
  came from the bootstrap locator. Windows' own Startup Apps switch can still
  disable a Run entry; Hachidori does not read or write that state.
- **Start minimized** (opt-in, stored in `%LOCALAPPDATA%\Hachidori\desktop.json`
  beside the bootstrap locator) applies to `--background` launches only: the
  window starts hidden in the tray (only if the tray icon exists) while the
  runtime starts as usual. It applies only to a configured, installed home
  whose runtime could be bound: first run, recovery and a failed start always
  show the window (on `#diagnostics` for a failed start). An explicit launch
  always shows the window.
- A launch that cannot even reach the window (for example a missing WebView2
  Runtime or a busy listen address) reports the error, and a `--background`
  launch shows it in an error box; it exits 1 and is not retried.

Distribution: the binding is `github.com/wailsapp/go-webview2` with its pure-Go
loader (default build, no cgo). No `WebView2Loader.dll` is embedded, written
to disk or searched for; the WebView2 Runtime installed on Windows is the only
prerequisite. The UI stays the dashboard's embedded `html/template` page; there
is no frontend bundle. WebView2's user data folder (its browser profile) is
`HACHIDORI_HOME/cache/webview2`, so nothing is written to `%APPDATA%`. If the
binding hits a fatal WebView2 error after startup it prints the error and
exits the process immediately; deferred cleanup is then skipped as with a
hard kill (the worker still exits on stdin EOF).

`hachidori.exe` remains a console-subsystem program so explicit CLI commands keep normal terminal behavior. On the no-argument product path, Hachidori hides the console only when it is the sole process attached to that console (the normal Explorer/double-click case). A launch from an existing terminal never hides the caller's console.

## Windows first run and no-argument launch

On Windows, `hachidori.exe` with no arguments is the desktop product entry point
(`cmd/hachidori/launch_windows.go`; other platforms print the CLI usage exactly
as before, and every explicit subcommand keeps its CLI behavior). It uses the
same WebView2 preflight, single-instance guard, window policy and
`worker`/`server`/`dashboard` authorities as `hachidori desktop` (they are the
same composition, including the resident tray lifecycle above), plus:

1. `home.Discover("")` (explicit env `HACHIDORI_HOME` > bootstrap locator >
   unconfigured) and `firstrun.Decide`:
   - no locator: first-run wizard, no runtime, no home, nothing written;
   - locator naming an installed home: normal startup (`app.Controller.Start`),
     the window shows the dashboard;
   - locator naming a home without a valid activation record: the wizard with
     that home pre-selected;
   - locator naming a missing/unavailable home, or a malformed locator: a
     recovery screen that explains what was found and offers the same folder
     choice. Recovery never installs, recreates or forgets anything by itself.
2. The window origin is the loopback dashboard address for the whole session.
   Under `/wizard/` it serves the embedded first-run page (no remote content,
   Host/Origin/token checks like the dashboard); everything else is the
   dashboard once the runtime is READY (or, on a normal launch, bound).
3. The user chooses one storage root with **Browse...**, which runs the native
   Windows folder dialog (`IFileOpenDialog`, folders only) on the server side
   and returns only a path; the page cannot supply a path. `firstrun.Validate`
   then checks it: absolute, not a file, writable (probed with a temporary file
   that is removed; nothing is created yet), existing Hachidori home detection,
   and free space only when Windows reports it (no required-space estimate is
   invented). A non-empty folder that is not a Hachidori home is not mixed with
   Hachidori's directories: `<chosen>\Hachidori` is used and shown.
4. A valid existing installation is recognised and offered as **Use this
   installation**; setup never runs over it.
5. **Install** passes the explicit device (`cuda` or `cpu`, never changed by
   Hachidori) and the default catalog model to `app.Controller.Setup`. Progress
   is the controller's real setup phases (`preparing`, `runtime`, `model`,
   `activation`) and, inside the current phase, the step it is busy with
   (see "Operation progress" below). Setup's log is kept in
   `HOME/logs/setup.log`.
6. The bootstrap locator is written (`home.Remember`) only after setup has
   succeeded, immediately before the runtime is started. A failed or interrupted
   setup therefore never leaves a locator claiming a finished installation, and
   a relaunch returns to the screen it started from. If the locator cannot be
   written, the runtime is not started and the error is shown with Retry.
7. Start, warmup and READY come from the controller and the supervisor's
   `/v1/status`; READY shows the model, device and runtime identity of that
   document, then the window continues to the dashboard.
8. Failures keep the controller's structured failure (source, class, message,
   stderr tail). **Retry** repeats setup (one operation at a time) or the
   runtime start; **Change storage location** returns to selection. Both are
   available only before a successful activation: once a home has a valid
   activation record the wizard neither offers nor accepts another location.

WebView2's profile folder is `HOME/cache/webview2` when a home is selected at
launch. Before any home exists (first run, recovery) it is a temporary folder
that is removed when the window closes, so nothing Hachidori-owned is left
outside the selected home except the bootstrap locator.

Closing the main window follows the resident desktop rule above: it hides to
the tray and leaves the controller/runtime running. **Quit Hachidori** performs
the bounded controller shutdown. An in-progress setup is not interruptible;
Quit waits for it up to the shutdown bound, and setup's atomic publish means an
interrupted setup never activates a partial runtime.

The executable remains console-subsystem for CLI compatibility. A no-argument
Explorer/double-click launch hides only a console owned solely by Hachidori; a
launch from an existing terminal leaves that terminal visible. Fatal startup
errors (for example a missing WebView2 Runtime) are shown in a native message
box even when the owned console has been hidden.

## Application lifecycle (`internal/app`)

`internal/app` is the platform-neutral application controller used by desktop
surfaces. It projects one user-facing state over the existing setup,
`worker.Lifecycle` / supervisor, active-runtime, and `server.StatusBody`
authorities; it does not create a second runtime state model.

States are `unconfigured`, `not_installed`, `installing`, `installed`,
`starting`, `warming`, `ready`, `stopping`, and `failed`. Setup and the
maintenance actions report only real phases entered, and a percentage only for
a step that has a measurable total.

### Operation progress

Every setup and maintenance action (`setup`, `materialize`, `repair`,
`activate`, `verify`, `remove`) is accepted by the controller at once and runs
in the background (never on the caller's or the UI's thread); only the
rejection rules (one action at a time, a running worker, an unreadable home)
are answered synchronously. The action's `Operation` carries:

- `plan`: the phases it goes through, and `phase`/`phases`: the ones it has
  entered. Setup is `preparing`, `runtime`, `model`, `activation`; Materialize
  and Repair end in `publish` instead; Activate is `runtime`, `model`,
  `activation`; Verify and Remove are the phase of their artifact. A phase is
  reported only when it is really entered.
- `progress`: the step inside the current phase, from `internal/setup`
  (`setup.Progress`, reported through `setup.Observer`): `downloading` (bytes
  of the response, `Content-Length` as the total when the server states one),
  `verifying` (bytes hashed of each file, whose size is known),
  `materializing` (the private uv's steps: installing Python, creating the
  environment, installing the locked packages), `publishing`, `activating`,
  `removing`. A step without a measurable total has no `total`; it is shown as
  in progress, and no percentage is ever derived from it. Model downloads are
  per file ("file 3 of 7"); the catalog does not pin sizes, so there is no
  invented overall total.
- `failure`: `source`, the `phase` and `step` it failed in, and the cause. A
  failed maintenance action changes neither the application state nor the
  activation record.

The outcome of an explicit Verify is kept per artifact by the controller
(`checks`). The output of an action goes to `HOME/logs/setup.log`, headed by the
action.

For the worker, the same vocabulary is the supervisor's own phase:
`preflight` and `spawning` are the supervisor's, `importing`, `loading` and
`warming` are reported by the worker process itself, then `ready`. None has a
total, so the UI shows the current one as a step with an indeterminate bar. A worker failure carries the phase it had reached, its class,
and the tail of its stderr. Everything the worker wrote before it exited is
applied before the exit is judged, so a reported phase or fatal class is never
replaced by a generic startup failure.

### Worker launch contract

The worker script is half of the contract with the private Python process (its
command line and protocol); the other half is the executable. An immutable
runtime keeps the script it was materialized with, so a runtime from an older
build can be internally consistent (identity, manifest and digests agree) and
still not understand the arguments this build passes. `setup.CheckWorkerContract`
therefore requires the activated runtime to carry the worker script this build
embeds. The supervisor runs that check (`worker.Config.Preflight`) before every
launch; a runtime that fails it is not started: the supervisor is `failed` with
class `preflight`, phase `preflight` and the cause and the recovery as the
message (Materialize the current runtime and Activate it, or
`hachidori setup --device <device>`; installed models are reused). The binding
and the dashboard still come up, so Settings is available for that recovery.
`doctor` fails its runtime check the same way, and `Inventory.ActiveProblem`
reports it for the Settings manager and the Runtime page.

The controller serializes setup/start/stop/restart actions so UI retries cannot
create duplicate runtime ownership. Setup accepts the same device and catalog
model selection as the CLI. Home discovery remains outside the controller.

## Runtime materialization

`hachidori setup` is declarative. The desired runtime is a **Runtime Spec**:
Go data in `internal/setup/spec.go` plus the uv project embedded from
`internal/setup/runtimespec/` (`pyproject.toml` and the locked `uv.lock`,
with one uv extra per PyTorch flavor: `cpu`, `cu128`). The spec records the
schema, platform, CPython version, the model providers (`laya==0.3.21` and
`opendecider==0.3.0`, one runtime carries both), exact torch build, flavor,
pinned uv version and executable digest, the SHA-256 of both uv project files
and of the worker script. The model is not part of it: selecting any catalog
checkpoint reuses the same runtime. A runtime materialized before OpenDecider
was carried has a different identity and a different worker script, so this
build cannot start it (see "Worker launch contract"): after an upgrade the
current runtime has to be materialized first (`hachidori setup`, or Materialize
and Activate in the manager), which reuses already verified models.

The runtime identity is `<flavor>-<first 16 hex of sha256(canonical spec JSON)>`,
for example `cu128-…` / `cpu-…`. Any semantic change (lock, versions, worker,
uv, platform, flavor) yields a new identity and a new directory; an existing
runtime is never modified.

```text
desired Runtime Spec (device -> flavor)
  -> private uv: tools/uv/<version>/uv[.exe]; archive and executable sha256 pinned,
     executable re-verified before every invocation, invoked by absolute path
  -> identity
  -> runtime/<identity>/ present?  yes: verify (manifest identity, worker digest, interpreter probe) and reuse
                                   no:  runtime/.staging-<identity>/ :
                                        uv python install <python> --no-bin --no-registry
                                        uv venv --relocatable --python <python> env
                                        uv sync --locked --no-build --no-install-project --extra <flavor>
                                        write worker, verify, write manifest.json last
  -> model (catalog entry selected by --model): reuse if every file matches its pinned digest,
     else download to staging, verify, rename (a download that receives no data
     for 2 minutes, headers included, is abandoned and leaves nothing behind; the
     bound is on progress, never on total time, and the digest stays the authority)
  -> rename staging -> runtime/<identity>, verify again
  -> write state/active-runtime.json
```

uv owns CPython acquisition (into `tools/uv/<version>/python/`, the uv-pinned
python-build-standalone build) and package acquisition from the lock; Hachidori
does not resolve dependencies, write requirements files or run pip. uv runs with
a constructed environment: `UV_NO_CONFIG=1`, `UV_MANAGED_PYTHON=1`, cache and
Python install dirs under `HACHIDORI_HOME`, `PATH` = its own directory; user
uv/pip configuration, indexes and interpreters are ignored. A uv from `PATH`,
system Python or system pip is never used; a failed uv bootstrap fails setup.

Verification is Hachidori's: the private interpreter (`-I`, offline, constructed
env) must report exactly the spec's Python version, `laya==0.3.21`,
`opendecider==0.3.0` and `torch==2.11.0+<flavor>`, its prefix must be the runtime's `env/` and its base
interpreter the uv-managed CPython under `tools/`. Failure or interruption at any
step leaves `state/active-runtime.json` unchanged; staging is never treated as a
runtime and is recreated on the next run. There is no retry loop beyond uv's
own. A runtime directory that exists but does not verify is an explicit error
(remove it to rematerialize). Runtimes created by the former pip-based setup
(`runtime/0.1.0-*`) never match an identity and are never completed or
modified in place. A runtime is valid as the active runtime only if its manifest
identity is the activation record's runtime and the one its own Runtime Spec
derives (`home.Home.LoadActive`); activation, serve, the desktop and `doctor`
all apply that one rule, so an active legacy runtime is `runtime_invalid`
everywhere and is never READY. `hachidori setup` (or the desktop install) is the
repair: it materializes the current runtime next to the legacy one, reuses the
verified model without downloading it, and switches `state/active-runtime.json`
only after both verified, for the requested device.

### Model catalog

Models are immutable catalog identities declared in `internal/setup/spec.go`
(`setup.Models`): a stable Hachidori model ID, provider kind (`laya` or
`opendecider`), upstream repository, immutable commit revision and every
required file with its SHA-256.

| ID | provider | checkpoint |
|---|---|---|
| `laya-base` (default) | `laya` | `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, the upstream English ModernBERT-large checkpoint |
| `opendecider-nano` (candidate) | `opendecider` | `manjunathshiva/opendecider-nano@7e42a1508d2beef44717d044831e87f2fc4db9f2`, Apache-2.0, Ettin-encoder-400m + MLP head (7 files, about 0.8 GB) |

`hachidori setup --model <id>` selects one; omitting `--model` selects
`laya-base`. Only catalog IDs are accepted (an unknown ID fails before anything
is changed); there is no repository or revision input. Adding a checkpoint
for an existing provider means declaring another entry, not changing the worker. The model directory `models/<owner>--<repo>/<revision>/` and
its `hachidori-model.json` are derived from the entry. `state/active-runtime.json`
records the runtime identity, the model ID (`model_id`), the model directory and
the device. A failed materialization of a newly selected model leaves the active
runtime and model unchanged. Changing the active model takes effect when the
host is restarted; a READY worker never switches models. By default exactly
one model is resident; `--resident` keeps further catalog models resident, each
in its own worker, and a request may target one of them directly (see
Multi-resident serving). There is no automatic routing or handoff.

`serve` resolves only the activated model: the activation's model ID must be a
catalog entry, the directory must be the one derived from it, the materialized
manifest must equal the entry and the runtime must carry the entry's provider;
the worker then verifies every file digest before loading. Nothing is
downloaded. `/v1/status` (and the dashboard) report `runtime.model_id` and
`runtime.model`, and, from the worker itself, the provider, its version, the
loaded model's ID and revision, and the device and dtype it is actually on; `doctor` verifies the selected
entry's files and prints its ID, repository and revision. Activation records
from before model selection (no `model_id`) are resolved by their directory.

Network is needed only while materializing. `serve` and `doctor` use the
published runtime's absolute interpreter and do not need uv, its cache or
package indexes.

Updating the environment: edit `runtimespec/pyproject.toml`, run
`uv lock --directory internal/setup/runtimespec` with the pinned uv version,
commit both files. Setup runs `uv sync --locked`, which refuses a stale lock.

### Models and runtimes manager

The Windows desktop's Settings workspace has a Models & runtimes section over
the model catalog (Laya and OpenDecider-nano). It is the one place a model is
chosen, materialized and activated, and a view over typed operations
of `internal/setup` (`Inspect`, `Materialize`, `Verify`, `Repair`, `Activate`,
`Remove`), reached through `app.Controller`; the dashboard never inspects or
deletes directories. `serve` and the browser `dashboard` do not offer it, and
`hachidori setup` is unchanged.

- Inventory lists only catalog-pinned identities: one runtime per device
  (`cuda`, `cpu`) and every catalog model, each with its immutable facts
  (runtime identity, platform, Python, provider, torch; model repository,
  revision, pinned file count) and whether it is supported, materialized,
  verified and active. Listing is read-only and offline; full verification
  (interpreter probe, file digests) runs only on an explicit Verify.
- Three facts are shown separately and never conflated: the model chosen in the
  form (choosing changes nothing: no download, activation or restart), the
  **active** model (the activation record, what the next start serves) and the
  model **running** now (what the resident worker was started with, from the
  status authority). After an activation under a running worker the active model
  reads "applies on restart" and the restart-required banner offers the explicit
  Restart.
- Every action is accepted at once and runs in the background; the manager
  shows the phases of its plan, the current step and, where the step has a
  total, bytes and a percentage (an indeterminate bar otherwise), then keeps
  the outcome on screen (DONE, or FAILED with the phase, the step, the cause,
  the active-state note and a Diagnostics link) until the next action. The
  Runtime page shows the same panel, the worker's startup phases, and a failed
  worker's phase, class, hint and stderr tail.
- With the application hosted, the Runtime page's Start, Stop and Restart are
  the application's (`app.Controller`), so a restart after an activation or a
  failure binds the active runtime and model, never the runtime the page was
  created for. "Model" on the Runtime page is what that runtime is configured
  for ("serving now" only when READY); "Next start" is the activation record.
- Materialize and Repair use the same staged, verified path as setup and never
  write `state/active-runtime.json`; a failure leaves the active state as it
  was. Materialize may run beside a running worker; Repair is refused while
  the worker runs. Repair moves an artifact that fails verification aside,
  rebuilds it, and restores it if the rebuild fails.
- Activate is explicit and offline: both artifacts must already be
  materialized and verify, and only then is the activation record replaced. It
  never restarts the worker. With a running worker the application reports
  restart required (`restart_required` in the snapshot); Restart, an
  `app.Controller` action, then stops the old binding and binds the new
  activation. A requested device is never replaced by another: activating
  `cuda` without a materialized CUDA runtime fails.
- Remove deletes one materialized, unused catalog artifact. The active
  runtime/model, non-catalog names, an unreadable activation record, a
  restart-required state, and any path that does not resolve to a real
  directory beneath `HACHIDORI_HOME` are refused.
- Serving performs no network resolution; only Materialize and Repair use the
  network.

### Updates (Windows executable)

Settings > Updates replaces `hachidori.exe` with a newer release of this
repository. It is manual, verified and bounded (`internal/update`); `serve`,
the browser `dashboard` and `hachidori setup` do not offer it.

**Network boundary.** Hachidori never checks for updates by itself. Startup,
opening Settings, opening Updates, rendering any page and changing the channel
use no network and no timer exists. Exactly two operator actions use it:

| action | requests |
|---|---|
| **Check for updates** | `GET https://api.github.com/repos/yohn-jp/hachidori/releases?per_page=100&page=N` (at most 5 pages), no redirects followed |
| **Download** | `GET https://github.com/yohn-jp/hachidori/releases/download/<tag>/<asset>` for the release's checksum file and executable; redirects only to GitHub's release-asset hosts (`objects.githubusercontent.com`, `release-assets.githubusercontent.com`, `github-releases.githubusercontent.com`) over https, at most 3, no credentials |

**Release authority.** Fixed in code (`internal/update/release.go`): the official
`yohn-jp/hachidori` GitHub Releases. No repository or URL is configurable, no
request is built from release metadata (a download URL in metadata must equal
the canonical one or the release is not installable) and no credential or
telemetry is sent.

**Release contract** (the shared Windows release workflow that
`.github/workflows/release.yml` calls):

- tags are SemVer without a `v` prefix: development `0.2.N-dev` (published by
  the development line), stable `X.Y.Z`. Legacy tags such as `dev-24`, drafts
  and other pre-release kinds are not offered.
- a release carries exactly one `hachidori-windows-amd64.exe` and one
  `hachidori-windows-amd64.exe.sha256` (`<sha256>  hachidori-windows-amd64.exe`).
  Zero or several of either, an asset that is not fully uploaded or one that
  lists an implausible size makes that release not downloadable.
- ordering is SemVer precedence, never lexical (`0.2.10-dev` is newer than
  `0.2.9-dev`; `0.2.5-dev` is older than `0.2.5`).

**Channels.** *Stable* offers stable releases. *Development* offers `-dev`
releases and stable releases. The channel is stored in `settings.json`
(`updates.channel`) and only selects what an explicit check offers; changing it
discards the displayed results and never downloads, installs or downgrades.
Releases not newer than the installed one are listed but cannot be downloaded:
an update never goes backwards, and Restart & update refuses a ready update that
the selected channel no longer offers or that is not newer.

**Installed version.** The executable carries no version string. It is known
from the helper's result for exactly this executable's SHA-256, or from an
explicit check that matched the executable's SHA-256 to the digest GitHub
reports for a release asset (kept in `settings.json` `updates.installed`, bound
to that digest). Otherwise it is unknown, and Updates says releases cannot be
compared.

**Download and verification.** Download fetches the checksum file first, then the
executable into `state/updates/<tag>/hachidori-windows-amd64.exe.part`. The
checksum file must hold exactly one line binding one SHA-256 to exactly that
asset name; if GitHub also reports a digest for the executable it must agree.
The size must equal the release metadata. The file becomes ready only after its
SHA-256 equals the checksum and it is a Windows executable: then it is renamed
into place and `state/updates/ready.json` is written last. Any failure removes
the partial file and writes no ready record, and an earlier ready update is kept.
Failures are classed (release metadata/network, malformed metadata, asset not
found or ambiguous, download, verification artifact, checksum mismatch,
replacement, restart) and shown with the next step. Authenticode is not part of
the release pipeline; none is invented, and a signature check can be added
beside the checksum later. Progress uses the long-running operation view
(phases checksum, download, verify; a percentage for the byte-counted download).

**Restart & update.** Nothing is overwritten in-process. The application
re-verifies the staged file, copies its own executable to
`state/updates/helper/` and starts that copy detached as
`hachidori apply-update --home <home> --pid <pid> --target <exe> [--restart-home <home>]`,
then closes. The helper (`update.Apply`) has no network code and no source or
URL argument: its source is the staged file the ready record names, and its only
destination is the executable the record was prepared for (a clean absolute
`*.exe` path, outside `state/updates`, a regular Windows executable). After the
application exited it re-verifies the staged file, copies it beside the
destination (`<exe>.new`) and verifies the copy, renames the current executable
to `<exe>.old`, and renames the copy into place; if the second rename fails the
previous executable is restored. Then it restarts `<exe>` (the new one after a
replacement, the previous one after a failure) and writes
`state/updates/result.json`. `<exe>.old` stays as the recoverable previous
executable. A helper that is refused or whose application does not exit changes
nothing. Restart & update is refused while an application operation (setup,
maintenance, start or stop) is in progress.

`HACHIDORI_HOME` (models, runtimes, history, evidence, settings) is never
replaced, recreated or deleted by an update; the new executable migrates through
the existing authorities.

## HACHIDORI_HOME

```text
HACHIDORI_HOME/
  tools/uv/0.12.19/
    uv[.exe]                   pinned private uv (sha256 verified)
    python/cpython-3.12.11-…/  uv-managed base CPython
  runtime/<flavor>-<digest>/   immutable once published
    env/                       relocatable venv (bin/python, Scripts\python.exe on Windows)
    spec/pyproject.toml, spec/uv.lock   the exact spec files it was materialized from
    worker/hachidori_worker.py (sha256 in spec and manifest)
    manifest.json              identity, Runtime Spec, verified Python version, installed distributions, worker digest
  runtime/.staging-<identity>/ in-progress materialization, never activated
  packages/                    downloaded uv release archive
  models/<owner>--<repo>/<revision>/   one per materialized catalog model
    model.safetensors …        every file sha256 pinned
    hachidori-model.json       the catalog entry (id, provider, repo, revision, files)
  cache/{uv,huggingface,torch,pip,xdg,nv,tmp,home}
  logs/worker.log, logs/doctor-worker.log
  state/active-runtime.json
  state/dashboard.json         last tunnel form values (non-secret), dashboard only
  state/updates/               explicit update downloads only: <tag>/ staged executable, ready.json,
                               result.json, helper/ (Windows desktop; see Updates)
  cache/webview2/              WebView2 browser profile of `hachidori desktop` (Windows only, disposable)
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

Pins: uv 0.12.19 (linux-amd64, windows-amd64; `internal/setup/spec.go`),
CPython 3.12.11, `torch==2.11.0+cu128` (NVIDIA driver ≥ 570, RTX 20xx–50xx) or
`torch==2.11.0+cpu`, `laya==0.3.21`, `opendecider==0.3.0`, `transformers==5.17.0`
and the full locked package set (`internal/setup/runtimespec/uv.lock`), default model `laya-base` =
`convaiinnovations/laya@55cf4c4e…` (Laya's own reviewed revision).

### Windows bootstrap locator

The only Hachidori state outside `HACHIDORI_HOME`, Windows only
(architecture §6.2.1):

```text
%LOCALAPPDATA%\Hachidori\bootstrap.json   {"schema":"hachidori.bootstrap/1","home":"D:\\Hachidori"}
```

`home.Discover` (desktop): explicit > `HACHIDORI_HOME` > valid locator >
unconfigured. Errors are typed: `ErrBootstrapMalformed`,
`ErrBootstrapUnknownSchema`, `ErrStoredHomeMissing` (distinct from the clean
unconfigured result). `home.Remember` writes it atomically (parent created only
then, no setup run); `home.Forget` removes only this file. Non-Windows builds
return `ErrBootstrapUnsupported` and create no bootstrap state.

Removing the executable and `HACHIDORI_HOME` removes everything Hachidori owns
(plus, on Windows, the tiny locator above, which holds no runtime state).

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

### Decision Evidence

Eval/benchmark reports are versioned as `hachidori.evidence.v1`. They retain
the full probability distribution, exact question digest, optional Question
Definition identity, request/inference timing, dataset digest, and the served
runtime/provider identity snapshotted from `GET /v1/status`. Expected labels
and evidence remain caller-side.

`hachidori replay` reloads the original dataset, verifies its SHA-256 and each
recorded question digest, reconstructs the original `/v1/decide` requests, and
can either print them or re-send them for comparison. Definition-backed datasets
must supply the same caller-side `--questions` definitions when replayed.

`testdata/eval/contract-example.jsonl` is a three-case format example, **not**
benchmark evidence. The coding-agent benchmark from architecture §12 is not in
this repository and must be supplied as a local JSONL file in this format.

### Question Definitions (caller side)

A Question Definition is a repository-owned, versioned v1 question. One JSON
object per file; unknown fields are rejected; no imports, inheritance or
templating:

```json
{"schema": "hachidori.question.v1", "id": "scope_expansion", "version": 1, "type": "choice",
 "instructions": "Did the agent modify files outside the scope the user requested?",
 "choices": ["yes", "no"], "descriptions": {"yes": "…"}}
```

- `version` is an explicit integer ≥ 1; `type`, instructions, choices and
  descriptions follow the same rules and limits as `/v1/decide` questions.
- It compiles to the v1 question `{id, type, instructions, choices,
  descriptions}` (choice order kept, empty descriptions omitted).
- Identity is `id@version` plus `digest` = `sha256:` of the canonical JSON
  (fixed field order, sorted description keys, no whitespace). Any change to
  id, version, instructions, choices (including order) or descriptions changes
  the digest; file path, file name and file formatting do not.

A dataset case may use `question_refs` instead of inline `questions`:

```json
{"id": "case-001", "state": "…", "question_refs": [{"id": "scope_expansion", "version": 1}], "expected": {"scope_expansion": "yes"}}
```

`--questions` (files or directories of `*.json`, repeatable) supplies the
definitions. References are resolved and compiled before the first request;
an unknown id/version, a duplicate definition or reference, a mismatching
optional `digest` pin, mixing `questions` and `question_refs` in one case, or
one question id bound to different versions (or to both inline and a
definition) within a dataset fails locally. The endpoint receives ordinary v1
requests only — never references, versions, digests or paths. Inline datasets
work unchanged. The report lists the resolved definitions as
`question_definitions` (`id`, `version`, `digest`). `testdata/questions/` and
`testdata/eval/contract-example-refs.jsonl` are the reference-form equivalent
of the contract example.
