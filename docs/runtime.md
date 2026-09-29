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
| `hachidori setup [--home H] [--device cuda\|cpu] [--model ID]` | host | reconcile `HACHIDORI_HOME` with the Runtime Spec: private uv materializes the locked Python environment, the selected catalog model (default `laya-base`) is materialized separately, then activate |
| `hachidori serve [--home H] [--listen 127.0.0.1:7843]` | host | run HTTP + one resident worker; non-loopback binds are refused |
| `hachidori dashboard [--home H] [--listen 127.0.0.1:7843] [--addr 127.0.0.1:7844] [--ssh ssh]` | host | `serve` plus the host-local dashboard (see below) |
| `hachidori desktop [--home H] [--listen …] [--addr …] [--ssh ssh] [--background]` | host (Windows) | the same desktop composition as a no-argument `hachidori.exe`: first run/recovery or normal start in a resident WebView2 window with a tray icon (see below); fails with a clear error on other systems |
| `hachidori doctor [--home H]` | host | verify the installation including a real HTTP→worker→model smoke inference |
| `hachidori status [--endpoint URL]` | client | print `/v1/status` |
| `hachidori decide [--endpoint URL] <request.json\|->` | client | send one v1 decide request |
| `hachidori eval [--endpoint URL] [--questions PATH]... [--out report.json] <dataset.jsonl>` | client | caller-side evaluation |
| `hachidori benchmark [--endpoint URL] [--questions PATH]... [--warmup N] [--passes N] [--out report.json] <dataset.jsonl>` | client | eval plus warmup and repeated passes for latency |
| `hachidori question <definition.json\|dir>...` | client | validate Question Definitions locally; print id, version, digest and the compiled v1 question (no endpoint) |
| `hachidori replay [--questions PATH]... [--dataset D] [--case ID]... [--question ID]... [--print] [--out replay.json] <report.json>` | client | reconstruct and optionally re-send decisions recorded by eval/benchmark |

`--home` defaults to `HACHIDORI_HOME`; the CLI has no implicit home (the Windows
desktop bootstrap locator below is never consulted by CLI commands). `--endpoint`
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
| Experiment Runner (`/experiments`) | `internal/eval` (`question.Load`, `eval.Load`, `eval.RunEvidence`) run caller-side in the dashboard process against the same inference API address |
| Question Workbench (`/workbench`) | a caller of the existing `POST /v1/decide` on the dashboard's inference API address; `internal/question` / `internal/api` validation and compilation |

Every state-changing action is a same-origin `POST` carrying a per-process form
token; `GET` never changes state. All rendered values go through `html/template`
escaping. The page polls `/live` every 3 s; there is no frontend build.

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
   `activation`); there is no percentage. Setup's log is kept in
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
`starting`, `warming`, `ready`, `stopping`, and `failed`. Setup reports
only real phases entered (`preparing`, `runtime`, `model`, `activation`);
no synthetic percentage is exposed.

The controller serializes setup/start/stop/restart actions so UI retries cannot
create duplicate runtime ownership. Setup accepts the same device and catalog
model selection as the CLI. Home discovery remains outside the controller.

## Runtime materialization

`hachidori setup` is declarative. The desired runtime is a **Runtime Spec**:
Go data in `internal/setup/spec.go` plus the uv project embedded from
`internal/setup/runtimespec/` (`pyproject.toml` and the locked `uv.lock`,
with one uv extra per PyTorch flavor: `cpu`, `cu128`). The spec records the
schema, platform, CPython version, provider (`laya==0.3.21`), exact torch build,
flavor, pinned uv version and executable digest, the SHA-256 of both uv project
files and of the worker script. The model is not part of it: selecting a
different compatible checkpoint reuses the same runtime.

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
     else download to staging, verify, rename
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
env) must report exactly the spec's Python version, `laya==0.3.21` and
`torch==2.11.0+<flavor>`, its prefix must be the runtime's `env/` and its base
interpreter the uv-managed CPython under `tools/`. Failure or interruption at any
step leaves `state/active-runtime.json` unchanged; staging is never treated as a
runtime and is recreated on the next run. There is no retry loop beyond uv's
own. A runtime directory that exists but does not verify is an explicit error
(remove it to rematerialize). Runtimes created by the former pip-based setup
(`runtime/0.1.0-*`) never match an identity; setup materializes a new one next
to them and `doctor` reports them as `runtime_invalid`.

### Model catalog

Models are immutable catalog identities declared in `internal/setup/spec.go`
(`setup.Models`): a stable Hachidori model ID, provider kind (`laya`), upstream
repository, immutable commit revision and every required file with its SHA-256.

| ID | checkpoint |
|---|---|
| `laya-base` (default) | `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, the upstream English ModernBERT-large checkpoint |

`hachidori setup --model <id>` selects one; omitting `--model` selects
`laya-base`. Only catalog IDs are accepted (an unknown ID fails before anything
is changed); there is no repository or revision input. Adding a checkpoint
(for example a fine-tuned Laya) means declaring another entry, not changing the
worker or provider. The model directory `models/<owner>--<repo>/<revision>/` and
its `hachidori-model.json` are derived from the entry. `state/active-runtime.json`
records the runtime identity, the model ID (`model_id`), the model directory and
the device. A failed materialization of a newly selected model leaves the active
runtime and model unchanged. Changing the active model takes effect when the
host is restarted; a READY worker never switches models.

`serve` resolves only the activated model: the activation's model ID must be a
catalog entry, the directory must be the one derived from it and the
materialized manifest must equal the entry; the worker then verifies every file
digest while loading. Nothing is downloaded. `/v1/status` (and the dashboard)
report `runtime.model_id` and `runtime.model`; `doctor` verifies the selected
entry's files and prints its ID, repository and revision. Activation records
from before model selection (no `model_id`) are resolved by their directory.

Network is needed only while materializing. `serve` and `doctor` use the
published runtime's absolute interpreter and do not need uv, its cache or
package indexes.

Updating the environment: edit `runtimespec/pyproject.toml`, run
`uv lock --directory internal/setup/runtimespec` with the pinned uv version,
commit both files. Setup runs `uv sync --locked`, which refuses a stale lock.

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
`torch==2.11.0+cpu`, `laya==0.3.21`, `transformers==5.17.0` and the full locked
package set (`internal/setup/runtimespec/uv.lock`), default model `laya-base` =
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
