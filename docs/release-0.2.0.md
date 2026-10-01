# Hachidori 0.2.0

Hachidori 0.2.0 is the first release that presents Hachidori as a Windows-first
local semantic inference workstation: one executable that sets up, runs and
supervises a resident inference worker on the accelerator host, and gives the
operator a desktop window to ask questions, run experiments and inspect
evidence.

The 0.1.x development line was the bootstrap: the `hachidori` CLI, the HTTP API
(`hachidori.v1`), a browser dashboard and a first WebView2 shell. 0.2.0
documents what that grew into. The authorities for each area stay
[architecture.md](architecture.md), [runtime.md](runtime.md),
[desktop.md](desktop.md) and [certification.md](certification.md); this page
summarizes them and does not replace them.

## Release status

- This page describes `main` as prepared for 0.2.0. Development builds of this
  line are published automatically as `0.2.N-dev` pre-releases by the shared
  development-release workflow (`development-line: "0.2"` in
  `.github/workflows/release.yml`).
- The stable `0.2.0` tag and GitHub Release are **not** created by the change
  that prepares them. They are a separate, explicit publication step.
- The executable carries no embedded version string. Its identity is the release
  asset and its SHA-256 (`hachidori-windows-amd64.exe.sha256`); the Diagnostics
  bundle records that digest.

## What 0.2.0 is

| Area | What an operator gets |
|---|---|
| Desktop shell | `hachidori.exe` with no arguments opens one native window over Microsoft WebView2, with a tray icon, close-to-tray, a single-instance guard, and opt-in start at sign-in and start minimized. The UI is the same server-rendered dashboard the runtime hosts; the window owns no runtime logic. |
| First run and setup | A storage-folder choice through the native Windows folder dialog, an explicit CUDA or CPU choice, and installation of the private runtime and default model with real setup phases. The bootstrap locator is written only after setup succeeds. Existing installations are recognised, and a missing or corrupt home gives a recovery screen instead of an exit. |
| Resident runtime | One supervised Python worker stays loaded on the host. Requests never start Python or load the model. Start, Stop and Restart act on the one controller shared by the window, the tray and the API. |
| Runtime and Diagnostics | Runtime shows readiness, model and device identity and lifecycle actions, with the operator summary separated from diagnostic evidence. Diagnostics holds doctor, the last worker failure, the tunnel launcher, the Desktop panel and an explicit local diagnostic bundle export. |
| Workbench | An interactive Question Workbench over `POST /v1/decide`, with Question Definition editing and export as `hachidori.question.v1`. |
| Experiments, Evidence, history | The Experiment Runner executes the caller-side evaluation harness and produces `hachidori.evidence.v1`. The Evidence (Error Explorer) workspace analyses a report deterministically. Saved experiments persist as immutable history entries under `state/history` and can be compared. |
| Settings | A typed settings authority: runtime defaults, interface language (English or Japanese), desktop preferences, Development Connection profiles, and a Models & runtimes manager. |
| Development Connections | Named, non-secret SSH reverse-tunnel profiles driven through the single tunnel manager. The window shows the exact `HACHIDORI_ENDPOINT=...` value to set on the development host. |
| Model and runtime management | Inventory of catalog-pinned runtimes and models, with explicit Materialize, Verify, Repair, Activate and Remove. Activation never restarts the worker and a requested device is never replaced by another. |
| Recovery and diagnostics | A worker that dies after READY is restarted at most 3 times in 10 minutes, then the runtime reports Needs attention until the operator chooses Restart Runtime. A WebView2 render-process failure reloads a bounded number of times, then reports. Worker text returned to API callers is redacted. |
| Localization | The operator interface is available in English and Japanese through one catalog (`internal/i18n`). |
| Native file selection | Native Windows file and folder pickers in the Workbench, Experiments and Evidence file workflows, in addition to first-run storage selection. Browser-hosted dashboards keep typed absolute paths. |
| DPI and layout | Per-monitor DPI awareness, reflow at 1180 px and 820 px, text sizes that follow the operator's preference, and no WebView zoom. |
| Desktop interaction and performance | One visual token system, a defined pending/failed/completed vocabulary, visible focus, reduced-motion and contrast-theme support. Hidden windows stop polling, unchanged live regions are not re-rendered, and the history table renders 100 rows per page. |
| Executable icon | The Windows executable embeds the current Hachidori icon as a multi-resolution resource (16, 24, 32, 48, 64, 128 and 256 px), and the development release build links it. |

The CLI is unchanged in purpose and remains available: `setup`, `doctor`,
`serve`, `dashboard`, `desktop`, `status`, `decide`, `eval`, `benchmark`,
`question` and `replay`. Every explicit subcommand keeps its terminal
behavior; only the no-argument launch is the desktop product path.

## Compatibility and operational boundaries

- **Platform.** The release asset is Windows amd64. The desktop needs the
  Microsoft WebView2 Runtime, which Hachidori detects but never downloads or
  installs. The CLI, API, dashboard and worker also build and run on Linux.
  WSL is not a desktop or runtime target. There is no installer and no
  automatic updater.
- **Loopback first.** The API (`127.0.0.1:7843`) and the dashboard
  (`127.0.0.1:7844`) refuse non-loopback binds. Requests whose `Host` is not
  loopback are rejected. State-changing dashboard actions are same-origin
  `POST`s with a per-process form token. The window can navigate only to the
  dashboard origin, and there is no JavaScript-to-Go bridge.
- **One mutable root.** `HACHIDORI_HOME` holds the managed runtime, models,
  caches, logs, history and activation state. On Windows, three small per-user
  files live beside it in `%LOCALAPPDATA%\Hachidori`: the bootstrap locator
  `bootstrap.json`, `desktop.json` and `settings.json`. Removing the executable,
  `HACHIDORI_HOME` and that folder removes everything Hachidori owns.
- **Deterministic setup.** Runtimes are declared by a Runtime Spec and built
  with a private, digest-pinned `uv`; models are immutable catalog identities
  with every file digest pinned. Serving never resolves packages or downloads
  anything. System Python, pip and uv are never used.
- **No silent fallback.** CUDA needs an NVIDIA driver 570 or newer (`cu128`
  build). When the requested device is unusable the failure is reported;
  Hachidori never switches from CUDA to CPU.
- **Single provider.** The model catalog has one entry, `laya-base`, on the
  Laya provider.

## Upgrading from 0.1.x

- Runtimes created by the former pip-based setup (`runtime/0.1.0-*`) are never
  reused or modified. An active legacy runtime, or one whose manifest lacks the
  declarative identity, is reported as `runtime_invalid` everywhere and is never
  READY. Run `hachidori setup` (or Install in the desktop) to materialize the
  current runtime next to it; the verified model is reused without downloading.
- The development executable is replaced by downloading the new asset and
  verifying its SHA-256. Existing homes are discovered automatically.
- No API schema change: the API stays `hachidori.v1`, evidence stays
  `hachidori.evidence.v1`, questions stay `hachidori.question.v1`.

## Verification state

**Automated, on Linux.** `gofmt`, `go vet`, `go test -race`, the Windows amd64
non-CGo cross-build and cross-vet, and a test that builds the Windows executable
and checks that its PE resource directory holds the icon group and every `.ico`
image byte-for-byte, with the `.ico` pixel-identical to `assets/icons`
(`go test ./internal/winres`). These are portable evidence. They are **not** a
`PASS` for any physical Windows item.

**Physical Windows.** No physical Windows run has been performed for this
release. Per [windows-certification-checklist.md](windows-certification-checklist.md),
every item below is `NOT_CHECKED`. The checklist also holds the identity block a
run must record. A cross-build never implies a physical `PASS`.

| Item | Outcome |
|---|---|
| W01 Clean first run | NOT_CHECKED |
| W02 Native folder picker | NOT_CHECKED |
| W03 Install to READY (CPU) | NOT_CHECKED |
| W04 Remembered-home launch | NOT_CHECKED |
| W05 WebView2 navigation and rendering | NOT_CHECKED |
| W06 Tray and reopen | NOT_CHECKED |
| W07 Start at sign-in | NOT_CHECKED |
| W08 Runtime restart | NOT_CHECKED |
| W09 Worker crash and recovery | NOT_CHECKED |
| W10 WebView2 process failure | NOT_CHECKED |
| W11 Diagnostic bundle export | NOT_CHECKED |
| W12 CPU runtime | NOT_CHECKED |
| W13 CUDA runtime | NOT_CHECKED |
| W14 Executable icon (Explorer, taskbar, window, tray) | NOT_CHECKED |

Desktop startup, navigation, idle CPU and memory budgets in
[desktop.md](desktop.md) are likewise server-side measurements only; the
WebView2 and physical-Windows parts are `NOT_CHECKED`. GPU/runtime
certification is a separate evidence boundary and has not been claimed here.

## Not in 0.2.0

- Multi-provider support and OpenJEV (#6).
- Inference-performance research and optimization (#94). The matrix is research,
  not shipped behavior.
- An installer, an automatic updater, or a non-Windows desktop.
- Changes to runtime, provider or API semantics.

## Known follow-ups

Recorded in [desktop.md](desktop.md) and not fixed in this release: saving
experiment history rescans every entry (O(n) per save), the Compare selectors
keep every history entry selectable, and the Experiments page decodes the whole
history index on every render.
