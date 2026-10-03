# Windows physical certification checklist

This is the record template for one physical Windows certification run of
`hachidori.exe`. The procedures live in [certification.md](certification.md);
this file fixes the outcome vocabulary, the identity that must accompany every
result and the complete item list. It also classifies every assertion of every
item as `CI_AUTOMATED`, `PHYSICAL_REQUIRED` or `OPTIONAL_HARDWARE` (see
[Assertion classification](#assertion-classification)), so it is explicit which
assertions the main-push Windows E2E certification
([test/windows-e2e](../test/windows-e2e/README.md)) may prove on hosted runners
and which remain physical.

## Outcome vocabulary

Every item is exactly one of:

| outcome | meaning |
|---|---|
| `PASS` | The item was run on a real Windows desktop with the executable identified below, and the expected result was observed. |
| `FAIL` | The item was run and the expected result was not observed. Record what was seen. |
| `NOT_CHECKED` | The item was not run on that machine with that executable. This is the default. |

Rules:

- A cross-build, a portable unit test, a Linux run or an emulated environment is
  never `PASS` for a physical item. Automation may assist (for example
  `go test ./internal/desktop`), but each physical assertion is made by a person
  looking at the real Windows desktop.
- Hardware-dependent items (CUDA) are `NOT_CHECKED` when the hardware does not
  exist on the machine. Do not substitute CPU results.
- Do not edit a `FAIL` or `NOT_CHECKED` to `PASS` without a new run that records
  its own identity block.

## Identity block (fill in once per run)

Every result below is valid only for this identity.

| field | value |
|---|---|
| Executable file name | |
| Executable SHA-256 (`Get-FileHash -Algorithm SHA256`) | |
| Executable source (release asset name, or local build commit) | |
| Windows edition, version and build (`winver`) | |
| CPU / GPU / NVIDIA driver (CUDA items only) | |
| WebView2 Runtime version (console line `WebView2 Runtime <version>`) | |
| Device of the home under test (`cpu` / `cuda`) | |
| Diagnostic bundle file (Diagnostics > Export bundle) | |
| Run date and operator | |

The diagnostic bundle (`state\diagnostics\hachidori-diagnostics-*.zip` under the
Hachidori home) records the executable name and SHA-256, runtime, model, device,
WebView2 version, OS and architecture, and worker recovery state. It is created
only by **Export bundle** on the Diagnostics page, stays on the machine and is
safe to attach to an Issue: it excludes request/state/question text, datasets,
environment variables, SSH keys and known_hosts, and model files. Windows
edition and build are not in the bundle; take them from `winver`.

## Checklist

Record one outcome per row. Steps refer to the sections in
[certification.md](certification.md) ("Windows desktop shell", "Windows
first-run wizard", "Windows recovery boundaries").

| id | item | procedure | outcome | evidence / notes |
|---|---|---|---|---|
| W01 | Clean first run | First-run wizard 1: clean profile, double-click, "Hachidori" window shows the first-run page, no companion console | NOT_CHECKED | |
| W02 | Native folder picker | First-run wizard 2: **Browse...** opens the standard Windows folder dialog and returns the chosen folder (#47 regression: no `No such interface supported`; cancel still leaves nothing chosen) | NOT_CHECKED | |
| W03 | Install to READY (CPU) | First-run wizard 3-4: CPU install, READY, `bootstrap.json` holds only `schema` and `home` | NOT_CHECKED | |
| W04 | Remembered-home launch | First-run wizard 5: after Quit, a new double-click reaches READY without the wizard from the same home; wizard 6-7: renamed folder and corrupt `bootstrap.json` give a recovery screen, not an exit | NOT_CHECKED | |
| W05 | WebView2 navigation and rendering | Desktop shell 1, 3, 5 (#41 regression: a visible first-run launch renders the first-run UI, never a blank white surface; a render-process failure reloads at most 3 times, then one message box and tray notice, and `WebView2 navigation/process failure` lines appear on the console) | NOT_CHECKED | |
| W06 | Tray and reopen | Desktop shell 2, 4 and wizard 5, 9: close hides to tray with the same worker pid, tray Open restores the same window, a second launch activates it, exactly one tray icon, **Quit Hachidori** frees the ports and leaves no `python` or `msedgewebview2.exe` child | NOT_CHECKED | |
| W07 | Start at sign-in | Wizard 10: enable creates one `HKCU\...\Run` value `Hachidori`, disable removes it, and after a real sign-out/sign-in the app starts in the background and reaches Ready | NOT_CHECKED | |
| W08 | Runtime restart | Wizard 9 and Recovery 2: Restart Runtime gives a new worker pid with no second owner; Stop and Quit never restart the worker by themselves | NOT_CHECKED | |
| W09 | Worker crash and recovery | Recovery 1: end the `python` worker; Diagnostics shows "Recovering from an unexpected worker exit", the worker returns to READY with a new pid; more than 3 kills in 10 minutes ends in Needs attention and "Automatic recovery stopped" until Restart Runtime | NOT_CHECKED | |
| W10 | WebView2 process failure | Recovery 4: end a `msedgewebview2.exe` render process of this run; bounded reload, then message box and tray notice | NOT_CHECKED | |
| W11 | Diagnostic bundle export | Diagnostics > Export bundle writes one `.zip` under `state\diagnostics`; it contains only `manifest.json`, `facts.json` and `worker-log-tail.txt`, and the facts show the WebView2 version, the executable SHA-256 matching the identity block, and the recovery state seen in W09 | NOT_CHECKED | |
| W12 | CPU runtime | Windows RTX host steps 2-3 with `--device cpu`: `setup`, `doctor` all checks pass, real smoke inference | NOT_CHECKED | |
| W13 | CUDA runtime (hardware required) | Same with `--device cuda` on an NVIDIA GPU with driver 570 or newer; with an unusable GPU the failure is reported and Hachidori never switches to CPU. `NOT_CHECKED` when no such hardware exists | NOT_CHECKED | |
| W14 | Executable icon | In Explorer, the taskbar, the window title bar, Alt+Tab and the tray, `hachidori.exe` shows the current Hachidori icon (`assets/icons`). The embedded resource itself is verified deterministically by `go test ./internal/winres`. Explorer caches icons per path, so judge a stale-looking icon only after copying the executable to a new path or clearing the icon cache | NOT_CHECKED | |
| W15 | OpenDecider-nano selection (#113) | In Settings > Models & runtimes: choose `opendecider-nano` (nothing downloads or changes by choosing), **Materialize**, confirm it is not active; **Activate** and read "applies on restart"; **Restart runtime** and confirm Serving now / `runtime.model_id` / `provider.provider` = `opendecider`, `provider.dtype` and `provider.device` (`cuda:0` on a CUDA home); a real decision answers; switch back to `laya-base` the same way. A forced load failure (for example a damaged model file) keeps the previous model active | NOT_CHECKED | |
| W16 | Laya vs OpenDecider-nano comparison on the RTX 3060 (#113) | "Decision-model comparison" in [certification.md](certification.md): same corpus and device, accuracy, ECE, high-confidence errors, p50/p95, cold load and warmup, resident/peak VRAM for both models. Never inferred from CI or another machine. `NOT_CHECKED` when no such hardware exists | NOT_CHECKED | |
| W17 | Model switch failure and progress (#117) | On the host that showed `FAILED` / `spawning` / `worker_startup: worker exited: exit status 2` with a runtime from before #116: Runtime and Settings name the active pair as not startable by this build (older worker) and Restart does not start a process; Materialize the current `cuda` runtime and `opendecider-nano`, watching the phases, the download bytes (determinate) and the uv steps (indeterminate); Activate; **Restart runtime** starts `opendecider-nano` on `cuda` (Serving now, `provider.device` `cuda:0`) and a real decision answers; a forced load failure shows the failed phase, the class and the stderr tail on Runtime | NOT_CHECKED | |
| W18 | Manual verified update (#126) | Settings > Updates on a copy of `hachidori.exe` placed in a scratch folder with a throwaway home: opening Settings and Updates and switching Stable/Development shows no request (confirm with a network monitor: nothing leaves the host); **Check for updates** lists releases; **Download** shows phases and bytes and reaches "matches the release checksum"; a corrupted download (alter the staged file or the checksum) is rejected and **Restart & update** stays unavailable; **Restart & update** closes the window, replaces the executable, reopens the same path, shows UPDATED, `<exe>.old` holds the previous executable and the home, models, settings, history and evidence are unchanged; a held lock on the executable (for example a scanner) ends in NOT UPDATED with the previous executable still starting. Use a throwaway home, never a production one | NOT_CHECKED | |
| W19 | OpenDecider-nano FP32 vs BF16 on the RTX 3060 (#136) | "OpenDecider-nano FP32 vs BF16" in [certification.md](certification.md): `hachidori status` shows `provider.dtype` `torch.float32` by default and `torch.bfloat16` with `HACHIDORI_OPENDECIDER_DTYPE=bfloat16` on `cuda:0`; an unsupported value refuses the launch; `hachidori precision` over the FP32 and BF16 resident comparisons reports flips, probability/confidence deltas, quality, latency, scaling, memory, load/warmup and errors. Never inferred from CI. `NOT_CHECKED` when no such hardware exists | NOT_CHECKED | |
| W20 | Clef-Flash source and CPU reference (#142) | "System One variant certification" in [certification.md](certification.md) step 1: `setup --device cpu --model clef-flash` (about 19 GB) verifies every file; `serve` with `HACHIDORI_CLEF_DTYPE=float32` (and again unset) reaches READY on the CPU, `hachidori status` shows `provider.provider` = `clef`, `provider.device` = `cpu`, `provider.dtype`, `provider.host_rss_bytes` and no `runtime.variant`; a real decision answers. Record the load time, host RAM and latency; whether it fits in 64 GB is whatever was measured. `NOT_CHECKED` when not run | NOT_CHECKED | |
| W21 | Q4 variant build (#142) | `hachidori variant optimize --model clef-flash`: the optimizer runtime materializes once, then phases starting, loading_source, resolving, quantizing, serializing, verifying, publish; `variant list` shows the variant with its preserved modules; `variant verify` passes; the source directory is unchanged; killing the build mid-way leaves no variant and no listed staging; `--reproduce` reports identical or the differing files. Record the build time and the output size | NOT_CHECKED | |
| W22 | Q4 variant on the RTX 3060 (#142) | `setup --device cuda --model clef-flash`, `activate --device cuda --model clef-flash --variant <id> --experimental`, `serve`: `hachidori status` shows `provider.device` `cuda:0`, `provider.dtype` `torch.bfloat16`, `provider.quantized_execution`, `runtime.variant.certification` `experimental/uncertified`; a real decision answers; record the exact VRAM (`worker.accelerator`), RAM and p50/p95. A CUDA failure is reported and never becomes a CPU run. `NOT_CHECKED` when no such hardware exists | NOT_CHECKED | |
| W23 | Fidelity certification of the real variant (#142) | `certify run` against the CPU reference and against the variant on the same corpus, then `certify evaluate`: record the flips, probability and confidence drift, divergence, labelled deltas if labels exist, resource facts (`MEASURED` / `NOT_CHECKED`), the policy and its verdict; a rejecting verdict keeps its evidence and blocks normal activation | NOT_CHECKED | |
| W24 | Variant lifecycle on the desktop (#142) | Settings > Models & runtimes: the variant row shows recipe, precision, certification state and selected/pending/running; **Optimize** and **Certify** show their phases and failure evidence; **Activate** is offered only for an accepted variant (**Activate as experimental** for an unjudged one, marked in status); **Restart runtime** applies it and Runtime shows the variant; a corrupted variant file is refused at Activate and at launch and the source is never started in its place | NOT_CHECKED | |
| W25 | Resumable Clef-Flash download (#148) | `hachidori forge preflight materialize --device cpu` then `setup --device cpu --model clef-flash`; interrupt it (Ctrl-C, or cut the network) part-way through a `model-0000N-of-00004.safetensors` file; `variant list` / Settings show the kept partial and no materialized model; run `setup` again: the log says "resuming ... at byte N of M", the progress counts the held bytes, the file completes and every pinned digest verifies. Also run it once with the partial's `.part.json` deleted to see the explicit restart from byte 0. Record the bytes saved by the resume. `NOT_CHECKED` when not run on the real Hub | NOT_CHECKED | |
| W26 | Forge preflight on the workstation (#148) | `forge preflight optimize` and `forge preflight probe --variant <id> --device cuda` on the RTX 3060 host: the report shows the real free disk, installed and available RAM (`unknown` for fit, never `pass`), the CUDA device name, capability and VRAM from torch, and blocks when the disk is short or CUDA is unusable (never a cpu run). Record the report (`-json`). `NOT_CHECKED` when not run | NOT_CHECKED | |
| W27 | Probe of the real persisted variant (#148) | `forge probe --device cuda <variant-id>` after `variant optimize`: the worker loads `variants/clef-flash/<id>` (provenance `variant_id`, `device` `cuda:0`, `dtype`), answers one decision, and exits; the running runtime, `state/active-runtime.json` and `state/certifications` are unchanged; record load, warmup, request, host RSS and VRAM. A forced failure (for example a damaged variant file) fails in its phase with a diagnostic. `NOT_CHECKED` when no such hardware exists | NOT_CHECKED | |
| W28 | Forge diagnostic after a real failure (#148) | After a failed step (for example an optimization stopped for lack of RAM): `forge diagnostics list`, `show` and `export` give a bounded document with the operation and phase, source/variant/recipe identities, versions, RAM/VRAM/disk, the error chain and the stderr/log tails, with no token, path of the home or dataset content; Settings, Models & runtimes links the same document from the failed operation. `NOT_CHECKED` when no failure was produced | NOT_CHECKED | |
| W29 | Update progress and repeated actions (#137; formerly the second `W19`) | Settings > Updates on a throwaway home with a copy of `hachidori.exe`: clicking **Check for updates** shows the button change to "Checking…" and the "RUNNING" panel (release list, indeterminate) immediately, even on a slow connection; while it runs **Check for updates**, **Download**, **Save channel** and **Restart & update** are disabled with the in-progress note and a second click starts nothing (confirm a single request burst in a network monitor); **Download** shows "Starting", then checksum, download with bytes and percent, and verify; reloading Updates mid-download shows the same operation; completion points to **Restart & update**; with the network disabled mid-download the failure shows its phase, cause and "partial download was discarded". Use a throwaway home, never a production one | NOT_CHECKED | |

## Assertion classification

Each item above is made of assertions. Every assertion has one stable ID
(`W<item>.<n>`) and exactly one class:

| class | meaning |
|---|---|
| `CI_AUTOMATED` | Deterministic and observable without a human or special hardware. The main-push Windows E2E certification, or the named portable test, may prove it. The proof is CI evidence for the exact candidate executable; it is **never** a physical `PASS`. |
| `PHYSICAL_REQUIRED` | Needs a person at a real Windows desktop or a production-scale model or machine state: human-visible WebView2 rendering, native picker, tray/taskbar/Explorer, sign-out/sign-in, sleep/resume, multi-monitor/DPI, a real production model. Always `NOT_CHECKED` until a physical run records it. |
| `OPTIONAL_HARDWARE` | Needs hardware hosted runners do not have (NVIDIA CUDA, VRAM, RTX 3060 performance, FP32/BF16 hardware certification). `NOT_CHECKED` when the hardware does not exist; never substituted by CPU results. |

Rules:

- The IDs are permanent. An ID is never reused for another assertion and a
  retired assertion keeps its row. The item that was previously a second `W19`
  (update progress and repeated actions, #137) is now `W29`; `W19` is the
  OpenDecider-nano FP32 vs BF16 item only.
- The `Proof` column of a `CI_AUTOMATED` assertion is `shard:<name>` (one of
  `bootstrap`, `runtime`, `recovery`, `single-instance`, `update`,
  `diagnostics`, the six proof boundaries of the Windows E2E workflow) or
  `go test <package>` for an existing portable test. It is `-` otherwise.
- Hosted-runner proof of a `CI_AUTOMATED` assertion does not change the meaning
  of a physical outcome above. A physical item is `PASS` only when its
  procedure was observed on a real Windows desktop; #66 remains the authority for
  physical Windows certification.
- Scenario-level evidence of the hosted runs is described in
  [test/windows-e2e/README.md](../test/windows-e2e/README.md).

| id | assertion | class | proof |
|---|---|---|---|
| W01.1 | The "Hachidori" window visibly shows the first-run page with no companion console | PHYSICAL_REQUIRED | - |
| W01.2 | A clean profile (no `bootstrap.json`, no `HACHIDORI_HOME`) starts in the first-run state and writes nothing before Install | CI_AUTOMATED | shard:bootstrap |
| W02.1 | **Browse...** opens the standard Windows folder dialog and returns the chosen folder (#47); cancel leaves nothing chosen | PHYSICAL_REQUIRED | - |
| W02.2 | A chosen folder (spaces, Unicode, supported deep paths) is accepted and reported with its free space through the controller | CI_AUTOMATED | shard:bootstrap |
| W03.1 | A deterministic CPU setup reaches READY with model, device and runtime identity | CI_AUTOMATED | shard:runtime |
| W03.2 | `bootstrap.json` holds only `schema` and `home`; the heavy state is under the chosen home; owned staging is cleaned up | CI_AUTOMATED | shard:bootstrap |
| W03.3 | The window shows the setup phases as they begin, then the runtime warming, then READY | PHYSICAL_REQUIRED | - |
| W03.4 | A CUDA install reaches READY on a supported NVIDIA machine | OPTIONAL_HARDWARE | - |
| W04.1 | After Quit, a new launch reaches READY from the same home without the wizard | CI_AUTOMATED | shard:runtime |
| W04.2 | A renamed or missing home gives a recovery state naming the folder and installs nothing | CI_AUTOMATED | shard:bootstrap |
| W04.3 | A corrupt `bootstrap.json` gives a diagnostic recovery state, not an exit | CI_AUTOMATED | shard:bootstrap |
| W04.4 | Browse to locate the renamed-back folder, then "Use this installation" starts it without running setup | PHYSICAL_REQUIRED | - |
| W05.1 | A visible first-run launch renders the first-run UI, never a blank white surface (#41) | PHYSICAL_REQUIRED | - |
| W05.2 | A render-process failure reloads at most 3 times, then one message box and tray notice, with `WebView2 navigation/process failure` console lines | PHYSICAL_REQUIRED | - |
| W05.3 | F12 / Ctrl+Shift+I open no DevTools and right-click shows no browser context menu | PHYSICAL_REQUIRED | - |
| W05.4 | Without the WebView2 Runtime the command fails before any worker starts with the WebView2 diagnostic and downloads nothing | PHYSICAL_REQUIRED | - |
| W06.1 | Closing the window hides it to the tray with the same worker pid; tray Open restores the same window | PHYSICAL_REQUIRED | - |
| W06.2 | A second launch exits 0, starts no second worker or endpoint and leaves the first owner's worker pid unchanged (non-visual activation contract) | CI_AUTOMATED | shard:single-instance |
| W06.3 | Exactly one tray icon exists; its tooltip and menu header read Starting, Ready, Stopped and Needs attention with a one-time notice, and Open lands on Diagnostics | PHYSICAL_REQUIRED | - |
| W06.4 | **Quit Hachidori** frees the API and dashboard ports and leaves no owned worker process | CI_AUTOMATED | shard:runtime |
| W06.5 | After Quit the tray icon is gone and no `msedgewebview2.exe` child of the run remains | PHYSICAL_REQUIRED | - |
| W06.6 | A second launch visibly restores and focuses the running window, also from the tray | PHYSICAL_REQUIRED | - |
| W07.1 | Enabling start at sign-in creates exactly one `HKCU\...\Run` value `Hachidori`, enabling again changes nothing, disabling removes it | PHYSICAL_REQUIRED | - |
| W07.2 | After a real sign-out/sign-in the app starts in the background and reaches Ready | PHYSICAL_REQUIRED | - |
| W08.1 | Restart Runtime gives a new worker pid with no second owner | CI_AUTOMATED | shard:runtime |
| W08.2 | Stop and Quit never restart the worker by themselves | CI_AUTOMATED | shard:runtime |
| W09.1 | Ending the worker reports "Recovering from an unexpected worker exit" and the worker returns to READY with a new pid | CI_AUTOMATED | shard:recovery |
| W09.2 | More than 3 worker kills in 10 minutes end in Needs attention and "Automatic recovery stopped" until Restart Runtime | CI_AUTOMATED | shard:recovery |
| W09.3 | Persisted state is consistent after worker termination and supported corrupt-state or interruption recovery | CI_AUTOMATED | shard:recovery |
| W10.1 | Ending a `msedgewebview2.exe` render process gives a bounded reload, then message box and tray notice | PHYSICAL_REQUIRED | - |
| W11.1 | Export bundle writes one `.zip` under `state\diagnostics` holding only `manifest.json`, `facts.json` and `worker-log-tail.txt` | CI_AUTOMATED | shard:diagnostics |
| W11.2 | The facts carry the executable name and SHA-256 matching the candidate, runtime, model, device and the recovery state | CI_AUTOMATED | shard:diagnostics |
| W11.3 | The facts carry the WebView2 Runtime version of the desktop shell | PHYSICAL_REQUIRED | - |
| W11.4 | The bundle excludes request/state/question text, datasets, environment variables, SSH keys, known_hosts, credentials and model files | CI_AUTOMATED | shard:diagnostics |
| W12.1 | `setup`, `doctor` (all checks pass) and a real smoke inference with the production model succeed with `--device cpu` | PHYSICAL_REQUIRED | - |
| W12.2 | A deterministic fixture CPU home crosses packaged executable, controller, worker, HTTP API and typed inference; this is not a production-model result | CI_AUTOMATED | shard:runtime |
| W13.1 | The CUDA runtime sets up, passes `doctor` and answers a real decision on an NVIDIA GPU with driver 570 or newer | OPTIONAL_HARDWARE | - |
| W13.2 | Requesting CUDA with no usable GPU reports the failure and never switches to CPU | CI_AUTOMATED | shard:runtime |
| W14.1 | `hachidori.exe` shows the current Hachidori icon in Explorer, the taskbar, the title bar, Alt+Tab and the tray | PHYSICAL_REQUIRED | - |
| W14.2 | The embedded icon resource matches `assets/icons` byte-for-byte | CI_AUTOMATED | go test ./internal/winres |
| W15.1 | Selecting, Materializing, Activating and restarting `opendecider-nano` serves it on `cuda:0` and a real decision answers; switching back to `laya-base` works | OPTIONAL_HARDWARE | - |
| W15.2 | A forced load failure (for example a damaged model file) keeps the previous model active | PHYSICAL_REQUIRED | - |
| W16.1 | Laya vs OpenDecider-nano comparison on the RTX 3060: accuracy, ECE, high-confidence errors, latency, cold load and warmup, VRAM | OPTIONAL_HARDWARE | - |
| W17.1 | A runtime from before #116 is named as not startable by this build and Restart starts no process | PHYSICAL_REQUIRED | - |
| W17.2 | Materializing the current `cuda` runtime and `opendecider-nano`, Activate and Restart serve it on `cuda:0` and a real decision answers | OPTIONAL_HARDWARE | - |
| W17.3 | A forced load failure shows the failed phase, the class and the stderr tail on Runtime | PHYSICAL_REQUIRED | - |
| W18.1 | Opening Settings > Updates and switching Stable/Development sends no request | CI_AUTOMATED | shard:update |
| W18.2 | Check for updates lists releases; Download reports phases and bytes and reaches "matches the release checksum" | CI_AUTOMATED | shard:update |
| W18.3 | A corrupted download or checksum is rejected and Restart & update stays unavailable | CI_AUTOMATED | shard:update |
| W18.4 | Restart & update replaces the executable of a disposable copy, reopens the same path, reports UPDATED, keeps the previous executable as `<exe>.old` and leaves home, models, settings, history and evidence unchanged | CI_AUTOMATED | shard:update |
| W18.5 | A held lock on the executable ends in NOT UPDATED with the previous executable still starting | CI_AUTOMATED | shard:update |
| W18.6 | The window visibly closes, replaces the executable and reopens at the same path | PHYSICAL_REQUIRED | - |
| W19.1 | OpenDecider-nano FP32 vs BF16 on the RTX 3060 (#136): `provider.dtype`, an unsupported value refusing the launch, and the `hachidori precision` comparison | OPTIONAL_HARDWARE | - |
| W20.1 | Clef-Flash source download and CPU reference run (about 19 GB) with a real decision | PHYSICAL_REQUIRED | - |
| W21.1 | Q4 variant build, `variant list`/`verify`, kill-mid-build cleanup and `--reproduce` on the real model | PHYSICAL_REQUIRED | - |
| W22.1 | Q4 variant on the RTX 3060: `cuda:0`, `torch.bfloat16`, a real decision, exact VRAM, and a CUDA failure never becoming a CPU run | OPTIONAL_HARDWARE | - |
| W23.1 | Fidelity certification of the real variant against the CPU reference (`certify run`, `certify evaluate`) | PHYSICAL_REQUIRED | - |
| W24.1 | Variant lifecycle on the desktop: Optimize, Certify, Activate, Restart and refusal of a corrupted variant | PHYSICAL_REQUIRED | - |
| W25.1 | Resumable Clef-Flash download against the real Hub, with and without the `.part.json` | PHYSICAL_REQUIRED | - |
| W26.1 | Forge preflight on the workstation: real disk, RAM, CUDA device facts and blocking when CUDA is unusable | OPTIONAL_HARDWARE | - |
| W27.1 | `forge probe --device cuda` of the real persisted variant leaves the running runtime and certification state unchanged | OPTIONAL_HARDWARE | - |
| W28.1 | Forge diagnostics after a real failure give a bounded document without token, home path or dataset content, linked from Settings | PHYSICAL_REQUIRED | - |
| W29.1 | Check for updates shows "Checking…" and the RUNNING panel immediately; while it runs Check, Download, Save channel and Restart & update are disabled with the in-progress note | PHYSICAL_REQUIRED | - |
| W29.2 | A second Check for updates while one runs starts no second request burst | CI_AUTOMATED | shard:update |
| W29.3 | Download reports Starting, checksum, download with bytes and percent, and verify; the same operation is observable after a reload; completion points to Restart & update | CI_AUTOMATED | shard:update |
| W29.4 | A network failure mid-download reports phase, cause and "partial download was discarded" | CI_AUTOMATED | shard:update |

## Recorded state for the change that introduced this checklist

The change that added the diagnostic bundle and this checklist was prepared
without physical Windows access. Every row above is therefore `NOT_CHECKED`; the
portable tests and the Windows amd64 non-CGo cross-build/vet reported in the pull
request are not a `PASS` for any row.
