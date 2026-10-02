# Windows physical certification checklist

This is the record template for one physical Windows certification run of
`hachidori.exe`. The procedures live in [certification.md](certification.md);
this file fixes the outcome vocabulary, the identity that must accompany every
result and the complete item list.

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

| W19 | Update progress and repeated actions (#137) | Settings > Updates on a throwaway home with a copy of `hachidori.exe`: clicking **Check for updates** shows the button change to "Checking…" and the "RUNNING" panel (release list, indeterminate) immediately, even on a slow connection; while it runs **Check for updates**, **Download**, **Save channel** and **Restart & update** are disabled with the in-progress note and a second click starts nothing (confirm a single request burst in a network monitor); **Download** shows "Starting", then checksum, download with bytes and percent, and verify; reloading Updates mid-download shows the same operation; completion points to **Restart & update**; with the network disabled mid-download the failure shows its phase, cause and "partial download was discarded". Use a throwaway home, never a production one | NOT_CHECKED | |

| W20 | Clef-Flash source and CPU reference (#142) | "System One variant certification" in [certification.md](certification.md) step 1: `setup --device cpu --model clef-flash` (about 19 GB) verifies every file; `serve` with `HACHIDORI_CLEF_DTYPE=float32` (and again unset) reaches READY on the CPU, `hachidori status` shows `provider.provider` = `clef`, `provider.device` = `cpu`, `provider.dtype`, `provider.host_rss_bytes` and no `runtime.variant`; a real decision answers. Record the load time, host RAM and latency; whether it fits in 64 GB is whatever was measured. `NOT_CHECKED` when not run | NOT_CHECKED | |
| W21 | Q4 variant build (#142) | `hachidori variant optimize --model clef-flash`: the optimizer runtime materializes once, then phases starting, loading_source, resolving, quantizing, serializing, verifying, publish; `variant list` shows the variant with its preserved modules; `variant verify` passes; the source directory is unchanged; killing the build mid-way leaves no variant and no listed staging; `--reproduce` reports identical or the differing files. Record the build time and the output size | NOT_CHECKED | |
| W22 | Q4 variant on the RTX 3060 (#142) | `setup --device cuda --model clef-flash`, `activate --device cuda --model clef-flash --variant <id> --experimental`, `serve`: `hachidori status` shows `provider.device` `cuda:0`, `provider.dtype` `torch.bfloat16`, `provider.quantized_execution`, `runtime.variant.certification` `experimental/uncertified`; a real decision answers; record the exact VRAM (`worker.accelerator`), RAM and p50/p95. A CUDA failure is reported and never becomes a CPU run. `NOT_CHECKED` when no such hardware exists | NOT_CHECKED | |
| W23 | Fidelity certification of the real variant (#142) | `certify run` against the CPU reference and against the variant on the same corpus, then `certify evaluate`: record the flips, probability and confidence drift, divergence, labelled deltas if labels exist, resource facts (`MEASURED` / `NOT_CHECKED`), the policy and its verdict; a rejecting verdict keeps its evidence and blocks normal activation | NOT_CHECKED | |
| W24 | Variant lifecycle on the desktop (#142) | Settings > Models & runtimes: the variant row shows recipe, precision, certification state and selected/pending/running; **Optimize** and **Certify** show their phases and failure evidence; **Activate** is offered only for an accepted variant (**Activate as experimental** for an unjudged one, marked in status); **Restart runtime** applies it and Runtime shows the variant; a corrupted variant file is refused at Activate and at launch and the source is never started in its place | NOT_CHECKED | |

## Recorded state for the change that introduced this checklist

The change that added the diagnostic bundle and this checklist was prepared
without physical Windows access. Every row above is therefore `NOT_CHECKED`; the
portable tests and the Windows amd64 non-CGo cross-build/vet reported in the pull
request are not a `PASS` for any row.
