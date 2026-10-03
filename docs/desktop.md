# Desktop visual, interaction and performance contract

This is the product contract for how the Windows desktop looks, behaves and performs. It covers the workstation (Runtime, Workbench, Experiments, Evidence, Settings, Diagnostics) and first run (Issue #105). It changes no runtime, inference, evidence or API semantics. `docs/architecture.md` and `docs/runtime.md` stay authoritative for those.

Architecture invariants:

- Go is the application and runtime authority.
- WebView2 is the renderer.
- Pages are server-rendered Go templates with small inline scripts. There is no frontend framework and no second presentation state.
- Localization goes through `internal/i18n` only.
- Raw evidence is always reachable.
- The WebView is never zoomed.

## 1. Visual system

**The one definition.** `internal/ui/system.css` defines every token and shared primitive. The dashboard and the first-run page both inline it; the CSP admits inline styles only. A surface may add layout rules but never redefines a token. Tests enforce this (`TestWorkspacesUseTheOneVisualSystem`, `TestFirstRunUsesTheVisualSystem`, `TestVisualSystemContract`).

Visual priorities, in order:

1. information hierarchy
2. typography
3. spacing
4. alignment
5. content measure
6. semantic state
7. feedback
8. decoration

| Family | Tokens | Rule |
|---|---|---|
| Typography | `--fs-display` 34px (the one metric a workspace is about), `--fs-title` 22px (h1), `--fs-figure` 24px (the operational state word, a result or a progress figure), `--fs-value` 18px (metric), `--fs-heading` 15px (h2), `--fs-body` 14px, `--fs-small` 13px (notes, secondary facts), `--fs-label` 11px uppercase (column, metric and group labels), `--fs-code` 13px mono (paths, identifiers, raw evidence) | All sizes are `rem` against `html { font-size: 100% }`, so the operator's text-size preference scales the UI. Numbers are tabular. |
| Spacing | `--sp-1`…`--sp-7` = 4, 8, 12, 16, 24, 32, 40px | Spacing and alignment separate content before borders or fills do. |
| Measure | `--measure` 72ch for long-form prose only (`.lede`, `.prose`, `.empty`); `--ws-max` 88rem workspace width | Tables and evidence may use the full workspace width. Short helper text (`.note`), fields and controls use the width of the section that holds them and wrap only where it ends; a numeric entry is as wide as its digits (`12ch`). A page never adds a width of its own. |
| Geometry | `--ctl-h` 32px for every button, input, select, nav item and disclosure summary; `--r` 6px for controls; `--r-lg` 8px and `--shadow-pop` for transient surfaces that sit above the page (dialogs, popovers); `--r-pill` for badges; `--col-min` 14rem, the narrowest column that may hold a label and a machine identity | Content is never grouped by radius, fill or shadow. The only elevation is `--shadow-pop`, and only for a surface that genuinely occupies a higher plane. The focus ring is the only other shadow. |
| Color roles | `--bg`, `--chrome`, `--surface`, `--raised`, `--inset`; `--text`, `--text-2`, `--text-3`; `--line`, `--line-2`; `--accent` | Dark is the default. Light follows `prefers-color-scheme`. Every text role meets WCAG AA (≥ 4.5:1) on every background role in both themes. The old `--text-3` measured 3.3–4.2:1 and was raised. |
| Semantic state | `.tone-ok`, `.tone-warn`, `.tone-bad`, `.tone-idle`, `.tone-active` set `--tone` and `--tone-soft` | Shown through one vocabulary: `.badge`, `.state-word`, `.dot`, `.msg.{ok,warn,bad}`, `.last` (action outcome), `.attention` / `.alert`. The same state never gets a different treatment on another surface. A state is also carried by the shape of its glyph (`--glyph-clip`: ok and active a disc, attention a triangle, failure a diamond, idle a hollow ring) and always by its word, never by color alone. |
| Focus | `--focus` (2px accent outline), `--focus-ring` | Every interactive element has `:focus-visible`. Under Windows contrast themes (`forced-colors: active`), focus becomes a `Highlight` outline. States carried by fills get borders. |
| Motion | `--dur-fast` .12s, `--dur` .2s, `--ease` | See §2. |

**Composition before containers.** The page is the canvas. Hierarchy comes from type, spacing, alignment and hairline rules, in this order of operational weight:

1. Primary operational state: readiness or failure, the current model, critical actions (`.state-word`, `.actions`).
2. Technical state: provider, device, runtime identity, resident models, lifecycle (`.spec`, `table.data`).
3. Telemetry: latency, requests, errors, memory, transport (`.metrics`).

Shared primitives (`internal/ui/system.css`):

| Primitive | Use |
|---|---|
| `.section` | A group on the canvas: a 1px `--line` rule, then its content. Every headed region (`.region.section`), Experiments run, Workbench question and first-run step is one. It has no fill, radius, shadow or state edge. |
| `.spec` | Label/value facts in a column grid (`repeat(auto-fit, minmax(min(100%, var(--col-min)), 1fr))`), so a column never collapses below a readable width. Technical state on any workspace. |
| `.id` | A machine identity (model, runtime, digest, endpoint): one monospace treatment, wrapped at the container edge, never given a width of its own. Prose and labels stay in the UI typeface. |
| `.metrics` / `.metric` | Telemetry on one column grid so values align across rows. Numbers are tabular. |
| `.items > .item` | A list of like objects: rows divided by rules. |
| `.msg`, `.last`, `.attention` | State notices. They color their label, text and rule by state; they never fill the surface. |
| `.btn`, `.btn.primary`, `.btn.danger` | Secondary, primary and destructive actions. `.danger` is one outlined red control, also when `.quiet`; it is never a bare colored word. |
| `input`, `textarea`, `select` | One entry-control box and one family of states: hover, `:focus-visible`, `::placeholder`, `[readonly]` (flat, dashed), `:disabled`, `[aria-invalid]` (double red rule, with `.field-error` words), `.optional`. A `select` is native and reads as a selection through a raised face and a pointer. `.field` holds one control, `.fields` a grid of them, `.rows` a repeatable option list. |

Forbidden: a card (rounded, filled or shadowed container) for information, a decorative edge accent on a container, glow, gradient, glass, or ornamental animation. Strong color is reserved for state and for the one primary action. The Runtime workspace is the reference composition (`TestRuntimeReferenceComposition`); Workbench, Experiments, Evidence, Settings and Diagnostics use the same primitives (`TestWorkspacesComposeWithoutCardChrome`). Any later workspace UI, such as resident selection, uses these primitives rather than a new container style.

**Navigation.** The selected item is a 2px `--accent` rule (bottom edge in the narrow strip), full text color and heavier weight. It has no fill, radius or shadow. Hover shows `--raised`; keyboard focus shows the shared focus outline; `aria-current="page"` carries the state for assistive technology and contrast themes.

**Progressive disclosure.** A workspace shows the operator's decision or action first: readiness, the result, failure counts, the run state. Evidence that explains it comes next. Raw diagnostics (provider info, stderr, argv, request JSON, evidence identity) sit in `details.disclosure`, one keyboard step away. They are never removed. Disclosures marked `data-keep` keep the operator's open/closed state across live refreshes.

## 2. Interaction system

| Behavior | Contract |
|---|---|
| Loading / progressive hydration | Every workspace is a complete server-rendered document. Live regions (`[data-live]` slots) refresh from `/live` every 3 s and Experiments progress from `/experiments/live` every 1.5 s. A slot is replaced only when its content changed: no relayout and no repeated screen-reader announcement. Nothing is fetched while the window is hidden (`document.hidden`). Showing the window refreshes immediately. **Operator-local state.** Replacement keeps only presentation the operator chose: `uistate.js` records a disclosure's open or closed state (on the click or key that toggles its summary, never from a rendered default), a console's follow preference and its scroll, keyed by `data-keep` else `id`, and applies them to the incoming markup before it replaces the old. Every disclosure therefore carries a stable `id` or `data-keep`; content stays server-authoritative, and nothing is persisted across a page load. |
| Pending / working / completed / failed | A submitted form and its submit control get `aria-busy="true"`. The control shows a progress spinner (static under reduced motion), and the form cannot be submitted twice. The Workbench result pane dims while a Run is pending (`data-busy`). Completed and failed actions render as one `.last` outcome (`role="status"`). Inline failures render as `.msg.bad` (`role="alert"`). |
| Destructive / irreversible | Stop and Disconnect are `.btn.danger` and act immediately (reversible). Deleting stored data asks first through the shell's one handler (`data-confirm`): removing a runtime or model, removing a saved connection, deleting a history entry. |
| Inline validation | The server validates (the Go authority). A rejected form keeps what was typed and reports the reason in one alert next to the form. Native `required` marks mandatory fields. |
| Long-running operations | Experiments show requests done of total in a `role="progressbar"`. Doctor shows a running badge. Model/runtime maintenance runs in the background and refreshes only its Settings section in place, so focus and the rest of Settings stay put; it shows the phase list, the current step and, only when the step has a total, a determinate `role="progressbar"` with bytes (an indeterminate bar otherwise), and keeps its outcome (DONE, or FAILED with phase and step) visible until the next action. The worker's startup phases are shown the same way on Runtime. First run announces each new phase once (`role="status"`). The workspace that owns an operation (Models: setup, materialize, repair, activate, verify, remove, desired state; Forge: build, certify, probe, preflight, apply) shows its full phases and progress while it runs, and a console: a closed-by-default disclosure with the bounded, redacted recent tail of the operation's own setup-log section (the same tail, line bound and scrubbing a Forge diagnostic uses; `app.ConsoleTail`) and the age of the latest real backend activity (the last reported phase or step, or the last write of that log), never a percentage the backend did not report. Every other workspace shows only a one-line headline in the shell. When the operation ends, the owning workspace collapses it to one compact summary (action, DONE or FAILED, target, elapsed time, what it accomplished) with the phases and the console kept in a closed `Details · Phases and output` disclosure; other workspaces show the summary and a link. |
| Notifications / status changes | The live/stale sync state is announced only when it flips, never on each tick. Tray notifications stay native (`docs/runtime.md`). |
| Workspace transitions | Full navigations on one origin. The skip link moves to the workspace. On first run, each new step moves focus to its heading. |
| Motion | Allowed only to explain a state transition, causality, selection or completion: the progress bar advancing, a disclosure chevron turning, work in progress (`.tone-active` pulse, busy spinner). Warnings and failures hold still. `prefers-reduced-motion: reduce` removes all animation and transitions. There is no decorative motion, glass or blur. |

## 3. Accessibility

- **Keyboard.** All primary workflows use native links, buttons, form controls and `details`/`summary`, so they are keyboard-operable in document order. Evidence rows are inspected through a link in their first cell.
- **Visible focus.** See §1.
- **Semantic HTML and accessible names.** Landmarks (`header`, `nav`, `main`, `footer`), labeled regions, `fieldset`/`legend` for grouped choices, and table headers (including a visually hidden "Actions" header). Accessible names go through the catalog, so Japanese operators hear Japanese names. Glyph-only marks (✗) have a role and a name.
- **Contrast.** See §1.
- **Display and text scaling.** The window is per-monitor DPI aware (#88/#95). Layout reflows at 1180px (single column) and 820px (navigation becomes a strip). Text sizes follow the operator's preference. No zoom factor is set anywhere (`TestNoWebViewZoom`).
- **Reduced motion and contrast themes.** See §1 and §2.

## 4. Windows-native integration

These stay as specified in `docs/runtime.md`:

- one window
- per-monitor DPI
- native file and folder pickers (#93/#97)
- tray, close-to-tray and start at sign-in
- light/dark following the OS

This contract adds one behavior: the shell sets WebView2 `IsVisible=false` when the window is hidden to the tray or minimized, and `true` when it is shown. The page then reports `document.hidden`, stops its live polling, and WebView2 may throttle rendering. `ShowWindow` alone does not tell WebView2 this.

The WebView2 profile lives under `HACHIDORI_HOME/cache/webview2`. During first run, before a home exists, it lives in a throwaway temporary folder. It is reused across launches.

Settings > Updates is a view over `internal/update` ([runtime.md](runtime.md#updates-windows-executable)): it renders local state and forwards four explicit actions (save channel, Check for updates, Download, Restart & update). Opening it, opening Settings and changing the channel use no network; its progress panel is the shared long-running operation view. Check for updates and Download are accepted at once and run in the background; while one runs the other update actions are disabled and refused, and the panel shows its phase, byte progress (only when the size is known) and, on failure, what was left behind. Restart & update hands replacement to a separate helper process after the window closes; the window owns no replacement logic.

WSL is not a UI or runtime target.

## 5. Performance budgets and measurements

Perceived performance is part of the contract.

**What the startup path does.** It does no runtime discovery or probing before the window:

- preflight: WebView2 version and the single-instance guard
- home discovery, which reads the bootstrap locator
- the activation check: three small JSON reads
- controller start, which spawns the worker supervisor goroutine and returns at once
- the loopback listeners
- window creation

CUDA and provider probing happen inside the worker process, after the window exists; the UI shows `starting` until the worker is ready. The model inventory is read only when Settings renders. History is read only when Experiments renders.

**Startup marks.** The desktop logs startup milestones to stderr as `+Nms since entry`. The epoch is `desktop.Epoch`, taken as the first statement of `main` and passed explicitly to the composition and the window. It is executable entry, not OS process creation. The OS loader and Go runtime initialization before `main` are not measured, and the budgets below are defined from entry. The marks are:

- `hachidori: startup shell composed +Nms`
- `native window shown`
- `WebView2 controller ready`
- `first navigation completed`

A physical Windows run records the startup budgets from these marks, with no profiler.

**Reproducing the measurements.** Server-side measurements are reproducible with:

- `go test ./internal/dashboard -run XXX -bench 'Workspaces|EvidenceRecords|HistoryRecords'`
- `go test ./cmd/hachidori -run XXX -bench LaunchToWindow`

| Budget | Target | Evidence (reference: i5-14600KF, Linux, Go 1.26; server/Go side only) |
|---|---|---|
| Executable entry → first visible native window | ≤ 1 s cold, ≤ 500 ms warm | Go composition to the window request: **0.13 ms** (`BenchmarkLaunchToWindow`, fake WebView2/runtime). The WebView2 part is **NOT_CHECKED**: it needs physical Windows (startup marks). |
| Executable entry → first meaningful shell | ≤ 1.5 s cold | Runtime document render **0.18 ms**, 36 KB. WebView2 navigation is **NOT_CHECKED**. |
| Shell → interactive navigation / workspace switch | ≤ 300 ms | Workspace documents render in 0.1–0.25 ms (31–42 KB). WebView2 parse/paint is **NOT_CHECKED**. |
| Warm launch | ≤ 500 ms to a shown window | A second launch activates the running instance (no new runtime). **NOT_CHECKED** on Windows. |
| Runtime refresh | ≤ 100 ms server per poll | `/live` **0.19 ms**, 8 KB. The client now skips unchanged slots and hidden windows. |
| Question submit → visible feedback | ≤ 100 ms | Feedback is client-side and immediate on submit: busy control and dimmed result pane, with no round trip. Inference time is #94's scope, not this contract's. |
| Idle CPU / memory | No periodic work while hidden; ≤ 1 request per 3 s while shown | Hidden windows no longer poll; shown windows poll at an unchanged rate. CPU and memory figures are **NOT_CHECKED** (physical Windows). |
| Evidence view, 100 / 1,000 / 10,000 observations | ≤ 50 ms server, ≤ 500 KB | 1.7 / 13.7 / 14.5 ms, 74 / 362 / 362 KB. Already bounded to 1,000 rendered rows (export includes all). |
| History view, 100 / 1,000 / 10,000 entries | ≤ 150 ms server, ≤ 2.5 MB | See below. |

History view measurements, before → after:

| Entries | Before | After |
|---|---|---|
| 100 | 2.6 ms, 148 KB | 2.7 ms, 160 KB |
| 1,000 | 24 ms, 1.2 MB | 12.7 ms, 344 KB |
| 10,000 | 268 ms, 11.8 MB | 119 ms, 2.2 MB |

The history table now renders 100 rows per page (Newer/Older links). Compare still lists every entry.

Headless Chromium 154 loaded the 10,000-entry history document as a proxy for WebView2, which uses the same engine. First frame after `load` (median of 3) went from **3.68 s to 0.72 s**. WebView2 itself is **NOT_CHECKED**.

### Recorded follow-ups (measured, not fixed here)

- **`history.Store.Save` rescans every entry to rebuild its index.** Saving is O(n), so seeding n saves is O(n²): seeding 10,000 saves did not finish in 9 minutes. It belongs to the history store, not the desktop UI.
- **The remaining ~2.2 MB at 10,000 history entries is the two Compare `<select>`s.** They deliberately keep every entry comparable. Bounding them needs a Compare interaction change.
- **`List` decodes the whole history index on every Experiments render** (about 100 ms at 10,000). A cached index would remove it.
