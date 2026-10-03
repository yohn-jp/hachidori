# Golden-path certification

Certification levels are reported separately. An environment that cannot run a
level reports it as **blocked / not checked**, never as passed.

| level | command | needs |
|---|---|---|
| unit / focused | `go vet ./... && go test -race ./...` | Go toolchain (the Python protocol test also uses a host `python3` when present, with stub `torch`/`laya`/`opendecider`) |
| integration (real worker, real Laya) | `HACHIDORI_CERT_HOME=<home> go test ./internal/doctor -run TestCertifyRealProvider -v` | materialized home |
| real Laya provider | `hachidori doctor --home <home>` | materialized home |
| CUDA | same two commands with a home set up by `--device cuda` | NVIDIA GPU + driver ≥ 570 |
| benchmark | `hachidori benchmark --out report.json <coding-agent.jsonl>` | running `serve`, the external benchmark dataset |

## Windows RTX host (PowerShell)

For dogfood, use the development prerelease assets from
https://github.com/yohn-jp/hachidori/releases:

- `hachidori-windows-amd64.exe`
- `hachidori-windows-amd64.exe.sha256`

```powershell
# 1. verify the downloaded development executable.
$expected = (Get-Content .\hachidori-windows-amd64.exe.sha256).Split()[0].ToLowerInvariant()
$actual = (Get-FileHash .\hachidori-windows-amd64.exe -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Hachidori SHA-256 mismatch" }
Copy-Item .\hachidori-windows-amd64.exe .\hachidori.exe

# Alternatively, build the same entry point locally:
# go build -o hachidori.exe ./cmd/hachidori

# 2. materialize runtime + model (network needed only here). Use a clean
#    home; no system Python, pip or uv is needed (Hachidori bootstraps its
#    pinned uv into $env:HACHIDORI_HOME\tools\uv\0.12.19\uv.exe).
$env:HACHIDORI_HOME = "D:\Hachidori-test"
.\hachidori.exe setup --device cuda
# expect: "private uv 0.12.19 ready", uv python install / venv / sync lines,
#         "runtime cu128-<digest> published", "active: runtime=cu128-<digest> …"

# 3. doctor: home, runtime, model digests, isolation, provider import,
#    CUDA, worker start/load/warmup, real HTTP smoke inference
.\hachidori.exe doctor

# 4. integration certification against the real worker on CUDA
$env:HACHIDORI_CERT_HOME = $env:HACHIDORI_HOME
go test ./internal/doctor -run TestCertifyRealProvider -v -count=1

# 5. serve (loopback only). To show serving does not use uv or package
#    resolution, first move the private uv away and cut the network, e.g.
#    Rename-Item $env:HACHIDORI_HOME\tools\uv\0.12.19\uv.exe uv.exe.off
.\hachidori.exe serve
```

In another shell on the host, or from the NixOS VM through an SSH local
forward (`ssh -N -L 7843:127.0.0.1:7843 <windows-host>`):

```sh
export HACHIDORI_ENDPOINT=http://127.0.0.1:7843
hachidori status                      # runtime.model_id=laya-base, worker.state=ready, provider.device=cuda, provider.device_name, accelerator memory
hachidori decide examples/decide.json # repeat: worker.pid and worker.starts in status stay unchanged
hachidori benchmark --warmup 5 --passes 3 --out coding-agent-report.json <coding-agent.jsonl>
```

Evidence to record: `doctor` output, the integration test log (device, load and
warmup ms, warm p50/p95, same pid over 20 requests), `status` before and after
the benchmark (pid/starts unchanged, request counters), and the benchmark report
compared with the architecture §12 baseline (accuracy ≈ 0.8333, ECE ≈ 0.0784,
mean confidence ≈ 0.8215, p50 ≈ 25.3 ms, p95 ≈ 30.5 ms).

Note: the §12 baseline was measured outside Hachidori; its checkpoint and
question wording are not recorded in this repository. If the benchmark uses a
different checkpoint than the pinned `convaiinnovations/laya` revision, the
accuracy comparison is not like-for-like.

## Decision-model comparison (#113)

`laya-base` is the default model. `opendecider-nano` is a candidate that is
selectable and certifiable, and becomes the default only if the evidence below
supports it for Hachidori's own workload. Model size, upstream benchmark numbers
and upstream latency claims are not evidence here; neither are portable tests,
which use stubs and never download the real model.

Run the **same** fixed corpus and Question Definitions against each model on the
same host, with the same device and the same `hachidori.exe`, one model active at
a time (activate, **Restart runtime**, confirm `runtime.model_id` and
`provider.provider`/`provider.dtype`/`provider.device` in `hachidori status`):

```powershell
.\hachidori.exe setup --device cuda --model laya-base          # or Materialize + Activate in Models
.\hachidori.exe serve                                          # another shell:
hachidori status                                               # model_id, provider, device, dtype, load_ms, warmup_ms
hachidori benchmark --questions <defs> --warmup 5 --passes 5 --out laya-base.json <corpus.jsonl>
# stop serve, then the same with: setup --device cuda --model opendecider-nano
hachidori benchmark --questions <defs> --warmup 5 --passes 5 --out opendecider-nano.json <corpus.jsonl>
```

The corpus and definitions must be identical (same dataset SHA-256 and question
identities in both reports; `internal/eval` compares only like with like and
refuses a mismatch). Record, per model:

| measure | where it comes from |
|---|---|
| accuracy, mean confidence, ECE (15 bins) | the benchmark report |
| calibration beyond ECE (Brier, NLL) | not produced by the single-model benchmark report; use the resident comparison below, or compute it from the report's per-observation probabilities and expected labels, or record "not available" |
| high-confidence errors | the Errors workspace over the report (outcome `high_confidence_wrong` at a stated confidence threshold) |
| p50 / p95 latency | the benchmark report (server inference and client round trip) |
| cold load and warmup time | `hachidori status`, `provider.load_ms` and `provider.warmup_ms` of a freshly started worker |
| resident and peak VRAM | `hachidori status` accelerator `memory_allocated`/`memory_reserved` after warmup (resident) and after the benchmark (peak is the highest value seen; note the sampling) |
| state truncation | for OpenDecider-nano, the number of `state truncated` lines in `logs/worker.log` for the corpus (its context is 2,048 tokens; Laya's differs) |

Rules for a default change: it needs a recorded comparison on the target
accelerator (the RTX 3060 for the Windows host) in which the candidate is at
least as accurate and as well calibrated with no more high-confidence errors on
the fixed corpus, with latency, load/warmup and VRAM reported alongside. Without
that record the default stays `laya-base`. A failure of the candidate to load,
warm up or stay on the requested CUDA device is a `FAIL`, never a fallback to CPU.

### Resident comparison without reload (#123)

When both models are resident at once (multi-resident serving, see
[runtime.md](runtime.md)), one command evaluates them on the same dataset and
Question Definitions while both stay loaded; no activation, restart or reload
happens between the model passes:

```powershell
hachidori benchmark --questions <defs> --models laya-base,opendecider-nano --warmup 5 --passes 5 --out comparison.json <corpus.jsonl>
# optional declared controls (defaults in parentheses):
#   --high-confidence 0.9  --thresholds 0.5,0.6,0.7,0.8,0.9,0.95,0.99
#   --length-edges 256,512,1024,2048,4096,8192,16384  --family question_id=family
```

`eval` takes the same flags (one pass, no warmup). Without `--models` both
commands are unchanged and write ordinary `hachidori.evidence.v1` evidence.

Every model is targeted directly by its catalog ID (`model` on the decide
request); a model that is not running or not ready stops the run before any
request, and an answer whose `served` provenance is not the targeted model is an
error that is never scored. The report is `hachidori.resident-comparison.v1`, a
separate document: v1 evidence and history entries are unchanged and
`ModelRun.Evidence` turns any run into a plain v1 report.

What it records, per model (from the report, no manual computation):

| measure | field |
|---|---|
| identity: catalog model, provider, revision, runtime, digest | `model`, `provider`, `identity` (the resident's own status), and `served_model`/`served_provider`/`identity_sha256` on every observation |
| same inputs | `alignment` (`aligned` or `refused` with every difference listed), `input_sha256`, `sent_sha256` (digest of every normalized request actually sent, model selector excluded) |
| accuracy, macro-F1 | `quality.accuracy`, `quality.macro_f1` (mean F1 over (question, label) classes) |
| ECE, Brier, NLL | `quality.calibration` (15 bins; Brier is the multi-class sum of squared errors; NLL clips p at 1e-15). Null, never 0, when undefined; `probability_undefined` counts the observations they skip |
| high-confidence errors | `quality.high_confidence` at the declared threshold (wrong and confidence >= threshold): count, rate of all observations, rate of high-confidence answers |
| threshold x coverage x conditional accuracy | `quality.thresholds` |
| per question, per family | `per_question`, `per_family` (families are declared with `--family`; others are `unassigned`) |
| latency by input length | `length_buckets`: cases, requests, errors, accuracy and request/inference p50/p95 per range of state characters; every bucket is present for every model |
| p50 / p95 | `request_latency` (client round trip) and `inference_latency` (server), warm requests of all passes, warmup excluded |
| cold load and warmup | `startup.load_ms`, `startup.warmup_ms` from the resident status, read before the first request; never part of request latency |
| resident and peak accelerator memory | `memory`: samples before the run, after warmup, after every pass and after all runs; `resident_*` is the after-warmup reading, `peak_*` the highest sampled value (a sampled maximum, not continuous). Absent, not zero, for a resident without accelerator statistics |
| request and error counts | `requests`, `succeeded`, `error_count`, `errors_by_class`, `errors` |
| no reload | `resident_stable` per model and `residents_stable` overall: identity, PID, worker start count, load/warmup timing unchanged and uptime not reset between the status read before the first request and the one after the last |

The input-length buckets are how the long-input behaviour of OpenDecider-nano
(its context is 2,048 tokens) is measured reproducibly: put long and short
states in the same corpus and read latency and accuracy per bucket. The edges
are state characters, not tokens; token counts and `state truncated` log lines
remain provider-internal and are still read from `logs/worker.log`.

The comparison states measurements only. It produces no winner, ranking, pass or
fail, and a default-model change still needs the recorded decision above.

### OpenDecider-nano FP32 vs BF16 (#136)

OpenDecider-nano runs in `float32` (what upstream evaluated). `bfloat16` is an
explicit, opt-in evaluation control; `float32` stays the default until the record
below supports changing it. Nothing else changes: the same pinned checkpoint,
typed closed-choice scoring, explicit CUDA placement, no generation, parsing or
retry. A resident whose encoder is not in the dtype that was asked for fails to
load (`model_load`); there is no silent dtype fallback, and no CPU fallback.

Select the dtype for one runtime launch with `HACHIDORI_OPENDECIDER_DTYPE`
(`float32` or `bfloat16`; anything else refuses the launch; it applies only to an
OpenDecider resident and is never persisted). `hachidori status` reports the
dtype the resident is actually on (`provider.dtype`: `torch.float32` or
`torch.bfloat16`) and the resident comparison records it in each run's identity.

```powershell
# 1. baseline: FP32 (variable unset), both models resident, fixed corpus and questions
hachidori benchmark --questions <defs> --models laya-base,opendecider-nano --warmup 5 --passes 5 --out fp32.json <corpus.jsonl>
# 2. stop serve; set the variable; start serve again (cold load and warmup are measured on a fresh worker)
$env:HACHIDORI_OPENDECIDER_DTYPE = "bfloat16"
hachidori status                                    # provider.dtype = torch.bfloat16 for opendecider-nano, device cuda:0
hachidori benchmark --questions <defs> --models laya-base,opendecider-nano --warmup 5 --passes 5 --out bf16.json <corpus.jsonl>
# 3. pair the two runs of OpenDecider-nano (reads files only)
hachidori precision -model opendecider-nano -out fp32-vs-bf16.json fp32.json bf16.json
```

`precision` refuses (listing every reason) unless both runs are the same model,
revision, provider, runtime, device and torch build, differ only in dtype, and
received the same dataset, question identities and normalized requests with the
same declared controls. It never writes a verdict. It reports:

| measure | field |
|---|---|
| choice flips, each one with expected label, both choices and confidences, and whether it `fixed`, `broke` or left both wrong | `choice_flips`, `flip_rate`, `flips[]` |
| probability deltas (per observation, the largest absolute change over its options: mean, p50, p95, max) | `probability_abs_delta` |
| confidence deltas (signed mean and absolute spread) | `confidence_delta` |
| accuracy, macro-F1, ECE, Brier, NLL, high-confidence errors (candidate minus baseline; both runs' own values under `baseline.quality` and `candidate.quality`) | `quality_delta` |
| p50 / p95 request and inference latency as candidate/baseline ratios, overall and per input-length bucket | `request_latency`, `inference_latency`, `length_buckets[]` |
| resident and peak accelerator memory ratios | `memory` |
| cold load and warmup ratios | `load_ms`, `warmup_ms` |
| requests, errors by class, resident stability | `baseline`/`candidate`: `requests`, `error_count`, `errors_by_class`, `resident_stable` |

Input-length scaling is the per-bucket table (a corpus with short and long
states). Question-count scaling is a series: repeat steps 1-3 on corpora whose
Question Definition sets have 1, 2, 4 and more questions per case and tabulate
each report's `question_count` against the ratios.

Rules for changing the default dtype to `bfloat16` (all of them, on the target
accelerator, here the RTX 3060): zero `broken` flips or a flip rate and
accuracy/ECE/Brier/high-confidence-error change the owner accepts and records,
a meaningful latency or memory ratio, no more request errors, residents stable,
and the report archived with the pull request that changes the default. Without
that record the default stays `float32`.

#### Recorded state (#136)

| item | outcome | note |
|---|---|---|
| portable tests: dtype launch control, worker dtype verification and no-fallback, paired comparison | see the pull request | stubs and fixtures only; no real model is loaded |
| real OpenDecider-nano FP32 vs BF16 on CUDA, fixed corpus (flips, probability/confidence deltas, accuracy, ECE, Brier, latency, scaling, memory, load/warmup, errors) | NOT_CHECKED | no GPU, torch or corpus in the authoring environment; nothing is inferred from CPU, upstream claims or other models |
| physical Windows, RTX 3060, `bfloat16` OpenDecider-nano | NOT_CHECKED | user-side; never inferred from CI |
| default dtype | unchanged: `float32` | no recorded evidence supports a change |

### System One variant certification (#142)

A System One variant (a derived, quantized execution artifact of a pinned
model, runtime.md "System One variants") is certified by one question: **how
much did compression change the typed decision model?** It is answered in the
space Hachidori serves (the typed choice decisions), not by perplexity, and it
needs no labels. The reference is the pinned source model at high precision
(`float32` or `bfloat16`, on the CPU if that is what fits, however slowly); the
candidate is the exact variant. Reference fidelity and labelled correctness are
different things: the reference is the authority for *fidelity only* and is
never treated as ground truth. The core evaluator states evidence and never
names a "best" model; a policy turns evidence into `accepted` or `rejected`
against explicit thresholds and changes none of it.

The evaluator is `internal/eval` (`certify*.go`, `resident_run.go`): it reuses
the resident-comparison machinery (direct resident targeting, the status
identity, alignment, `QualityOf`, the calibration and slice formulas) and adds
only what pairing a reference with a variant needs.

Before the expensive steps, check readiness and smoke-test what was built
(runtime.md "System One Forge readiness"). None of these commands certifies
anything; an interrupted source download resumes by repeating `setup`; a failed
step leaves a diagnostic (`hachidori forge diagnostics show`):

```powershell
hachidori forge preflight materialize --device cpu                 # before the ~19 GB download
hachidori forge preflight optimize                                 # before the build
hachidori forge preflight probe --variant <variant-id> --device cuda
hachidori forge probe --device cuda <variant-id>                   # the persisted variant loads and answers one typed decision
hachidori forge preflight certify --variant <variant-id> --device cpu [--reference-dtype float32]   # RAM lower bound of the reference run
```

A passing probe says only that the exact persisted variant loaded and produced
one valid typed decision on that device; its latency, RAM and VRAM figures are
single observations, not certification evidence, and `NOT_CHECKED` stays the
state of every physical claim until the run happens on the workstation.

```powershell
# 0. a corpus: eval JSONL; "expected" labels are optional (all questions or none)
# 1. reference: serve the source on the CPU in high precision, record the run
$env:HACHIDORI_CLEF_DTYPE = "float32"        # or leave unset for bfloat16, what the release ships
hachidori setup --device cpu --model clef-flash
hachidori serve                              # slow load and slow requests are expected
hachidori certify run --model clef-flash --questions <defs> --warmup 5 --passes 3 --out reference.json <corpus.jsonl>   # the same --warmup and --passes as the candidate run

# 2. build the variant, then record the candidate run on the variant
Remove-Item Env:HACHIDORI_CLEF_DTYPE
hachidori variant optimize --model clef-flash
hachidori setup --device cuda --model clef-flash
hachidori activate --device cuda --model clef-flash --variant <variant-id> --experimental   # uncertified candidates run only as experimental
hachidori serve
hachidori certify run --model clef-flash --questions <defs> --warmup 5 --passes 3 --out candidate.json <corpus.jsonl>

# 3. certify: identical normalized inputs, the exact variant, an explicit policy
hachidori certify evaluate --variant <variant-id> --reference reference.json --candidate candidate.json [--policy policy.json] [--out certification.json]
hachidori certify show <variant-id>          # add -json for the full report

# 4. activate for real (the experimental mark is cleared), restart, status, decide
hachidori activate --device cuda --model clef-flash --variant <variant-id>
```

An accepted variant is applied with one explicit operation instead of
activate, restart, inspect and smoke by hand: `hachidori variant apply --device
cuda <variant-id>` (`Controller.ApplyCertifiedVariant`). It refuses an
uncertified, rejected, ambiguous or stale variant before anything changes, then
activates, rebinds the runtime, waits for READY, proves from the worker and the
status that the exact quantized variant executes on the requested device, and
answers one typed decision through the serving path; any failure restores the
previous activation and running configuration and verifies it (see runtime.md,
Certified variant apply). An accepted certification is necessary, never
sufficient: certification does not apply a variant, and the serving proof is the
final gate. Physical Windows/CUDA/Clef behavior remains NOT_CHECKED until the
operator runs it.

`hachidori forge execute` produces the same `ResidentRun` without a resident:
it starts the exact source or the exact persisted variant on an explicit device
as temporary maintenance work (no experimental activation, no certification
needed), proves from the READY worker what executed, runs the dataset and keeps
the run under the home as `hachidori.forge-run.v1`, bound to the source and
variant identities, runtime, device, dtype, dataset and Question Definitions,
returning an evidence ID instead of a path (see runtime.md, Forge execution
sessions). It decides nothing: certification stays the comparison below.

**The normal path is one command.** `hachidori forge certify --device cuda <variant-id> <corpus.jsonl>`
(with `--questions`, `--policy`, `--reference-device`, `--reference-dtype`,
`--materialize`) does steps 1 to 3 above and nothing else: it resolves the exact
source, variant, policy and corpus; runs the preflights and, if allowed,
materializes a missing runtime; probes the variant; executes the pinned source
(reference, default `cpu`, `bfloat16`) and the exact variant (candidate, the
explicit device) as `forge execute` does; proves from the stored evidence that
each is the exact target; proves the two runs are the same normalized dataset,
Question Definitions, questions, requests and case order; calls `eval.Certify`
with the unchanged policy; and records the certification. No run file is named,
no experimental activation is needed and nothing is activated, whatever the
verdict. The report and its record carry a versioned `producer` block
(`hachidori.certification-producer.v1`): the evidence ID and digest of the
reference and candidate runs, their exact targets (source and variant identity,
runtime, requested and actual device and dtype, quantized execution) and the
outcomes of the preflight and the probe. Activation re-verifies that the record's
and the report's producer blocks name the same runs; certifications of run files
and records written before the block existed carry none and stay valid. A failure
leaves a diagnostic naming its phase (`resolving`, `preflight`, `materializing`,
`probe`, `reference_run`, `candidate_run`, `aligning`, `certifying`,
`persisting`) and the evidence of any run that did complete, and never a
certification. The reference remains the authority for fidelity only. The rest of
this section describes the run-file level, which stays as the low-level surface.

The desktop's Forge workspace is the same path as a form: it asks for the
dataset, the Question Definitions, the optional policy and the devices, calls
`Controller.CertifyVariant`, and has no reference or candidate run input.
Accepted does not apply: the explicit Apply of the same workspace calls the
controller's apply transaction (`StartApply`, the same transaction as
`ApplyCertifiedVariant`). Experimental activation is a separate, collapsed
Advanced action and is never part of this path (runtime.md, Models and Forge
workspaces).

`certify run` records one resident's pass as `hachidori.resident-run.v1` (the
same per-model evidence a resident comparison records: identity from the
resident's own status, startup load and warmup, accelerator and host memory
samples, latency, every observation with its input digest, errors). The two
runs need not exist at the same time. `certify evaluate` **refuses**, listing
every reason and computing no delta, unless the runs are the pinned source
model at the pinned revision at high precision and the exact variant (its
`variant_id`, and the dtype its manifest declares), received the same dataset,
question identities, normalized requests and requests actually sent in the same
order, with the same declared controls, and are both labelled or both not.

The report (`hachidori.system-one-certification/1`) binds: the source identity;
the variant ID, manifest digest, build ID, recipe and engine; the reference and
candidate execution identities (model, provider, revision, device, dtype,
quantized execution, status-identity digest, stability); the dataset digest,
question identities and normalized-request digest; the policy, embedded whole,
with its digest. It reports:

| evidence | fields |
|---|---|
| **fidelity** (no labels needed) | `choice_flips`, `flip_rate`, `top_choice_preserved_rate`; per-question transition matrix (`transitions`); each flip (`flips[]`); confidence delta (signed mean, spread); probability deltas per observation (mean over options and max over options, mean/p50/p95/max); Jensen-Shannon divergence per observation in bits (bounded by 1); ranking preserved; high-confidence reference decisions and how many flipped; per-question fidelity; question types (only `choice` is served by `hachidori.v1`, so noul and score drift are stated as not measurable, not reported as zero) |
| **labelled quality** (only with labels) | accuracy, macro-F1, Brier, ECE, NLL and high-confidence-error deltas (candidate minus reference, from each run's own `QualityOf`), threshold x coverage x conditional-accuracy deltas, per-question and per-family slices, and whether each flip fixed, broke or left both wrong |
| **resources** | load and warmup, request and inference p50/p95 and their ratios, accelerator memory (resident and peak) and host RAM of the worker where it reported them, request and error counts, and per fact `MEASURED` or `NOT_CHECKED` (`checks`): nothing is estimated, and a CPU resident reports no VRAM |

The default policy `system-one-fidelity/1` (`eval.DefaultPolicy`; a policy file,
`hachidori.certification-policy/1`, replaces it, and any threshold change is a
new versioned ID and a new digest): at least 100 paired observations; no
unpaired observation and no request error; both residents stable; flip rate at
most 0.03; high-confidence (>= 0.9) reference decisions flipping at most 0.01;
mean JS at most 0.01 bits; p95 of the largest probability change at most 0.10
and the maximum at most 0.50; and, for labelled runs only, accuracy at most 0.02
lower, macro-F1 at most 0.03 lower, ECE and Brier at most 0.03 higher, NLL at most
0.10 higher and the high-confidence error rate at most 0.01 higher. Boundaries
are inclusive. These are starting thresholds for the first physical
certification, not a claim about Clef-Flash; the operator records the policy
they accept. A criterion whose evidence does not exist (labelled criteria without
labels) is reported as not applied and neither passes nor fails.

`certify evaluate` writes the report and then its record under
`state/certifications/<variant-id>/` for accepted and rejected alike (a
rejecting verdict exits non-zero but the evidence is kept whole). Activation
trusts a record only after re-verifying it: it names this variant by ID and
manifest digest and source, its report hashes to the recorded digest and binds
the same identities, the report's policy digests to the recorded one, and the
verdict recomputed from that policy and the report's evidence is the recorded
verdict, so a hand-edited verdict or report never certifies a variant. The latest
valid record decides; a later rejecting record withdraws an earlier acceptance.
"Latest" is a persisted total order, not a clock: each record
(`hachidori.certification-record/2`) carries the variant's next `sequence`,
claimed exclusively on disk (`<sequence>.order`) before the record is written,
so a certification written in the same second as an earlier one still
supersedes it, and neither the report digest, the record's file name nor the
order a directory is listed in ever decides. Records written before sequences
existed (`hachidori.certification-record/1`) stay readable and are older than
every sequenced record; among themselves only a strictly later `created_at`
orders them, and two of them created within the same second resolve to
`ambiguous`: neither accepted nor uncertified, refused for activation (even
experimental) until the variant is certified again.

Historical evidence is unchanged: `hachidori.evidence.v1`,
`hachidori.resident-comparison.v1` and `hachidori.precision-comparison.v1` keep
their schemas and readers; the only additions are optional `host_rss_bytes`
fields in memory samples and `hachidori.resident-run.v1` (new).

#### Recorded state (#142)

| item | outcome | note |
|---|---|---|
| portable tests: variant identity and manifest contract, builder failure/cleanup/reproduction, activation gating, launch provenance and no-fallback, fidelity metrics, policy boundaries, record verification, controller and Settings projection, CLI workflow | see the pull request | fakes and tiny fixtures only; no real model is downloaded, no multi-GB optimization runs |
| real LLM Compressor build and real `clef` adapter on a tiny random model with the Clef-Flash layout (`go test ./internal/optimize -run TinyClefEndToEnd` with `HACHIDORI_TEST_OPTIMIZER_ENV` and `HACHIDORI_TEST_JOINT_SCHEMA`): build, byte-identical reproduction, packed int4 execution, no CUDA fallback, dtype guard, corrupt-variant refusal | see the pull request | optional, outside normal CI; a tiny random model proves the machinery, not the quality of any real model |
| Clef-Flash BF16 or FP32 reference on the CPU of a 64 GB workstation (fits, succeeds, load time, RAM, latency) | NOT_CHECKED | requires the 19 GB download and the user's machine |
| real Q4 build of Clef-Flash with the canonical recipe (time, output size, determinism on the real model) | NOT_CHECKED | requires the real source and the optimizer runtime |
| Q4 variant on the RTX 3060 (fits, exact VRAM, RAM, load, latency) | NOT_CHECKED | physical; never inferred from CI or the tiny model |
| decision fidelity of the real variant (flips, drift, divergence, labelled deltas) and the policy verdict | NOT_CHECKED | needs the two real runs on the operator's corpus |
| physical Windows desktop: Optimize, Certify, Activate, Restart for a variant | NOT_CHECKED | Windows checklist W20-W24 |
| `forge certify` of the real Clef-Flash W4A16 variant on Windows / CUDA / Clef (probe, BF16 reference, candidate run, certification) | NOT_CHECKED | portable tests use fake workers and fixtures only; no real weights are downloaded; user-side |

### Deterministic resident routing (#124)

With both models resident and a routing policy bound (`serve --resident
opendecider-nano --routing-policy policy.json [--routing-evidence comparison.json]`,
see [runtime.md](runtime.md)), one application can run laya-only, nano-only and
`route: "auto"` calls without a reload of either model. The evidence of a routed
run is in the response and in status, not in a separate report:

| claim | evidence |
|---|---|
| which model produced each final result and why | `routing.results[].served` and `reason` (stable codes) on every response |
| what a handoff replaced | `routing.results[].first_path` (model, choice, confidence) |
| the policy that decided | `routing.policy` `id` and SHA-256; `routing.calibration` (evidence digest, dataset digest, observations per threshold rule) when `--routing-evidence` was given |
| per-provider latency contribution, handoff counts | `routing.providers` per response; `routing` counters in `/v1/status` (`handoffs`, `handoff_failures`, `reasons`, per-resident calls and inference ms) |
| no reload | each resident's `pid` and `starts` in `/v1/status` unchanged across direct and routed calls (the same stability evidence as the resident comparison) |
| required handoff failure is explicit | `routing_failed` with `first_path_failed` / `required_handoff_failed`, no results |

A routing certification records, for a stated policy and corpus: a kept
first-path result that did not wake the alternate (alternate `requests` counter
unchanged), a selective handoff that replaced only the selected results,
alternating direct and routed calls with unchanged PIDs and `starts`, and a
required handoff to a stopped resident failing explicitly while the other
resident keeps serving. Thresholds are justified from the resident comparison
(`per_family`, `quality.thresholds`, `quality.calibration`), not set from a
single universal confidence value.

Portable tests cover all of this with fake residents and real fake-worker
processes and download no model. They are not physical evidence.

### Recorded state for the change that introduced OpenDecider-nano

| item | outcome | note |
|---|---|---|
| portable tests, race, vet, Windows amd64 build/vet | see the pull request | stubs and fixtures only; no real model download in CI |
| real OpenDecider-nano on CPU, Linux (setup, doctor, serve, status, decide, switch to and from Laya) | run by the author in a scratch home on one Linux CPU host | functional evidence only; not a comparison and not a performance claim |
| comparative run, Laya vs OpenDecider-nano, fixed Hachidori corpus | NOT_CHECKED | the external corpus is not in this repository; no comparative evidence exists, so the default is unchanged |
| resident comparison (#123) on real models, including physical Windows / RTX 3060 | NOT_CHECKED | implemented and tested with fake residents only; running it with both real models resident is user-side and never inferred from CI |
| physical Windows, RTX 3060, OpenDecider-nano (materialize, activate, restart, CUDA load/warmup, latency, VRAM) | NOT_CHECKED | user-side; never inferred from CI or the Linux run |
| deterministic resident routing and selective handoff (#124) on real models, including physical Windows / RTX 3060 (co-residency, keep/handoff/required-failure routes, no reload, latency, VRAM) | NOT_CHECKED | implemented and tested with fake residents and fake worker processes only; running it with both real models resident is user-side and never inferred from CI |

## Linux GPU host

Same commands with `export HACHIDORI_HOME=/srv/hachidori`. On NixOS the private uv and
CPython (the uv-managed python-build-standalone build, glibc) need a dynamic loader (`nix-ld` or an
FHS environment) and `LD_LIBRARY_PATH` must include the driver libraries
(`/run/opengl-driver/lib`).

## Windows desktop shell (manual, real Windows only)

Generic CI and cross-compilation cannot open a WebView2 window; report these
items as **not checked** unless they were run on a real Windows desktop with
the resulting `hachidori.exe`.

```powershell
go test ./internal/desktop -v -count=1          # Windows-only tests: instance guard, runtime detection
.\hachidori.exe desktop --home D:\Hachidori-test
```

1. With the WebView2 Runtime installed: a native window titled "Hachidori"
   shows the dashboard (status, Start/Stop/Restart, doctor, tunnel).
   - From an existing terminal, the terminal remains visible and prints
     `WebView2 Runtime <version>`.
   - From Explorer/no-argument double-click, the process-owned console is hidden
     and only the desktop window is visible.
   Start/Stop/Restart work.
2. A second `hachidori.exe desktop` by the same user activates the running
   window (restores and focuses it, also when it is hidden in the tray), exits 0
   and starts no worker (`status` pid unchanged).
3. F12 / Ctrl+Shift+I open no DevTools and right-click shows no browser
   context menu. The navigation allow-list itself is covered by unit tests
   (`TestPolicyAllowsOnlyTheDashboardOrigin`); any navigation the window
   cancels is logged on the console as `desktop blocked`.
4. Closing the window hides it to the tray (first time: a notice) and the API
   keeps answering with the same worker pid. Tray Open restores the same window.
   **Quit Hachidori** stops the process: the API and dashboard ports are free, the
   tray icon is gone and no `python` worker or `msedgewebview2.exe` child of this
   run remains.
5. On a machine without the WebView2 Runtime (or with it uninstalled), the
   command fails before any worker starts with the WebView2 diagnostic, and
   nothing is downloaded.
6. Unified composition: a no-argument `hachidori.exe` and `hachidori.exe desktop`
   behave identically. From a first run, after Install reaches READY the same
   window stays open with the tray icon (Open/Restart/Diagnostics/Quit act on
   the one worker; closing hides to the tray); a second launch activates it. A
   `--background` launch with Start minimized starts in the tray only when the
   runtime is up, and shows the window when first run/recovery is needed or the
   start failed.

## Windows first-run wizard (manual, real Windows only)

Generic CI and cross-compilation cannot show a WebView2 window, the native
folder dialog or a GPU. Report every item below as **not checked** unless it was
run on a real Windows machine with the resulting `hachidori.exe`, and mark the
GPU items separately.

Prepare a clean user profile: no `%LOCALAPPDATA%\Hachidori\bootstrap.json`, no
`HACHIDORI_HOME` in the environment, WebView2 Runtime installed.

1. Double-click `hachidori.exe`. A window titled "Hachidori" shows the first-run
   page with no companion console window; no terminal input is needed. Running
   the same no-argument executable from an existing terminal must leave that
   terminal visible.
2. **Browse...** opens the normal Windows folder dialog. Choose a folder on a
   non-system drive. The page shows the chosen path (and free space when
   reported). No bootstrap file exists yet.
3. Choose CPU (or CUDA on a supported NVIDIA machine) and **Install**. The page
   shows the setup phases as they begin, then the runtime warming, then READY
   with the model, device and runtime identity.
4. `%LOCALAPPDATA%\Hachidori\bootstrap.json` now names the chosen folder and
   contains only `schema` and `home`; the heavy state is under the chosen folder.
5. Close the main window: it hides to the tray and the API keeps answering with
   the same worker pid. Use **Quit Hachidori** from the tray and confirm the
   process exits, the API/dashboard ports are released, and no `python` worker
   or `msedgewebview2.exe` child from this run remains. Double-click again:
   the dashboard appears and reaches READY without the wizard, from the same
   home.
6. Rename the chosen folder and relaunch: a recovery screen names the missing
   folder and nothing is installed. Rename it back and use Browse to locate it:
   "Use this installation" starts it without running setup.
7. Corrupt `bootstrap.json` and relaunch: a diagnostic recovery screen, not an
   exit.
8. With the GPU unusable, choose CUDA: setup or startup reports the failure and
   Retry / Change location are offered; Hachidori never switches to CPU.
9. Tray: exactly one icon; its tooltip and menu header read Starting, then Ready;
   stopping the worker from the dashboard reads Stopped; forcing a worker failure
   reads Needs attention with a one-time notice, and Open lands on Diagnostics.
   Restart Runtime restarts the worker (new pid) without a second owner.
10. Start at sign-in (non-elevated shell): enabling it creates exactly one
   `Hachidori` value under `HKCU\...\Run`; enabling again changes nothing;
   unchecking removes it. `go test ./internal/desktop -run RunKey -v` passes. After
   a real sign-out/sign-in the app starts in the background (hidden when Start
   minimized is on) and reaches Ready.
11. With the runtime uninstalled, a `--background` launch shows the recovery /
   needs-attention UI rather than silently remaining hidden or relaunching
   itself.

## Windows recovery boundaries (manual, real Windows only)

Status of every item below: **NOT_CHECKED**. The recovery producer described
here is covered by portable unit tests (`go test ./internal/app
./internal/desktop ./internal/dashboard ./internal/worker`) and a Windows amd64
non-CGo build/vet; none of that is a physical Windows or GPU PASS. Record each
item as PASS / FAIL / NOT_CHECKED with the exact `hachidori.exe` identity and
the Windows and WebView2 versions. Diagnostic bundle export and the physical
certification run are recorded through the checklist below.

1. Worker crash: with the worker READY, end the `python` worker process from
   Task Manager. Diagnostics reports "Recovering from an unexpected worker exit",
   the worker returns to READY with a new pid, and the app never shows this as
   an operator Stop. Repeating the kill more than the restart budget (3 within
   10 minutes) stops the automatic restarts: the tray reads Needs attention and
   Diagnostics reads "Automatic recovery stopped" with the last worker failure.
   Restart Runtime (tray or dashboard) then starts a worker with a fresh budget.
2. Operator Stop / Quit: Stop from the dashboard, or Quit Hachidori, never shows
   a recovery notice and never restarts the worker by itself.
3. Device: on a CUDA home whose GPU is unusable the failure is reported and the
   requested device is unchanged; recovery never switches to CPU.
4. WebView2: end a `msedgewebview2.exe` render process of this run. The window
   reloads the loopback dashboard (at most 3 times, then one native message box
   and tray notice naming the failure); it is never left as only a blank
   surface. The console logs `WebView2 process failure` / `WebView2 navigation
   failure` lines. Navigation is still restricted to the loopback dashboard.
5. Existing first-run, tray, single-instance and start-at-sign-in checks above
   still pass unchanged.

## Diagnostic bundle and physical certification record

**Diagnostic bundle.** Diagnostics > **Export bundle** (an explicit operator
action; nothing runs in the background) writes one bounded local archive,
`state/diagnostics/hachidori-diagnostics-<UTC time>.zip` under the Hachidori
home. It is never uploaded. Its format is versioned
(`hachidori.diagnostics.manifest/v1`, `hachidori.diagnostics.facts/v1`) and it
holds exactly three entries:

- `manifest.json`: schema, creation time, per-entry size and SHA-256, and the
  declared exclusions;
- `facts.json`: application and executable identity (name, SHA-256, size, Go
  version), OS and architecture, runtime, model, device, provider/torch/CUDA
  versions, WebView2 Runtime version (desktop shell only) and worker state with
  the recovery state (`recovering` / `gave_up`) and last failure class/message;
- `worker-log-tail.txt`: at most the last 200 lines (512 bytes each) written by
  Hachidori's own worker log, with the Hachidori home path replaced by a
  placeholder.

It never contains request, state or question text, datasets or experiment
contents, environment variables, SSH credentials or known_hosts, model files,
raw worker stderr or doctor output. `hachidori doctor` output is unchanged.

**Physical certification record.** Every Windows item is recorded in
[windows-certification-checklist.md](windows-certification-checklist.md) as
exactly `PASS`, `FAIL` or `NOT_CHECKED`, together with the executable SHA-256,
Windows build, WebView2 version and device of that run. It covers clean first
run, the native folder picker (#47), remembered-home launch, WebView2
navigation and rendering (#41), tray/reopen, start at sign-in, runtime restart,
worker crash and recovery, CPU, and CUDA when the hardware exists. Cross-builds
and portable tests never produce `PASS` for a physical item.

## Clef Gated DeltaNet kernels (#221)

The selected CUDA path uses `fla-core==0.5.2` for both convolution and chunk
Gated DeltaNet. Linux uses `triton==3.6.0`; Windows amd64 uses
`triton-windows==3.6.0.post26`. These are wheel-only dependencies in the serving
Runtime Spec/uv lock, not user-site packages or worker-time Hub downloads.
The kernels require CUDA compute capability >= 8.0 and the Clef `bfloat16`
compute path. CPU, older CUDA capabilities and `float32` explicitly execute
references. Supported CUDA startup/inference errors propagate without a
reference or CPU retry. This does not change the W4A16 recipe, artifact identity,
question encoding, joint head, or preservation policy.

Investigation on 2026-10-03 used the exact locked `transformers==5.17.0` wheel
(SHA-256 `78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801`).
Its Qwen3.5 and Hub integration sources match the upstream v5.17.0 tag:

- [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py):
  Clef's `use_cache=False` forward uses `causal_conv1d_fn` and
  `torch_chunk_gated_delta_rule`, not the cached update/recurrent functions.
- [Transformers dispatch](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py):
  optional-package dispatch freezes a callable at import time and catches import
  errors. Merely installing a distribution does not establish optimized execution.
  `use_kernels=True` can substitute Hub functions, but its whole-layer
  `Atlas-Inference/gdn` replacement is SM121-only, excluding the RTX 3060 (SM86).
- [causal-conv1d 1.7.0 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)
  has Linux wheels, no Windows wheels, and no Torch 2.11 wheel. Its PyPI sdist
  cannot satisfy `uv sync --no-build`. Convolution Hub builds similarly do not
  establish a native Windows solution. Installing `kernels` alone neither pins
  its Hub artifacts in uv nor supplies a matching Windows convolution binary.
- [FLA's upstream convolution](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/conv/causal_conv1d.py)
  offers a Triton backend with the same causal depthwise convolution, weight
  layout and SiLU activation. Hachidori's adapter transposes inputs/outputs and
  explicitly selects that backend without recurrent state. The
  [chunk delta API](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py)
  accepts the pinned Qwen3.5 positional Q/K/V and named gate, beta, state,
  normalization and sequence-length arguments directly.
- [FLA Windows support](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/utils/_device.py)
  explicitly recognizes `triton-windows`.
  [Triton Windows compatibility](https://github.com/triton-lang/triton-windows/blob/e113e6b604f25e38dd45565a875d12069f9d7245/README.md)
  pairs Torch 2.11 with Triton 3.6, supports Ampere BF16, and bundles a CUDA 12.8
  toolchain. The locked CPython 3.12 Windows amd64 wheel SHA-256 is
  `189d8c57911aa9d2ff983a715e5c967b325f576307db60924cab22b501a36515`.

`kernel_paths` in resident provider information distinguishes availability from
successful execution (see `runtime.md`). Physical benchmarks may claim
optimized execution only if **both** operations report `optimized_active` on
the same resident runtime. Portable tests exercise selection, layout/argument
adaptation, error propagation, explicit references, and existing Clef result
and variant contracts with simulated kernels; they do not certify CUDA.

| Physical acceptance | Result |
|---|---|
| Native Windows amd64 private-runtime materialization and kernel JIT | NOT_CHECKED |
| RTX 3060 startup with both operations `optimized_active` | NOT_CHECKED |
| Removal of both observed reference-fallback warnings | NOT_CHECKED |
| Fixed accepted W4A16 before/after latency, correctness, peak VRAM, long-state behavior | NOT_CHECKED |
| Numerical equivalence and accuracy regression | NOT_CHECKED |

For the physical before/after comparison, retain the exact accepted W4A16
artifact, states, questions, options and dtype. Require identical choices and
maximum absolute per-option probability difference <= 0.01 across the fixed
workload, and no accuracy regression against its expected labels. Record the
long-state result or timeout explicitly. This kernel comparison tolerance does
not replace or relax the existing variant preservation/certification policy.
