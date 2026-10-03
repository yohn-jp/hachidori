# Operator Abstraction and Model Tuning Architecture

Status: target architecture and UX authority
Issue: #169
Parent Epic: #176
Baseline: main at 51e4be6d6416c4662b4e5a7c2e52cd3880597c70

## 1. Decision

Hachidori is an intent-first model-engineering workstation.

The normal user interface MUST ask the operator for decisions that change model-engineering intent and MUST NOT require the operator to execute deterministic runtime or artifact lifecycle mechanics that Hachidori can execute itself.

The governing interaction rule is:

> Automatic by default, manually overridable where the choice is meaningful, and internally observable at all times.

A second rule is equally important:

> Parameter override is allowed. Lifecycle orchestration is not delegated back to the human.

For example, an operator may pin CUDA, FP32 reference execution, a remote port, or stronger preservation for one model region. The normal workflow must not require a sequence such as Preflight -> Materialize -> Verify -> Activate -> Restart.

Those phases remain real. They become phases of one backend transaction and remain visible as progress and evidence.

This architecture extends the existing Hachidori runtime, System One Forge, certification, resident execution, evidence, and Windows workstation architecture. It does not replace their authorities.

## 2. Authority and precedence

docs/architecture.md remains the top-level product architecture authority.

This document is the detailed authority for:

- desktop/operator abstraction;
- control classification;
- automatic parameter resolution and explicit override;
- desired-state Models behavior;
- intent-driven Forge behavior;
- the first-class Tuning workspace;
- model-region preservation profiles;
- Evidence -> Tuning feedback;
- Settings/Connections abstraction;
- resource-selection UX;
- workstation visual and interaction hierarchy.

Where older documentation describes a now-obsolete placement of a desktop control or requires the operator to execute a mechanical lifecycle sequence, this document controls the operator-facing responsibility. Existing backend identity, lifecycle, security, evaluation, certification, and runtime contracts continue to control their respective domains.

## 3. Non-negotiable backend boundaries

The redesign MUST preserve these authorities:

- internal/app.Controller and existing composed operations own runtime/application lifecycle.
- internal/home and internal/setup own runtime, model, source, variant, materialization, activation, and immutable identity.
- internal/worker and internal/server own exact process execution and serving.
- internal/optimize owns optimizer execution.
- internal/eval owns evaluation and certification evidence.
- internal/dashboard projects typed state and collects operator intent. It is not a lifecycle authority.
- HACHIDORI_HOME remains the managed mutable root, subject only to the narrow existing Windows bootstrap exceptions.
- Source identity and variant identity remain distinct.
- Certification is evidence. It never silently promotes or activates a candidate.
- Apply remains an explicit operator action.
- Requested model, variant, provider, device, and dtype never silently fall back to another target.
- Production UI remains server-rendered Go + WebView2. No SPA state authority is introduced.

## 4. Operator-control taxonomy

Every operator-facing control MUST be classified as exactly one of the following.

### 4.1 INTENT

A human decision about what Hachidori should accomplish.

Examples:

- model or candidate to use;
- models to keep resident;
- evaluation suite;
- optimization objective;
- fidelity/resource trade-off;
- preservation strength for a semantic model region;
- whether to apply an accepted candidate;
- connection destination;
- experiment A/B selection.

INTENT belongs in primary UI.

### 4.2 AUTO+OVERRIDE

A parameter for which Hachidori can determine a safe/default value but an expert may legitimately need a specific value.

Examples:

- execution device;
- reference device;
- reference dtype;
- optimizer profile details;
- bind address;
- remote/local port;
- selected resource location where a default exists.

The UI MUST show both selection mode and resolved value.

Example:

~~~text
Device               Auto -> RTX 3060 / cuda
Reference precision  Auto -> bfloat16
Remote port          Auto -> 7843
~~~

When an operator pins a value, evidence MUST distinguish it from an automatic resolution. An explicit override that cannot be honored fails explicitly; it is never silently replaced.

### 4.3 ORCHESTRATED

A deterministic prerequisite or lifecycle phase owned by the backend.

Examples:

- preflight;
- runtime/model materialization;
- managed-artifact repair;
- verification;
- stop/rebind/start;
- restart after activation change;
- optimizer probe;
- reference run;
- candidate run;
- exact-serving verification;
- rollback/restore.

These phases MAY appear as progress, failure location, history, or diagnostics. They MUST NOT be the required normal-workflow sequence of buttons.

### 4.4 ADVANCED

A valid expert control that is not normally needed to express intent.

Examples:

- pin CPU when accelerator Auto would choose CUDA;
- force FP32 reference execution;
- pin a transport port;
- inspect or override generated model-region mapping;
- choose an exact supported optimizer algorithm when multiple engines exist;
- experimental/uncertified activation.

Advanced controls remain semantic where possible. “Advanced” is not permission to make raw module regexes, command fragments, or file-system mechanics the primary interface.

### 4.5 EVIDENCE

A fact useful for audit, reproduction, diagnosis, support, or confidence, but not normally an operator input.

Examples:

- PID;
- exact runtime/variant ID;
- requested/actual device and dtype;
- optimizer/package versions;
- RecipeSHA256 / manifest digest;
- artifact path;
- exact SSH command;
- worker log tail;
- stage timings;
- exact generated module mapping.

Evidence remains inspectable, usually through Details or Diagnostics.

## 5. Workspace responsibility map

| Workspace | Primary question | Primary human decisions |
| --- | --- | --- |
| Runtime | What is running and is it usable? | normally none; recovery only when needed |
| Models | What should run or remain resident? | model/candidate, residency, optional execution override |
| Forge | What artifact should be built, evaluated, and explicitly applied? | source, optimization profile, evaluation suite, build/evaluate, apply |
| Tuning | What model behavior should be preserved under which resource objective? | fidelity/resource objective and semantic-region preservation |
| Workbench | What does the model think about this bounded state? | state/question semantics |
| Experiments | How does the model perform on this corpus? | dataset/suite, comparison, run intent |
| Evidence | Where and how did behavior regress or fail? | investigation/filter/comparison |
| Settings | What durable application preferences should be used? | preferences and connection intent |
| Diagnostics | What exact internal state explains a problem? | explicit maintenance/recovery actions only |

Runtime is observational. Models is desired-state. Forge is artifact lifecycle. Tuning is model engineering. Evidence is investigation. Diagnostics is internals.

## 6. Shared interaction architecture

### 6.1 One focal task

Each action-oriented workspace SHOULD have one obvious focal decision and one dominant primary action.

Examples:

- Models: desired model/residency -> Use model / Apply desired state.
- Forge: source/profile/evaluation -> Build & evaluate.
- Tuning: objective/preservation -> Build candidate.
- Experiments: evaluation setup -> Run experiment.

Secondary actions must not visually compete with the primary operation.

### 6.2 Auto-resolution

Each AUTO+OVERRIDE value has:

1. selection mode: Auto or pinned;
2. resolved/requested value;
3. actual value when execution exists;
4. resolution reason where useful;
5. evidence of whether the value was automatic or operator-pinned.

“Auto” never means hidden fallback.

### 6.3 Operation projection

Long-running operations use one operation projection:

~~~text
Requested intent
  -> resolved plan
  -> current phase
  -> completed phases
  -> measured result
  -> failure + rollback status when applicable
~~~

The browser does not advance the state machine. It renders backend operation state.

### 6.4 Advanced and Details

Advanced contains optional input overrides.

Details/Evidence contains resolved facts and implementation internals.

These are different concepts and MUST NOT be merged into one generic disclosure.

A workspace SHOULD use a small number of coherent disclosures instead of many independent “advanced” drawers.

## 7. Runtime and Models target design

### 7.1 Runtime

Runtime answers:

- Ready / Starting / Needs attention / Stopped;
- exact SOURCE or VARIANT currently serving;
- exact model/variant identity;
- requested/actual device and dtype;
- compact operational telemetry.

PID, process details, raw paths, command lines, and extended environment facts belong in Details/Diagnostics.

Restart/Stop may remain as explicit maintenance/recovery actions, but routine model switching should not require the operator to reason about restart.

### 7.2 Models

Models expresses desired state:

~~~text
Desired execution target
Desired resident set
Optional execution override
        |
        v
Apply desired state
~~~

The backend transaction is conceptually:

~~~text
resolve
-> validate
-> materialize or repair managed artifacts if required
-> verify
-> stop/rebind/start as required
-> wait READY
-> verify exact serving identity
-> typed-decision smoke
-> commit success
-> restore/rollback previous usable state on recoverable failure
~~~

Preflight, Materialize, Repair, Verify, Activate, Save resident models, and Restart remain backend phases or maintenance tools. They are not the normal model-change workflow.

## 8. Forge target design

Forge owns artifact lifecycle, not model-tuning semantics.

The normal lifecycle is:

~~~text
Source
  -> Optimization profile
  -> Evaluation suite
  -> Build & evaluate
  -> Candidate evidence
  -> explicit Apply
~~~

Build & evaluate is one backend-owned operation whose internal phases may include:

- input/resource resolution;
- preflight;
- prerequisite materialization;
- optimizer execution;
- variant integrity verification;
- candidate probe;
- exact high-precision reference execution;
- exact candidate execution;
- certification/evidence persistence.

The UI shows progress through these phases but does not require separate buttons.

Primary Forge inputs are semantic resources and product concepts. Candidate/reference device, reference precision, and provisioning policy are AUTO+OVERRIDE.

Accepted certification never applies automatically.

Experimental activation remains available only as an explicitly separated Advanced escape hatch.

## 9. Tuning workspace

Tuning is a first-class /tuning workspace, not “Forge > Advanced”.

Its purpose is to expose the part of quantization where human judgment is valuable: quality/resource trade-offs and selective precision preservation.

### 9.1 Objective model

The primary objective expresses a trade-off among:

- decision fidelity;
- artifact size;
- VRAM/RAM;
- latency.

The UI may offer high-level profiles such as Balanced or Maximum fidelity, but the underlying persisted profile is explicit and versioned.

### 9.2 Semantic model regions

The operator MUST NOT need raw module regexes or tensor paths.

Hachidori analyzes the exact source model into semantic regions. The region schema is model-family-specific and versioned.

For a Clef-family model, an analysis may expose regions such as:

- embedding/input projection;
- early attention;
- middle attention;
- late/decision-forming attention;
- early feed-forward;
- late feed-forward;
- output head;
- vision tower;
- other architecture-specific sensitive components.

Layer ranges are not globally hard-coded. They are the output of the model-family analyzer for the exact source identity.

### 9.3 Preservation levels

Each region uses an operator vocabulary such as:

- AUTO;
- Preserve +;
- Preserve ++;
- Maximum / PRESERVED.

AUTO means Hachidori uses the canonical safe policy for that source/profile. It does not mean “quantize everything”.

Current components intentionally preserved at higher precision remain preserved when introducing this layer unless a later accepted design explicitly changes that policy.

### 9.4 Profile contract

A tuning profile records at least:

- schema/version;
- exact source model identity/revision;
- analyzer version;
- compiler version;
- objective;
- region preservation choices;
- which choices are Auto vs pinned;
- any supported expert overrides.

The profile is deterministic and reproducible.

### 9.5 Profile -> Recipe compiler

The compiler maps semantic intent to the existing canonical optimization contract:

~~~text
Tuning intent
-> semantic profile
-> resolved preservation policy
-> canonical Recipe
-> RecipeSHA256
-> BuildID
-> immutable Variant
~~~

The compiler produces exact PreservedModule mappings consumed by the existing recipe/variant architecture.

Generated module mappings are EVIDENCE. They remain inspectable.

A semantic profile change that changes the canonical recipe MUST change recipe/build/variant identity through the existing identity chain.

The initial AUTO profile MUST reproduce the existing canonical optimizer behavior unless separately changed by an accepted issue and evidence.

### 9.6 Impact

Tuning presents the consequences of the current profile:

- model size;
- VRAM/RAM;
- latency;
- fidelity/quality.

Every number is explicitly labeled:

- MEASURED — directly observed from compatible evidence;
- ESTIMATED — derived by a documented estimator;
- NOT_CHECKED — no valid evidence.

Estimated values must never be styled or worded as measured facts.

### 9.7 Tuning does not own Forge

Tuning edits intent and may request “Build candidate”.

Forge still owns optimization/certification/apply lifecycle. Tuning hands the exact profile/recipe to Forge and receives resulting evidence.

### 9.8 Layer-wise profiles (schema 2)

Issue #225 refines the semantic regions of 9.2 into stable layer groups. A schema 2 profile holds one choice for every group of the exact source analysis; the schema 1 profile (one choice per region) remains readable and buildable.

**Groups.** The analyzer (`clef-qwen3.5/2`) partitions each region by transformer block: `block.NN.linear-attn`, `block.NN.linear-attn.decay-gate`, `block.NN.linear-attn.beta-gate`, `block.NN.full-attn`, `block.NN.mlp`, plus `output-embeddings`, `vision-tower` and `joint-schema-head`. Identifiers derive from the declared layout, never from tensor names. For the pinned Clef-Flash revision this is 32 blocks (24 linear-attention, 8 full-attention) and 115 groups over 359 Linear modules. The analysis digest covers the groups, so a saved profile cannot apply to another revision or to a structurally different model.

**Policies.** A group is AUTO or carries one named policy from an ordered set that contains only what the optimizer executes today:

| rank | policy | transformation |
| --- | --- | --- |
| 0 | `source-precision` | the group's Linear modules stay at the source precision (bfloat16) |
| 1 | `w4a16` | weight-only int4, symmetric, group size 128, round to nearest |

Two policies are a discrete choice, not a slider; a slider is admissible only for a longer ordered set whose every position is a named, executable policy. Other precisions (for example W8A16) are not offered: the optimizer writes one compressed-tensors quantization group, `setup.CheckPreserved` refuses any variant whose saved config is not that single group, and the only per-module selector the backend addresses is its ignore list. Per-group mixed precision therefore requires an accepted backend change first.

**AUTO and required groups.** AUTO resolves to the canonical recipe's policy (`clef-auto/1`): W4A16 for the backbone projections. The groups the canonical contract always preserves (both linear-attention gates, output embeddings, vision tower, joint head) are required: they have no control and an override is rejected, so a profile can never quantize them.

**Plan and identity.** Compiling a profile resolves a plan: for every group the requested policy (`auto` or the override), the effective policy, AUTO versus OVERRIDDEN, and the basis. The plan digest is recorded in the build provenance (`hachidori.tuning-provenance/2`), so the complete effective profile, AUTO resolutions included, is part of the BuildID and the variant ID; an override that restates AUTO leaves the recipe unchanged but is a different effective profile. A profile also records the canonical recipe it was authored against and is refused if that recipe changes.

**Validation before a build.** Unsupported policies, overrides of required groups, a profile that would quantize nothing, a profile bound to another source, structure or recipe are refused when the profile is saved and again while Forge resolves its inputs, before preflight and before the optimizer starts.

**Evidence.** The optimizer builder checks the optimizer's own report against the plan before anything is digested and refuses a build whose applied transformation differs from the resolved policy of any group. It then writes `hachidori-tuning-evidence.json` into the variant (covered by the variant digests): for every group the requested and effective policy, AUTO or OVERRIDDEN, the transformation applied and the written dtype of preserved modules.

**Baseline comparison.** The Tuning page lists the groups whose effective policy differs from the baseline: the accepted baseline's applied evidence, or the canonical recipe's all-AUTO resolution when the baseline was built by the canonical recipe, and NOT_CHECKED when neither is known. For a built candidate it shows accuracy, fidelity, latency, VRAM and artifact size beside the accepted baseline (the active variant when its certification is accepted), each MEASURED or NOT_CHECKED; evaluated figures are compared only when both come from the same dataset and questions.

**Migration.** A schema 1 profile is shown as its exact schema 2 equivalent: every group of a pinned region becomes an override to `source-precision`, everything else is AUTO, pins on regions the canonical contract already requires are recorded as redundant. Migration is refused unless the current analysis describes the same declared layout and both compile to the same set of preserved modules. The legacy profile is never changed; saving the equivalent creates a new profile that names its lineage.

### 9.9 RAM-resident tuning trials (Candidates)

Issue #226 makes layer-wise experiments cheap enough to run in the tens or hundreds. A full Forge build quantizes the whole model, serializes a multi-gigabyte artifact, unloads and reloads it; a trial instead changes the representation of the few groups that differ from the previous trial, in place, on a resident model.

~~~text
source (canonical weights, RAM)
  -> transformed component cache (RAM, by deterministic identity)
  -> the one model on the accelerator, changed by a bounded delta
  -> the existing resident evaluation
  -> Candidate + trial Evidence      (repeat)
  -> Forge, only for a selected finalist
  -> immutable Variant -> clean-load certification -> Apply
~~~

Three lifecycle objects are kept apart:

| object | what it is | persisted as |
| --- | --- | --- |
| **Trial** | the ephemeral composition being measured: the resolved plan applied to the resident model | nothing; reversible, disposable |
| **Candidate** | a reproducible result: exact source, profile, resolved plan and component set, with immutable trial Evidence | `state/tuning-trials/<candidate id>/` |
| **Variant** | the immutable Forge artifact | `variants/…` (unchanged) |

**Plan, not a second policy.** A trial is driven by the resolved `home.TuningPlan` of 9.8. The delta between two trials is computed from the two plans' effective policies per stable group (`trial.Diff`); plans of different structures are refused.

**Components.** A transformed component is identified by the digest of everything that can change its bytes or the safety of reusing it: the exact source (revision and every pinned file digest), the plan schema, the stable group and its modules with their source tensor shape and dtype, the transformation (policy, scheme parameters read from the same declaration the Forge builder verifies variants against, transformation implementation version, numerical backend version). Profile identity and display state are not part of it, so two profiles that need the same transformation of the same group share the component. A component is never reused across a source revision, plan schema, tensor shape or dtype, transformation parameter or implementation version.

**RAM tier.** The session has an explicit byte budget (there is no default and nothing is assumed about the host). The canonical source and the transformed components count against it. Eviction is deterministic (least recently used first, by a logical clock) and never touches a component the current trial needs, the known-good state a rollback needs, or one that is being built; a trial that cannot fit fails before it changes the model. Cache hits, misses, evictions, bytes RAM to GPU, bytes released, RAM in use and assembly time are recorded in every trial's Evidence.

**Transactions.** Application is validate, stage on the accelerator beside the old weights, swap, release the old weights, and read back what the modules actually are; a mismatch with the requested plan is a failure, never an evaluated model. A failed or cancelled trial restores the previous trial's state from RAM (reverse delta, or an explicit full reconstruction), keeps the valid cache, releases partial allocations and records nothing. If the state cannot be restored the session is broken and refuses further trials. A delta the worker cannot apply in place is refused, or, only when `--allow-reconstruct` is given, rebuilt explicitly and recorded as `reconstruct` in the Evidence. The canonical source and every Variant are never written.

**Evidence is not certification.** Trial Evidence (`hachidori.trial-evidence/1`) states `ephemeral: true`, `certification: NOT_CERTIFIED` and the evaluation mode `trial-fast`, binds the candidate, plan, component set, dataset, inputs and questions it was measured on, and records the worker's own reported execution (`trial`, never a variant). It is a different schema from Decision Evidence, Forge runs and certification and is never read as one.

**Finalists.** `hachidori forge finalist <candidate-id>` (or Continue in Forge from the candidate's row) builds the candidate's exact resolved plan through the normal Forge optimizer, and is refused if the candidate's profile no longer resolves to byte-for-byte the plan that was measured. The Variant is independently loadable and is then certified from a clean load with the existing certification; it is linked to the Candidate by its own provenance.

**Commands.**

~~~text
hachidori forge trial --device cuda --ram-budget 40GiB <dataset.jsonl> <profile-id>...
hachidori forge candidates
hachidori forge finalist <candidate-id>
~~~

The Tuning page shows, for the shown plan, whether a trial was recorded, that it is an ephemeral trial and not certification, its figures, and whether the candidate has been built as a Variant; cache and transfer detail is in Evidence.

**Backend limits (code evidence).** The component cache is exactly as fine as one stable group: the transformation (`TorchOps.build`) is applied per Linear module with the compressed-tensors pack-quantized primitives, and a module executes through the same `packed_forward` binding the Variant loader uses (`bind_packed`). The existing packed W4A16 forward still dequantizes into `F.linear`; a fused/packed kernel (#222) would be a new representation value with its own transformation, and Trial, Candidate and component identity do not change shape for it. Only the two policies of 9.8 exist; no other precision is offered.

**Validation boundary.** Portable tests prove identity, orchestration, rollback, cache behavior, provenance and state transitions, and (with torch, compressed-tensors and llmcompressor installed) that a trial component equals what the Forge optimizer writes, bit for bit, on CPU. Real RAM residency, CUDA replacement, transfer behavior, GPU/RAM figures and wall-clock throughput on the Windows RTX workstation are NOT_CHECKED until measured there. To compare: the old path is the Forge phases `quantizing`, `serializing`, `verifying`, `publish` plus the variant load of a probe/execution; the new path is each trial's `assembly_ms` plus `evaluation_ms` (and `total_ms`), with the session's one-time worker start (`start_ms`) reported separately.

## 10. Evidence feedback loop

The target product loop is:

~~~text
Analyze
  -> Tune
  -> Build
  -> Evaluate
  -> Evidence
  -> Tune when useful
  -> explicit Apply
~~~

Experiments/Evidence may open Tuning with exact context:

- source identity;
- candidate identity;
- tuning-profile identity;
- dataset/question identity;
- certification/evidence identity.

Evidence can be projected beside the tuning decision it informs.

Example:

~~~text
Why?

Long-context discrimination       -3.8%  MEASURED
Late attention sensitivity        HIGH   supported mapping

Recommendation
Preserve late attention one level more.
~~~

Recommendations are advisory. Unsupported or ambiguous evidence yields no recommendation.

Applying a recommendation is an explicit operator action and creates a distinct profile revision/identity. Evidence never mutates tuning state by itself.

## 11. Resource selection

On Windows desktop, resource selection SHOULD be object-first rather than path-first.

Examples:

- Choose dataset
- Choose Question Definitions
- Choose evidence report
- Choose export destination

Use the existing native file/folder/save selection infrastructure as the primary desktop interaction.

Exact host paths remain available in Details/Advanced for reproduction and remain valid in browser/server mode where native desktop selection is unavailable.

This removes path manipulation from the normal Windows workflow without hiding exact resource identity.

## 12. Development Connections and SSH

SSH itself is not an abstraction leak.

The operator’s meaningful intent is:

- connection name;
- SSH destination such as user@development-host;
- connect/disconnect/reconnect.

Authentication, keys, agent state, known_hosts, and SSH configuration remain delegated to the host SSH client.

Transport mechanics are classified as:

| Field | Disposition |
| --- | --- |
| connection name | INTENT |
| destination | INTENT |
| remote bind | AUTO+OVERRIDE |
| remote port | AUTO+OVERRIDE |
| local port | AUTO+OVERRIDE |
| exact ssh -N -R command | EVIDENCE |
| process PID / stderr / exit | EVIDENCE |
| endpoint value for caller | RESULT / EVIDENCE |

Auto resolution preserves the existing loopback-only security boundary unless a separate accepted architecture change authorizes wider exposure.

Hachidori does not become an SSH credential manager or remote-host provisioner.

## 13. Visual system

The workstation SHOULD read as a technical document/instrument, not a collection of generic SaaS cards.

### 13.1 Reading order

A page should read in a deliberate sequence such as:

~~~text
CONTEXT / SOURCE
OBJECTIVE / INTENT
MODEL-ENGINEERING CONTROL
IMPACT / EVIDENCE
PRIMARY ACTION
ADVANCED
DETAILS / DIAGNOSTICS
~~~

Section labels such as SOURCE, OBJECTIVE, PRESERVATION, EVIDENCE, IMPACT, and ADVANCED create a stable reading grammar.

### 13.2 Hierarchy before containers

Use hierarchy in this order:

1. typography;
2. whitespace;
3. alignment/grid;
4. hairline separators;
5. bounded container only where enclosure has interaction/comparison meaning.

Ordinary informational grouping MUST NOT automatically become a large rounded card.

Shadows belong to real raised planes such as dialogs/popovers, not routine data grouping.

### 13.3 State vocabulary

Use one bounded cross-workspace vocabulary:

- AUTO
- OVERRIDDEN
- MEASURED
- ESTIMATED
- PRESERVED
- NOT_CHECKED
- READY
- WORKING
- NEEDS ATTENTION
- FAILED

Color is semantic and restrained.

### 13.4 Evidence near the decision

When evidence explains a model-engineering control, the relevant summary may appear next to that control. This reduces the need to mentally join an isolated report to a tuning action.

The canonical evidence object remains in the Evidence authority; the tuning view is a projection with traceable identity.

### 13.5 Motion

Motion communicates state change, progress, selection, or completion only. Respect reduced-motion preferences. No decorative ambient motion.

## 14. Committed sample-data boundary

Repository-owned documentation, screenshots, HTML mocks, fixtures, examples, and test data MUST NOT contain copied developer-specific environment values.

Do not commit personal:

- usernames;
- hostnames;
- local paths;
- SSH destinations;
- machine names;
- credentials or key material;
- private endpoints.

Use explicit fictional/generic values such as:

- user@development-host
- development-host
- D:\Hachidori\...
- example dataset/profile names

Real runtime evidence follows the repository’s existing bounded evidence/security rules and is not “sample data”.

## 15. State and identity additions

The implementation may add versioned state under HACHIDORI_HOME for:

~~~text
state/
  model-analysis/
    <source-identity>.json
  tuning/
    profiles/
      <profile-id>.json
~~~

Exact layout is implementation detail, but the following requirements are architectural:

- model analysis is bound to exact source identity and analyzer version;
- tuning profile is versioned and reproducible;
- a built candidate records exact profile/recipe linkage;
- evidence records exact candidate/source/evaluation linkage;
- no UI-local database becomes authority.

## 16. Migration plan

### Phase 0 — architecture authority

Commit this document, reference it from docs/architecture.md, and commit a non-production high-fidelity HTML mock.

No production behavior changes.

### Phase 1 — shared primitives

Add canonical Auto+Override, operation progress, resource selection, Advanced, Details/Evidence, and state-vocabulary primitives.

### Phase 2 — Runtime and Models

Migrate to observational Runtime + desired-state Models.

### Phase 3 — Forge

Replace mechanical lifecycle inputs/buttons with intent-driven Build & evaluate and explicit Apply.

### Phase 4 — Tuning

Add model analysis, semantic regions, profile editor, deterministic profile->recipe compiler, impact projection, and Forge handoff.

### Phase 5 — Evidence loop

Carry exact Experiment/Evidence context into Tuning and add conservative explainable recommendations.

### Phase 6 — Settings/Connections/resources

Move bind/ports/device/resource mechanics to Auto+Override or native resource selection while keeping exact evidence available.

## 17. Testing and certification strategy

Portable tests must prove:

- Auto fields expose both mode and resolved value;
- explicit overrides are never silently replaced;
- browser code does not sequence backend lifecycle calls;
- desired-state Models uses one backend operation;
- Forge Build & evaluate uses one backend operation;
- operation failure reports actual phase and rollback result;
- tuning analysis is deterministic for exact source identity;
- tuning profile validation is deterministic;
- profile -> canonical Recipe compilation is deterministic;
- semantic profile changes alter recipe/build identity when semantics change;
- initial Auto profile reproduces current canonical preservation behavior;
- MEASURED / ESTIMATED / NOT_CHECKED cannot be conflated;
- resource picker has a host-path fallback where required;
- Auto SSH bind/ports remain explicit in evidence;
- mock/test/sample data contains no developer-specific environment identifiers.

The existing optional real LLM Compressor tiny-model end-to-end test should eventually prove at least two distinct supported preservation profiles compile and execute as declared. It proves machinery, not real-model quality.

Physical Windows / RTX certification remains required for:

- WebView2 composition and responsive interaction;
- native file/folder/save selection;
- desired-state Models transaction;
- Build & evaluate;
- real Clef model analysis;
- at least two real tuning candidates;
- measured model size, VRAM/RAM, latency, and fidelity;
- Apply and rollback.

Portable CI MUST NOT report physical PASS.

## 18. Acceptance architecture

The architecture is considered implemented only when:

- normal model use requires no manual Preflight/Materialize/Verify/Activate/Restart sequence;
- normal Forge build/evaluation requires no manual Probe/reference-run/candidate-run sequence;
- every meaningful automatic parameter shows its resolved value and can be pinned when technically valid;
- explicit override failures are explicit and never silently fall back;
- raw paths are not the primary Windows resource-selection interaction where a native picker is possible;
- Diagnostics still exposes exact internal evidence;
- Tuning is a first-class workspace;
- Tuning exposes semantic model regions and preservation strength rather than raw module regexes;
- the initial Auto profile preserves current canonical safe behavior;
- tuning profiles compile deterministically into the existing recipe/variant identity chain;
- unmeasured impact is labeled ESTIMATED or NOT_CHECKED;
- evidence can inform the next tuning decision without becoming an optimizer authority;
- pages use a document/instrument visual hierarchy rather than card-heavy dashboard composition;
- repository-owned mock/sample content contains no developer-specific environment identifiers.

## 19. Non-goals

This architecture does not:

- turn Hachidori into a general model-training product;
- introduce arbitrary user-authored optimizer code;
- make the UI a raw optimizer-backend editor;
- hide diagnostic evidence;
- silently choose fallback execution after an explicit override fails;
- automatically apply a candidate because certification passed;
- claim quality impact without measurement;
- replace server-rendered Go/WebView2 with a SPA;
- add SSH credential management;
- authorize public/non-loopback dashboard exposure;
- authorize automatic evidence-driven tuning mutation.

## 20. Final target experience

The product should feel like a model-engineering workstation, not a runtime administration console.

The normal path is:

~~~text
Choose model
   ↓
Tune optimization intent — or keep Auto
   ↓
Build & evaluate
   ↓
Inspect Evidence
   ↓
Adjust Tuning when useful
   ↓
Apply accepted candidate
~~~

Underneath, Hachidori still performs explicit, verifiable operations:

~~~text
resolve
preflight
materialize
verify
optimize
probe
reference run
candidate run
certify
activate/rebind
serve verification
rollback if required
~~~

The architectural improvement is that the human is no longer the workflow engine.
