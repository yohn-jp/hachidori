# Hachidori Architecture

Status: target architecture  
Scope: initial product architecture and implementation authority  
Product: Hachidori

## 1. Purpose

Hachidori is a local semantic inference runtime for small, frequent, typed decisions.

Its primary role is to keep a compact inference model resident on an accelerator host and answer semantic questions with low latency. Typical callers are other products in the yohn-jp ecosystem that need inexpensive observations before they apply their own deterministic policy.

Examples include questions such as:

- is this operation blocked by an explicit contract?
- does this request require scope expansion?
- is verification evidence sufficient?
- is execution blocked by the environment?
- is the current work ready to finalize?
- is a reported failure limited to fixtures?

Hachidori is deliberately not the authority that decides what an agent, repository, or workflow should do. It measures or classifies semantic state and returns typed observations. The caller remains responsible for policy and action.

The architectural principle is:

```text
state / text
    |
    v
Hachidori
semantic observation
    |
    +-- blocked_by_contract     0.92
    +-- scope_expansion        0.88
    +-- ready_to_finalize      0.81
    |
    v
caller policy
    |
    v
action
```

This boundary is fundamental. Hachidori should remain useful even when models, accelerators, transports, and callers change.

## 2. Design goals

Hachidori MUST optimize for the following properties.

### 2.1 Resident inference

Model initialization is expensive relative to inference. A model MUST be loaded once, moved to the selected device once, warmed up once, and then reused across requests.

Per-request Python process startup or per-request model loading is outside the target architecture.

### 2.2 Explicit ownership of runtime state

The host machine MUST NOT depend on an arbitrary system Python installation, an incidental virtual environment, the user's global package state, or opaque model caches.

A Hachidori installation is defined by one distribution entry point plus one explicit mutable root:

```text
hachidori.exe
HACHIDORI_HOME
```

Everything Hachidori owns at runtime lives under `HACHIDORI_HOME`.

The single narrow exception is the Windows desktop bootstrap locator (§6.2.1): a per-user discovery record that only says where `HACHIDORI_HOME` is.

### 2.3 Reproducibility

The runtime must be reconstructible from versioned artifacts and manifests.

Installation MUST NOT mean "create a venv and pip install the latest compatible dependencies." Runtime artifacts, model revisions, and checksums must be pinned.

### 2.4 Transport independence

Callers should depend on an endpoint contract, not on whether the inference device is local, across a VM boundary, on another LAN host, or remote.

The same client contract should work for:

- local CPU
- local GPU
- a Windows GPU host reached from a NixOS VM
- a LAN GPU host
- a remote Linux GPU host
- a future cloud GPU service

### 2.5 Low-cost typed semantics

Hachidori is intended for frequent, narrow semantic questions where invoking a general-purpose large language model would be unnecessarily slow or expensive.

The output must be structured and machine-consumable.

### 2.6 Caller-owned evaluation

Datasets, expected labels, evaluation policy, and reports belong to the development or caller side. The inference host receives only the inputs needed for inference.

The runtime is not a benchmark database or experiment manager.

## 3. Non-goals

The initial product explicitly does not own:

- agent orchestration policy
- repository or pull-request governance
- workspace or filesystem authority
- Git process isolation
- agent lifecycle management
- final authorization decisions
- benchmark truth labels
- dataset storage
- SSH tunnel management
- general-purpose LLM chat
- arbitrary Python environment management
- model training

These responsibilities belong to callers or neighboring products.

## 4. System boundaries

The architecture is split into two planes.

### 4.1 Development / caller plane

The development environment owns:

- source code
- datasets
- semantic question definitions
- schemas
- expected labels
- evaluation logic
- benchmark runners
- reports and evidence
- client CLI
- caller-specific policy

For the current environment this is expected to run primarily in the NixOS development VM.

### 4.2 Inference host plane

The accelerator host owns:

- the Hachidori server
- the managed ML runtime
- model artifacts
- accelerator execution
- queueing and batching
- runtime cache
- logs
- runtime state
- operational metrics

For the current environment this is expected to be a Windows host with an NVIDIA GPU. The architecture must not encode Windows as a permanent semantic assumption.

### 4.3 Data flow

```text
NixOS / caller                              GPU host
-----------------                          --------------------------
dataset
question schema
evaluation logic
      |
      | HTTP request
      | state + questions + options
      +-----------------------------------> Hachidori server
                                             |
                                             v
                                        resident worker
                                             |
                                             v
                                        model on CUDA
                                             |
      <----------------------------------- typed observations
      |
local scoring
accuracy / ECE / latency
report / evidence
```

The dataset is not copied wholesale to the inference host. Evaluation remains caller-side.

## 5. Process architecture

Hachidori should use a native supervisor/server process and a private managed inference worker.

The initial implementation should use:

- Go for the CLI, server, supervision, lifecycle, configuration, health, metrics, and protocol handling
- Python for Laya / PyTorch inference

This is an implementation boundary, not a public API boundary.

### 5.1 Why Go owns the outer runtime

Go is appropriate for:

- a single user-facing executable
- deterministic process supervision
- HTTP serving
- portable CLI behavior
- runtime lifecycle
- host diagnostics
- structured status
- package/materialization control

### 5.2 Why Python remains inside the inference worker

The model stack already relies on Python and PyTorch. Reimplementing the model stack in Go would add risk and maintenance cost without improving the public contract.

The correct abstraction is not "embed Python logic into Go." It is:

> Hachidori materializes and owns an immutable ML runtime artifact, then supervises it as an internal implementation detail.

### 5.3 Worker lifecycle

The required lifecycle is:

```text
process start
    |
    v
resolve active runtime
    |
    v
start private Python worker
    |
    v
load pinned model
    |
    v
move model to device
    |
    v
warm up
    |
    v
READY
    |
    +--> predict
    +--> predict
    +--> batch predict
    +--> predict
    |
    v
shutdown
```

A request MUST NOT spawn Python or reload the model.

## 6. Runtime materialization

### 6.1 User-facing model

The user should be able to establish the runtime with commands in this shape:

```text
hachidori setup --home D:\Hachidori --device cuda
hachidori serve --home D:\Hachidori
```

The exact CLI syntax may evolve, but the ownership model must remain explicit.

### 6.2 HACHIDORI_HOME

The sole mutable root should have a layout equivalent to:

```text
HACHIDORI_HOME/
  runtime/
    <runtime-version>/
  packages/
  models/
    <model-id>/
  cache/
  logs/
  state/
    active-runtime.json
```

No runtime-owned mutable state should escape this root except operating-system resources that cannot reasonably be relocated, such as the installed GPU driver.

### 6.2.1 Windows bootstrap locator (the one exception)

So a double-clicked Windows desktop can rediscover the selected home without an environment variable or CLI flag, Hachidori keeps exactly one out-of-home file, on Windows only:

```text
%LOCALAPPDATA%\Hachidori\bootstrap.json
{"schema": "hachidori.bootstrap/1", "home": "D:\\Hachidori"}
```

It owns only discovery metadata. It MUST NOT contain runtime/model manifests, device/model selection, worker state, logs, caches, credentials/secrets, SSH material, semantic policy, benchmark/evaluation state, or copies of active-runtime state; it is never a second state root. The record is strictly versioned: unknown fields, a relative or unclean `home`, or invalid JSON are malformed, and any other `schema` is rejected, never treated as a first run.

Desktop discovery (`home.Discover`) uses the precedence explicit home from the desktop/controller > `HACHIDORI_HOME` > valid locator > unconfigured (first run). A locator whose home no longer exists is reported distinctly from never configured. Lower authorities are not read once a higher one applies. The CLI resolver (`home.Resolve`) stays strict and never reads the locator.

Selecting a home (`home.Remember`) requires an existing directory, normalizes it as a Windows absolute path, creates the locator's parent directory only then, and replaces the record atomically; it does not run setup or write under the home. Forgetting (`home.Forget`) removes only the locator file, never the home; the desktop then asks for a home again.

Non-Windows builds have no locator: they create no bootstrap state and desktop discovery resolves only explicit and `HACHIDORI_HOME`.

### 6.3 Isolation requirements

The supervised worker MUST be started with an explicitly constructed environment.

At minimum:

- invoke the private Python executable by absolute path
- disable user-site packages
- force Hugging Face and related caches into `HACHIDORI_HOME`
- avoid reliance on host `PATH`
- avoid reliance on a user-created venv
- avoid global pip state
- avoid normal user Hugging Face cache directories

For Python, `PYTHONNOUSERSITE=1` or an equivalent hard isolation measure is required.

### 6.4 Runtime artifacts

Setup should install a prebuilt, pinned, verified runtime bundle rather than resolving dependencies dynamically on the target machine.

An example artifact name is:

```text
hachidori-runtime-windows-amd64-cuda-0.1.0.tar.zst
```

The bundle may contain:

- a private Python distribution
- the exact PyTorch build
- the exact Laya version
- required Python dependencies
- the internal worker implementation
- a runtime manifest

Models should be separate pinned artifacts or revisions so that model and runtime upgrades can be managed independently.

### 6.5 Immutable versions and activation

Installed runtime versions should be immutable.

For example:

```text
runtime/
  0.1.0/
  0.2.0/
state/
  active-runtime.json
```

Upgrades install a new version and then change activation state. Rollback changes activation back to a previously installed version.

In-place mutation of an existing runtime version is prohibited.

### 6.6 Integrity

Every materialized artifact must have a pinned identity and cryptographic digest.

The active manifest should be sufficient to answer:

- which runtime version is active?
- which Python version is active?
- which PyTorch build is active?
- which model/revision is active?
- which worker version is active?
- which hashes were verified?
- which device/provider is selected?

After setup, normal serving should not require network access.

## 7. Provider boundary

Laya is the first provider, not the product definition.

The internal runtime should expose a provider abstraction with responsibilities equivalent to:

- initialize
- report capabilities
- warm up
- decide
- decide batch
- expose provider diagnostics
- shut down

The first provider is expected to be Laya over PyTorch CUDA.

Future providers may use:

- PyTorch CPU
- ONNX Runtime
- another local classification model
- another accelerator backend

Public callers must not need to know Python package layout or model implementation details.

## 8. API contract

The initial HTTP surface should stay small.

Recommended v0.1 endpoints:

```text
GET  /health
GET  /v1/status
POST /v1/decide
POST /v1/decide/batch
```

### 8.1 Health

`GET /health` answers only whether the service is able to accept requests.

Health should not be overloaded with a large diagnostics payload.

### 8.2 Status

`GET /v1/status` exposes structured operational state, including fields equivalent to:

- runtime version
- provider
- model identity/revision
- device
- worker state
- readiness
- accelerator availability
- accelerator memory where available
- request counters
- queue state
- latency summaries

### 8.3 Decide

A decide request conceptually contains:

```json
{
  "state": "...",
  "questions": [
    {
      "id": "blocked_by_contract",
      "type": "choice",
      "choices": ["yes", "no"]
    }
  ],
  "options": {}
}
```

The response conceptually contains one typed result per question:

```json
{
  "results": [
    {
      "id": "blocked_by_contract",
      "choice": "yes",
      "confidence": 0.92
    }
  ]
}
```

Exact schema versioning must be introduced before compatibility is promised, but the contract should remain centered on state plus typed questions producing typed observations.

### 8.4 Batch decide

Batching exists to amortize provider overhead and increase throughput.

Batch APIs should not leak model tensor details. They batch public semantic requests, not provider internals.

## 9. Client contract

Clients should resolve Hachidori through an endpoint.

The common environment contract should be:

```text
HACHIDORI_ENDPOINT=http://127.0.0.1:7843
```

A client should not need to know whether that endpoint is:

- a local process
- an SSH-forwarded remote process
- a LAN host
- a future remote deployment

Recommended client commands are:

```text
hachidori decide
hachidori eval
hachidori benchmark
hachidori status
hachidori doctor
```

The final CLI command organization may change during implementation, but the separation between runtime operations and caller-side evaluation must remain.

## 10. SSH and network topology

For the current NixOS-to-Windows topology, SSH local port forwarding is an acceptable transport mechanism.

Hachidori itself should normally bind to loopback by default, for example:

```text
127.0.0.1:7843
```

The caller may forward that endpoint into its own network namespace.

Important boundary:

> SSH establishes transport. Hachidori does not own SSH lifecycle.

Hachidori should not grow SSH key management, tunnel orchestration, or host provisioning into the inference runtime.

The host dashboard's transport launcher (issue #2) stays inside this boundary: it only starts, observes and stops an `ssh` child it created, with a fixed argument vector, using the host's existing SSH client, identity and configuration. The inference runtime itself does not depend on it.

If explicit non-loopback binding is later supported, it must be opt-in and accompanied by an authentication and exposure model appropriate to that deployment.

## 11. Evaluation architecture

Evaluation is caller-side because it owns truth.

A typical flow is:

```text
hachidori eval dataset.jsonl
    |
    +-- read examples locally
    +-- read expected labels locally
    +-- send only state/questions to HACHIDORI_ENDPOINT
    +-- collect typed observations
    +-- compute metrics locally
    +-- write evidence locally
```

Metrics may include:

- choice accuracy
- calibration error
- confidence
- p50 latency
- p95 latency
- per-question performance

The runtime may expose operational latency, but it does not own semantic correctness metrics because it does not own expected labels.

## 12. Initial empirical baseline

The first coding-agent semantic benchmark established a useful initial target for the Laya provider.

In the combined multi-question benchmark, six coding-agent questions achieved approximately:

- choice accuracy: 0.8333
- p50 latency: 25.3 ms
- p95 latency: 30.5 ms
- expected calibration error: 0.0784
- mean confidence: 0.8215

Per-question performance varied substantially, which is architecturally important. Semantic reliability depends on the feature, primitive, and question formulation. Hachidori must therefore preserve question identity and make calibration measurable rather than pretending every semantic feature is equally reliable.

This benchmark is not sufficient to establish production validity because it used controlled examples. The next validity step is held-out natural coding-session excerpts with human labels.

## 13. Question design

Hachidori should favor observable or narrowly semantic predicates over vague high-level judgments.

For example, questions such as:

- "is an explicit contract blocking the requested operation?"
- "does the requested action require touching files outside the declared scope?"

are preferable to broad prompts such as:

- "is this good?"
- "should the agent continue?"

The product should make narrow semantic features cheap. Final policy remains external.

Different question formulations may require calibration. The architecture must therefore allow:

- stable question IDs
- versioned question definitions
- caller-side benchmark evidence
- comparison of alternative prompts or primitives

Hachidori should not silently rewrite caller question semantics.

## 14. Runtime observability

The host runtime should expose enough information to understand whether inference is healthy without attaching a debugger.

A runtime status view may include:

- server state
- worker state
- provider
- active runtime version
- model/revision
- device
- GPU identity
- VRAM use
- process uptime
- request count
- error count
- queue depth
- current concurrency
- p50 latency
- p95 latency
- recent worker restart reason

This operational view belongs to the runtime.

Accuracy, expected labels, confusion matrices, and benchmark reports belong to caller-side evaluation.

## 15. Failure model

The outer runtime must distinguish at least these failure classes:

### Setup failure

The runtime or model artifact cannot be materialized or verified.

### Provider initialization failure

The private worker starts but cannot initialize the selected model/provider.

### Device failure

The configured accelerator is unavailable or incompatible.

### Worker failure

The inference worker crashes, becomes unresponsive, or violates the internal protocol.

### Request failure

The request is invalid or inference for that request fails.

### Capacity failure

The request cannot be accepted within configured queue or resource limits.

These should be represented as structured errors. A worker crash should not be returned as an ambiguous semantic "no."

## 16. Restart behavior

The Go supervisor should own worker restart policy.

The first implementation should prefer a conservative policy:

- mark readiness false when the worker is unavailable
- capture the worker termination reason
- restart only within a bounded policy
- avoid infinite rapid crash loops
- require successful initialization and warmup before readiness returns

The public endpoint should never report ready while the provider is still loading.

## 17. Security posture

The default deployment is local and should default to loopback-only exposure.

The model worker is an internal child process and should not expose its own network listener unless a later implementation demonstrates a concrete need.

Inputs should be treated as untrusted data. Request limits should exist for:

- state size
- number of questions
- batch size
- queue depth

The inference worker must not interpret model input as shell commands, paths, or executable code.

Runtime bundles and model artifacts must be integrity-checked before activation.

## 18. Upgrade model

Runtime and model upgrades are separate operations.

A safe upgrade sequence is:

```text
download new immutable artifact
    |
verify digest
    |
materialize side-by-side
    |
run local validation / doctor
    |
activate
    |
start / warm
    |
become ready
```

If activation fails, the previous immutable version remains available for rollback.

No upgrade should depend on mutating an existing runtime directory.

## 19. Doctor contract

`hachidori doctor` should verify the system, not merely print configuration.

Checks should include:

- `HACHIDORI_HOME` accessibility
- active runtime manifest validity
- runtime artifact checksums
- model artifact identity/checksum
- private Python executable
- isolation from user-site packages
- cache paths under Hachidori ownership
- provider importability
- configured device
- CUDA / accelerator availability when selected
- model initialization
- warmup inference
- end-to-end smoke inference

Doctor should clearly distinguish host prerequisites from Hachidori-owned failures.

The NVIDIA driver is a host prerequisite. The Python, PyTorch, model, and worker stack are Hachidori-owned.

## 20. Packaging boundary

The product should behave like a one-entry-point installation even though its internal runtime contains multiple components.

"One binary" means:

- one user-facing executable to install and invoke
- one explicit owned runtime root
- no user-managed Python environment
- no manual pip workflow
- no hidden model cache
- deterministic materialization and cleanup

It does not require physically embedding every ML runtime byte into the Go executable.

This distinction keeps the user experience simple without making the implementation brittle.

## 21. Uninstall and cleanup

Hachidori must be removable without forensic cleanup.

Removing the user-facing executable and `HACHIDORI_HOME` should remove all Hachidori-owned runtime state.

On Windows the desktop may additionally leave the bootstrap locator `%LOCALAPPDATA%\Hachidori\bootstrap.json` (§6.2.1), which holds no runtime state; deleting it never deletes `HACHIDORI_HOME`.

The product should avoid:

- arbitrary files in user profile caches
- package installation into system Python
- implicit PATH mutation
- unrelated registry state
- hidden Hugging Face cache use

## 22. First implementation milestone

The first milestone is complete when the following path works end to end:

```text
client CLI
    |
HTTP
    |
Go Hachidori runtime
    |
resident private Python worker
    |
Laya
    |
PyTorch CUDA
    |
RTX-class GPU
```

The implementation must then rerun the coding-agent semantic benchmark through the public Hachidori API.

Certification should demonstrate:

1. the model is loaded once and remains resident;
2. repeated inference does not start Python per request;
3. all mutable runtime/model/cache state is under `HACHIDORI_HOME`;
4. the runtime works without system Python or a user-managed venv;
5. the caller-side benchmark can run through `HACHIDORI_ENDPOINT`;
6. semantic accuracy remains consistent with the established baseline;
7. warm inference latency is measured separately from process/model startup;
8. `doctor` can detect the major broken states described above.

The existing approximately 83.3% combined coding-agent choice accuracy is a baseline to preserve during transport/runtime extraction, not a permanent product SLO.

## 23. Architectural invariants

The following are the initial architecture invariants.

1. Hachidori returns semantic observations; it does not own final policy.
2. The model remains resident across requests.
3. Python is an internal implementation detail, not a host prerequisite managed by the user.
4. Hachidori owns its runtime, model, and cache locations explicitly; the only out-of-home state is the Windows bootstrap locator, which holds nothing but the home's location (§6.2.1).
5. Runtime versions are pinned, verified, and immutable after materialization.
6. Models are versioned independently from the outer runtime.
7. Callers depend on the public endpoint, not on local Python APIs.
8. Datasets and expected labels stay on the caller/evaluation side.
9. SSH may carry traffic but is not part of Hachidori's domain.
10. Provider-specific implementation details do not leak into the public semantic API.
11. Readiness requires a successfully initialized and warmed provider.
12. Benchmark evidence must be able to detect semantic regressions across runtime changes.

Any implementation that violates one of these invariants requires an explicit architecture change, not an incidental code-level shortcut.
