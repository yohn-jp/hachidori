# Clef State execution residency validation

Status: **validated for implementation on the pinned Windows CUDA Clef-Flash W4A16 path**

Date: 2026-10-04

Source research: #267  
Feature authority: #265  
Capacity dependency: #264

## Executive decision

Physical Windows CUDA experiments establish that Hachidori can make Clef State model-execution-resident without assuming ordinary Transformer KV-cache semantics.

The validated resident representation is:

1. an immutable, content-addressed State identity;
2. a Qwen3.5 continuation `Cache` containing the model's hybrid continuation state;
3. accumulated prefix hidden states required by the existing Clef `JointSchemaHead`;
4. a per-request fork of the immutable prefix Cache;
5. Question/schema suffix execution only;
6. concatenation of resident prefix hidden states and request suffix hidden states before running the existing `JointSchemaHead`.

Long prefixes must not be built in one forward pass on the measured RTX 3060 12 GiB target. Prefix construction is instead incremental chunked prefill. Among the measured 128/256/512-token candidates, **512 tokens is the initial production default**: it was the fastest measured long-State construction while retaining low peak VRAM and equivalent decisions.

This is a model-specific Clef/Qwen3.5 execution optimization. It is not a generic KV-cache contract.

## Environment and artifact identity

The evidence was collected on the active native-Windows Hachidori serving environment:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 |
| Physical VRAM | 12.00 GiB |
| Python | 3.12.11 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.17.0 |
| CUDA available | true |
| Source model | `Cloudflare--clef-flash/17f0b0ad64efb65d273590632833508766b2aae6` |
| Active variant | `clef-flash--clef-flash-w4a16-rtn-g128--6cdd68bf9677` |
| Variant scheme | W4A16 |
| Group size | 128 |
| Compute dtype | bfloat16 |
| Quantized format | `compressed-tensors/pack-quantized` |
| Optimized W4 modules | 184 |
| Reference W4 modules | 16 |
| `joint_schema_model.py` SHA-256 | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |

The chunked-prefix run started with 10.99 GiB free VRAM. After loading the W4A16 model, PyTorch reported 8.42 GiB allocated and 1.39 GiB free.

## 1. Encoder contract: State is structurally prefixable

The pinned `joint_schema_model.encode_record` contract constructs model input in this order:

```text
system/user prefix
+ rendered State
+ rendered schema fields / Questions
+ assistant suffix
```

In source terms:

```python
input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
```

The encoder computes the schema first, reserves the fixed prefix/schema/suffix budget, and truncates only `state_ids` to the remaining `max_length`:

```python
fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
if fixed_length > max_length:
    raise ValueError(...)
state_ids = state_ids[: max_length - fixed_length]
```

Therefore resident execution must preserve this exact truncation contract. State residency must not introduce an alternate truncation, compression, summarization, or mutation rule.

### Prefix invariance matrix

A static probe varied State length/content and Question shape. Every successful case retained the entire State inside the common prefix, with a constant 50 tokens before Question-specific divergence:

| State case | Raw State tokens | Record lengths | Common prefix | Common prefix - State | Covers State |
| --- | ---: | --- | ---: | ---: | --- |
| tiny | 5 | 124 / 123 / 180 / 143 / 153 | 55 | 50 | yes |
| short | 47 | 166 / 165 / 222 / 185 / 195 | 97 | 50 | yes |
| medium | 959 | 1078 / 1077 / 1134 / 1097 / 1107 | 1009 | 50 | yes |
| long | 4799 | 4918 / 4917 / 4974 / 4937 / 4947 | 4849 | 50 | yes |
| Japanese | 5399 | 5518 / 5517 / 5574 / 5537 / 5547 | 5449 | 50 | yes |
| symbols | 7799 | 7918 / 7917 / 7974 / 7937 / 7947 | 7849 | 50 | yes |

Question count and ordering were also varied. For a 959-token State, one Question, multiple Questions, and reversed Question order all retained a 1009-token common prefix.

This establishes that State prefixability is an implementation property of the pinned encoder, not an accidental property of one prompt sample.

## 2. Qwen3.5 continuation contract is hybrid, not KV-only

The pinned Transformers 5.17.0 Qwen3.5 implementation accepts `past_key_values: Cache` through the model, decoder layers, attention, and Gated DeltaNet path.

Inspection of the installed implementation showed cache handling for:

- ordinary attention continuation state;
- convolution state via `update_conv_state`;
- recurrent state via `recurrent_states` / `update_recurrent_state`;
- `DynamicCache(config=self.config)` construction when caching is enabled.

The current Clef serving path intentionally disables this:

```python
outputs = text_model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    use_cache=False,
    return_dict=True,
    **media,
)
```

Residency therefore requires enabling and owning Qwen3.5's model-specific continuation state. Implementing only an attention KV cache would be incomplete and is prohibited by #265.

## 3. Clef head requires resident prefix hidden states

`ClefModel.forward` currently gives the entire `outputs.last_hidden_state` to `JointSchemaHead`.

The head does not consume only Question-token positions. It also:

- normalizes the full sequence;
- projects the full sequence into `memory`;
- uses the last sequence hidden state as `global_vector`;
- computes Question vectors and option-context vectors from spans;
- routes options and fields through layers that attend to the full sequence memory.

Therefore retaining only Qwen3.5 `Cache` is insufficient for exact Clef behavior. The resident representation must also retain prefix hidden states, then concatenate suffix hidden states before invoking the existing head.

Validated execution shape:

```text
State prefix
   |
   +-- Qwen3.5 hybrid Cache ----------------------+
   |                                              |
   +-- accumulated prefix hidden states           |
                                                  |
request Question/schema suffix                    |
   |                                              |
   +-- fork immutable Cache ----------------------+
   |
   +-- suffix forward
   |
   +-- suffix hidden states
   |
[prefix hidden states + suffix hidden states]
   |
existing JointSchemaHead
   |
existing logits/probabilities/choice mapping
```

## 4. GPU continuation proof

The execution PoC loaded the active W4A16 variant through the same numerical path used by Hachidori: 184 optimized W4 modules and 16 reference W4 modules, BF16 compute, and the pinned Windows CUDA stack.

### Medium State: 959 raw State tokens / 1009-token resident prefix

Full forward measurements:

| Question set | Full latency |
| --- | ---: |
| scope | 8742.080 ms |
| danger | 3789.112 ms |
| multi-question | 3702.867 ms |

Resident prefix:

| Measurement | Value |
| --- | ---: |
| Prefix build | 3427.594 ms |
| Prefix peak allocated | 8.75 GiB |
| Qwen Cache tensor bytes | 81.03 MiB |
| Prefix hidden-state bytes | 7.88 MiB |
| Logical resident payload | 88.91 MiB |
| CUDA allocated with resident payload | 8.51 GiB |

Resident requests:

| Question set | Cache clone | Suffix + head | Total resident request | Full baseline | Steady-state speedup | Max probability delta | Choice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scope | 4.103 ms | 210.378 ms | 214.481 ms | 8742.080 ms | 40.759x | 0.00053155422 | equal |
| danger | 4.070 ms | 209.636 ms | 213.706 ms | 3789.112 ms | 17.731x | 0 | equal |
| multi | 4.115 ms | 356.128 ms | 360.243 ms | 3702.867 ms | 10.279x | 0.0082375407 | equal |

A second `scope` request forked from the same untouched resident prefix produced identical probabilities and choice (`prob_max_abs=0`). This establishes the required immutable-prefix / independent-request fork behavior for the measured path.

The PoC acceptance guardrail was exact choice equality and probability delta no greater than 0.01. All measured medium-State requests passed.

The raw whole-sequence hidden-state maximum absolute difference was large (12.78125 in this experiment), so whole-tensor hidden equality is not itself an acceptance criterion. The production regression contract must evaluate the Clef-observable result boundary and, where hidden-state diagnostics are retained, focus them on the positions and derived values consumed by the head rather than treating a raw whole-sequence maximum as semantic equivalence.

## 5. One-shot long prefix is not an acceptable construction path

For a 4799-token State / 4849-token common prefix, full forward remained semantically valid but was extremely expensive:

| Question set | Full latency | Peak allocated |
| --- | ---: | ---: |
| scope | 36953.757 ms | 12.30 GiB |
| danger | 22302.271 ms | 12.30 GiB |
| multi-question | 22286.509 ms | 12.38 GiB |

The one-shot resident prefix build took **71524.642 ms** and reported **12.38 GiB peak allocated** on a nominal 12 GiB GPU. The run became pathologically slow under this memory pressure.

The completed resident payload itself was not the cause:

| Resident component | Size |
| --- | ---: |
| Qwen Cache | 201.03 MiB |
| Prefix hidden states | 37.88 MiB |
| Total logical resident payload | 238.91 MiB |

The issue is transient prefill working memory for a very long one-shot sequence, not the steady resident State payload. Production must therefore bound prefill work independently from steady-state residency.

## 6. Chunked prefill proof

A second physical PoC constructed the same prefix incrementally through Qwen3.5 continuation state. It intentionally did not repeat the known-bad long one-shot prefill.

### Medium State direct comparison against one-shot

State: 959 raw tokens, 1009-token prefix.

| Construction | Build latency | Peak allocated | Scope probability delta vs one-shot | Danger probability delta vs one-shot | Choices |
| --- | ---: | ---: | ---: | ---: | --- |
| one-shot | 3566.573 ms | 8.75 GiB | reference | reference | reference |
| 128-token chunks | 2788.373 ms | 8.63 GiB | 0.0013444498 | 0.0010953769 | equal |
| 256-token chunks | 3622.224 ms | 8.64 GiB | 0.0029875264 | 0.00083118677 | equal |
| 512-token chunks | 3260.631 ms | 8.66 GiB | 0.00053155422 | 0.00082289428 | equal |

All candidates preserved choices and remained inside the 0.01 probability guardrail.

### Long State

State: 4799 raw tokens, 4849-token prefix.

| Chunk size | Prefix build | Peak allocated | Allocated with resident State | Scope suffix + head | Danger suffix + head |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 15420.778 ms | 8.82 GiB | 8.74 GiB | 240.239 ms | 227.394 ms |
| 256 | 13863.422 ms | 8.83 GiB | 8.74 GiB | 231.230 ms | 227.721 ms |
| 512 | **13554.583 ms** | **8.84 GiB** | 8.74 GiB | 233.617 ms | 238.487 ms |

Chunking reduced measured long-prefix peak from the unacceptable one-shot 12.38 GiB to approximately 8.82-8.84 GiB.

Cross-chunk decision equivalence:

| Comparison | Question | Probability delta | Choice |
| --- | --- | ---: | --- |
| 128 vs 256 | scope | 0.0010156408 | equal |
| 128 vs 256 | danger | 0.00057924539 | equal |
| 128 vs 512 | scope | 0.00033962727 | equal |
| 128 vs 512 | danger | 0.0020108372 | equal |

Maximum measured long-State cross-chunk probability delta: **0.0020108372**.

The measured PoC therefore selects **512-token chunked prefill** as the initial default. It was the fastest of the tested long-State candidates and remained comfortably below the one-shot VRAM peak.

This value is a measured default, not a public semantic constant. Future tuning may change it if equivalent evidence is produced on the supported execution matrix.

## 7. Architecture decision

Production implementation derived from #265 should use the following model-specific contract for Clef:

### Resident State object

A resident Clef State owns, at minimum:

- immutable content-derived State identity;
- the exact encoded/truncated prefix identity needed to prevent mismatched reuse;
- Qwen3.5 continuation `Cache`;
- accumulated prefix hidden states;
- resident byte accounting;
- model/runtime/variant identity sufficient to prevent reuse across incompatible execution artifacts;
- lifecycle/accounting metadata required by #263/#264.

The object is immutable after successful construction.

### Construction

1. Resolve State content through the canonical State identity.
2. Encode with the existing Clef `encode_record` semantics; do not invent a second prompt/truncation path.
3. Identify the stable prefix boundary before Question-specific schema content.
4. Prefill incrementally in bounded **512-token chunks** on the measured path.
5. Accumulate prefix hidden states.
6. Retain the resulting Qwen3.5 hybrid Cache.
7. Admit the completed resident payload only through #264 capacity accounting.
8. On construction failure, OOM, cancellation, model/runtime change, or incomplete state, expose no partially valid resident entry.

### Per-request execution

1. Resolve the immutable resident State by identity.
2. Fork its continuation Cache so request execution cannot mutate the shared resident object.
3. Execute only the Question/schema suffix.
4. Concatenate resident prefix hidden states with suffix hidden states.
5. Invoke the existing `JointSchemaHead` unchanged.
6. Preserve existing result IDs, ordering, probability/choice mapping, timeout/error boundaries, and observability.
7. Release request-local fork state after completion.

### Capacity and eviction

This report proves feasibility; it does not define the global capacity policy.

The resident payload is substantial but bounded (approximately 88.91 MiB for the measured 1009-token prefix and 238.91 MiB for the measured 4849-token prefix). The model itself occupied about 8.42 GiB allocated on the tested device. Therefore:

- residency must integrate with #264;
- registration does not imply GPU admission;
- working-set headroom must be reserved in addition to resident payload bytes;
- entries must be evictable under an explicit policy;
- an evicted State reference remains an identity, but GPU execution residency may need to be rebuilt;
- no hidden cache may bypass capacity reporting.

## 8. Rejected or prohibited designs

### Generic KV-only State cache

Rejected. Qwen3.5 continuation includes model-specific convolution and recurrent state in addition to attention state, and Clef requires prefix hidden states for the head.

### Cache-only residency

Rejected. `JointSchemaHead` consumes full-sequence memory and global/question/option representations derived from hidden states.

### One-shot long prefix construction

Rejected for the measured RTX 3060 target. It produced a 12.38 GiB reported peak and pathological latency.

### Re-encoding State with a new resident-only prompt format

Rejected. Residency must preserve the existing `encode_record` and truncation semantics.

### Mutable shared Cache used directly by requests

Rejected. Request execution mutates continuation state. The resident prefix must be immutable and forked per request.

### Unlimited resident-State map

Rejected. Model occupancy plus State payloads require explicit #264 admission/eviction.

## 9. Implementation and regression requirements

The implementation derived from #265 should not be considered complete unless repository tests cover at least:

- deterministic immutable State identity;
- inline `state` compatibility and `state_ref` exclusivity;
- unknown/unavailable State reference failure;
- resident object binding to model/runtime/variant identity;
- exact preservation of current State truncation semantics;
- chunked prefix construction;
- Cache fork isolation;
- multiple different Question suffixes from one resident State;
- no reuse across different State identities;
- existing Clef head/result path retained;
- exact choice agreement on deterministic fixtures;
- bounded probability tolerance defined by the existing numerical/certification policy rather than an undocumented cache-specific tolerance;
- resident and request working-set accounting through #264;
- eviction/rebuild behavior;
- request provenance and lifecycle visibility through #263;
- same-State/different-Question coalescing remaining distinct from the public batch transport contract;
- no model-specific resident execution enabled for providers or Clef variants that have not established the required continuation capability.

Physical Windows CUDA certification should preserve a benchmark/equivalence case equivalent to this report, because CPU/unit mocks cannot establish the W4A16 + Qwen3.5 + FLA execution behavior measured here.

## 10. Final conclusion

The precondition in #265 has been satisfied for the pinned Clef-Flash W4A16 Windows CUDA execution path.

**Decision: implement State execution residency with 512-token chunked prefix prefill.**

The evidence supports all of the following simultaneously:

- State is a deterministic Question-independent encoder prefix.
- The pinned Qwen3.5 implementation can continue from its hybrid Cache.
- The Cache can be independently forked for different Question suffixes.
- Clef semantic choices are preserved in the measured comparisons.
- The measured probability differences stay within the PoC's 0.01 guardrail.
- Steady resident requests are dramatically faster than repeating full long-State forwards.
- Resident payload size is manageable relative to the model.
- One-shot long-prefix construction is unsafe for the measured 12 GiB target.
- Chunked prefill removes that transient VRAM problem.
- 512 tokens is the best measured initial chunk size among 128/256/512.

The remaining work is production engineering under #265, with capacity governed by #264 and request/provenance observability governed by #263.
