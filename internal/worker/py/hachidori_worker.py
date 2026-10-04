"""Hachidori private inference worker.

Internal implementation detail of the Hachidori runtime. It is started by the Go
supervisor with an explicitly constructed environment and speaks newline-delimited
JSON over stdin/stdout. It never opens a network listener.

One provider adapter serves the single active model: Laya, OpenDecider-nano or a
Clef System One model (the pinned upstream release, or a Hachidori-built variant of it).
All of them score the request's closed choices and return a probability distribution;
nothing generates free-form text. Provider-specific input construction and result
shapes stay inside the adapters; the protocol below them is the same.

Lifecycle: import provider -> load pinned model -> move to device -> warm up -> ready,
then serve requests until stdin closes or a shutdown request arrives.

Library output written to stdout (Laya prints warnings there) is redirected to stderr
so the protocol channel carries protocol messages only.
"""
import argparse
import copy
import functools
import hashlib
import inspect
import json
import os
import signal
import sys
import time
import traceback
import types

PROTOCOL = "hachidori-worker.v1"

# Take ownership of the real stdout for the protocol, then point fd 1 and sys.stdout
# at stderr so no library can corrupt the protocol stream.
_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(msg):
    _proto.write(json.dumps(msg, separators=(",", ":")) + "\n")
    _proto.flush()


def log(text):
    sys.stderr.write("[worker] %s\n" % text)
    sys.stderr.flush()


def fatal(cls, message):
    emit({"event": "fatal", "class": cls, "message": message})
    log("fatal %s: %s" % (cls, message))
    sys.exit(3)


# A failed provider import or model load keeps a bounded traceback in the worker log,
# whose tail the supervisor attaches (redacted and bounded again) to the failure.
TRACE_MAX_FRAMES = 12
TRACE_MAX_LINES = 32
TRACE_MAX_LINE = 400


def log_failure_trace(phase, exc):
    """Log the traceback of a startup failure: the last TRACE_MAX_FRAMES frames of
    every exception in its chain, the last TRACE_MAX_LINES lines of that text, each
    cut to TRACE_MAX_LINE characters. Source lines only; no locals or values."""
    try:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__,
                                                  limit=-TRACE_MAX_FRAMES))
    except Exception:  # noqa: BLE001 - diagnostics never replace the failure they describe
        return
    lines = text.rstrip("\n").splitlines()
    omitted = max(0, len(lines) - TRACE_MAX_LINES)
    log("%s traceback%s:" % (phase, " (%d earlier lines omitted)" % omitted if omitted else ""))
    for line in lines[-TRACE_MAX_LINES:]:
        line = line.rstrip()
        if len(line) > TRACE_MAX_LINE:
            line = line[:TRACE_MAX_LINE] + "...[truncated]"
        log("  " + line)


def require_account_name():
    """Fail with the real cause when the process has no account name.

    torch._inductor.codecache computes a cache directory from getpass.getuser() at
    module level, after it has already registered InductorCacheArtifact. Without
    LOGNAME/USER/LNAME/USERNAME, native Windows has no pwd module to fall back on:
    that first import fails after registering, an optional-import guard swallows
    the ImportError, and the next import of the module re-runs the registration as
    the misleading "Artifact of type=inductor already registered" assertion.
    The supervisor supplies USERNAME (home.Env); this checks it before torch is
    imported so a missing name is reported as itself.
    """
    import getpass
    try:
        getpass.getuser()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "the worker process has no account name (getpass.getuser() failed with %s: %s); "
            "torch._inductor.codecache needs one at import time and a failed first import "
            "surfaces later as a duplicate inductor cache registration"
            % (type(e).__name__, e)) from e


def to_typed(questions):
    # Laya and OpenDecider share this typed-question shape.
    out = {}
    for q in questions:
        desc = q.get("descriptions") or {}
        out[q["id"]] = {
            "type": "choice",
            "instructions": q["instructions"],
            "criteria": {c: desc.get(c) for c in q["choices"]},
        }
    return out


def from_typed(questions, answers):
    results = []
    for q in questions:
        a = answers[q["id"]]
        results.append({
            "id": q["id"],
            "type": "choice",
            "choice": a["choice"],
            "confidence": a["answer_confidence"],
            "probabilities": a["probabilities"],
        })
    return results


WARMUP_STATE = "The user asked to rename a variable in utils.py; the change is complete and tests pass."
WARMUP_QUESTIONS = [{
    "id": "warmup",
    "type": "choice",
    "instructions": "Is the requested work complete?",
    "choices": ["yes", "no"],
}]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clef_reference(dispatch, operation):
    """Recover the reference frozen by the pinned Transformers 5.17 decorator.

    Even a CPU launch in a CUDA environment must not inherit FLA's GPU callable
    from Transformers' import-time optional-package selection.
    """
    closure = inspect.getclosurevars(dispatch).nonlocals
    implementation = closure.get("implementation")
    reference = closure.get("torch_function")
    optimized = closure.get("is_new_implementation")
    if (not callable(implementation) or not callable(reference)
            or type(optimized) is not bool or optimized != (implementation is not reference)):
        raise RuntimeError("unrecognized Transformers Clef kernel dispatch: %s" % operation)
    params = inspect.signature(reference).parameters

    @functools.wraps(reference)
    def filtered(*args, **kwargs):
        return reference(*args, **{k: v for k, v in kwargs.items() if k in params})

    return filtered


def fla_clef_conv(conv, hidden_states, weight, bias=None, activation=None, **kwargs):
    # Transformers: [B, D, T] -> FLA: [B, T, D]. No caching or packed
    # variable-length sequences are used by the Clef joint-schema forward.
    output, _ = conv(x=hidden_states.transpose(1, 2).contiguous(), weight=weight,
                     bias=bias, activation=activation, backend="triton",
                     output_final_state=False)
    return output.transpose(1, 2)


def configure_clef_kernels(qwen, torch, device, dtype, platform=None):
    """Bind upstream FLA kernels for supported CUDA inference, or explicit refs.

    Import success establishes availability only. Successful calls through these
    bindings establish active execution; failures propagate without CPU or
    reference retries. The resident never downloads or kernelizes from the Hub.
    """
    platform = platform or sys.platform
    reason = "CPU reference execution"
    supported = False
    if device == "cuda":
        capability = torch.cuda.get_device_capability()
        supported = platform == "win32" and capability[0] >= 8 and dtype == "bfloat16"
        if platform != "win32":
            reason = "optimized Clef kernels are supported only on native Windows"
        elif capability[0] < 8:
            reason = "Triton 3.6 requires CUDA compute capability >= 8.0"
        else:
            reason = "float32 uses the high-precision reference path"
    bindings = [
        ("causal_conv1d_fn", "causal_conv1d_fn"),
        ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
    ]
    references = {op: clef_reference(getattr(qwen, symbol), op) for symbol, op in bindings}
    implementations = references.copy()
    if supported:
        # fla-core, not the full model/training distribution, owns these APIs.
        # Missing/incompatible kernels are a startup error on the supported path.
        from fla.modules.conv import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        implementations = {
            "causal_conv1d_fn": functools.partial(fla_clef_conv, causal_conv1d),
            "chunk_gated_delta_rule": chunk_gated_delta_rule,
        }
        reason = "pinned fla-core Triton kernels"
    evidence = {}
    for symbol, operation in bindings:
        implementation = implementations[operation]
        name = ("fla.modules.conv.causal_conv1d" if operation == "causal_conv1d_fn"
                else "fla.ops.gated_delta_rule.chunk_gated_delta_rule") if supported else (
                    references[operation].__module__ + "." + references[operation].__name__)
        entry = {"selected": "optimized" if supported else "reference",
                 "availability": "available" if supported else "unavailable",
                 "implementation": name, "execution": "not_observed", "reason": reason}
        evidence[operation] = entry

        def observed(*args, _implementation=implementation, _entry=entry, **kwargs):
            result = _implementation(*args, **kwargs)
            _entry["execution"] = _entry["selected"] + "_active"
            return result

        setattr(qwen, symbol, observed)
    return evidence


class CapacityError(Exception):
    """A deterministic pre-device admission denial, never a worker failure."""
    def __init__(self, metric, limit, observed):
        self.details = {"metric": metric, "limit": limit, "observed": observed}
        super().__init__("%s: observed %s; limit/required %s" % (metric, observed, limit))


class Provider:
    """The provider-neutral part of the resident model: device policy, lifecycle
    phases, warmup, grouping of requests and status. Adapters supply import, load,
    prediction and the facts they actually observed about the loaded model."""

    name = ""

    def __init__(self, model_dir, device, manifest, dtype=None, variant_dir=None, variant=None, trial_session=False):
        self.trial_session = trial_session
        self.trial = None  # the tuning trial executor of a trial session
        self.model_dir = model_dir
        self.requested = device
        self.dtype = dtype
        self.manifest = manifest
        self.digests = manifest["files"]
        self.variant_dir = variant_dir
        self.variant = variant
        self.torch = None

    # -- adapter hooks -------------------------------------------------------
    def import_provider(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def placed_device(self):
        """The torch device the loaded model is actually on."""
        raise NotImplementedError

    def placed_dtype(self):
        raise NotImplementedError

    def predict(self, states, questions):
        """states, typed questions -> one {"answers": {id: {choice, answer_confidence, probabilities}}} per state."""
        raise NotImplementedError

    def version(self):
        raise NotImplementedError

    def extra_info(self):
        return {}

    # -- lifecycle -----------------------------------------------------------
    def initialize(self):
        emit({"event": "phase", "phase": "importing"})
        try:
            require_account_name()
            import torch
            self.import_provider()
        except Exception as e:  # noqa: BLE001 - any import failure is a provider failure
            log_failure_trace("provider_import", e)
            fatal("provider_import", "%s: %s" % (type(e).__name__, e))
        self.torch = torch
        if self.requested == "cuda":
            if not torch.cuda.is_available():
                fatal("device_unavailable",
                      "CUDA requested but torch.cuda.is_available() is false "
                      "(torch %s, built for CUDA %s, device_count=%d). Check the NVIDIA driver."
                      % (torch.__version__, torch.version.cuda, torch.cuda.device_count()))
        elif self.requested != "cpu":
            fatal("device_unavailable", "unsupported device %r" % self.requested)

        emit({"event": "phase", "phase": "loading"})
        t0 = time.perf_counter()
        try:
            self.load()
        except Exception as e:  # noqa: BLE001
            log_failure_trace("model_load", e)
            fatal("model_load", "%s: %s" % (type(e).__name__, e))
        # A requested device is never substituted: a model that did not land on it
        # is a device failure for Hachidori, not a degraded success.
        placed = self.placed_device()
        if placed.type != self.requested:
            fatal("device_unavailable", "model placed on %s instead of requested %s"
                  % (placed, self.requested))
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        if self.name == "clef":
            try:
                self.check_capacity_readiness()
            except CapacityError as e:
                fatal("capacity", str(e))

    def warmup(self):
        emit({"event": "phase", "phase": "warming"})
        t0 = time.perf_counter()
        try:
            for _ in range(3):
                out = self.decide([{"state": WARMUP_STATE, "questions": WARMUP_QUESTIONS}])
            choice = out[0][0]["choice"]
            if choice not in ("yes", "no"):
                raise RuntimeError("warmup produced invalid choice %r" % (choice,))
        except SystemExit:
            raise
        except CapacityError as e:
            fatal("capacity", str(e))
        except Exception as e:  # noqa: BLE001
            fatal("warmup", "%s: %s" % (type(e).__name__, e))
        self.warmup_ms = (time.perf_counter() - t0) * 1000.0
        if self.name == "clef":
            try:
                self.check_capacity_readiness()
            except CapacityError as e:
                fatal("capacity", str(e))

    def sync(self):
        if self.requested == "cuda":
            self.torch.cuda.synchronize()

    def decide(self, items):
        """Answer items; items sharing a question set share forward passes."""
        groups = {}
        for i, item in enumerate(items):
            key = json.dumps(item["questions"], sort_keys=True)
            groups.setdefault(key, []).append(i)
        results = [None] * len(items)
        for idxs in groups.values():
            questions = items[idxs[0]]["questions"]
            outs = self.predict([items[i]["state"] for i in idxs], to_typed(questions))
            for i, out in zip(idxs, outs):
                results[i] = from_typed(questions, out["answers"])
        self.sync()
        return results

    def info(self):
        torch = self.torch
        device = self.placed_device()
        info = {
            "provider": self.name,
            "provider_version": self.version(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "model_id": self.manifest.get("id", ""),
            "model_revision": self.manifest.get("revision", ""),
            "device": str(device),
            "dtype": str(self.placed_dtype()),
            "model_dir": self.model_dir,
            "load_ms": round(self.load_ms, 1),
            "warmup_ms": round(self.warmup_ms, 1),
            "no_user_site": bool(sys.flags.no_user_site),
            "hf_home": os.environ.get("HF_HOME", ""),
        }
        info.update(self.extra_info())
        if self.variant:
            # The semantic model stays model_id; the variant is what actually executes.
            info["variant_id"] = self.variant["id"]
            info["variant_dir"] = self.variant_dir
        if self.requested == "cuda":
            idx = device.index or 0
            info["device_name"] = torch.cuda.get_device_name(idx)
            info["device_capability"] = list(torch.cuda.get_device_capability(idx))
        return info

    def stats(self):
        out = {}
        if self.name == "clef":
            out["w4_linear_paths"] = self.extra_info()["w4_linear_paths"]
            out["clef_batch"] = dict(getattr(self, "batch_profile", {}))
            out["capacity"] = self.capacity_status()
            if self.requested == "cuda":
                out["memory_peak_allocated"] = self.torch.cuda.max_memory_allocated()
        # Host RAM of this worker process, only when it can actually be read.
        try:
            import psutil
            out["host_rss_bytes"] = int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - absent or unreadable is simply not reported
            pass
        if self.requested != "cuda":
            return out
        torch = self.torch
        free, total = torch.cuda.mem_get_info()
        out.update({
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "memory_free": free,
            "memory_total": total,
        })
        return out


class LayaProvider(Provider):
    name = "laya"

    def import_provider(self):
        import laya
        self.laya = laya

    def load(self):
        self.agent = self.laya.load(self.model_dir, device=self.requested,
                                    expected_sha256=self.digests)

    # Laya silently falls back to CPU when a device placement fails; the base
    # class turns that into a device failure.
    def placed_device(self):
        return self.agent.device

    def placed_dtype(self):
        return self.agent.dtype

    def predict(self, states, questions):
        return self.agent.predict_batch(states, questions)

    def version(self):
        return self.laya.__version__

    def extra_info(self):
        return {"laya_version": self.laya.__version__}


# Upper bound on sequences (states x questions) scored in one padded batch, so a
# large request cannot exhaust memory in a single forward pass.
OPENDECIDER_MAX_SEQUENCES = 16


# The precisions the OpenDecider adapter can be asked for. float32 is what upstream
# evaluated and stays the default; bfloat16 is an explicit, opt-in choice (#136).
OPENDECIDER_DTYPES = ("float32", "bfloat16")


class OpenDeciderProvider(Provider):
    """OpenDecider-nano: an Ettin encoder with one [MASK] marker per option and a
    small head. The opendecider package builds the marked input and reads one logit
    per option; all options of a question are scored in a single encoder pass and
    softmaxed. This adapter only maps Hachidori's typed choice questions onto it and
    its probabilities back; nothing is generated, parsed or retried."""

    name = "opendecider"

    def import_provider(self):
        import opendecider
        self.opendecider = opendecider

    def verify_files(self):
        # The same integrity guarantee Laya gets from expected_sha256: every pinned
        # file is checked against its digest before any of it is loaded.
        for rel, want in sorted(self.digests.items()):
            got = file_sha256(os.path.join(self.model_dir, *rel.split("/")))
            if got != want:
                raise RuntimeError("%s: sha256 %s, want %s" % (rel, got, want))

    def load(self):
        self.verify_files()
        # A local directory with opendecider.json is loaded as is; nothing is
        # resolved from the Hub. float32 is the dtype upstream evaluated; another
        # dtype is only ever the one that was explicitly requested.
        want = self.dtype or "float32"
        self.model = self.opendecider.load(self.model_dir, device=self.requested, dtype=want)
        # A requested dtype is never substituted either: the encoder must hold
        # exactly the dtype that was asked for, or the load is a failure.
        got = str(self.placed_dtype())
        if got != "torch." + want:
            raise RuntimeError("model placed as %s instead of requested %s" % (got, want))

    def parameter(self):
        return next(self.model.impl.enc.parameters())

    def placed_device(self):
        return self.parameter().device

    def placed_dtype(self):
        return self.parameter().dtype

    def predict(self, states, questions):
        per_state = max(1, OPENDECIDER_MAX_SEQUENCES // max(1, len(questions)))
        out = []
        for i in range(0, len(states), per_state):
            for result in self.model.system_one_batch(states[i:i + per_state], questions):
                answers = {}
                for qid, a in result["answers"].items():
                    if a.get("truncated"):
                        log("question %r: state truncated to the model's 2048-token context" % (qid,))
                    answers[qid] = {"choice": a["choice"], "answer_confidence": a["confidence"],
                                    "probabilities": a["probabilities"]}
                out.append({"answers": answers})
        return out

    def version(self):
        return self.opendecider.__version__

    def extra_info(self):
        return {"opendecider_version": self.opendecider.__version__}


# Upper bound on the tokens of one System One record, the upstream default. A schema
# that does not fit is a request error; the state alone is truncated by the upstream
# encoder, which the adapter reports.
CLEF_MAX_LENGTH = 16384

# The precisions the Clef adapter can be asked for on the source model. bfloat16 is
# what the release ships and stays the default; float32 is the explicit
# high-precision reference. A variant executes at the precision it declares.
CLEF_DTYPES = ("float32", "bfloat16")


def packed_reference_weight(module, dtype):
    """Original reconstruction retained as the numerical/profiling reference."""
    if module._hachidori_w4[1] is not None:
        raise RuntimeError("reference reconstruction requires the canonical packed layout")
    out, inner, bits, group = module._hachidori_packed
    q = module._unpack(module.weight_packed, bits, module.torch_Size((out, inner)))
    return (q.to(dtype).view(out, inner // group, group)
            * module.weight_scale.to(dtype).unsqueeze(-1)).view(out, inner)


def select_w4_backend(torch, module, platform=None):
    """Derived execution layout, not an artifact or a new quantizer.

    Torch 2.11 CUDA tinygemm uses offset-8 nibbles and BF16 [scale, zero]
    pairs. Additive zero is exactly zero for our symmetric weights.
    API/layout authority: pytorch v2.11.0 aten/src/ATen/native/cuda/int4mm.cu
    and torch/testing/_internal/common_quantization.py. No requantization.
    Probe the actual wheel/device; API presence does not establish support.
    """
    state = {"selected": "packed-reference", "available": False,
             "executed": None, "optimized_calls": 0, "reference_calls": 0,
             "reason": None}
    out, inner, bits, group = module._hachidori_packed
    platform = sys.platform if platform is None else platform
    if platform != "win32":
        state["reason"] = "native Windows CUDA backend only"
    elif str(torch.__version__) != "2.11.0+cu128":
        state["reason"] = "requires pinned torch 2.11.0+cu128"
    elif module.weight_packed.device.type != "cuda":
        state["reason"] = "requires CUDA resident weights"
    elif module.weight_scale.dtype != torch.bfloat16:
        state["reason"] = "requires BF16 scales/compute"
    elif module.bias is not None:
        state["reason"] = "native kernel has no fused bias; reference preserves bias rounding"
    elif bits != 4 or group != 128 or inner % 128 or out % 8:
        state["reason"] = "requires W4 group 128 and aligned Linear shape"
    elif torch.cuda.get_device_capability(module.weight_packed.device)[0] < 8:
        state["reason"] = "requires NVIDIA SM80 or newer"
    else:
        try:
            # Integer unpacking once; never a resident dense BF16 weight copy.
            q = module._unpack(module.weight_packed, bits, torch.Size((out, inner)))
            u = (q.to(torch.int16) + 8).to(torch.uint8)
            pairs = ((u[:, 0::2] << 4) | u[:, 1::2]).contiguous()
            layout = torch._convert_weight_to_int4pack(pairs, 8)
            scales = torch.stack((module.weight_scale.t(), torch.zeros_like(module.weight_scale.t())), -1).contiguous()
            # Windows upstream tests skip this private API. Require a real,
            # deterministic output comparison on the installed wheel/device.
            probe = torch.sin(torch.arange(inner, device=q.device, dtype=torch.float32)).to(torch.bfloat16).view(1, inner)
            dense = (q.to(torch.bfloat16).view(out, inner // group, group)
                     * module.weight_scale.unsqueeze(-1)).view(out, inner)
            expected = torch.nn.functional.linear(probe, dense)
            actual = torch._weight_int4pack_mm(probe, layout, group, scales)
            torch.testing.assert_close(actual, expected, rtol=0.016, atol=1e-5)
            torch.cuda.synchronize(q.device)
            state.update(selected="torch-cuda-int4pack", available=True)
            return state, (layout, scales, torch._weight_int4pack_mm, torch.bfloat16)
        except (RuntimeError, AttributeError, NotImplementedError, AssertionError) as e:
            state["reason"] = "%s: %s" % (type(e).__name__, e)
    return state, None


def packed_forward(self, x):
    """Linear forward of a module that holds its weight as compressed-tensors
    int4 weights: a native weight-aware kernel when capability probing succeeded,
    otherwise explicit per-call reconstruction. Neither path expands the model."""
    out, inner, _, group = self._hachidori_packed
    state, backend = self._hachidori_w4
    if backend is not None and x.dtype == backend[1]:
        mm, _ = backend
        # A failure after probing is surfaced, never silently retried.
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], out))
        y = mm(x.reshape(-1, inner).contiguous(), self.weight_packed, group, self._hachidori_w4_scales)
        y = y.reshape(*x.shape[:-1], out)
        if self.bias is not None:
            y = y + self.bias
        state["executed"] = "torch-cuda-int4pack"
        state["optimized_calls"] += 1
        return y
    if backend is not None:
        raise RuntimeError("optimized W4 Linear requires BF16 activations")
    w = packed_reference_weight(self, x.dtype)
    y = self.torch_linear(x, w, self.bias)
    state["executed"] = "packed-reference"
    state["reference_calls"] += 1
    return y


def bind_packed(torch, unpack, module, out, inner, bits, group):
    """Bind one capability-gated execution contract to packed int4 Linear.
    The variant loader and the tuning trials
    bind modules through this one function, so a trial executes exactly as the
    variant built from the same plan does."""
    module._hachidori_packed = (out, inner, bits, group)
    module._unpack = unpack
    module.torch_Size = torch.Size
    module.torch_linear = torch.nn.functional.linear
    state, backend = select_w4_backend(torch, module)
    if backend is not None:
        layout, scales, mm, dtype = backend
        # Replace, do not duplicate, packed residency. The immutable on-disk and
        # RAM component formats remain compressed-tensors; this layout is derived.
        module._parameters.pop("weight_packed", None)
        module._buffers.pop("weight_packed", None)
        module.register_parameter("weight_packed", torch.nn.Parameter(layout, requires_grad=False))
        module.register_buffer("_hachidori_w4_scales", scales)
        backend = (mm, dtype)
    module._hachidori_w4 = (state, backend)
    module.forward = types.MethodType(packed_forward, module)


# -- RAM-resident tuning trials ------------------------------------------------
#
# A trial session keeps the canonical source weights of every Linear module in
# system RAM, builds transformed components there once, and changes only the
# representation of the modules of the groups that differ from one trial to the
# next. The model object, its module identities and the canonical source are
# never replaced or written; a failed replacement leaves the model exactly as it
# was. Inference is unchanged: it runs on the accelerator through the very same
# decide path as every other execution.

TRIAL_TRANSFORM = "hachidori.trial-transform/1"  # must equal trial.TransformImplementation in Go
REPRESENTATION_DENSE = "dense"
REPRESENTATION_PACKED = "packed-int4"
# Bounds the accelerator memory one replacement chunk stages beside the weights
# it replaces. A larger delta is applied chunk by chunk; a failed chunk reverses
# the chunks already applied, from system RAM.
TRIAL_STAGE_BYTES = 1 << 30


class TrialError(Exception):
    """A trial operation refused or failed. cls is one of trial_unsupported (an
    explicit full reconstruction can resolve it), trial_incompatible (the
    requested representation cannot be produced on this model), trial_state_lost
    (the model is neither as asked nor as it was) and trial_failed (nothing was
    changed)."""

    def __init__(self, cls, message):
        super().__init__(message)
        self.cls = cls


class TorchOps:
    """The torch-facing half of the trial executor: everything that touches a
    tensor or a module. It is deliberately thin; the transactional logic lives in
    TrialExecutor and is tested without torch."""

    def __init__(self, torch, device, dtype_name):
        self.torch = torch
        self.device = device
        self.dtype_name = dtype_name
        self.pinned = False
        self._unpack = None
        self._ct = None
        self._unsupported = None
        try:
            from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
            from compressed_tensors.quantization import preset_name_to_scheme
            from compressed_tensors.quantization.lifecycle.forward import quantize
            from compressed_tensors.quantization.utils import calculate_qparams
            import importlib.metadata as md
            self._unpack = unpack_from_int32
            self._ct = {"pack": pack_to_int32, "quantize": quantize, "qparams": calculate_qparams,
                        "scheme": preset_name_to_scheme, "version": md.version("compressed-tensors")}
        except Exception as e:  # noqa: BLE001 - reported, and packed replacement is then unsupported
            self._unsupported = "%s: %s" % (type(e).__name__, e)

    # -- identity of the numerical stack ---------------------------------------
    def backend(self):
        if self._ct is None:
            return "unavailable"
        return "compressed-tensors %s" % self._ct["version"]

    def packed_support(self):
        """None when packed modules can be built and executed, else the reason."""
        return self._unsupported

    # -- inspection ------------------------------------------------------------
    def describe(self, module):
        params, bufs = module._parameters, module._buffers
        info = {"class": type(module).__name__, "hooked": bool(getattr(module, "_hf_hook", None)),
                "bias": params.get("bias") is not None}
        weight = params.get("weight")
        packed = params.get("weight_packed", bufs.get("weight_packed"))
        if weight is not None and packed is None:
            info.update(representation=REPRESENTATION_DENSE, shape=tuple(int(d) for d in weight.shape),
                        dtype=str(weight.dtype).replace("torch.", ""), device=weight.device.type)
        elif weight is None and packed is not None and hasattr(module, "_hachidori_packed"):
            out, inner = module._hachidori_packed[:2]
            scale = params.get("weight_scale", bufs.get("weight_scale"))
            info.update(representation=REPRESENTATION_PACKED, shape=(int(out), int(inner)),
                        dtype=str(scale.dtype).replace("torch.", ""), device=packed.device.type)
        else:
            info.update(representation="unknown", shape=(), dtype="", device="")
        return info

    def shared_weights(self, model):
        """Module names whose weight Parameter is also reachable under another name
        (tied or shared): replacing one alias would not replace the other."""
        owners = {}
        for name, p in model.named_parameters(remove_duplicate=False):
            owners.setdefault(id(p), []).append(name)
        shared = set()
        for names in owners.values():
            if len(names) > 1:
                shared.update(n[:-len(".weight")] for n in names if n.endswith(".weight"))
        return shared

    def canonical(self, module):
        w = module._parameters["weight"]
        return {"tensor": w.detach().to("cpu", copy=True), "requires_grad": bool(w.requires_grad)}

    def tensor_bytes(self, tensor):
        return int(tensor.numel()) * int(tensor.element_size())

    # -- components --------------------------------------------------------------
    def build(self, weight, params):
        """Quantize one canonical weight exactly as the compressed-tensors pack
        quantized compressor stores it: per-row, per-group symmetric min-max scale
        (the round-to-nearest observer), int4 values packed into int32 words."""
        torch, ct = self.torch, self._ct
        scheme = ct["scheme"]("W4A16", ["Linear"])
        args = scheme.weights
        if (args.num_bits, args.group_size, bool(args.symmetric)) != (params["bits"], params["group_size"], bool(params["symmetric"])):
            raise TrialError("trial_incompatible", "the library's W4A16 scheme is %s bits/group %s/symmetric %s, the transformation asks %s/%s/%s"
                             % (args.num_bits, args.group_size, args.symmetric, params["bits"], params["group_size"], params["symmetric"]))
        out, inner = weight.shape
        group = args.group_size
        grouped = weight.reshape(out, inner // group, group)
        scale, zero_point = ct["qparams"](torch.amin(grouped, dim=-1), torch.amax(grouped, dim=-1), args)
        q = ct["quantize"](x=weight, scale=scale, zero_point=zero_point, args=args, dtype=torch.int8)
        packed = ct["pack"](q, args.num_bits)
        tensors = {"packed": packed.contiguous(), "scale": scale.to(weight.dtype).contiguous(),
                   "shape": torch.tensor([int(out), int(inner)], dtype=torch.int64)}
        if self.device == "cuda":
            # Pinned host memory makes the repeated RAM -> GPU copies of a cached
            # component faster and lets them overlap; a refusal to pin (no more
            # lockable memory) leaves the component pageable, still correct.
            try:
                tensors = {k: v.pin_memory() for k, v in tensors.items()}
                self.pinned = True
            except RuntimeError:
                pass
        return tensors

    def component_bytes(self, tensors):
        return sum(self.tensor_bytes(t) for t in tensors.values())

    def digest(self, items):
        import hashlib
        h = hashlib.sha256()
        for name, tensors in items:
            h.update(name.encode("utf-8") + b"\0")
            for key in ("packed", "scale", "shape"):
                t = tensors[key].contiguous().view(self.torch.uint8)
                h.update(key.encode("ascii") + b"\0")
                h.update(t.numpy().tobytes())
        return h.hexdigest()

    # -- staging and installation -------------------------------------------------
    def stage_dense(self, canonical):
        return {"weight": canonical["tensor"].to(self.device, non_blocking=True), "requires_grad": canonical["requires_grad"]}

    def stage_packed(self, tensors):
        return {k: v.to(self.device, non_blocking=True) for k, v in tensors.items()}

    def staged_bytes(self, staged):
        return sum(self.tensor_bytes(t) for k, t in staged.items() if k != "requires_grad")

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def snapshot(self, module):
        keep = {}
        for kind, table in (("p", module._parameters), ("b", module._buffers)):
            for key in ("weight", "weight_packed", "weight_scale", "weight_shape", "_hachidori_w4_scales"):
                if key in table:
                    keep[(kind, key)] = table[key]
        extra = {k: module.__dict__[k] for k in ("_hachidori_packed", "_hachidori_w4", "_unpack", "torch_Size", "torch_linear", "forward") if k in module.__dict__}
        return {"tables": keep, "extra": extra}

    def snapshot_bytes(self, snap):
        return sum(self.tensor_bytes(t) for t in snap["tables"].values() if t is not None)

    def _clear(self, module):
        for table in (module._parameters, module._buffers):
            for key in ("weight", "weight_packed", "weight_scale", "weight_shape", "_hachidori_w4_scales"):
                table.pop(key, None)
        for key in ("_hachidori_packed", "_hachidori_w4", "_unpack", "torch_Size", "torch_linear", "forward"):
            module.__dict__.pop(key, None)

    def install_dense(self, module, staged):
        self._clear(module)
        module.register_parameter("weight", self.torch.nn.Parameter(staged["weight"], requires_grad=staged["requires_grad"]))

    def install_packed(self, module, staged, params):
        self._clear(module)
        nn = self.torch.nn
        module.register_parameter("weight_packed", nn.Parameter(staged["packed"], requires_grad=False))
        module.register_parameter("weight_scale", nn.Parameter(staged["scale"], requires_grad=False))
        module.register_parameter("weight_shape", nn.Parameter(staged["shape"], requires_grad=False))
        out, inner = (int(v) for v in staged["shape"].tolist())
        bind_packed(self.torch, self._unpack, module, out, inner, params["bits"], params["group_size"])

    def restore(self, module, snap):
        self._clear(module)
        for (kind, key), value in snap["tables"].items():
            (module._parameters if kind == "p" else module._buffers)[key] = value
        module.__dict__.update(snap["extra"])

    def memory(self):
        out = {}
        if self.device == "cuda":
            out["gpu_allocated"] = int(self.torch.cuda.memory_allocated())
        try:
            import psutil
            out["host_rss"] = int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - absent or unreadable is simply not reported
            pass
        return out


def trial_params(transformation):
    """The transformation parameters a packed component is built with, checked to
    be exactly what this executor can produce."""
    t = transformation
    if t.get("implementation") != TRIAL_TRANSFORM:
        raise TrialError("trial_incompatible", "transformation implementation %r is not %r" % (t.get("implementation"), TRIAL_TRANSFORM))
    if t.get("representation") == REPRESENTATION_DENSE:
        return None
    if t.get("representation") != REPRESENTATION_PACKED:
        raise TrialError("trial_unsupported", "representation %r is not one this worker can execute" % (t.get("representation"),))
    if (t.get("format") != "compressed-tensors/pack-quantized" or t.get("bits") != 4 or not t.get("symmetric")
            or t.get("algorithm") != "rtn" or int(t.get("group_size", 0)) <= 0):
        raise TrialError("trial_incompatible", "unsupported packed transformation %s" % json.dumps(t, sort_keys=True))
    return {"bits": int(t["bits"]), "group_size": int(t["group_size"]), "symmetric": bool(t["symmetric"]),
            "compute_dtype": t.get("compute_dtype")}


class TrialExecutor:
    """Transactional in-place replacement of module weight representations.

    Every module of a replacement is checked before anything changes; the new
    representation is staged on the accelerator beside the old one and swapped in
    by attribute assignment, which keeps the old tensors until the swap is done,
    so a failure at any point restores the old representation without allocating.
    A delta larger than one staging chunk is applied chunk by chunk and a failed
    chunk reverses the chunks before it from system RAM."""

    def __init__(self, ops, modules, device, dtype_name, stage_bytes=TRIAL_STAGE_BYTES, shared=()):
        self.ops = ops
        self.modules = modules  # name -> the module object, captured at open
        self.device = device
        self.dtype_name = dtype_name
        self.stage_bytes = stage_bytes
        self.shared = set(shared)
        self.groups = {}
        self.canon = {}  # name -> canonical weight in system RAM
        self.components = {}  # component id -> {"modules": {name: tensors}, "bytes": int}
        self.current = {}  # module -> (representation, component id or None)

    # -- session -------------------------------------------------------------------
    def open(self, groups):
        self.groups = {g: list(m) for g, m in groups.items()}
        listed = []
        canonical_bytes = 0
        for name in sorted(self.modules):
            module = self.modules[name]
            info = self.ops.describe(module)
            if info["representation"] != REPRESENTATION_DENSE:
                raise TrialError("trial_incompatible", "module %s starts as %s, not as the dense source" % (name, info["representation"]))
            self.canon[name] = self.ops.canonical(module)
            canonical_bytes += self.ops.tensor_bytes(self.canon[name]["tensor"])
            self.current[name] = (REPRESENTATION_DENSE, None)
            listed.append({"module": name, "shape": list(info["shape"]), "dtype": info["dtype"]})
        return {"device": self.device, "dtype": self.dtype_name, "backend": self.ops.backend(), "modules": listed,
                "canonical_bytes": canonical_bytes, "transform": TRIAL_TRANSFORM, "stage_bytes": self.stage_bytes}

    # -- components ----------------------------------------------------------------
    def transform(self, identity):
        t = identity["transformation"]
        params = trial_params(t)
        if params is None:
            raise TrialError("trial_incompatible", "source-precision material is the canonical source; there is nothing to build")
        if self.ops.packed_support() is not None:
            raise TrialError("trial_incompatible", "packed replacement is unavailable: %s" % self.ops.packed_support())
        if t.get("backend") != self.ops.backend():
            raise TrialError("trial_incompatible", "the component is identified under backend %r, this worker is %r" % (t.get("backend"), self.ops.backend()))
        cid = identity["id"]
        if cid in self.components:
            raise TrialError("trial_failed", "component %s is already resident" % cid)
        built, total = {}, 0
        try:
            for member in identity["members"]:
                name = member["module"]
                canon = self.canon.get(name)
                if canon is None:
                    raise TrialError("trial_incompatible", "module %s is not in the resident source" % name)
                weight = canon["tensor"]
                if list(weight.shape) != list(member["shape"]) or str(weight.dtype).replace("torch.", "") != member["dtype"]:
                    raise TrialError("trial_incompatible", "module %s is %s %s, the component was identified for %s %s"
                                     % (name, list(weight.shape), weight.dtype, member["shape"], member["dtype"]))
                if len(weight.shape) != 2 or weight.shape[1] % params["group_size"] != 0:
                    raise TrialError("trial_incompatible", "module %s width is not a multiple of group size %d" % (name, params["group_size"]))
                tensors = self.ops.build(weight, params)
                built[name] = tensors
                total += self.ops.component_bytes(tensors)
        except TrialError:
            built.clear()
            raise
        except Exception as e:  # noqa: BLE001 - nothing partial is kept
            built.clear()
            raise TrialError("trial_failed", "%s: %s" % (type(e).__name__, e))
        digest = self.ops.digest(sorted(built.items()))
        self.components[cid] = {"modules": built, "bytes": total, "params": params}
        return {"bytes": total, "digest": digest, "pinned": bool(getattr(self.ops, "pinned", False))}

    def release(self, cid):
        comp = self.components.get(cid)
        if comp is None:
            return {"released": 0}
        for name, rep in self.current.items():
            if rep[1] == cid:
                raise TrialError("trial_failed", "component %s is in use by module %s" % (cid, name))
        del self.components[cid]
        return {"released": comp["bytes"]}

    # -- validation ----------------------------------------------------------------
    def _check(self, rep, force):
        group = rep["group"]
        t = rep["transformation"]
        params = trial_params(t)
        if rep["policy"] is None or not rep["modules"]:
            raise TrialError("trial_incompatible", "group %s names no policy or modules" % group)
        if params is not None:
            if self.ops.packed_support() is not None:
                raise TrialError("trial_incompatible", "group %s: packed replacement is unavailable: %s" % (group, self.ops.packed_support()))
            if params["compute_dtype"] != self.dtype_name:
                raise TrialError("trial_incompatible", "group %s: the component computes in %s, the model in %s" % (group, params["compute_dtype"], self.dtype_name))
            comp = self.components.get(rep.get("component"))
            if comp is None:
                raise TrialError("trial_incompatible", "group %s: component %s is not resident" % (group, rep.get("component")))
        for name in rep["modules"]:
            module = self.modules.get(name)
            if module is None:
                raise TrialError("trial_incompatible", "group %s: module %s is not in the resident model" % (group, name))
            if name in self.shared:
                raise TrialError("trial_incompatible", "group %s: module %s shares its weight with another module" % (group, name))
            info = self.ops.describe(module)
            canon = self.canon[name]["tensor"]
            if info["class"] != "Linear" or info["hooked"]:
                raise TrialError("trial_incompatible", "group %s: module %s is %s%s, not a plain Linear" % (group, name, info["class"], " with a dispatch hook" if info["hooked"] else ""))
            if info["representation"] == "unknown":
                raise TrialError("trial_unsupported", "group %s: module %s is in a representation this worker does not recognize" % (group, name))
            if list(info["shape"]) != list(canon.shape) or info["dtype"] != self.dtype_name or info["device"] != self.device:
                raise TrialError("trial_incompatible", "group %s: module %s is %s %s on %s, the source is %s %s on %s"
                                 % (group, name, list(info["shape"]), info["dtype"], info["device"], list(canon.shape), self.dtype_name, self.device))
            if params is not None:
                tensors = comp["modules"].get(name)
                if tensors is None or list(tensors["shape"].tolist()) != list(canon.shape):
                    raise TrialError("trial_incompatible", "group %s: component %s does not cover module %s at its shape" % (group, rep.get("component"), name))
            if not force and info["representation"] != self.current[name][0]:
                raise TrialError("trial_unsupported", "group %s: module %s is observed as %s but the session tracked %s"
                                 % (group, name, info["representation"], self.current[name][0]))

    def validate(self, reps, force=False):
        seen = set()
        for rep in reps:
            for name in rep["modules"]:
                if name in seen:
                    raise TrialError("trial_incompatible", "module %s is replaced twice in one request" % name)
                seen.add(name)
            self._check(rep, force)
        return {"ok": True, "modules": len(seen)}

    # -- application ---------------------------------------------------------------
    def _jobs(self, reps):
        jobs = []
        for rep in reps:
            params = trial_params(rep["transformation"])
            for name in rep["modules"]:
                jobs.append((name, rep.get("component") if params is not None else None, params))
        return jobs

    def _job_bytes(self, job):
        name, cid, _ = job
        if cid is None:
            return self.ops.tensor_bytes(self.canon[name]["tensor"])
        return self.ops.component_bytes(self.components[cid]["modules"][name])

    def _chunks(self, jobs):
        chunk, size = [], 0
        for job in jobs:
            b = self._job_bytes(job)
            if chunk and size + b > self.stage_bytes:
                yield chunk
                chunk, size = [], 0
            chunk.append(job)
            size += b
        if chunk:
            yield chunk

    def _stage(self, job):
        name, cid, _ = job
        if cid is None:
            return self.ops.stage_dense(self.canon[name])
        return self.ops.stage_packed(self.components[cid]["modules"][name])

    def _apply_chunk(self, chunk):
        """Stage a chunk beside the old weights, swap it in, then drop the old
        weights. Returns (bytes copied to the accelerator, bytes released). On any
        failure the old representation is back in place and nothing is returned."""
        staged = []
        try:
            for job in chunk:
                staged.append(self._stage(job))
            self.ops.sync()
        except Exception as e:  # noqa: BLE001 - includes accelerator out-of-memory
            del staged[:]
            raise TrialError("trial_failed", "staging on the accelerator failed: %s: %s" % (type(e).__name__, e))
        swapped = []
        try:
            for job, new in zip(chunk, staged):
                name, cid, params = job
                module = self.modules[name]
                snap = self.ops.snapshot(module)
                swapped.append((module, snap))
                if cid is None:
                    self.ops.install_dense(module, new)
                else:
                    self.ops.install_packed(module, new, params)
        except Exception as e:  # noqa: BLE001
            try:
                for module, snap in reversed(swapped):
                    self.ops.restore(module, snap)
            except Exception as r:  # noqa: BLE001
                raise TrialError("trial_state_lost", "installing failed (%s: %s) and its undo failed (%s: %s)" % (type(e).__name__, e, type(r).__name__, r))
            raise TrialError("trial_failed", "installing failed and was undone: %s: %s" % (type(e).__name__, e))
        copied = sum(self.ops.staged_bytes(s) for s in staged)
        released = sum(self.ops.snapshot_bytes(snap) for _, snap in swapped)
        for job in chunk:
            name, cid, _ = job
            self.current[name] = (REPRESENTATION_DENSE, None) if cid is None else (REPRESENTATION_PACKED, cid)
        del swapped[:]
        del staged[:]
        return copied, released

    def apply(self, reps, force=False):
        self.validate(reps, force=force)
        jobs = self._jobs(reps)
        before = {name: self.current[name] for name, _, _ in jobs}
        done, copied, released = [], 0, 0
        for chunk in self._chunks(jobs):
            try:
                c, r = self._apply_chunk(chunk)
            except TrialError as e:
                if done:
                    # Reverse the chunks that were applied, from system RAM.
                    back = []
                    for name, _, _ in done:
                        rep, cid = before[name]
                        back.append((name, None if rep == REPRESENTATION_DENSE else cid,
                                     None if rep == REPRESENTATION_DENSE else self._params_of(cid)))
                    try:
                        for undo in self._chunks(back):
                            self._apply_chunk(undo)
                    except TrialError as u:
                        raise TrialError("trial_state_lost", "%s; reversing the applied chunks failed: %s" % (e, u))
                raise
            copied, released = copied + c, released + r
            done.extend(chunk)
        return {"bytes_to_gpu": copied, "bytes_released": released, "modules": len(jobs)}

    def _params_of(self, cid):
        return self.components[cid]["params"]

    def reconstruct(self, reps):
        return self.apply(reps, force=True)

    # -- state ---------------------------------------------------------------------
    def state(self):
        out = {}
        for group, names in self.groups.items():
            reps = {self.ops.describe(self.modules[n])["representation"] for n in names}
            out[group] = reps.pop() if len(reps) == 1 else ("mixed" if reps else REPRESENTATION_DENSE)
        res = {"groups": out}
        res.update(self.ops.memory())
        return res


def trial_dispatch(executor, op, req):
    """One trial protocol request on an executor."""
    if op == "trial_open":
        return executor.open(req["groups"])
    if op == "trial_transform":
        return executor.transform(req["identity"])
    if op == "trial_release":
        return executor.release(req["component"])
    if op == "trial_validate":
        return executor.validate(req["replacements"])
    if op == "trial_apply":
        return executor.apply(req["replacements"])
    if op == "trial_reconstruct":
        return executor.reconstruct(req["replacements"])
    if op == "trial_state":
        return executor.state()
    raise ValueError("unknown trial op %r" % op)


def _clef_tensor_bytes(value):
    """Count unique tensor storages in the Clef continuation and hidden states."""
    seen, storages = set(), set()
    total = 0

    def visit(obj):
        nonlocal total
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "untyped_storage") and hasattr(obj, "device"):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr())
            if key not in storages:
                storages.add(key)
                total += storage.nbytes()
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        elif hasattr(obj, "__dict__"):
            for item in vars(obj).values():
                visit(item)

    visit(value)
    return total


class ClefResidentState:
    """Completed Clef/Qwen3.5 continuation; the cache is never forwarded directly."""

    __slots__ = ("_identity", "_prefix", "_cache", "_hidden", "_payload_bytes", "_sealed")

    def __init__(self, identity, prefix, cache, hidden):
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_prefix", tuple(prefix))
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_hidden", hidden)
        object.__setattr__(self, "_payload_bytes", _clef_tensor_bytes((cache, hidden)))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("resident State is immutable")
        object.__setattr__(self, name, value)

    @property
    def identity(self):
        return self._identity

    @property
    def payload_bytes(self):
        return self._payload_bytes

    @property
    def prefix_tokens(self):
        return len(self._prefix)


class ClefProvider(Provider):
    """Clef System One (a Qwen3.5 backbone with the joint schema head). The upstream
    module joint_schema_model.py, imported from the digest-verified model directory,
    encodes the state and the typed choice questions and returns one logit per allowed
    option of every question in a single forward pass; this adapter only maps
    Hachidori's choice questions onto it and the softmaxed logits back. Nothing is
    generated, parsed or retried.

    The model is either the pinned upstream release (model_dir) or a Hachidori variant
    of it (variant_dir): a quantized backbone with the joint head and tokenizer carried
    over unchanged. The variant is loaded from its own directory only; it never falls
    back to the source."""

    name = "clef"

    def import_provider(self):
        import safetensors.torch
        import transformers
        self.safetensors_torch = safetensors.torch
        self.transformers = transformers

    @property
    def path(self):
        return self.variant_dir or self.model_dir

    def verify_files(self):
        # Every pinned file is checked against its digest before any of it is loaded;
        # in particular before joint_schema_model.py is imported and executed.
        files = self.variant["files"] if self.variant else self.digests
        for rel, want in sorted(files.items()):
            got = file_sha256(os.path.join(self.path, *rel.split("/")))
            if got != want:
                raise RuntimeError("%s: sha256 %s, want %s" % (rel, got, want))

    def load(self):
        torch = self.torch
        self.verify_files()
        if self.variant:
            declared = self.variant["weights"]["dtype"]
            if self.dtype not in (None, declared):
                raise RuntimeError("variant %s executes at %s; %s was requested"
                                   % (self.variant["id"], declared, self.dtype))
            want = declared
        else:
            want = self.dtype or "bfloat16"
        self.want_dtype = want
        sys.path.insert(0, self.path)
        import joint_schema_model
        self.jsm = joint_schema_model
        dtype = getattr(torch, want)
        from transformers.models.qwen3_5 import modeling_qwen3_5
        self.kernel_paths = configure_clef_kernels(modeling_qwen3_5, torch, self.requested, want)
        log("Clef kernel selection: " + json.dumps(self.kernel_paths, sort_keys=True))
        # The same composition as upstream load_release_model, without the multimodal
        # processor: typed decisions are text-only, and the tokenizer is all they use.
        backbone = self.transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            self.path, dtype=dtype, device_map={"": self.requested})
        backbone.config.use_cache = False
        self.quantized = 0
        if self.variant:
            self.install_packed_linears(backbone)
        with open(os.path.join(self.path, "joint_head_config.json"), encoding="utf-8") as f:
            head_config = json.load(f)
        head = joint_schema_model.JointSchemaHead(**head_config)
        head.load_state_dict(self.safetensors_torch.load_file(os.path.join(self.path, "joint_head.safetensors")), strict=True)
        head = head.to(device=self.requested, dtype=dtype)
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(self.path)
        self.model = joint_schema_model.ClefModel(backbone, head).eval()
        self.backbone = backbone
        # Nothing may sit on another device: no offload, no partial placement.
        devices = {t.device.type for t in list(self.model.parameters()) + list(self.model.buffers())}
        if devices != {self.requested}:
            raise RuntimeError("model tensors on %s, requested only %s" % (sorted(devices), self.requested))
        if self.variant and self.quantized == 0:
            raise RuntimeError("variant %s loaded no quantized weights" % self.variant["id"])
        if not self.variant and self.quantized:
            raise RuntimeError("the source model loaded quantized weights")
        if self.trial_session:
            self.trial = self.build_trial(backbone)

    def build_trial(self, backbone):
        """The tuning trial executor over this model's Linear modules. A trial
        session is the pinned source model on the accelerator; nothing else."""
        torch = self.torch
        ops = TorchOps(torch, self.requested, self.want_dtype)
        modules = {n: m for n, m in backbone.named_modules() if isinstance(m, torch.nn.Linear)}
        return TrialExecutor(ops, modules, self.requested, self.want_dtype, shared=ops.shared_weights(backbone))

    def install_packed_linears(self, backbone):
        """Run every packed Linear at four bits. transformers would expand the whole
        model to the dense dtype on its first forward pass (a hook on the root model);
        that is removed, and each packed module selects its execution backend."""
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        torch = self.torch
        weights = self.variant["weights"]
        if weights["bits"] != 4 or not weights["symmetric"] or weights["format"] != "compressed-tensors/pack-quantized":
            raise RuntimeError("unsupported variant weights %s" % json.dumps(weights, sort_keys=True))
        hook = getattr(backbone, "ct_decompress_hook", None)
        if hook is not None:
            hook.remove()
            del backbone.ct_decompress_hook
        for _, module in backbone.named_modules():
            if not hasattr(module, "weight_packed"):
                continue
            out, inner = (int(v) for v in module.weight_shape.tolist())
            if inner % weights["group_size"] != 0:
                raise RuntimeError("packed module width %d is not a multiple of group size %d" % (inner, weights["group_size"]))
            bind_packed(torch, unpack_from_int32, module, out, inner, weights["bits"], weights["group_size"])
            self.quantized += 1

    def reference(self):
        return self.model.head.hidden_norm.weight

    def placed_device(self):
        return self.reference().device

    def placed_dtype(self):
        return self.reference().dtype

    def _resident_artifact(self):
        """Restrict continuation to the physically validated CUDA artifact.

        Runtime kernel evidence mutates its execution observation after first use;
        identity must bind only immutable selection facts, never that mutable field.
        """
        files = self.variant.get("files", {}) if self.variant else self.digests
        kernel_identity = {}
        for name, entry in sorted(self.kernel_paths.items()):
            if isinstance(entry, dict):
                kernel_identity[name] = {
                    key: entry.get(key)
                    for key in ("selected", "availability", "implementation", "reason")
                }
            else:
                kernel_identity[name] = entry
        if (self.requested != "cuda" or self.want_dtype != "bfloat16" or
                not self.variant or self.variant.get("id") !=
                "clef-flash--clef-flash-w4a16-rtn-g128--6cdd68bf9677" or
                self.torch.__version__ != "2.11.0+cu128" or
                self.transformers.__version__ != "5.17.0" or
                files.get("joint_schema_model.py") !=
                "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"):
            raise RuntimeError("Clef continuation is not certified for this execution artifact")
        facts = (files, self.variant, kernel_identity, self.want_dtype,
                 self.requested, self.torch.__version__, self.transformers.__version__)
        return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()

    def _resident_text_model(self):
        base = (self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model")
                else self.backbone)
        text = base.model
        return text.language_model if hasattr(text, "language_model") else text

    def _resident_record(self, state, questions):
        record = self.jsm.encode_record(self.tokenizer, {"state": state, "questions": questions},
                                        max_length=CLEF_MAX_LENGTH)
        if record.media is not None:
            raise ValueError("Clef resident State does not support media")
        return record

    def build_resident(self, state, questions):
        """Construct a complete admitted State prefix, never a partial cache.

        Capacity admission owns the no-silent-truncation contract. Two sentinel
        States with the same Questions then locate the upstream encoder's fixed
        pre-State boundary for the already-admitted full record.
        """
        artifact = self._resident_artifact()
        record = self._encode_inputs([state], questions)[0]
        empty = self._resident_record("", questions)
        other = self._resident_record(0, questions)
        limit = min(len(empty.input_ids), len(other.input_ids))
        start = next((i for i in range(limit)
                      if empty.input_ids[i] != other.input_ids[i]), limit)
        empty_ids = self.jsm._tokens(self.tokenizer, self.jsm.render(""))
        state_ids = self.jsm._tokens(self.tokenizer, self.jsm.render(state))
        fixed = len(empty.input_ids) - len(empty_ids)
        effective = len(record.input_ids) - fixed
        boundary = start + effective
        if (start == limit or start < 1 or effective < 0 or effective > len(state_ids) or
                tuple(empty.input_ids[start:start + len(empty_ids)]) != tuple(empty_ids) or
                tuple(record.input_ids[start:boundary]) != tuple(state_ids[:effective]) or
                tuple(record.input_ids[boundary:]) != tuple(empty.input_ids[start + len(empty_ids):])):
            raise ValueError("cannot establish exact effective Clef State prefix")
        prefix = tuple(record.input_ids[:boundary])
        identity = (artifact, hashlib.sha256(json.dumps(prefix).encode()).hexdigest())
        torch = self.torch
        cache, hidden = None, []
        with torch.inference_mode():
            for start in range(0, len(prefix), 512):
                chunk = torch.tensor([prefix[start:start + 512]], dtype=torch.long,
                                     device=self.reference().device)
                outputs = self._resident_text_model()(input_ids=chunk, past_key_values=cache,
                                                       use_cache=True, return_dict=True)
                cache = outputs.past_key_values
                if cache is None:
                    raise RuntimeError("Qwen3.5 prefill returned no continuation Cache")
                hidden.append(outputs.last_hidden_state)
            return ClefResidentState(identity, prefix, cache, torch.cat(hidden, dim=1))

    def register_resident(self, ref, state, questions):
        """Attach one admitted immutable continuation to an explicit State ref.

        The content digest names the registry object, not the execution cache:
        build_resident binds the exact effective prefix and pinned artifact.
        """
        if ref != "sha256:" + hashlib.sha256(state.encode("utf-8")).hexdigest():
            raise ValueError("state_ref does not match State content")
        if self.requested != "cuda":
            # The #270 continuation is certified only for its pinned CUDA
            # artifact; referenced CPU requests retain ordinary execution.
            return {"supported": False}
        current = getattr(self, "registered_resident", None)
        if current is None:
            try:
                self._resident_artifact()
            except RuntimeError as e:
                if "not certified" in str(e):
                    return {"supported": False}
                raise
        if current is not None and current[0] == ref:
            if current[1] != state:
                raise ValueError("State reference collision")
            # The primitive checks exact artifact and effective prefix on use.
            return self.resident_info(current[2])
        # Bound residency to one completed object; failed construction is not
        # published. #264 checks input shape before build touches the device.
        self.registered_resident = None
        self.resident_usage = {}
        self.check_capacity_readiness()
        resident = self.build_resident(state, to_typed(questions))
        self.registered_resident = (ref, state, resident)
        return self.resident_info(resident)

    def resident_info(self, resident):
        return {"supported": True, "artifact": resident.identity[0], "effective_prefix": resident.identity[1],
                "prefix_tokens": resident.prefix_tokens, "payload_bytes": resident.payload_bytes}

    def decide_resident(self, items):
        current = getattr(self, "registered_resident", None)
        if current is None:
            raise ValueError("no registered resident State")
        # Admit all questions before any suffix is forwarded. Do not reuse a
        # resident if the State content, execution artifact or prefix differs.
        for item in items:
            if item.get("state_ref") != current[0] or item["state"] != current[1]:
                raise ValueError("incompatible resident State reference")
            self._encode_inputs([item["state"]], to_typed(item["questions"]))
        self.check_capacity_readiness()
        results = []
        for item in items:
            out, usage = self.predict_resident(current[2], item["state"], to_typed(item["questions"]))
            self.resident_usage = {**self.resident_info(current[2]), **usage}
            results.append(from_typed(item["questions"], out["answers"]))
        self.sync()
        return results

    def predict_resident(self, resident, state, questions):
        """Run only a compatible suffix against a request-owned hybrid Cache."""
        if not isinstance(resident, ClefResidentState):
            raise TypeError("expected completed Clef resident State")
        artifact = self._resident_artifact()
        # Resident execution remains subject to the same pre-device capacity
        # admission and no-silent-truncation contract as ordinary Clef requests.
        record = self._encode_inputs([state], questions)[0]
        prefix = tuple(record.input_ids[:resident.prefix_tokens])
        identity = (artifact, hashlib.sha256(json.dumps(prefix).encode()).hexdigest())
        if (identity != resident.identity or prefix != resident._prefix or
                len(record.input_ids) <= resident.prefix_tokens):
            raise ValueError("incompatible Clef resident prefix or execution artifact")
        torch = self.torch
        with torch.inference_mode():
            # The Qwen3.5 cache includes attention KV and recurrent/conv state.
            cache = copy.deepcopy(resident._cache)
            fork_bytes = _clef_tensor_bytes(cache)
            suffix = torch.tensor([record.input_ids[resident.prefix_tokens:]],
                                  dtype=torch.long, device=self.reference().device)
            outputs = self._resident_text_model()(input_ids=suffix, past_key_values=cache,
                                                   use_cache=True, return_dict=True)
            if outputs.past_key_values is None:
                raise RuntimeError("Qwen3.5 suffix returned no continuation Cache")
            hidden = torch.cat((resident._hidden, outputs.last_hidden_state), dim=1)
            batch = self.jsm.collate_records([record], self.tokenizer.pad_token_id,
                                             self.reference().device)
            base = (self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model")
                    else self.backbone)
            logits = self.model.head(hidden, batch["input_ids"], batch["attention_mask"],
                                     [record], base.get_output_embeddings().weight)
            result = self._map_clef_results([record], logits, torch)[0]
            working_bytes = _clef_tensor_bytes((cache, suffix, outputs.last_hidden_state,
                                                hidden, batch["input_ids"], batch["attention_mask"]))
        return result, {"fork_bytes": fork_bytes, "working_bytes": working_bytes}

    def capacity_status(self):
        current = getattr(self, "registered_resident", None)
        resident = ({"state_ref": current[0], **self.resident_info(current[2]),
                     **getattr(self, "resident_usage", {})} if current else None)
        return {"profile": getattr(self, "capacity_profile", None),
                "model_context_tokens": CLEF_MAX_LENGTH,
                "headroom": getattr(self, "capacity_headroom", None),
                "last_rejection": getattr(self, "capacity_rejection", None),
                "resident": resident}

    def check_capacity_readiness(self):
        profile = getattr(self, "capacity_profile", None)
        if profile and profile["dtype"] != self.want_dtype:
            fatal("capacity", "capacity profile dtype does not match loaded execution dtype")
        if self.requested == "cuda" and not profile:
            fatal("capacity", "capacity_profile: " + getattr(self, "capacity_profile_error", "missing"))
        if profile and profile["max_input_tokens"] > CLEF_MAX_LENGTH:
            raise CapacityError("max_input_tokens", CLEF_MAX_LENGTH, profile["max_input_tokens"])
        if self.requested != "cuda":
            return
        free, total = self.torch.cuda.mem_get_info(self.placed_device())
        # Include reusable allocator blocks in the usable workspace budget:
        # warmup may reserve them, but they remain available for inference.
        reserved = self.torch.cuda.memory_reserved(self.placed_device())
        allocated = self.torch.cuda.memory_allocated(self.placed_device())
        usable = free + max(0, reserved - allocated)
        required = profile["required_gpu_headroom_bytes"]
        self.capacity_headroom = {"required_bytes": required, "usable_bytes": usable,
                                  "device_total_bytes": total, "resident_allocated_bytes": allocated,
                                  "allocator_reserved_bytes": reserved}
        if usable < required:
            raise CapacityError("gpu_headroom_bytes", required, usable)

    def _encode_inputs(self, states, questions):
        if not hasattr(self, "batch_profile"):
            self.batch_profile = {"forwards": 0, "records": 0, "encode_ms": 0.0,
                                  "collate_ms": 0.0, "model_enqueue_ms": 0.0, "post_transfer_ms": 0.0,
                                  "padded_tokens": 0}
        start = time.perf_counter()
        encoded = []
        profile = getattr(self, "capacity_profile", None)
        for state in states:
            # The pinned encoder slices to max_length. sys.maxsize disables
            # that slicing; reject the full CPU shape before device collation.
            record = self.jsm.encode_record(self.tokenizer, {"state": state, "questions": questions},
                                            max_length=sys.maxsize)
            limit = min(CLEF_MAX_LENGTH, profile["max_input_tokens"]) if profile else CLEF_MAX_LENGTH
            self._capacity_limit("input_tokens", limit, len(record.input_ids))
            if profile:
                state_count = len(self.jsm._tokens(self.tokenizer, self.jsm.render(state)))
                self._capacity_limit("state_tokens", profile["max_state_tokens"], state_count)
            encoded.append(record)
        self.batch_profile["encode_ms"] += (time.perf_counter() - start) * 1000
        for batch in self._encoded_batches(encoded):
            if profile:
                self._capacity_limit("batch_items", profile["max_batch_items"], len(batch))
                self._capacity_limit("batch_padded_tokens", profile["max_batch_padded_tokens"],
                                     len(batch) * max(len(r.input_ids) for _, r in batch))
        return encoded

    def _capacity_limit(self, metric, limit, observed):
        if observed > limit:
            error = CapacityError(metric, limit, observed)
            self.capacity_rejection = error.details
            raise error

    def _encoded_batches(self, encoded):
        pending = []
        for index, record in sorted(enumerate(encoded), key=lambda pair: len(pair[1].input_ids)):
            length = len(record.input_ids)
            if pending and (len(pending) >= 8 or
                            max(max(len(r.input_ids) for _, r in pending), length) * (len(pending) + 1) > 8192 or
                            length > 2 * min(len(r.input_ids) for _, r in pending) or
                            2 * length < max(len(r.input_ids) for _, r in pending)):
                yield pending
                pending = []
            pending.append((index, record))
        if pending:
            yield pending

    def decide(self, items):
        # Admit every question group before executing any group on the GPU.
        groups = {}
        for i, item in enumerate(items):
            groups.setdefault(json.dumps(item["questions"], sort_keys=True), []).append(i)
        prepared = []
        for indices in groups.values():
            questions = to_typed(items[indices[0]]["questions"])
            records = self._encode_inputs([items[i]["state"] for i in indices], questions)
            prepared.append((indices, items[indices[0]]["questions"], records))
        results = [None] * len(items)
        for indices, questions, records in prepared:
            outputs = self._predict_encoded(records)
            for i, out in zip(indices, outputs):
                results[i] = from_typed(questions, out["answers"])
        self.sync()
        return results

    def predict(self, states, questions):
        return self._predict_encoded(self._encode_inputs(states, questions))

    def _predict_encoded(self, encoded):
        out = [None] * len(encoded)
        for pending in self._encoded_batches(encoded):
            batch = self._predict_batch([r for _, r in pending], self.torch)
            for (original, _), result in zip(pending, batch):
                out[original] = result
        return out

    def _predict_batch(self, records, torch):
        self.batch_profile["padded_tokens"] += len(records) * max(len(r.input_ids) for r in records)
        start = time.perf_counter()
        batch = self.jsm.collate_records(records, self.tokenizer.pad_token_id, self.reference().device)
        self.batch_profile["collate_ms"] += (time.perf_counter() - start) * 1000
        with torch.inference_mode():
            start = time.perf_counter()
            all_logits = self.model(batch)
            self.batch_profile["forwards"] += 1
            self.batch_profile["records"] += len(records)
            # On CUDA this is enqueue time, not synchronized GPU execution time.
            self.batch_profile["model_enqueue_ms"] += (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            out = self._map_clef_results(records, all_logits, torch)
            self.batch_profile["post_transfer_ms"] += (time.perf_counter() - start) * 1000
        return out

    def _map_clef_results(self, records, all_logits, torch):
        # Ragged question sizes require separate softmaxes and one host extraction.
        flat = [qlogits.float().softmax(-1) for logits in all_logits for qlogits in logits]
        values = torch.cat(flat).tolist()
        offset = 0
        out = []
        for record, logits in zip(records, all_logits):
            answers = {}
            for question, _ in zip(record.questions, logits):
                count = len(question.option_ids)
                by_option = dict(zip(question.option_ids, values[offset:offset + count]))
                offset += count
                choice = max(question.option_ids, key=by_option.__getitem__)
                answers[question.question_id] = {"choice": choice, "answer_confidence": by_option[choice],
                                                 "probabilities": by_option}
            out.append({"answers": answers})
        return out

    def version(self):
        return "clef/" + self.transformers.__version__

    def extra_info(self):
        execution = "variant" if self.variant else ("trial" if self.trial_session else "source")
        info = {"transformers_version": self.transformers.__version__, "weights_quantized_modules": self.quantized,
                "execution": execution, "kernel_paths": self.kernel_paths}
        info["capacity"] = self.capacity_status()
        if getattr(self, "registered_resident", None) is not None:
            info["resident_state"] = {"state_ref": self.registered_resident[0],
                                      **self.resident_info(self.registered_resident[2]),
                                      **getattr(self, "resident_usage", {})}
        info["w4_linear_paths"] = {name: dict(module._hachidori_w4[0])
                                   for name, module in self.backbone.named_modules()
                                   if hasattr(module, "_hachidori_w4")}
        if self.trial_session:
            info["trial_session"] = True
            info["trial_transform"] = TRIAL_TRANSFORM
        if self.variant:
            weights = self.variant["weights"]
            info["quantization_scheme"] = weights["scheme"]
            info["quantized_execution"] = "%s %s weights, %s compute" % (weights["format"], weights["scheme"], self.want_dtype)
        return info


PROVIDERS = {"laya": LayaProvider, "opendecider": OpenDeciderProvider, "clef": ClefProvider}


def serve(provider):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            fatal("protocol", "invalid request line: %s" % e)
        rid = req.get("id")
        op = req.get("op")
        if op == "shutdown":
            emit({"id": rid, "ok": True})
            return
        try:
            if op == "resident_register":
                if not isinstance(provider, ClefProvider):
                    raise ValueError("resident State requires Clef")
                emit({"id": rid, "ok": True, "result": provider.register_resident(
                    req["state_ref"], req["state"], req["questions"])})
            elif op == "resident_decide":
                if not isinstance(provider, ClefProvider):
                    raise ValueError("resident State requires Clef")
                t0 = time.perf_counter()
                results = provider.decide_resident(req["items"])
                emit({"id": rid, "ok": True, "results": results,
                      "inference_ms": (time.perf_counter() - t0) * 1000.0})
            elif op == "decide":
                t0 = time.perf_counter()
                results = provider.decide(req["items"])
                emit({"id": rid, "ok": True, "results": results,
                      "inference_ms": (time.perf_counter() - t0) * 1000.0})
            elif op == "stats":
                emit({"id": rid, "ok": True, "stats": provider.stats()})
            elif isinstance(op, str) and op.startswith("trial_"):
                if provider.trial is None:
                    emit({"id": rid, "ok": False,
                          "error": {"class": "request_invalid", "message": "this worker is not a tuning trial session"}})
                else:
                    emit({"id": rid, "ok": True, "result": trial_dispatch(provider.trial, op, req)})
            else:
                emit({"id": rid, "ok": False,
                      "error": {"class": "request_invalid", "message": "unknown op %r" % op}})
        except CapacityError as e:
            emit({"id": rid, "ok": False,
                  "error": {"class": "capacity", "message": str(e), "capacity": e.details}})
        except TrialError as e:
            emit({"id": rid, "ok": False, "error": {"class": e.cls, "message": str(e)}})
        except (ValueError, TypeError, KeyError) as e:
            emit({"id": rid, "ok": False,
                  "error": {"class": "request_invalid", "message": "%s: %s" % (type(e).__name__, e)}})
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc())
            emit({"id": rid, "ok": False,
                  "error": {"class": "inference_failed", "message": "%s: %s" % (type(e).__name__, e)}})


def main():
    # The supervisor owns the lifecycle: a console Ctrl+C reaches the whole process
    # group, but the worker must only stop on shutdown or stdin EOF.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", required=True, choices=["cuda", "cpu"])
    ap.add_argument("--manifest", required=True, help="model manifest with pinned file digests")
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS), help="provider that loads the model")
    ap.add_argument("--dtype", choices=sorted(set(OPENDECIDER_DTYPES + CLEF_DTYPES)),
                    help="opendecider: inference dtype (default float32); clef: bfloat16 (default) or float32 on the source model")
    ap.add_argument("--variant-dir", help="clef only: execute this Hachidori variant of the model instead of the source")
    ap.add_argument("--variant-manifest", help="clef only: the variant manifest (hachidori.variant/1) of --variant-dir")
    ap.add_argument("--trial-session", action="store_true",
                    help="clef only: serve the pinned source model as a RAM-resident tuning trial session")
    ap.add_argument("--capacity-profile", help="exact target-bound configured Clef capacity JSON")
    ap.add_argument("--capacity-profile-error", help="typed missing/invalid profile condition from host")
    args = ap.parse_args()
    if args.dtype and args.provider not in ("opendecider", "clef"):
        ap.error("--dtype is only supported by the opendecider and clef providers")
    if bool(args.variant_dir) != bool(args.variant_manifest):
        ap.error("--variant-dir and --variant-manifest go together")
    if args.variant_dir and args.provider != "clef":
        ap.error("variants are only supported by the clef provider")
    if args.trial_session and (args.provider != "clef" or args.variant_dir):
        ap.error("--trial-session is only supported by the clef provider on the source model")
    emit({"event": "hello", "protocol": PROTOCOL, "pid": os.getpid()})
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    variant = None
    if args.variant_manifest:
        with open(args.variant_manifest, encoding="utf-8") as f:
            variant = json.load(f)
    provider = PROVIDERS[args.provider](args.model_dir, args.device, manifest, args.dtype, args.variant_dir, variant, args.trial_session)
    if args.provider == "clef":
        provider.capacity_profile = json.loads(args.capacity_profile) if args.capacity_profile else None
        provider.capacity_profile_error = args.capacity_profile_error or "missing"
    if args.provider == "clef" and args.device == "cuda" and not provider.capacity_profile:
        fatal("capacity", "capacity_profile: " + provider.capacity_profile_error)
    provider.initialize()
    provider.warmup()
    emit({"event": "ready", "info": provider.info()})
    serve(provider)


if __name__ == "__main__":
    main()
