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
import hashlib
import json
import os
import signal
import sys
import time
import traceback

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


class Provider:
    """The provider-neutral part of the resident model: device policy, lifecycle
    phases, warmup, grouping of requests and status. Adapters supply import, load,
    prediction and the facts they actually observed about the loaded model."""

    name = ""

    def __init__(self, model_dir, device, manifest, dtype=None, variant_dir=None, variant=None):
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
            import torch
            self.import_provider()
        except Exception as e:  # noqa: BLE001 - any import failure is a provider failure
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
            fatal("model_load", "%s: %s" % (type(e).__name__, e))
        # A requested device is never substituted: a model that did not land on it
        # is a device failure for Hachidori, not a degraded success.
        placed = self.placed_device()
        if placed.type != self.requested:
            fatal("device_unavailable", "model placed on %s instead of requested %s"
                  % (placed, self.requested))
        self.load_ms = (time.perf_counter() - t0) * 1000.0

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
        except Exception as e:  # noqa: BLE001
            fatal("warmup", "%s: %s" % (type(e).__name__, e))
        self.warmup_ms = (time.perf_counter() - t0) * 1000.0

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


def packed_forward(self, x):
    """Linear forward of a module that holds its weight as compressed-tensors
    pack-quantized int4 (weight_packed, weight_scale): the weight is expanded for this
    one call only, so the module stays at four bits in memory. This is exactly the
    dequantization compressed-tensors itself applies (pack_quantized, symmetric, group
    quantization), without ever materializing the dense model."""
    out, inner, bits, group = self._hachidori_packed
    q = self._unpack(self.weight_packed, bits, self.torch_Size((out, inner)))
    w = (q.to(x.dtype).view(out, inner // group, group) * self.weight_scale.to(x.dtype).unsqueeze(-1)).view(out, inner)
    return self.torch_linear(x, w, self.bias)


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

    def install_packed_linears(self, backbone):
        """Run every packed Linear at four bits. transformers would expand the whole
        model to the dense dtype on its first forward pass (a hook on the root model);
        that is removed, and each packed module dequantizes its own weight per call."""
        import types
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
            module._hachidori_packed = (out, inner, weights["bits"], weights["group_size"])
            module._unpack = unpack_from_int32
            module.torch_Size = torch.Size
            module.torch_linear = torch.nn.functional.linear
            module.forward = types.MethodType(packed_forward, module)
            self.quantized += 1

    def reference(self):
        return self.model.head.hidden_norm.weight

    def placed_device(self):
        return self.reference().device

    def placed_dtype(self):
        return self.reference().dtype

    def predict(self, states, questions):
        torch = self.torch
        out = []
        for state in states:
            record = {"state": state, "questions": questions}
            encoded = self.jsm.encode_record(self.tokenizer, record, max_length=CLEF_MAX_LENGTH)
            if len(encoded.input_ids) >= CLEF_MAX_LENGTH:
                log("state truncated to the model's %d-token context" % CLEF_MAX_LENGTH)
            batch = self.jsm.collate_records([encoded], self.tokenizer.pad_token_id, self.reference().device)
            with torch.inference_mode():
                logits = self.model(batch)[0]
            answers = {}
            for question, qlogits in zip(encoded.questions, logits):
                probs = qlogits.float().softmax(-1).tolist()
                by_option = dict(zip(question.option_ids, probs))
                choice = max(question.option_ids, key=by_option.__getitem__)
                answers[question.question_id] = {"choice": choice, "answer_confidence": by_option[choice],
                                                 "probabilities": by_option}
            out.append({"answers": answers})
        return out

    def version(self):
        return "clef/" + self.transformers.__version__

    def extra_info(self):
        info = {"transformers_version": self.transformers.__version__, "weights_quantized_modules": self.quantized,
                "execution": "variant" if self.variant else "source"}
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
            if op == "decide":
                t0 = time.perf_counter()
                results = provider.decide(req["items"])
                emit({"id": rid, "ok": True, "results": results,
                      "inference_ms": (time.perf_counter() - t0) * 1000.0})
            elif op == "stats":
                emit({"id": rid, "ok": True, "stats": provider.stats()})
            else:
                emit({"id": rid, "ok": False,
                      "error": {"class": "request_invalid", "message": "unknown op %r" % op}})
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
    args = ap.parse_args()
    if args.dtype and args.provider not in ("opendecider", "clef"):
        ap.error("--dtype is only supported by the opendecider and clef providers")
    if bool(args.variant_dir) != bool(args.variant_manifest):
        ap.error("--variant-dir and --variant-manifest go together")
    if args.variant_dir and args.provider != "clef":
        ap.error("variants are only supported by the clef provider")
    emit({"event": "hello", "protocol": PROTOCOL, "pid": os.getpid()})
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    variant = None
    if args.variant_manifest:
        with open(args.variant_manifest, encoding="utf-8") as f:
            variant = json.load(f)
    provider = PROVIDERS[args.provider](args.model_dir, args.device, manifest, args.dtype, args.variant_dir, variant)
    provider.initialize()
    provider.warmup()
    emit({"event": "ready", "info": provider.info()})
    serve(provider)


if __name__ == "__main__":
    main()
