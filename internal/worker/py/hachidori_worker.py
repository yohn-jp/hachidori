"""Hachidori private inference worker.

Internal implementation detail of the Hachidori runtime. It is started by the Go
supervisor with an explicitly constructed environment and speaks newline-delimited
JSON over stdin/stdout. It never opens a network listener.

One provider adapter serves the single active model: Laya or OpenDecider-nano.
Both score the request's closed choices and return a probability distribution;
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

    def __init__(self, model_dir, device, manifest):
        self.model_dir = model_dir
        self.requested = device
        self.manifest = manifest
        self.digests = manifest["files"]
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
        if self.requested == "cuda":
            idx = device.index or 0
            info["device_name"] = torch.cuda.get_device_name(idx)
            info["device_capability"] = list(torch.cuda.get_device_capability(idx))
        return info

    def stats(self):
        if self.requested != "cuda":
            return {}
        torch = self.torch
        free, total = torch.cuda.mem_get_info()
        return {
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "memory_free": free,
            "memory_total": total,
        }


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
        # resolved from the Hub. float32 is the dtype upstream evaluated.
        self.model = self.opendecider.load(self.model_dir, device=self.requested, dtype="float32")

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


PROVIDERS = {"laya": LayaProvider, "opendecider": OpenDeciderProvider}


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
    args = ap.parse_args()
    emit({"event": "hello", "protocol": PROTOCOL, "pid": os.getpid()})
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    provider = PROVIDERS[args.provider](args.model_dir, args.device, manifest)
    provider.initialize()
    provider.warmup()
    emit({"event": "ready", "info": provider.info()})
    serve(provider)


if __name__ == "__main__":
    main()
