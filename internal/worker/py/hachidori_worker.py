"""Hachidori private inference worker (Laya provider).

Internal implementation detail of the Hachidori runtime. It is started by the Go
supervisor with an explicitly constructed environment and speaks newline-delimited
JSON over stdin/stdout. It never opens a network listener.

Lifecycle: import provider -> load pinned model -> move to device -> warm up -> ready,
then serve requests until stdin closes or a shutdown request arrives.

Library output written to stdout (Laya prints warnings there) is redirected to stderr
so the protocol channel carries protocol messages only.
"""
import argparse
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


def to_laya(questions):
    out = {}
    for q in questions:
        desc = q.get("descriptions") or {}
        out[q["id"]] = {
            "type": "choice",
            "instructions": q["instructions"],
            "criteria": {c: desc.get(c) for c in q["choices"]},
        }
    return out


def from_laya(questions, answers):
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


class LayaProvider:
    def __init__(self, model_dir, device, digests):
        self.model_dir = model_dir
        self.requested = device
        self.digests = digests
        self.agent = None
        self.torch = None

    def initialize(self):
        emit({"event": "phase", "phase": "importing"})
        try:
            import torch
            import laya
        except Exception as e:  # noqa: BLE001 - any import failure is a provider failure
            fatal("provider_import", "%s: %s" % (type(e).__name__, e))
        self.torch = torch
        self.laya = laya
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
            self.agent = laya.load(self.model_dir, device=self.requested,
                                   expected_sha256=self.digests)
        except Exception as e:  # noqa: BLE001
            fatal("model_load", "%s: %s" % (type(e).__name__, e))
        # Laya silently falls back to CPU when a device placement fails; that is a
        # device failure for Hachidori, not a degraded success.
        if self.agent.device.type != self.requested:
            fatal("device_unavailable", "model placed on %s instead of requested %s"
                  % (self.agent.device, self.requested))
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
            outs = self.agent.predict_batch([items[i]["state"] for i in idxs], to_laya(questions))
            for i, out in zip(idxs, outs):
                results[i] = from_laya(questions, out["answers"])
        self.sync()
        return results

    def info(self):
        torch = self.torch
        info = {
            "provider": "laya",
            "laya_version": self.laya.__version__,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "device": str(self.agent.device),
            "dtype": str(self.agent.dtype),
            "model_dir": self.model_dir,
            "load_ms": round(self.load_ms, 1),
            "warmup_ms": round(self.warmup_ms, 1),
            "no_user_site": bool(sys.flags.no_user_site),
            "hf_home": os.environ.get("HF_HOME", ""),
        }
        if self.requested == "cuda":
            idx = self.agent.device.index or 0
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
    args = ap.parse_args()
    emit({"event": "hello", "protocol": PROTOCOL, "pid": os.getpid()})
    with open(args.manifest, encoding="utf-8") as f:
        digests = json.load(f)["files"]
    provider = LayaProvider(args.model_dir, args.device, digests)
    provider.initialize()
    provider.warmup()
    emit({"event": "ready", "info": provider.info()})
    serve(provider)


if __name__ == "__main__":
    main()
