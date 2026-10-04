"""Hachidori private System One optimizer.

Internal implementation detail of the Hachidori optimizer runtime. It is started by
the Go builder with an explicitly constructed environment and speaks newline-delimited
JSON on stdout (events only). It never opens a network listener and it performs no
model discovery: it receives one already materialized, digest-verified source
directory and one canonical Hachidori recipe, and writes the quantized model into the
staging directory it is given. Publication, digests and the variant manifest belong
to the Go builder; nothing written here is valid until the builder publishes it.

Lifecycle: import -> load source -> prepare -> quantize -> serialize -> verify -> done.

The first backend is LLM Compressor (compressed-tensors): a data-free, weight-only
W4A16 round-to-nearest recipe on the Linear modules of the Clef backbone. Modules the
recipe declares as preserved are excluded from quantization (the compressed-tensors
ignore list) and stay in the source precision; the output metadata is checked for
exactly that. No scheme is ever substituted: a recipe the backend cannot apply as
declared fails.

Device policy. The Go builder names the concrete device (--device cpu|cuda) and starts
the interpreter of the matching runtime flavor. This process never chooses or changes
it: it resolves the device the backend will use BEFORE loading the source, requires it
to be the requested one, reports it, and fails with a typed class (cuda_unavailable,
device_mismatch, cuda_out_of_memory) rather than continuing on another device.

CUDA does not mean whole-model residency. The source is loaded on the host and stays in
host memory; no code here or in the pinned backend calls model.cuda() or model.to().
In the pinned llmcompressor 0.14.0 / compressed-tensors 0.19.0 the data-free pipeline
(llmcompressor.pipelines.data_free.pipeline.DataFreePipeline) calls
llmcompressor.utils.dev.get_main_device() and
compressed_tensors.offload.set_onload_device(model, device): every module is wrapped in
a CPU offload cache whose tensors are moved to the accelerator only when the module's
weight is accessed (weight observation, then per-module compression when saving) and
released afterwards. get_main_device() is also what silently falls back to the cpu
("No accelerator available") when torch has no accelerator; the explicit check below
makes that a failure when cuda was requested.
"""
import argparse
import json
import os
import re
import shutil
import sys
import traceback

PROTOCOL = "hachidori-optimizer.v1"

_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

REPORT = "hachidori-optimizer-report.json"

# Files the backend itself writes for the backbone; everything else a variant needs
# is carried over, byte for byte, from the verified source as the recipe declares.
BACKBONE_OUTPUTS = ("config.json", "generation_config.json", "model.safetensors.index.json")


def emit(msg):
    _proto.write(json.dumps(msg, separators=(",", ":")) + "\n")
    _proto.flush()


def log(text):
    sys.stderr.write("[optimizer] %s\n" % text)
    sys.stderr.flush()


def fail(cls, message):
    emit({"event": "fatal", "class": cls, "message": message})
    log("fatal %s: %s" % (cls, message))
    sys.exit(3)


def phase(name, detail=""):
    emit({"event": "phase", "phase": name, "detail": detail})


def pattern_matches(pattern, name):
    """compressed-tensors ignore semantics: "re:" prefix is a regular expression
    matched from the start of the module name, anything else is an exact name."""
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return pattern == name


def module_precision(module):
    weight = getattr(module, "weight", None)
    if weight is not None:
        return str(weight.dtype).replace("torch.", "")
    packed = getattr(module, "weight_packed", None)
    return "packed:" + str(packed.dtype).replace("torch.", "") if packed is not None else "unknown"


def resolve_backend_device(torch, get_main_device, requested):
    """The device the pinned backend's data-free pipeline will onload to, checked against
    the requested one. Fails (typed) instead of continuing on another device."""
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.version.cuda is None:
            fail("cuda_unavailable", "cuda was requested but torch %s (CUDA %s) reports no usable CUDA device; "
                 "the build is not continued on the cpu" % (torch.__version__, torch.version.cuda))
    try:
        backend = get_main_device()
    except Exception as e:  # noqa: BLE001
        fail("device_mismatch", "the backend could not resolve its device: %s: %s" % (type(e).__name__, e))
    if backend.type != requested:
        fail("cuda_unavailable" if requested == "cuda" else "device_mismatch",
             "%s was requested but the backend would run on %s; the build is not continued on another device" % (requested, backend))
    return backend


def device_event(torch, requested, backend):
    ev = {"event": "device", "device": requested, "backend": str(backend)}
    if requested == "cuda":
        free, total = torch.cuda.mem_get_info(backend)
        ev.update({"name": torch.cuda.get_device_name(backend), "cuda": torch.version.cuda,
                   "vram_total_bytes": int(total), "vram_free_bytes": int(free)})
    return ev


def failure_class(torch, e, default):
    """cuda_out_of_memory for an accelerator allocation failure, otherwise the stage's class."""
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return "cuda_out_of_memory" if oom is not None and isinstance(e, oom) else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--recipe", required=True, help="canonical Hachidori recipe (JSON)")
    ap.add_argument("--out", required=True, help="empty staging directory")
    ap.add_argument("--device", required=True, choices=["cpu", "cuda"], help="concrete device the quantization runs on; never changed by this process")
    args = ap.parse_args()
    emit({"event": "hello", "protocol": PROTOCOL, "pid": os.getpid()})

    with open(args.recipe, encoding="utf-8") as f:
        recipe = json.load(f)
    if recipe.get("engine") != "llmcompressor" or recipe.get("scheme") != "W4A16" or recipe.get("algorithm") != "rtn":
        fail("recipe_unsupported", "this optimizer applies engine llmcompressor, scheme W4A16, algorithm rtn; got %s/%s/%s"
             % (recipe.get("engine"), recipe.get("scheme"), recipe.get("algorithm")))
    if os.path.exists(args.out) and os.listdir(args.out):
        fail("staging_not_empty", "staging directory %s is not empty" % args.out)
    os.makedirs(args.out, exist_ok=True)

    phase("importing")
    try:
        import importlib.metadata as md
        import torch
        import transformers
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.utils.dev import get_main_device
        from transformers import Qwen3_5ForConditionalGeneration
    except Exception as e:  # noqa: BLE001
        fail("optimizer_import", "%s: %s" % (type(e).__name__, e))
    versions = {d: md.version(d) for d in ("llmcompressor", "compressed-tensors", "transformers", "torch", "accelerate")}
    emit({"event": "engine", "engine": "llmcompressor", "version": versions["llmcompressor"], "versions": versions})

    # Resolve the device the backend will use before anything is loaded and require it
    # to be the requested one: no silent cuda -> cpu fallback, no accidental cpu -> cuda.
    backend_device = resolve_backend_device(torch, get_main_device, args.device)
    emit(device_event(torch, args.device, backend_device))

    backbone_patterns = [p["pattern"] for p in recipe.get("preserved", []) if p.get("scope") == "backbone"]

    phase("loading_source", args.source_dir)
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.source_dir, dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        fail("source_load", "%s: %s" % (type(e).__name__, e))
    model.eval()

    phase("preparing")
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    preserved = {n: [p for p in backbone_patterns if pattern_matches(p, n)] for n in linears}
    preserved = {n: ps for n, ps in preserved.items() if ps}
    unmatched = [p for p in backbone_patterns if not any(pattern_matches(p, n) for n in linears)]
    if unmatched:
        # A preserved-module declaration that matches no module of the pinned graph is a
        # recipe that does not describe this model, not something to ignore silently.
        fail("preserved_unmatched", "preserved patterns match no Linear module of the source: %s" % ", ".join(unmatched))
    targets = [n for n in linears if n not in preserved]
    emit({"event": "plan", "linear_modules": len(linears), "quantized": len(targets), "preserved": len(preserved)})

    phase("quantizing", "%d Linear modules" % len(targets))
    try:
        modifier = QuantizationModifier(targets=list(recipe["targets"]), scheme="W4A16", ignore=backbone_patterns)
        oneshot(model=model, recipe=modifier)
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        fail(failure_class(torch, e, "quantize"), "%s: %s" % (type(e).__name__, e))

    quantized = sorted(n for n, m in model.named_modules() if getattr(m, "quantization_scheme", None) is not None
                       and isinstance(m, torch.nn.Linear))
    if sorted(targets) != quantized:
        fail("quantize", "quantized %d modules, recipe selects %d" % (len(quantized), len(targets)))
    precision = {n: module_precision(dict(model.named_modules())[n]) for n in preserved}

    phase("serializing")
    try:
        model.save_pretrained(args.out, save_compressed=True)
        carried = []
        for rel in recipe.get("carry", []):
            src = os.path.join(args.source_dir, *rel.split("/"))
            dst = os.path.join(args.out, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            carried.append(rel)
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        fail(failure_class(torch, e, "serialize"), "%s: %s" % (type(e).__name__, e))

    phase("verifying")
    try:
        with open(os.path.join(args.out, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        if qc.get("quant_method") != "compressed-tensors":
            raise RuntimeError("saved config has quant_method %r, want compressed-tensors" % qc.get("quant_method"))
        # compressed-tensors resolves the recipe's patterns to the module names they matched;
        # the saved ignore list must be exactly the modules the recipe preserves.
        saved_ignore = set(qc.get("ignore") or [])
        if saved_ignore != set(preserved):
            raise RuntimeError("saved quantization_config.ignore differs from the preserved modules: missing %s, unexpected %s"
                               % (sorted(set(preserved) - saved_ignore), sorted(saved_ignore - set(preserved))))
        groups = qc.get("config_groups") or {}
        weights = [g.get("weights") or {} for g in groups.values()]
        if len(weights) != 1 or weights[0].get("num_bits") != 4 or weights[0].get("type") != "int":
            raise RuntimeError("saved scheme is not a single int4 weight group: %s" % json.dumps(groups, sort_keys=True))
        if (weights[0].get("input_activations") or groups[next(iter(groups))].get("input_activations")) is not None:
            raise RuntimeError("saved scheme quantizes activations; the recipe is weight-only")
        # Preserved modules must hold their source precision in the written tensors.
        from safetensors import safe_open
        index_path = os.path.join(args.out, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        else:
            weight_map = None
        dtypes = {}
        shards = sorted({v for v in weight_map.values()}) if weight_map else ["model.safetensors"]
        for shard in shards:
            with safe_open(os.path.join(args.out, shard), framework="pt") as f:
                for key in f.keys():
                    dtypes[key] = str(f.get_slice(key).get_dtype())
        written = {}
        for name in preserved:
            key = name + ".weight"
            if key not in dtypes:
                raise RuntimeError("preserved module %s has no %s tensor in the output" % (name, key))
            if dtypes[key] not in ("BF16", "F16", "F32"):
                raise RuntimeError("preserved module %s was written as %s, not a higher precision" % (name, dtypes[key]))
            written[name] = dtypes[key]
        for name in targets:
            if name + ".weight_packed" not in dtypes:
                raise RuntimeError("quantized module %s was not written packed" % name)
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        fail("verify", "%s: %s" % (type(e).__name__, e))

    report = {
        "schema": "hachidori.optimizer-report/1",
        "engine": "llmcompressor",
        "versions": versions,
        "scheme": recipe["scheme"],
        "algorithm": recipe["algorithm"],
        "device": args.device,
        "linear_modules": len(linears),
        "quantized_modules": len(quantized),
        "preserved_modules": {n: {"patterns": preserved[n], "source_precision": precision[n], "written_dtype": written[n]}
                              for n in sorted(preserved)},
        "carried_files": carried,
        "quantization_config_ignore": qc.get("ignore"),
    }
    with open(os.path.join(args.out, REPORT), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    emit({"event": "done", "report": REPORT})


if __name__ == "__main__":
    main()
