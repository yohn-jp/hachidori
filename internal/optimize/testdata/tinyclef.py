"""Builds a tiny, deterministic, randomly initialized System One model laid out like
the pinned Clef-Flash release (same module graph and file names, a few hundred
kilobytes), for the optional end-to-end tests of the optimizer and the System One
provider. No real weights are involved and nothing is downloaded.

    python tinyclef.py --out DIR --joint-schema PATH/TO/joint_schema_model.py

joint_schema_model.py is the upstream module of the release being exercised; it is
copied unchanged because the provider imports it from the model directory.
"""
import argparse
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

BASE = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "dtype": "bfloat16",
    "image_token_id": 500,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True, "bos_token_id": None,
        "dtype": "bfloat16", "eos_token_id": 1, "full_attention_interval": 4, "head_dim": 64,
        "hidden_act": "silu", "hidden_size": 256, "initializer_range": 0.02, "intermediate_size": 512,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_num_key_heads": 4,
        "linear_num_value_heads": 8, "linear_value_head_dim": 32, "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096, "mlp_only_layers": [], "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": False, "num_attention_heads": 4,
        "num_hidden_layers": 4, "num_key_value_heads": 2, "pad_token_id": None, "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "rope_type": "default"},
        "tie_word_embeddings": False, "use_cache": True, "vocab_size": 512,
    },
    "tie_word_embeddings": False,
    "video_token_id": 501,
    "vision_config": {
        "deepstack_visual_indexes": [], "depth": 2, "dtype": "bfloat16", "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 128, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 256,
        "model_type": "qwen3_5_vision", "num_heads": 4, "num_position_embeddings": 64, "out_hidden_size": 256,
        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
    },
    "vision_end_token_id": 499,
    "vision_start_token_id": 498,
}
HEAD = {"hidden_size": 256, "width": 64, "routing_layers": 1, "layers": 1, "heads": 4, "feedforward": 64}
SPECIAL = ["<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
CORPUS = [
    "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of the allowed options.",
    "SCHEMA FIELDS: FIELD ID TYPE choice INSTRUCTION ALLOWED OPTIONS OPTION END FIELD STATE JOINT SCHEMA DECISIONS",
    "The user asked to rename a variable in utils.py; the change is complete and tests pass. Is the requested work complete? yes no",
    "option_id description true false noul score choice billing technical urgent routine",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--joint-schema", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(BASE)).to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.joint_schema)))
    import joint_schema_model as mod
    head = mod.JointSchemaHead(**HEAD).to(torch.bfloat16)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, os.path.join(args.out, "joint_head.safetensors"))
    with open(os.path.join(args.out, "joint_head_config.json"), "w") as f:
        json.dump(HEAD, f)
    shutil.copyfile(args.joint_schema, os.path.join(args.out, "joint_schema_model.py"))

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=480, special_tokens=SPECIAL,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(os.path.join(args.out, "tokenizer.json"))
    with open(os.path.join(args.out, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "pad_token": "<|endoftext|>", "eos_token": "<|im_end|>"}, f)

    # The remaining files of the release layout, as placeholders.
    for name, body in (("LICENSE", "fixture license\n"), ("chat_template.jinja", "{{ messages }}\n"), ("processor_config.json", "{}\n")):
        with open(os.path.join(args.out, name), "w") as f:
            f.write(body)

    files = {}
    for root, _, names in os.walk(args.out):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, args.out).replace(os.sep, "/")
            if rel != "hachidori-model.json":
                files[rel] = sha256(p)
    manifest = {"id": "clef-flash", "provider": "clef", "repo": "tiny/clef-flash-fixture", "revision": "0" * 40,
                "description": "tiny deterministic fixture", "files": files}
    with open(os.path.join(args.out, "hachidori-model.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps({"out": args.out, "files": len(files)}))


if __name__ == "__main__":
    main()
