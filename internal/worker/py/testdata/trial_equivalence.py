"""A trial component equals what the Forge optimizer writes, bit for bit.

Quantizes a tiny randomly initialized Qwen3.5 backbone exactly as the optimizer
does (LLM Compressor oneshot, W4A16 round to nearest, the canonical preserved
modules ignored, save_compressed) and compares every quantized module's
weight_packed, weight_scale and weight_shape with the component the trial
executor builds from the same source weights. Needs torch, transformers,
compressed-tensors and llmcompressor at the pinned versions, so it reports
itself skipped where they are absent; it is evidence about the transformation on
CPU, not about the accelerator or the Windows runtime.
"""
import importlib.util
import os
import re
import runpy
import sys
import tempfile
import unittest

try:
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
except Exception as e:  # noqa: BLE001
    print('SKIP: %s' % e)
    sys.exit(0)

worker = runpy.run_path(sys.argv.pop(1))
tinyclef = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location('tinyclef', tinyclef)
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

IGNORE = ['lm_head', r're:.*linear_attn\.in_proj_a$', r're:.*linear_attn\.in_proj_b$', r're:model\.visual.*']


def matches(pattern, name):
    return re.match(pattern[3:], name) is not None if pattern.startswith('re:') else pattern == name


class Equivalence(unittest.TestCase):
    def test_component_bytes_equal_the_optimizer_output(self):
        torch.manual_seed(7)
        source = tempfile.mkdtemp()
        Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(tc.BASE)).to(torch.bfloat16).save_pretrained(source)
        load = lambda: Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16).eval()
        built, resident = load(), load()
        linears = {n: m for n, m in resident.named_modules() if isinstance(m, torch.nn.Linear)}
        targets = [n for n in linears if not any(matches(p, n) for p in IGNORE)]
        self.assertGreater(len(targets), 4)
        oneshot(model=built, recipe=QuantizationModifier(targets=['Linear'], scheme='W4A16', ignore=IGNORE))
        out = tempfile.mkdtemp()
        built.save_pretrained(out, save_compressed=True)
        written = {}
        for f in os.listdir(out):
            if f.endswith('.safetensors'):
                with safe_open(os.path.join(out, f), framework='pt') as sf:
                    for k in sf.keys():
                        written[k] = sf.get_tensor(k)
        ops = worker['TorchOps'](torch, 'cpu', 'bfloat16')
        for name in targets:
            tensors = ops.build(linears[name].weight.detach().to('cpu'), {'bits': 4, 'group_size': 128, 'symmetric': True})
            for key, suffix in (('packed', 'weight_packed'), ('scale', 'weight_scale'), ('shape', 'weight_shape')):
                want = written[name + '.' + suffix]
                self.assertEqual((want.dtype, tuple(want.shape)), (tensors[key].dtype, tuple(tensors[key].shape)), name + '.' + suffix)
                self.assertTrue(torch.equal(want, tensors[key]), name + '.' + suffix)


if __name__ == '__main__':
    unittest.main()
