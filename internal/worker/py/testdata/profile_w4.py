"""W4 reference/auto benchmark on the same fixed inputs; emits JSON evidence.

Micro: python profile_w4.py ../hachidori_worker.py --out 3072 --inner 3072
Model: python profile_w4.py ../hachidori_worker.py --config fixed.json
Config: {"provider": {"model_dir": "...", "device": "cuda", "manifest": {...},
"variant_dir": "...", "variant": {...}}, "workloads": [
{"name": "one-question", "items": [...]}, {"name": "multi-question", "items": [...]}]}
Use the accepted immutable variant and identical fixed decide items for both paths.
Rows in microbenchmarks are activation rows, NOT semantic questions. Only model
mode measures probabilities/decisions. Run with the pinned runtime interpreter.
No output from another OS/GPU constitutes physical Windows RTX 3060 evidence.
Tolerance is fixed at BF16 rtol=0.016, atol=1e-5; model probability absolute
error <=0.001 and exactly equal choices. Failures are recorded, never relaxed.
"""
import argparse
import gc
import hashlib
import json
import runpy
import statistics
import sys
import time
from unittest.mock import patch

import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('worker')
ap.add_argument('--config')
ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
ap.add_argument('--out', type=int, default=128)
ap.add_argument('--inner', type=int, default=256)
ap.add_argument('--repeats', type=int, default=10)
args = ap.parse_args()
if args.repeats < 1:
    ap.error('--repeats must be positive')
w = runpy.run_path(args.worker)
bind = w['bind_packed']
select = w['select_w4_backend']


def forced_reference(*unused):
    return ({'selected': 'packed-reference', 'available': False, 'executed': None,
             'optimized_calls': 0, 'reference_calls': 0, 'reason': 'benchmark reference override'}, None)


def measure(fn, device):
    cuda = device == 'cuda'
    sync = torch.cuda.synchronize if cuda else lambda: None
    fn()  # fixed warmup outside measurements
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() if cuda else None
    samples = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            sync()
            result = None  # do not retain a previous dense reference temporary
            start = time.perf_counter()
            result = fn()
            sync()
            samples.append((time.perf_counter() - start) * 1000)
    return result, {'samples_ms': samples, 'median_ms': statistics.median(samples),
                    'base_allocated_bytes': base,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated() if cuda else None,
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved() if cuda else None}


report = {'torch': str(torch.__version__), 'platform': sys.platform,
          'physical_windows_rtx3060': 'NOT_CHECKED', 'bf16_tolerance': {'rtol': 0.016, 'atol': 1e-5}}
if args.config:
    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)
    report['fixed_config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    names = {c['name'] for c in config['workloads']}
    if not {'one-question', 'multi-question'} <= names:
        ap.error('config must include one-question and multi-question workloads')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    pristine = {name: getattr(qwen, name) for name in ('causal_conv1d_fn', 'torch_chunk_gated_delta_rule')}
    runs = {}
    for path in ['reference', 'auto']:
        # Provider initialization wraps upstream dispatch; restore the originals
        # between loads so the second run sees the same pinned kernel contract.
        torch.manual_seed(222)
        with patch.multiple(qwen, **pristine), patch.dict(bind.__globals__, select_w4_backend=forced_reference if path == 'reference' else select):
            p = w['ClefProvider'](**config['provider'])
            p.initialize()
            p.warmup()
            cases = {}
            for case in config['workloads']:
                outputs, timing = measure(lambda: p.decide(case['items']), p.requested)
                cases[case['name']] = {'outputs': outputs, 'timing': timing, 'evidence': p.info()}
            runs[path] = cases
            del p
        gc.collect()
        if config['provider']['device'] == 'cuda':
            torch.cuda.empty_cache()
    comparisons = {}
    for name in names:
        a, b = runs['reference'][name]['outputs'], runs['auto'][name]['outputs']
        errors = []
        choices = True
        if len(a) != len(b):
            raise RuntimeError('item count differs')
        for ra, rb in zip(a, b):
            if len(ra) != len(rb):
                raise RuntimeError('answer count differs')
            for aa, bb in zip(ra, rb):
                if aa['id'] != bb['id']:
                    raise RuntimeError('question IDs differ')
                if aa['probabilities'].keys() != bb['probabilities'].keys():
                    raise RuntimeError('probability options differ')
                choices = choices and aa['choice'] == bb['choice']
                errors.extend(abs(prob - bb['probabilities'][option]) for option, prob in aa['probabilities'].items())
        comparisons[name] = {'choices_equal': choices, 'max_probability_error': max(errors, default=0),
                             'equivalent': choices and max(errors, default=0) <= 0.001}
    report.update(runs=runs, comparisons=comparisons)
else:
    torch.manual_seed(222)
    ops = w['TorchOps'](torch, args.device, 'bfloat16')
    m = torch.nn.Linear(args.inner, args.out, bias=False).to(torch.bfloat16)
    tensors = ops.build(m.weight.detach(), {'bits': 4, 'group_size': 128, 'symmetric': True})
    with patch.dict(bind.__globals__, select_w4_backend=forced_reference):
        ops.install_packed(m, ops.stage_packed(tensors), {'bits': 4, 'group_size': 128})
    cases = {}
    for rows in [1, 4, 128]:
        torch.manual_seed(222 + rows)
        x = torch.randn(rows, args.inner, device=args.device, dtype=torch.bfloat16)
        dense, dequant = measure(lambda: w['packed_reference_weight'](m, x.dtype), args.device)
        expected, linear = measure(lambda: torch.nn.functional.linear(x, dense), args.device)
        del dense
        _, total = measure(lambda: m(x), args.device)
        cases[str(rows)] = {'dequant_materialize': dequant, 'linear': linear, 'reference_total': total,
                            'reference_output_sample': expected.flatten()[:8].cpu().float().tolist()}
    bind(torch, m._unpack, m, args.out, args.inner, 4, 128)
    for rows in [1, 4, 128]:
        torch.manual_seed(222 + rows)
        x = torch.randn(rows, args.inner, device=args.device, dtype=torch.bfloat16)
        # Independent canonical component reference; never attached to serving.
        q = ops._unpack(tensors['packed'].to(args.device), 4, torch.Size((args.out, args.inner)))
        dense = (q.to(x.dtype).view(args.out, args.inner // 128, 128) * tensors['scale'].to(args.device).unsqueeze(-1)).view(args.out, args.inner)
        expected = torch.nn.functional.linear(x, dense)
        del dense, q
        actual, timing = measure(lambda: m(x), args.device)
        try:
            torch.testing.assert_close(actual, expected, rtol=0.016, atol=1e-5)
            equivalent = True
        except AssertionError:
            equivalent = False
        cases[str(rows)].update(auto=timing, equivalent=equivalent,
                               max_abs_error=(actual.float() - expected.float()).abs().max().item(),
                               evidence=dict(m._hachidori_w4[0]))
    report.update(device=args.device, shape=[args.out, args.inner], micro=cases,
                  semantic_workloads='NOT_CHECKED: use --config')
device = config['provider']['device'] if args.config else args.device
report['python'] = sys.version
report['gpu'] = torch.cuda.get_device_name() if device == 'cuda' else None
w['emit'](report)
if args.config:
    sys.exit(0 if all(c['equivalent'] for c in comparisons.values()) else 1)
sys.exit(0 if all(c['equivalent'] for c in cases.values()) else 1)
