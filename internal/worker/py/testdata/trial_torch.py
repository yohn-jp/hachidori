"""Tuning trial replacement semantics on real torch modules (CPU).

Needs torch and compressed-tensors (the worker runtime's own dependencies); when
either is missing the module reports itself skipped, so portable CI without them
stays green. This proves what replacing a Linear module's representation does to
a real torch module; it is not evidence about CUDA behavior, transfer speed or
the Windows runtime.
"""
import runpy
import sys
import unittest

try:
    import torch
    import compressed_tensors  # noqa: F401
except Exception as e:  # noqa: BLE001
    print('SKIP: %s' % e)
    sys.exit(0)

worker = runpy.run_path(sys.argv.pop(1))
TorchOps, TrialExecutor, TrialError = worker['TorchOps'], worker['TrialExecutor'], worker['TrialError']
TRANSFORM = worker['TRIAL_TRANSFORM']


def model(shared=False):
    torch.manual_seed(3)
    m = torch.nn.ModuleDict({n: torch.nn.Linear(256, 128, bias=False) for n in 'abcd'}).to(torch.bfloat16).eval()
    if shared:
        m['d'].weight = m['c'].weight
    return m


def opened(m, **kw):
    ops = TorchOps(torch, 'cpu', 'bfloat16')
    mods = {'a': m['a'], 'b': m['b'], 'c': m['c'], 'd': m['d']}
    ex = TrialExecutor(ops, mods, 'cpu', 'bfloat16', shared=ops.shared_weights(m), **kw)
    ex.open({'g1': ['a', 'b'], 'g2': ['c', 'd']})
    return ex, ops


def transformation(representation='packed-int4'):
    t = {'policy': 'w4a16', 'representation': representation, 'implementation': TRANSFORM, 'backend': TorchOps(torch, 'cpu', 'bfloat16').backend(),
         'scheme': 'W4A16', 'algorithm': 'rtn', 'bits': 4, 'group_size': 128, 'symmetric': True,
         'format': 'compressed-tensors/pack-quantized', 'compute_dtype': 'bfloat16'}
    return t


def identity(cid, modules):
    return {'id': cid, 'transformation': transformation(),
            'members': [{'module': n, 'shape': [128, 256], 'dtype': 'bfloat16'} for n in modules]}


def packed(group, cid, modules):
    return {'group': group, 'policy': 'w4a16', 'component': cid, 'modules': modules, 'transformation': transformation()}


def dense(group, modules):
    return {'group': group, 'policy': 'source-precision', 'modules': modules,
            'transformation': {'policy': 'source-precision', 'representation': 'dense', 'implementation': TRANSFORM,
                               'backend': TorchOps(torch, 'cpu', 'bfloat16').backend()}}


class TorchTrialTests(unittest.TestCase):
    def test_packed_replacement_runs_through_the_shared_packed_forward(self):
        m = model()
        original = {n: m[n].weight.detach().clone() for n in 'abcd'}
        x = torch.randn(3, 256, dtype=torch.bfloat16)
        want = {n: torch.nn.functional.linear(x, original[n]) for n in 'ab'}
        ex, ops = opened(m)
        ex.transform(identity('c1', ['a', 'b']))
        res = ex.apply([packed('g1', 'c1', ['a', 'b'])])
        self.assertGreater(res['bytes_to_gpu'], 0)
        self.assertEqual(res['modules'], 2)
        for n in 'ab':
            mod = m[n]
            self.assertIsNone(mod._parameters.get('weight'))
            self.assertEqual(mod.weight_packed.dtype, torch.int32)
            self.assertEqual(tuple(mod.weight_packed.shape), (128, 32))
            self.assertEqual(tuple(mod.weight_scale.shape), (128, 2))
            self.assertIn('forward', mod.__dict__)
            got = mod(x)
            # int4 round-to-nearest: close to, but not bit-identical with, the source
            self.assertFalse(torch.equal(got, want[n]))
            self.assertLess((got.float() - want[n].float()).abs().max().item(), 0.35 * want[n].float().abs().max().item())
        # untouched group keeps its exact source parameter object
        self.assertIs(ex.modules['c'], m['c'])
        self.assertTrue(torch.equal(m['c'].weight, original['c']))
        self.assertEqual(ex.state()['groups'], {'g1': 'packed-int4', 'g2': 'dense'})

    def test_reverse_restores_the_source_bit_exactly_and_unbinds_the_wrapper(self):
        m = model()
        original = {n: m[n].weight.detach().clone() for n in 'abcd'}
        ex, ops = opened(m)
        ex.transform(identity('c1', ['a', 'b']))
        ex.apply([packed('g1', 'c1', ['a', 'b'])])
        ex.apply([dense('g1', ['a', 'b'])])
        for n in 'ab':
            mod = m[n]
            self.assertTrue(torch.equal(mod.weight, original[n]))
            self.assertEqual(sorted(mod._parameters), ['bias', 'weight'])
            self.assertEqual(sorted(mod._buffers), [])
            for stale in ('forward', '_hachidori_packed', '_unpack'):
                self.assertNotIn(stale, mod.__dict__)
            self.assertFalse(hasattr(mod, 'weight_packed'))
        self.assertEqual(ex.state()['groups'], {'g1': 'dense', 'g2': 'dense'})

    def test_failed_application_leaves_the_module_untouched(self):
        m = model()
        original = {n: m[n].weight.detach().clone() for n in 'abcd'}
        ex, ops = opened(m)
        ex.transform(identity('c1', ['a', 'b']))
        calls = []
        real = ops.install_packed

        def failing(module, staged, params):
            calls.append(module)
            if len(calls) == 2:
                raise RuntimeError('second install fails')
            real(module, staged, params)
        ops.install_packed = failing
        with self.assertRaises(TrialError) as cm:
            ex.apply([packed('g1', 'c1', ['a', 'b'])])
        self.assertEqual(cm.exception.cls, 'trial_failed')
        for n in 'ab':
            self.assertTrue(torch.equal(m[n].weight, original[n]))
            self.assertNotIn('forward', m[n].__dict__)
            self.assertFalse(hasattr(m[n], 'weight_packed'))
        self.assertEqual(ex.state()['groups'], {'g1': 'dense', 'g2': 'dense'})

    def test_shared_weights_are_not_replaced(self):
        m = model(shared=True)
        ex, ops = opened(m)
        ex.transform(identity('c1', ['c']))
        with self.assertRaises(TrialError) as cm:
            ex.validate([packed('g2', 'c1', ['c'])])
        self.assertIn('shares its weight', str(cm.exception))

    def test_component_is_the_exact_compressed_tensors_quantization(self):
        # Independent check of the transformation: unpacking with the library and
        # applying the stored scale reproduces the source within about half a step
        # (plus bfloat16 rounding of the scale and of the product).
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        m = model()
        w = m['a'].weight.detach().clone()
        ops = TorchOps(torch, 'cpu', 'bfloat16')
        t = ops.build(w, {'bits': 4, 'group_size': 128, 'symmetric': True})
        q = unpack_from_int32(t['packed'], 4, torch.Size((128, 256)))
        dq = q.to(torch.bfloat16).view(128, 2, 128) * t['scale'].unsqueeze(-1)
        err = (dq.view(128, 256).float() - w.float()).abs().view(128, 2, 128)
        bound = t['scale'].float().unsqueeze(-1) * 0.6 + w.float().abs().view(128, 2, 128) * 2 ** -6 + 1e-6
        self.assertTrue(bool((err <= bound).all()))
        self.assertEqual(t['shape'].tolist(), [128, 256])
        self.assertEqual(ops.digest([('a', t)]), ops.digest([('a', ops.build(w, {'bits': 4, 'group_size': 128, 'symmetric': True}))]))


if __name__ == '__main__':
    unittest.main()
