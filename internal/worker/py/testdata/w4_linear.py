"""Deterministic W4 execution tests. Mock dispatch is not physical evidence."""
import runpy
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
    import compressed_tensors
except ImportError as e:
    print('SKIP: %s' % e)
    sys.exit(0)

w = runpy.run_path(sys.argv.pop(1))
bind = w['bind_packed']
select = w['select_w4_backend']
reference = w['packed_reference_weight']
ops = w['TorchOps'](torch, 'cpu', 'bfloat16')


def module(device='cpu', bias=True):
    torch.manual_seed(222)
    m = torch.nn.Linear(256, 128, bias=bias).to(torch.bfloat16)
    tensors = ops.build(m.weight.detach(), {'bits': 4, 'group_size': 128, 'symmetric': True})
    ops.install_packed(m, tensors, {'bits': 4, 'group_size': 128})
    return m.to(device)


class W4Tests(unittest.TestCase):
    def test_reference_outputs_and_evidence(self):
        m = module()
        state = m._hachidori_w4[0]
        self.assertIsNone(state['executed'])
        self.assertFalse(state['available'])
        self.assertTrue(state['reason'])
        for shape in [(256,), (1, 256), (4, 7, 256)]:
            x = torch.randn(shape, dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(x, reference(m, x.dtype), m.bias)
            self.assertTrue(torch.equal(m(x), expected))
        self.assertEqual(state['executed'], 'packed-reference')
        self.assertEqual(state['optimized_calls'], 0)
        self.assertEqual(state['reference_calls'], 3)
        p = w['ClefProvider']('unused', 'cpu', {'files': {}})
        p.backbone = torch.nn.ModuleDict({'linear': m})
        p.transformers = types.SimpleNamespace(__version__='test')
        p.quantized, p.kernel_paths = 1, {}
        self.assertEqual(p.extra_info()['w4_linear_paths']['linear'], state)
        self.assertEqual(p.stats()['w4_linear_paths']['linear']['reference_calls'], 3)

    def test_capability_denials(self):
        m = module()
        self.assertIn('Windows', select(torch, m, platform='linux')[0]['reason'])
        self.assertIn('pinned', select(torch, m, platform='win32')[0]['reason'])
        fake = types.SimpleNamespace(__version__='2.11.0+cu128')
        self.assertIn('CUDA', select(fake, m, platform='win32')[0]['reason'])

    def test_mock_native_layout_and_execution_not_reconstruction(self):
        # Mock only the device/dispatch APIs; all q/scales/outputs are real torch.
        # This checks integration and nibble semantics, NOT the CUDA kernel.
        m = module(bias=False)
        canonical = m.weight_packed
        q = m._unpack(canonical, 4, torch.Size((128, 256)))
        expected_weight = reference(m, torch.bfloat16)
        m.weight_packed = None
        m.__dict__['weight_packed'] = types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))
        m._unpack = lambda *args: q
        pairs = []
        def convert(p, tiles):
            self.assertEqual(tiles, 8)
            pairs.append(p.clone())
            return canonical
        def mm(x, layout, group, scales):
            p = pairs[-1]
            decoded = torch.stack((p >> 4, p & 15), -1).reshape(128, 256).to(torch.int16) - 8
            dense = (decoded.to(torch.bfloat16).view(128, 2, 128) * scales[:, :, 0].t().unsqueeze(-1)).reshape(128, 256)
            self.assertTrue(torch.equal(dense, expected_weight))
            self.assertEqual(group, 128)
            self.assertEqual(scales[:, :, 1].count_nonzero(), 0)
            return torch.nn.functional.linear(x, dense)
        fake = types.SimpleNamespace(__version__='2.11.0+cu128', cuda=types.SimpleNamespace(get_device_capability=lambda d: (8, 6), synchronize=lambda d: None), _convert_weight_to_int4pack=convert, _weight_int4pack_mm=mm)
        for attr in ('bfloat16', 'int16', 'uint8', 'Size', 'stack', 'zeros_like', 'sin', 'arange', 'float32', 'nn', 'testing'):
            setattr(fake, attr, getattr(torch, attr))
        state, backend = select(fake, m, platform='win32')
        self.assertTrue(state['available'])
        self.assertIsNone(state['executed'])  # preflight is not serving evidence
        fake._weight_int4pack_mm = lambda *a: torch.zeros((1, 128), dtype=torch.bfloat16)
        refused, absent = select(fake, m, platform='win32')
        self.assertIsNone(absent)
        self.assertFalse(refused['available'])
        self.assertEqual(refused['selected'], 'packed-reference')
        self.assertIn('AssertionError', refused['reason'])
        self.assertIsNone(refused['executed'])
        def missing(*a):
            raise RuntimeError('kernel not compiled in this wheel')
        fake._weight_int4pack_mm = missing
        refused, absent = select(fake, m, platform='win32')
        self.assertIsNone(absent)
        self.assertIn('kernel not compiled', refused['reason'])
        fake._weight_int4pack_mm = mm
        m.__dict__.pop('weight_packed')
        m._parameters.pop('weight_packed')
        m.register_parameter('weight_packed', torch.nn.Parameter(canonical, requires_grad=False))
        with patch.dict(bind.__globals__, select_w4_backend=lambda *a: (state, backend)):
            bind(torch, lambda *a: self.fail('per-forward unpack'), m, 128, 256, 4, 128)
        x = torch.randn(2, 3, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(m(x), torch.nn.functional.linear(x, expected_weight, m.bias), rtol=0.016, atol=1e-5)
        self.assertEqual(state['optimized_calls'], 1)
        self.assertEqual(state['reference_calls'], 0)
        with self.assertRaisesRegex(RuntimeError, 'BF16'):
            m(x.float())
        self.assertEqual(state['optimized_calls'], 1)
        m._hachidori_w4 = (state, (missing, torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, 'not compiled'):
            m(x)
        self.assertEqual(state['reference_calls'], 0)
        m._hachidori_w4 = (state, (mm, torch.bfloat16))
        snap = ops.snapshot(m)
        ops.install_dense(m, {'weight': expected_weight, 'requires_grad': False})
        self.assertFalse(hasattr(m, '_hachidori_w4_scales'))
        ops.restore(m, snap)
        torch.testing.assert_close(m(x), torch.nn.functional.linear(x, expected_weight, m.bias), rtol=0.016, atol=1e-5)

    def test_physical_native_outputs(self):
        if sys.platform != 'win32' or str(torch.__version__) != '2.11.0+cu128' or not torch.cuda.is_available():
            self.skipTest('NOT_CHECKED: pinned Windows CUDA unavailable')
        m = module('cuda', bias=False)
        expected_weight = reference(m, torch.bfloat16)
        bind(torch, m._unpack, m, 128, 256, 4, 128)
        if not m._hachidori_w4[0]['available']:
            self.skipTest('incompatible: %s' % m._hachidori_w4[0]['reason'])
        for rows in [1, 4, 32, 257]:
            x = torch.randn(rows, 256, device='cuda', dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(x, expected_weight, m.bias)
            torch.testing.assert_close(m(x), expected, rtol=0.016, atol=1e-5)
        self.assertEqual(m._hachidori_w4[0]['optimized_calls'], 4)


if __name__ == '__main__':
    unittest.main()
