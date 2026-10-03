"""Portable worker contract tests; no CUDA or physical performance evidence."""
import functools
import json
import os
import runpy
import sys
import types
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

worker = runpy.run_path(sys.argv.pop(1))
configure = worker['configure_clef_kernels']
ClefProvider = worker['ClefProvider']


def dispatch(optimized=False, fail=False):
    # Closure contract inspected in the exact Transformers 5.17.0 wheel.
    def torch_function(value, activation=None, **kwargs):
        if fail:
            raise RuntimeError('kernel failure')
        return value

    def fast(value, **kwargs):
        raise AssertionError('the import-time selection must be replaced')

    implementation = fast if optimized else torch_function
    is_new_implementation = implementation is not torch_function

    @functools.wraps(torch_function)
    def wrapped(*args, **kwargs):
        if is_new_implementation and kwargs.pop('exporting', False):
            return torch_function(*args, **kwargs)
        return implementation(*args, **kwargs)

    return wrapped


def qwen(optimized=False, fail=False):
    return types.SimpleNamespace(causal_conv1d_fn=dispatch(optimized, fail),
                                 torch_chunk_gated_delta_rule=dispatch(optimized, fail))


def torch(capability=(8, 6)):
    return types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: capability))


class Tensor:
    def __init__(self, data): self.data = data
    def contiguous(self): return self
    def transpose(self, first, second):
        assert (first, second) == (1, 2)
        return Tensor([[list(row) for row in zip(*batch)] for batch in self.data])


def kernel_modules(conv=None, delta=None):
    def default_conv(**kwargs):
        assert kwargs['backend'] == 'triton'
        assert kwargs['output_final_state'] is False
        return kwargs['x'], None

    return {'fla': types.ModuleType('fla'),
            'fla.modules': types.ModuleType('fla.modules'),
            'fla.modules.conv': types.SimpleNamespace(causal_conv1d=conv or default_conv),
            'fla.ops': types.ModuleType('fla.ops'),
            'fla.ops.gated_delta_rule': types.SimpleNamespace(chunk_gated_delta_rule=delta or (lambda value, **kw: value))}


class KernelTests(unittest.TestCase):
    def test_load_configures_kernels_before_backbone(self):
        # Exercise the actual Clef load hook, including refusal before model
        # loading when a supported runtime lacks either upstream kernel.
        for available in [False, True]:
            with self.subTest(available=available), tempfile.TemporaryDirectory() as path:
                with open(os.path.join(path, 'joint_head_config.json'), 'w') as f:
                    json.dump({}, f)
                module = qwen()
                head = types.SimpleNamespace(load_state_dict=lambda *a, **kw: None)
                head.to = lambda **kw: head
                model = types.SimpleNamespace(
                    parameters=lambda: [types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))],
                    buffers=lambda: [])
                model.eval = lambda: model
                backbone = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True))
                loads = []
                def load(*args, **kwargs):
                    loads.append((args, kwargs))
                    # The bound delta kernel must execute before we claim it active.
                    self.assertEqual(module.torch_chunk_gated_delta_rule('input'), 'input')
                    return backbone
                provider = ClefProvider(path, 'cuda', {'files': {}})
                provider.torch = torch()
                provider.torch.bfloat16 = 'bfloat16'
                provider.transformers = types.SimpleNamespace(
                    Qwen3_5ForConditionalGeneration=types.SimpleNamespace(from_pretrained=load),
                    AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: object()))
                provider.safetensors_torch = types.SimpleNamespace(load_file=lambda path: {})
                modules = kernel_modules()
                modules.update({
                    'joint_schema_model': types.SimpleNamespace(JointSchemaHead=lambda: head,
                                                               ClefModel=lambda *args: model),
                    'transformers': types.ModuleType('transformers'),
                    'transformers.models': types.ModuleType('transformers.models'),
                    'transformers.models.qwen3_5': types.SimpleNamespace(modeling_qwen3_5=module),
                })
                if not available:
                    modules['fla.modules.conv'] = None
                with patch.object(sys, 'platform', 'win32'), patch.dict(sys.modules, modules):
                    if available:
                        provider.load()
                        self.assertFalse(backbone.config.use_cache)
                        self.assertEqual(len(loads), 1)
                        self.assertEqual(loads[0][1]['device_map'], {'': 'cuda'})
                        self.assertEqual(provider.kernel_paths['chunk_gated_delta_rule']['execution'], 'optimized_active')
                        self.assertEqual(provider.kernel_paths['causal_conv1d_fn']['execution'], 'not_observed')
                    else:
                        with self.assertRaises(ImportError):
                            provider.load()
                        self.assertEqual(loads, [])
                # load follows the pre-existing model-local import contract.
                sys.path.remove(path)

    def test_supported_selection_is_not_execution(self):
        module = qwen()
        with patch.dict(sys.modules, kernel_modules()):
            paths = configure(module, torch(), 'cuda', 'bfloat16', platform='win32')
        for entry in paths.values():
            self.assertEqual(entry['execution'], 'not_observed')
            self.assertEqual(entry['availability'], 'available')
        value = Tensor([[[1, 2, 3], [10, 20, 30]]])
        self.assertEqual(module.causal_conv1d_fn(value, 'weights', activation='silu').data, value.data)
        self.assertIs(module.torch_chunk_gated_delta_rule(value), value)
        for entry in paths.values():
            self.assertEqual(entry['execution'], 'optimized_active')

    def test_convolution_layout_and_arguments(self):
        weights, bias = object(), object()
        def conv(**kw):
            self.assertEqual(kw['x'].data, [[[1, 10], [2, 20], [3, 30]]])
            self.assertIs(kw['weight'], weights)
            self.assertIs(kw['bias'], bias)
            self.assertEqual(kw['activation'], 'silu')
            self.assertEqual(kw['backend'], 'triton')
            self.assertFalse(kw['output_final_state'])
            return Tensor([[[3, 50], [8, 140], [13, 230]]]), None
        module = qwen()
        with patch.dict(sys.modules, kernel_modules(conv=conv)):
            configure(module, torch(), 'cuda', 'bfloat16', platform='win32')
        result = module.causal_conv1d_fn(Tensor([[[1, 2, 3], [10, 20, 30]]]),
                                        weights, bias, activation='silu')
        self.assertEqual(result.data, [[[3, 8, 13], [50, 140, 230]]])

    def test_delta_rule_arguments_are_preserved(self):
        calls = []
        def delta(*args, **kwargs):
            calls.append((args, kwargs))
            return 'output', None
        module = qwen()
        with patch.dict(sys.modules, kernel_modules(delta=delta)):
            configure(module, torch(), 'cuda', 'bfloat16', platform='win32')
        tensors = tuple(object() for _ in range(5))
        kwargs = {'g': tensors[3], 'beta': tensors[4], 'initial_state': None,
                  'output_final_state': False, 'use_qk_l2norm_in_kernel': True,
                  'cu_seqlens': None}
        self.assertEqual(module.torch_chunk_gated_delta_rule(*tensors[:3], **kwargs), ('output', None))
        self.assertEqual(calls, [(tensors[:3], kwargs)])

    def test_cpu_unsupported_gpu_and_dtype_use_explicit_references(self):
        for device, capability, dtype, reason in [
            ('cpu', (8, 6), 'bfloat16', 'CPU reference'),
            ('cpu', (8, 6), 'float32', 'CPU reference'),
            ('cuda', (7, 5), 'bfloat16', 'capability >= 8.0'),
            ('cuda', (8, 6), 'float32', 'high-precision reference'),
        ]:
            with self.subTest(device=device, capability=capability, dtype=dtype):
                # Even an optimized callable auto-selected by Transformers must
                # be replaced on unsupported/CPU launches.
                module = qwen(optimized=True)
                paths = configure(module, torch(capability), device, dtype, platform='win32')
                for entry in paths.values():
                    self.assertIn(reason, entry['reason'])
                    self.assertEqual(entry['availability'], 'unavailable')
                value = object()
                self.assertIs(module.causal_conv1d_fn(value, activation='silu'), value)
                self.assertIs(module.torch_chunk_gated_delta_rule(value), value)
                for entry in paths.values():
                    self.assertEqual(entry['execution'], 'reference_active')

    def test_non_windows_cuda_uses_reference_path(self):
        module = qwen()
        paths = configure(module, torch(), 'cuda', 'bfloat16', platform='linux')
        for entry in paths.values():
            self.assertEqual(entry['selected'], 'reference')
            self.assertEqual(entry['availability'], 'unavailable')
            self.assertIn('only on native Windows', entry['reason'])
        value = object()
        self.assertIs(module.causal_conv1d_fn(value), value)
        self.assertIs(module.torch_chunk_gated_delta_rule(value), value)
        for entry in paths.values():
            self.assertEqual(entry['execution'], 'reference_active')

    def test_supported_cuda_without_kernels_fails(self):
        for missing in ['fla.modules.conv', 'fla.ops.gated_delta_rule']:
            modules = kernel_modules()
            modules[missing] = None
            with patch.dict(sys.modules, modules):
                with self.assertRaises(ImportError):
                    configure(qwen(), torch(), 'cuda', 'bfloat16', platform='win32')

    def test_failed_kernel_has_no_reference_retry_or_active_claim(self):
        def failed(*args, **kwargs):
            raise RuntimeError('kernel failure')
        module = qwen()
        with patch.dict(sys.modules, kernel_modules(conv=failed, delta=failed)):
            paths = configure(module, torch(), 'cuda', 'bfloat16', platform='win32')
        with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
            module.causal_conv1d_fn(Tensor([[[1]]]), 'weights')
        with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
            module.torch_chunk_gated_delta_rule(object())
        for entry in paths.values():
            self.assertEqual(entry['execution'], 'not_observed')
        module = qwen(fail=True)
        paths = configure(module, torch(), 'cpu', 'float32')
        with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
            module.causal_conv1d_fn(1)
        self.assertEqual(paths['causal_conv1d_fn']['execution'], 'not_observed')

    def test_unknown_transformers_dispatch_is_refused(self):
        module = qwen()
        module.causal_conv1d_fn = lambda value: value
        with self.assertRaisesRegex(RuntimeError, 'unrecognized Transformers'):
            configure(module, torch(), 'cuda', 'bfloat16', platform='win32')

    def test_clef_predictions_and_variant_evidence_preserved(self):
        for device in ['cpu', 'cuda']:
            module = qwen()
            provider = ClefProvider('model', device, {'files': {}})
            with patch.dict(sys.modules, kernel_modules()):
                provider.kernel_paths = configure(
                    module, torch(), device, 'bfloat16',
                    platform='win32' if device == 'cuda' else 'linux')
            provider.transformers = types.SimpleNamespace(__version__='5.17.0')
            provider.backbone = types.SimpleNamespace(named_modules=lambda: [])
            provider.quantized = 7
            provider.want_dtype = 'bfloat16'
            provider.variant = {'weights': {'scheme': 'W4A16', 'format': 'compressed-tensors/pack-quantized'}}
            provider.torch = types.SimpleNamespace(inference_mode=nullcontext)
            question = types.SimpleNamespace(question_id='decision', option_ids=['no', 'yes'])
            encoded = types.SimpleNamespace(input_ids=[1, 2], questions=[question])
            seen = []
            provider.jsm = types.SimpleNamespace(
                encode_record=lambda tokenizer, record, max_length: (seen.append(record) or encoded),
                collate_records=lambda records, pad, device: Tensor([[[1, 2]]]))
            provider.tokenizer = types.SimpleNamespace(pad_token_id=0)
            provider.reference = lambda: types.SimpleNamespace(device=device)

            class Logits:
                def float(self): return self
                def softmax(self, dim): return self
                def tolist(self): return [0.25, 0.75]

            def model(batch):
                module.causal_conv1d_fn(batch, 'weights') if device == 'cuda' else module.causal_conv1d_fn(batch)
                module.torch_chunk_gated_delta_rule(batch)
                return [[Logits()]]

            provider.model = model
            questions = [{'id': 'decision', 'choices': ['yes', 'no']}]
            result = provider.predict(['unchanged state'], questions)
            self.assertEqual(seen, [{'state': 'unchanged state', 'questions': questions}])
            self.assertEqual(result, [{'answers': {'decision': {
                'choice': 'yes', 'answer_confidence': 0.75,
                'probabilities': {'no': 0.25, 'yes': 0.75}}}}])
            info = provider.extra_info()
            self.assertEqual(info['execution'], 'variant')
            self.assertEqual(info['weights_quantized_modules'], 7)
            self.assertEqual(info['quantization_scheme'], 'W4A16')
            self.assertEqual(info['kernel_paths']['chunk_gated_delta_rule']['execution'],
                             'optimized_active' if device == 'cuda' else 'reference_active')


unittest.main()
