"""CPU-only admission/readiness contracts; no CUDA certification claim."""
import contextlib
import io
import json
import runpy
import sys
import types
import unittest
from unittest.mock import patch
w = runpy.run_path(sys.argv.pop(1))
Clef = w['ClefProvider']
CapacityError = w['CapacityError']


def provider(profile=None):
    p = Clef.__new__(Clef)
    p.capacity_profile = profile
    p.requested = 'cpu'
    p.want_dtype = 'bfloat16'
    p.tokenizer = object()
    p.sync = lambda: None
    p.calls = []
    def encode(tokenizer, record, max_length):
        assert max_length == sys.maxsize, 'encoder must not truncate'
        # State + full question schema are counted before any model call.
        size = len(record['state']) + len(record['questions'])
        return types.SimpleNamespace(input_ids=tuple(range(size)))
    p.jsm = types.SimpleNamespace(encode_record=encode, render=lambda s: s,
                                 _tokens=lambda t, s: list(s))
    p.torch = object()
    p._predict_batch = lambda records, torch: p.calls.append(len(records)) or [{} for _ in records]
    return p


def profile(**overrides):
    x = dict(dtype='bfloat16', max_state_tokens=8, max_input_tokens=10, max_batch_items=2,
             max_batch_padded_tokens=20, required_gpu_headroom_bytes=100)
    x.update(overrides)
    return x


class Capacity(unittest.TestCase):
    def test_boundaries_and_no_truncation(self):
        for count in (7, 8):
            p = provider(profile())
            self.assertEqual(len(p.predict(['s' * count], {'q': {}})), 1)
            self.assertEqual(p.calls, [1])
        p = provider(profile())
        with self.assertRaises(CapacityError) as e:
            p.predict(['s' * 9], {'q': {}})
        self.assertEqual(e.exception.details, dict(metric='state_tokens', limit=8, observed=9))
        self.assertEqual(p.calls, [])
        p = provider(profile(max_state_tokens=10))
        with self.assertRaises(CapacityError) as e:
            p.predict(['s' * 10], {'q': {}})
        self.assertEqual(e.exception.details['metric'], 'input_tokens')
        self.assertEqual(p.calls, [])

    def test_requested_input_limit_only_tightens_safe_bound(self):
        p = provider(profile(requested_max_input_tokens=5))
        status = p.capacity_status()
        self.assertEqual(status['requested_max_input_tokens'], 5)
        self.assertEqual(status['effective_max_input_tokens'], 5)
        with self.assertRaises(CapacityError) as e:
            p.predict(['s' * 5], {'q': {}})
        self.assertEqual(e.exception.details, dict(metric='input_tokens', limit=5, observed=6))
        self.assertEqual(p.calls, [])

        p = provider(profile(max_state_tokens=12, max_input_tokens=12,
                             max_batch_padded_tokens=24, requested_max_input_tokens=15))
        status = p.capacity_status()
        self.assertEqual(status['requested_max_input_tokens'], 15)
        self.assertEqual(status['effective_max_input_tokens'], 12)
        with self.assertRaises(CapacityError) as e:
            p.predict(['s' * 12], {'q': {}})
        self.assertEqual(e.exception.details, dict(metric='input_tokens', limit=12, observed=13))
        self.assertEqual(p.calls, [])

    def test_batch_forward_shape_and_determinism(self):
        for _ in range(2):
            p = provider(profile(max_batch_padded_tokens=17))
            with self.assertRaises(CapacityError) as e:
                p.predict(['s' * 8, 's' * 8], {'q': {}})
            self.assertEqual(e.exception.details, dict(metric='batch_padded_tokens', limit=17, observed=18))
            self.assertEqual(p.calls, [])
        p = provider(profile(max_batch_padded_tokens=18))
        p.predict(['s' * 8, 's' * 8], {'q': {}})
        self.assertEqual(p.calls, [2])
        p = provider(profile())
        with self.assertRaises(CapacityError):
            p.predict(['s'] * 3, {'q': {}})
        self.assertEqual(p.calls, [])

    def test_every_question_group_admitted_before_execution(self):
        p = provider(profile())
        items = [{'state': 'ok', 'questions': []},
                 {'state': 's' * 9, 'questions': [{'id': 'q', 'instructions': 'i', 'choices': ['y', 'n']}]}]
        with self.assertRaises(CapacityError):
            p.decide(items)
        self.assertEqual(p.calls, [])

    def test_cpu_context_without_gpu_profile(self):
        p = provider()
        p.check_capacity_readiness()
        with self.assertRaises(CapacityError) as e:
            p.predict(['s' * (w['CLEF_MAX_LENGTH'] + 1)], {})
        self.assertEqual(e.exception.details['limit'], w['CLEF_MAX_LENGTH'])
        self.assertEqual(p.calls, [])

    def test_gpu_readiness_includes_reusable_allocator_workspace(self):
        p = provider(profile())
        p.requested = 'cuda'
        p.placed_device = lambda: 'cuda:0'
        p.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            mem_get_info=lambda d: (60, 1000), memory_reserved=lambda d: 90,
            memory_allocated=lambda d: 50))
        p.check_capacity_readiness()
        self.assertEqual(p.capacity_status()['headroom']['usable_bytes'], 100)
        p.capacity_profile['required_gpu_headroom_bytes'] = 101
        with self.assertRaises(CapacityError) as e:
            p.check_capacity_readiness()
        self.assertEqual(e.exception.details, dict(metric='gpu_headroom_bytes', limit=101, observed=100))

    def test_missing_profile_is_typed_readiness_failure(self):
        p = provider()
        p.requested = 'cuda'
        messages = []
        with patch.dict(w['ClefProvider'].check_capacity_readiness.__globals__, {'emit': messages.append}):
            with self.assertRaises(SystemExit):
                p.check_capacity_readiness()
        self.assertEqual(messages[-1]['class'], 'capacity')

    def test_protocol_denial_keeps_worker_alive(self):
        p = provider(profile())
        request = {'id': 1, 'op': 'decide', 'items': [{'state': 's' * 9, 'questions': []}]}
        output = []
        namespace = w['serve'].__globals__
        with patch.dict(namespace, {'emit': output.append}), patch('sys.stdin', io.StringIO(json.dumps(request)+'\n'+json.dumps({'id':2,'op':'shutdown'})+'\n')):
            w['serve'](p)
        self.assertEqual(output[0]['error']['class'], 'capacity')
        self.assertEqual(output[0]['error']['capacity']['observed'], 9)
        self.assertTrue(output[1]['ok'])
        self.assertEqual(p.calls, [])

if __name__ == '__main__': unittest.main()
