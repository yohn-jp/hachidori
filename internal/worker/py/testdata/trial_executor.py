"""Portable contract tests of the tuning trial executor's transactional logic.

The executor's rollback, validation and accounting are independent of torch: the
torch-facing half is an injected adapter, replaced here by a fake whose failures
can be injected at every step. These tests prove orchestration and invariants;
they say nothing about real accelerator behavior (see trial_torch.py and the
physical validation).
"""
import io
import json
import runpy
import sys
import unittest

worker = runpy.run_path(sys.argv.pop(1))
TrialExecutor = worker['TrialExecutor']
TrialError = worker['TrialError']
trial_dispatch = worker['trial_dispatch']
TRANSFORM = worker['TRIAL_TRANSFORM']

DENSE, PACKED = 'dense', 'packed-int4'


class Tensor:
    def __init__(self, shape, dtype='bfloat16', nbytes=None, value=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.nbytes = nbytes if nbytes is not None else 2 * shape[0] * shape[1]
        self.value = value

    def tolist(self):
        return list(self.value)


class Module:
    """A Linear module: its representation is the only thing a trial changes."""

    def __init__(self, shape, cls='Linear', hooked=False, rep=DENSE):
        self.shape = shape
        self.cls = cls
        self.hooked = hooked
        self.state = (rep, None)
        self.writes = 0


class Ops:
    """Fake of TorchOps. Failures are injected by name: the Nth call (1-based) of
    an operation raises."""

    def __init__(self, fail=None):
        self.fail = fail or {}
        self.calls = {}
        self.pinned = False
        self.unsupported = None
        self.gpu = 0

    def _hit(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.fail.get(name) == self.calls[name]:
            raise RuntimeError('injected %s failure' % name)

    def backend(self): return 'fake 1'
    def packed_support(self): return self.unsupported

    def describe(self, m):
        rep = m.state[0]
        return {'class': m.cls, 'hooked': m.hooked, 'bias': False, 'representation': rep,
                'shape': m.shape, 'dtype': 'bfloat16', 'device': 'cuda'}

    def canonical(self, m):
        return {'tensor': Tensor(m.shape), 'requires_grad': False}

    def tensor_bytes(self, t): return t.nbytes

    def build(self, weight, params):
        self._hit('build')
        out, inner = weight.shape
        return {'packed': Tensor((out, inner // 8), 'int32', nbytes=out * inner // 2),
                'scale': Tensor((out, inner // params['group_size']), 'bfloat16'),
                'shape': Tensor((2,), 'int64', nbytes=16, value=[out, inner])}

    def component_bytes(self, tensors): return sum(t.nbytes for t in tensors.values())
    def digest(self, items): return 'digest:' + ','.join(n for n, _ in items)

    def stage_dense(self, canon):
        self._hit('stage')
        return {'kind': DENSE, 'bytes': canon['tensor'].nbytes}

    def stage_packed(self, tensors):
        self._hit('stage')
        return {'kind': PACKED, 'bytes': sum(t.nbytes for t in tensors.values())}

    def staged_bytes(self, s): return s['bytes']
    def sync(self): self._hit('sync')

    def snapshot(self, m): return {'state': m.state, 'bytes': 7}
    def snapshot_bytes(self, snap): return snap['bytes']

    def install_dense(self, m, staged):
        self._hit('install')
        m.state = (DENSE, None)
        m.writes += 1

    def install_packed(self, m, staged, params):
        self._hit('install')
        m.state = (PACKED, 'x')
        m.writes += 1

    def restore(self, m, snap):
        self._hit('restore')
        m.state = snap['state']

    def memory(self): return {'gpu_allocated': 1}


def transformation(representation=PACKED, **kw):
    t = {'policy': 'w4a16' if representation == PACKED else 'source-precision', 'representation': representation,
         'implementation': TRANSFORM, 'backend': 'fake 1'}
    if representation == PACKED:
        t.update(scheme='W4A16', algorithm='rtn', bits=4, group_size=128, symmetric=True,
                 format='compressed-tensors/pack-quantized', compute_dtype='bfloat16')
    t.update(kw)
    return t


def identity(cid, modules, shape=(256, 256), **kw):
    return {'id': cid, 'transformation': transformation(**kw),
            'members': [{'module': m, 'shape': list(shape), 'dtype': 'bfloat16'} for m in modules]}


def rep(group, policy, modules, cid=None, representation=None, **kw):
    r = {'group': group, 'policy': policy, 'modules': modules, 'transformation': transformation(
        representation or (PACKED if policy == 'w4a16' else DENSE), **kw)}
    if cid:
        r['component'] = cid
    return r


def make(ops=None, names=('a', 'b', 'c', 'd'), shape=(256, 256), stage_bytes=1 << 30, **mod):
    ops = ops or Ops()
    modules = {n: Module(shape, **mod) for n in names}
    ex = TrialExecutor(ops, modules, 'cuda', 'bfloat16', stage_bytes=stage_bytes)
    opened = ex.open({'g1': ['a', 'b'], 'g2': ['c', 'd']})
    return ex, modules, ops, opened


def states(modules): return {n: m.state[0] for n, m in sorted(modules.items())}


class ExecutorTests(unittest.TestCase):
    def test_open_reports_source_and_keeps_canonical_in_ram(self):
        ex, modules, ops, opened = make()
        self.assertEqual(opened['canonical_bytes'], 4 * 2 * 256 * 256)
        self.assertEqual([m['module'] for m in opened['modules']], ['a', 'b', 'c', 'd'])
        self.assertEqual(opened['backend'], 'fake 1')
        self.assertEqual(set(states(modules).values()), {DENSE})
        self.assertEqual(ex.state()['groups'], {'g1': DENSE, 'g2': DENSE})

    def test_open_refuses_a_model_that_is_not_dense_source(self):
        ops = Ops()
        modules = {'a': Module((256, 256), rep=PACKED)}
        with self.assertRaises(TrialError) as cm:
            TrialExecutor(ops, modules, 'cuda', 'bfloat16').open({'g': ['a']})
        self.assertEqual(cm.exception.cls, 'trial_incompatible')

    def test_transform_apply_and_reverse_replace_only_the_named_group(self):
        ex, modules, ops, _ = make()
        built = ex.transform(identity('c1', ['a', 'b']))
        self.assertGreater(built['bytes'], 0)
        self.assertEqual(built['digest'], 'digest:a,b')
        res = ex.apply([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        self.assertEqual(states(modules), {'a': PACKED, 'b': PACKED, 'c': DENSE, 'd': DENSE})
        self.assertEqual(modules['c'].writes + modules['d'].writes, 0)
        self.assertEqual(res['modules'], 2)
        self.assertEqual(res['bytes_released'], 14)
        self.assertEqual(ex.state()['groups'], {'g1': PACKED, 'g2': DENSE})
        ex.apply([rep('g1', 'source-precision', ['a', 'b'])])
        self.assertEqual(set(states(modules).values()), {DENSE})

    def test_release_refuses_a_component_in_use(self):
        ex, modules, _, _ = make()
        ex.transform(identity('c1', ['a']))
        ex.apply([rep('g1', 'w4a16', ['a'], 'c1')])
        with self.assertRaises(TrialError):
            ex.release('c1')
        ex.apply([rep('g1', 'source-precision', ['a'])])
        self.assertGreater(ex.release('c1')['released'], 0)
        self.assertEqual(ex.release('c1')['released'], 0)

    def test_canonical_source_is_never_replaced_or_written(self):
        ex, modules, _, _ = make()
        before = {n: ex.canon[n]['tensor'] for n in modules}
        ex.transform(identity('c1', ['a', 'b']))
        ex.apply([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        ex.apply([rep('g1', 'source-precision', ['a', 'b'])])
        for n in modules:
            self.assertIs(ex.canon[n]['tensor'], before[n])

    def test_transform_failure_leaves_nothing_behind(self):
        ex, modules, ops, _ = make(Ops(fail={'build': 2}))
        with self.assertRaises(TrialError) as cm:
            ex.transform(identity('c1', ['a', 'b']))
        self.assertEqual(cm.exception.cls, 'trial_failed')
        self.assertNotIn('c1', ex.components)
        self.assertEqual(set(states(modules).values()), {DENSE})

    def test_staging_failure_changes_nothing(self):
        ops = Ops(fail={'stage': 2})
        ex, modules, _, _ = make(ops)
        ex.transform(identity('c1', ['a', 'b']))
        with self.assertRaises(TrialError) as cm:
            ex.apply([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        self.assertEqual(cm.exception.cls, 'trial_failed')
        self.assertEqual(set(states(modules).values()), {DENSE})
        self.assertEqual(sum(m.writes for m in modules.values()), 0)

    def test_install_failure_is_undone_in_place(self):
        ops = Ops(fail={'install': 2})
        ex, modules, _, _ = make(ops)
        ex.transform(identity('c1', ['a', 'b']))
        with self.assertRaises(TrialError) as cm:
            ex.apply([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        self.assertEqual(cm.exception.cls, 'trial_failed')
        self.assertEqual(set(states(modules).values()), {DENSE})

    def test_failed_undo_reports_the_state_as_lost(self):
        ops = Ops(fail={'install': 2, 'restore': 1})
        ex, modules, _, _ = make(ops)
        ex.transform(identity('c1', ['a', 'b']))
        with self.assertRaises(TrialError) as cm:
            ex.apply([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        self.assertEqual(cm.exception.cls, 'trial_state_lost')

    def test_failed_chunk_reverses_the_chunks_before_it(self):
        # one module per chunk: the third module's staging fails after two chunks
        ops = Ops()
        ex, modules, _, _ = make(ops, stage_bytes=1)
        ex.transform(identity('c1', ['a', 'b', 'c']))
        ops.fail = {'stage': ops.calls.get('stage', 0) + 3}
        with self.assertRaises(TrialError) as cm:
            ex.apply([rep('g1', 'w4a16', ['a', 'b', 'c'], 'c1')])
        self.assertEqual(cm.exception.cls, 'trial_failed')
        self.assertEqual(set(states(modules).values()), {DENSE})

    def test_validation_rejections(self):
        ex, modules, ops, _ = make()
        ex.transform(identity('c1', ['a', 'b']))
        ok = rep('g1', 'w4a16', ['a', 'b'], 'c1')
        self.assertTrue(ex.validate([ok])['ok'])
        # unknown module, missing component, wrong compute dtype: incompatible
        for bad, why in [(rep('g1', 'w4a16', ['zz'], 'c1'), 'module'),
                         (rep('g1', 'w4a16', ['a'], 'nope'), 'component'),
                         (rep('g1', 'w4a16', ['a'], 'c1', compute_dtype='float32'), 'dtype')]:
            with self.assertRaises(TrialError, msg=why) as cm:
                ex.validate([bad])
            self.assertEqual(cm.exception.cls, 'trial_incompatible', why)
        with self.assertRaises(TrialError) as cm:
            ex.validate([ok, rep('g2', 'source-precision', ['a'])])
        self.assertIn('twice', str(cm.exception))

    def test_validation_rejects_modules_that_cannot_be_swapped_safely(self):
        for kw, text in [({'cls': 'QuantLinear'}, 'plain Linear'), ({'hooked': True}, 'dispatch hook')]:
            ex, modules, _, _ = make(**kw)
            ex.transform(identity('c1', ['a']))
            with self.assertRaises(TrialError) as cm:
                ex.validate([rep('g1', 'w4a16', ['a'], 'c1')])
            self.assertEqual(cm.exception.cls, 'trial_incompatible')
            self.assertIn(text, str(cm.exception))
        ops = Ops()
        modules = {'a': Module((256, 256)), 'b': Module((256, 256))}
        ex = TrialExecutor(ops, modules, 'cuda', 'bfloat16', shared={'a'})
        ex.open({'g': ['a', 'b']})
        ex.transform(identity('c1', ['a']))
        with self.assertRaises(TrialError) as cm:
            ex.validate([rep('g', 'w4a16', ['a'], 'c1')])
        self.assertIn('shares its weight', str(cm.exception))

    def test_shape_mismatch_between_component_and_module_is_incompatible(self):
        ex, modules, _, _ = make()
        with self.assertRaises(TrialError) as cm:
            ex.transform(identity('c1', ['a'], shape=(128, 256)))
        self.assertEqual(cm.exception.cls, 'trial_incompatible')

    def test_unsupported_transformations(self):
        ex, modules, ops, _ = make()
        with self.assertRaises(TrialError) as cm:
            ex.transform(identity('c1', ['a'], bits=8))
        self.assertEqual(cm.exception.cls, 'trial_incompatible')
        with self.assertRaises(TrialError) as cm:
            ex.validate([rep('g1', 'fused', ['a'], 'c1', representation='fused-int4')])
        self.assertEqual(cm.exception.cls, 'trial_unsupported')
        with self.assertRaises(TrialError) as cm:
            ex.transform(identity('c1', ['a'], implementation='other'))
        self.assertEqual(cm.exception.cls, 'trial_incompatible')
        ops.unsupported = 'compressed_tensors missing'
        with self.assertRaises(TrialError) as cm:
            ex.transform(identity('c2', ['a']))
        self.assertEqual(cm.exception.cls, 'trial_incompatible')

    def test_drift_is_unsupported_for_apply_and_repaired_by_reconstruct(self):
        ex, modules, _, _ = make()
        ex.transform(identity('c1', ['a', 'b']))
        modules['a'].state = (PACKED, 'external')  # observed differs from tracked
        with self.assertRaises(TrialError) as cm:
            ex.apply([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        self.assertEqual(cm.exception.cls, 'trial_unsupported')
        ex.reconstruct([rep('g1', 'w4a16', ['a', 'b'], 'c1')])
        self.assertEqual(states(modules)['a'], PACKED)
        self.assertEqual(ex.current['a'], (PACKED, 'c1'))

    def test_dispatch(self):
        ex, modules, _, _ = make()
        self.assertEqual(trial_dispatch(ex, 'trial_state', {})['groups'], {'g1': DENSE, 'g2': DENSE})
        with self.assertRaises(ValueError):
            trial_dispatch(ex, 'trial_nope', {})


class Provider:
    """Just enough of a provider for the worker's request loop."""

    def __init__(self, trial=None):
        self.trial = trial

    def stats(self):
        return {}


def serve_lines(provider, requests):
    out = io.StringIO()
    # runpy hands back a copy of the script's globals; the functions keep the
    # originals, which is where the protocol stream lives.
    live = worker['emit'].__globals__
    saved_proto, saved_stdin = live['_proto'], sys.stdin
    live['_proto'], sys.stdin = out, io.StringIO(''.join(json.dumps(r) + '\n' for r in requests))
    try:
        worker['serve'](provider)
    finally:
        live['_proto'], sys.stdin = saved_proto, saved_stdin
    return [json.loads(line) for line in out.getvalue().splitlines()]


class ProtocolTests(unittest.TestCase):
    def test_trial_operations_travel_over_the_worker_protocol(self):
        ex, modules, _, _ = make()
        replies = serve_lines(Provider(ex), [
            {'id': 1, 'op': 'trial_state'},
            {'id': 2, 'op': 'trial_transform', 'identity': identity('c1', ['a', 'b'])},
            {'id': 3, 'op': 'trial_validate', 'replacements': [rep('g1', 'w4a16', ['a', 'b'], 'c1')]},
            {'id': 4, 'op': 'trial_apply', 'replacements': [rep('g1', 'w4a16', ['a', 'b'], 'c1')]},
            {'id': 5, 'op': 'trial_state'},
        ])
        self.assertTrue(all(r['ok'] for r in replies), replies)
        self.assertEqual(replies[0]['result']['groups'], {'g1': DENSE, 'g2': DENSE})
        self.assertEqual(replies[4]['result']['groups'], {'g1': PACKED, 'g2': DENSE})
        self.assertEqual(replies[3]['result']['modules'], 2)

    def test_refusals_carry_the_trial_error_class(self):
        ex, modules, _, _ = make()
        replies = serve_lines(Provider(ex), [
            {'id': 1, 'op': 'trial_validate', 'replacements': [rep('g1', 'w4a16', ['zz'], 'nope')]},
            {'id': 2, 'op': 'trial_validate', 'replacements': [rep('g1', 'fused', ['a'], 'c1', representation='fused-int4')]},
            {'id': 3, 'op': 'trial_state'},
        ])
        self.assertEqual([r['ok'] for r in replies], [False, False, True])
        self.assertEqual(replies[0]['error']['class'], 'trial_incompatible')
        self.assertEqual(replies[1]['error']['class'], 'trial_unsupported')

    def test_a_worker_that_is_not_a_trial_session_refuses_trial_operations(self):
        replies = serve_lines(Provider(None), [{'id': 1, 'op': 'trial_state'}])
        self.assertFalse(replies[0]['ok'])
        self.assertEqual(replies[0]['error']['class'], 'request_invalid')


if __name__ == '__main__':
    unittest.main()
