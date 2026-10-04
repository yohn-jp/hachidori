"""Portable worker startup contract tests (issue #259); no torch, no CUDA.

The failure class: torch._inductor.codecache registers InductorCacheArtifact and
then, at module level, needs getpass.getuser(). Where that raises, the first import
fails after registering and a later import of the module re-runs the one-shot
registration. These tests prove the worker's import/load sequence reports the real
cause before such a module is imported, and that a startup failure keeps a bounded
traceback. They are not evidence about Windows, CUDA or PyTorch itself."""
import io
import json
import os
import runpy
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

worker = runpy.run_path(sys.argv.pop(1))
Provider = worker['Provider']
require_account_name = worker['require_account_name']
IDENTITY = ('LOGNAME', 'USER', 'LNAME', 'USERNAME')


def without_identity():
    env = {k: v for k, v in os.environ.items() if k not in IDENTITY}
    return patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {'pwd': None})


class Probe(Provider):
    """A provider whose import and load steps are scripted by the test."""
    name = 'probe'

    def __init__(self, imports=None, load=None):
        super().__init__('m', 'cpu', {'files': {}})
        self.imports, self.loader = imports, load

    def import_provider(self):
        if self.imports:
            self.imports()

    def load(self):
        if self.loader:
            self.loader()

    def placed_device(self):
        return types.SimpleNamespace(type='cpu')


def initialize(provider):
    """Run Provider.initialize; returns (exit code, protocol events, worker log)."""
    proto, log = io.StringIO(), io.StringIO()
    code = None
    with patch.dict(Provider.initialize.__globals__, {'_proto': proto}), redirect_stderr(log):
        try:
            provider.initialize()
        except SystemExit as e:
            code = e.code
    return code, [json.loads(line) for line in proto.getvalue().splitlines()], log.getvalue().splitlines()


class AccountNameTests(unittest.TestCase):
    def test_missing_account_name_is_reported_as_itself(self):
        env, pwd = without_identity()
        with env, pwd:
            with self.assertRaises(RuntimeError) as caught:
                require_account_name()
        self.assertIn('no account name', str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_supplied_account_name_passes(self):
        env, pwd = without_identity()
        with env, pwd, patch.dict(os.environ, {'USERNAME': 'hachidori'}):
            require_account_name()


class ImportSequenceTests(unittest.TestCase):
    """A stand-in for torch._inductor.codecache: registers once, then needs the account name."""

    def module(self, path):
        with open(os.path.join(path, 'fake_codecache.py'), 'w') as f:
            f.write(textwrap.dedent('''
                import getpass
                REGISTRY = []
                assert 'inductor' not in REGISTRY, 'Artifact of type=inductor already registered'
                REGISTRY.append('inductor')
                CACHE_DIR = getpass.getuser()
            '''))

    def run_sequence(self, identity):
        with tempfile.TemporaryDirectory() as path:
            self.module(path)
            seen = []

            def imports():
                # The worker's real order: provider imports, then load imports the
                # kernel-adjacent modules. The stand-in is imported at load time.
                seen.append('import_provider')

            def load():
                seen.append('load')
                import fake_codecache  # noqa: F401

            env, pwd = without_identity()
            extra = {'USERNAME': 'hachidori'} if identity else {}
            sys.modules.pop('fake_codecache', None)
            with env, pwd, patch.dict(os.environ, extra), patch.object(sys, 'path', [path] + sys.path), \
                    patch.dict(sys.modules, {'torch': type(sys)('torch')}):
                result = initialize(Probe(imports, load))
                registry = list(getattr(sys.modules.get('fake_codecache'), 'REGISTRY', []))
            sys.modules.pop('fake_codecache', None)
            return result, seen, registry

    def test_without_account_name_nothing_registers_and_cause_is_reported(self):
        (code, events, log), seen, registry = self.run_sequence(identity=False)
        self.assertEqual(code, 3)
        self.assertEqual(seen, [], 'neither provider import nor load may run')
        self.assertEqual(registry, [])
        fatal = [e for e in events if e['event'] == 'fatal']
        self.assertEqual(len(fatal), 1)
        self.assertEqual(fatal[0]['class'], 'provider_import')
        self.assertIn('no account name', fatal[0]['message'])
        self.assertNotIn('already registered', json.dumps(events))
        self.assertTrue(any('provider_import traceback' in line for line in log))

    def test_with_account_name_registers_exactly_once(self):
        (code, events, log), seen, registry = self.run_sequence(identity=True)
        self.assertIsNone(code)
        self.assertEqual(seen, ['import_provider', 'load'])
        self.assertEqual(registry, ['inductor'])
        self.assertEqual([e['event'] for e in events], ['phase', 'phase'])


class TraceTests(unittest.TestCase):
    def failing_load(self, depth):
        # Distinct frames: the traceback module folds a repeated frame into one line.
        namespace = {}
        exec('def level0():\n    raise ImportError("x" * 5000)\n' + ''.join(
            'def level%d():\n    level%d()\n' % (i, i - 1) for i in range(1, depth + 1)), namespace)

        def deeper(n):
            namespace['level%d' % n]()

        def load():
            try:
                deeper(depth)
            except ImportError as e:
                raise AssertionError('Artifact of type=inductor already registered in mega-cache artifact factory') from e
        return load

    def run_load(self, depth):
        env, pwd = without_identity()
        with env, pwd, patch.dict(os.environ, {'USERNAME': 'hachidori'}), \
                patch.dict(sys.modules, {'torch': type(sys)('torch')}):
            return initialize(Probe(None, self.failing_load(depth)))

    def test_model_load_failure_keeps_bounded_traceback(self):
        code, events, log = self.run_load(depth=60)
        self.assertEqual(code, 3)
        fatal = [e for e in events if e['event'] == 'fatal'][0]
        self.assertEqual(fatal['class'], 'model_load')
        self.assertTrue(fatal['message'].startswith('AssertionError: Artifact of type=inductor'))
        trace = [line for line in log if line.startswith('[worker] ')]
        self.assertLessEqual(len(trace), worker['TRACE_MAX_LINES'] + 2)  # header, trace, fatal line
        limit = len('[worker]   ') + worker['TRACE_MAX_LINE'] + len('...[truncated]')
        self.assertTrue(all(len(line) <= limit or line.startswith('[worker] fatal') for line in trace), trace)
        text = '\n'.join(log)
        self.assertIn('model_load traceback:', text)
        self.assertIn('level0', text)  # the raising end is retained
        self.assertNotIn('level%d' % (60 - worker['TRACE_MAX_FRAMES'] - 1), text.replace('level0', ''))
        self.assertIn('AssertionError', text)  # the failing end of the chain is retained
        self.assertIn('...[truncated]', text)
        # The tail the supervisor keeps ends with the fatal line, the trace just before it.
        self.assertTrue(log[-1].startswith('[worker] fatal model_load'))

    def test_long_exception_chain_keeps_the_final_lines(self):
        def load():
            error = None
            for i in range(10):
                try:
                    raise RuntimeError('link %d' % i) from error
                except RuntimeError as e:
                    error = e
            raise error
        env, pwd = without_identity()
        with env, pwd, patch.dict(os.environ, {'USERNAME': 'hachidori'}), \
                patch.dict(sys.modules, {'torch': type(sys)('torch')}):
            code, events, log = initialize(Probe(None, load))
        text = '\n'.join(log)
        self.assertEqual(code, 3)
        self.assertRegex(text, r'model_load traceback \(\d+ earlier lines omitted\):')
        self.assertIn('RuntimeError: link 9', text)
        self.assertNotIn('RuntimeError: link 0', text)
        self.assertLessEqual(len([line for line in log if line.startswith('[worker]   ')]), worker['TRACE_MAX_LINES'])

    def test_shallow_failure_is_complete_and_not_marked_omitted(self):
        code, events, log = self.run_load(depth=1)
        self.assertEqual(code, 3)
        text = '\n'.join(log)
        self.assertIn('model_load traceback:', text)
        self.assertNotIn('omitted', text)
        self.assertIn('level0', text)


if __name__ == '__main__':
    unittest.main()
