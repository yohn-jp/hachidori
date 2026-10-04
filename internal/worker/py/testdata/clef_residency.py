"""Clef continuation ownership tests without CUDA or model weights."""
import contextlib
import runpy
import sys
import types
import unittest

worker = runpy.run_path(sys.argv.pop(1))
ClefProvider = worker['ClefProvider']
ClefResidentState = worker['ClefResidentState']


class Storage:
    def __init__(self, size, pointer):
        self.size, self.pointer = size, pointer

    def nbytes(self): return self.size
    def data_ptr(self): return self.pointer


class Tensor:
    next_pointer = 0

    def __init__(self, values):
        self.values = values
        self.device = 'cuda'
        Tensor.next_pointer += 1
        self.storage = Storage(len(values) * 2, Tensor.next_pointer)

    def untyped_storage(self): return self.storage
    def float(self): return self
    def softmax(self, dim): return self
    def tolist(self): return self.values


class Cache:
    def __init__(self):
        self.kv = Tensor([1, 2])
        self.recurrent = Tensor([3, 4])
        self.positions = []


class Model:
    def __init__(self):
        self.calls = []
        self.fail_at = None

    def __call__(self, input_ids, past_key_values, use_cache, return_dict):
        assert use_cache and return_dict
        self.calls.append(len(input_ids.values))
        if self.fail_at == len(self.calls):
            raise RuntimeError('prefill failed')
        cache = past_key_values or Cache()
        cache.positions.extend(input_ids.values)
        return types.SimpleNamespace(past_key_values=cache,
                                     last_hidden_state=Tensor(list(input_ids.values)))


class ResidentContract(unittest.TestCase):
    def setUp(self):
        self.provider = p = ClefProvider.__new__(ClefProvider)
        p.requested, p.want_dtype = 'cuda', 'bfloat16'
        p.variant = {'id': 'clef-flash--clef-flash-w4a16-rtn-g128--6cdd68bf9677'}
        p.digests = {'joint_schema_model.py':
                     '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'}
        p.kernel_paths = {'fla': 'pinned'}
        p.transformers = types.SimpleNamespace(__version__='5.17.0')
        p.torch = types.SimpleNamespace(__version__='2.11.0+cu128', long='long',
                                        inference_mode=contextlib.nullcontext,
                                        tensor=lambda xs, **kw: Tensor(xs[0]),
                                        cat=lambda xs, dim=0: Tensor(sum((x.values for x in xs), [])))
        p.tokenizer = types.SimpleNamespace(pad_token_id=0)
        p.reference = lambda: types.SimpleNamespace(device='cuda')
        self.text = Model()
        head_calls = []
        def head(hidden, ids, mask, records, weight):
            head_calls.append((hidden.values, ids.values))
            return [[Tensor([0.2, 0.8])] for _ in records]
        p.model = types.SimpleNamespace(head=head)
        p.backbone = types.SimpleNamespace(model=self.text,
                        get_output_embeddings=lambda: types.SimpleNamespace(weight='weights'))
        self.head_calls = head_calls

        def encode(tokenizer, record, max_length):
            state = [ord(c) for c in str(record['state'])][:max_length - 4]
            questions = record['questions']
            q = next(iter(questions))
            suffix = [11 if q == '__resident_probe__' else 12, len(q)]
            return types.SimpleNamespace(input_ids=tuple([1] + state + [2] + suffix),
                media=None, questions=[types.SimpleNamespace(question_id=q,
                option_ids=['no', 'yes'])])
        def collate(records, pad, device):
            ids = Tensor(list(records[0].input_ids))
            return {'input_ids': ids, 'attention_mask': Tensor([1] * len(ids.values))}
        p.jsm = types.SimpleNamespace(encode_record=encode, collate_records=collate,
                                      render=str, _tokens=lambda tokenizer, text: [ord(c) for c in text])

    def test_chunk_forks_identity_and_accounting(self):
        p = self.provider
        state = 's' * 1100
        resident = p.build_resident(state, {'q1': {}})
        self.assertIsInstance(resident, ClefResidentState)
        self.assertEqual(self.text.calls, [512, 512, 77])
        self.assertEqual(resident.prefix_tokens, 1101)
        self.assertEqual(resident._hidden.values, list(resident._prefix))
        self.assertEqual(resident._cache.positions, list(resident._prefix))
        self.assertEqual(resident.payload_bytes, 8 + 2 * resident.prefix_tokens)
        self.assertEqual(resident.identity, p.build_resident(state, {'q1': {}}).identity)
        self.assertEqual(resident.payload_bytes, p.build_resident(state, {'q1': {}}).payload_bytes)
        with self.assertRaises(AttributeError):
            resident._cache = Cache()
        initial = list(resident._cache.positions)
        results = []
        for q in ['q1', 'q2', 'q1']:
            result, usage = p.predict_resident(resident, state, {q: {}})
            results.append(result)
            self.assertEqual(usage['fork_bytes'], 8)
            self.assertGreater(usage['working_bytes'], usage['fork_bytes'])
            self.assertEqual(resident._cache.positions, initial)
        self.assertEqual(results[0]['answers']['q1'], results[2]['answers']['q1'])
        self.assertEqual(results[1]['answers']['q2']['choice'], 'yes')
        self.assertEqual(self.text.calls[-3:], [3, 3, 3])
        self.assertEqual(len(self.head_calls), 3)
        self.assertEqual(self.head_calls[0][0], list(resident._prefix) + [2, 12, 2])
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            p.predict_resident(resident, 'other', {'q1': {}})
        p.variant = dict(p.variant, revision='changed')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            p.predict_resident(resident, state, {'q1': {}})

    def test_effective_prefix_uses_encoder_truncation(self):
        p = self.provider
        state = 's' * 17000
        resident = p.build_resident(state, {'q1': {}})
        self.assertEqual(resident.prefix_tokens, worker['CLEF_MAX_LENGTH'] - 3)
        self.assertTrue(all(n <= 512 for n in self.text.calls))
        self.assertEqual(sum(self.text.calls), resident.prefix_tokens)
        # Identical effective prefix is the identity, even if raw State differs
        # only after the encoder's truncation boundary.
        same = p.build_resident(state + 'more', {'q1': {}})
        self.assertEqual(resident.identity, same.identity)
        self.assertEqual(p.predict_resident(resident, state + 'more', {'q2': {}})[0]
                         ['answers']['q2']['choice'], 'yes')

    def test_failed_construction_never_returns_resident(self):
        self.text.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, 'prefill failed'):
            self.provider.build_resident('s' * 1100, {'q1': {}})
        self.assertEqual(self.text.calls, [512, 512])

    def test_uncertified_artifact_refused_without_fallback(self):
        self.provider.requested = 'cpu'
        with self.assertRaisesRegex(RuntimeError, 'not certified'):
            self.provider.build_resident('state', {'q1': {}})
        self.assertEqual(self.text.calls, [])


if __name__ == '__main__':
    unittest.main()
