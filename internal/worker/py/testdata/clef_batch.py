"""Lightweight Clef batch contract tests, without model weights or CUDA."""
import contextlib
import runpy
import sys
import types
import unittest

worker = runpy.run_path(sys.argv.pop(1))
ClefProvider = worker['ClefProvider']


class Tensor:
    def __init__(self, values): self.values = values
    def float(self): return self
    def softmax(self, dim): return self
    def tolist(self): return self.values


class ClefBatch(unittest.TestCase):
    def test_ragged_questions_and_multiple_states(self):
        provider = ClefProvider.__new__(ClefProvider)
        transfers = []
        def combine(ts):
            class Combined(Tensor):
                def tolist(self):
                    transfers.append(True)
                    return self.values
            return Combined(sum((t.values for t in ts), []))
        provider.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, cat=combine)
        provider.tokenizer = types.SimpleNamespace(pad_token_id=0)
        provider.reference = lambda: types.SimpleNamespace(device='cpu')
        calls = []
        def encode(tokenizer, record, max_length):
            return types.SimpleNamespace(input_ids=[1, 2, 3], questions=[
                types.SimpleNamespace(question_id='a', option_ids=['x', 'y']),
                types.SimpleNamespace(question_id='b', option_ids=['one', 'two', 'three'])])
        def collate(records, pad, device):
            calls.append(len(records))
            return records
        provider.jsm = types.SimpleNamespace(encode_record=encode, collate_records=collate)
        provider.model = lambda records: [[Tensor([0.2, 0.8]), Tensor([0.1, 0.3, 0.6])]
                                           for _ in records]
        result = provider.predict(['first', 'second'], [])
        self.assertEqual(calls, [2])
        self.assertEqual(len(transfers), 1)
        self.assertEqual(provider.batch_profile['forwards'], 1)
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertEqual(row['answers']['a']['probabilities'], {'x': 0.2, 'y': 0.8})
            self.assertEqual(row['answers']['b']['probabilities'], {'one': 0.1, 'two': 0.3, 'three': 0.6})
            self.assertEqual(row['answers']['b']['choice'], 'three')
        single = provider.predict(['first'], [])
        self.assertEqual(single, result[:1])
        self.assertEqual(calls, [2, 1])
        self.assertEqual(len(transfers), 2)

    def test_encoded_length_splits_within_bound(self):
        provider = ClefProvider.__new__(ClefProvider)
        provider.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                               cat=lambda ts: Tensor(sum((t.values for t in ts), [])))
        provider.tokenizer = types.SimpleNamespace(pad_token_id=0)
        provider.reference = lambda: types.SimpleNamespace(device='cpu')
        calls = []
        def encode(tokenizer, record, max_length):
            return types.SimpleNamespace(input_ids=[1] * int(record['state']), questions=[
                types.SimpleNamespace(question_id='q', option_ids=['a','b'])])
        provider.jsm = types.SimpleNamespace(encode_record=encode,
                     collate_records=lambda records, pad, device: (calls.append(len(records)) or records))
        provider.model = lambda records: [[Tensor([0.6,0.4])] for _ in records]
        result = provider.predict(['10','10','100','100'], [])
        self.assertEqual(calls, [2,2])
        self.assertEqual(len(result),4)
        self.assertEqual(provider.batch_profile['forwards'],2)
        self.assertEqual(result, [provider.predict([length], [])[0]
                                  for length in ['10','10','100','100']])
        calls.clear()
        provider.predict(['5000','5000'], [])
        self.assertEqual(calls, [1,1])


if __name__ == '__main__': unittest.main()
