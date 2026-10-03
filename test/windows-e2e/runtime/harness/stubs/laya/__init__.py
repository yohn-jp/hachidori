"""Windows E2E fixture standing in for the Laya provider and its weights.

This is NOT Laya and loads no model. It is the lightweight, deterministic model
payload behind the real hachidori_worker.py: closed-option scoring by a fixed,
documented rule, so a typed decision answered through the real packaged
executable, application controller, worker process and HTTP API can be checked
exactly.

Rule (mirrored by harness.StubAnswer in Go): for every choice c of a question,
weight(c) = 1 + the number of whole-word, case-insensitive occurrences of c in
the state; the probabilities are the weights normalised to sum 1; the reported
choice is the first choice with the largest probability.
"""

import re

__version__ = "0.3.21+e2e-fixture"


class _Device:
    type = "cpu"

    def __str__(self):
        return "cpu"


class Agent:
    device = _Device()
    dtype = "float32"

    def predict_batch(self, states, questions):
        out = []
        for state in states:
            text = state.lower()
            answers = {}
            for qid, q in questions.items():
                choices = list(q["criteria"].keys())
                weights = [
                    1 + len(re.findall(r"\b" + re.escape(c.lower()) + r"\b", text))
                    for c in choices
                ]
                total = float(sum(weights))
                probs = {c: w / total for c, w in zip(choices, weights)}
                best = max(range(len(choices)), key=lambda i: (weights[i], -i))
                answers[qid] = {
                    "choice": choices[best],
                    "answer_confidence": probs[choices[best]],
                    "probabilities": probs,
                }
            out.append({"answers": answers})
        return out


def load(model_dir, device="cpu", expected_sha256=None):
    # The fixture never places anything on another device: a request for one is
    # refused, never served on the CPU instead.
    if device != "cpu":
        raise RuntimeError("the E2E fixture model only exists on the cpu, not %r" % (device,))
    return Agent()
