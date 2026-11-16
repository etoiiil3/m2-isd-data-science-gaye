"""Demo contract (NOT graded: marked 'stretch'). Run before the live demo: pytest -m stretch"""
import numpy as np
import pytest

from asr_demo.contract import OfflineBackend
from asr_demo.metrics import run_session_offline

pytestmark = pytest.mark.stretch  # not in `pytest -m d3`: the stretch is optional
WORDS = [f"w{i}" for i in range(30)]


class MockModel(OfflineBackend):
    """One word per 0.4 s of audio; the last word is unstable (changes between calls), like real models."""
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        n = int(len(audio) / 16000 / 0.4)
        out = WORDS[:n]
        if out and self.calls % 2:
            out = out[:-1] + ["x" + out[-1]]
        return " ".join(out)


def test_local_agreement_never_retracts_and_finishes_complete():
    from speech.streaming_live import LocalAgreementBackend
    b = LocalAgreementBackend(MockModel(), step_s=0.5, n_agree=2)
    s = b.new_session()
    y = np.zeros(16000 * 6, np.float32)
    committed_prefixes = []
    for i in range(0, len(y), 1600):
        t = s.accept_chunk(y[i:i + 1600])
        if t:
            committed_prefixes.append(list(s.committed))
    for a, c in zip(committed_prefixes, committed_prefixes[1:]):
        assert c[:len(a)] == a, "committed words must never change"
    final = s.finish().split()
    assert len(final) == 15 and final[:14] == WORDS[:14]  # the mock may garble its last word on any call
    assert committed_prefixes and len(committed_prefixes[-1]) > 5


def test_harness_metrics():
    b = MockModel()
    b.name, b.calls = "mock", 1  # next call is even: no garbled word
    m = run_session_offline(b, np.zeros(16000 * 2, np.float32), 16000, reference="w0 w1 w2 w3 w4")
    assert m["wer"] == 0 and m["rtf"] >= 0 and m["first_partial_ms"] is None and m["audio_s"] == 2.0
