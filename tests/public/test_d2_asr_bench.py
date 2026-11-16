import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.d2
TEXTS = ["hello world", "the cat sat", "good morning"]


class FakeASR:
    """Pretends to be a transformers ASR pipeline; utterance i is tagged by its first sample (i+1).
    The 3rd utterance is transcribed wrongly. Like the real pipeline, it consumes (pops) the input dicts."""
    model = types.SimpleNamespace(parameters=lambda: [types.SimpleNamespace(numel=lambda: 10), types.SimpleNamespace(numel=lambda: 5)])

    def __call__(self, inputs, batch_size=1):
        out = []
        for x in inputs:
            i = int(x.pop("raw")[0]) - 1
            x.pop("sampling_rate")
            out.append({"text": "WRONG WORDS" if i == 2 else TEXTS[i]})
        return out


@pytest.fixture
def fake_transformers(monkeypatch, fake_torch):
    tr = types.ModuleType("transformers")
    tr.pipeline = lambda *a, **k: FakeASR()
    monkeypatch.setitem(sys.modules, "transformers", tr)


def test_bench_model_metrics(fake_transformers):
    from speech.asr_benchmark import bench_model
    data = [dict(audio=np.r_[float(i + 1), np.zeros(15999)], sr=16000, text=t) for i, t in enumerate(TEXTS)]
    m, hyps = bench_model("fake/model", data, 1, "cpu", False)
    assert len(hyps) == 3
    assert m["wer"] == pytest.approx(2 / 7)  # 'good morning' -> 'wrong words': 2 errors / 7 reference words
    assert 0 < m["rtf"] < 10 and "lat_p90_ms" in m and m["n_params_M"] > 0
