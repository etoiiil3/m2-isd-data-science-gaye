import numpy as np
import pytest

from speech.streaming_sim import _mock, _mock_native, commit_lags, local_agreement, simulate_streaming, stream_native

pytestmark = pytest.mark.d3


def setup(seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(50)]
    ends = np.cumsum(rng.uniform(0.25, 0.6, 50)) + 0.3
    audio = np.zeros(int((ends[-1] + 0.5) * 16000), np.float32)
    return rng, words, ends, audio


def test_local_agreement_is_the_common_prefix_of_the_last_n():
    h = [["a", "b", "x"], ["the", "cat", "sad"], ["the", "cat", "sat", "on"]]
    assert local_agreement(h, 2) == ["the", "cat"]
    assert local_agreement(h, 3) == []
    assert local_agreement(h[-1:], 2) == []           # fewer than n hypotheses: nothing yet
    assert local_agreement([["a", "b"], ["a", "b"]], 2) == ["a", "b"]


def test_fast_model_streams_with_bounded_lag():
    import jiwer
    rng, words, ends, audio = setup()
    r = simulate_streaming(audio, 16000, _mock(words, ends, 16000, rng), step_s=0.5, cost_model=lambda d: 0.05 * d + 0.02)
    assert jiwer.wer(" ".join(words), r["text"]) < 0.12
    assert np.median(commit_lags(r["words"], r["commit_times"], words, ends)) < 2.5
    assert not r["behind"]


def test_slow_model_falls_behind():
    rng, words, ends, audio = setup(1)
    r = simulate_streaming(audio, 16000, _mock(words, ends, 16000, rng), step_s=0.5, cost_model=lambda d: 0.5 * d + 0.1)
    assert r["behind"]


def test_the_clock_waits_for_a_busy_model():
    # constant cost 0.3 s per update, step 1 s: each update ends 0.3 s after its audio; 1 + 5 s of audio
    calls = []

    def transcribe(buf):
        calls.append(len(buf) / 16000)
        return "a b c"
    r = simulate_streaming(np.zeros(5 * 16000, np.float32), 16000, transcribe, step_s=1.0, cost_model=lambda d: 0.3)
    assert calls == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.allclose(r["commit_times"], [2.3, 2.3, 2.3])     # "a b c" agreed at the 2nd update (t = 2 s)
    assert r["finish_delay_s"] == pytest.approx(0.3) and not r["behind"]
    slow = simulate_streaming(np.zeros(5 * 16000, np.float32), 16000, lambda b: "a", step_s=1.0, cost_model=lambda d: 1.5)
    assert slow["updates"] < 5 and slow["behind"]             # busy: updates are skipped, not queued


def test_native_streaming_commits_only_finished_words():
    _, words, ends, audio = setup()
    r = stream_native(audio, 16000, _mock_native(words, ends), chunk_s=0.1)
    assert r["words"] == words                                # no half word ("w1" for "w12") is ever committed
    lags = commit_lags(r["words"], r["commit_times"], words, ends)
    assert 0.3 <= np.median(lags) < 1.0                       # one word later than the 0.3 s look-ahead at most
    assert np.all(np.diff(r["commit_times"]) >= 0) and r["finish_delay_s"] >= 0


def test_names_from_before_19_nov_still_import():
    """Groups that imported these names before the day-3 update still can, after the pull."""
    from asr_demo.backends import SherpaTransducer, SherpaZipformer
    from speech.streaming_live import common_prefix

    assert SherpaZipformer is SherpaTransducer
    assert common_prefix(["the", "cat"], ["the", "car"]) == ["the"]


def test_load_system_wraps_hf_failure_as_backend_unavailable(monkeypatch):
    import transformers
    from asr_demo.backends import BackendUnavailable
    from speech.streaming_sim import load_system

    def fail_pipeline(*args, **kwargs):
        raise OSError("offline test")

    monkeypatch.setattr(transformers, "pipeline", fail_pipeline)
    with pytest.raises(BackendUnavailable):
        load_system("nonexistent/offline-model")
