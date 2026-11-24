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


def test_a_backend_built_on_another_loads_without_deadlock():
    """A factory may get() its inner backend (LocalAgreement around a model): the registry lock is re-entrant."""
    import threading

    from asr_demo import backends
    backends.register("test/inner", "")(MockModel)
    backends.register("test/outer", "")(lambda: (backends.get("test/inner"), MockModel())[1])
    out = {}
    t = threading.Thread(target=lambda: out.update(b=backends.get("test/outer")), daemon=True)
    t.start()
    t.join(timeout=10)
    try:
        assert not t.is_alive(), "backends.get deadlocked on a backend built on another one"
        assert isinstance(out["b"], MockModel) and isinstance(backends.get("test/inner"), MockModel)
    finally:
        for name in ("test/inner", "test/outer"):
            backends.REGISTRY.pop(name, None)
            backends._CACHE.pop(name, None)


def test_register_evicts_cache_and_wrappers():
    from asr_demo import backends
    from asr_demo.contract import Backend

    class V1(MockModel):
        tag = "v1"

    class V2(MockModel):
        tag = "v2"

    class Wrapper(Backend):
        def __init__(self, inner):
            self.inner = inner

        def new_session(self):
            return self.inner.new_session()

    try:
        backends.register("test/inner_reload", "")(V1)
        backends.register("test/outer_reload", "")(lambda: Wrapper(backends.get("test/inner_reload")))
        assert backends.get("test/outer_reload").inner.tag == "v1"
        backends.register("test/inner_reload", "")(V2)
        assert backends.get("test/inner_reload").tag == "v2"
        assert backends.get("test/outer_reload").inner.tag == "v2"
    finally:
        for name in ("test/inner_reload", "test/outer_reload"):
            backends.REGISTRY.pop(name, None)
            backends._CACHE.pop(name, None)
            backends._FAILED.pop(name, None)


def test_realtime_replay_respects_chunk_arrival_and_backlog():
    import time
    from asr_demo.contract import Backend, Session
    from asr_demo.metrics import run_session_offline, wer_or_none

    class SlowChunkSession(Session):
        def __init__(self, sleep_s):
            self.sleep_s = sleep_s
            self.chunks = 0

        def accept_chunk(self, audio):
            if self.sleep_s:
                time.sleep(self.sleep_s)
            self.chunks += 1
            return "hi" if self.chunks == 1 else None

        def finish(self):
            time.sleep(0.01)
            return "hi"

    class SlowChunkBackend(Backend):
        def __init__(self, sleep_s):
            self.name = "test/slow"
            self.sleep_s = sleep_s

        def new_session(self):
            return SlowChunkSession(self.sleep_s)

    y = np.zeros(int(16000 * 0.3), np.float32)
    rt_fast = run_session_offline(SlowChunkBackend(0.0), y, 16000, realtime=True, chunk_s=0.1)
    assert rt_fast["first_partial_ms"] >= 95.0
    off_slow = run_session_offline(SlowChunkBackend(0.16), y, 16000, realtime=False, chunk_s=0.1)
    assert off_slow["final_latency_ms"] >= 140.0
    rt_slow = run_session_offline(SlowChunkBackend(0.16), y, 16000, realtime=True, chunk_s=0.1)
    assert rt_slow["final_latency_ms"] >= 140.0
    assert wer_or_none("...", "hi") is None


def test_live_ws_wait_includes_queued_audio(tmp_path, monkeypatch):
    import asyncio
    import json
    import time
    from asr_demo import backends, server
    from asr_demo.contract import Backend, Session

    class SlowLiveSession(Session):
        def accept_chunk(self, audio):
            time.sleep(0.16)
            return "hi"

        def finish(self):
            time.sleep(0.01)
            return "hi"

    class SlowLiveBackend(Backend):
        def new_session(self):
            return SlowLiveSession()

    class FakeWebSocket:
        def __init__(self, msgs):
            self._msgs = msgs
            self.sent = []

        async def accept(self):
            pass

        async def receive_text(self):
            return json.dumps({"type": "start", "backend": "test/slow_live"})

        async def receive(self):
            kind, delay, data = await self._msgs.get()
            if delay:
                await asyncio.sleep(delay)
            if kind == "bytes":
                return {"type": "websocket.receive", "bytes": data}
            return {"type": "websocket.receive", "text": json.dumps(data)}

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self):
            pass

    monkeypatch.setattr(server, "REC", tmp_path / "rec")
    monkeypatch.setattr(server, "LOG", tmp_path / "results.jsonl")
    backends.register("test/slow_live", "")(SlowLiveBackend)
    try:
        async def run():
            q = asyncio.Queue()
            chunk = np.zeros(1600, np.float32).tobytes()
            await q.put(("bytes", 0.0, chunk))
            await q.put(("bytes", 0.0, chunk))
            await q.put(("bytes", 0.0, chunk))
            await q.put(("text", 0.04, {"type": "stop"}))
            ws = FakeWebSocket(q)
            await server.ws_endpoint(ws)
            return ws.sent

        sent = asyncio.run(run())
    finally:
        backends.REGISTRY.pop("test/slow_live", None)
        backends._CACHE.pop("test/slow_live", None)
        backends._FAILED.pop("test/slow_live", None)
    finals = [m for m in sent if m.get("type") == "final"]
    assert len(finals) == 1
    assert finals[0]["final_latency_ms"] >= 200.0


def test_replay_skips_digits_by_default_and_matches_wav_keys(tmp_path, monkeypatch):
    import soundfile as sf
    from asr_demo import backends, replay as replay_mod

    wav = tmp_path / "0.wav"
    sf.write(wav, np.zeros(3200, np.float32), 16000)
    (tmp_path / "trans.txt").write_text("0.wav w0 w1 w2 w3 w4\n", encoding="utf-8")

    backends.register("test/mock_replay", "")(MockModel)
    monkeypatch.setattr(
        backends,
        "available",
        lambda: [
            {"name": "digits/fake", "description": "", "loaded": False, "error": None, "has_prompt": False},
            {"name": "test/mock_replay", "description": "", "loaded": False, "error": None, "has_prompt": False},
        ],
    )
    try:
        df = replay_mod.replay(tmp_path, trans="trans.txt", jsonl=None)
    finally:
        backends.REGISTRY.pop("test/mock_replay", None)
        backends._CACHE.pop("test/mock_replay", None)
        backends._FAILED.pop("test/mock_replay", None)
    assert list(df["backend"]) == ["test/mock_replay"]
    assert df["wer"].notna().all()
