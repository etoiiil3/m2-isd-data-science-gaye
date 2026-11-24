"""Day 3: your streaming policy, LIVE (stretch, not graded). Turns any offline recogniser into a streaming one that
fits the demo contract (asr_demo/contract.py): re-decode the growing buffer every `step_s` seconds of new audio and
commit words with LocalAgreement-n, the same `local_agreement` as speech/streaming_sim.py, driven by real chunks.
Committed words are never retracted; the partial shown = committed + tentative tail (lecture L6).
Unlike `simulate_streaming`, `LocalAgreementSession` processes chunks as they arrive without buffer flushing or
skipping updates, so a slow backend queues work and increases the wait after release.
With a record button there is no endpointing to do: release = finish().
"""
from __future__ import annotations

import numpy as np

from asr_demo.contract import Backend, OfflineBackend, Session
from speech.streaming_sim import _common_prefix as common_prefix  # noqa: F401  (kept for code written before 19 Nov)
from speech.streaming_sim import local_agreement


class LocalAgreementSession(Session):
    def __init__(self, inner: OfflineBackend, step_s: float, n_agree: int):
        self.inner, self.step, self.n = inner, step_s, n_agree
        self.buf: list[np.ndarray] = []
        self.new = 0                          # samples received since the last update
        self.committed: list[str] = []        # shown, never retracted
        self.history: list[list[str]] = []    # last hypotheses, without the committed words

    def accept_chunk(self, audio):
        """Buffer the chunk; every step_s seconds of NEW audio, re-transcribe the whole buffer and commit with
        LocalAgreement-n. Returns committed + tentative words as one string, or None when there was no update."""
        self.buf.append(np.asarray(audio, np.float32))
        self.new += len(audio)
        if self.new < self.step * self.inner.sample_rate:
            return None
        self.new = 0
        words = self.inner.transcribe(np.concatenate(self.buf)).split()
        tail = words[len(self.committed):]    # the words after the committed ones
        self.history = (self.history + [tail])[-self.n:]
        raise NotImplementedError("TODO(d3): agreed = local_agreement(self.history, self.n); if not empty: add it to self.committed and remove its words from the front of every hypothesis in self.history")
        return " ".join(self.committed + self.history[-1])

    def finish(self):
        """Release: transcribe the full buffer once more; the committed words stay, the rest of the final
        hypothesis (its words after len(self.committed)) is appended."""
        if not self.buf:
            return " ".join(self.committed)
        words = self.inner.transcribe(np.concatenate(self.buf)).split()
        raise NotImplementedError("TODO(d3): return the committed words followed by the words of the final hypothesis after the len(self.committed) first ones, joined by spaces")


class _Probe(OfflineBackend):
    def transcribe(self, audio):
        return "a"


class LocalAgreementBackend(Backend):
    def __init__(self, inner: OfflineBackend, step_s: float = 1.0, n_agree: int = 2):
        probe = LocalAgreementSession(_Probe(), step_s=0.0, n_agree=1)
        probe.accept_chunk(np.zeros(1, np.float32))  # fail on load if a TODO stub in this file is not implemented yet
        probe.finish()
        self.inner, self.step_s, self.n_agree = inner, step_s, n_agree
        self.sample_rate = inner.sample_rate
        self.description = f"LocalAgreement-{n_agree} every {step_s}s around {getattr(inner, 'name', 'model')}"

    def new_session(self):
        return LocalAgreementSession(self.inner, self.step_s, self.n_agree)
