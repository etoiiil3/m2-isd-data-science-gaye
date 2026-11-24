"""Metrics shared by the live server and the replay harness, so live and offline numbers are comparable.
  first_partial_ms : audio start -> first partial text (streaming feel)
  final_latency_ms : record button released -> final text (what the user waits for)
  rtf              : compute time / audio duration (must stay < 1 to keep up)
"""
from __future__ import annotations

import time

import numpy as np


def wer_or_none(reference: str, hyp: str):
    if not reference.strip():
        return None
    import jiwer
    from speech.data import normalise
    ref = normalise(reference)
    if not ref:
        return None
    return float(jiwer.wer(ref, normalise(hyp) or "<empty>"))


def summarise(name, streaming, y, sr, compute_s, first_partial_ms, final_latency_ms, n_partials, text, reference, load_ms=0.0):
    dur = len(y) / sr if len(y) else 0.0
    return {"backend": name, "streaming": bool(streaming), "text": text, "audio_s": round(dur, 3),
            "first_partial_ms": None if first_partial_ms is None else round(first_partial_ms, 1),
            "final_latency_ms": round(final_latency_ms, 1), "rtf": round(compute_s / dur, 4) if dur else None,
            "n_partials": n_partials, "wer": wer_or_none(reference, text), "load_ms": round(load_ms, 1)}


def run_session_offline(backend, y, sr, reference="", realtime=False, chunk_s=0.1, keep_partials=False):
    """Feed a recording through a backend exactly like the browser does (100 ms chunks, then finish).
    realtime=True sleeps so that audio arrives at the speed of speech (true first-partial latency).
    keep_partials=True adds "partials": [(t_s, text), ...], t_s = audio received when the partial came out
    (realtime: wall-clock seconds since the start)."""
    from asr_demo.contract import resample
    y = resample(y, sr, backend.sample_rate)
    sr = backend.sample_rate
    s = backend.new_session()
    n = int(chunk_s * sr)
    compute, first, nparts, partials = 0.0, None, 0, []
    t_free = 0.0
    t0 = time.perf_counter()
    for i in range(0, len(y), n):
        chunk_end = min(i + n, len(y)) / sr
        if realtime:
            time.sleep(max(0.0, t0 + chunk_end - time.perf_counter()))
        c = time.perf_counter()
        txt = s.accept_chunk(y[i:i + n])
        dt = time.perf_counter() - c
        compute += dt
        t_free = max(chunk_end, t_free) + dt
        if txt:
            nparts += 1
            now_ms = (time.perf_counter() - t0) * 1000 if realtime else chunk_end * 1000
            first = first or now_ms
            partials.append((round(now_ms / 1000, 3), txt))
    c = time.perf_counter()
    final = s.finish()
    done = time.perf_counter()
    fin = done - c
    dur = len(y) / sr if len(y) else 0.0
    t_free = max(dur, t_free) + fin
    wait_s = max(fin, done - (t0 + dur)) if realtime else (t_free - dur)
    m = summarise(backend.name, backend.streaming, y, sr, compute + fin, first, wait_s * 1000, nparts, final, reference)
    if keep_partials:
        m["partials"] = partials
    return m
