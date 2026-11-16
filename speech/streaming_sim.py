"""Turn ANY offline recogniser into a simulated streaming one, and measure what real time costs.

Audio arrives in real time. Every `step_s` seconds (or as soon as the model is free, if it is slower
than that) the current buffer is re-transcribed. LocalAgreement-n (as in the whisper_streaming paper,
Machacek et al. 2023): a word is COMMITTED (shown, never retracted) once n consecutive hypotheses
agree on it. Metrics:
  * WER of the committed text           (vs offline WER of the same model on the full file)
  * commit lag per word = commit time - time the word ended in the audio (needs word end times,
    e.g. from the model's own offline word timestamps or a forced aligner)
  * falling behind: if compute per update > step_s, lag grows with time -> not real-time.

    python -m speech.streaming_sim --selftest            # mock recogniser, runs in seconds
    python -m speech.streaming_sim --model openai/whisper-base.en   # LibriSpeech (data/LibriSpeech), cuda if available
Limitation (on purpose, for discussion): when the buffer exceeds max_buffer_s it is flushed and
restarted, so a word cut at the boundary may be lost. whisper_streaming trims at timestamps instead.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np

from common.bench import default_device


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def simulate_streaming(audio: np.ndarray, sr: int, transcribe: Callable[[np.ndarray], str], step_s: float = 1.0,
                       n_agree: int = 2, max_buffer_s: float = 15.0,
                       cost_model: Callable[[float], float] | None = None) -> dict:
    """Simulate streaming recognition of `audio` (sampled at `sr`) with an offline recogniser.

    transcribe(buffer) -> text, where buffer = audio[buf_start * sr : t * sr] is the audio received since the last
    flush. If transcribe has an attribute `wants_offset = True` (the mocks of the tests do), call it as
    transcribe(buffer, buf_start) instead: buf_start, in seconds, is where the buffer starts in `audio`.
    cost_model(buffer_seconds) -> compute seconds, to simulate a model of a given speed;
    if None, the real wall-clock time of `transcribe` is used.
    Words are text.lower().split(). Times are seconds from the start of the audio.

    Returns a dict with exactly these keys:
      text            the committed words joined by spaces;
      words           the committed words (list of str), in order;
      commit_times    np.ndarray, one commit time per committed word, non-decreasing;
      updates         number of transcribe calls;
      mean_compute_s  mean compute time per call (seconds);
      behind          mean_compute_s > step_s (the model cannot keep up with real time);
      finish_delay_s  time of the last commit minus the audio duration (>= 0).
    """
    raise NotImplementedError("TODO(d3): real-time clock: re-transcribe the buffer every step_s or when the model is free; LocalAgreement-n commits (never retract); flush at max_buffer_s; return the documented dict")


def commit_lags(committed: list[str], commit_times: np.ndarray, ref_words: list[str], ref_ends: np.ndarray):
    """Align committed words to reference words; lag for each correctly recognised word."""
    import jiwer
    out = jiwer.process_words(" ".join(ref_words), " ".join(committed) or "<e>")
    lags = []
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                lags.append(commit_times[j] - ref_ends[i])
    return np.array(lags)


# --------------------------------------------------------------------------- #
def _mock(ref_words, ref_ends, sr, rng, unstable_s=0.6, err=0.02):
    """Mock recogniser: words that ended > unstable_s before the buffer end are right (except a
    small error rate); the last ones are unstable (random garbage), like a real model's tail."""
    starts = np.r_[0.0, ref_ends[:-1]]

    def transcribe(buf, offset=0.0):
        end = offset + len(buf) / sr
        out = []
        for w, st, e in zip(ref_words, starts, ref_ends):
            if st < offset - 0.05:  # word started before the buffer: cut off, not recognised
                continue
            if e > end:
                break
            if e > end - unstable_s:
                out.append(w if rng.random() < 0.5 else "xx" + str(rng.integers(9)))
            else:
                out.append(w if rng.random() > err else "yy")
        return " ".join(out)
    transcribe.wants_offset = True
    return transcribe


def selftest():
    import jiwer
    import pandas as pd
    rng = np.random.default_rng(0)
    sr = 16000
    rows = []
    for trial in range(5):
        n = 60
        ref_words = [f"w{i}" for i in range(n)]
        ref_ends = np.cumsum(rng.uniform(0.25, 0.6, n)) + 0.3
        audio = np.zeros(int((ref_ends[-1] + 0.5) * sr), np.float32)
        for model, cost in (("fast (RTF 0.05)", lambda d: 0.05 * d + 0.02), ("slow (RTF 0.5)", lambda d: 0.5 * d + 0.1)):
            for step in (0.25, 0.5, 1.0, 2.0):
                r = simulate_streaming(audio, sr, _mock(ref_words, ref_ends, sr, rng), step_s=step, cost_model=cost)
                lags = commit_lags(r["words"], r["commit_times"], ref_words, ref_ends)
                rows.append(dict(model=model, step_s=step, wer=jiwer.wer(" ".join(ref_words), r["text"]),
                                 lag_p50=np.median(lags), lag_p90=np.percentile(lags, 90),
                                 updates=r["updates"], behind=r["behind"]))
    df = pd.DataFrame(rows).groupby(["model", "step_s"]).mean(numeric_only=True).round(3)
    print(df.to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--librispeech", type=Path, default=Path("data/LibriSpeech"))
    ap.add_argument("--model", default="openai/whisper-base.en")
    ap.add_argument("--steps", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", type=Path, default=Path("results/streaming.csv"))
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    import pandas as pd
    from transformers import pipeline
    from speech.data import load_librispeech, normalise
    asr = pipeline("automatic-speech-recognition", model=args.model, device=args.device)
    # long-ish audio: concatenate utterances of one speaker into ~1 min streams
    data = load_librispeech(args.librispeech, n=args.n)
    rows = []
    for d in data:
        off = asr({"raw": d["audio"], "sampling_rate": d["sr"]}, return_timestamps="word")
        # word end times from the model's own timestamps; normalise each chunk, so that the words are tokenised
        # exactly as the streamed text below ("well-known" -> "well known": two words, same end time)
        ref_words, ends = [], []
        for c in off["chunks"]:
            for w in normalise(c["text"]).split():
                ref_words.append(w)
                ends.append(c["timestamp"][1] or 0.0)
        ends = np.array(ends)
        for step in args.steps:
            r = simulate_streaming(d["audio"], d["sr"], lambda b: asr({"raw": b, "sampling_rate": d["sr"]})["text"]
                                   if len(b) > 1600 else "", step_s=step)
            lags = commit_lags(normalise(r["text"]).split(), r["commit_times"], ref_words, ends)
            from speech.data import wer
            rows.append(dict(model=args.model, id=d["id"], step_s=step, wer_stream=wer([d["text"]], [r["text"]]),
                             wer_offline=wer([d["text"]], [off["text"]]), lag_p50=np.median(lags) if len(lags) else None,
                             lag_p90=np.percentile(lags, 90) if len(lags) else None, behind=r["behind"]))
            print(rows[-1])
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    from common.bench import BenchRecord, log_record
    for step, g in df.groupby("step_s"):
        log_record(BenchRecord("asr_streaming", args.model, dict(step_s=step),
                               dict(wer_stream=float(g.wer_stream.mean()), wer_offline=float(g.wer_offline.mean()),
                                    lag_p50=float(g.lag_p50.median()), lag_p90=float(g.lag_p90.median()),
                                    behind_frac=float(g.behind.mean()))), args.jsonl)


if __name__ == "__main__":
    main()
