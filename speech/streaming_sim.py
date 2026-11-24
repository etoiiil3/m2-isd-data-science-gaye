"""Day 3: streaming recognition on a simulated real-time clock, and what it costs (lecture L5).

Two ways to show text while the user is still speaking:
  * re-decode and agree: an OFFLINE recogniser re-transcribes the growing buffer every `step_s` seconds, and
    LocalAgreement-n (Machacek et al., 2023) commits a word once the last n hypotheses agree on it;
  * native streaming: a streaming model (the demo's transducers) extends its hypothesis chunk after chunk, and a
    word is committed once the model has started the next one.
Both run on the same clock: audio arrives in real time, the model is busy for the compute time of each call
(measured, or given by a cost model), and every committed word gets a commit time. Per utterance:
  * WER of the committed text            (vs the offline WER of the same model on the whole file)
  * commit lag of each correct word = commit time - time the word ends in the audio. The end times come from a
    CTC forced alignment of the reference transcript (`reference_word_ends`), the same for every model;
  * finish delay = end of the final call (t_free) - end of the audio: the wait after the user releases the button;
  * behind: compute per update > step_s, the model cannot keep up (updates are skipped, lag is set by compute or grows with the buffer).

    python -m speech.streaming_sim --selftest      # mock recogniser, a few seconds (lecture L5's table)
    python -m speech.streaming_sim                 # the graded run (d3): 3 systems x 20 utterances, about 6 min
    python -m speech.streaming_sim --models openai/whisper-base.en --steps 1 2      # run extra system(s) and append their rows to results.jsonl (replaces defaults for this call)
    python -m speech.streaming_sim --models student/ctc+3gram   # any backend registered in my_backends.py

Limitation (on purpose, for discussion): when the buffer exceeds max_buffer_s it is flushed and restarted, so a
word cut at the boundary may be lost. whisper_streaming trims the buffer at word timestamps instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from common.bench import default_device

ALIGNER = "facebook/wav2vec2-base-960h"   # CTC model used for the forced alignment of the references
DEFAULT_SYSTEMS = ["openai/whisper-tiny.en", "facebook/wav2vec2-base-960h", "reference/fastconformer-114M-int8"]


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def local_agreement(history: list[list[str]], n_agree: int) -> list[str]:
    """LocalAgreement-n: the words to commit now.

    history: the last hypotheses, oldest first; each is a list of words, without the words already committed.
    Returns the longest common prefix of the last n_agree hypotheses, or [] while there are fewer than n_agree.
    Example: history [["the", "cat", "sad"], ["the", "cat", "sat", "on"]], n_agree=2 -> ["the", "cat"].
    """
    if len(history) < n_agree:
        return []
    last = history[len(history) - n_agree:]   # the n_agree most recent hypotheses
    raise NotImplementedError("TODO(d3): agreed = the longest common prefix of the hypotheses in last (start from last[0], fold _common_prefix over the others); return it")


def simulate_streaming(audio: np.ndarray, sr: int, transcribe: Callable[[np.ndarray], str], step_s: float = 1.0,
                       n_agree: int = 2, max_buffer_s: float = 15.0,
                       cost_model: Callable[[float], float] | None = None) -> dict:
    """Simulate streaming recognition of `audio` (sampled at `sr`) with an OFFLINE recogniser, re-decoding the
    buffer every step_s seconds and committing words with LocalAgreement-n.

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
      finish_delay_s  end of the final call (t_free) minus the audio duration (>= 0).
    """
    dur = len(audio) / sr
    t, t_free, buf_start = step_s, 0.0, 0.0   # audio received so far, time the model becomes free, buffer start (s)
    committed, commit_times = [], []
    in_buf_committed = 0                      # committed words that come from the current buffer
    history: list[list[str]] = []             # last hypotheses, without their committed words
    compute = []
    while True:
        t = min(t, dur)
        start = max(t, t_free)                # an update starts when its audio is there AND the model is free
        buf = audio[int(buf_start * sr):int(t * sr)]
        w0 = time.perf_counter()
        # mocks/oracles may need the absolute position of the buffer; real models only see audio
        text = transcribe(buf, buf_start) if getattr(transcribe, "wants_offset", False) else transcribe(buf)
        c = cost_model(len(buf) / sr) if cost_model else time.perf_counter() - w0
        compute.append(c)
        raise NotImplementedError("TODO(d3): the clock: this update ends (and its words are shown) at t_free = start + c")
        tail = text.lower().split()[in_buf_committed:]
        history = (history + [tail])[-n_agree:]
        final = t >= dur
        new = tail if final else local_agreement(history, n_agree)  # end of the audio: commit everything
        if new:                               # committed: shown at t_free, never retracted
            committed.extend(new)
            commit_times.extend([t_free] * len(new))
            in_buf_committed += len(new)
            history = [h[len(new):] for h in history]
        if final:
            break
        if t - buf_start > max_buffer_s:      # flush: commit what we have, restart the buffer
            rest = tail[len(new):]
            committed.extend(rest)
            commit_times.extend([t_free] * len(rest))
            buf_start, in_buf_committed, history = t, 0, []
        raise NotImplementedError("TODO(d3): the next update: when step_s more seconds of audio have arrived AND the model is free: t = the later of t + step_s and t_free")
    return dict(text=" ".join(committed), words=committed, commit_times=np.array(commit_times),
                updates=len(compute), mean_compute_s=float(np.mean(compute)), behind=bool(np.mean(compute) > step_s),
                finish_delay_s=t_free - dur)


def stream_native(audio: np.ndarray, sr: int, backend, chunk_s: float = 0.1) -> dict:
    """Simulate a NATIVE streaming recogniser on the same clock as simulate_streaming; same returned dict.

    backend follows the demo contract (asr_demo/contract.py): session = backend.new_session(), then
    session.accept_chunk(chunk) -> current hypothesis or None (unchanged), and session.finish() -> final text.
    Chunk i (chunk_s seconds) has arrived at t = (i + 1) * chunk_s; it is processed when it has arrived and the
    previous chunk is done (measured wall-clock compute). A streaming transducer decoded greedily only extends its
    hypothesis, but its last word may still grow ("the" -> "there"): a word is committed once a next word exists.
    finish() commits the rest. `updates` counts the calls (chunks + finish); `behind` = mean compute > chunk_s.
    """
    from asr_demo.contract import resample
    y = resample(np.asarray(audio, np.float32), sr, backend.sample_rate)
    sr = backend.sample_rate
    dur = len(y) / sr
    n = max(1, int(round(chunk_s * sr)))
    session = backend.new_session()
    t_free, committed, commit_times, compute = 0.0, [], [], []
    for i in range(0, len(y), n):
        start = max(min(i + n, len(y)) / sr, t_free)
        w0 = time.perf_counter()
        text = session.accept_chunk(y[i:i + n])
        compute.append(time.perf_counter() - w0)
        t_free = start + compute[-1]
        if text is None:
            continue
        words = text.lower().split()
        raise NotImplementedError("TODO(d3): the stable words: every word of words except the last one (it may still grow), minus the len(committed) words already committed: new = words[:-1][len(committed):]")
        committed.extend(new)
        commit_times.extend([t_free] * len(new))
    start = max(dur, t_free)                  # release: the final call starts when the last chunk is done
    w0 = time.perf_counter()
    final = session.finish().lower().split()
    compute.append(time.perf_counter() - w0)
    t_free = start + compute[-1]
    rest = final[len(committed):]
    committed.extend(rest)
    commit_times.extend([t_free] * len(rest))
    return dict(text=" ".join(committed), words=committed, commit_times=np.array(commit_times),
                updates=len(compute), mean_compute_s=float(np.mean(compute)), behind=bool(np.mean(compute) > chunk_s),
                finish_delay_s=t_free - dur)


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


def normalise_committed(words: list[str], times) -> tuple[list[str], np.ndarray]:
    """Normalise committed words as the WER does (speech.data.normalise), keeping one commit time per word:
    "Hello," -> "hello"; "well-known" -> "well", "known" (same time); a lone "--" disappears."""
    from speech.data import normalise
    out_w, out_t = [], []
    for w, c in zip(words, times):
        for piece in normalise(w).split():
            out_w.append(piece)
            out_t.append(c)
    return out_w, np.array(out_t)


# --------------------------------------------------------------------------- reference word end times
def ctc_forced_align(logprobs: np.ndarray, targets: list[int], blank: int) -> np.ndarray:
    """Best CTC alignment of a KNOWN token sequence: lecture L3's lattice (y' = blank y_1 blank ... y_U blank), with
    max instead of sum (Viterbi, as in L2). logprobs: (T, K) log P_t(k); targets: the U token ids of y.
    Returns, for each target token, the last frame it occupies on the best path (shape (U,))."""
    T, U = len(logprobs), len(targets)
    ext = np.full(2 * U + 1, blank)
    ext[1::2] = targets
    S = len(ext)
    skip = np.zeros(S, bool)                     # a token state may come from 2 states back, unless the token repeats
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    delta = np.full(S, -np.inf)
    delta[0], delta[1] = logprobs[0, ext[0]], logprobs[0, ext[1]]
    back = np.zeros((T, S), np.int8)             # 0: stay, 1: from s-1, 2: from s-2
    for t in range(1, T):
        cand = np.stack([delta, np.r_[-np.inf, delta[:-1]], np.where(skip, np.r_[-np.inf, -np.inf, delta[:-2]], -np.inf)])
        back[t] = cand.argmax(0)
        delta = cand.max(0) + logprobs[t, ext]
    s = S - 1 if delta[S - 1] >= delta[S - 2] else S - 2
    last = np.full(U, -1)
    for t in range(T - 1, -1, -1):
        if s % 2 == 1 and last[(s - 1) // 2] < 0:
            last[(s - 1) // 2] = t
        s -= int(back[t, s])
    return last


def reference_word_ends(data: list[dict], device: str = "cpu", cache: Path | None = None) -> dict:
    """For each utterance of `data` (speech.data.load_librispeech): its reference words, normalised as for the WER,
    and the time (s) each one ends in the audio, from a CTC forced alignment of the reference transcript with
    wav2vec2-base-960h (20 ms frames). The same end times serve every model, so commit lags are comparable.
    Returns {id: (words, ends)}; cached as JSON in `cache` (computed once, about 10 s for 20 utterances)."""
    from speech.data import normalise
    stored = json.loads(cache.read_text()) if cache and cache.exists() else {}
    todo = [d for d in data if d["id"] not in stored]
    if todo:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        proc = AutoProcessor.from_pretrained(ALIGNER)
        model = Wav2Vec2ForCTC.from_pretrained(ALIGNER).to(device).eval()
        vocab = proc.tokenizer.get_vocab()
        frame_s = 320 / 16000                    # wav2vec2: one frame every 320 samples
        for d in todo:
            words = d["text"].upper().split()
            ids, word_of = [], []
            for i, w in enumerate(words):
                if i:
                    ids.append(vocab["|"])
                    word_of.append(-1)
                for ch in w:
                    ids.append(vocab.get(ch, vocab["<unk>"]))
                    word_of.append(i)
            with torch.inference_mode():
                x = proc(d["audio"], sampling_rate=d["sr"], return_tensors="pt").input_values.to(device)
                lp = torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy()
            last = ctc_forced_align(lp, ids, proc.tokenizer.pad_token_id)
            ends = np.zeros(len(words))
            for j, i in enumerate(word_of):
                if i >= 0:
                    ends[i] = (last[j] + 1) * frame_s
            nw, ne = [], []
            for w, e in zip(words, ends):
                for piece in normalise(w).split():
                    nw.append(piece)
                    ne.append(round(float(e), 3))
            stored[d["id"]] = {"words": nw, "ends": ne}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f"{cache.suffix}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(stored), encoding="utf-8")
            os.replace(tmp, cache)
    return {d["id"]: (stored[d["id"]]["words"], np.array(stored[d["id"]]["ends"])) for d in data}


# --------------------------------------------------------------------------- mocks (tests, self-test, slides)
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


def _mock_native(ref_words, ref_ends, look_ahead_s=0.3, sr=16000):
    """Mock native streaming backend: a word appears look_ahead_s after it ends, but first as a growing prefix
    ("w12" is shown as "w", "w1", then "w12"), like a transducer that emits sub-word tokens."""
    from asr_demo.contract import Backend, Session

    class _Session(Session):
        def __init__(self):
            self.n, self.last = 0, None

        def _text(self, now):
            out = [w for w, e in zip(ref_words, ref_ends) if e + look_ahead_s <= now]
            nxt = len(out)
            if nxt < len(ref_words) and ref_ends[nxt] - 0.2 <= now:   # the next word is being emitted, piece by piece
                frac = min(1.0, (now - ref_ends[nxt] + 0.2) / (0.2 + look_ahead_s))
                out.append(ref_words[nxt][:max(1, int(len(ref_words[nxt]) * frac))])
            return " ".join(out)

        def accept_chunk(self, audio):
            self.n += len(audio)
            t = self._text(self.n / sr)
            if t != self.last:
                self.last = t
                return t
            return None

        def finish(self):
            return " ".join(ref_words[:sum(e <= self.n / sr + 1e-9 for e in ref_ends)])

    class _Backend(Backend):
        sample_rate = sr

        def new_session(self):
            return _Session()
    return _Backend()


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


# --------------------------------------------------------------------------- the graded run
def load_system(name: str, device: str = "cpu"):
    """A system to stream, by name: a backend of the demo (asr_demo/backends.py, e.g.
    reference/fastconformer-114M-int8, or one registered in my_backends.py) or a Hugging Face model id
    (openai/whisper-tiny.en). Returns ("native", backend) for a native streaming backend, ("redecode", transcribe)
    for an offline one (OfflineBackend, LocalAgreementBackend unwrapped to its .inner OfflineBackend so steps can be
    swept, or Hugging Face pipeline). Raises asr_demo.backends.BackendUnavailable."""
    from asr_demo import backends
    from asr_demo.contract import OfflineBackend
    if name in backends.REGISTRY:
        b = backends.get(name)
        if hasattr(b, "inner") and isinstance(b.inner, OfflineBackend):
            b = b.inner
        return ("redecode", b.transcribe) if isinstance(b, OfflineBackend) else ("native", b)
    from transformers import pipeline
    try:
        asr = pipeline("automatic-speech-recognition", model=name, device=device)
    except Exception as e:
        raise backends.BackendUnavailable(f"{name}: {backends.explain(e)}") from e
    return "redecode", lambda buf: asr({"raw": buf, "sampling_rate": 16000})["text"] if len(buf) >= 1600 else ""


def run_system(name: str, data: list[dict], ends: dict, steps=(0.5, 1.0, 2.0), chunk_s: float = 0.1,
               n_agree: int = 2, device: str = "cpu", verbose: bool = True):
    """Stream every utterance of `data` through one system; returns (summary rows, per-utterance rows).
    Re-decoded systems: one configuration per step (LocalAgreement-n_agree); native ones: one (chunk_s).
    Summary metrics: wer_stream, wer_offline (corpus WER), lag_p50 / lag_p90 over all correct words (s),
    finish_p50 (median finish delay: the wait after release, s), rtf (compute / audio), behind_frac."""
    from asr_demo import backends
    from speech.data import wer
    kind, system = load_system(name, device)
    actual_device = "cpu" if name in backends.REGISTRY else device
    configs = [("native", chunk_s)] if kind == "native" else [(f"local-agreement-{n_agree}", s) for s in steps]
    per_utt = []
    for k, d in enumerate(data):
        audio, sr = d["audio"], d["sr"]
        if kind == "native":                  # offline = the whole file in one chunk
            s = system.new_session()
            s.accept_chunk(audio)
            offline = s.finish()
        else:
            offline = system(audio)
        ref_words, ref_ends = ends[d["id"]]
        for policy, step in configs:
            if kind == "native":
                r = stream_native(audio, sr, system, chunk_s=step)
            else:
                r = simulate_streaming(audio, sr, system, step_s=step, n_agree=n_agree)
            words, times = normalise_committed(r["words"], r["commit_times"])
            per_utt.append(dict(model=name, policy=policy, step_s=step, id=d["id"], audio_s=len(audio) / sr,
                                ref=d["text"], hyp=r["text"], offline=offline,
                                lags=commit_lags(words, times, ref_words, ref_ends).tolist(),
                                finish_delay_s=r["finish_delay_s"], compute_s=r["mean_compute_s"] * r["updates"],
                                behind=r["behind"]))
        if verbose:
            print(f"  {name}: {k + 1}/{len(data)} utterances", end="\r", flush=True)
    rows = []
    for (policy, step), g in _group(per_utt, ("policy", "step_s")):
        lags = np.concatenate([np.asarray(u["lags"], float) for u in g]) if g else np.zeros(0)
        rows.append(dict(model=name, policy=policy, step_s=step, n_utts=len(g),
                         wer_stream=wer([u["ref"] for u in g], [u["hyp"] for u in g]),
                         wer_offline=wer([u["ref"] for u in g], [u["offline"] for u in g]),
                         lag_p50=float(np.median(lags)) if len(lags) else None,
                         lag_p90=float(np.percentile(lags, 90)) if len(lags) else None,
                         finish_p50=float(np.median([u["finish_delay_s"] for u in g])),
                         rtf=sum(u["compute_s"] for u in g) / sum(u["audio_s"] for u in g),
                         behind_frac=float(np.mean([u["behind"] for u in g])),
                         device=actual_device))
    if verbose:
        print()
    return rows, per_utt


def _group(items: list[dict], keys: tuple) -> list:
    groups: dict = {}
    for it in items:
        groups.setdefault(tuple(it[k] for k in keys), []).append(it)
    return list(groups.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--librispeech", type=Path, default=Path("data/LibriSpeech"))
    ap.add_argument("--models", nargs="+", default=DEFAULT_SYSTEMS,
                    help="Hugging Face ids (re-decoded) and/or demo backends (replaces the 3 defaults for this run; run without --models first, then append extra systems)")
    ap.add_argument("--model", dest="models", nargs="+", help=argparse.SUPPRESS)   # name used before 19 Nov
    ap.add_argument("--steps", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="re-decoding steps (s)")
    ap.add_argument("--chunk", type=float, default=0.1, help="chunk of a native streaming model (s), as in the demo")
    ap.add_argument("--n-agree", type=int, default=2)
    ap.add_argument("--n", type=int, default=20, help="LibriSpeech test-clean utterances (the same for every system)")
    ap.add_argument("--device", default="cpu", help="cpu by default: the demo machine is the laptop "
                    f"(this machine's accelerator: {default_device()})")
    ap.add_argument("--out", type=Path, default=Path("results/streaming.csv"))
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    import pandas as pd
    from asr_demo.backends import BackendUnavailable
    from common.bench import BenchRecord, log_record
    from speech.data import load_librispeech
    data = load_librispeech(args.librispeech, n=args.n)
    print(f"{len(data)} utterances, {sum(len(d['audio']) / d['sr'] for d in data):.0f} s of audio; aligning the references")
    ends = reference_word_ends(data, cache=args.out.parent / "streaming_word_ends.json")
    summary, per_utt, skipped = [], [], []
    for name in args.models:
        t0 = time.perf_counter()
        try:
            rows, utts = run_system(name, data, ends, args.steps, args.chunk, args.n_agree, args.device)
        except BackendUnavailable as e:   # a missing model or package: say so, run the others (a TODO is raised)
            print(f"[skip] {name}: {e}")
            skipped.append(name)
            continue
        summary += rows
        per_utt += utts
        for r in rows:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
            log_record(BenchRecord("asr_streaming", name, dict(policy=r["policy"], step_s=r["step_s"], n_agree=args.n_agree,
                                                               n_utts=r["n_utts"], device=r.get("device", args.device)),
                                   {k: r[k] for k in ("wer_stream", "wer_offline", "lag_p50", "lag_p90", "finish_p50",
                                                      "rtf", "behind_frac")}), args.jsonl)
        print(f"  {name}: {time.perf_counter() - t0:.0f} s")
    if not summary:
        print("no system ran")
        sys.exit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_utt).to_csv(args.out, index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    if skipped:
        print(f"\nWARNING: skipped {', '.join(skipped)} (see [skip] above). Once fixed, add the missing rows with:\n"
              f"  python -m speech.streaming_sim --models {' '.join(skipped)}")


if __name__ == "__main__":
    main()
