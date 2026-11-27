"""Limits study (final): how does our deployed system hold up when the audio gets noisy?

Controlled: the same 20 LibriSpeech test-clean utterances as speech.streaming_sim, with white noise added
at a fixed signal-to-noise ratio (speech/degrade.py, same noise for every system). Only the SNR changes.
Systems: our deployed system (wav2vec2-base-960h + our LocalAgreement-2, step 0.5 s, decision D5) and the
provided reference streaming transducer (the bar to beat).

    DSP_MILESTONE=final python -m speech.limits_noise

Writes results/limits_noise.jsonl (one row per system x SNR), results/limits_noise.csv and
results/figures/limits_noise.png. About 5 minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common.bench import BenchRecord, log_record
from speech.data import load_librispeech
from speech.degrade import degrade
from speech.streaming_sim import reference_word_ends, run_system

SYSTEMS = ["facebook/wav2vec2-base-960h", "reference/fastconformer-114M-int8"]
LABELS = {"facebook/wav2vec2-base-960h": "ours: wav2vec2 + LocalAgreement (0.5 s)",
          "reference/fastconformer-114M-int8": "reference transducer"}


def add_noise(data: list[dict], snr: str) -> list[dict]:
    """A copy of `data` with white noise at `snr` dB (fixed seed per utterance), or `data` itself for 'clean'."""
    if snr == "clean":
        return data
    return [dict(d, audio=degrade(d["audio"], d["sr"], "noise", snr_db=float(snr),
                                  rng=np.random.default_rng(k)).astype(np.float32))
            for k, d in enumerate(data)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--librispeech", type=Path, default=Path("data/LibriSpeech"))
    ap.add_argument("--n", type=int, default=20, help="LibriSpeech test-clean utterances (same as streaming_sim)")
    ap.add_argument("--snr", nargs="+", default=["clean", "20", "10", "5", "0"], help="'clean' and/or SNRs in dB")
    ap.add_argument("--models", nargs="+", default=SYSTEMS)
    ap.add_argument("--step", type=float, default=0.5, help="LocalAgreement step of the re-decoded systems (s)")
    ap.add_argument("--jsonl", type=Path, default=Path("results/limits_noise.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/limits_noise.csv"))
    ap.add_argument("--fig", type=Path, default=Path("results/figures/limits_noise.png"))
    args = ap.parse_args()

    import pandas as pd
    data = load_librispeech(args.librispeech, n=args.n)
    print(f"{len(data)} utterances, {sum(len(d['audio']) / d['sr'] for d in data):.0f} s of audio")
    # word end times come from the CLEAN audio: additive noise does not move the words in time
    ends = reference_word_ends(data, cache=Path("results/streaming_word_ends.json"))

    rows = []
    for snr in args.snr:
        noisy = add_noise(data, snr)
        for name in args.models:
            t0 = time.perf_counter()
            summary, _ = run_system(name, noisy, ends, steps=(args.step,), device="cpu", verbose=False)
            for r in summary:
                row = dict(snr=snr, **r)
                rows.append(row)
                log_record(BenchRecord("limits_noise", name,
                                       dict(snr_db=snr, policy=r["policy"], step_s=r["step_s"], n_utts=r["n_utts"]),
                                       {k: r[k] for k in ("wer_stream", "wer_offline", "lag_p50", "lag_p90",
                                                          "finish_p50", "rtf", "behind_frac")}), args.jsonl)
                print(f"SNR {snr:>5}  {name:<36} WER {r['wer_stream']:.3f}  wait {r['finish_p50'] * 1000:.0f} ms"
                      f"  ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print()
    print(df.pivot(index="snr", columns="model", values="wer_stream").reindex(args.snr).round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(args.snr))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in df.groupby("model", sort=False):
        g = g.set_index("snr").reindex(args.snr)
        ax.plot(x, 100 * g["wer_stream"], marker="o", label=LABELS.get(name, name))
    ax.axhline(5, color="grey", ls="--", lw=1, label="our WER budget (5%)")
    ax.set_xticks(x, [s if s == "clean" else f"{s} dB" for s in args.snr])
    ax.set_xlabel("white noise, signal-to-noise ratio")
    ax.set_ylabel("WER of the committed text (%)")
    ax.set_title(f"Limits study: white noise ({len(data)} utterances)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=130)
    print(f"\n{args.out}, {args.jsonl}, {args.fig}")


if __name__ == "__main__":
    main()