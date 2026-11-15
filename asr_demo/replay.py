"""Run saved recordings through several backends, the way the browser would (100 ms chunks, then release).

    python -m asr_demo.replay --wavs results/demo_recordings --backends reference/zipformer-20M-int8 student/my-model
    python -m asr_demo.replay --wavs models/<reference-model>/test_wavs --trans trans.txt --backends ...

References: <name>.txt next to each wav, or a Kaldi-style file ("<name> TEXT") given with --trans.
Rows go to results/results.jsonl (task demo_replay): objective numbers next to what you heard live.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

from asr_demo import backends
from asr_demo.metrics import run_session_offline
from common.bench import BenchRecord, log_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", type=Path, required=True)
    ap.add_argument("--trans", default=None, help="file inside --wavs with '<name> <reference>' lines")
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--realtime", action="store_true", help="feed audio at speaking speed")
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    a = ap.parse_args()
    refs = {}
    if a.trans:
        for line in (a.wavs / a.trans).read_text(encoding="utf-8").splitlines():
            k, _, t = line.partition(" ")
            refs[k] = t
    rows = []
    for name in a.backends or [b["name"] for b in backends.available()]:
        try:
            b = backends.get(name)
        except Exception as e:  # missing model / package: report and continue
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        for w in sorted(a.wavs.glob("*.wav")):
            y, sr = sf.read(w, dtype="float32")
            ref = refs.get(w.stem) or ((w.with_suffix(".txt")).read_text(encoding="utf-8") if w.with_suffix(".txt").exists() else "")
            m = run_session_offline(b, y, sr, ref, a.realtime)
            m["recording"] = w.stem
            rows.append(m)
            log_record(BenchRecord("demo_replay", name, {"recording": w.stem, "realtime": a.realtime},
                                   {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}), a.jsonl)
    df = pd.DataFrame(rows)
    if len(df):
        cols = ["backend", "recording", "audio_s", "first_partial_ms", "final_latency_ms", "rtf", "wer", "text"]
        pd.set_option("display.max_colwidth", 60)
        print(df[cols].to_string(index=False))
        print("\nper backend (mean):")
        print(df.groupby("backend")[["final_latency_ms", "rtf", "wer"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
