"""Run saved recordings through several backends, the way the browser would (100 ms chunks, then release).

    python -m asr_demo.replay --wavs results/demo_recordings --backends reference/fastconformer-114M-int8 student/my-model
    python -m asr_demo.replay --librispeech data/LibriSpeech --n 10 --backends ...   # no microphone needed
    python -m asr_demo.replay --wavs models/<reference-model>/test_wavs --trans trans.txt --backends ...

References: <name>.txt next to each wav, or a Kaldi-style file ("<name> TEXT") given with --trans.
--librispeech writes n test-clean utterances (wav + txt) to results/librispeech_wavs/ once, then replays them.
Rows go to results/results.jsonl (task demo_replay): objective numbers next to what you heard live.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from asr_demo import backends
from asr_demo.metrics import run_session_offline
from common.bench import BenchRecord, log_record


def export_librispeech(root: Path, folder: Path, n: int = 10, max_s: float = 20.0) -> Path:
    """Write n LibriSpeech test-clean utterances (the same ones as the benchmarks: speech.data.load_librispeech)
    to folder/<id>.wav with their reference in folder/<id>.txt. Existing files are kept."""
    from speech.data import load_librispeech
    folder.mkdir(parents=True, exist_ok=True)
    items = load_librispeech(root, n=n, max_s=max_s)
    for d in items:
        if not (folder / f"{d['id']}.wav").exists():
            sf.write(folder / f"{d['id']}.wav", d["audio"], d["sr"])
            (folder / f"{d['id']}.txt").write_text(d["text"].lower(), encoding="utf-8")
    (folder / ".manifest.json").write_text(json.dumps([d["id"] for d in items]), encoding="utf-8")
    return folder


def replay(wavs: Path, names: list[str] | None = None, trans: str | None = None, realtime: bool = False,
           jsonl: Path | None = Path("results/results.jsonl"), limit: int | None = None) -> pd.DataFrame:
    """Every wav of `wavs` (manifest order or sorted, at most `limit`) through every backend of `names`
    (default: all registered speech backends, excluding digits/*).
    One row per (backend, recording): the metrics of asr_demo/metrics.py plus "recording". A backend that cannot
    load is reported ([skip]) and left out. Rows are also logged to `jsonl` (task demo_replay) unless it is None."""
    refs = {}
    if trans:
        for line in (wavs / trans).read_text(encoding="utf-8").splitlines():
            k, _, t = line.partition(" ")
            refs[k] = t
    manifest = wavs / ".manifest.json"
    if manifest.exists() and trans is None:
        ids = json.loads(manifest.read_text(encoding="utf-8"))
        files = [wavs / f"{i}.wav" for i in ids if (wavs / f"{i}.wav").exists()][:limit]
    else:
        files = []
    if not files:
        files = sorted(wavs.glob("*.wav"))[:limit]
    rows = []
    default_names = [b["name"] for b in backends.available() if not b["name"].startswith("digits/")]
    for name in names or default_names:
        try:
            b = backends.get(name)
        except backends.BackendUnavailable as e:  # missing model / package / TODO: report and continue
            print(f"[skip] {e}")
            continue
        for w in files:
            y, sr = sf.read(w, dtype="float32")
            ref = refs.get(w.name) or refs.get(w.stem) or (w.with_suffix(".txt").read_text(encoding="utf-8") if w.with_suffix(".txt").exists() else "")
            m = run_session_offline(b, y, sr, ref, realtime)
            m["recording"] = w.stem
            rows.append(m)
            if jsonl is not None:
                log_record(BenchRecord("demo_replay", name, {"recording": w.stem, "realtime": realtime},
                                       {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}), jsonl)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", type=Path, default=None, help="folder of recordings (default: results/demo_recordings)")
    ap.add_argument("--librispeech", type=Path, default=None, help="replay LibriSpeech test-clean utterances instead")
    ap.add_argument("--n", type=int, default=10, help="with --librispeech: how many utterances")
    ap.add_argument("--trans", default=None, help="file inside --wavs with '<name> <reference>' lines")
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--realtime", action="store_true", help="feed audio at speaking speed")
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    a = ap.parse_args()
    wavs = a.wavs or Path("results/demo_recordings")
    if a.librispeech:
        wavs = export_librispeech(a.librispeech, a.wavs or Path("results/librispeech_wavs"), a.n)
    df = replay(wavs, a.backends, a.trans, a.realtime, a.jsonl, limit=a.n if a.librispeech else None)
    if len(df):
        cols = ["backend", "recording", "audio_s", "first_partial_ms", "final_latency_ms", "rtf", "wer", "text"]
        pd.set_option("display.max_colwidth", 60)
        print(df[cols].to_string(index=False))
        print("\nper backend (mean):")
        print(df.groupby("backend")[["first_partial_ms", "final_latency_ms", "rtf", "wer"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
