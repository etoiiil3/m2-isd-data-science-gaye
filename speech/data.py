"""LibriSpeech loading (local OpenSLR folder, the most robust option) + text normalisation.

Download test-clean / test-other from https://www.openslr.org/12 once, put them on a Snowflake
stage, and every group reads the same files. Layout: LibriSpeech/test-clean/<spk>/<chap>/*.flac
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_librispeech(root: Path, subset: str = "test-clean", n: int | None = 200, max_s: float | None = 20.0,
                     seed: int = 0):
    import soundfile as sf
    items = []
    for trans in sorted((Path(root) / subset).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            items.append((trans.parent / f"{uid}.flac", text))
    if not items:
        raise FileNotFoundError(f"no LibriSpeech transcripts (*/*/*.trans.txt) under {Path(root) / subset}: "
                                "run `python scripts/fetch_data.py --only librispeech`, or pass --librispeech")
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    out = []
    for path, text in items:
        y, sr = sf.read(path, dtype="float32")
        if max_s and len(y) / sr > max_s:
            continue
        out.append(dict(id=path.stem, audio=y, sr=sr, text=text))
        if n and len(out) >= n:
            break
    return out


_PUNCT = re.compile(r"[^a-z0-9' ]+")


def normalise(text: str) -> str:
    """Minimal normaliser: lowercase, strip punctuation, squash spaces.
    LIMITATION worth showing students: Whisper writes '20' / 'Mr.' / 'colour' where LibriSpeech
    references say 'twenty' / 'mister' / 'color' -> WER inflated by formatting, not recognition.
    Compare with the Whisper English normaliser (pip install whisper-normalizer) and discuss."""
    t = _PUNCT.sub(" ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer
    refs = [normalise(r) for r in refs]
    hyps = [normalise(h) or "<empty>" for h in hyps]
    return jiwer.wer(refs, hyps)
