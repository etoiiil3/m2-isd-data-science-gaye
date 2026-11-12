"""Day-1 systems in the live demo: one menu entry per stage of the lab, all on spoken digits (FSDD).

    digits/0-dtw-baseline        provided: template matching with DTW, no statistical model
    digits/1-gmm                 one GMM per digit (a 1-state GMM-HMM: frame order ignored)   needs recognise_isolated
    digits/2-hmm                 one 5-state left-to-right GMM-HMM per digit                  needs recognise_isolated
    digits/3-word-loop           connected digits: word-loop Viterbi, no LM                   needs _build, decode_emissions
    digits/4-word-loop+bigram    the same with a bigram LM                                    needs train_bigram too

An entry that reaches one of YOUR stubs says which one when you select it: implement it, then restart the server.
Entries 0-2 expect ONE digit per recording; entries 3-4 a digit string (short pauses between digits help).

Training data: FSDD (`bash scripts/fetch_data.sh data`, or DSP_FSDD=<folder of wav>), official split (files 5-49
of each speaker). Models are trained on first use with the provided code, exactly as
`run_classical_baseline --part isolated --split standard` does, and cached as results/speech_classical/gmm_S<s>M<m>.pkl
(the files your experiments write are reused). The bigram LM is trained on 5000 strings of a synthetic grammar in
which each digit has two likely successors; "New prompt" draws from the same grammar, so the LM can help.

Try your own tuned lambda and beta by registering another entry in my_backends.py, e.g.
    from asr_demo.backends import register
    from asr_demo.digits import ConnectedDigits, prompt_grammar
    register("student/bigram(λ=4,β=-150)", "my dev-tuned weights", prompt=prompt_grammar)(
        lambda: ConnectedDigits(bigram=True, lm_weight=4.0, word_penalty=-150.0))
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from asr_demo.contract import OfflineBackend, resample

ROOT = Path(__file__).resolve().parents[1]
FSDD = Path(os.environ.get("DSP_FSDD", ROOT / "data/fsdd/recordings"))
CACHE = ROOT / "results/speech_classical"
DIGITS = [str(d) for d in range(10)]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
SR = 8000
TOP_DB = 30  # same trimming threshold as the training data (run_classical_baseline.load_fsdd)


# ----------------------------------------------------------------- data and models (shared by the entries)
@lru_cache(maxsize=1)
def training_data():
    """(training files, per-word normalised training features, global normaliser), as in --part isolated."""
    from speech import run_classical_baseline as R
    if not any(FSDD.glob("*.wav")):
        raise FileNotFoundError(f"no FSDD recordings in {FSDD}: run `bash scripts/fetch_data.sh data` "
                                "(or set DSP_FSDD to the folder of wav files)")
    rng = np.random.default_rng(0)  # same seed and order of draws as run_classical_baseline
    data, sil_gaps = R.prepare(SimpleNamespace(fsdd=FSDD), rng)
    tr, _ = R.split_data(data, "standard", None)
    train, norm = R.training_set(tr, sil_gaps)
    return tr, train, norm


def word_models(n_states: int, n_mix: int):
    """(WordModels, Normaliser) for one configuration: cached file if any, else trained now and cached."""
    path = CACHE / f"gmm_S{n_states}M{n_mix}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    from speech.gmm_hmm import train_word_models
    _, train, norm = training_data()
    wm = train_word_models(train, n_states, n_mix)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((wm, norm), f)
    return wm, norm


def to_training_conditions(audio: np.ndarray, trim: bool) -> np.ndarray:
    """Browser audio (16 kHz) -> 8 kHz, speech at the training level (TARGET_RMS); `trim` cuts the
    leading/trailing silence like the training words."""
    import librosa
    from speech.run_classical_baseline import TARGET_RMS
    y = resample(np.asarray(audio, np.float32), 16000, SR)
    if trim:
        y = librosa.effects.trim(y, top_db=TOP_DB, frame_length=256, hop_length=64)[0]
    parts = librosa.effects.split(y, top_db=TOP_DB, frame_length=256, hop_length=64)
    speech = np.concatenate([y[a:b] for a, b in parts]) if len(parts) else y
    return y * (TARGET_RMS / (np.sqrt(np.mean(speech ** 2)) + 1e-9))


# ----------------------------------------------------------------- prompts ("New prompt" in the page)
_PROMPT_RNG = np.random.default_rng()


@lru_cache(maxsize=1)
def grammar() -> np.ndarray:
    """The digit grammar of the bigram entry (fixed seed): row 10 = sentence start."""
    from speech.run_classical_baseline import markov_chain
    return markov_chain(np.random.default_rng(1))


def prompt_single() -> str:
    return WORDS[_PROMPT_RNG.integers(10)]


def prompt_uniform() -> str:
    return " ".join(WORDS[d] for d in _PROMPT_RNG.integers(10, size=_PROMPT_RNG.integers(3, 6)))


def prompt_grammar() -> str:
    from speech.run_classical_baseline import sample_sequences
    return " ".join(WORDS[int(d)] for d in sample_sequences(_PROMPT_RNG, grammar(), 1, lmin=3, lmax=5)[0])


def needs(fn, *dummy_args):
    """Fail now, with the stub's own message, if `fn` is still a TODO stub (a stub raises before using its
    arguments). An implemented `fn` usually fails on the dummy arguments: that is ignored."""
    try:
        fn(*dummy_args)
    except NotImplementedError:
        raise
    except Exception:
        pass


# ----------------------------------------------------------------- the five entries
class DTWDigits(OfflineBackend):
    """Provided baseline: compare the recording with stored examples of each digit by dynamic time warping
    (13 MFCC, per-utterance normalisation), answer the digit of the closest example. No training, no model:
    the reference point that the statistical models must beat (about 0.95 on the FSDD test split)."""

    def __init__(self, per_speaker: int = 2):
        from speech.gmm_hmm import mfcc_features
        self.mfcc = mfcc_features
        tr, _, _ = training_data()
        self.templates = {w: [] for w in DIGITS}
        for d in tr:
            mine = self.templates[d["digit"]]
            if sum(t[0] == d["speaker"] for t in mine) < per_speaker:
                mine.append((d["speaker"], self._feats(d["y"])))

    def _feats(self, y):
        return self.mfcc(y, SR, deltas=False, cmvn=True)

    def transcribe(self, audio):
        import librosa
        y = to_training_conditions(audio, trim=True)
        if len(y) < SR // 10:
            return ""
        f = self._feats(y)

        def cost(x):  # accumulated distance along the best alignment, per aligned frame pair
            return librosa.sequence.dtw(f.T, x.T)[0][-1, -1] / (len(f) + len(x))
        return WORDS[int(min(DIGITS, key=lambda w: min(cost(x) for _, x in self.templates[w])))]


class IsolatedDigits(OfflineBackend):
    """One GMM-HMM per digit (provided training, hmmlearn) + YOUR recognise_isolated. n_states=1 is a plain GMM:
    every frame is scored by the same mixture, so the order of the frames is ignored."""

    def __init__(self, n_states: int, n_mix: int):
        from speech.gmm_hmm import mfcc_features, recognise_isolated
        needs(recognise_isolated, None, None, None)  # before training: say at once what is missing
        self.recognise, self.mfcc = recognise_isolated, mfcc_features
        self.wm, self.norm = word_models(n_states, n_mix)

    def transcribe(self, audio):
        y = to_training_conditions(audio, trim=True)
        if len(y) < SR // 10:
            return ""
        return WORDS[int(self.recognise(self.wm, self.norm(self.mfcc(y, SR)), DIGITS))]


class ConnectedDigits(OfflineBackend):
    """Digit strings: the 5-state word models + silence, YOUR word-loop decoder, and (bigram=True) YOUR
    train_bigram on the grammar's text. Default lambda/beta: the best clean, matched cells of the reference
    run (acoustic only: beta=-300; bigram: lambda=2, beta=-200)."""

    def __init__(self, bigram: bool, lm_weight: float | None = None, word_penalty: float | None = None):
        from speech.gmm_hmm import WordLoopDecoder, mfcc_features, train_bigram
        from speech.run_classical_baseline import STANDARD_MODEL, sample_sequences
        needs(WordLoopDecoder._build, None)  # before training: say at once what is missing
        needs(WordLoopDecoder.decode_emissions, None, None)
        if bigram:
            needs(train_bigram, [], [])
        self.mfcc = mfcc_features
        self.wm, self.norm = word_models(*STANDARD_MODEL)
        lm = None
        if bigram:
            text = sample_sequences(np.random.default_rng(2), grammar(), 5000)
            lm = train_bigram(text, DIGITS)
        lm_weight = lm_weight if lm_weight is not None else (2.0 if bigram else 1.0)
        word_penalty = word_penalty if word_penalty is not None else (-200.0 if bigram else -300.0)
        self.dec = WordLoopDecoder(self.wm, DIGITS + ["sil"], lm, lm_weight=lm_weight, word_penalty=word_penalty)

    def transcribe(self, audio):
        y = to_training_conditions(audio, trim=False)
        if len(y) < SR // 10:
            return ""
        return " ".join(WORDS[int(d)] for d in self.dec.decode(self.norm(self.mfcc(y, SR))))
