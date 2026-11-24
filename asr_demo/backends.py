"""Backend registry. The demo lists every registered backend; the browser picks one PER RECORDING.

Register your own (this is how you "hijack" the demo) in my_backends.py at the bundle root:

    from asr_demo.backends import register
    from asr_demo.contract import OfflineBackend

    @register("student/my-model", "what it is, in one line")
    class Mine(OfflineBackend):
        def __init__(self): ...                 # load once
        def transcribe(self, audio): return "..."

Factories are lazy: a model is loaded the first time someone selects it, then cached.
"""
from __future__ import annotations

import importlib
import os
import threading
import traceback
from pathlib import Path

import numpy as np

from asr_demo import digits
from asr_demo.contract import Backend, OfflineBackend, Session

REGISTRY: dict[str, dict] = {}
_CACHE: dict[str, Backend] = {}
_LOCK = threading.RLock()  # re-entrant: a factory may get() its inner backend (LocalAgreement around a model)
_FAILED: dict[str, str] = {}  # name -> one-line reason; a failed factory is not retried until the server restarts
ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("DSP_MODELS", ROOT / "models"))
DATA = Path(os.environ.get("DSP_DATA", ROOT / "data"))


class BackendUnavailable(RuntimeError):
    """Unknown backend, or a factory failed (missing package, missing model file, ...). str(e) says what to do."""


def explain(e: BaseException) -> str:
    if isinstance(e, ModuleNotFoundError):
        return f"missing Python module '{e.name}' (install it in the demo's environment)"
    if isinstance(e, NotImplementedError):  # one of the TODO stubs: say which one
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        code = tb.tb_frame.f_code
        where = f"{Path(code.co_filename).parent.name}/{Path(code.co_filename).name}::{getattr(code, 'co_qualname', code.co_name)}"
        return f"needs your code: {where} is not implemented yet ({e}). Implement it, then restart the server"
    return f"{type(e).__name__}: {e}"


def register(name: str, description: str = "", prompt=None):
    """`prompt`: optional function returning a sentence to read aloud ("New prompt" in the page)."""
    def deco(factory):
        with _LOCK:
            REGISTRY[name] = {"factory": factory, "description": description, "prompt": prompt}
            _CACHE.pop(name, None)
            _FAILED.pop(name, None)
            for cached_name in [k for k, b in _CACHE.items() if getattr(getattr(b, "inner", None), "name", None) == name]:
                _CACHE.pop(cached_name, None)
        return factory
    return deco


def get(name: str) -> Backend:
    with _LOCK:  # the page preloads on selection while a recording may start: never build the same model twice
        return _get(name)


def _get(name: str) -> Backend:
    if name in _CACHE:
        return _CACHE[name]
    if name not in REGISTRY:
        raise BackendUnavailable(f"unknown backend {name}")
    try:
        b = REGISTRY[name]["factory"]()
    except Exception as e:
        _FAILED[name] = explain(e)
        if not isinstance(e, (ModuleNotFoundError, FileNotFoundError, NotImplementedError)):  # explained, not bugs
            traceback.print_exc()
        raise BackendUnavailable(f"{name} is unavailable: {_FAILED[name]}. Fix it, then restart the server.") from e
    _FAILED.pop(name, None)
    b.name = name
    _CACHE[name] = b
    return b


def available() -> list[dict]:
    return [{"name": n, "description": r["description"], "loaded": n in _CACHE, "error": _FAILED.get(n),
             "has_prompt": r["prompt"] is not None} for n, r in REGISTRY.items()]


def prompt(name: str) -> str | None:
    r = REGISTRY.get(name)
    return r["prompt"]() if r and r["prompt"] else None


# ----------------------------------------------------------------- day 1: digits, one entry per stage of the lab
# (asr_demo/digits.py: provided baseline + your recognise_isolated, word-loop decoder and bigram)
register("digits/0-dtw-baseline", "PROVIDED baseline, one digit: template matching (DTW), no model",
         prompt=digits.prompt_single)(lambda: digits.DTWDigits())
register("digits/1-gmm", "one digit: one GMM per digit, frame order ignored — uses your recognise_isolated",
         prompt=digits.prompt_single)(lambda: digits.IsolatedDigits(n_states=1, n_mix=8))
register("digits/2-hmm", "one digit: 5-state GMM-HMM per digit — uses your recognise_isolated",
         prompt=digits.prompt_single)(lambda: digits.IsolatedDigits(n_states=5, n_mix=2))
register("digits/3-word-loop", "digit strings, no LM — uses your _build + decode_emissions (Viterbi)",
         prompt=digits.prompt_uniform)(lambda: digits.ConnectedDigits(bigram=False))
register("digits/4-word-loop+bigram", "digit strings + bigram LM (λ=2, β=-200) — uses your train_bigram too",
         prompt=digits.prompt_grammar)(lambda: digits.ConnectedDigits(bigram=True))


# ----------------------------------------------------------------- reference: on-device streaming transducers
class _SherpaSession(Session):
    def __init__(self, rec, lead_s=0.0):
        self.rec, self.s, self.last = rec, rec.create_stream(), ""
        if lead_s:  # silence before the speech, fed when the session opens (before any audio arrives)
            self.s.accept_waveform(16000, np.zeros(int(lead_s * 16000), np.float32))

    def _text(self):
        r = self.rec.get_result(self.s)
        return (r if isinstance(r, str) else r.text).strip().lower()

    def accept_chunk(self, audio):
        self.s.accept_waveform(16000, audio)
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
        t = self._text()
        if t != self.last:
            self.last = t
            return t
        return None

    def finish(self):
        self.s.accept_waveform(16000, np.zeros(int(0.5 * 16000), np.float32))  # tail padding flushes the encoder
        self.s.input_finished()
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
        return self._text()


class SherpaTransducer(Backend):
    """Streaming transducer exported to ONNX (folder from scripts/fetch_data.py), run by sherpa-onnx on CPU with
    greedy search: its hypothesis only grows, chunk after chunk. lead_s: seconds of silence fed before the speech
    (the 20M zipformer loses the start of an utterance without it: lecture L6)."""
    def __init__(self, folder: str, int8: bool = True, threads: int = 2, lead_s: float = 0.0):
        import sherpa_onnx
        d = MODELS / folder
        if not (d / "tokens.txt").exists():
            raise FileNotFoundError(f"{d} not found: run `python scripts/fetch_data.py` (downloads the demo models)")

        def part(name):
            files = sorted(f for f in d.glob(f"{name}*.onnx") if (".int8." in f.name) == int8)
            if not files:
                raise FileNotFoundError(f"no {'int8' if int8 else 'fp32'} {name}*.onnx in {d}")
            return str(files[0])
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=part("encoder"), decoder=part("decoder"), joiner=part("joiner"),
            num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
        self.lead_s = lead_s

    def new_session(self):
        return _SherpaSession(self.rec, self.lead_s)


SherpaZipformer = SherpaTransducer  # name used before 19 Nov
FASTCONFORMER_114M = "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8"
ZIPFORMER_20M = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
register("reference/fastconformer-114M-int8",
         "THE REFERENCE: streaming transducer (NVIDIA FastConformer, 80 ms look-ahead), 114M params, int8, CPU")(
    lambda: SherpaTransducer(FASTCONFORMER_114M, int8=True))
register("reference/zipformer-20M-int8", "small streaming transducer (k2 zipformer), 23M params, int8: loses first words")(
    lambda: SherpaTransducer(ZIPFORMER_20M, int8=True))
register("reference/zipformer-20M-fp32", "same small model in fp32: what does quantisation cost or save?")(
    lambda: SherpaTransducer(ZIPFORMER_20M, int8=False))


# ----------------------------------------------------------------- Hugging Face offline models
class HFPipeline(OfflineBackend):
    def __init__(self, model_id: str, partial_every_s=None):
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        self.partial_every_s = partial_every_s

    def transcribe(self, audio):
        if len(audio) < 1600:  # under 0.1 s: nothing to recognise (and CTC models fail on very short inputs)
            return ""
        return self.asr({"raw": audio, "sampling_rate": 16000})["text"].strip().lower()


class CTCWithLM(OfflineBackend):
    """Your day-2 system as a demo backend: a wav2vec2 CTC model + pyctcdecode's prefix beam search with a KenLM
    n-gram (speech/ctc_lm.py::make_decoder). arpa=None: beam search without LM. alpha = lambda (LM weight),
    beta = beta (word bonus), beam = N_beam. Use the values your d2 sweep chose on dev (DECISIONS.md)."""
    def __init__(self, model_id: str = "facebook/wav2vec2-base-100h", arpa: str | None = "lm/3-gram.pruned.1e-7.arpa",
                 alpha: float = 0.5, beta: float = 1.0, beam: int = 10):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        from speech.ctc_lm import labels_from_processor, make_decoder
        if arpa:  # as given, or relative to the repository root, or to the data folder
            found = next((c for c in (Path(arpa), ROOT / arpa, DATA / arpa) if c.exists()), None)
            if found is None:
                raise FileNotFoundError(f"{arpa} not found: run `python scripts/fetch_data.py --only lm`")
            arpa = str(found)
        self.torch, self.beam = torch, beam
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        labels, _ = labels_from_processor(self.proc)
        self.decoder = make_decoder(labels, arpa, alpha, beta)

    def transcribe(self, audio):
        if len(audio) < 1600:
            return ""
        with self.torch.inference_mode():
            x = self.proc(audio, sampling_rate=16000, return_tensors="pt").input_values
            lp = self.torch.log_softmax(self.model(x).logits[0].float(), -1).numpy()
        return self.decoder.decode(lp, beam_width=self.beam).strip()


register("hf/whisper-tiny.en", "Whisper tiny (39M), offline: text after release")(lambda: HFPipeline("openai/whisper-tiny.en"))
register("hf/whisper-base.en+partials", "Whisper base, re-decoded every 1 s for live text")(
    lambda: HFPipeline("openai/whisper-base.en", partial_every_s=1.0))
register("hf/wav2vec2-base-960h", "CTC, greedy decoding, no LM")(lambda: HFPipeline("facebook/wav2vec2-base-960h"))
register("hf/wav2vec2-base-100h+3gram", "CTC + 3-gram (alpha=0.5, beta=1, N_beam=10): day 2, provided (needs kenlm)")(
    lambda: CTCWithLM("facebook/wav2vec2-base-100h", "lm/3-gram.pruned.1e-7.arpa", alpha=0.5, beta=1.0, beam=10))


def _local_agreement(inner: str, step_s: float = 1.0):
    def factory():
        from speech.streaming_live import LocalAgreementBackend, _Probe
        LocalAgreementBackend(_Probe(), step_s=step_s)  # check the TODO stubs before loading the inner model
        return LocalAgreementBackend(get(inner), step_s=step_s)
    return factory


register("student/local-agreement(whisper-tiny.en)", "your day-3 streaming policy around Whisper tiny, 1 s step")(
    _local_agreement("hf/whisper-tiny.en"))
register("student/local-agreement(wav2vec2-base-960h)", "your day-3 streaming policy around wav2vec2 CTC, 0.5 s step")(
    _local_agreement("hf/wav2vec2-base-960h", step_s=0.5))

# students' own backends
try:
    importlib.import_module("my_backends")
except ModuleNotFoundError as e:
    if e.name != "my_backends":
        raise
