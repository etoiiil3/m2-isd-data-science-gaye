"""Register YOUR systems here: they appear in the demo's backend list (python -m asr_demo.server), in
`python -m asr_demo.replay --backends ...` and in `python -m speech.streaming_sim --models ...`.
Same contract as the provided ones (asr_demo/contract.py). Uncomment and adapt (lecture L6, "Hijack the demo")."""
# from asr_demo.backends import CTCWithLM, SherpaTransducer, ZIPFORMER_20M, get, register
# from speech.streaming_live import LocalAgreementBackend
#
# 1. Your day-2 decoder, with the values your sweep chose on dev (DECISIONS.md, d2):
# register("student/ctc+3gram", "wav2vec2-base-100h + 3-gram, our d2 choice (alpha=0.5, beta=1, N_beam=10)")(
#     lambda: CTCWithLM("facebook/wav2vec2-base-100h", "lm/3-gram.pruned.1e-7.arpa", alpha=0.5, beta=1.0, beam=10))
#
# 2. Your day-3 streaming policy around it (stretch: speech/streaming_live.py):
# register("student/local-agreement(ctc+3gram)", "our LocalAgreement-2, 0.5 s step")(
#     lambda: LocalAgreementBackend(get("student/ctc+3gram"), step_s=0.5))
#
# 3. A streaming transducer with silence before the speech (the 20M zipformer loses first words without it):
# register("student/zipformer-20M+lead1s", "20M zipformer, 1 s of silence first")(
#     lambda: SherpaTransducer(ZIPFORMER_20M, int8=True, lead_s=1.0))
#
# 4. Anything else: a class with transcribe(audio) -> lower-case words (16 kHz float32 in).
# import numpy as np
# from asr_demo.contract import OfflineBackend
#
# @register("student/my-model", "what it is, in one line")
# class Mine(OfflineBackend):
#     def __init__(self):
#         ...                                   # load the model ONCE
#     def transcribe(self, audio: np.ndarray) -> str:
#         ...
