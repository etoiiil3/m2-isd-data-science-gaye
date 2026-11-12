# Subject announcement — Real-time speech recognition

**Real-time speech recognition: accuracy, latency and robustness**

The goal of this project is to build a real-time automatic speech recognition (ASR) system in the spirit of a voice assistant: press a button, speak, and the transcript appears. The project follows the history of the field. We start from a classical system whose core components you implement yourselves: MFCC acoustic features, hidden Markov models with Gaussian mixtures (GMM-HMM), n-gram language models and Viterbi decoding. We then move to modern neural models (wav2vec 2.0, Whisper, streaming transducers), complemented by external language models. The aim is not to train large models but to compare, combine and configure pre-trained models available on Hugging Face.

Every system is evaluated from two angles. Objectively, we measure word error rate (WER), the latency felt by the user, compute cost and robustness to acoustic degradations (noise, reverberation, telephone channel). These measurements are drawn as Pareto fronts, which show what each gain in accuracy costs. Qualitatively, we characterise the trade-offs of each point on the front: a fast model that drops words, an accurate model too slow for conversation, a language model that produces fluent but wrong sentences. The project ends with a live demo in which each group plugs its systems in next to an on-device reference system. Assessment rests mainly on the oral defence, where each group defends its scientific choices. No prior knowledge of speech processing is required; fluency in Python and basic probability are enough.

**Practical:** groups of 4 (3 possible) · bring a laptop with a microphone (a headset is fine) on which you can install Python packages.
