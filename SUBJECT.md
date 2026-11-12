# Project S — Speech recognition: accuracy, latency, and what language knowledge buys you

## The question
How much does **language information** add to **acoustic evidence**, what does it cost, and which
system would you actually deploy on a laptop for a push-to-talk assistant? You go from a classical
recogniser to modern neural models, add a language model at each level, measure everything as
**Pareto fronts** (WER vs latency / real-time factor / size / hardware), and finally plug your systems
into a live demo next to a provided state-of-the-art on-device recogniser.

You do not train neural models: you select, combine, configure and measure existing ones. The
measurement harness, data, metrics, degradations, mock recognisers and the demo are provided; you
implement the scientific core of each stage (the `TODO(dX)` stubs), then benchmark, optimise, decide.

## How you work
- One Git repository per group, started from this bundle; tag each delivery `d1`, `d2`, `d3`, `final`.
- Before running experiments: `export DSP_MILESTONE=d1` (d2, d3): every logged row is tagged.
- Measurements go to `results/results.jsonl` (the provided scripts write them); figures to `results/figures/`.
- Every scientific choice gets an entry in `DECISIONS.md`. The oral exam is built on it.
- `pytest -m d1` runs the public tests of a day. They define "implemented", not "good".
- Use any tool you like, LLMs included. You are responsible for everything you submit.
- Everything runs on a laptop CPU. A GPU (Snowflake) only speeds up the large models of day 2.

## Day 1 — classical recognition (10 November)
**Morning: the two components, each measured on its own.**
| | |
|---|---|
| Implement | `common/pareto.py::pareto_mask_array`; `speech/gmm_hmm.py::recognise_isolated`; `speech/ngram.py`: `NGramLM.fit`, `prob`, `perplexity` |
| Acoustic model | Free Spoken Digit Dataset, MFCC + one GMM-HMM per word: ≥ 8 (states × mixtures) configurations, accuracy vs latency vs parameters; held-out speaker; degradations (white noise, babble, reverberation, telephone) — `python -m speech.run_classical_baseline --part isolated` |
| Language model | perplexity vs order N = 1…5, training size and smoothing (add-k vs interpolation) on a digit grammar with 2nd-order structure — `--part lm`. Which N wins, and when? |
| Try it live | `python -m asr_demo.server`: say one digit to `digits/0-dtw-baseline` (provided), then to `digits/1-gmm` and `digits/2-hmm` (your `recognise_isolated`) |

**Afternoon: putting them together.**
| | |
|---|---|
| Implement | `speech/gmm_hmm.py`: `WordLoopDecoder._build` (decoding graph with the LM), `decode_emissions` (Viterbi; pass `delta` through the provided `self.prune` at every frame), `train_bigram` |
| Benchmark | connected digits: acoustic-only vs bigram LM, LM weight sweep, insertion penalty tuned on a dev subset, matched vs mismatched LM domain, ≥ 3 SNRs — `--part connected` |
| Benchmark | beam search: WER vs active states per frame for a score beam and a top-N beam (`WordLoopDecoder(beam=..., n_beam=...)`), clean and 15 dB — `--part beam`. Choose a beam width and justify it in `DECISIONS.md` |
| Think | why does the decoding graph grow with N? (a trigram needs one copy of each word per previous word) |
| Try it live | say digit strings to `digits/3-word-loop` (your decoder) and `digits/4-word-loop+bigram` (+ your bigram); replay the same recording on both |
| Deliver `d1` | rows `gmm_isolated` (≥ 8), `ngram_lm` (≥ 8), `gmm_connected` (≥ 6), `gmm_beam` (≥ 4); figures; ≥ 1 decision tagged d1 · `pytest -m d1` passes |

## Day 2 — neural recognition (17 November)
**Morning: model zoo.**
| | |
|---|---|
| Implement | `speech/asr_benchmark.py::bench_model` |
| Benchmark | ≥ 6 configurations from ≥ 3 families (Whisper sizes, distilled Whisper, wav2vec2 CTC, Moonshine, …) on the **same** 200 LibriSpeech test-clean utterances; batch 1 (latency) vs batch 16 (throughput); CPU vs GPU and fp16/fp32 for ≥ 1 model |

**Afternoon: adding language information.**
| | |
|---|---|
| Implement | `speech/ctc_lm.py`: `greedy_ctc`, `sweep` |
| Benchmark | a CTC model: greedy vs beam vs beam + n-gram (3-gram, and 4-gram if time) — sweep alpha (LM weight), beta (word bonus), beam; WER vs decoding time. Explicit LM (CTC + n-gram) vs implicit LM (Whisper's decoder): what does each cost? |
| Deliver `d2` | rows `asr` (≥ 6) and `ctc_decoding` (≥ 6); fronts; ≥ 1 decision tagged d2 · `pytest -m d2` passes |

## Day 3 — deployment (24 November)
**Morning: decide.** Merge your fronts (classical, neural, + LM). Measure the budget **on your laptop**
(the demo machine): what final latency after release is acceptable, what RTF can it sustain? Choose the
configurations to deploy and predict their latency and WER. Implement `speech/streaming_sim.py::simulate_streaming`
and measure commit lag vs WER for ≥ 2 models × step ∈ {0.5, 1, 2} s.

**Afternoon: the live demo (not graded).** `python -m asr_demo.server`, open `http://localhost:8000`.
The provided reference is an on-device streaming recogniser (`reference/zipformer-20M-*`). Hijack the
demo: register your systems in `my_backends.py` (contract in `asr_demo/contract.py`), implement the live
LocalAgreement streamer (`speech/streaming_live.py`, `pytest -m stretch`), and compare backends **on the
same recordings** (Replay button, `python -m asr_demo.replay`): what you hear vs first-partial latency,
wait after release, RTF, WER. Did your prediction from the morning hold?
| | |
|---|---|
| Deliver `d3` | rows `asr_streaming` (≥ 3); `results/pareto.yaml` with your deployment budget; ≥ 1 decision tagged d3 (your deployment choice, and how the demo confirmed or contradicted it) · `pytest -m d3` passes |

## Final day — defence
Tag `final` before: `REPORT.md` (sections in `SUBMISSION_SPEC.md`), slides, one controlled limits study
(noise, accents/test-other, long audio, LM domain mismatch, …). 15 min presentation + questions on your
decisions; every member answers.
