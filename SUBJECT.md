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
- **Set up first, today, not at the start of the lab:** `README.md`, "Setup" (uv or pip; macOS, Linux and
  Windows; the extra day-2 packages and what to do if `kenlm` does not compile).
- One Git repository per group, started from this bundle; tag each delivery `d1`, `d2`, `d3`, `final`.
- Before running experiments: `export DSP_MILESTONE=d1` (PowerShell: `$env:DSP_MILESTONE = "d1"`; d2, d3 likewise): every logged row is tagged.
- Measurements go to `results/results.jsonl` (the provided scripts write them); figures to `results/figures/`.
- Every scientific choice gets an entry in `DECISIONS.md`. The oral exam is built on it.
- `pytest -m d1` runs the public tests of a day. They define "implemented", not "good".
- Use any tool you like, LLMs included (`AGENTS.md` tells coding assistants how to help you here). You are
  responsible for everything you submit, and you must be able to explain it.
- Everything runs on a laptop CPU (macOS, Linux, Windows). A GPU (Snowflake) only speeds up the large models of day 2.
- Live demo: `python -m asr_demo.server`, then http://localhost:8000. `bash run_smoke.sh` needs bash: `uv run bash run_smoke.sh`
  (Windows: Git Bash or WSL, or in PowerShell skip `set -euo pipefail` and paste only the `python -m ...` lines; pass `-m "not stretch"` to `pytest` if you skip the optional day-3 stretch).

## Day 1 — classical recognition (10 November)
**Morning: the two components, each measured on its own.**
| | |
|---|---|
| Implement | `common/pareto.py::pareto_mask_array`; `speech/gmm_hmm.py::recognise_isolated`; `speech/ngram.py`: `NGramLM.fit`, `prob`, `perplexity` |
| Acoustic model | Free Spoken Digit Dataset, MFCC + one GMM-HMM per word: ≥ 8 (states × mixtures) configurations, accuracy vs latency vs parameters; held-out speaker; degradations (white noise, babble, reverberation, telephone) — `python -m speech.run_classical_baseline --part isolated` |
| Language model | perplexity vs order N = 1…5, training size and smoothing (add-k vs interpolation) on a digit grammar with 2nd-order structure — `--part lm`. Which N wins, and when? |
| See it | `notebooks/d1_morning.ipynb`: MFCC step by step, EM on a GMM, a word HMM's state path, then your recogniser, your Pareto front and your n-grams as figures (no code to write in the notebook) |
| Try it live | `python -m asr_demo.server`: say one digit to `digits/0-dtw-baseline` (provided), then to `digits/1-gmm` and `digits/2-hmm` (your `recognise_isolated`) |

Each lab is planned for 2 h of work, which leaves time to get stuck and ask: the header of its notebook gives the
timeline. Each day-1 function is one `TODO(d1)` stub for its whole body; on days 2 and 3, the functions to
implement are written except a few lines, each marked by a `TODO(dX)` that says what goes there. **Since 17 November the
day-1 reference solution is on branch `solutions` of the starter repository, never on `main`**, so pulling `main`
never replaces your code. Compare it with your own, or take a function you need (`git fetch course`, then `git diff HEAD course/solutions -- speech/gmm_hmm.py` to compare, or `git checkout course/solutions -- speech/gmm_hmm.py` to take one file; use
`origin` instead of `course` if that is your name for the starter repository). Your `d1` tag and rows stay as
delivered.

**Afternoon: putting them together.**
| | |
|---|---|
| Implement | `speech/gmm_hmm.py`: `WordLoopDecoder._build` (decoding graph with the LM), `decode_emissions` (Viterbi; pass `delta` through the provided `self.prune` at every frame), `train_bigram` |
| Benchmark | connected digits: acoustic-only vs bigram LM, LM weight sweep, insertion penalty tuned on a dev subset, matched vs mismatched LM domain, ≥ 3 SNRs — `--part connected` |
| Benchmark | beam search: WER vs active states per frame for a score beam and a top-N beam (`WordLoopDecoder(beam=..., n_beam=...)`), clean and 15 dB — `--part beam`. Choose a beam width and justify it in `DECISIONS.md` |
| Think | why does the decoding graph grow with N? (a trigram needs one copy of each word per previous word) |
| See it | `notebooks/d1_afternoon.ipynb`: your decoding graph next to the grammar, the Viterbi trellis of one utterance, a (λ, β) grid on dev, matched vs mismatched text, beam width vs WER and search errors, and your benchmark CSVs as figures |
| Try it live | say digit strings to `digits/3-word-loop` (your decoder) and `digits/4-word-loop+bigram` (+ your bigram); replay the same recording on both |
| Deliver `d1` | rows `gmm_isolated` (≥ 8), `ngram_lm` (≥ 8), `gmm_connected` (≥ 6), `gmm_beam` (≥ 4); figures; ≥ 1 decision tagged d1 · `pytest -m d1` passes |

## Day 2 — neural recognition (17 November)
**Morning: model zoo.**
| | |
|---|---|
| Implement | `speech/asr_benchmark.py::bench_model` |
| Benchmark | ≥ 6 configurations from ≥ 3 families (Whisper sizes, distilled Whisper, wav2vec2 CTC, Moonshine, …) on the **same** 200 LibriSpeech test-clean utterances; batch 1 (latency) vs batch 16 (throughput); with a GPU (optional: Kaggle, Snowflake), CPU vs GPU and fp16/fp32 for ≥ 1 model |
| See it | `notebooks/d2_morning.ipynb`: wav2vec2's frames and labels, Whisper and the normaliser, your `bench_model` on 16 utterances, batch 1 vs 8 |
| Run | `python -m speech.asr_benchmark`: 3 small models (Whisper tiny.en, wav2vec2-base, Moonshine tiny) × batch 1 and 16, about 11 min on a laptop CPU; then add models, e.g. `--models openai/whisper-base.en --n 50` (larger Whisper models need about 1-2 s per utterance on CPU). `--plot` draws the fronts into `results/figures/` |

**Afternoon: adding language information.**
| | |
|---|---|
| Implement | `speech/ctc_lm.py`: `greedy_ctc`, `sweep` |
| Benchmark | a CTC model: greedy vs beam vs beam + n-gram (3-gram, and 4-gram if time) — sweep alpha (LM weight), beta (word bonus), beam; WER vs decoding time. Explicit LM (CTC + n-gram) vs implicit LM (Whisper's decoder): what does each cost? |
| See it | `notebooks/d2_afternoon.ipynb`: your greedy output and its non-words, L4's toy prefix beam search, the LM's two knobs on one utterance, your sweep on 20 dev / 20 test utterances |
| Run | `python -m speech.ctc_lm --arpa data/lm/3-gram.pruned.1e-7.arpa`, about 2 min on a laptop CPU: the grid is swept on 100 dev utterances, the best point decoded once on the other 100 (rows tagged `cfg_split`). 4-gram (optional, 1.3 GB): `python scripts/fetch_data.py --only lm4`, then `--arpa data/lm/4-gram.arpa` |
| Deliver `d2` | rows `asr` (≥ 6) and `ctc_decoding` (≥ 6); fronts; ≥ 1 decision tagged d2 · `pytest -m d2` passes |

## Day 3 — deployment (24 November)
Like day 2, the day-3 functions are written except a few lines: each `TODO(d3)` says what goes there. First,
`python scripts/fetch_data.py` (it skips what you have): the demo's reference streaming transducer changed on
19 November (`reference/fastconformer-114M-int8`, about 100 MB).

**Morning: decide.**
| | |
|---|---|
| Implement | `speech/streaming_sim.py`: `local_agreement` (1 hole), `simulate_streaming` (2 holes: the clock), `stream_native` (1 hole) |
| Budget | merge your fronts and re-measure your candidates **on your laptop** (the demo machine): `python -m asr_demo.replay --librispeech data/LibriSpeech --n 8 --backends ...`. Set your budget (wait after release, WER) from the push-to-talk use case |
| Benchmark | commit lag vs WER: `python -m speech.streaming_sim`, about 6 min on a laptop CPU: Whisper tiny and wav2vec2 re-decoded with LocalAgreement-2 at step ∈ {0.5, 1, 2} s, and the reference streaming transducer, on the same 20 LibriSpeech utterances. Run the default command first (7 rows); then append extra systems with `--models` (another model id, or a backend of `my_backends.py` — `--models` replaces the defaults for that call) |
| See it | `notebooks/d3_morning.ipynb`: 4 candidates on the same recordings, the wait you predict, LocalAgreement and the clock on a mock, the transducer's partials, one utterance through three systems, your streaming front |
| Decide | before the afternoon: the system you deploy, your budget, and your predicted first partial, wait after release and WER, in `DECISIONS.md` |

**Afternoon: the live demo (not graded).** `python -m asr_demo.server`, open `http://localhost:8000`. The
reference is an on-device streaming transducer (`reference/fastconformer-114M-int8`; a smaller one,
`reference/zipformer-20M-*`, is there too).
| | |
|---|---|
| Hijack | register your systems in `my_backends.py` (examples inside: your day-2 decoder with your λ, β, N_beam, and a streaming wrapper around it); contract in `asr_demo/contract.py` |
| Stretch | `speech/streaming_live.py::LocalAgreementSession` (2 holes; it reuses your `local_agreement`), `pytest -m stretch` |
| Compare | the same recordings on several backends (Replay button, `python -m asr_demo.replay`): what you hear vs first partial, wait after release, RTF, WER. Did your prediction from the morning hold? |
| See it | `notebooks/d3_afternoon.ipynb`: the contract, what the demo lists, your systems, your live streamer partial by partial, replays, prediction vs measurement |
| Deliver `d3` | rows `asr_streaming` (≥ 7: ≥ 2 re-decoded models × 3 steps + the reference transducer); `results/pareto.yaml` with your deployment budget (`wer_stream` min, `finish_p50` or `lag_p50` min, budget in **seconds**, e.g. `0.2` for 200 ms); ≥ 1 decision tagged d3 (your deployment choice, and how the demo confirmed or contradicted it) · `pytest -m d3` passes |

Example `results/pareto.yaml` for day 3 (`budget` is in **seconds**, the unit of `finish_p50` and `lag_p50`):
```yaml
quality: {column: wer_stream, direction: min}
cost:    {column: finish_p50, direction: min}
budget:  0.2
claimed_front: [...]
chosen: ...
```

## Final day — defence
Tag `final` before: `REPORT.md` (sections in `SUBMISSION_SPEC.md`), slides, one controlled limits study
(noise, accents/test-other, long audio, LM domain mismatch, …). 15 min presentation + questions on your
decisions; every member answers.
