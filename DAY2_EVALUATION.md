# Day 2 — Neural ASR model zoo (morning)

## What was implemented
`speech/asr_benchmark.py::bench_model`: loads a Hugging Face ASR pipeline (fp16 only on GPU), counts the parameters, warms up on 2 utterances, then times every call of the pipeline on fresh copies of the inputs (batch 1 or 16, `cuda_sync` before reading the clock). Returns WER (provided normaliser), RTF = compute time / audio time, latency percentiles (p50, p90, p99), peak memory and energy (None on CPU). Public test: `pytest -m d2` (test_bench_model_metrics passes).

## Setup
LibriSpeech test-clean, 200 utterances (22.5 min of audio); whisper-base.en on the first 50 only. Laptop CPU, Linux x86_64, 16 cores, fp32. Command: `DSP_MILESTONE=d2 python -m speech.asr_benchmark` (+ `--models openai/whisper-base.en --n 50`, then `--plot`).

## Results (batch 1 = latency; batch 16 = throughput)
| Model | Params | WER b1 | RTF b1 | p50 b1 | p90 b1 | WER b16 | RTF b16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| whisper-tiny.en | 38 M | 4.97% | 0.141 | 853 ms | 1514 ms | 4.97% | 0.214 |
| whisper-base.en (50 utt.) | 73 M | 3.10% | 0.224 | 1244 ms | 2248 ms | 3.10% | 0.383 |
| wav2vec2-base-960h | 94 M | 3.32% | 0.043 | 222 ms | 553 ms | 24.47% | 0.131 |
| moonshine-tiny | 27 M | 3.61% | 0.077 | 408 ms | 1050 ms | 3.64% | 0.211 |

## Observations
- wav2vec2 at batch 1 dominates the CPU front: lowest RTF and tail latency, lowest WER on the shared 200 utterances.
- On CPU, batch 16 is slower than batch 1 for every model: batching pads each utterance to the longest one, and without a GPU the padding is wasted compute.
- wav2vec2-base-960h degrades from 3.3% to 24.5% WER when batched: the zero padding changes its internal normalisation. It must be run one utterance at a time.
- Whisper's WER is inflated by formatting (numbers, abbreviations) that the provided normaliser does not map.
- whisper-base.en was measured on 50 utterances only: its WER is not strictly comparable.

## Figure
`results/figures/asr_pareto.png` (WER vs RTF, one panel per batch size).