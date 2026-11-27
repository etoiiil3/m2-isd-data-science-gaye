# Limits of our deployed system

Deployed system (decision D5): **wav2vec2-base-960h + our LocalAgreement-2, step 0.5 s, greedy CTC, no LM**,
on a laptop CPU, for a push-to-talk assistant (short English commands of 2-3 s).
Budget: wait after release ≤ 0.2 s and WER ≤ 5%.
This file gathers every limit we measured or observed. It is the source of the "Limits" section of REPORT.md.

## 1. Controlled limits study: white noise (the graded study)

**Protocol.** Same 20 LibriSpeech test-clean utterances as `speech.streaming_sim` (137 s of audio). White noise is
added at a fixed signal-to-noise ratio with the provided `speech/degrade.py`, with the same noise (fixed seed per
utterance) for every system. Only the SNR changes. Both systems are streamed exactly as on day 3. Our system uses
LocalAgreement-2 at step 0.5 s; the provided transducer streams natively in 100 ms chunks. Word end times come from
the clean audio, because additive noise does not move the words in time.
Command: `DSP_MILESTONE=final python -m speech.limits_noise`.

**Evidence:** `results/limits_noise.csv`, `results/limits_noise.jsonl`, `results/figures/limits_noise.png`.

| SNR | ours: wav2vec2 + LocalAgreement (WER) | reference transducer (WER) |
|---|---:|---:|
| clean | 3.4% | 2.1% |
| 20 dB | 5.5% | 3.4% |
| 10 dB | 17.8% | 5.5% |
| 5 dB | 51.2% | 8.6% |
| 0 dB | 84.6% | 28.5% |

**Finding.** Our system meets the 5% WER budget **only on clean audio**. It is just over the budget at 20 dB, and it
breaks down from 10 dB (5 times more errors than clean). The provided transducer degrades much more slowly: it stays
near the budget at 10 dB and is still usable at 5 dB.
**Consequence.** Our deployment choice is valid for a quiet room only. In a noisy room we would deploy the provided
transducer (this is the first "Would change our mind if" condition of D5).
**Link with day 1.** With the GMM-HMM, the LM helped most in noise (D1). Section 2b measures the same question for
the neural model.

## 2. No language model in the deployed system

D4 chose an explicit LM (CTC + 3-gram) on wav2vec2-base-100h, but the model chosen in D3, and deployed in D5, is
base-960h, and it runs without LM. We re-ran the D4 protocol on base-960h (same 200 utterances, grid on the first
100, best point once on the other 100).

**Evidence:** rows `task=ctc_decoding, cfg_milestone=final`; `results/speech_ctc_lm/ctc_lm_wav2vec2-base-960h.csv`.

| test set (100 utterances) | base-100h (D4) | base-960h (final) |
|---|---:|---:|
| greedy, no LM | 5.92% | 3.39% |
| beam, no LM | 6.19% | 3.28% |
| beam + 3-gram (alpha 0.5, beta 1) | 3.93% | 2.85% |
| relative gain of the 3-gram | -34% | -16% |

**Finding.** The 3-gram still helps on base-960h offline, but its relative gain is halved: the 960h model makes fewer
acoustic errors, so the LM has fewer to correct, as anticipated in D4. As on day 2, alpha 2 is worse than no LM.
**Not measured:** the 3-gram on base-960h **in streaming** and **in noise** (section 2b measures both on base-100h).

## 2b. Does the 3-gram help more in noise? (same protocol as section 1, base-100h with and without LM)

**Protocol.** Same 20 utterances, same white noise and same streaming (LocalAgreement-2, step 0.5 s) as section 1.
The acoustic model is the same on both sides (wav2vec2-base-100h): one side uses greedy CTC, the other our D4 decoder
(beam + 3-gram, alpha 0.5, beta 1, N_beam 10). Only the LM changes.
Command: `DSP_MILESTONE=final python -m speech.limits_noise --models facebook/wav2vec2-base-100h hf/wav2vec2-base-100h+3gram --jsonl results/limits_noise_3gram.jsonl --out results/limits_noise_3gram.csv --fig results/figures/limits_noise_3gram.png`.

**Evidence:** `results/limits_noise_3gram.csv`, `results/limits_noise_3gram.jsonl`, `results/figures/limits_noise_3gram.png`.

| SNR | base-100h, no LM | base-100h + 3-gram | errors removed by the LM |
|---|---:|---:|---:|
| clean | 4.7% | 3.7% | -1.0 pt (-21%) |
| 20 dB | 10.2% | 7.8% | -2.4 pts (-24%) |
| 10 dB | 20.1% | 17.8% | -2.3 pts (-11%) |
| 5 dB | 49.9% | 42.6% | -7.3 pts (-15%) |
| 0 dB | 80.9% | 75.7% | -5.2 pts (-6%) |

**Finding.** The 3-gram helps in **every** condition, and in streaming too (4.7% to 3.7% on clean audio). In absolute
points it removes more errors as the noise grows, but the **share** of errors it corrects falls, from about a quarter
(clean, 20 dB) to 6% at 0 dB. When the acoustic evidence collapses, the LM cannot rebuild the sentence: at 10 dB the
system with LM is still at 17.8%, far above our 5% budget. This differs from day 1, where the bigram halved the
errors in noise (matched domain, 15 dB: 37.7% to 18.9%). There the LM was trained on the same kind of digit
sequences as the test; here a book-text LM meets read speech, with alpha tuned on clean audio.
**Limit of this test.** alpha was tuned on clean speech (D4). A larger alpha might help in noise, as on day 1, but
we did not tune it per condition. The cost of the LM is about 20 ms more wait (243 to 269 ms on clean audio, Grace's
laptop). For reference, at 5 dB and 0 dB base-100h + 3-gram (42.6%, 75.7%) is better than our deployed base-960h
without LM (51.2%, 84.6%, section 1). The acoustic models differ, so this is not a controlled comparison, but it
suggests adding the 3-gram to base-960h would help in noise.

## 3. Our own voices: the live demo (no longer read LibriSpeech)

**Evidence:** rows `task=demo` and `demo_replay`; `results/asr_demo_history.csv`; `results/demo_recordings/`.
Five commands, six recordings (29 words), our voices, laptop microphone; the same recordings were replayed on the
reference.

- **WER about 10%** (3 errors in 29 words), instead of 3.4% on LibriSpeech: "weather" became "weazor", "Yoan" became
  "join", and "call" became "cor". The causes are non-native accents, a first name absent from LibriSpeech, and a
  laptop microphone. The reference also rose, to about 17% (5 errors). On 29 words the two WERs are **not
  significantly different**.
- **Names and domain words.** Without an LM, a CTC model spells what it hears. Words outside LibriSpeech, such as
  first names or app names, are the most fragile. A 3-gram retrained on the assistant's commands is the natural fix.

## 4. Language: English only

"appelle maman" (French) fails on every system: ours gave "i pay him mammo", Whisper "atel mamo." and the reference
"apel mammon". All the models are English-only. A French assistant needs another acoustic model.

## 5. Latency depends on the machine and on the length of the request

- **Machine.** The wait after release was 97 ms on Anastassia's laptop (the demo machine), but 280-317 ms on Grace's
  slower laptop, measured during the limits study. On that laptop our system **goes over the 0.2 s budget**, while
  the reference stays at about 95 ms. This is the course rule in action: a point measured elsewhere is not on your
  front, so re-measure on the deployment machine.
- **Length.** Re-decoding transcribes the whole buffer again at release, so the wait grows with the length of the
  request: 99-116 ms for 2.5-4.7 s commands, but 199 ms for a 6.6 s one, right at the budget. Long dictation (20 s)
  would exceed it. The streaming transducer's wait stays constant (29-40 ms) whatever the length.
- **First text.** It appeared after 1.3-1.8 s instead of the predicted 1 s. There is silence before speaking, and
  LocalAgreement needs two agreeing hypotheses before it shows a word.

## 6. Implicit LM hallucinations (why we prefer an explicit LM)

**Evidence:** `results/call_your_mother_whisper_partial.png`, `results/noisy_audio_whisper_final_out.png`, rows
`task=demo_from_file`.

- On a noisy upload of "call your mother", `whisper-base.en+partials` showed an intermediate partial that repeated
  "come on," dozens of times, then gave the wrong final text "go, your mother." after an **8.4 s wait** (RTF 2.35).
- On another noisy recording it looped on "i'll see you in a minute" **more than 50 times**, with a **16 s wait**
  (RTF 3.55).
- **Why.** Whisper's decoder writes each token from the previous ones (an implicit LM). When the acoustic evidence is
  weak, the LM takes over and produces fluent but wrong text, or loops. CTC emits one label per frame, so it cannot
  produce more words than there is audio: our system made phonetic errors ("waa ta at y a goi rygers an") but invented
  nothing. The reference transducer recovered "call your mother".
- This supports D4: an explicit LM can be weighted (alpha), tuned and retrained; an implicit LM cannot.

## 7. Our system is not on the Pareto front

On every axis we measured (WER, wait after release, commit lag, robustness to noise), the provided transducer is
better: 2.1% WER and a 37 ms wait, against 3.4% and 97 ms. Our system meets the budget in quiet conditions, and we
deploy it because it is our own pipeline (D3 model, our streaming policy, an LM we can add and retrain). Re-decoding an
offline model does not reach a model designed for streaming.

## 8. Statistical limits of our measurements

- The sets are small: 20 utterances (383 words) for streaming and noise, 8 utterances for the replay, 29 live words,
  and 100 test utterances for the LM.
- We report no confidence intervals. Differences of about 1 point of WER (for example the 3-gram on base-960h:
  3.39% to 2.85%) may not be significant. A paired bootstrap on per-utterance errors (course L5) is the next check.
- Each measurement was run once, on one laptop, with one noise type (white noise). Babble noise, reverberation and the
  telephone channel are untested on the neural systems.