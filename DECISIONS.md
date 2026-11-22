# Decision log

One entry per scientific choice, at least one per milestone. Keep entries short and factual: the oral
exam asks you to defend them, including the alternatives you rejected.

## D1 — Which LM weight to use in the connected-digit decoder   (milestone: d1)
- Question: how much should the bigram LM count compared to the acoustic scores, and does it still help when the LM was trained on a different kind of digit sequences?
- Alternatives: acoustic-only (uniform LM); bigram with weight 0.5, 1, 2, 4, 8, 16, 32 or 64.
- Protocol: connected digits built from FSDD recordings, 3 to 7 digits per utterance. Clean, 20, 15 and 10 dB SNR. The bigram is trained on 5,000 text sentences. "Matched" test sequences come from the same Markov chain as the LM text; "mismatched" ones are uniformly random digits. The word insertion penalty is chosen on 30 dev utterances for each setting, and the WER is measured on the remaining ones.
- Evidence: `results/speech_classical/connected_digits_lm.csv` and `connected_digits_lm.png`. Matched domain, 15 dB: WER 37.7% acoustic-only, 18.9% with weight 64. Mismatched domain, 10 dB: 54.7% acoustic-only, 44.3% with weight 16, but 61.4% with weight 64. On clean audio the LM changes almost nothing (about 1.6% to 1%).
- Decision: use a weight of about 16 as the default. It helps in both domains. Weights of 32 to 64 are only better when we know the LM matches the data, and with a mismatched LM weight 64 is worse than no LM at all.
- Would change our mind if: we knew the real digit statistics of the application (then a higher weight is safe), or if we tested on a different speaker or another set of recordings and the best weight moved.

## D1 — Score beam width for connected-digit decoding   (milestone: d1)
- Question: how aggressively can we prune HMM states while preserving recognition quality?
- Alternatives: exact Viterbi search, score beams from 0 to 800, and top-N limits from 1 to 48 active states.
- Protocol: compare on the same matched-text clean and 15 dB test conditions as the connected-digit experiment, using 150 test utterances per condition. Tune the insertion penalty on a separate 30-utterance development set. A search error means pruning did not retain the exact decoder's best-scoring path.
- Evidence: `results/speech_classical/beam_search.csv`. Exact search gives WER 0.82% clean and 16.97% at 15 dB, with 82.0 average active states/frame and about 3.68 ms/utterance. A score beam of **400** preserves both WERs with zero search errors, while reducing active states to 43.8 clean and 65.8 at 15 dB; latency is about 4.03 ms/utterance in both conditions. Tighter beams can retain fewer states, but at 15 dB beam 350 already has search errors (0.8%), and lower widths sharply increase them and WER.
- Decision: use score beam width **400** when preserving the exact-search result across these tested conditions is the priority. This is a quality-preserving pruning setting, not a measured speed optimization: the current dense decoder is slower with pruning than exact search.
- Would change our mind if: a sparse/active-state decoder makes pruning faster, or measurements on more speakers and noise conditions show a narrower beam preserves WER and exact-search paths reliably.

## D3 — ASR model for the push-to-talk assistant on a laptop CPU   (milestone: d2)
- Question: which pretrained model to deploy on the laptop CPU for push-to-talk (latency), and which for offline transcription (throughput)?
- Alternatives: whisper-tiny.en, whisper-base.en, wav2vec2-base-960h (CTC), moonshine-tiny; batch 1 vs batch 16.
- Protocol: same 200 LibriSpeech test-clean utterances (22.5 min) for tiny/wav2vec2/moonshine, 50 for whisper-base.en (slower); laptop CPU (x86_64, 16 cores), fp32, provided normaliser.
- Evidence: rows task=asr, cfg_milestone=d2 in results/results.jsonl; results/figures/asr_pareto.png. Batch 1: wav2vec2 WER 3.3%, RTF 0.043, p90 553 ms; moonshine 3.6%, RTF 0.077, p90 1050 ms; whisper-tiny 5.0%, RTF 0.141, p90 1514 ms; whisper-base 3.1% (50 utt.), RTF 0.224, p90 2248 ms. Batch 16 is slower than batch 1 for every model on CPU (padding), and wav2vec2 degrades to 24.5% WER when batched.
- Decision: wav2vec2-base-960h at batch 1 for push-to-talk: best RTF and tail latency, and the best WER on the shared 200 utterances. Offline on this CPU: batch 1 as well; batching only pays off on a GPU.
- Would change our mind if: CTC without a language model makes too many spelling errors (afternoon), a stronger normaliser closes the gap for Whisper, whisper-base keeps its lower WER on all 200 utterances, or the demo machine has a GPU.


## D4 — External language model for the CTC recogniser (explicit vs implicit LM)   (milestone: d2)
- Question: does an external n-gram LM help the CTC model, at what decoding cost, and which beam width should we use?
- Alternatives: greedy CTC; prefix beam search without LM (N_beam 10, 50, 100); beam + pruned LibriSpeech 3-gram (alpha 0.3, 0.5, 1, 2 x beta 0, 1, 2 x N_beam 10, 50, 100); implicit LM of Whisper (morning, D3).
- Protocol: wav2vec2-base-100h logits on 200 LibriSpeech test-clean utterances, computed once; grid swept on the first 100 (dev), best point decoded once on the other 100 (test); laptop CPU; ms_per_utt is decoding time only (logits precomputed).
- Evidence: rows task=ctc_decoding, cfg_milestone=d2 (cfg_split dev/test) in results/results.jsonl; results/speech_ctc_lm/ctc_lm_wav2vec2-base-100h.csv; results/figures/ctc_decoding_pareto.png. Best on dev: beam + 3-gram, alpha 0.5, beta 1, N_beam 10. Test: greedy 5.92% (0.03 ms/utt), beam without LM 6.19% (7.4 ms/utt), beam + 3-gram 3.93% (5.9 ms/utt). Self-test (closed 21-word vocabulary): greedy 0.33-0.93 -> beam + bigram 0.02-0.06. On dev, alpha 2 is worse than no LM (best 6.43% vs 6.11% greedy), and for alpha 0.5, beta 1 the beam width barely matters (N_beam 10 / 50 / 100: 3.46% / 3.56% / 3.51%).
- Decision: use the explicit LM: CTC + 3-gram with alpha 0.5, beta 1, N_beam 10. It removes about a third of the errors for ~6 ms of decoding per utterance; a beam without LM brings nothing because CTC posteriors are spiky. Together with D3, our push-to-talk system is wav2vec2 (batch 1) + 3-gram: a separate LM can also be retrained on the assistant's own text, which Whisper's implicit LM cannot.
- Would change our mind if: a smaller beam (N_beam 10 is the edge of the grid, but the WER is nearly flat in N_beam) gives the same WER faster; the gain is much smaller on wav2vec2-base-960h (the model chosen in D3, which makes fewer acoustic errors); or the LibriSpeech book-text LM hurts on the assistant's real commands (domain mismatch, as on day 1).