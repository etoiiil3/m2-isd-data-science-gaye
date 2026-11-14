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

