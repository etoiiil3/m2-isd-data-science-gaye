# Day 1 — Acoustic and language model evaluation

## Goal

Day 1 compares classical digit recognition from two perspectives: acoustic evidence from MFCC + GMM-HMM models, and language information from n-gram models. This report collects the benchmark setup and interpretation for both evaluations. Architecture decisions for the vocal assistant belong in `DECISIONS.md`.

## Experimental setup

- Dataset: Free Spoken Digit Dataset (FSDD), 8 kHz audio.
- Split: `standard`; recordings with indices 5 and above are used for training, and indices 0–4 for testing. This is **not** a held-out-speaker evaluation: speakers may occur in both partitions.
- Configurations: 4 state counts (2, 3, 5, 8) × 3 mixture counts (1, 2, 4), for 12 configurations.
- Conditions: clean, white noise at 10 dB SNR, babble at 5 dB SNR, reverberation with RT60 0.6 s, and telephone-channel degradation.
- Run: `DSP_MILESTONE=d1 .venv/bin/python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part isolated`
- Artifacts: `results/speech_classical/isolated_digits.csv` and rows tagged `cfg_milestone=d1` in `results/results.jsonl`.

## Isolated acoustic model sweep

## Clean audio results

Latency is the mean time to classify one test recording, in milliseconds. It excludes audio loading and MFCC extraction. `SxMy` means x HMM states and y Gaussian mixtures per state. The Pareto front is computed over latency (minimize) and accuracy (maximize).

| Configuration | Accuracy | Latency (ms/recording) | Parameters | Pareto front |
|---|---:|---:|---:|:---:|
| S2M1 | 95.67% | 1.55 | 1,866 | Yes |
| S2M2 | 97.33% | 1.66 | 3,683 | Yes |
| S2M4 | 99.00% | 1.83 | 6,843 | Yes |
| S3M1 | 96.00% | 1.99 | 2,706 | No |
| S3M2 | 98.33% | 2.14 | 5,313 | No |
| S3M4 | 99.00% | 2.44 | 10,053 | No |
| S5M1 | 98.00% | 2.80 | 4,446 | No |
| S5M2 | 99.00% | 3.01 | 8,633 | No |
| S5M4 | 99.67% | 3.42 | 16,533 | Yes |
| S8M1 | 97.67% | 3.97 | 7,206 | No |
| S8M2 | 99.00% | 4.34 | 13,763 | No |
| S8M4 | 99.00% | 4.99 | 26,403 | No |

## Interpretation

The clean-audio Pareto front contains S2M1, S2M2, S2M4, and S5M4. Each gives a trade-off: the smaller 2-state systems are faster, while S5M4 has the highest observed accuracy and higher latency. For example, S2M2 is non-dominated because it improves accuracy over S2M1 while taking slightly longer; neither configuration is better on both measures.

Several larger configurations are dominated. S3M4 has the same observed clean accuracy as S2M4 (99%) but takes longer and has more parameters. S8M2 and S8M4 also reach only 99%, at substantially higher latency than S2M4.

The test set contains 300 recordings, so one additional correct answer changes accuracy by about 0.33 percentage points. S5M4's 99.67% is 299/300 correct, compared with 297/300 for S2M4. That two-recording difference is suggestive, but should not be treated as strong evidence that S5M4 generalizes better without repeated runs or another test set.

## Degradation results

Mean accuracy across the 12 configurations for each condition:

| Condition | Mean accuracy | Range across configurations |
|---|---:|---:|
| Clean | 98.1% | 95.7–99.7% |
| White noise, 10 dB SNR | 68.9% | 54.7–79.0% |
| Babble, 5 dB SNR | 63.4% | 56.7–69.0% |
| Reverberation, RT60 0.6 s | 92.6% | 80.7–96.7% |
| Telephone channel | 77.6% | 60.7–89.3% |

Noise and babble cause the largest average accuracy drops. More mixtures help in some degraded conditions—S5M2 reaches 79% on 10 dB white noise—but gains are not consistent across every configuration or condition. The recogniser is strong on clean digits but remains sensitive to acoustic mismatch.

All measured real-time factors were below 0.014 on this machine, so isolated classification was much faster than the audio duration. Training time rose with model complexity in general; it is a one-time fitting cost, separate from per-recording recognition latency.

## Held-out-speaker evaluation: Theo

The unseen-speaker run trains without Theo and evaluates on all 500 Theo recordings, using clean audio. The CSV contains 12 `unseen_theo` rows and the JSONL contains the corresponding 12 `gmm_isolated` rows tagged `cfg_milestone=d1`; the measurements agree between the two files.

| Configuration | Accuracy | Latency (ms/recording) | Parameters | Pareto front |
|---|---:|---:|---:|:---:|
| S2M1 | 95.8% | 1.58 | 1,866 | Yes |
| S2M2 | 98.2% | 1.68 | 3,683 | Yes |
| S2M4 | 97.8% | 1.84 | 6,843 | No |
| S3M1 | 98.2% | 2.00 | 2,706 | No |
| S3M2 | 98.8% | 2.14 | 5,313 | Yes |
| S3M4 | 98.4% | 2.37 | 10,053 | No |
| S5M1 | 98.6% | 2.78 | 4,446 | No |
| S5M2 | 97.8% | 3.03 | 8,633 | No |
| S5M4 | 98.0% | 3.42 | 16,533 | No |
| S8M1 | 98.2% | 3.99 | 7,206 | No |
| S8M2 | 97.6% | 4.31 | 13,763 | No |
| S8M4 | 97.8% | 4.97 | 26,403 | No |

On this speaker, accuracy ranges from 95.8% to 98.8%, with a mean of 97.9% across configurations. The held-out Pareto front is S2M1, S2M2, and S3M2: S3M2 reaches the highest accuracy, while the 2-state models offer lower latency. Increasing states and mixtures does not consistently improve results; for example, S5M4 is slower and less accurate than S3M2.

The standard clean split averaged about 98.1% across configurations, close to Theo's 97.9%. This suggests no large accuracy collapse on this one unseen speaker, but it is not enough to establish generalization across speakers. The two evaluations also have different test-set sizes (300 standard recordings versus 500 Theo recordings), so compare the broad pattern rather than small percentage differences. Each Theo test example changes accuracy by 0.2 percentage points.

The run emitted a scikit-learn GMM initialization convergence warning during S5M2 training. Training continued and the resulting GMM-HMM was evaluated; retain the result with this caveat rather than treating it as a failed test.

## N-gram language model evaluation

The experiment compared n-gram orders 1–5 and two smoothing methods on a synthetic digit grammar where the next digit depends on the previous two. It used 50, 500, and 5,000 training sentences and the same generated test set of 1,000 sentences. Lower perplexity is better.

| Training sentences | Smoothing | 1-gram | 2-gram | 3-gram | 4-gram | 5-gram | Best order |
|---:|---|---:|---:|---:|---:|---:|---|
| 50 | Add-k | 10.75 | 11.15 | 8.08 | 8.61 | 8.63 | 3 |
| 50 | Interpolated | 10.75 | 9.77 | 7.76 | **7.73** | 7.99 | 4 |
| 500 | Add-k | 10.69 | 8.99 | **5.69** | 6.65 | 7.16 | 3 |
| 500 | Interpolated | 10.69 | 9.19 | 6.00 | **5.83** | 6.22 | 4 |
| 5,000 | Add-k | 10.69 | 8.89 | **5.05** | 5.16 | 5.67 | 3 |
| 5,000 | Interpolated | 10.69 | 9.14 | 5.73 | **5.17** | 5.23 | 4 |

The unigram baseline stays near 10.7, while trigrams improve substantially because they can use the two-token context that generated the data. With 500 and 5,000 sentences, the add-k trigram is best overall. This matches the source grammar's true order. With only 50 sentences, the interpolated 4-gram has the lowest perplexity (7.73), narrowly beating the interpolated trigram (7.76): mixing shorter-order estimates helps when longer histories are sparse. Interpolation also helps the 4- and 5-gram estimates at the larger training sizes, although the add-k trigram remains best overall.

Higher order does not automatically improve perplexity. For example, at 5,000 training sentences, the add-k 5-gram scores 5.67, worse than the trigram's 5.05. The model has to estimate many more distinct histories and continuations: the number of distinct n-gram entries across all orders grows from 1,035 for the 3-gram to 12,693 for the 5-gram. These results illustrate the trade-off between context length and data sparsity. They come from one fixed synthetic test set, so small differences should be treated as suggestive rather than universal.

The run produced 30 `ngram_lm` rows tagged `cfg_milestone=d1` in `results/results.jsonl` and the full measurements in `results/speech_classical/ngram_perplexity.csv`.

## Connected digits: acoustic model + bigram LM

Setup: utterances of 3 to 7 digits are built by gluing FSDD recordings with short silences. The decoder is a word loop where each digit HMM (and a `sil` HMM) is connected to the others, and the transition from word u to word v gets `lm_weight * log P(v|u) + word_penalty`. The bigram is trained with add-k (k = 0.1) on 5,000 text sentences. In the "matched" test the digit sequences come from the same Markov chain as the LM text. In the "mismatched" test they are uniformly random. The insertion penalty is picked on 30 dev utterances for each setting, then the WER is measured on the other utterances. The run gives 72 `gmm_connected` rows tagged `cfg_milestone=d1`.

Best WER per setting (lowest over LM weights):

| SNR | Acoustic only | Matched LM | Mismatched LM |
|---|---:|---:|---:|
| Clean | 1.6% | 1.0% | 2.0% |
| 20 dB | 14.8% | 8.2% | 9.2% |
| 15 dB | 37.7% | 18.9% | 27.3% |
| 10 dB | 56.5% | 35.1% | 44.3% |

What we see:
- On clean audio the acoustic model is already good, so the LM brings almost nothing.
- The noisier the audio, the more the LM helps. At 15 dB the matched LM divides the WER by two.
- A matched LM keeps improving up to weight 32-64. A mismatched LM has an optimum around 16, and at weight 64 it gets worse than acoustic-only (61% against 55% at 10 dB), because the LM pushes the decoder toward sequences that are not in the data.
- Our results come from one random draw of utterances and one set of FSDD recordings, so small differences should not be over-interpreted.

## Why does the decoding graph grow with N?

In a bigram decoder the state we leave from tells us the previous word, so the graph only needs one copy of each word HMM. With a trigram we must know the two previous words, so we need one copy of each word for each previous word. With V words this gives about V copies for a bigram, V^2 for a trigram, and V^(N-1) for an N-gram, and every copy has its own HMM states. The Viterbi search then has to handle many more states, so it is slower and uses more memory. This is also why a higher order is not always worth it: the model is bigger and needs more data to estimate (see the n-gram sparsity results above).

## Beam pruning evaluation

Beam pruning limits the decoder to promising HMM states at each audio frame. The **active states** column is the average number of states retained for the next frame. This reduces the search space in principle, while increasing the risk that the correct path is discarded. The experiment compared exact decoding with score-threshold beams and top-N state limits on 150 test utterances per condition (clean and 15 dB SNR); insertion penalties were tuned on a separate 30-utterance development set. Results are in `results/speech_classical/beam_search.csv` and the matching `gmm_beam` records in `results/results.jsonl`.

| Condition and setting | WER | Active states/frame | Search errors | Viterbi time (ms/utterance) |
|---|---:|---:|---:|---:|
| Clean, exact | 0.82% | 82.0 | 0% | 3.68 |
| Clean, score beam 250 | 0.82% | 19.8 | 0% | 4.02 |
| Clean, score beam 225 | 0.82% | 16.0 | 5.8% | 4.01 |
| Clean, top-N 32 | 0.82% | 31.9 | 0% | 4.45 |
| 15 dB, exact | 16.97% | 82.0 | 0% | 3.68 |
| 15 dB, score beam 400 | 16.97% | 65.8 | 0% | 4.01 |
| 15 dB, score beam 350 | 16.97% | 55.3 | 0.8% | 4.06 |
| 15 dB, top-N 48 | 16.97% | 47.7 | 2.5% | 4.41 |

On clean audio, a score beam of 250 cuts the active-state count by about three quarters without changing WER or losing the exact best-scoring path. Tightening it to 225 retains even fewer states but begins to prune paths that exact decoding would choose. At 15 dB, the acoustic condition is harder, and the tested beams retain more states to keep WER near the exact baseline. Search errors indicate the percentage of utterances where pruning loses the exact decoder's best-scoring path; they do not directly indicate WER errors. For example, a pruned path can have a lower model score yet produce the same recognized digits.

Every tested pruned setting took longer than exact decoding in this implementation, even when its active-state count was much lower. The key reason is when pruning happens: for each frame, `decode_emissions` first forms the dense transition-candidate matrix for all source and destination states and finds the best incoming transition for every destination. It then applies the beam or top-N pruning. States set to negative infinity by the previous frame still occupy entries in that dense calculation, so fewer retained states do not proportionally reduce the main matrix work. Pruning also adds score comparisons, top-N selection, and array handling. With a small active set, that extra work can be a larger share of the total, making the pruned run slower. To turn fewer active states into a speedup, decoding would need to expand transitions only from surviving states (or otherwise use a sparse transition structure) before doing the expensive scoring work.

## Limits and next measurements

- The standard split allows speaker overlap. Run `--split unseen_speaker` to assess generalization to a voice absent from training.
- Accuracy differences are based on one 300-recording test set and should be interpreted at that resolution.
- The generated measurements contain mean `lat_ms`, but no latency tail percentile such as `lat_p90_ms`, which the submission specification asks for.
- The plots are generated (`isolated_digits.png`, `connected_digits_lm.png`). The LM weight choice is recorded in `DECISIONS.md` (D1).
