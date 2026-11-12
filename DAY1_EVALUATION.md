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

## Limits and next measurements

- The standard split allows speaker overlap. Run `--split unseen_speaker` to assess generalization to a voice absent from training.
- Accuracy differences are based on one 300-recording test set and should be interpreted at that resolution.
- The generated measurements contain mean `lat_ms`, but no latency tail percentile such as `lat_p90_ms`, which the submission specification asks for.
- Generate the plot artifacts with `--part plot` after connected-digit results exist; the plotting command also reads the connected-digit CSV. Record any model choice and its rationale in the appropriate Day 1 decision entry in `DECISIONS.md`.
