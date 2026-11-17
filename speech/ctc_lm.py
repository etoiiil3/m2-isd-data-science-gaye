"""Stage 2 of the speech track: the same acoustic model, with and without language information.

    acoustic only : CTC model (wav2vec2-style) + greedy argmax decoding
    + search      : CTC prefix beam search, no LM (isolates the effect of search)
    + n-gram LM   : beam search with KenLM, sweeping alpha (LM weight) and beta (word bonus)
                    -- the same two knobs as the GMM-HMM word-loop decoder of stage 1.

Install:  uv sync --extra lm        (pyproject.toml ignores pyctcdecode's numpy<2 pin; kenlm is compiled: README.md)
          or, with pip:  pip install -r requirements.txt && pip install --no-deps pyctcdecode==0.5.0 && pip install kenlm
          (pyctcdecode's metadata pins numpy<2 and would DOWNGRADE numpy, breaking librosa/scipy; it works fine
          with numpy 2, hence --no-deps)
LM:       LibriSpeech n-gram ARPA files from https://www.openslr.org/11 (the pruned 3-gram is small),
          or write_arpa_bigram() below for a toy LM trained on any text.

    python -m speech.ctc_lm --selftest                      # synthetic posteriors, runs on CPU in seconds
    python -m speech.ctc_lm --arpa data/lm/3-gram.pruned.1e-7.arpa   # LibriSpeech, wav2vec2-base-100h

Protocol of main: logits of the 200 utterances computed once; alpha x beta x beam swept on the first half (dev);
the best point is decoded once on the other half (test). Rows go to results/results.jsonl (task "ctc_decoding",
cfg_split dev or test). Defaults: --librispeech data/LibriSpeech, --device cuda if available else cpu.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common.bench import default_device


# --------------------------------------------------------------------------- #
def greedy_ctc(logprobs: np.ndarray, labels: list[str], blank: int = 0) -> str:
    arg_max = logprobs.argmax(axis=-1)
    # Collapse consecutive repeats before removing blanks (A, blank, A -> AA).
    arg_max = arg_max[np.r_[True, arg_max[1:] != arg_max[:-1]]]
    arg_max = arg_max[arg_max != blank]
    return "".join(labels[i] for i in arg_max).replace("|", " ").strip()  # '|' = space


def labels_from_processor(processor) -> tuple[list[str], int]:
    """pyctcdecode wants labels ordered by id, blank as '' and word delimiter as ' '."""
    vocab = processor.tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, i in vocab.items():
        labels[i] = tok
    blank = processor.tokenizer.pad_token_id
    labels = [
        "" if i == blank else (" " if t == "|" else t.lower())
        for i, t in enumerate(labels)
    ]
    return labels, blank


def write_arpa_bigram(sentences: list[list[str]], path: Path, discount: float = 0.75):
    """Interpolated absolute-discounting bigram in ARPA format (exact backoff semantics)."""
    uni, bi = Counter(), Counter()
    for s in sentences:
        toks = ["<s>"] + s + ["</s>"]
        uni.update(toks)
        bi.update(zip(toks[:-1], toks[1:]))
    total = sum(v for k, v in uni.items() if k != "<s>")
    p_uni = {w: c / total for w, c in uni.items() if w != "<s>"}
    p_uni["<unk>"] = 1e-6
    follow, n_types = (
        Counter(),
        Counter(),
    )  # total successor tokens / distinct successor types
    for (u, _), c in bi.items():
        follow[u] += c
        n_types[u] += 1
    lines_bi, bo = [], {}
    for u in uni:
        if u == "</s>":
            continue
        lam = discount * n_types[u] / follow[u] if follow[u] else 1.0
        bo[u] = lam
    for (u, w), c in bi.items():
        p = max(c - discount, 0) / follow[u] + bo[u] * p_uni[w]
        lines_bi.append(f"{math.log10(p):.6f}\t{u} {w}")
    with open(
        path, "w", encoding="utf-8", newline="\n"
    ) as f:  # LF on Windows too: kenlm reads the ARPA
        f.write("\\data\\\n")
        f.write(f"ngram 1={len(p_uni) + 1}\nngram 2={len(lines_bi)}\n\n\\1-grams:\n")
        f.write(f"-99\t<s>\t{math.log10(bo['<s>']):.6f}\n")
        for w, p in p_uni.items():
            b = f"\t{math.log10(bo[w]):.6f}" if w in bo else ""
            f.write(f"{math.log10(p):.6f}\t{w}{b}\n")
        f.write("\n\\2-grams:\n" + "\n".join(lines_bi) + "\n\n\\end\\\n")


def make_decoder(labels, arpa: str | None, alpha: float, beta: float):
    from pyctcdecode import build_ctcdecoder

    return build_ctcdecoder(labels, kenlm_model_path=arpa, alpha=alpha, beta=beta)


def sweep(
    logprobs_list, refs, labels, blank, arpa, alphas, betas, beams, unigrams=None
):
    """Decode every utterance with each decoder; one row (a dict) per decoder configuration, in this order:
      - decoder="greedy (acoustic only)", alpha=0, beta=0, beam=1;
      - decoder="beam, no LM", alpha=0, beta=0, for each beam in beams;
      - decoder="beam + n-gram", for each alpha x beta x beam, only when arpa is given.
    Keys: decoder, alpha, beta, beam, wer (jiwer.wer over all utterances; an empty hypothesis counts as "<e>"),
    ms_per_utt (decoding time per utterance). The decoder names are used by the tests and the plots: keep them.
    unigrams: unused (kept for compatibility)."""
    raise NotImplementedError(
        "TODO(d2): rows for greedy, beam without LM (each beam width) and beam + KenLM (alpha x beta x beam): WER and ms per utterance"
    )


# --------------------------------------------------------------------------- #
def synthetic_posteriors(
    text: str, labels: list[str], rng, p_ok: float = 0.6, sharp: float = 3.0
):
    """Fake CTC output: each char held 2-4 frames, blanks in between; with prob 1-p_ok the char's
    frames favour a random wrong char. Lets students test decoders without any acoustic model.
    """
    idx = {c: i for i, c in enumerate(labels)}
    seq = []
    for c in text:
        k = idx[c]
        n = int(rng.integers(2, 5))
        wrong = rng.random() > p_ok
        alt = int(rng.integers(2, len(labels))) if wrong else k
        for _ in range(n):
            seq.append((k, alt))
        seq.append((0, 0))
    V = len(labels)
    lp = []
    for k, alt in seq:
        z = rng.normal(0, 1, V)
        z[alt] += sharp
        z[k] += sharp * (
            0.85 if alt != k else 1.0
        )  # the right char is still close -> LM can recover it
        lp.append(z - np.logaddexp.reduce(z))
    return np.array(lp, dtype=np.float32)


def selftest(out: Path):
    import pandas as pd

    rng = np.random.default_rng(0)
    vocab = (
        "the cat dog bird sat ran flew on under over a red big small mat tree house roof "
        "garden near in"
    ).split()
    nouns = ["cat", "dog", "bird", "mat", "tree", "house", "roof", "garden"]

    def sent():  # noqa: E306
        return [
            "the",
            rng.choice(["red", "big", "small"]),
            rng.choice(nouns[:3]),
            rng.choice(["sat", "ran", "flew"]),
            rng.choice(["on", "under", "over", "near", "in"]),
            "the",
            rng.choice(nouns[3:]),
        ]

    corpus = [[str(w) for w in sent()] for _ in range(3000)]
    out.mkdir(parents=True, exist_ok=True)
    arpa = out / "toy_bigram.arpa"
    write_arpa_bigram(corpus, arpa)
    labels = ["", " "] + sorted(set("".join(vocab)))
    tests = [" ".join(str(w) for w in sent()) for _ in range(60)]
    rows = []
    for p_ok in (0.9, 0.75, 0.6):
        lps = [synthetic_posteriors(t, labels, rng, p_ok) for t in tests]
        for r in sweep(
            lps,
            tests,
            labels,
            0,
            str(arpa),
            alphas=[0.5, 1.0, 2.0],
            betas=[0.0, 1.5],
            beams=[1, 10, 50],
        ):
            rows.append(dict(char_acc_sim=p_ok, **r))
    df = pd.DataFrame(rows)
    df.to_csv(out / "ctc_lm_selftest.csv", index=False)
    best = df.loc[df.groupby(["char_acc_sim", "decoder"]).wer.idxmin()]
    print(
        best[["char_acc_sim", "decoder", "alpha", "beta", "beam", "wer", "ms_per_utt"]]
        .round(3)
        .to_string(index=False)
    )
    return df


# --------------------------------------------------------------------------- #
def librispeech_logits(root, model_id, n, device):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from speech.data import load_librispeech

    proc = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    data = load_librispeech(root, n=n)
    lps = []
    with torch.inference_mode():
        for d in data:
            x = proc(
                d["audio"], sampling_rate=d["sr"], return_tensors="pt"
            ).input_values.to(device)
            lps.append(torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy())
    labels, blank = labels_from_processor(proc)
    return lps, [d["text"].lower() for d in data], labels, blank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--librispeech", type=Path, default=Path("data/LibriSpeech"))
    ap.add_argument("--model", default="facebook/wav2vec2-base-100h")
    ap.add_argument("--arpa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", type=Path, default=Path("results/speech_ctc_lm"))
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.out)
    import pandas as pd
    from common.bench import BenchRecord, log_record

    lps, refs, labels, blank = librispeech_logits(
        args.librispeech, args.model, args.n, args.device
    )
    half = (
        len(refs) // 2
    )  # the utterances are already shuffled (load_librispeech): dev = first half, test = the rest
    dev = sweep(
        lps[:half],
        refs[:half],
        labels,
        blank,
        args.arpa,
        alphas=[0.3, 0.5, 1.0, 2.0],
        betas=[0.0, 1.0, 2.0],
        beams=[10, 50, 100],
    )
    best = min(
        (r for r in dev if r["decoder"] != "greedy (acoustic only)"),
        key=lambda r: r["wer"],
    )
    print(f"best on dev ({half} utterances): {best}")
    test = sweep(
        lps[half:],
        refs[half:],
        labels,
        blank,
        args.arpa if best["decoder"] == "beam + n-gram" else None,
        alphas=[best["alpha"]],
        betas=[best["beta"]],
        beams=[best["beam"]],
    )
    rows = [dict(split="dev", **r) for r in dev] + [
        dict(split="test", **r) for r in test
    ]
    print(pd.DataFrame(rows[-len(test) :]).round(4).to_string(index=False))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.out / f"ctc_lm_{args.model.split('/')[-1]}.csv", index=False
    )
    for r in rows:
        log_record(
            BenchRecord(
                "ctc_decoding",
                f"{args.model}|{r['decoder']}",
                dict(
                    alpha=r["alpha"],
                    beta=r["beta"],
                    beam=r["beam"],
                    split=r["split"],
                    n=half if r["split"] == "dev" else len(refs) - half,
                    device=args.device,
                ),
                dict(wer=r["wer"], ms_per_utt=r["ms_per_utt"]),
            ),
            args.jsonl,
        )


if __name__ == "__main__":
    main()
