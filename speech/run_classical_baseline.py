"""Stage 1 of the speech track: acoustic-only GMM-HMM, then the same acoustic model + a bigram LM.

Data: Free Spoken Digit Dataset (git clone https://github.com/Jakobovski/free-spoken-digit-dataset)
      3000 wav @ 8 kHz, 6 speakers, files '<digit>_<speaker>_<idx>.wav'; idx 0-4 = official test split.

Experiments
  A. Isolated digits: sweep (n_states, n_mix) -> accuracy vs latency vs #params  => a first Pareto front,
     plus a speaker-held-out condition and a noise condition (limits of a purely acoustic model).
  B. Connected digits: concatenate test words into utterances whose word order follows a Markov chain
     ("structured" domain, e.g. codes/phone numbers). Decode with
        - uniform LM (acoustic only)
        - bigram LM trained on TEXT ONLY from the same domain, sweeping the LM weight
        - the same bigram on a mismatched domain (uniformly random digit strings)
     at several SNRs. Shows: LM helps more when acoustics degrade; too much LM weight hurts;
     a mismatched LM hurts.
  C. Beam pruning (--part beam): the same matched test utterances, decoded with a score beam and a top-N beam
     (WordLoopDecoder(beam=..., n_beam=...)). WER vs active states per frame, and how often the beam causes a
     search error. Shows: half the states are free, then a cliff; the safe width depends on the condition.

    # each part is independent and appends to CSVs in --out (handy on small machines)
    python -m speech.run_classical_baseline --fsdd fsdd/recordings --part isolated --split standard
    python -m speech.run_classical_baseline --fsdd fsdd/recordings --part isolated --split unseen_speaker
    python -m speech.run_classical_baseline --fsdd fsdd/recordings --part connected
    python -m speech.run_classical_baseline --fsdd fsdd/recordings --part beam
    python -m speech.run_classical_baseline --fsdd fsdd/recordings --part plot
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd

from common.bench import BenchRecord, log_record
from common.pareto import plot_pareto
from speech.degrade import degrade
from speech.ngram import NGramLM, sample_second_order, second_order_chain
from speech.gmm_hmm import (Normaliser, WordLoopDecoder, add_noise, mfcc_features, recognise_isolated,
                            train_bigram, train_word_models)

DIGITS = [str(d) for d in range(10)]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
SR = 8000
TARGET_RMS = 0.05
GAP_STD = 5e-4  # -40 dB "silence" between words
LM_WEIGHTS = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)  # lambda grid of --part connected
PENALTIES = (-600.0, -400.0, -300.0, -200.0, -150.0, -100.0, -60.0, -30.0, -15.0, 0.0)  # beta grid, tuned on dev
BEAM_CONDITIONS = ((None, 32), (15, 64))  # (SNR, lambda) of --part beam: the best lambda of each condition in --part connected
BEAMS = (np.inf, 800, 600, 500, 400, 350, 300, 275, 250, 225, 200, 175, 150, 100, 50, 0)  # score beam Omega
N_BEAMS = (48, 40, 32, 28, 24, 20, 16, 12, 8, 4, 2, 1)  # top-N beam; 1 = greedy
STANDARD_MODEL = (5, 2)  # (n_states, n_mix) saved as gmm_standard.pkl: used by --part connected (the demo uses gmm_S5M2.pkl)


def load_fsdd(root: Path):
    data = []
    for f in sorted(root.glob("*.wav")):
        d, spk, idx = f.stem.split("_")
        y, _ = librosa.load(f, sr=SR)
        # Trim leading/trailing non-speech: word models should model the WORD, and the
        # recording-specific background noise goes to the silence model instead.
        # (Without this, the noisiest-sounding word -- "six" -- absorbs every pause and
        # connected decoding suffers massive insertions: a nice bug to let students find.)
        yt, (a, b) = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
        scale = TARGET_RMS / (np.sqrt(np.mean(yt ** 2)) + 1e-9)
        edges = [e * scale for e in (y[:a], y[b:]) if len(e) > SR * 0.04]
        data.append(dict(digit=d, speaker=spk, idx=int(idx), y=yt * scale, edges=edges))
    return data


def make_silence(rng, n, sr=SR):
    return [rng.normal(0, GAP_STD, int(sr * rng.uniform(0.08, 0.4))) for _ in range(n)]


def markov_chain(rng):
    """Each digit has two 'likely' successors (80 % of the mass)."""
    P = np.full((11, 10), 0.2 / 8)  # row 10 = sentence start
    for u in range(11):
        fav = rng.choice(10, 2, replace=False)
        P[u, fav] = 0.4
    return P / P.sum(1, keepdims=True)


def sample_sequences(rng, P, n, lmin=3, lmax=7, uniform=False):
    out = []
    for _ in range(n):
        L = rng.integers(lmin, lmax + 1)
        s, prev = [], 10
        for _ in range(L):
            d = rng.integers(10) if uniform else rng.choice(10, p=P[prev])
            s.append(str(d))
            prev = d
        out.append(s)
    return out


def build_utterance(seq, pool, rng):
    parts = [rng.normal(0, GAP_STD, int(SR * 0.15))]
    for d in seq:
        parts.append(pool[d][rng.integers(len(pool[d]))])
        parts.append(rng.normal(0, GAP_STD, int(SR * rng.uniform(0.05, 0.25))))
    return np.concatenate(parts)


def to_words(seq):
    return " ".join(WORDS[int(d)] for d in seq)


def prepare(args, rng):
    data = load_fsdd(args.fsdd)
    for d in data:
        d["f_raw"] = mfcc_features(d["y"], SR)
    sil_gaps = [mfcc_features(s, SR) for s in make_silence(rng, 300)]
    return data, sil_gaps


def split_data(data, split, held_out):
    if split == "standard":
        return [d for d in data if d["idx"] >= 5], [d for d in data if d["idx"] < 5]
    return [d for d in data if d["speaker"] != held_out], [d for d in data if d["speaker"] == held_out]


def training_set(tr, sil_gaps):
    sil_train = sil_gaps + [mfcc_features(e, SR) for d in tr for e in d["edges"]]
    norm = Normaliser().fit([d["f_raw"] for d in tr] + sil_train)
    train = defaultdict(list)
    for d in tr:
        train[d["digit"]].append(norm(d["f_raw"]))
    train["sil"] = [norm(f) for f in sil_train]
    return train, norm


def append_rows(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def part_isolated(args, rng):
    data, sil_gaps = prepare(args, rng)
    tr, te = split_data(data, args.split, args.held_out_speaker)
    train, norm = training_set(tr, sil_gaps)
    others = [d["y"] for d in tr]  # babble = other speakers from the TRAINING set
    conds = args.conditions if args.split == "standard" else ["clean"]

    def apply(y, cond):
        kind, _, val = cond.partition(":")
        kw = {"snr_db": float(val)} if kind in ("noise", "babble") and val else {"rt60": float(val)} if kind == "reverb" and val else {}
        return degrade(y, SR, kind, rng=rng, others=others, **kw)
    te_feats = {c: [norm(mfcc_features(apply(d["y"], c), SR)) for d in te] for c in conds}
    audio_s = sum(len(d["y"]) for d in te) / SR
    rows = []
    for n_states in args.states:
        for n_mix in args.mixes:
            wm = train_word_models(train, n_states, n_mix)
            for snr, feats in te_feats.items():
                t1 = time.perf_counter()
                pred = [recognise_isolated(wm, f, DIGITS) for f in feats]
                dt = time.perf_counter() - t1
                rows.append(dict(split=args.split if args.split == "standard" else f"unseen_{args.held_out_speaker}", snr=snr, n_states=n_states,
                                 n_mix=n_mix, acc=float(np.mean([p == d["digit"] for p, d in zip(pred, te)])),
                                 lat_ms=dt / len(feats) * 1000, rtf=dt / audio_s, n_params=wm.n_params,
                                 train_s=wm.train_time_s, name=f"S{n_states}M{n_mix}"))
                print(rows[-1], flush=True)
            if args.split == "standard":  # every config is kept as gmm_S<s>M<m>.pkl
                pickle.dump((wm, norm), open(args.out / f"gmm_S{n_states}M{n_mix}.pkl", "wb"))
                if (n_states, n_mix) == STANDARD_MODEL:  # --part connected and the demo load this one (or DSP_GMM_MODEL)
                    pickle.dump((wm, norm), open(args.out / "gmm_standard.pkl", "wb"))
    append_rows(rows, args.out / "isolated_digits.csv")
    for r in rows:
        log_record(BenchRecord("gmm_isolated", r["name"], dict(split=r["split"], condition=r["snr"], n_states=r["n_states"],
                                                               n_mix=r["n_mix"]),
                               dict(acc=r["acc"], lat_ms=r["lat_ms"], rtf=r["rtf"], n_params=r["n_params"],
                                    train_s=r["train_s"])), args.jsonl)


def connected_setup(args, rng):
    """Shared by --part connected and --part beam (same seed => same models, LM and test utterances)."""
    data, sil_gaps = prepare(args, rng)
    tr, te = split_data(data, "standard", None)
    pk = args.out / "gmm_standard.pkl"
    if pk.exists():
        wm, norm = pickle.load(open(pk, "rb"))
    else:
        train, norm = training_set(tr, sil_gaps)
        wm = train_word_models(train, *STANDARD_MODEL)
    pool = defaultdict(list)
    for d in te:
        pool[d["digit"]].append(d["y"])
    P = markov_chain(rng)
    lm = train_bigram(sample_sequences(rng, P, 5000), DIGITS)  # text-only LM corpus
    return wm, norm, pool, P, lm


def connected_emissions(wm, norm, pool, P, rng, n_utts, domain):
    """(references, {snr: emission matrices}) for one text domain; draws from rng in a fixed order."""
    seqs = sample_sequences(rng, P, n_utts, uniform=(domain == "mismatched"))
    utts = [build_utterance(s, pool, rng) for s in seqs]
    dec0 = WordLoopDecoder(wm, DIGITS + ["sil"], None)
    B = {snr: [dec0.emissions(norm(mfcc_features(add_noise(u, snr, rng), SR))) for u in utts]
         for snr in (None, 20, 15, 10)}
    return [to_words(s) for s in seqs], B


def tune_penalty(wm, vocab, bg, w, B_dev, refs_dev):
    """Best word penalty beta on the dev utterances (exact search)."""
    best = None
    for pen in PENALTIES:
        dec = WordLoopDecoder(wm, vocab, bg, lm_weight=w, word_penalty=pen)
        hyp = [to_words(dec.decode_emissions(b)) or "<empty>" for b in B_dev]
        e = jiwer.wer(refs_dev, hyp)
        if best is None or e < best[0]:
            best = (e, pen)
    return best[1]


def part_connected(args, rng):
    wm, norm, pool, P, lm = connected_setup(args, rng)
    vocab = DIGITS + ["sil"]
    n_dev = 30
    rows = []
    for domain in ("matched", "mismatched"):
        refs, Bs = connected_emissions(wm, norm, pool, P, rng, args.n_utts, domain)
        for snr in (None, 20, 15, 10):
            B = Bs[snr]
            configs = [("acoustic-only", None, 1.0)] + [(f"bigram w={w}", lm, w) for w in LM_WEIGHTS]
            for name, bg, w in configs:
                # word insertion penalty tuned on a dev subset. Its scale must match the acoustic
                # log-likelihoods (tens of nats per frame), not ~1.
                best = (None, tune_penalty(wm, vocab, bg, w, B[:n_dev], refs[:n_dev]))
                dec = WordLoopDecoder(wm, vocab, bg, lm_weight=w, word_penalty=best[1])
                t1 = time.perf_counter()
                hyp = [to_words(dec.decode_emissions(b)) or "<empty>" for b in B[n_dev:]]
                dt = time.perf_counter() - t1
                o = jiwer.process_words(refs[n_dev:], hyp)
                rows.append(dict(domain=domain, snr="clean" if snr is None else snr, decoder=name,
                                 lm_weight=w if bg is not None else 0, penalty=best[1], wer=o.wer,
                                 sub=o.substitutions, dele=o.deletions, ins=o.insertions,
                                 viterbi_ms_per_utt=dt / len(hyp) * 1000))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out / "connected_digits_lm.csv", index=False)
    for r in rows:
        log_record(BenchRecord("gmm_connected", r["decoder"], dict(domain=r["domain"], snr=r["snr"], lm_weight=r["lm_weight"],
                                                                   penalty=r["penalty"]),
                               dict(wer=r["wer"], viterbi_ms_per_utt=r["viterbi_ms_per_utt"])), args.jsonl)


def part_beam(args, rng):
    """Beam pruning on the matched-text test utterances of --part connected (same seed, same data): WER vs the
    number of active states per frame, for a score beam (Omega) and a top-N beam. beta is tuned on dev with
    exact search, as in --part connected. A search error = the pruned decoder ends with a lower score than exact
    search, under the same model."""
    wm, norm, pool, P, lm = connected_setup(args, rng)
    vocab = DIGITS + ["sil"]
    n_dev = 30
    refs, Bs = connected_emissions(wm, norm, pool, P, rng, args.n_utts, "matched")
    rows = []
    for snr, w in BEAM_CONDITIONS:
        pen = tune_penalty(wm, vocab, lm, w, Bs[snr][:n_dev], refs[:n_dev])
        B_test, refs_test = Bs[snr][n_dev:], refs[n_dev:]
        grid = [("exact", np.inf, None)] + [("score", b, None) for b in BEAMS if np.isfinite(b)] + \
               [("top-N", np.inf, n) for n in N_BEAMS]
        exact_scores = None
        for rule, beam, n_beam in grid:
            dec = WordLoopDecoder(wm, vocab, lm, lm_weight=w, word_penalty=pen, beam=beam, n_beam=n_beam)
            hyp, scores = [], []
            t1 = time.perf_counter()
            for b in B_test:
                hyp.append(to_words(dec.decode_emissions(b)) or "<empty>")
                scores.append(dec.final_score)
            dt = time.perf_counter() - t1
            if not dec.n_active:
                raise RuntimeError("decode_emissions never called self.prune(delta): see its docstring")
            exact_scores = exact_scores or scores
            o = jiwer.process_words(refs_test, hyp)
            rows.append(dict(snr="clean" if snr is None else snr, lm_weight=w, penalty=pen, rule=rule,
                             beam=beam, n_beam=n_beam, wer=o.wer, sub=o.substitutions, dele=o.deletions,
                             ins=o.insertions, active_states=float(np.mean(dec.n_active)),
                             search_errors=float(np.mean([s < e - 1e-6 for s, e in zip(scores, exact_scores)])),
                             viterbi_ms_per_utt=dt / len(hyp) * 1000))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out / "beam_search.csv", index=False)
    for r in rows:
        name = r["rule"] if r["rule"] == "exact" else (f"beam={r['beam']:g}" if r["rule"] == "score" else f"n_beam={r['n_beam']}")
        beam = r["beam"] if np.isfinite(r["beam"]) else None  # null = no score beam (JSON has no infinity)
        log_record(BenchRecord("gmm_beam", name, dict(snr=r["snr"], lm_weight=r["lm_weight"], penalty=r["penalty"],
                                                      beam=beam, n_beam=r["n_beam"]),
                               dict(wer=r["wer"], active_states=r["active_states"], search_errors=r["search_errors"],
                                    viterbi_ms_per_utt=r["viterbi_ms_per_utt"])), args.jsonl)


def part_lm(args, rng):
    """Perplexity of n-gram LMs vs order, training size and smoothing, on a 2nd-order digit grammar."""
    P = second_order_chain(rng)
    test = sample_second_order(rng, P, 1000)
    rows = []
    for n_train in args.lm_sizes:
        train = sample_second_order(rng, P, n_train)
        for order in range(1, args.max_order + 1):
            for sm in ("add-k", "interpolated"):
                t0 = time.perf_counter()
                lm = NGramLM(order, sm).fit(train, DIGITS)
                pp = lm.perplexity(test)
                rows.append(dict(n_train=n_train, order=order, smoothing=sm, perplexity=pp, n_entries=lm.n_entries,
                                 ms=(time.perf_counter() - t0) * 1000))
                log_record(BenchRecord("ngram_lm", f"{order}-gram/{sm}", dict(order=order, smoothing=sm, n_train=n_train),
                                       dict(perplexity=pp, n_entries=lm.n_entries)), args.jsonl)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "ngram_perplexity.csv", index=False)
    print(df.pivot_table(index=["smoothing", "n_train"], columns="order", values="perplexity").round(2))


def part_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dfA = pd.read_csv(args.out / "isolated_digits.csv")
    dfA["snr"] = dfA["snr"].astype(str).replace({"None": "clean"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    clean = dfA[(dfA.split == "standard") & (dfA.snr == "clean")].reset_index(drop=True)
    plot_pareto(clean, "lat_ms", "acc", "min", "max", label="name", ax=axes[0],
                title="Isolated digits, seen speakers, clean: accuracy vs latency")
    # 95% binomial half-width: with 300 test files, 1 error = 0.33 pt. Most of the front is noise!
    n_test = 300
    ci = 1.96 * np.sqrt(clean.acc * (1 - clean.acc) / n_test)
    axes[0].errorbar(clean.lat_ms, clean.acc, yerr=ci, fmt="none", ecolor="gray", alpha=0.5, capsize=2)
    comp = dfA[(dfA.n_states == dfA.n_states.max()) & (dfA.n_mix == dfA.n_mix.min())].copy()
    comp["condition"] = comp.split + " / " + comp.snr
    axes[1].bar(comp.condition, comp.acc, color="tab:gray")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Acoustic degradations and speakers (one configuration)")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(comp.acc):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(args.out / "isolated_digits.png", dpi=130)

    dfB = pd.read_csv(args.out / "connected_digits_lm.csv")
    dfB["snr"] = dfB["snr"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dom in zip(axes, ("matched", "mismatched")):
        sub = dfB[dfB.domain == dom]
        for snr, g in sub.groupby("snr", sort=False):
            base = g[g.decoder == "acoustic-only"].wer.values[0]
            gb = g[g.decoder != "acoustic-only"]
            (line,) = ax.plot(gb.lm_weight, gb.wer, "o-", label=f"SNR {snr}")
            ax.axhline(base, ls=":", color=line.get_color(), lw=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("LM weight  (dotted line = acoustic only, same SNR)")
        ax.set_title(f"Connected digits, LM trained on {dom} text domain")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("WER")
    fig.tight_layout()
    fig.savefig(args.out / "connected_digits_lm.png", dpi=130)

    if (args.out / "beam_search.csv").exists():
        dfC = pd.read_csv(args.out / "beam_search.csv")
        fig, ax = plt.subplots(figsize=(7, 5))
        for (snr, rule), g in dfC[dfC.rule != "exact"].groupby(["snr", "rule"], sort=False):
            ax.plot(g.active_states, g.wer, "o-" if rule == "score" else "s--", label=f"SNR {snr}, {rule} beam")
        ex = dfC[dfC.rule == "exact"]
        ax.plot(ex.active_states, ex.wer, "k*", ms=12, label="exact Viterbi")
        ax.set_yscale("log")
        ax.set_xlabel("active states per frame (cost)")
        ax.set_ylabel("WER")
        ax.set_title("Beam pruning on connected digits")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "beam_search.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdd", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/speech_classical"))
    ap.add_argument("--part", choices=["isolated", "lm", "connected", "beam", "plot"], required=True)
    ap.add_argument("--conditions", nargs="+", default=["clean", "noise:10", "babble:5", "reverb:0.6", "telephone"],
                    help="kind[:value] — noise/babble value = SNR dB, reverb value = RT60 s")
    ap.add_argument("--lm_sizes", type=int, nargs="+", default=[50, 500, 5000])
    ap.add_argument("--max_order", type=int, default=5)
    ap.add_argument("--split", choices=["standard", "unseen_speaker"], default="standard")
    ap.add_argument("--states", type=int, nargs="+", default=[2, 3, 5, 8])
    ap.add_argument("--mixes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n_utts", type=int, default=150)
    ap.add_argument("--held_out_speaker", default="theo")
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    {"isolated": part_isolated, "lm": part_lm, "connected": part_connected, "beam": part_beam}.get(args.part, lambda a, r: part_plot(a))(args, rng)
    print(f"[{args.part}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
