"""N-gram language models measured ON THEIR OWN (before any decoding): perplexity vs order N,
training-set size and smoothing. Day 1 morning.

    lm = NGramLM(order=3, smoothing="interpolated").fit(train_sentences, vocab)
    lm.perplexity(test_sentences)

The lesson: higher N captures longer dependencies but sees each history less often (sparsity), so the
best N depends on the amount of text and on smoothing. The decoding graph also grows with N (lecture).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

BOS, EOS = "<s>", "</s>"


class NGramLM:
    def __init__(self, order: int = 2, smoothing: str = "add-k", k: float = 0.1, lambdas: list | None = None):
        assert order >= 1 and smoothing in ("add-k", "interpolated")
        self.order, self.smoothing, self.k = order, smoothing, k
        # interpolation weights for orders N, N-1, ..., 1 (sum to 1)
        self.lambdas = lambdas or [0.6, 0.3, 0.1, 0.05, 0.05][:order]
        self.lambdas = [w / sum(self.lambdas) for w in self.lambdas]

    def fit(self, sentences: list[list[str]], vocab: list[str]):
        """Count n-grams of every order 1..N. Each sentence is padded with N-1 BOS and one EOS.
        self.counts[n][ngram_tuple] = count, self.context[n][history_tuple] = count (n = 1..N)."""
        # BOS is context only; EOS is a predictable token. Preserve the input
        # vocabulary order while removing duplicates and reserved symbols.
        self.vocab = list(dict.fromkeys(w for w in vocab if w not in (BOS, EOS))) + [EOS]
        vocab_set = set(self.vocab)
        for sentence in sentences:
            unknown = set(sentence) - vocab_set
            if unknown:
                raise ValueError(f"training sentence contains tokens outside vocab: {sorted(unknown)}")

        self.counts = {n: Counter() for n in range(1, self.order + 1)}
        self.context = {n: Counter() for n in range(1, self.order + 1)}

        for sentence in sentences:
            padded = [BOS] * (self.order - 1) + list(sentence) + [EOS]
            # Each position after the BOS padding is a prediction target,
            # including EOS. Shorter orders use the suffix of the same history.
            for end in range(self.order - 1, len(padded)):
                for n in range(1, self.order + 1):
                    history = tuple(padded[end - n + 1:end]) if n > 1 else ()
                    ngram = history + (padded[end],)
                    self.counts[n][ngram] += 1
                    self.context[n][history] += 1
        return self

    def prob(self, word: str, history: list[str]) -> float:
        """P(word | last N-1 tokens of history). history may contain BOS tokens."""
        if not hasattr(self, "counts"):
            raise RuntimeError("fit must be called before prob")
        if word not in self.vocab:
            return 0.0

        vocab_size = len(self.vocab)

        def add_k_estimate(n: int) -> float:
            # At the beginning of a sentence, restore missing left context
            # with BOS markers, just as fit() did when counting the corpus.
            hlen = n - 1
            h = list(history)
            if len(h) < hlen:
                h = [BOS] * (hlen - len(h)) + h
            h = tuple(h[-hlen:]) if hlen else ()

            numerator = self.counts[n].get(h + (word,), 0) + self.k
            denominator = self.context[n].get(h, 0) + self.k * vocab_size
            # With k=0 an entirely unseen context has no empirical estimate;
            # use a uniform estimate to keep the distribution defined.
            if denominator == 0:
                return 1.0 / vocab_size
            return numerator / denominator

        if self.smoothing == "add-k":
            return add_k_estimate(self.order)

        # self.lambdas is ordered N, N-1, ..., 1.
        return sum(self.lambdas[self.order - n] * add_k_estimate(n)
                   for n in range(self.order, 0, -1))

    def sentence_logprob(self, sentence: list[str]) -> float:
        hist, lp = [], 0.0
        for w in list(sentence) + [EOS]:
            p = self.prob(w, hist)
            if p <= 0.0:
                return -math.inf
            lp += math.log(p)
            hist.append(w)
        return lp

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(-(sum of log P) / number of predicted tokens), EOS included."""
        total_logprob = 0.0
        n_predicted = 0
        for sentence in sentences:
            # Each sentence predicts its words and one end-of-sentence token.
            n_predicted += len(sentence) + 1
            sentence_lp = self.sentence_logprob(sentence)
            if sentence_lp == -math.inf:
                return math.inf
            total_logprob += sentence_lp

        if n_predicted == 0:
            raise ValueError("perplexity requires at least one predicted token")
        return math.exp(-total_logprob / n_predicted)

    @property
    def n_entries(self) -> int:
        return sum(len(c) for c in self.counts.values())

    def as_bigram(self, vocab: list[str]) -> dict:
        """Bigram table {u: {v: P(v|u)}} for the word-loop decoder (uses this model's own smoothing)."""
        rows = {}
        for u in [BOS] + list(vocab):
            p = {v: self.prob(v, [u]) for v in vocab}
            z = sum(p.values())
            rows[u] = {v: x / z for v, x in p.items()}
        return rows


def second_order_chain(rng, n_symbols: int = 10, concentration: float = 0.3):
    """A digit 'grammar' where the next digit depends on the previous TWO: a trigram has something to
    learn that a bigram cannot. Returns P[a, b, c] = P(c | a, b), with index n_symbols = start."""
    P = rng.dirichlet(np.full(n_symbols, concentration), size=(n_symbols + 1, n_symbols + 1))
    return P


def sample_second_order(rng, P, n: int, lmin: int = 4, lmax: int = 9) -> list[list[str]]:
    S = P.shape[-1]
    out = []
    for _ in range(n):
        a = b = S  # start symbols
        s = []
        for _ in range(int(rng.integers(lmin, lmax + 1))):
            c = int(rng.choice(S, p=P[a, b]))
            s.append(str(c))
            a, b = b, c
        out.append(s)
    return out
