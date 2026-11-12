import numpy as np
import pytest

from speech.gmm_hmm import WordLoopDecoder, recognise_isolated, train_bigram, train_word_models

pytestmark = pytest.mark.d1
MEANS = {"a": 3.0, "b": -3.0, "sil": 0.0}


def frames(word, n, rng):
    return rng.normal(MEANS[word], 0.5 if word != "sil" else 0.2, (n, 2))


@pytest.fixture(scope="module")
def toy_models():
    rng = np.random.default_rng(0)
    train = {w: [frames(w, int(rng.integers(8, 15)), rng) for _ in range(20)] for w in MEANS}
    return train_word_models(train, n_states=3, n_mix=1, n_iter=5)


def test_bigram_rows_sum_to_one():
    lm = train_bigram([["a", "b"], ["a", "a"]], ["a", "b"])
    for u, row in lm.items():
        assert abs(sum(row.values()) - 1) < 1e-9, u
    assert lm["a"]["b"] > 0 and lm["b"]["b"] > 0  # smoothing: unseen bigrams keep some mass


def test_isolated(toy_models):
    rng = np.random.default_rng(1)
    assert recognise_isolated(toy_models, frames("a", 10, rng), ["a", "b"]) == "a"
    assert recognise_isolated(toy_models, frames("b", 10, rng), ["a", "b"]) == "b"


def test_connected_decoding(toy_models):
    rng = np.random.default_rng(2)
    seq = ["a", "b", "a"]
    X = np.vstack([frames("sil", 5, rng)] + [np.vstack([frames(w, 10, rng), frames("sil", 5, rng)]) for w in seq])
    dec = WordLoopDecoder(toy_models, ["a", "b", "sil"], None, word_penalty=-5.0)
    assert dec.decode(X) == seq


def test_decoder_uses_the_beam(toy_models):
    """decode_emissions must pass delta through self.prune at every frame (see its docstring)."""
    rng = np.random.default_rng(2)
    X = np.vstack([frames("sil", 5, rng), frames("a", 10, rng), frames("sil", 5, rng), frames("b", 10, rng)])
    exact = WordLoopDecoder(toy_models, ["a", "b", "sil"], None, word_penalty=-5.0)
    assert exact.decode(X) == ["a", "b"]
    assert len(exact.n_active) == len(X) and max(exact.n_active) > 1  # one prune() call per frame; no beam
    greedy = WordLoopDecoder(toy_models, ["a", "b", "sil"], None, word_penalty=-5.0, n_beam=1)
    greedy.decode(X)
    assert greedy.n_active == [1] * len(X)
    wide = WordLoopDecoder(toy_models, ["a", "b", "sil"], None, word_penalty=-5.0, beam=1e6)
    assert wide.decode(X) == ["a", "b"] and wide.final_score == exact.final_score
