"""The day-1 entries of the live demo (python -m asr_demo.server): menu order, prompts, and how a stub is reported.
These tests do not use your code; they pass from the start."""
import pytest

from asr_demo import backends, digits

pytestmark = pytest.mark.d1
DAY1 = ["digits/0-dtw-baseline", "digits/1-gmm", "digits/2-hmm", "digits/3-word-loop", "digits/4-word-loop+bigram"]


def test_day1_entries_come_first_in_the_menu():
    assert list(backends.REGISTRY)[:5] == DAY1
    assert all(b["has_prompt"] for b in backends.available()[:5])


def test_prompts_are_digit_words():
    for name, n_words in [(DAY1[0], (1, 1)), (DAY1[3], (3, 5)), (DAY1[4], (3, 5))]:
        words = backends.prompt(name).split()
        assert n_words[0] <= len(words) <= n_words[1] and set(words) <= set(digits.WORDS), (name, words)
    assert backends.prompt("hf/whisper-tiny.en") is None  # the page then uses its own English prompts


def _stub(x):
    raise NotImplementedError("fake stub message")


def _implemented(x):
    return x.attribute_that_a_dummy_argument_lacks


def test_a_stub_is_reported_by_name_before_any_training():
    with pytest.raises(NotImplementedError) as e:
        digits.needs(_stub, None)
    msg = backends.explain(e.value)
    assert "test_d1_demo.py::_stub" in msg and "fake stub message" in msg and "restart" in msg
    digits.needs(_implemented, None)  # an implemented function failing on the dummy argument is fine
