"""The refusal numbers in the README are a fixture, not a fresh run.

CI has no Ollama, so it cannot embed a question. What it can do is pin the scores the
gate was measured on and fail if the threshold, or the held-out result, drifts.
"""

import sys

from eval.refusal import REFUSE_BELOW

# cosine of the closest paragraph, held-out twenty, measured 2026-10-25
HELD_OUT = [
    0.561,
    0.658,
    0.566,
    0.502,
    0.546,
    0.639,
    0.565,
    0.681,
    0.626,
    0.610,
    0.575,
    0.644,
    0.552,
    0.621,
    0.586,
    0.579,
    0.623,
    0.530,
    0.622,
    0.586,
]
ANSWERABLE_MIN = 0.584  # lowest score over the 60 tuned and 40 blind questions


def test_the_fitted_line_is_the_one_quoted():
    assert REFUSE_BELOW == 0.58


def test_held_out_half_still_catches_nine_of_twenty():
    caught = [score for score in HELD_OUT if score < REFUSE_BELOW]
    assert len(caught) == 9


def test_the_line_sits_below_every_answerable_question():
    assert REFUSE_BELOW < ANSWERABLE_MIN


def test_the_module_is_importable_without_a_model():
    assert "app.refusal" not in sys.modules
