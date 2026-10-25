"""The eval scripts make claims; these check the little pieces that turn text into them."""

from eval.judge import parse_reply
from eval.run_eval import wilson


def test_verdict_is_read_from_the_reply():
    verdict, why = parse_reply("VERDICT: SUPPORTED\nWHY: the sentence is on the page")
    assert verdict == "SUPPORTED"
    assert why == "the sentence is on the page"


def test_verdict_survives_chatter_around_it():
    assert parse_reply("Sure. VERDICT: PARTIAL because it stops short")[0] == "PARTIAL"


def test_an_unparseable_reply_is_not_silently_counted():
    verdict, why = parse_reply("the passage seems relevant")
    assert verdict == "UNCLEAR"
    assert why == "the passage seems relevant"


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson(55, 60)
    assert 0 <= low < 55 / 60 < high <= 1


def test_wilson_handles_a_perfect_and_an_empty_result():
    assert wilson(60, 60)[1] == 1.0
    assert wilson(0, 10)[0] == 0.0
