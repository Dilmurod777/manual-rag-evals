"""Three checks that fail when the parsers or the ranks break."""

from app.corpus import paragraphs
from app.main import evidence
from app.search import BM25


def test_torque_page_comes_first():
    chunks = list(paragraphs())
    ranked = BM25(chunks).search("torque values for AN bolts", 1)
    assert ranked[0][1]["page"] == "7-9"


def test_evidence_marks_the_answering_sentence():
    chunks = list(paragraphs())
    ranked = BM25(chunks).search("circuit breaker as a switch", 5)
    marked = [
        s["text"] for s in evidence("circuit breaker", ranked)["sentences"] if s["hit"]
    ]
    assert any("switch" in text and "life" in text for text in marked)


def test_corpus_pages_are_citable():
    chunks = list(paragraphs())
    assert len(chunks) == 1758
    assert all(
        c["page"].count("-") == 1 or c["page"].startswith("pdf-") for c in chunks
    )
