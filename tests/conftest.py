"""The API tests must not need a 24 MB embedding cache or a running model."""

import pytest

import app.main as main
from app.corpus import paragraphs

CHUNKS = list(paragraphs())


class FakeVector:
    def search(self, question, k=5):
        return [(0.9 - i * 0.01, chunk) for i, chunk in enumerate(CHUNKS[:k])]


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch):
    monkeypatch.setattr(main, "vector", FakeVector())
    monkeypatch.setattr(main, "reranked", lambda q, k=5: FakeVector().search(q, k))
