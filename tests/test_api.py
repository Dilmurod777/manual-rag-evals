from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def test_short_question_is_rejected_before_search():
    assert client.post("/api/ask", json={"q": "hi"}).status_code == 422


def test_refusal_does_not_call_the_model(monkeypatch):
    monkeypatch.setattr(main, "confidence", lambda question: 0.10)
    calls = []
    monkeypatch.setattr(
        main, "answer", lambda q, ranked: calls.append(q) or ("x", False)
    )
    body = client.post("/api/ask", json={"q": "What is the takeoff weight?"}).json()
    assert body["vector"]["refused"] is True
    assert calls == []


def test_grounded_is_false_when_the_page_was_not_shown(monkeypatch):
    monkeypatch.setattr(main, "confidence", lambda question: 0.90)
    monkeypatch.setattr(main, "answer", lambda q, ranked: ("Yes. (Page 99-99)", False))
    body = client.post("/api/ask", json={"q": "circuit breaker switch"}).json()
    assert body["vector"]["grounded"] is False


def test_stream_sends_done_for_every_arm(monkeypatch):
    monkeypatch.setattr(main, "confidence", lambda question: 0.90)
    monkeypatch.setattr(main, "answer", lambda q, ranked: ("Yes. (Page 11-16)", False))
    with client.stream(
        "POST", "/api/ask/stream", json={"q": "circuit breaker switch"}
    ) as reply:
        body = "".join(reply.iter_text())
    assert reply.status_code == 200
    assert body.count('"kind": "done"') == 3
    assert "grounded" in body
