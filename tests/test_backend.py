"""Тесты этапа 5: OpenAI-совместимый бэкенд."""

from contextlib import contextmanager

from fastapi.testclient import TestClient

from codeqa.backend import create_app


@contextmanager
def _client(cfg):
    with TestClient(create_app(cfg)) as client:  # with — срабатывает shutdown
        yield client


def test_models_and_health(indexed):
    with _client(indexed["cfg"]) as client:
        assert client.get("/health").json()["status"] == "ok"
        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "codeqa-assistant"


def test_chat_answer_with_alias(indexed):
    with _client(indexed["cfg"]) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "codeqa-assistant",
                "messages": [
                    {"role": "user", "content": "как работает calculate_total в биллинге?"}
                ],
            },
        )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "MOCK-ANSWER" in content      # ответ прошёл через mock LLM
    assert "Источники" in content        # приложены ссылки на код
    assert "main.py" in content


def test_chat_clarification_then_number(indexed):
    with _client(indexed["cfg"]) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "что это?"}]},
        )
        content = resp.json()["choices"][0]["message"]["content"]
        assert "Уточните проект" in content

        history = [
            {"role": "user", "content": "что это?"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "1"},
        ]
        resp2 = client.post("/v1/chat/completions", json={"messages": history})
        content2 = resp2.json()["choices"][0]["message"]["content"]
        assert "MOCK-ANSWER" in content2     # проект определён, ответ получен


def test_chat_streaming(indexed):
    with _client(indexed["cfg"]) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "как устроен магазин?"}],
                "stream": True,
            },
        ) as resp:
            body = "".join(resp.iter_text())
    assert "data:" in body
    assert "[DONE]" in body
    assert "MOCK-ANSWER" in body
