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


def test_malformed_json_is_400(indexed):
    with _client(indexed["cfg"]) as client:
        resp = client.post(
            "/v1/chat/completions",
            content=b"{broken",
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 400


def test_internal_error_generic_message(indexed, monkeypatch):
    """Детали исключения не утекают пользователю."""
    import codeqa.backend.app as app_mod

    def boom(*a, **kw):
        raise RuntimeError("секретный путь /data/index.sqlite сломан")

    monkeypatch.setattr(app_mod, "answer_question", boom)
    with _client(indexed["cfg"]) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [
                {"role": "user", "content": "calculate_total в биллинге"}
            ]},
        )
    content = resp.json()["choices"][0]["message"]["content"]
    assert "Внутренняя ошибка codeqa" in content
    assert "index.sqlite" not in content


def test_question_after_clarification_keeps_original():
    """После ответа «1» на уточнение в ретрив уходит исходный вопрос."""
    from codeqa.backend.app import _question_for_retrieval

    history = [
        {"role": "user", "content": "где считается налог?"},
        {"role": "assistant", "content": "Не могу однозначно... Уточните проект:\n\n1) **billing**\n2) **shop**"},
        {"role": "user", "content": "1"},
    ]
    assert _question_for_retrieval(history) == "где считается налог?"
    # без уточнения — вопрос как есть
    assert _question_for_retrieval(history[:1]) == "где считается налог?"
    # пустая история
    assert _question_for_retrieval([]) == ""
