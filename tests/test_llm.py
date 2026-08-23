from codeqa.config import LLMConfig
from codeqa.llm import LLMClient
from codeqa.llm.mock_server import EMBED_DIM


def _client(url: str) -> LLMClient:
    return LLMClient(
        LLMConfig(base_url=url, api_key="sk-test", chat_model="mock-qwen", embed_model="mock-qwen")
    )


def _serve(app):
    """FastAPI-приложение на своём порту (для проверки отдельного endpoint'а)."""
    import socket
    import threading
    import time

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("тестовый сервер не стартовал")
    return f"http://127.0.0.1:{port}", server


def test_models(mock_llm_url):
    with _client(mock_llm_url) as client:
        assert "mock-qwen" in client.models()


def test_chat(mock_llm_url):
    with _client(mock_llm_url) as client:
        answer = client.chat([{"role": "user", "content": "diag:ping"}])
        assert answer == "diag:pong"


def test_base_url_with_and_without_v1(mock_llm_url):
    for url in (mock_llm_url, mock_llm_url + "/v1", mock_llm_url + "/v1/"):
        with _client(url) as client:
            assert client.chat([{"role": "user", "content": "diag:ping"}]) == "diag:pong"


def test_embeddings_deterministic_and_lexical(mock_llm_url):
    with _client(mock_llm_url) as client:
        v1, v2, v3 = client.embed(
            [
                "функция расчёта стоимости заказа",
                "функция расчёта стоимости доставки",
                "погода завтра в городе",
            ]
        )
    assert v1 == client_embed_again(mock_llm_url, "функция расчёта стоимости заказа")
    sim_close = _cosine(v1, v2)
    sim_far = _cosine(v1, v3)
    assert sim_close > sim_far  # bag-of-words: лексически близкие тексты ближе


def test_embed_on_separate_endpoint(mock_llm_url):
    """embed_base_url задан: эмбеддинги уходят на отдельный API, чат — на основной."""
    from fastapi import FastAPI, Request

    embed_app = FastAPI()

    @embed_app.post("/v1/embeddings")
    async def _emb(request: Request):
        payload = await request.json()
        texts = payload.get("input", [])
        texts = [texts] if isinstance(texts, str) else texts
        # отличительная размерность 8 — докажет, что запрос пришёл именно сюда
        return {"data": [{"index": i, "embedding": [0.5] * 8} for i in range(len(texts))]}

    url, server = _serve(embed_app)
    try:
        cfg = LLMConfig(
            base_url=mock_llm_url, api_key="sk-chat",
            embed_base_url=url, embed_api_key="sk-embed",
            chat_model="mock-qwen", embed_model="mock-embed",
        )
        with LLMClient(cfg) as client:
            assert client.embed_endpoint == url + "/v1"
            assert client.chat_endpoint == mock_llm_url + "/v1"
            assert all(len(v) == 8 for v in client.embed(["a", "b"]))
            # чат при этом ходит на основной сервер
            answer = client.chat([{"role": "user", "content": "diag:ping"}])
            assert answer == "diag:pong"
    finally:
        server.should_exit = True


def test_embed_falls_back_to_main_endpoint(mock_llm_url):
    """embed_base_url пуст — эмбеддинги идут через основной endpoint."""
    cfg = LLMConfig(
        base_url=mock_llm_url, api_key="sk-test",
        chat_model="mock-qwen", embed_model="mock-qwen",
    )
    with LLMClient(cfg) as client:
        assert client.embed_endpoint == client.chat_endpoint
        assert len(client.embed(["текст"])[0]) == EMBED_DIM


def client_embed_again(url, text):
    with _client(url) as client:
        return client.embed([text])[0]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)
