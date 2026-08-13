from codeqa.config import LLMConfig
from codeqa.llm import LLMClient


def _client(url: str) -> LLMClient:
    return LLMClient(
        LLMConfig(base_url=url, api_key="sk-test", chat_model="mock-qwen", embed_model="mock-qwen")
    )


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


def client_embed_again(url, text):
    with _client(url) as client:
        return client.embed([text])[0]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)
