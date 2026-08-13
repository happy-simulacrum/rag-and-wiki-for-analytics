"""OpenAI-совместимый клиент для LiteLLM/Qwen: чат и эмбеддинги одной моделью."""

from __future__ import annotations

import httpx

from codeqa.config import LLMConfig


class LLMError(Exception):
    """Ошибка обращения к LLM API."""


def _normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


class LLMClient:
    """Синхронный клиент. base_url — с "/v1" или без (добавим сами)."""

    def __init__(self, cfg: LLMConfig):
        self._base = _normalize_base_url(cfg.base_url)
        self._chat_model = cfg.chat_model
        self._embed_model = cfg.embed_model
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=httpx.Timeout(cfg.timeout_sec, connect=10.0),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base}{path}"
        try:
            resp = self._http.post(url, json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"Сеть: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code} от {url}: {resp.text[:500]}")
        return resp.json()

    def models(self) -> list[str]:
        try:
            resp = self._http.get(f"{self._base}/models")
        except httpx.HTTPError as e:
            raise LLMError(f"Сеть: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return [m.get("id", "?") for m in resp.json().get("data", [])]

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        data = self._post(
            "/chat/completions",
            {
                "model": self._chat_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Неожиданный формат ответа chat: {data!r:.500}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(
            "/embeddings",
            {"model": self._embed_model, "input": texts},
        )
        try:
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except (KeyError, TypeError) as e:
            raise LLMError(f"Неожиданный формат ответа embeddings: {data!r:.500}") from e
