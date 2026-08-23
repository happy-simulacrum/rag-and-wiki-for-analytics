"""OpenAI-совместимый клиент: чат через LiteLLM/Qwen, эмбеддинги — отдельным endpoint'ом."""

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
    """Синхронный клиент. base_url — с "/v1" или без (добавим сами).

    Если заданы embed_base_url/embed_api_key — эмбеддинги уходят на
    отдельный endpoint (своя модель); иначе используется основной.
    """

    def __init__(self, cfg: LLMConfig):
        self._base = _normalize_base_url(cfg.base_url)
        self._chat_model = cfg.chat_model
        self._embed_model = cfg.embed_model
        self._embed_base = (
            _normalize_base_url(cfg.embed_base_url) if cfg.embed_base_url else self._base
        )
        embed_key = cfg.embed_api_key or cfg.api_key
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=httpx.Timeout(cfg.timeout_sec, connect=10.0),
        )
        if self._embed_base != self._base or embed_key != cfg.api_key:
            self._embed_http = httpx.Client(
                headers={"Authorization": f"Bearer {embed_key}"},
                timeout=httpx.Timeout(cfg.timeout_sec, connect=10.0),
            )
        else:
            self._embed_http = self._http

    @property
    def chat_endpoint(self) -> str:
        """Нормализованный URL основного API (для diag)."""
        return self._base

    @property
    def embed_endpoint(self) -> str:
        """Нормализованный URL API эмбеддингов (для diag)."""
        return self._embed_base

    def close(self) -> None:
        self._http.close()
        if self._embed_http is not self._http:
            self._embed_http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _post(http: httpx.Client, url: str, payload: dict) -> dict:
        try:
            resp = http.post(url, json=payload)
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
            self._http,
            f"{self._base}/chat/completions",
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
            self._embed_http,
            f"{self._embed_base}/embeddings",
            {"model": self._embed_model, "input": texts},
        )
        try:
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except (KeyError, TypeError) as e:
            raise LLMError(f"Неожиданный формат ответа embeddings: {data!r:.500}") from e
