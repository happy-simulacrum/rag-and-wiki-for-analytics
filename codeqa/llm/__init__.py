"""OpenAI-совместимый клиент: чат + эмбеддинги (LiteLLM/Qwen)."""

from .client import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
