"""OpenAI-совместимый клиент: чат (LiteLLM/Qwen) + эмбеддинги (отдельный endpoint или тот же)."""

from .client import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
