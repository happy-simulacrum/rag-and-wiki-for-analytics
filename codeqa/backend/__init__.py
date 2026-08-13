"""FastAPI-бэкенд: OpenAI-совместимый endpoint «единого окна»."""

from .app import create_app

__all__ = ["create_app"]
