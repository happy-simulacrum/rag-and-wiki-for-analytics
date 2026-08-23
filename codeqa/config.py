"""Загрузка конфигурации: YAML-файл + переопределения из окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ENV_PREFIX = "CODEQA_"


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:4000"
    api_key: str = ""
    chat_model: str = "qwen3.5"
    embed_model: str = "qwen3.5"
    # отдельный endpoint эмбеддингов (своя модель); пусто — как base_url/api_key
    embed_base_url: str = ""
    embed_api_key: str = ""
    timeout_sec: int = 120
    max_context_tokens: int = 256000
    answer_context_budget: int = 200000


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    repos_root: str = "./repos"


@dataclass
class RetrievalConfig:
    vector_top_k: int = 40
    lexical_top_k: int = 40
    faq_max_entries: int = 50
    faq_max_tokens: int = 20000
    # роутер: молча роутим, если cos >= silent_threshold и отрыв >= margin
    route_silent_threshold: float = 0.75
    route_margin: float = 0.1
    # wiki-first: минимальная близость страницы вики к вопросу
    wiki_threshold: float = 0.7


@dataclass
class WebConfig:
    port: int = 8080


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    web: WebConfig = field(default_factory=WebConfig)
    # URL сервера Qdrant; пусто = local-режим (data_dir/qdrant)
    qdrant_url: str = ""


def default_config_path() -> Path | None:
    """Путь к конфигу по умолчанию: env, ./config.yaml, ~/.config/codeqa/config.yaml."""
    if os.environ.get(f"{ENV_PREFIX}CONFIG"):
        return Path(os.environ[f"{ENV_PREFIX}CONFIG"])
    for candidate in (Path("config.yaml"), Path.home() / ".config" / "codeqa" / "config.yaml"):
        if candidate.exists():
            return candidate
    return None


def _apply_yaml(cfg: Config, data: dict) -> None:
    for section_name in ("llm", "paths", "retrieval", "web"):
        section = data.get(section_name) or {}
        target = getattr(cfg, section_name)
        for key, value in section.items():
            if hasattr(target, key):
                setattr(target, key, value)
            else:
                raise ValueError(f"Неизвестный ключ конфига: {section_name}.{key}")
    if "qdrant_url" in data:
        cfg.qdrant_url = data["qdrant_url"]


def _apply_env(cfg: Config) -> None:
    """Явные переопределения из переменных окружения CODEQA_*."""
    mapping = {
        "LLM_BASE_URL": (cfg.llm, "base_url"),
        "LLM_API_KEY": (cfg.llm, "api_key"),
        "LLM_CHAT_MODEL": (cfg.llm, "chat_model"),
        "LLM_EMBED_MODEL": (cfg.llm, "embed_model"),
        "EMBED_BASE_URL": (cfg.llm, "embed_base_url"),
        "EMBED_API_KEY": (cfg.llm, "embed_api_key"),
        "DATA_DIR": (cfg.paths, "data_dir"),
        "REPOS_ROOT": (cfg.paths, "repos_root"),
        "QDRANT_URL": (cfg, "qdrant_url"),
    }
    for env_name, (obj, attr) in mapping.items():
        value = os.environ.get(f"{ENV_PREFIX}{env_name}")
        if value is not None:
            setattr(obj, attr, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    cfg_path = Path(path) if path else default_config_path()
    if cfg_path is not None:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        _apply_yaml(cfg, data)
    _apply_env(cfg)
    return cfg
