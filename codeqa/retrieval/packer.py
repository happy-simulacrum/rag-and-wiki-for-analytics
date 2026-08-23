"""Упаковка чанков в токен-бюджет контекста."""

from __future__ import annotations

# Кириллица у Qwen-токенизатора ~2-2.5 символа на токен, латиница ~4.
# Заниженная оценка переполняет контекст реальной модели — считаем по скриптам.
_CHARS_PER_TOKEN = {"cyr": 2.0, "other": 4.0}


def est_tokens(text: str) -> int:
    """Грубая оценка с поправкой на кириллицу."""
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    other = len(text) - cyr
    est = cyr / _CHARS_PER_TOKEN["cyr"] + other / _CHARS_PER_TOKEN["other"]
    return max(1, round(est))


def pack_chunks(chunks: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
    """Жадная упаковка по порядку ранжирования. Всегда берём хотя бы 1 чанк."""
    packed: list[dict] = []
    used = 0
    for c in chunks:
        t = est_tokens(c["text"])
        if used + t > budget_tokens and packed:
            break
        packed.append(c)
        used += t
    return packed, used
