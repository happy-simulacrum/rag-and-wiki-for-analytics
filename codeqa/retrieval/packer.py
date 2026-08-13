"""Упаковка чанков в токен-бюджет контекста."""

from __future__ import annotations


def est_tokens(text: str) -> int:
    """Грубая оценка: ~4 символа на токен."""
    return max(1, len(text) // 4)


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
