"""Общие утилиты (без зависимостей от остальных модулей codeqa)."""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    """Косинусная близость векторов одинаковой размерности."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
