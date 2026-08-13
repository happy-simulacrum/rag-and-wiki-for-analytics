"""Извлечение идентификаторов и терминов из вопроса (для лексического поиска)."""

from __future__ import annotations

import re

# camelCase / PascalCase / snake_case / dotted.path / file.ext
_PATTERNS = [
    re.compile(r"\b[a-zа-яё][\w]*[A-Z][\w]*\b"),                   # camelCase
    re.compile(r"\b[A-ZА-ЯЁ][a-zа-яё\d]+(?:[A-ZА-ЯЁ][\w]*)+\b"),  # PascalCase
    re.compile(r"\b\w*_\w+\b"),                                   # snake_case
    re.compile(r"\b[\w-]+(?:\.[\w-]+)+\b"),                       # dotted.path, file.ext
]
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")


def extract_identifiers(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            tok = m.group(0)
            if len(tok) >= 3 and tok.lower() not in seen:
                seen.add(tok.lower())
                out.append(tok)
    return out


def extract_terms(text: str) -> list[str]:
    """Термины для FTS: идентификаторы + значимые слова (>= 4 символов)."""
    idents = extract_identifiers(text)
    out = list(idents)
    seen = {i.lower() for i in idents}
    for w in _WORD_RE.findall(text):
        if len(w) >= 4 and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out
