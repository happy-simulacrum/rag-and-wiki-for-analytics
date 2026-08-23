"""Wiki-first поиск: faq.md (по записям) + concepts/*.md, по эмбеддингам."""

from __future__ import annotations

import re
from pathlib import Path

from codeqa.config import Config
from codeqa.llm import LLMClient
from codeqa.util import cosine as _cosine

_MAX_PAGE_CHARS = 4000


def _split_faq(text: str) -> list[tuple[str, str]]:
    """faq.md → [(заголовок, текст записи)] по заголовкам '## '."""
    parts = re.split(r"(?m)^## ", text)
    out = []
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        out.append((title.strip(), body.strip()))
    return out


class WikiSearch:
    def __init__(self, cfg: Config, llm: LLMClient):
        self._cfg = cfg
        self._llm = llm

    def _pages(self, project: str) -> list[dict]:
        wiki = Path(self._cfg.paths.data_dir) / "wiki" / project
        pages: list[dict] = []
        faq = wiki / "faq.md"
        if faq.exists():
            for title, body in _split_faq(faq.read_text(encoding="utf-8")):
                pages.append({"title": f"faq: {title}", "text": body[:_MAX_PAGE_CHARS]})
        concepts = wiki / "concepts"
        if concepts.exists():
            for page in sorted(concepts.glob("*.md")):
                pages.append({
                    "title": f"concepts: {page.stem}",
                    "text": page.read_text(encoding="utf-8")[:_MAX_PAGE_CHARS],
                })
        return pages

    def search(
        self, project: str, question: str, threshold: float = 0.7, limit: int = 3
    ) -> list[dict]:
        pages = self._pages(project)
        if not pages:
            return []
        qvec = self._llm.embed([question])[0]
        pvecs = self._llm.embed([p["text"] for p in pages])
        scored = [
            {**p, "score": _cosine(qvec, pv)} for p, pv in zip(pages, pvecs)
        ]
        scored = [s for s in scored if s["score"] >= threshold]
        scored.sort(key=lambda s: s["score"], reverse=True)
        return scored[:limit]
