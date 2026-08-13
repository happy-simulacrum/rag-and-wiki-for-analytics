"""Wiki lint: битые цитаты (файл исчез) + LLM-проверка противоречий/устаревшего."""

from __future__ import annotations

import re
from pathlib import Path

from codeqa.config import Config
from codeqa.indexer.wiki import append_log, wiki_dir
from codeqa.llm import LLMClient
from codeqa.store import ChunkStore

# цитата вида `path/to/file.py:123` или path/to/file.py:123
_CITATION_RE = re.compile(r"`?([\w./\-]+\.[a-zA-Z]+):(\d+)`?")

_LLM_PROMPT = """\
Ты — ревизор вики базы знаний по кодовой базе. Ниже — страницы вики проекта \
"{project}". Найди:
1) противоречия между страницами;
2) устаревшие или подозрительные утверждения;
3) страницы-дубликаты, которые стоит слить.
Если проблем нет — так и напиши. Отвечай по-русски, кратко, списком.

Страницы:
{pages}
"""


def _page_texts(out_dir: Path) -> dict[str, str]:
    pages: dict[str, str] = {}
    for page in sorted(out_dir.rglob("*.md")):
        rel = page.relative_to(out_dir).as_posix()
        if rel in ("index.md", "log.md", "lint_report.md"):
            continue
        pages[rel] = page.read_text(encoding="utf-8")
    return pages


def lint_project(cfg: Config, llm: LLMClient, store: ChunkStore, project: str) -> dict:
    out_dir = wiki_dir(cfg.paths.data_dir, project)
    pages = _page_texts(out_dir)

    # 1) детерминированная проверка: цитаты на несуществующие файлы
    stale: list[str] = []
    for rel, text in pages.items():
        for path, _line in _CITATION_RE.findall(text):
            if not store.has_relpath(project, path):
                stale.append(f"{rel}: цитата на отсутствующий файл `{path}`")

    # 2) LLM-ревизия содержимого
    joined = "\n\n".join(f"=== {rel} ===\n{text[:3000]}" for rel, text in pages.items())
    if joined.strip():
        review = llm.chat(
            [{"role": "user", "content": _LLM_PROMPT.format(project=project, pages=joined[:20000])}],
            max_tokens=2048,
        )
    else:
        review = "Вики пуста."

    report = ["# Lint-отчёт", ""]
    report.append(f"## Битые цитаты ({len(stale)})")
    report.extend(f"- {s}" for s in stale) if stale else report.append("- нет")
    report.append("")
    report.append("## Ревизия LLM")
    report.append(review)
    report_text = "\n".join(report) + "\n"
    (out_dir / "lint_report.md").write_text(report_text, encoding="utf-8")
    append_log(out_dir, f"Lint: битых цитат {len(stale)}.")
    return {"stale_citations": stale, "report": report_text}
