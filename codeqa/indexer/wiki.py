"""Wiki-фаза индексации: карточка проекта (overview.md), index.md, log.md."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from codeqa.llm import LLMClient
from codeqa.registry import Project

from .walker import SourceFile

_OVERVIEW_PROMPT = """\
Ты — библиотекарь базы знаний по кодовой базе. Составь карточку проекта \
для маршрутизации вопросов аналитиков.

Проект: {name}
Алиасы: {aliases}
Описание от разработчиков: {description}
Сабмодули: {modules}
Языки (файлов): {languages}
Каталоги верхнего уровня: {top_dirs}
Всего индексируемых файлов: {file_count}

README (фрагмент):
{readme}
{previous}
Требования к карточке (markdown, до 1500 слов):
- назначение проекта (1-2 абзаца)
- технологический стек
- ключевые модули/каталоги и их роль
- типичные слова и синонимы, которыми аналитики могут называть этот проект
Пиши по-русски, кратко и по делу. Ответ — только markdown карточки.
"""

_PREV_BLOCK = """\
Предыдущая версия карточки (обнови её, сохрани верное, выкинь устаревшее):
---
{card}
---
С тех пор изменились файлы: {changed}
"""


def wiki_dir(data_dir: str | Path, project: str) -> Path:
    path = Path(data_dir) / "wiki" / project
    (path / "concepts").mkdir(parents=True, exist_ok=True)
    return path


def _readme_excerpt(repo: Path, limit: int = 2000) -> str:
    for candidate in ("README.md", "README", "readme.md", "README.txt"):
        f = repo / candidate
        if f.exists():
            return f.read_text(encoding="utf-8", errors="ignore")[:limit]
    return "(нет README)"


def update_overview(
    llm: LLMClient,
    project: Project,
    files: list[SourceFile],
    changed: list[str] | None,
    out_dir: Path,
) -> Path:
    languages = Counter(f.language for f in files)
    repo = Path(project.path)
    top_dirs = sorted(
        {p.parts[0] for f in files for p in [Path(f.relpath)] if len(p.parts) > 1}
    )[:30]
    modules = sorted({f.module for f in files if f.module})

    previous = ""
    overview_path = out_dir / "overview.md"
    if overview_path.exists() and changed is not None:
        previous = _PREV_BLOCK.format(
            card=overview_path.read_text(encoding="utf-8")[:4000],
            changed=", ".join(changed[:50]) or "(нет)",
        )

    prompt = _OVERVIEW_PROMPT.format(
        name=project.name,
        aliases=", ".join(project.aliases) or "(не заданы)",
        description=project.description or "(нет)",
        modules=", ".join(modules) or "(нет)",
        languages=", ".join(f"{lang}: {n}" for lang, n in languages.most_common()),
        top_dirs=", ".join(top_dirs),
        file_count=len(files),
        readme=_readme_excerpt(repo),
        previous=previous,
    )
    card = llm.chat([{"role": "user", "content": prompt}], max_tokens=2048, temperature=0.2)
    overview_path.write_text(card.strip() + "\n", encoding="utf-8")
    return overview_path


def append_log(out_dir: Path, entry: str) -> None:
    log_path = out_dir / "log.md"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {ts}\n{entry}\n")


def update_index(out_dir: Path) -> None:
    lines = ["# Индекс вики проекта", ""]
    for page in sorted(out_dir.rglob("*.md")):
        rel = page.relative_to(out_dir).as_posix()
        if rel in ("index.md", "log.md"):
            continue
        lines.append(f"- [[{rel[:-3]}]]")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
