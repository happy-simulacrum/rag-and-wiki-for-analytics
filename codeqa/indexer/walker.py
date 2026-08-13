"""Обход репозитория: фильтры, сабмодули, определение языка."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Расширение → язык (грамматики tree-sitter)
LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".go": "go",
}

# Текстовые файлы без грамматики — индексируем скользящим окном
WINDOW_EXTS = {
    ".md", ".txt", ".sql", ".xml", ".yaml", ".yml", ".json", ".properties",
    ".sh", ".bat", ".cmd", ".ps1", ".ini", ".cfg", ".toml", ".gradle",
    ".csproj", ".sln", ".csv", ".html", ".css",
}

SKIP_DIRS = {
    ".git", ".svn", ".hg", ".idea", ".vs", ".vscode",
    "node_modules", "bower_components", "vendor",
    "bin", "obj", "target", "build", "dist", "out",
    "__pycache__", ".gradle", ".mypy_cache", ".pytest_cache", "packages",
}

SKIP_FILE_PATTERNS = [
    re.compile(r".*\.lock$"), re.compile(r"package-lock\.json$"),
    re.compile(r"yarn\.lock$"), re.compile(r"go\.sum$"),
    re.compile(r".*\.min\.(js|css)$"), re.compile(r".*\.(png|jpg|jpeg|gif|ico|svg|woff2?|ttf|eot)$"),
    re.compile(r".*\.(zip|jar|war|dll|exe|so|dylib|class|pyc)$"),
]

MAX_FILE_SIZE = 512 * 1024  # 512 КБ


@dataclass
class SourceFile:
    module: str      # имя сабмодуля или "" для корня
    relpath: str     # путь относительно корня репозитория
    abspath: Path
    language: str    # язык грамматики или "text"


def parse_gitmodules(repo_root: Path) -> dict[str, str]:
    """Путь сабмодуля → его имя (из .gitmodules)."""
    gitmodules = repo_root / ".gitmodules"
    mapping: dict[str, str] = {}
    if not gitmodules.exists():
        return mapping
    current_path: str | None = None
    for line in gitmodules.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        m = re.match(r'\[submodule "(.+)"\]', line)
        if m:
            current_path = None
            current_name = m.group(1)
            mapping[f"__name__{current_name}"] = current_name
        elif line.startswith("path") and "=" in line:
            current_path = line.split("=", 1)[1].strip()
            # имя сабмодуля = последняя секция [submodule "..."]; упростим:
            # имя = basename пути, если секцию не сопоставили
            mapping.setdefault(current_path, Path(current_path).name)
    return {k: v for k, v in mapping.items() if not k.startswith("__name__")}


def classify(path: Path) -> str | None:
    """Язык файла или None, если файл индексировать не нужно."""
    ext = path.suffix.lower()
    if ext in LANGUAGE_BY_EXT:
        return LANGUAGE_BY_EXT[ext]
    if ext in WINDOW_EXTS:
        return "text"
    return None


def _skipped(name: str) -> bool:
    return any(p.match(name) for p in SKIP_FILE_PATTERNS)


def walk_repo(repo_root: Path, only_relpaths: set[str] | None = None) -> list[SourceFile]:
    """Список индексируемых файлов. only_relpaths — ограничение (инкремент)."""
    repo_root = repo_root.resolve()
    submodules = parse_gitmodules(repo_root)
    result: list[SourceFile] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        parts = rel.parts
        if any(part in SKIP_DIRS for part in parts[:-1]):
            continue
        if _skipped(path.name):
            continue
        relpath = rel.as_posix()
        if only_relpaths is not None and relpath not in only_relpaths:
            continue
        language = classify(path)
        if language is None:
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        module = ""
        for sub_path, sub_name in submodules.items():
            if relpath.startswith(sub_path.rstrip("/") + "/"):
                module = sub_name
                break
        result.append(SourceFile(module=module, relpath=relpath, abspath=path, language=language))
    return result
