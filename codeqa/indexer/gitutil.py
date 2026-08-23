"""Минимальные git-операции для инкрементальной индексации (через CLI git)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def is_git_repo(repo: Path) -> bool:
    try:
        _git(repo, "rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def current_commit(repo: Path) -> str | None:
    try:
        return _git(repo, "rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError:
        return None


def changed_files(repo: Path, old_commit: str) -> tuple[list[str], list[str]]:
    """Изменённые/новые и удалённые файлы (worktree против old_commit + untracked)."""
    changed: list[str] = []
    deleted: list[str] = []
    out = _git(
        repo, "diff", "--name-status", "--no-renames",
        "--diff-filter=ACMD", old_commit,
    )
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        if status.startswith("D"):
            deleted.append(path)
        else:
            changed.append(path)
    # untracked — тоже новые файлы
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    changed.extend(p for p in untracked.splitlines() if p)
    return changed, deleted
