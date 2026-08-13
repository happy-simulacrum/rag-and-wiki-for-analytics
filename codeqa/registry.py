"""Реестр проектов: projects.yaml (источник правды для роутера и индексера)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class Project:
    name: str
    path: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""


def registry_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "projects.yaml"


def load_registry(data_dir: str | Path) -> list[Project]:
    path = registry_path(data_dir)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Project(**item) for item in data.get("projects", [])]


def save_registry(data_dir: str | Path, projects: list[Project]) -> None:
    path = registry_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"projects": [asdict(p) for p in projects]}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def get_project(data_dir: str | Path, name: str) -> Project:
    for p in load_registry(data_dir):
        if p.name == name:
            return p
    raise KeyError(f"Проект не найден: {name}")


def add_project(data_dir: str | Path, project: Project) -> None:
    projects = load_registry(data_dir)
    if any(p.name == project.name for p in projects):
        raise ValueError(f"Проект уже существует: {project.name}")
    projects.append(project)
    save_registry(data_dir, projects)


def update_project(data_dir: str | Path, name: str, **changes) -> Project:
    projects = load_registry(data_dir)
    for p in projects:
        if p.name == name:
            for key, value in changes.items():
                if value is not None:
                    if not hasattr(p, key):
                        raise ValueError(f"Неизвестное поле проекта: {key}")
                    setattr(p, key, value)
            save_registry(data_dir, projects)
            return p
    raise KeyError(f"Проект не найден: {name}")


def remove_project(data_dir: str | Path, name: str) -> None:
    projects = load_registry(data_dir)
    remaining = [p for p in projects if p.name != name]
    if len(remaining) == len(projects):
        raise KeyError(f"Проект не найден: {name}")
    save_registry(data_dir, remaining)
