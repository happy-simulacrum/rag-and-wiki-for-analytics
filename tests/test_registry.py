"""Тесты реестра проектов: валидация имён, дубликаты, обновление."""

import pytest

from codeqa.registry import (
    Project, add_project, get_project, load_registry, update_project,
)


def test_add_and_get(tmp_path):
    add_project(tmp_path, Project(name="billing", path="/repos/billing", aliases=["биллинг"]))
    p = get_project(tmp_path, "billing")
    assert p.aliases == ["биллинг"]
    assert [x.name for x in load_registry(tmp_path)] == ["billing"]


def test_add_duplicate_rejected(tmp_path):
    add_project(tmp_path, Project(name="shop", path="/repos/shop"))
    with pytest.raises(ValueError, match="уже существует"):
        add_project(tmp_path, Project(name="shop", path="/repos/other"))


@pytest.mark.parametrize("bad", ["../evil", "my.app", "a b", "", "/abs"])
def test_add_invalid_name_rejected(tmp_path, bad):
    """Имя становится каталогом вики и частью имени qdrant-коллекции."""
    with pytest.raises(ValueError, match="Недопустимое имя"):
        add_project(tmp_path, Project(name=bad, path="/repos/x"))


def test_update_unknown_field_rejected(tmp_path):
    add_project(tmp_path, Project(name="billing", path="/repos/billing"))
    with pytest.raises(ValueError, match="Неизвестное поле"):
        update_project(tmp_path, "billing", nonexistent="x")


def test_update_missing_project_rejected(tmp_path):
    with pytest.raises(KeyError, match="не найден"):
        update_project(tmp_path, "ghost", description="x")
