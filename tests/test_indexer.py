"""Тесты этапа 2: walker, chunker, pipeline (полная + инкрементная индексация)."""

import subprocess
from pathlib import Path

import pytest

from codeqa.config import Config, LLMConfig, PathsConfig
from codeqa.indexer import IndexPipeline
from codeqa.indexer.chunker import chunk_text
from codeqa.indexer.walker import walk_repo
from codeqa.llm import LLMClient
from codeqa.registry import Project, add_project, load_registry
from codeqa.store import ChunkStore

MAIN_PY = '''"""Биллинг: основной модуль."""

TAX_RATE = 0.2


def calculate_total(items):
    """Полная стоимость заказа с налогом."""
    return sum(i.price for i in items) * (1 + TAX_RATE)


def format_invoice(order_id, total):
    return f"Invoice #{order_id}: {total:.2f}"


class DiscountPolicy:
    """Политика скидок."""

    def __init__(self, percent):
        self.percent = percent

    def apply(self, amount):
        return amount * (1 - self.percent / 100)
'''

UTIL_JAVA = """package billing;

public class Util {
    public static double round2(double value) {
        return Math.round(value * 100.0) / 100.0;
    }

    public static String greet(String name) {
        return "Hello, " + name;
    }
}
"""

CORE_PY = """def core_helper():
    return "core"
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def _write(repo: Path, rel: str, content: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message)


@pytest.fixture()
def repo(tmp_path):
    repo = tmp_path / "repos" / "billing"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _write(repo, "main.py", MAIN_PY)
    _write(repo, "src/Util.java", UTIL_JAVA)
    _write(repo, "lib/core/helper.py", CORE_PY)
    _write(repo, ".gitmodules", '[submodule "lib/core"]\n\tpath = lib/core\n\turl = ../core\n')
    _write(repo, "README.md", "# Billing\nСервис биллинга: заказы, инвойсы, скидки.\n")
    _write(repo, "package-lock.json", "{}")
    _write(repo, "node_modules/dep/index.js", "module.exports = 1;\n")
    _commit(repo, "init")
    return repo


def _cfg(tmp_path, mock_url) -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        base_url=mock_url, api_key="sk-test",
        chat_model="mock-qwen", embed_model="mock-qwen",
    )
    cfg.paths = PathsConfig(data_dir=str(tmp_path / "data"), repos_root=str(tmp_path / "repos"))
    return cfg


# ---- walker ----

def test_walk_filters_and_submodules(repo):
    files = walk_repo(repo)
    rels = {f.relpath for f in files}
    assert "main.py" in rels and "src/Util.java" in rels
    assert "package-lock.json" not in rels          # lock-файл пропущен
    assert not any(r.startswith("node_modules/") for r in rels)
    by_rel = {f.relpath: f for f in files}
    assert by_rel["lib/core/helper.py"].module == "core"   # сабмодуль распознан
    assert by_rel["main.py"].module == ""
    assert by_rel["src/Util.java"].language == "java"


# ---- chunker ----

def test_chunker_python_symbols(repo):
    text = (repo / "main.py").read_text(encoding="utf-8")
    chunks = chunk_text("billing", "", "main.py", "python", text)
    symbols = {c.symbol for c in chunks}
    assert "calculate_total" in symbols
    assert "DiscountPolicy" in symbols
    assert all(c.text.startswith("# main.py") for c in chunks)


def test_chunker_java(repo):
    text = (repo / "src/Util.java").read_text(encoding="utf-8")
    chunks = chunk_text("billing", "", "src/Util.java", "java", text)
    assert any(c.symbol == "Util" for c in chunks)


def test_chunker_huge_function_window_split():
    big = "def monster():\n" + "\n".join(f"    x{i} = {i}" for i in range(250)) + "\n"
    chunks = chunk_text("p", "", "big.py", "python", big)
    assert len(chunks) > 1
    assert all(c.symbol.startswith("lines") for c in chunks)


def test_chunker_text_windows():
    text = "\n".join(f"строка {i}" for i in range(400))
    chunks = chunk_text("p", "", "notes.md", "text", text)
    assert len(chunks) >= 3  # 400 строк / (150-30 overlap)
    assert chunks[0].start_line == 1


# ---- pipeline ----

def test_pipeline_full_and_incremental(repo, tmp_path, mock_llm_url):
    cfg = _cfg(tmp_path, mock_llm_url)
    add_project(cfg.paths.data_dir, Project(
        name="billing", path=str(repo), aliases=["биллинг"],
    ))
    project = load_registry(cfg.paths.data_dir)[0]

    with LLMClient(cfg.llm) as llm, IndexPipeline(cfg, llm) as pipe:
        stats = pipe.run(project, full=True)
        assert stats.chunks_indexed > 0
        assert not stats.incremental

        store = ChunkStore(Path(cfg.paths.data_dir) / "index.sqlite")
        assert store.count_chunks("billing") == stats.chunks_indexed
        # лексический поиск находит идентификатор
        hits = store.lexical_search("billing", ["calculate_total"])
        assert hits, "calculate_total должен находиться"

        # ---- инкремент: переименовали функцию ----
        _write(repo, "main.py", MAIN_PY.replace("calculate_total", "compute_total"))
        _commit(repo, "rename")
        stats2 = pipe.run(project)
        assert stats2.incremental
        assert "main.py" in stats2.changed_files
        assert store.lexical_search("billing", ["compute_total"])
        assert not store.lexical_search("billing", ["calculate_total"])

        # ---- инкремент: удалили файл ----
        (repo / "src" / "Util.java").unlink()
        _commit(repo, "drop Util")
        before = store.count_chunks("billing")
        stats3 = pipe.run(project)
        assert stats3.files_deleted == 1
        assert store.count_chunks("billing") < before
        store.close()

    # wiki-фаза: карточка, лог, индекс
    wiki = Path(cfg.paths.data_dir) / "wiki" / "billing"
    assert (wiki / "overview.md").exists()
    assert "Индексация" in (wiki / "log.md").read_text(encoding="utf-8")
    assert "overview" in (wiki / "index.md").read_text(encoding="utf-8")


def test_drop_project(repo, tmp_path, mock_llm_url):
    cfg = _cfg(tmp_path, mock_llm_url)
    project = Project(name="billing", path=str(repo))
    with LLMClient(cfg.llm) as llm, IndexPipeline(cfg, llm) as pipe:
        pipe.run(project, full=True)
        store = ChunkStore(Path(cfg.paths.data_dir) / "index.sqlite")
        assert store.count_chunks("billing") > 0
        pipe.drop_project("billing")
        assert store.count_chunks("billing") == 0
        store.close()
