import socket
import threading
import time

import pytest
import uvicorn

from codeqa.llm.mock_server import app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_llm_url():
    """Запускает mock LLM-сервер в потоке, возвращает base_url."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("mock-сервер не стартовал")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


# ---- общая среда: два проиндексированных проекта (billing, shop) ----

_BILLING_PY = '''"""Биллинг: расчёты."""

TAX_RATE = 0.2


def calculate_total(items):
    """Полная стоимость заказа с налогом."""
    return sum(i.price for i in items) * (1 + TAX_RATE)


class DiscountPolicy:
    """Политика скидок."""

    def apply(self, amount):
        return amount * (1 - self.percent / 100)
'''

_SHOP_PY = '''def find_product(catalog, sku):
    return next((p for p in catalog if p.sku == sku), None)


class Cart:
    """Корзина покупок."""

    def checkout(self, customer):
        return {"customer": customer, "total": self.total()}
'''


def _git(repo, *args):
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(path, files):
    import subprocess  # noqa: F401

    path.mkdir(parents=True)
    _git(path, "init", "-q")
    for rel, content in files.items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, mock_llm_url):
    """Два проиндексированных проекта + cfg. Возвращает dict(cfg=..., root=...)."""
    from codeqa.config import Config, LLMConfig, PathsConfig
    from codeqa.indexer import IndexPipeline
    from codeqa.llm import LLMClient
    from codeqa.registry import Project, add_project

    root = tmp_path_factory.mktemp("indexed")
    _make_repo(root / "repos" / "billing", {"main.py": _BILLING_PY})
    _make_repo(root / "repos" / "shop", {"catalog.py": _SHOP_PY})

    cfg = Config()
    cfg.llm = LLMConfig(
        base_url=mock_llm_url, api_key="sk-test",
        chat_model="mock-qwen", embed_model="mock-qwen",
    )
    cfg.paths = PathsConfig(data_dir=str(root / "data"), repos_root=str(root / "repos"))

    add_project(cfg.paths.data_dir, Project(
        name="billing", path=str(root / "repos" / "billing"),
        aliases=["биллинг"], description="Расчёт заказов, инвойсы, налоги, скидки",
    ))
    add_project(cfg.paths.data_dir, Project(
        name="shop", path=str(root / "repos" / "shop"),
        aliases=["магазин"], description="Каталог товаров, корзина, доставка",
    ))

    with LLMClient(cfg.llm) as llm, IndexPipeline(cfg, llm) as pipe:
        for name in ("billing", "shop"):
            from codeqa.registry import get_project

            pipe.run(get_project(cfg.paths.data_dir, name), full=True)

    # осмысленные карточки вместо mock-текста (для тестов роутера по эмбеддингам)
    wiki = root / "data" / "wiki"
    (wiki / "billing" / "overview.md").write_text(
        "Проект billing: расчёт стоимости заказов, инвойсы, налоги, скидки.\n",
        encoding="utf-8",
    )
    (wiki / "shop" / "overview.md").write_text(
        "Проект shop: каталог товаров, корзина покупок, оформление доставки.\n",
        encoding="utf-8",
    )
    return {"cfg": cfg, "root": root}
