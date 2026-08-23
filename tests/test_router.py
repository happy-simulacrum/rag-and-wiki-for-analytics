"""Тесты этапа 4: роутер проектов (алиасы, эмбеддинги, уточнения)."""

from codeqa.llm import LLMClient
from codeqa.retrieval.router import ProjectRouter
from codeqa.registry import Project


def test_route_by_alias(indexed):
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        result = router.route([{"role": "user", "content": "Как устроен биллинг?"}])
    assert result.project is not None
    assert result.project.name == "billing"


def test_route_by_embedding(indexed):
    cfg = indexed["cfg"]
    # BoW-эмбеддинги mock'а слабее настоящих — порог снижаем, логика та же
    cfg.retrieval.route_silent_threshold = 0.1
    cfg.retrieval.route_margin = 0.05
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        result = router.route([{"role": "user", "content": "инвойсы и налоги"}])
    assert result.project is not None
    assert result.project.name == "billing"
    assert "эмбеддинг" in result.reason


def test_route_ambiguous_clarifies(indexed):
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        result = router.route([{"role": "user", "content": "что это?"}])
    assert result.project is None
    assert len(result.candidates) >= 2
    msg = router.clarification_message(result.candidates)
    assert "Уточните проект" in msg
    assert "1)" in msg and "2)" in msg


def test_clarification_resolved_by_number(indexed):
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        clarify = router.clarification_message(router.projects)
        history = [
            {"role": "user", "content": "что это?"},
            {"role": "assistant", "content": clarify},
            {"role": "user", "content": "2"},
        ]
        result = router.route(history)
    assert result.project is not None
    assert result.project.name == "shop"  # второй в списке


def test_clarification_resolved_by_name(indexed):
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        clarify = router.clarification_message(router.projects)
        history = [
            {"role": "user", "content": "что это?"},
            {"role": "assistant", "content": clarify},
            {"role": "user", "content": "проект billing"},
        ]
        result = router.route(history)
    assert result.project is not None
    assert result.project.name == "billing"


def test_route_from_history_alias(indexed):
    """Проект упомянут раньше в диалоге — follow-up вопросы идут в него."""
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        history = [
            {"role": "user", "content": "расскажи про магазин"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "а как там устроен checkout?"},
        ]
        result = router.route(history)
    assert result.project is not None
    assert result.project.name == "shop"


def test_card_cache_invalidated_when_projects_change(indexed):
    """Проект, добавленный при живом бэкенде, не ломает route() через KeyError."""
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        router = ProjectRouter(cfg, llm)
        base = router.projects
        vecs = router._card_vectors(base)
        assert set(vecs) == {"billing", "shop"}
        assert router._card_vectors(base) is vecs  # состав не менялся — кэш жив
        extended = base + [Project(name="newproj", path=str(indexed["root"]))]
        vecs2 = router._card_vectors(extended)
        assert "newproj" in vecs2
        assert "billing" in vecs2
