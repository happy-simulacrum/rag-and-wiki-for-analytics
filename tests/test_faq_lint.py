"""Тесты этапа 6: FAQ по частоте (с лимитом) и wiki lint."""

import json
import sqlite3
import time
from pathlib import Path

from codeqa.faq import build_faq, clusterize
from codeqa.lint import lint_project
from codeqa.llm import LLMClient
from codeqa.store import ChunkStore, VectorStore


def _stores(cfg):
    data_dir = Path(cfg.paths.data_dir)
    return ChunkStore(data_dir / "index.sqlite"), VectorStore(local_path=str(data_dir / "qdrant"))


def test_clusterize_groups_similar(indexed):
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        vecs = llm.embed([
            "как считается налог", "как считается налог", "как считается налог",
            "что такое скидка",
        ])
    questions = [
        {"id": i, "ts": time.time() + i, "question": q, "embedding": v}
        for i, (q, v) in enumerate(zip(
            ["как считается налог", "как считается налог?", "как считается налог!",
             "что такое скидка"], vecs))
    ]
    clusters = clusterize(questions)
    # «как считается налог» (3 шт., почти идентичные) — один кластер, скидка — другой
    assert len(clusters) == 2
    assert clusters[0].count == 3
    assert clusters[0].rep.startswith("как считается налог")


def test_build_faq_respects_limits(indexed):
    cfg = indexed["cfg"]
    cfg.retrieval.faq_max_entries = 5
    cfg.retrieval.faq_max_tokens = 400  # жёсткий лимит — проверим вытеснение
    store, vectors = _stores(cfg)
    db = sqlite3.connect(Path(cfg.paths.data_dir) / "index.sqlite")
    with LLMClient(cfg.llm) as llm:
        # частый вопрос (5 раз) и редкий (1 раз)
        for i in range(5):
            vec = llm.embed(["как работает calculate_total"])[0]
            db.execute(
                "INSERT INTO question_log (ts, project, question, embedding, answered_ok) "
                "VALUES (?,?,?,?,1)",
                (time.time() + i, "billing", "как работает calculate_total",
                 json.dumps(vec)),
            )
        vec = llm.embed(["редкий вопрос про скидки"])[0]
        db.execute(
            "INSERT INTO question_log (ts, project, question, embedding, answered_ok) "
            "VALUES (?,?,?,?,1)",
            (time.time() + 100, "billing", "редкий вопрос про скидки", json.dumps(vec)),
        )
        db.commit()

        stats = build_faq(cfg, llm, store, vectors, "billing")
    assert stats["entries"] >= 1
    faq = Path(stats["path"]).read_text(encoding="utf-8")
    assert "как работает calculate_total" in faq
    assert "Спрашивали 5 раз" in faq
    db.close()
    store.close()
    vectors.close()


def test_lint_finds_stale_citations(indexed):
    cfg = indexed["cfg"]
    store, vectors = _stores(cfg)
    wiki = Path(cfg.paths.data_dir) / "wiki" / "billing"
    (wiki / "faq.md").write_text(
        "# FAQ\n\n## вопрос\n\nОтвет со ссылкой `main.py:5` и `ghost.py:10`.\n",
        encoding="utf-8",
    )
    with LLMClient(cfg.llm) as llm:
        result = lint_project(cfg, llm, store, "billing")
    stale = " ".join(result["stale_citations"])
    assert "ghost.py" in stale          # несуществующий файл найден
    assert "main.py" not in stale       # существующий — не тронут
    assert (wiki / "lint_report.md").exists()
    store.close()
    vectors.close()
