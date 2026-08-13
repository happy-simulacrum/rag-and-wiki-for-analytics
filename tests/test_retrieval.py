"""Тесты этапа 3: идентификаторы, упаковка, гибридный ретрив, ответы."""

import sqlite3
from pathlib import Path

from codeqa.answer import answer_question, format_sources
from codeqa.llm import LLMClient
from codeqa.retrieval import (
    HybridRetriever, est_tokens, extract_identifiers, extract_terms, pack_chunks,
)
from codeqa.store import ChunkStore, VectorStore


def test_extract_identifiers():
    text = "Как работает calculateTotal и DiscountPolicy.apply в cart_service.py?"
    idents = extract_identifiers(text)
    assert "calculateTotal" in idents
    assert "DiscountPolicy" in idents
    assert "DiscountPolicy.apply" in idents or "cart_service.py" in idents
    assert "cart_service.py" in extract_identifiers(text)


def test_extract_terms_includes_russian():
    terms = extract_terms("как считается налог в calculate_total")
    assert "calculate_total" in terms
    assert "считается" in terms
    assert "в" not in terms  # короткие слова отбрасываем


def test_packer_budget():
    chunks = [{"text": "x" * 400}, {"text": "y" * 400}, {"text": "z" * 400}]
    packed, used = pack_chunks(chunks, budget_tokens=210)  # ~2 чанка по 100 токенов
    assert len(packed) == 2
    assert used <= 210 + est_tokens("x" * 400)
    # пустой бюджет — всё равно хотя бы один чанк
    packed1, _ = pack_chunks(chunks, budget_tokens=1)
    assert len(packed1) == 1


def _stores(cfg):
    data_dir = Path(cfg.paths.data_dir)
    store = ChunkStore(data_dir / "index.sqlite")
    vectors = VectorStore(local_path=str(data_dir / "qdrant"))
    return store, vectors


def test_hybrid_retrieval_finds_symbol(indexed):
    cfg = indexed["cfg"]
    store, vectors = _stores(cfg)
    with LLMClient(cfg.llm) as llm:
        r = HybridRetriever(store, vectors, llm)
        chunks, qvec = r.retrieve("billing", "как работает calculate_total?")
    assert chunks, "ретрив ничего не нашёл"
    assert chunks[0]["symbol"] == "calculate_total"
    assert len(qvec) > 0
    store.close()
    vectors.close()


def test_answer_question_flow(indexed):
    cfg = indexed["cfg"]
    store, vectors = _stores(cfg)
    with LLMClient(cfg.llm) as llm:
        result = answer_question(
            cfg, llm, store, vectors, "billing", "как работает calculate_total?"
        )
    assert result["answer"]            # mock вернул непустой ответ
    assert result["chunks_used"] > 0
    assert any(s["symbol"] == "calculate_total" for s in result["sources"])
    assert "main.py" in format_sources(result["sources"])
    # вопрос записан в лог (питает FAQ)
    db = sqlite3.connect(Path(cfg.paths.data_dir) / "index.sqlite")
    n = db.execute("SELECT COUNT(*) FROM question_log WHERE project='billing'").fetchone()[0]
    db.close()
    assert n >= 1
    store.close()
    vectors.close()
