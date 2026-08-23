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


def test_est_tokens_cyrillic_not_underestimated():
    """Кириллица ~2 символа на токен: len//4 занижал бы вдвое."""
    ru = "привет " * 100          # 700 символов кириллицы
    en = "word " * 100            # 500 символов латиницы
    assert est_tokens(ru) >= 300
    assert est_tokens(en) <= 150
    assert est_tokens("") == 1


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


def _chunk(chunk_id: str):
    from codeqa.indexer.chunker import Chunk

    return Chunk(
        chunk_id=chunk_id, project="p", module="", relpath="f.py",
        language="python", symbol="f", start_line=1, end_line=2, text="# f",
    )


def test_vector_store_dim_change_recreates(tmp_path):
    """Смена модели эмбеддингов (другая размерность) → коллекция пересоздаётся."""
    vs = VectorStore(local_path=str(tmp_path / "qdrant"))
    try:
        vs.upsert("p", [_chunk("a" * 16)], [[1.0] * 4])
        assert len(vs.search("p", [1.0] * 4)) == 1
        # пришёл вектор новой размерности — старая коллекция несовместима
        vs.upsert("p", [_chunk("b" * 16)], [[0.5] * 8])
        hits = vs.search("p", [0.5] * 8)
        assert [h[0] for h in hits] == ["b" * 16]
        # запрос старой размерности к новому индексу честно возвращает пусто
        assert vs.search("p", [1.0] * 4) == []
    finally:
        vs.close()
