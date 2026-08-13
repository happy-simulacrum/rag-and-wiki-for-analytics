"""Гибридный ретрив: FTS5 (лексика) + Qdrant (вектор) → RRF + буст символов."""

from __future__ import annotations

from codeqa.llm import LLMClient
from codeqa.store import ChunkStore, VectorStore

from .identifiers import extract_identifiers, extract_terms

RRF_K = 60           # константа reciprocal rank fusion
SYMBOL_BOOST = 3.0   # буст чанка, чей символ точно назван в вопросе


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        vectors: VectorStore,
        llm: LLMClient,
        vector_top_k: int = 40,
        lexical_top_k: int = 40,
    ):
        self._store = store
        self._vectors = vectors
        self._llm = llm
        self._vec_k = vector_top_k
        self._lex_k = lexical_top_k

    def retrieve(self, project: str, question: str, limit: int = 60) -> tuple[list[dict], list[float]]:
        """Возвращает (чанки по убыванию релевантности, эмбеддинг вопроса)."""
        terms = extract_terms(question)
        lex_ids = self._store.lexical_search(project, terms, self._lex_k)

        qvec = self._llm.embed([question])[0]
        vec_hits = self._vectors.search(project, qvec, self._vec_k)

        scores: dict[str, float] = {}
        for rank, cid in enumerate(lex_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (cid, _score) in enumerate(vec_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        chunks = self._store.get_chunks(list(scores))
        idents = {i.lower() for i in extract_identifiers(question)}
        for c in chunks:
            if c["symbol"] and c["symbol"].lower() in idents:
                scores[c["chunk_id"]] *= SYMBOL_BOOST

        chunks.sort(key=lambda c: scores.get(c["chunk_id"], 0.0), reverse=True)
        return chunks[:limit], qvec
