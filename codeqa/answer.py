"""Ответный пайплайн: ретрив → упаковка → промпт → LLM → ответ с цитатами."""

from __future__ import annotations

from pathlib import Path

from codeqa.config import Config
from codeqa.llm import LLMClient
from codeqa.retrieval import HybridRetriever, pack_chunks
from codeqa.store import ChunkStore, VectorStore

SYSTEM_PROMPT = (
    "Ты — помощник аналитиков, отвечающий на вопросы по кодовой базе проекта. "
    "Отвечай по-русски, опираясь ТОЛЬКО на приведённый контекст. "
    "Каждое утверждение о коде сопровождай ссылкой в формате `файл:строка`. "
    "Если контекста недостаточно — честно скажи «не знаю» и предложи, "
    "какая информация нужна."
)

_USER_TEMPLATE = """\
Карточка проекта:
{card}

Вопрос аналитика: {question}

{wiki_block}Контекст кода:
{context}

Ответь на вопрос, цитируя источники как `файл:строка`.\
"""


def _read_card(cfg: Config, project: str, limit: int = 2000) -> str:
    path = Path(cfg.paths.data_dir) / "wiki" / project / "overview.md"
    if path.exists():
        return path.read_text(encoding="utf-8")[:limit]
    return "(карточка отсутствует)"


def _format_code_block(c: dict) -> str:
    return (
        f"=== {c['relpath']}:{c['start_line']}-{c['end_line']} ({c['symbol']}) ===\n"
        f"{c['text']}"
    )


def answer_question(
    cfg: Config,
    llm: LLMClient,
    store: ChunkStore,
    vectors: VectorStore,
    project: str,
    question: str,
    wiki_hits: list[dict] | None = None,
    log_to_faq: bool = True,
) -> dict:
    """Полный цикл ответа. Возвращает answer + sources + chunks_used.

    log_to_faq=False — не писать вопрос в лог (генерация FAQ, иначе
    частоты кластеров раздувается сами собой).
    """
    retriever = HybridRetriever(
        store, vectors, llm,
        vector_top_k=cfg.retrieval.vector_top_k,
        lexical_top_k=cfg.retrieval.lexical_top_k,
    )
    chunks, qvec = retriever.retrieve(project, question)
    packed, used_tokens = pack_chunks(chunks, cfg.llm.answer_context_budget)

    wiki_block = ""
    if wiki_hits:
        parts = [
            f"=== wiki: {h['title']} ===\n{h['text']}" for h in wiki_hits
        ]
        wiki_block = "Накопленные знания вики (приоритетный контекст):\n" + "\n\n".join(parts) + "\n\n"

    context = "\n\n".join(_format_code_block(c) for c in packed) or "(код не найден)"
    user = _USER_TEMPLATE.format(
        card=_read_card(cfg, project),
        question=question,
        wiki_block=wiki_block,
        context=context,
    )
    answer = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    if log_to_faq:
        store.log_question(project, question, qvec, ok=True)
    sources = [
        {
            "relpath": c["relpath"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "symbol": c["symbol"],
        }
        for c in packed
    ]
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(packed),
        "context_tokens": used_tokens,
    }


def format_sources(sources: list[dict], limit: int = 10) -> str:
    lines = ["", "---", "**Источники:**"]
    for s in sources[:limit]:
        lines.append(f"- `{s['relpath']}:{s['start_line']}` ({s['symbol']})")
    return "\n".join(lines)
