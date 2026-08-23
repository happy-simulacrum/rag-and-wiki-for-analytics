"""OpenAI-совместимый endpoint: Open WebUI подключается как к обычной модели.

Пайплайн: роутер проектов → wiki-first → гибридный ретрив → ответ с цитатами.
Stateless: вся история диалога приходит в каждом запросе.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from codeqa.answer import answer_question, format_sources
from codeqa.config import Config
from codeqa.llm import LLMClient
from codeqa.retrieval.router import CLARIFY_MARKER, ProjectRouter
from codeqa.store import ChunkStore, VectorStore
from codeqa.wiki_search import WikiSearch

MODEL_ID = "codeqa-assistant"
log = logging.getLogger("codeqa.backend")


def _question_for_retrieval(messages: list[dict]) -> str:
    """Исходный вопрос: если последний ответ был уточнением роутера,
    вопросом для ретрива остаётся реплика ДО уточнения, а не ответ «1»."""
    users = [
        (i, m.get("content", "")) for i, m in enumerate(messages)
        if m.get("role") == "user"
    ]
    if not users:
        return ""
    idx, content = users[-1]
    for m in reversed(messages[:idx]):
        if m.get("role") == "assistant":
            if CLARIFY_MARKER in m.get("content", ""):
                prev = [c for i2, c in users if i2 < idx]
                return prev[-1] if prev else content
            break
    return content


def create_app(cfg: Config) -> FastAPI:
    data_dir = Path(cfg.paths.data_dir)
    llm = LLMClient(cfg.llm)
    store = ChunkStore(data_dir / "index.sqlite")
    vectors = (
        VectorStore(url=cfg.qdrant_url)
        if cfg.qdrant_url
        else VectorStore(local_path=str(data_dir / "qdrant"))
    )
    router = ProjectRouter(cfg, llm)
    wiki = WikiSearch(cfg, llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        llm.close()
        store.close()
        vectors.close()

    app = FastAPI(title="codeqa-backend", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    def _handle(messages: list[dict]) -> str:
        route = router.route(messages)
        if route.project is None:
            if not route.candidates:
                return "Реестр проектов пуст. Обратитесь к руководителю разработки."
            return router.clarification_message(route.candidates)
        question = _question_for_retrieval(messages)
        project = route.project.name
        wiki_hits = wiki.search(
            project, question, threshold=cfg.retrieval.wiki_threshold
        )
        result = answer_question(cfg, llm, store, vectors, project, question, wiki_hits)
        return result["answer"] + format_sources(result["sources"])

    def _response_payload(content: str) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    def _sse(content: str):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": content},
                 "finish_reason": None}
            ],
        }
        done = {
            "id": chunk["id"], "object": "chat.completion.chunk",
            "created": chunk["created"], "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Тело запроса не является корректным JSON.",
                           "type": "invalid_request_error"}},
                status_code=400,
            )
        messages = payload.get("messages", [])
        try:
            # _handle блокирующий (LLM/SQLite/Qdrant) — не держим event loop
            content = await run_in_threadpool(_handle, messages)
        except Exception:
            # детали исключения — в журнал сервера, пользователю общий текст
            log.exception("ошибка обработки запроса")
            content = (
                "Внутренняя ошибка codeqa. Подробности в журнале backend; "
                "повторите вопрос или обратитесь к администратору."
            )
        if payload.get("stream"):
            return StreamingResponse(_sse(content), media_type="text/event-stream")
        return JSONResponse(_response_payload(content))

    return app
