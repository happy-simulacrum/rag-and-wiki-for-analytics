"""OpenAI-совместимый endpoint: Open WebUI подключается как к обычной модели.

Пайплайн: роутер проектов → wiki-first → гибридный ретрив → ответ с цитатами.
Stateless: вся история диалога приходит в каждом запросе.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codeqa.answer import answer_question, format_sources
from codeqa.config import Config
from codeqa.llm import LLMClient
from codeqa.retrieval.router import ProjectRouter
from codeqa.store import ChunkStore, VectorStore
from codeqa.wiki_search import WikiSearch

MODEL_ID = "codeqa-assistant"


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
        question = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
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
        payload = await request.json()
        messages = payload.get("messages", [])
        try:
            content = _handle(messages)
        except Exception as e:  # не отдаём 500 в чат — отвечаем понятным текстом
            content = f"Внутренняя ошибка codeqa: {e}"
        if payload.get("stream"):
            return StreamingResponse(_sse(content), media_type="text/event-stream")
        return JSONResponse(_response_payload(content))

    return app
