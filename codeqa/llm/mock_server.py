"""Mock LLM-сервер для разработки и тестов (OpenAI-совместимый).

Детерминированный: эмбеддинги — bag-of-words хеширование (лексически похожие
тексты получают похожие векторы), чат — предсказуемые ответы. Лимит контекста
имитируется переменной окружения MOCK_CONTEXT_TOKENS.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

EMBED_DIM = int(os.environ.get("MOCK_EMBED_DIM", "256"))
MODEL_ID = "mock-qwen"

_token_re = re.compile(r"[0-9A-Za-zА-Яа-я_]+")


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    for tok in _token_re.findall(text.lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def create_app() -> FastAPI:
    app = FastAPI(title="mock-llm")

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        payload = await request.json()
        texts = payload.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        data = [
            {"object": "embedding", "index": i, "embedding": _embed(t)}
            for i, t in enumerate(texts)
        ]
        return {"object": "list", "data": data, "model": payload.get("model", MODEL_ID)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        messages = payload.get("messages", [])
        total = sum(_est_tokens(m.get("content", "")) for m in messages)
        max_ctx = int(os.environ.get("MOCK_CONTEXT_TOKENS", "262144"))
        if total > max_ctx:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"context length exceeded: {total} > {max_ctx}",
                        "type": "context_length_exceeded",
                    }
                },
            )
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if last_user.strip() == "diag:ping":
            answer = "diag:pong"
        else:
            answer = f"MOCK-ANSWER messages={len(messages)} user_chars={len(last_user)}"
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", MODEL_ID),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": total,
                "completion_tokens": _est_tokens(answer),
                "total_tokens": total + _est_tokens(answer),
            },
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock LLM-сервер")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8399)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
