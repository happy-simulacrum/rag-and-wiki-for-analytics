"""Qdrant: векторный индекс. Local-режим (dev) или сервер (prod, через qdrant_url)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.http import models

if TYPE_CHECKING:
    from codeqa.indexer.chunker import Chunk


def collection_name(project: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", project)
    return f"codeqa_{safe}"


class VectorStore:
    def __init__(self, url: str | None = None, local_path: str | None = None):
        if url:
            self._client = QdrantClient(url=url)
        elif local_path:
            self._client = QdrantClient(path=local_path)
        else:
            raise ValueError("нужен url (сервер) или local_path (local-режим)")

    def close(self) -> None:
        self._client.close()

    def _ensure(self, project: str, dim: int) -> str:
        name = collection_name(project)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        return name

    def upsert(self, project: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        name = self._ensure(project, len(vectors[0]))
        points = [
            models.PointStruct(
                id=self._point_id(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "relpath": c.relpath,
                    "symbol": c.symbol,
                    "module": c.module,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=name, points=points)

    def search(self, project: str, vector: list[float], limit: int = 40) -> list[tuple[str, float]]:
        name = collection_name(project)
        if not self._client.collection_exists(name):
            return []
        hits = self._client.query_points(
            collection_name=name, query=vector, limit=limit, with_payload=True
        ).points
        return [(h.payload["chunk_id"], h.score) for h in hits]

    def delete_chunks(self, project: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        name = collection_name(project)
        if not self._client.collection_exists(name):
            return
        self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchAny(any=chunk_ids),
                    )]
                )
            ),
        )

    def drop_project(self, project: str) -> None:
        name = collection_name(project)
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def count(self, project: str) -> int:
        name = collection_name(project)
        if not self._client.collection_exists(name):
            return 0
        return self._client.count(collection_name=name).count

    @staticmethod
    def _point_id(chunk_id: str) -> int:
        # qdrant требует uint или UUID; берём первые 15 hex-символов sha1 как int
        return int(chunk_id[:15], 16)
