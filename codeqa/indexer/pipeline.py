"""Пайплайн индексации: RAG-фаза (чанки → эмбеддинги → хранилища) + wiki-фаза."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from codeqa.config import Config
from codeqa.indexer import gitutil
from codeqa.indexer.chunker import Chunk, chunk_text
from codeqa.indexer.walker import SourceFile, walk_repo
from codeqa.indexer.wiki import append_log, update_index, update_overview, wiki_dir
from codeqa.llm import LLMClient
from codeqa.registry import Project
from codeqa.store import ChunkStore, VectorStore

log = logging.getLogger("codeqa.indexer")

EMBED_BATCH = 32


@dataclass
class IndexStats:
    project: str
    files_seen: int = 0
    files_deleted: int = 0
    chunks_indexed: int = 0
    commit: str = ""
    incremental: bool = False
    duration_sec: float = 0.0
    changed_files: list[str] = field(default_factory=list)


class IndexPipeline:
    def __init__(self, cfg: Config, llm: LLMClient):
        self._cfg = cfg
        self._llm = llm
        data_dir = Path(cfg.paths.data_dir)
        self._chunks_db = ChunkStore(data_dir / "index.sqlite")
        qdrant_url = getattr(cfg, "qdrant_url", "") or ""
        self._vectors = (
            VectorStore(url=qdrant_url)
            if qdrant_url
            else VectorStore(local_path=str(data_dir / "qdrant"))
        )

    def close(self) -> None:
        self._chunks_db.close()
        self._vectors.close()

    def __enter__(self) -> "IndexPipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(self, project: Project, full: bool = False) -> IndexStats:
        start = time.monotonic()
        repo = Path(project.path).resolve()
        stats = IndexStats(project=project.name)

        state = self._chunks_db.get_state(project.name)
        head = gitutil.current_commit(repo) if gitutil.is_git_repo(repo) else None
        stats.commit = head or ""

        incremental = bool(state and head and not full)
        stats.incremental = incremental

        if incremental:
            changed, deleted = gitutil.changed_files(repo, state["last_commit"])
            stats.changed_files = changed
            removed_ids = self._chunks_db.delete_files(project.name, deleted + changed)
            self._vectors.delete_chunks(project.name, removed_ids)
            stats.files_deleted = len(deleted)
            only = set(changed)
        else:
            # полную переиндексацию делаем без предварительного удаления:
            # старые чанки живут до успешной записи новых (сбой LLM не оставит
            # пустой индекс), устаревшее чистится в конце по разности id
            old_ids = self._chunks_db.all_chunk_ids(project.name)
            changed = None  # все файлы
            only = None

        files = walk_repo(repo, only_relpaths=only)
        stats.files_seen = len(files)

        chunks: list[Chunk] = []
        for sf in files:
            text = sf.abspath.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(
                chunk_text(project.name, sf.module, sf.relpath, sf.language, text)
            )

        new_ids: set[str] = set()
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            vectors = self._llm.embed([c.text for c in batch])
            self._chunks_db.upsert_chunks(batch)
            self._vectors.upsert(project.name, batch, vectors)
            stats.chunks_indexed += len(batch)
            new_ids.update(c.chunk_id for c in batch)

        if not incremental and old_ids:
            stale = sorted(old_ids - new_ids)
            if stale:
                self._chunks_db.delete_chunk_ids(project.name, stale)
                self._vectors.delete_chunks(project.name, stale)
                log.info("'%s': удалено устаревших чанков %d", project.name, len(stale))

        total = self._chunks_db.count_chunks(project.name)
        self._chunks_db.set_state(project.name, head or "", total)

        # ---- wiki-фаза ----
        all_files = files if only is None else walk_repo(repo)
        out_dir = wiki_dir(self._cfg.paths.data_dir, project.name)
        update_overview(self._llm, project, all_files, changed, out_dir)
        append_log(
            out_dir,
            f"Индексация ({'инкремент' if incremental else 'полная'}): "
            f"файлов {stats.files_seen}, чанков {stats.chunks_indexed}, "
            f"удалено файлов {stats.files_deleted}, коммит {stats.commit[:12] or '—'}.",
        )
        update_index(out_dir)

        stats.duration_sec = time.monotonic() - start
        return stats

    def drop_project(self, project_name: str) -> None:
        self._chunks_db.drop_project(project_name)
        self._vectors.drop_project(project_name)
