"""FAQ: кластеризация лога вопросов по частоте → faq.md (авто, с лимитами)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from codeqa.answer import answer_question
from codeqa.config import Config
from codeqa.indexer.wiki import append_log, update_index, wiki_dir
from codeqa.llm import LLMClient
from codeqa.retrieval import est_tokens
from codeqa.store import ChunkStore, VectorStore

CLUSTER_COSINE = 0.85  # порог «это тот же вопрос»


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


@dataclass
class _Cluster:
    rep: str
    count: int
    last_ts: float
    centroid: list[float]
    n: int = field(default=1)

    def add(self, vec: list[float], ts: float) -> None:
        self.centroid = [
            (c * self.n + v) / (self.n + 1) for c, v in zip(self.centroid, vec)
        ]
        self.n += 1
        self.count += 1
        self.last_ts = max(self.last_ts, ts)


def clusterize(questions: list[dict]) -> list[_Cluster]:
    """Жадная кластеризация по косинусной близости эмбеддингов."""
    clusters: list[_Cluster] = []
    for q in questions:
        vec = q["embedding"]
        best, best_sim = None, CLUSTER_COSINE
        for c in clusters:
            sim = _cosine(vec, c.centroid)
            if sim >= best_sim:
                best, best_sim = c, sim
        if best is None:
            clusters.append(_Cluster(rep=q["question"], count=1, last_ts=q["ts"], centroid=vec))
        else:
            best.add(vec, q["ts"])
    clusters.sort(key=lambda c: (c.count, c.last_ts), reverse=True)
    return clusters


def build_faq(
    cfg: Config,
    llm: LLMClient,
    store: ChunkStore,
    vectors: VectorStore,
    project: str,
) -> dict:
    """Топ частых вопросов → свежие ответы → faq.md. Лимиты из конфига."""
    questions = store.get_questions(project)
    clusters = clusterize(questions)
    max_entries = cfg.retrieval.faq_max_entries
    max_tokens = cfg.retrieval.faq_max_tokens

    entries: list[str] = []
    used = est_tokens(f"# FAQ проекта {project}\n")
    for cluster in clusters[:max_entries]:
        result = answer_question(cfg, llm, store, vectors, project, cluster.rep)
        sources = ", ".join(
            f"`{s['relpath']}:{s['start_line']}`" for s in result["sources"][:5]
        )
        entry = (
            f"## {cluster.rep}\n\n_Спрашивали {cluster.count} раз._\n\n"
            f"{result['answer']}\n\nИсточники: {sources}\n"
        )
        t = est_tokens(entry)
        if used + t > max_tokens:
            break  # лимит по размеру: редкие вытесняются частыми
        entries.append(entry)
        used += t

    out_dir = wiki_dir(cfg.paths.data_dir, project)
    faq_path = out_dir / "faq.md"
    header = f"# FAQ проекта {project}\n\n_Автогенерация из лога вопросов. Записей: {len(entries)}._\n\n"
    faq_path.write_text(header + "\n".join(entries), encoding="utf-8")
    update_index(out_dir)
    append_log(out_dir, f"FAQ перегенерирован: вопросов {len(questions)}, "
                        f"кластеров {len(clusters)}, записей {len(entries)}.")
    return {
        "questions": len(questions),
        "clusters": len(clusters),
        "entries": len(entries),
        "path": str(faq_path),
    }
