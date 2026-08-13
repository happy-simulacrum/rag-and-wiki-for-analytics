"""Индексация: walker → chunker → embed → Qdrant + FTS5; затем wiki-фаза."""

from .pipeline import IndexPipeline, IndexStats

__all__ = ["IndexPipeline", "IndexStats"]
