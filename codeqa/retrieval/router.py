"""Роутер проектов: к какому проекту относится вопрос аналитика.

Гибридная стратегия (по плану):
1. ответ на предыдущее уточнение (цифра/название) — из истории диалога;
2. точное упоминание имени/алиаса проекта → молча;
3. эмбеддинг вопроса против карточек проектов → молча при уверенности,
   иначе — уточняющий нумерованный список.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from codeqa.config import Config
from codeqa.llm import LLMClient
from codeqa.registry import Project, load_registry
from codeqa.util import cosine as _cosine

CLARIFY_MARKER = "Уточните проект"


@dataclass
class RouteResult:
    project: Project | None
    candidates: list[Project] = field(default_factory=list)
    reason: str = ""


def _mentions(text: str, project: Project) -> bool:
    """Упоминание проекта: точное слово или слово с окончанием (склонение).

    Алиасы длиной >= 4 матчатся по префиксу слова: «биллинге» → «биллинг».
    """
    words = re.findall(r"\w+", text.lower())
    for name in [project.name, *project.aliases]:
        if not name:
            continue
        low = name.lower()
        for w in words:
            if w == low or (len(low) >= 4 and w.startswith(low)):
                return True
    return False


class ProjectRouter:
    def __init__(self, cfg: Config, llm: LLMClient):
        self._cfg = cfg
        self._llm = llm
        self._card_vecs: dict[str, list[float]] | None = None

    @property
    def projects(self) -> list[Project]:
        return load_registry(self._cfg.paths.data_dir)

    def _card_text(self, p: Project) -> str:
        path = Path(self._cfg.paths.data_dir) / "wiki" / p.name / "overview.md"
        if path.exists():
            return path.read_text(encoding="utf-8")[:3000]
        return f"{p.name} {p.description} {' '.join(p.aliases)}"

    def _card_vectors(self, projects: list[Project]) -> dict[str, list[float]]:
        # кэш инвалидируется при изменении состава проектов (add/remove на ходу)
        if self._card_vecs is None or set(self._card_vecs) != {p.name for p in projects}:
            texts = [self._card_text(p) for p in projects]
            vecs = self._llm.embed(texts) if texts else []
            self._card_vecs = {p.name: v for p, v in zip(projects, vecs)}
        return self._card_vecs

    # ---- основная логика ----

    def route(self, messages: list[dict]) -> RouteResult:
        projects = self.projects
        if not projects:
            return RouteResult(None, [], "реестр проектов пуст")
        if len(projects) == 1:
            return RouteResult(projects[0], projects, "единственный проект")

        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last = user_msgs[-1] if user_msgs else ""

        # 1) ответ на предыдущее уточнение
        resolved = self._resolve_clarification(messages, last)
        if resolved is not None:
            return RouteResult(resolved, [resolved], "выбор из уточнения")

        # 2) упоминание имени/алиаса (сначала в последнем сообщении, потом в истории)
        for text, tag in ((last, "упоминание в вопросе"),
                          (" ".join(user_msgs[:-1]), "упоминание в истории")):
            if not text:
                continue
            hits = [p for p in projects if _mentions(text, p)]
            if len(hits) == 1:
                return RouteResult(hits[0], hits, tag)
            if len(hits) > 1:
                return RouteResult(None, hits, "несколько проектов упомянуто")

        # 3) эмбеддинг против карточек
        if not last:
            return RouteResult(None, projects[:3], "пустой вопрос")
        qvec = self._llm.embed([last])[0]
        card_vecs = self._card_vectors(projects)
        scored = sorted(
            ((p, _cosine(qvec, card_vecs[p.name])) for p in projects),
            key=lambda x: x[1], reverse=True,
        )
        top, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        rt = self._cfg.retrieval
        if top_score >= rt.route_silent_threshold and top_score - second_score >= rt.route_margin:
            return RouteResult(top, [top], f"эмбеддинг ({top_score:.2f})")
        return RouteResult(
            None, [p for p, _ in scored[:3]],
            f"неоднозначно ({top_score:.2f} vs {second_score:.2f})",
        )

    # ---- уточнения ----

    def _resolve_clarification(self, messages: list[dict], last_user: str) -> Project | None:
        """Последнее assistant-сообщение со списком + ответ пользователя цифрой/именем."""
        clarify_msg = None
        for m in reversed(messages[:-1] if last_user else messages):
            if m.get("role") == "assistant" and CLARIFY_MARKER in m.get("content", ""):
                clarify_msg = m["content"]
                break
            if m.get("role") == "user" and m.get("content", "") != last_user:
                break  # была ещё реплика пользователя после уточнения — не наш случай
        if not clarify_msg or not last_user.strip():
            return None
        listed = re.findall(r"(?m)^\s*(\d+)\)\s*\*{0,2}([\w.\-]+)\*{0,2}", clarify_msg)
        if not listed:
            return None
        by_number = {n: name for n, name in listed}
        answer = last_user.strip().rstrip(".")
        projects = self.projects
        if answer in by_number:
            return self._find(by_number[answer], projects)
        for name in by_number.values():
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", answer, re.IGNORECASE):
                return self._find(name, projects)
        return None

    @staticmethod
    def _find(name: str, projects: list[Project]) -> Project | None:
        for p in projects:
            if p.name == name:
                return p
        return None

    @staticmethod
    def clarification_message(candidates: list[Project]) -> str:
        lines = [
            f"Не могу однозначно определить проект. {CLARIFY_MARKER}:",
            "",
        ]
        for i, p in enumerate(candidates, 1):
            desc = f" — {p.description}" if p.description else ""
            lines.append(f"{i}) **{p.name}**{desc}")
        lines.append("")
        lines.append("Ответьте номером или названием проекта.")
        return "\n".join(lines)
