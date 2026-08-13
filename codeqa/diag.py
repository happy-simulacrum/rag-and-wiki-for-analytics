"""Самопроверка развёртывания: чат, эмбеддинги, контекст, задержки.

Запускается в контуре после deploy.sh; вывод можно прислать для диагностики.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from codeqa.config import Config
from codeqa.llm import LLMClient, LLMError

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    elapsed_ms: int = 0


@dataclass
class DiagReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _timed(fn):
    start = time.monotonic()
    result = fn()
    return result, int((time.monotonic() - start) * 1000)


def check_models(client: LLMClient) -> Check:
    try:
        models, ms = _timed(client.models)
        return Check("models", True, f"доступны: {', '.join(models) or '(пусто)'}", ms)
    except LLMError as e:
        return Check("models", False, str(e))


def check_chat(client: LLMClient) -> Check:
    try:
        answer, ms = _timed(
            lambda: client.chat(
                [{"role": "user", "content": "diag:ping"}], max_tokens=16
            )
        )
        ok = bool(answer.strip())
        return Check("chat", ok, f"ответ: {answer.strip()[:80]!r}", ms)
    except LLMError as e:
        return Check("chat", False, str(e))


def check_embeddings(client: LLMClient) -> Check:
    try:
        vectors, ms = _timed(lambda: client.embed(["диагностика эмбеддингов codeqa"]))
        dim = len(vectors[0]) if vectors else 0
        return Check("embeddings", dim > 0, f"размерность: {dim}", ms)
    except LLMError as e:
        return Check("embeddings", False, str(e))


def probe_context(client: LLMClient, candidates: list[int]) -> Check:
    """Ступенчато растим вход, пока сервер не откажет. Возвращает max прошедший."""
    max_ok = 0
    start = time.monotonic()
    for size in candidates:
        filler = "код " * size  # ~4 символа на токен → ~size токенов
        try:
            client.chat([{"role": "user", "content": filler}], max_tokens=1)
            max_ok = size
        except LLMError:
            break
    ms = int((time.monotonic() - start) * 1000)
    if max_ok == 0:
        return Check("context", False, "не прошёл даже минимальный размер", ms)
    return Check("context", True, f"фактический лимит ≥ {max_ok} токенов", ms)


def run_diag(cfg: Config, with_context_probe: bool = False) -> DiagReport:
    report = DiagReport()
    with LLMClient(cfg.llm) as client:
        report.checks.append(check_models(client))
        report.checks.append(check_chat(client))
        report.checks.append(check_embeddings(client))
        if with_context_probe:
            candidates = [
                s
                for s in (8192, 32768, 65536, 131072, 200704, 262144)
                if s <= cfg.llm.max_context_tokens
            ]
            report.checks.append(probe_context(client, candidates))
    return report


def print_report(report: DiagReport) -> None:
    for c in report.checks:
        mark = f"{_GREEN}OK  {_RESET}" if c.ok else f"{_RED}FAIL{_RESET}"
        print(f"[{mark}] {c.name:<12} {c.elapsed_ms:>6} мс  {c.detail}")
    verdict = (
        f"{_GREEN}Все проверки пройдены{_RESET}"
        if report.ok
        else f"{_YELLOW}Есть проблемы — пришлите этот вывод разработчикам{_RESET}"
    )
    print(f"\nИтог: {verdict}")
