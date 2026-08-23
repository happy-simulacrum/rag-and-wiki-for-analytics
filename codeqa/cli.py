"""Точка входа CLI: codeqa <command>."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from codeqa.config import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(prog="codeqa", description="Единое окно Q&A по кодовым базам")
    parser.add_argument("--config", help="путь к config.yaml", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diag", help="самопроверка: чат, эмбеддинги, контекст")
    p_diag.add_argument(
        "--probe-context",
        action="store_true",
        help="замерить фактический лимит контекста (тяжёлые запросы)",
    )

    p_mock = sub.add_parser("mock", help="запустить mock LLM-сервер (разработка)")
    p_mock.add_argument("--host", default="127.0.0.1")
    p_mock.add_argument("--port", type=int, default=8399)

    p_project = sub.add_parser("project", help="управление проектами")
    psub = p_project.add_subparsers(dest="project_command", required=True)

    p_add = psub.add_parser("add", help="добавить проект и проиндексировать")
    p_add.add_argument("name")
    p_add.add_argument("--path", required=True, help="путь к репозиторию")
    p_add.add_argument("--aliases", default="", help="алиасы через запятую")
    p_add.add_argument("--description", default="")
    p_add.add_argument("--no-index", action="store_true", help="не индексировать сразу")

    psub.add_parser("list", help="список проектов")

    p_update = psub.add_parser("update", help="изменить путь/алиасы/описание")
    p_update.add_argument("name")
    p_update.add_argument("--path", default=None)
    p_update.add_argument("--aliases", default=None)
    p_update.add_argument("--description", default=None)

    p_remove = psub.add_parser("remove", help="удалить проект и его индекс")
    p_remove.add_argument("name")
    p_remove.add_argument("--yes", action="store_true", help="без подтверждения")

    p_reindex = psub.add_parser("reindex", help="переиндексировать (RAG + wiki)")
    p_reindex.add_argument("name")
    p_reindex.add_argument("--full", action="store_true", help="полная переиндексация")

    p_ask = sub.add_parser("ask", help="задать вопрос из консоли (проверка пайплайна)")
    p_ask.add_argument("question")
    p_ask.add_argument("--project", default=None, help="пропустить роутер")

    p_serve = sub.add_parser("serve", help="запустить бэкенд (OpenAI-совместимый API)")
    p_serve.add_argument("--host", default="127.0.0.1")  # наружу — только в контейнере
    p_serve.add_argument("--port", type=int, default=None)

    p_wiki = sub.add_parser("wiki", help="операции вики-слоя")
    wsub = p_wiki.add_subparsers(dest="wiki_command", required=True)

    p_faq = wsub.add_parser("faq", help="перегенерировать faq.md по частоте вопросов")
    p_faq.add_argument("--project", required=True)

    p_lint = wsub.add_parser("lint", help="проверка вики: битые цитаты, противоречия")
    p_lint.add_argument("--project", required=True)

    args = parser.parse_args()

    try:
        _dispatch(args)
    except KeyError as e:  # get_project: «Проект не найден: ...»
        print(f"Ошибка: {e.args[0]}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:  # валидация реестра и конфига
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


def _dispatch(args) -> None:
    if args.command == "diag":
        from codeqa.diag import print_report, run_diag

        cfg = load_config(args.config)
        report = run_diag(cfg, with_context_probe=args.probe_context)
        print_report(report)
        sys.exit(0 if report.ok else 1)

    if args.command == "mock":
        import uvicorn

        from codeqa.llm.mock_server import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    if args.command == "project":
        _run_project_command(args)
        return

    if args.command == "ask":
        _run_ask(args)
        return

    if args.command == "serve":
        import uvicorn

        from codeqa.backend import create_app

        cfg = load_config(args.config)
        uvicorn.run(
            create_app(cfg), host=args.host, port=args.port or cfg.web.port,
            log_level="info",
        )
        return

    if args.command == "wiki":
        _run_wiki_command(args)


def _run_wiki_command(args) -> None:
    from codeqa.faq import build_faq
    from codeqa.lint import lint_project
    from codeqa.llm import LLMClient
    from codeqa.registry import get_project
    from codeqa.store import ChunkStore, VectorStore

    cfg = load_config(args.config)
    data_dir = Path(cfg.paths.data_dir)
    get_project(data_dir, args.project)  # проверка, что проект существует
    store = ChunkStore(data_dir / "index.sqlite")
    vectors = (
        VectorStore(url=cfg.qdrant_url)
        if cfg.qdrant_url
        else VectorStore(local_path=str(data_dir / "qdrant"))
    )
    try:
        with LLMClient(cfg.llm) as llm:
            if args.wiki_command == "faq":
                stats = build_faq(cfg, llm, store, vectors, args.project)
                print(
                    f"faq.md обновлён: вопросов {stats['questions']}, "
                    f"кластеров {stats['clusters']}, записей {stats['entries']} "
                    f"→ {stats['path']}"
                )
            elif args.wiki_command == "lint":
                result = lint_project(cfg, llm, store, args.project)
                print(f"Битых цитат: {len(result['stale_citations'])}")
                print(result["report"])
    finally:
        store.close()
        vectors.close()


def _run_ask(args) -> None:
    from codeqa.answer import answer_question, format_sources
    from codeqa.llm import LLMClient
    from codeqa.registry import get_project
    from codeqa.retrieval.router import ProjectRouter
    from codeqa.store import ChunkStore, VectorStore
    from codeqa.wiki_search import WikiSearch

    cfg = load_config(args.config)
    data_dir = Path(cfg.paths.data_dir)
    with LLMClient(cfg.llm) as llm:
        store = ChunkStore(data_dir / "index.sqlite")
        vectors = (
            VectorStore(url=cfg.qdrant_url)
            if cfg.qdrant_url
            else VectorStore(local_path=str(data_dir / "qdrant"))
        )
        project_name = args.project
        if project_name is None:
            router = ProjectRouter(cfg, llm)
            route = router.route([{"role": "user", "content": args.question}])
            if route.project is None:
                print(router.clarification_message(route.candidates))
                return
            project_name = route.project.name
            print(f"[роутер → проект '{project_name}' ({route.reason})]\n")
        else:
            get_project(data_dir, project_name)  # проверка, что существует

        wiki_hits = WikiSearch(cfg, llm).search(
            project_name, args.question, threshold=cfg.retrieval.wiki_threshold
        )
        result = answer_question(
            cfg, llm, store, vectors, project_name, args.question, wiki_hits
        )
        store.close()
        vectors.close()
    print(result["answer"])
    print(format_sources(result["sources"]))
    print(f"\n[чанков: {result['chunks_used']}, контекст ~{result['context_tokens']} токенов]")


def _run_project_command(args) -> None:
    from codeqa.llm import LLMClient
    from codeqa.registry import (
        Project, add_project, get_project, load_registry, remove_project, update_project,
    )

    cfg = load_config(args.config)
    data_dir = cfg.paths.data_dir
    cmd = args.project_command

    if cmd == "list":
        projects = load_registry(data_dir)
        if not projects:
            print("Проектов нет. Добавьте: codeqa project add <name> --path <path>")
            return
        for p in projects:
            aliases = f" (алиасы: {', '.join(p.aliases)})" if p.aliases else ""
            print(f"{p.name}: {p.path}{aliases}")
        return

    if cmd == "add":
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
        add_project(data_dir, Project(
            name=args.name, path=args.path, aliases=aliases, description=args.description,
        ))
        print(f"Проект '{args.name}' добавлен в реестр.")
        if not args.no_index:
            _run_reindex(cfg, args.name, full=True)
        return

    if cmd == "update":
        aliases = None
        if args.aliases is not None:
            aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
        p = update_project(
            data_dir, args.name, path=args.path, aliases=aliases,
            description=args.description,
        )
        print(f"Проект '{p.name}' обновлён.")
        return

    if cmd == "remove":
        if not args.yes:
            answer = input(
                f"Удалить проект '{args.name}' и весь его индекс/вики? [y/N] "
            )
            if answer.lower() not in ("y", "yes"):
                print("Отменено.")
                return
        from codeqa.indexer import IndexPipeline

        with LLMClient(cfg.llm) as llm, _pipeline(cfg, llm) as pipe:
            pipe.drop_project(args.name)
        remove_project(data_dir, args.name)
        print(f"Проект '{args.name}' удалён (индекс очищен).")
        return

    if cmd == "reindex":
        _run_reindex(cfg, args.name, full=args.full)


def _pipeline(cfg, llm):
    from codeqa.indexer import IndexPipeline

    return IndexPipeline(cfg, llm)


def _run_reindex(cfg, name: str, full: bool) -> None:
    from codeqa.llm import LLMClient
    from codeqa.registry import get_project

    project = get_project(cfg.paths.data_dir, name)
    with LLMClient(cfg.llm) as llm, _pipeline(cfg, llm) as pipe:
        print(f"Индексация '{name}' ({'полная' if full else 'инкрементная'})...")
        stats = pipe.run(project, full=full)
    print(
        f"Готово за {stats.duration_sec:.1f} с: файлов {stats.files_seen}, "
        f"чанков {stats.chunks_indexed}, удалено файлов {stats.files_deleted}, "
        f"коммит {stats.commit[:12] or '—'}"
    )


if __name__ == "__main__":
    main()
