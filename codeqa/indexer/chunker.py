"""Чанкинг: tree-sitter по AST (определения функций/классов), fallback — окна."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Типы узлов-«определений» по языкам
DEF_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration", "generator_function_declaration",
        "class_declaration", "method_definition",
    },
    "typescript": {
        "function_declaration", "generator_function_declaration",
        "class_declaration", "method_definition", "interface_declaration",
        "enum_declaration", "type_alias_declaration",
    },
    "java": {
        "class_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "method_declaration", "constructor_declaration",
        "annotation_type_declaration",
    },
    "csharp": {
        "class_declaration", "interface_declaration", "struct_declaration",
        "enum_declaration", "record_declaration", "method_declaration",
        "constructor_declaration",
    },
    "c": {"function_definition", "struct_specifier", "enum_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "enum_specifier"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
}

# Контейнеры верхнего уровня, внутрь которых спускаемся на любом уровне
CONTAINER_TYPES = {
    "cpp": {"namespace_definition", "linkage_specification"},
    "csharp": {"namespace_declaration", "file_scoped_namespace_declaration"},
    "java": {"package_declaration"},
}

MAX_DEF_LINES = 200       # определение длиннее — дробим на вложенные/окна
WINDOW_LINES = 150        # скользящее окно для fallback
WINDOW_OVERLAP = 30

_parsers: dict[str, object] = {}


def _get_parser(language: str):
    if language in _parsers:
        return _parsers[language]
    import importlib

    from tree_sitter import Language, Parser

    modules = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "java": ("tree_sitter_java", "language"),
        "csharp": ("tree_sitter_c_sharp", "language"),
        "c": ("tree_sitter_c", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
        "go": ("tree_sitter_go", "language"),
    }
    mod_name, attr = modules[language]
    mod = importlib.import_module(mod_name)
    parser = Parser(Language(getattr(mod, attr)()))
    _parsers[language] = parser
    return parser


@dataclass
class Chunk:
    chunk_id: str
    project: str
    module: str
    relpath: str
    language: str
    symbol: str
    start_line: int
    end_line: int
    text: str


@dataclass
class _Ctx:
    project: str
    module: str
    relpath: str
    language: str
    def_types: set
    containers: set


def _make_chunk(ctx: _Ctx, symbol: str, start: int, end: int, text: str) -> Chunk:
    raw = f"{ctx.project}|{ctx.relpath}|{start}|{symbol}"
    chunk_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    header = f"# {ctx.relpath} :: {symbol}\n" if symbol else f"# {ctx.relpath}\n"
    return Chunk(
        chunk_id=chunk_id, project=ctx.project, module=ctx.module,
        relpath=ctx.relpath, language=ctx.language, symbol=symbol,
        start_line=start, end_line=end, text=header + text,
    )


def _window_chunks(ctx: _Ctx, lines: list[str], base: int = 0):
    start = 0
    while start < len(lines):
        end = min(start + WINDOW_LINES, len(lines))
        text = "\n".join(lines[start:end])
        yield _make_chunk(
            ctx, f"lines {base + start + 1}-{base + end}",
            base + start + 1, base + end, text,
        )
        if end == len(lines):
            break
        start = end - WINDOW_OVERLAP


def _node_symbol(node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", errors="ignore")
    return ""


def _definition_of(node):
    """decorated_definition → внутреннее определение (у него есть имя)."""
    if node.type == "decorated_definition":
        for child in node.named_children:
            if child.type != "decorator":
                return child
    return None


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _handle_def(node, source: bytes, ctx: _Ctx):
    """Чанк определения; слишком большое — дробим на вложенные defs/окна."""
    inner = _definition_of(node)
    n_lines = node.end_point[0] - node.start_point[0] + 1
    if n_lines <= MAX_DEF_LINES:
        yield _make_chunk(
            ctx, _node_symbol(inner or node),
            node.start_point[0] + 1, node.end_point[0] + 1, _node_text(node, source),
        )
        return
    sub: list[Chunk] = []
    for child in node.named_children:
        if child.type in ctx.def_types:
            sub.extend(_handle_def(child, source, ctx))
        else:
            sub.extend(_search_defs(child, source, ctx))
    if sub:
        yield from sub
    else:  # гигантская функция без вложенных определений — окнами
        yield from _window_chunks(
            ctx, _node_text(node, source).splitlines(), base=node.start_point[0]
        )


def _search_defs(node, source: bytes, ctx: _Ctx):
    """Спускаемся в любые узлы в поиске вложенных определений."""
    for child in node.named_children:
        if child.type in ctx.def_types:
            yield from _handle_def(child, source, ctx)
        else:
            yield from _search_defs(child, source, ctx)


def _chunk_top(node, source: bytes, ctx: _Ctx, depth: int):
    for child in node.named_children:
        if child.type in ctx.def_types:
            yield from _handle_def(child, source, ctx)
        elif child.type in ctx.containers or depth == 0:
            yield from _chunk_top(child, source, ctx, depth + 1)


def chunk_text(project: str, module: str, relpath: str, language: str, text: str) -> list[Chunk]:
    if language == "text" or language not in DEF_TYPES:
        ctx = _Ctx(project, module, relpath, "text", set(), set())
        return list(_window_chunks(ctx, text.splitlines()))
    source = text.encode("utf-8", errors="ignore")
    parser = _get_parser(language)
    tree = parser.parse(source)
    ctx = _Ctx(
        project, module, relpath, language,
        DEF_TYPES[language], CONTAINER_TYPES.get(language, set()),
    )
    chunks = list(_chunk_top(tree.root_node, source, ctx, depth=0))
    if not chunks:  # парсер ничего не нашёл — fallback окнами
        chunks = list(_window_chunks(ctx, text.splitlines()))
    return chunks
