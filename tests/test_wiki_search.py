"""Тесты wiki-first поиска: split_faq, пороги, пустая вики."""

from pathlib import Path

from codeqa.config import Config, LLMConfig, PathsConfig
from codeqa.llm import LLMClient
from codeqa.wiki_search import WikiSearch, _split_faq


def _cfg(tmp_path, mock_url) -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        base_url=mock_url, api_key="sk-test",
        chat_model="mock-qwen", embed_model="mock-qwen",
    )
    cfg.paths = PathsConfig(data_dir=str(tmp_path / "data"), repos_root=str(tmp_path / "repos"))
    return cfg


def test_split_faq():
    text = (
        "# FAQ проекта\n\n"
        "## Как считается налог?\n\nОтвет про налог.\n\n"
        "## Что такое скидка\n\nОтвет про скидку.\n"
    )
    entries = _split_faq(text)
    assert len(entries) == 2
    assert entries[0][0] == "Как считается налог?"
    assert entries[0][1] == "Ответ про налог."
    assert entries[1][0] == "Что такое скидка"


def test_search_empty_wiki(indexed):
    """Нет faq.md и concepts/ → пустой результат без вызова LLM."""
    cfg = indexed["cfg"]
    with LLMClient(cfg.llm) as llm:
        hits = WikiSearch(cfg, llm).search("no-such-project", "вопрос", threshold=0.5)
    assert hits == []


def test_search_filters_by_threshold(indexed, tmp_path, mock_llm_url):
    cfg = _cfg(tmp_path, mock_llm_url)
    wiki_dir = Path(cfg.paths.data_dir) / "wiki" / "billing"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "faq.md").write_text(
        "# FAQ\n\n## налоги\n\nрасчёт налога НДС ставка\n\n"
        "## доставка\n\nкурьер перевозка посылка\n",
        encoding="utf-8",
    )
    with LLMClient(cfg.llm) as llm:
        search = WikiSearch(cfg, llm)
        # нулевой порог — записи проходят; нерелевантный вопрос отсеивается высоким
        all_hits = search.search("billing", "налог", threshold=0.0)
        assert len(all_hits) == 2
        strict = search.search("billing", "полностью неродственное слово", threshold=0.99)
        assert strict == []
