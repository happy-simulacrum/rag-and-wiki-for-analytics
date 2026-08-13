from codeqa.config import Config, LLMConfig
from codeqa.diag import probe_context, run_diag
from codeqa.llm import LLMClient


def _cfg(url: str) -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        base_url=url, api_key="sk-test", chat_model="mock-qwen", embed_model="mock-qwen"
    )
    return cfg


def test_diag_all_pass(mock_llm_url):
    report = run_diag(_cfg(mock_llm_url))
    names = [c.name for c in report.checks]
    assert names == ["models", "chat", "embeddings"]
    assert report.ok, [c.detail for c in report.checks]


def test_diag_fails_on_wrong_url():
    report = run_diag(_cfg("http://127.0.0.1:1"))  # порт закрыт
    assert not report.ok


def test_context_probe(mock_llm_url, monkeypatch):
    monkeypatch.setenv("MOCK_CONTEXT_TOKENS", "4096")
    with LLMClient(_cfg(mock_llm_url).llm) as client:
        check = probe_context(client, candidates=[1024, 2048, 8192, 16384])
    assert check.ok
    # 1024 и 2048 проходят (filler ~size токенов), 8192 — уже отказ
    assert "2048" in check.detail
