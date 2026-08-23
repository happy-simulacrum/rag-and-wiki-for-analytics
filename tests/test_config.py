from codeqa.config import load_config


def test_load_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "llm:\n  base_url: http://example:1234\n  chat_model: qwen3.5\n"
        "paths:\n  data_dir: /tmp/data\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.llm.base_url == "http://example:1234"
    assert cfg.paths.data_dir == "/tmp/data"
    # значения по умолчанию сохраняются
    assert cfg.retrieval.faq_max_entries == 50


def test_env_overrides(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("llm:\n  base_url: http://example:1234\n", encoding="utf-8")
    monkeypatch.setenv("CODEQA_LLM_BASE_URL", "http://override:9999")
    monkeypatch.setenv("CODEQA_LLM_API_KEY", "sk-test")
    cfg = load_config(cfg_file)
    assert cfg.llm.base_url == "http://override:9999"
    assert cfg.llm.api_key == "sk-test"


def test_embed_endpoint_env_overrides(tmp_path, monkeypatch):
    """Отдельный endpoint эмбеддингов через CODEQA_EMBED_*; основной не задет."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("llm:\n  base_url: http://main:4000\n", encoding="utf-8")
    monkeypatch.setenv("CODEQA_EMBED_BASE_URL", "http://embed-api:8000")
    monkeypatch.setenv("CODEQA_EMBED_API_KEY", "sk-embed-only")
    cfg = load_config(cfg_file)
    assert cfg.llm.embed_base_url == "http://embed-api:8000"
    assert cfg.llm.embed_api_key == "sk-embed-only"
    assert cfg.llm.base_url == "http://main:4000"


def test_unknown_key_rejected(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("llm:\n  bogus_key: 1\n", encoding="utf-8")
    try:
        load_config(cfg_file)
    except ValueError as e:
        assert "bogus_key" in str(e)
    else:
        raise AssertionError("ожидали ValueError")


def test_no_config_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # нет config.yaml рядом
    monkeypatch.delenv("CODEQA_CONFIG", raising=False)
    cfg = load_config(None)
    assert cfg.llm.chat_model  # дефолт из датакласса
