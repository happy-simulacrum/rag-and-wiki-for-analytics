# AGENTS.md

## Что это

codeqa — «единое окно» Q&A по кодовым базам для аналитиков. Гибрид RAG
(FTS5 + Qdrant) и wiki-слоя по Карпатыху. LLM — корпоративная Qwen через
LiteLLM (чат И эмбеддинги одной моделью). Развёртывание — в закрытом контуре
(bundle + deploy.sh), разработка — здесь, на mock LLM.

## Команды

```bash
.venv/bin/python -m pytest tests -q        # тесты (36 шт., mock LLM)
.venv/bin/codeqa mock                      # mock LLM на :8399 (dev)
.venv/bin/codeqa diag --config <yaml>      # самопроверка LLM API
make bundle                                # собрать bundle для контура
```

## Архитектура (кратко)

- `codeqa/llm/` — OpenAI-совместимый клиент (`LLMClient`) + mock-сервер.
  Mock-эмбеддинги — bag-of-words хеширование (лексическая близость сохраняется).
- `codeqa/indexer/` — walker (сабмодули из .gitmodules, фильтры) → chunker
  (tree-sitter, defs ≤ 200 строк, fallback окна 150/30) → pipeline
  (RAG-фаза + wiki-фаза одной командой, инкремент по `git diff`).
- `codeqa/retrieval/` — identifiers (CamelCase/snake_case из вопроса),
  hybrid (FTS5 trigram + Qdrant, RRF k=60, буст символа ×3), packer
  (est: 4 символа/токен), router (алиасы со склонениями → эмбеддинг карточек →
  уточнение нумерованным списком; stateless, разбор ответа из истории).
- `codeqa/store/` — ChunkStore (SQLite + FTS5 + question_log), VectorStore
  (Qdrant: local path для dev, url для prod; коллекция на проект).
- `codeqa/backend/app.py` — FastAPI, OpenAI-совместимый /v1/chat/completions
  (+ SSE-стриминг одним чанком) для подключения Open WebUI.
- `codeqa/faq.py` — кластеризация question_log (cos ≥ 0.85), топ по частоте,
  лимиты retrieval.faq_max_entries/faq_max_tokens.
- `codeqa/lint.py` — битые цитаты (регекс + has_relpath) + LLM-ревизия.

## Конфигурация

`config.yaml` (см. config.example.yaml) + env `CODEQA_*` (LLM_BASE_URL,
LLM_API_KEY, LLM_CHAT_MODEL, LLM_EMBED_MODEL, DATA_DIR, REPOS_ROOT,
QDRANT_URL). `qdrant_url` пустой → Qdrant local-режим в data_dir/qdrant.

## Договорённости (из диалога с заказчиком)

- Эмбеддинги — только через основную Qwen (без отдельной модели).
- Индексация — только ручная (`project reindex`), по команде руководителя
  разработки; обновляет и RAG, и wiki.
- qa/ в вики НЕ ведём; faq.md — авто по частоте, без ручного подтверждения.
- Роутер: молча при уверенности, уточнение при неоднозначности.
- Ответы — по-русски, с цитатами `файл:строка`, честное «не знаю».

## Подводные камни

- SELinux: bind-mounts в compose/deploy.sh требуют суффикс `:z`.
- Qdrant local-режим — эксклюзивный лок на каталог: backend и CLI одновременно
  с одним data_dir не работают (в prod — qdrant_url, там проблемы нет).
- В образ backend обязательно ставить git (slim-образ его не имеет) —
  иначе инкрементальная индексация молча деградирует до полной.
- pgrep -f mock_server ловит собственную командную строку — проверять по
  порту (`ss -tlnp | grep 8399`).
