# codeqa — единое окно Q&A по кодовым базам

Веб-интерфейс для аналитиков: вопрос на русском → ответ по кодовой базе
с цитатами `файл:строка`. Гибрид: RAG (FTS5 + векторный поиск) + wiki-слой
(карточки проектов, FAQ, concepts). LLM — корпоративная Qwen через LiteLLM
(чат и эмбеддинги одной моделью).

## Развёртывание (закрытый контур, Docker уже установлен)

```bash
tar -xzf codeqa-bundle-0.1.0.tar.gz
cd bundle
./deploy.sh
```

Скрипт проверит окружение, загрузит образы, спросит параметры (URL/ключ
LiteLLM, имена моделей, корень репозиториев, порт, LDAP), прогонит `diag`
(проверка чата, `/v1/embeddings`, опционально — замер лимита контекста)
и поднимет сервисы. Повторный запуск безопасен. Лог — `deploy.log`.

Обновление: распаковать новый bundle рядом, `./update.sh` — данные
(индексы, вики, лог вопросов) сохраняются в docker volumes.

## Сервисы

| Сервис     | Назначение                                   | Порт          |
|------------|----------------------------------------------|---------------|
| open-webui | веб-чат для аналитиков (LDAP)                | `$WEB_PORT`   |
| backend    | OpenAI-совместимый API, роутинг, RAG, wiki   | `127.0.0.1:8088` |
| qdrant     | векторный индекс                             | внутренний    |

Open WebUI подключён к backend как OpenAI-совместимая модель
`codeqa-assistant`. Встроенный RAG Open WebUI не используется.

### LDAP (Open WebUI)

`deploy.sh` заполняет переменные `ENABLE_LDAP`, `LDAP_SERVER_HOST`,
`LDAP_SERVER_PORT`, `LDAP_APP_DN`, `LDAP_APP_PASSWORD`, `LDAP_SEARCH_BASE`,
`LDAP_SEARCH_FILTER`, `LDAP_ATTRIBUTE_FOR_USERNAME/MAIL` в `.env`.
Если схема вашего LDAP отличается — отредактируйте `.env` и
`docker compose up -d open-webui`. Имена переменных соответствуют
документации Open WebUI v0.6.x — при расхождениях сверяйтесь с ней.

## Работа руководителя разработки

Все команды — внутри контейнера backend (алиасы ради краткости опущены):

```bash
CD="docker compose exec backend codeqa --config /config/config.yaml"

# добавить проект (репозитории примонтированы в /repos, read-only)
$CD project add billing --path /repos/billing --aliases "биллинг,billing-core"

# после обновления исходников — переиндексация (RAG + wiki, инкрементально)
$CD project reindex billing            # или --full для полной

# список / правка / удаление
$CD project list
$CD project update billing --aliases "биллинг,билинг"
$CD project remove billing

# консольная проверка пайплайна без веб-интерфейса
$CD ask "как рассчитывается стоимость заказа?" 

# wiki-операции
$CD wiki faq --project billing         # FAQ по частоте вопросов (авто, лимит 50)
$CD wiki lint --project billing        # битые цитаты, противоречия

# самопроверка связи с LLM
$CD diag                               # + --probe-context для замера контекста
```

## Как это работает

1. **Роутер проектов**: по упоминанию имени/алиаса (со склонениями) или по
   близости вопроса к карточкам проектов. При неоднозначности — уточняющий
   нумерованный список; выбор запоминается в рамках диалога.
2. **Wiki-first**: если похожий вопрос уже есть в `faq.md`/`concepts/` —
   контекст вики идёт в ответ первым.
3. **Гибридный ретрив**: идентификаторы из вопроса → точный поиск FTS5
   (trigram) + векторный поиск Qdrant → RRF-слияние, буст точных символов →
   упаковка до ~200k токенов в контекст.
4. **Ответ**: LLM отвечает только по контексту, с цитатами `файл:строка`,
   при недостатке данных говорит «не знаю». Источники прилагаются.
5. **Индексация** — только вручную (`project reindex`): walker (сабмодули,
   фильтры) → tree-sitter чанкинг (Java/C#/C++/Python/JS/Go/TS) → эмбеддинги
   → Qdrant + FTS5; затем wiki-фаза обновляет карточку проекта, index.md,
   log.md.
6. **FAQ**: вопросы логируются; `wiki faq` кластеризует их по эмбеддингам,
   топ-частотные (≤50 записей, ≤20k токенов) получают свежие ответы в faq.md.

## Разработка (локально, без контура)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q
.venv/bin/codeqa mock &          # mock LLM (чат + эмбеддинги) на :8399
.venv/bin/codeqa diag --config config.example.yaml
```

Конфигурация: `config.yaml` (см. `config.example.yaml`) или переменные
окружения `CODEQA_*`. Сборка bundle для контура: `make bundle`.

## Структура

```
codeqa/
├── llm/           # OpenAI-клиент + mock-сервер
├── indexer/       # walker, chunker (tree-sitter), pipeline, git, wiki-фаза
├── retrieval/     # идентификаторы, гибридный ретрив, упаковка, роутер
├── store/         # SQLite (FTS5, метаданные, лог) + Qdrant
├── backend/       # FastAPI OpenAI-совместимый endpoint
├── faq.py, lint.py, wiki_search.py, answer.py, diag.py, cli.py
deploy/            # Dockerfile, compose, deploy.sh, update.sh
tests/             # pytest (mock LLM, фикстурные репозитории)
```
