#!/usr/bin/env bash
# deploy.sh — однокомандное развёртывание codeqa в закрытом контуре.
# Идемпотентен: повторный запуск безопасен. Лог — deploy.log.
set -euo pipefail
umask 077
cd "$(dirname "$0")"

LOG=deploy.log
exec > >(tee -a "$LOG") 2>&1

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
say()  { echo "${GREEN}==>${RESET} $*"; }
warn() { echo "${YELLOW}!! ${RESET} $*"; }
die()  { echo "${RED}XX ${RESET} $*"; exit 1; }

BACKEND_IMAGE="${CODEQA_BACKEND_IMAGE:-codeqa-backend:0.1.0}"
[[ -f backend-image.env ]] && source backend-image.env \
  && BACKEND_IMAGE="$CODEQA_BACKEND_IMAGE"
ENV_FILE=.env
CONFIG_FILE=config.yaml

# ---------- 0. pre-flight ----------
say "Проверка окружения"
command -v docker >/dev/null || die "docker не найден"
docker compose version >/dev/null 2>&1 || die "docker compose plugin не найден"
df -k . | awk 'NR==2 {if ($4 < 3*1024*1024) exit 1}' || warn "свободного места < 3 ГБ"

# ---------- 1. загрузка образов ----------
if [[ -d images ]]; then
  say "Загрузка образов из ./images"
  for tar in images/*.tar; do
    [[ -e "$tar" ]] || continue
    docker load -i "$tar"
  done
else
  warn "каталог images/ не найден — предполагаю, что образы уже загружены"
fi

# ---------- 2. конфигурация ----------
# неинтерактивный режим: ./deploy.sh --config existing.env
if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
  cp "$2" "$ENV_FILE"
fi

prompt() {  # prompt VAR "вопрос" [default]
  local var="$1" question="$2" default="${3:-}"
  local current="${!var:-$default}"
  if [[ -n "${NONINTERACTIVE:-}" ]]; then
    printf -v "$var" '%s' "$current"; return
  fi
  read -r -p "$question [$current]: " answer
  printf -v "$var" '%s' "${answer:-$current}"
}

prompt_secret() {  # как prompt, но значение не показывается и не попадает в лог
  local var="$1" question="$2"
  local current="${!var:-}"
  if [[ -n "${NONINTERACTIVE:-}" ]]; then
    printf -v "$var" '%s' "$current"; return
  fi
  local hint=""
  [[ -n "$current" ]] && hint="(Enter — оставить сохранённый)"
  read -r -s -p "$question $hint: " answer
  echo
  printf -v "$var" '%s' "${answer:-$current}"
}

[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a

say "Конфигурация (Enter — оставить значение в скобках)"
prompt LITELLM_URL "URL LiteLLM (OpenAI-совместимый API)" "http://llm.local:4000"
prompt_secret LITELLM_KEY "API-ключ LiteLLM"
prompt CHAT_MODEL  "Имя чат-модели" "qwen3.5"
prompt EMBED_MODEL "Имя модели эмбеддингов" "$CHAT_MODEL"
prompt REPOS_ROOT  "Корень репозиториев на этой VM" "/srv/repos"
prompt WEB_PORT    "Порт веб-интерфейса" "8080"
prompt ENABLE_LDAP "Включить LDAP-авторизацию? (true/false)" "false"
if [[ "$ENABLE_LDAP" == "true" ]]; then
  prompt LDAP_HOST "LDAP host" "ldap.local"
  prompt LDAP_PORT "LDAP port" "389"
  prompt LDAP_APP_DN "LDAP bind DN (service account)" ""
  prompt_secret LDAP_APP_PASSWORD "LDAP bind password"
  prompt LDAP_SEARCH_BASE "LDAP search base (ou=users,dc=...)" ""
fi

cat > "$ENV_FILE" <<EOF
LITELLM_URL=$LITELLM_URL
LITELLM_KEY=$LITELLM_KEY
CHAT_MODEL=$CHAT_MODEL
EMBED_MODEL=$EMBED_MODEL
REPOS_ROOT=$REPOS_ROOT
WEB_PORT=$WEB_PORT
ENABLE_LDAP=$ENABLE_LDAP
LDAP_HOST=${LDAP_HOST:-}
LDAP_PORT=${LDAP_PORT:-389}
LDAP_APP_DN=${LDAP_APP_DN:-}
LDAP_APP_PASSWORD=${LDAP_APP_PASSWORD:-}
LDAP_SEARCH_BASE=${LDAP_SEARCH_BASE:-}
LDAP_SEARCH_FILTER=${LDAP_SEARCH_FILTER:-}
CODEQA_BACKEND_IMAGE=$BACKEND_IMAGE
EOF
chmod 600 "$ENV_FILE" "$CONFIG_FILE"   # там api_key и пароли
say "Записан $ENV_FILE"

cat > "$CONFIG_FILE" <<EOF
llm:
  base_url: "$LITELLM_URL"     # переопределяется env в compose
  api_key: "$LITELLM_KEY"
  chat_model: "$CHAT_MODEL"
  embed_model: "$EMBED_MODEL"
  max_context_tokens: 256000
  answer_context_budget: 200000
paths:
  data_dir: "/data"
  repos_root: "/repos"
qdrant_url: "http://qdrant:6333"
web:
  port: 8080
EOF
say "Записан $CONFIG_FILE"

mkdir -p data

# ---------- 3. самопроверка LLM ----------
say "diag: проверка связи с LLM"
set +e
docker run --rm --network host \
  -v "$PWD/$CONFIG_FILE:/config/config.yaml:ro,z" \
  "$BACKEND_IMAGE" diag
DIAG_RC=$?
set -e
if [[ $DIAG_RC -ne 0 ]]; then
  die "diag нашёл проблемы (см. выше). Исправьте .env и перезапустите deploy.sh"
fi
read -r -p "Замерить фактический лимит контекста? Тяжёлые запросы. [y/N]: " probe
if [[ "${probe:-N}" =~ ^[yYдД]$ ]]; then
  docker run --rm --network host \
    -v "$PWD/$CONFIG_FILE:/config/config.yaml:ro,z" \
    "$BACKEND_IMAGE" diag --probe-context || true
fi

# ---------- 4. запуск ----------
say "Запуск сервисов"
docker compose --env-file "$ENV_FILE" up -d

say "Ожидание готовности backend"
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8088/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8088/health >/dev/null || die "backend не поднялся — см. docker compose logs backend"

cat <<EOF

${GREEN}Развёртывание завершено.${RESET}

Веб-интерфейс:  http://$(hostname -f 2>/dev/null || hostname):$WEB_PORT
Backend API:    http://127.0.0.1:8088/v1 (health: /health)

Следующие шаги (руководитель разработки):
  1. Добавить проект:
     docker compose exec backend codeqa --config /config/config.yaml \
       project add <имя> --path /repos/<репозиторий> --aliases "алиас1,алиас2"
     (индексация запустится сразу; повтор — project reindex <имя>)
  2. Проверить из консоли:
     ... ask "вопрос" 
  3. Периодические операции: wiki faq --project <имя>, wiki lint --project <имя>

Лог развёртывания: $LOG
EOF
