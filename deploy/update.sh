#!/usr/bin/env bash
# update.sh — накат нового bundle без потери данных.
# Конфиги и bind-mounts (.env, config.yaml, data/) живут в каталоге
# ПРЕДЫДУЩЕГО деплоя: update.sh копирует туда новый compose-файл и
# пересоздаёт сервисы оттуда. Volumes (qdrant, open-webui) не трогаем.
#
# Использование: ./update.sh [каталог_старого_деплоя]
#   Без аргумента — ищется единственный соседний каталог с .env.
set -euo pipefail
cd "$(dirname "$0")"

NEW_DIR="$PWD"

if [[ -n "${1:-}" ]]; then
  OLD_DIR="$(cd "$1" && pwd)"
elif [[ -f .env && -f docker-compose.yml ]]; then
  OLD_DIR="$NEW_DIR"          # обновление прямо поверх этого каталога
else
  OLD_DIR=""
  while IFS= read -r d; do
    if [[ -z "$OLD_DIR" ]]; then OLD_DIR="$d"; else
      echo "XX Найдено несколько каталогов деплоя рядом; укажите нужный:" >&2
      echo "   $0 <каталог_старого_деплоя>" >&2
      exit 1
    fi
  done < <(for d in ../*/; do
    d="${d%/}"; [[ -f "$d/.env" && -f "$d/docker-compose.yml" ]] && readlink -f "$d"
  done)
  if [[ -z "$OLD_DIR" ]]; then
    echo "XX Не найден предыдущий каталог деплоя (с .env и docker-compose.yml)." >&2
    echo "   Запустите: $0 <каталог_старого_деплоя>" >&2
    exit 1
  fi
fi

echo "==> Каталог деплоя: $OLD_DIR"

echo "==> Загрузка новых образов"
for tar in images/*.tar; do
  [[ -e "$tar" ]] || continue
  docker load -i "$tar"
done

echo "==> Обновление compose-файла"
cp -v "$NEW_DIR/docker-compose.yml" "$OLD_DIR/docker-compose.yml"
if [[ -f "$NEW_DIR/backend-image.env" ]]; then
  cp -v "$NEW_DIR/backend-image.env" "$OLD_DIR/"
  source "$NEW_DIR/backend-image.env"
  if grep -q '^CODEQA_BACKEND_IMAGE=' "$OLD_DIR/.env"; then
    sed -i "s|^CODEQA_BACKEND_IMAGE=.*|CODEQA_BACKEND_IMAGE=$CODEQA_BACKEND_IMAGE|" "$OLD_DIR/.env"
  else
    echo "CODEQA_BACKEND_IMAGE=$CODEQA_BACKEND_IMAGE" >> "$OLD_DIR/.env"
  fi
fi

echo "==> Пересоздание сервисов (данные сохраняются)"
(cd "$OLD_DIR" && docker compose --env-file .env up -d --force-recreate)

echo "==> Готово. Проверка:"
curl -fsS http://127.0.0.1:8088/health && echo " backend OK"
