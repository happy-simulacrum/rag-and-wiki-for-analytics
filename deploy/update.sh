#!/usr/bin/env bash
# update.sh — накат нового bundle без потери данных (volumes не трогаем).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Загрузка новых образов"
for tar in images/*.tar; do
  [[ -e "$tar" ]] || continue
  docker load -i "$tar"
done

echo "==> Пересоздание сервисов (данные в volumes сохраняются)"
docker compose --env-file .env up -d --force-recreate

echo "==> Готово. Проверка:"
curl -fsS http://127.0.0.1:8088/health && echo " backend OK"
