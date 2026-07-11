#!/usr/bin/env bash
# Rigenera il token Sanctum del bot Cognitor lato Laravel, lo scrive nell'.env di
# questo repo e riavvia il processo uvicorn locale perché lo ricarichi.
#
# Usa il comando artisan SENZA --fresh: aggiunge un nuovo token senza revocare
# quelli esistenti, quindi non rompe altri client che stessero ancora usando un
# token precedente. Da rilanciare ogni volta che si aggiunge una nuova ability
# in CreateCognitorToken::ABILITIES (Laravel) per una nuova azione chatbot.
set -euo pipefail

COGNITOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LARAVEL_CONTAINER="php-82"
LARAVEL_PATH="/var/www/programmato-backoffice-be"
ENV_FILE="$COGNITOR_DIR/.env"

echo "Genero un nuovo token Sanctum (additivo, non revoca i precedenti)..."
OUTPUT=$(docker exec "$LARAVEL_CONTAINER" sh -c "cd $LARAVEL_PATH && php artisan chatbot:cognitor-token")
TOKEN=$(echo "$OUTPUT" | sed -n '2p' | tr -d '\r')

if [ -z "$TOKEN" ]; then
    echo "Errore: non sono riuscito a estrarre il token dall'output del comando artisan." >&2
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Errore: $ENV_FILE non trovato." >&2
    exit 1
fi

TMP=$(mktemp)
awk -v tok="BACKEND_API_TOKEN=$TOKEN" '{ if ($0 ~ /^BACKEND_API_TOKEN=/) print tok; else print $0 }' "$ENV_FILE" > "$TMP"
mv "$TMP" "$ENV_FILE"
echo "Token aggiornato in $ENV_FILE (valore non stampato)."

PID=$(pgrep -f "uvicorn main:app" | head -1 || true)
if [ -n "$PID" ]; then
    echo "Riavvio processo Cognitor esistente (pid $PID)..."
    kill "$PID"
    sleep 2
fi

cd "$COGNITOR_DIR"
nohup .venv/bin/uvicorn main:app --reload --port 8000 > /tmp/cognitor_server.log 2>&1 &
disown
sleep 3

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || echo "000")
if [ "$STATUS" = "200" ]; then
    echo "Cognitor riavviato correttamente (health check OK)."
else
    echo "Attenzione: health check ha risposto $STATUS, controlla /tmp/cognitor_server.log" >&2
fi
