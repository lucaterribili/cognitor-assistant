#!/usr/bin/env bash
# Deploy di Cognitor su un server di staging/produzione. Il training si fa in locale: sul server arrivano
# il codice del branch corrente (solo i file versionati) e gli artefatti già addestrati.
#
# Uso (dalla cartella del branch da pubblicare):
#   scripts/deploy_staging.sh [--train] [--no-restart] [--smoke "domanda"] [--config deploy/staging.env]
#
#   --train        lancia prima `python -m pipeline` (training completo) con l'interprete di PYTHON_BIN
#   --no-restart   copia i file ma non riavvia il servizio
#   --smoke "..."  dopo il riavvio fa una domanda reale al bot (login con le credenziali del .env remoto)
#
# Cosa viaggia:  codice (git ls-files) · models/intent_model_fast.pth · models/dialogue_policy.pth · .cognitor/
# Cosa NON viaggia mai: .env, .venv, sessions/, fasttext_model.bin, translator/ (i modelli pesanti restano fuori).
# Cosa resta sul server: .env (segreti generati lì), .venv, il service systemd.
# Config: deploy/staging.env (SSH_HOST, SSH_PORT, SSH_USER, SSH_KEY, REMOTE_PATH, SERVICE, HEALTH_URL, EXPECTED_BRANCH).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

CONFIG="deploy/staging.env"; TRAIN=0; RESTART=1; SMOKE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --train) TRAIN=1 ;;
    --no-restart) RESTART=0 ;;
    --smoke) shift; SMOKE="${1:?--smoke vuole una domanda}" ;;
    --config) shift; CONFIG="${1:?--config vuole un file}" ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "Opzione sconosciuta: $1" >&2; exit 1 ;;
  esac
  shift
done

[ -f "$CONFIG" ] || { echo "Manca $CONFIG (vedi l'intestazione dello script)" >&2; exit 1; }
set -a; . "$CONFIG"; set +a
for v in SSH_HOST SSH_PORT SSH_USER SSH_KEY REMOTE_PATH SERVICE HEALTH_URL; do
  [ -n "${!v:-}" ] || { echo "Variabile mancante in $CONFIG: $v" >&2; exit 1; }
done
SSH_KEY="${SSH_KEY/#\$HOME/$HOME}"; SSH_KEY="${SSH_KEY/#\~/$HOME}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REMOTE="$SSH_USER@$SSH_HOST"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -p "$SSH_PORT")
rssh()   { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
rrsync() { rsync -az -e "ssh ${SSH_OPTS[*]}" "$@"; }

# 1) Il branch giusto e i file versionati puliti: si pubblica quello che è in git, non un albero a metà.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ -n "${EXPECTED_BRANCH:-}" ] && [ "$BRANCH" != "$EXPECTED_BRANCH" ]; then
  echo "Sei sul branch '$BRANCH', questo deploy è per '$EXPECTED_BRANCH'." >&2; exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Ci sono modifiche non committate: committa prima (il deploy invia i file versionati)." >&2
  git status --short --untracked-files=no >&2; exit 1
fi
COMMIT="$(git rev-parse --short HEAD)"
echo "==> Deploy di $BRANCH @ $COMMIT su $REMOTE:$REMOTE_PATH"

# 2) Training locale (opzionale) e controllo che gli artefatti ci siano e non siano più vecchi dei dati
if [ "$TRAIN" = 1 ]; then
  echo "==> Training locale: $PYTHON_BIN -m pipeline"
  "$PYTHON_BIN" -m pipeline
fi
ARTIFACTS=(models/intent_model_fast.pth models/dialogue_policy.pth .cognitor/vocab.json .cognitor/intent_dict.json \
           .cognitor/rules.yaml .cognitor/responses.yaml .cognitor/sentencepiece.model)
for f in "${ARTIFACTS[@]}"; do
  [ -e "$f" ] || { echo "Manca $f: esegui il training (--train o 'python -m pipeline')." >&2; exit 1; }
done
NEWEST_DATA="$(find training_data knowledge -type f \( -name '*.yaml' -o -name '*.yml' \) -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
MODEL_TIME="$(stat -c %Y models/intent_model_fast.pth)"
if [ -n "$NEWEST_DATA" ] && [ "${NEWEST_DATA%.*}" -gt "$MODEL_TIME" ]; then
  echo "ATTENZIONE: i dati in training_data/ o knowledge/ sono più recenti del modello (non riaddestrato?)." >&2
  [ "${FORCE:-0}" = 1 ] || { echo "Riaddestra con --train, oppure FORCE=1 per procedere comunque." >&2; exit 1; }
fi

# 3) Codice: solo i file versionati (nessuna cancellazione sul server: .env, .venv e modelli restano)
rssh "mkdir -p '$REMOTE_PATH/models' '$REMOTE_PATH/knowledge/intents'"
echo "==> Codice"
git ls-files -z | rrsync --from0 --files-from=- ./ "$REMOTE:$REMOTE_PATH/"

# 4) Modelli e artefatti del training (.cognitor/ viene sostituita per intero: niente file vecchi)
echo "==> Modelli e .cognitor/"
rrsync models/intent_model_fast.pth models/dialogue_policy.pth "$REMOTE:$REMOTE_PATH/models/"
rrsync --delete .cognitor/ "$REMOTE:$REMOTE_PATH/.cognitor/"

# 5) Dipendenze: solo se requirements.txt è cambiato rispetto all'ultimo deploy
LOCAL_REQ="$(sha256sum requirements.txt | cut -d' ' -f1)"
rssh "set -e; cd '$REMOTE_PATH'
  [ -d .venv ] || python3 -m venv .venv
  if [ ! -f .requirements.sha ] || [ \"\$(cat .requirements.sha)\" != '$LOCAL_REQ' ]; then
    echo '==> Dipendenze (requirements.txt cambiato)'; .venv/bin/pip install -q --upgrade pip; .venv/bin/pip install -q -r requirements.txt
    echo '$LOCAL_REQ' > .requirements.sha
  else echo '==> Dipendenze invariate'; fi
  echo '$BRANCH @ $COMMIT' > .deployed"

# 6) Riavvio e controllo
if [ "$RESTART" = 1 ]; then
  echo "==> Riavvio $SERVICE"
  rssh "sudo systemctl restart '$SERVICE'"
  for i in $(seq 1 30); do
    if rssh "curl -fs '$HEALTH_URL' >/dev/null"; then echo "    $HEALTH_URL risponde (dopo $((i*2)) s)"; break; fi
    [ "$i" = 30 ] && { echo "Il servizio non risponde: rssh 'journalctl -u $SERVICE -n 40'" >&2; exit 1; }
    sleep 2
  done
fi

# 7) Domanda di prova (opzionale), con le credenziali del .env remoto: non passano mai dalla macchina locale
if [ -n "$SMOKE" ]; then
  echo "==> Prova: $SMOKE"
  rssh "cd '$REMOTE_PATH'; set -a; . ./.env; set +a; export BASE='${HEALTH_URL%/health}' Q=$(printf %q "$SMOKE"); python3 -" <<'PY'
import json, os, urllib.parse, urllib.request

base = os.environ["BASE"]
form = urllib.parse.urlencode({"username": os.environ["AUTH_ADMIN_USERNAME"], "password": os.environ["AUTH_ADMIN_PASSWORD"]})
token = json.load(urllib.request.urlopen(urllib.request.Request(f"{base}/auth/token", form.encode())))["access_token"]
request = urllib.request.Request(
    f"{base}/chatbot/message",
    json.dumps({"sender": "deploy-smoke", "message": os.environ["Q"]}).encode(),
    {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
)
answer = json.load(urllib.request.urlopen(request))
print("   intento: ", answer["intent"], round(answer["confidence"], 2))
print("   risposta:", answer["response"])
PY
fi
echo "==> Fatto: $BRANCH @ $COMMIT"
