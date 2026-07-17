import os

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOPING_ACTIVE = False

# Intent confidence threshold
MIN_INTENT_CONFIDENCE = 0.20  # Soglia minima per accettare un intent (altrimenti fallback)

# Soglia di confidenza per abbandonare la raccolta slot (modalità "inputable") quando
# il messaggio dell'utente, pur non contenendo un valore utile per lo slot atteso,
# classifica con questa confidenza come un intent diverso da quello in attesa: si
# assume un cambio di argomento invece di forzare il testo come valore di slot non
# valido. Più alta di MIN_INTENT_CONFIDENCE perché abbandonare uno slot in attesa è
# una decisione più costosa di un semplice fallback.
INPUTABLE_SWITCH_CONFIDENCE = 0.60

# Integrazione con il backend REST del dominio corrente (specifico per branch,
# es. il backoffice Laravel "Programmato"). Token di servizio scoped, generato
# lato backend (per Programmato: php artisan chatbot:cognitor-token).
BACKEND_API_BASE_URL = os.getenv("BACKEND_API_BASE_URL", "http://localhost/api")
BACKEND_API_TOKEN = os.getenv("BACKEND_API_TOKEN", "")
BACKEND_API_TIMEOUT = float(os.getenv("BACKEND_API_TIMEOUT", "5"))
