import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOPING_ACTIVE = False

# Intent confidence threshold
MIN_INTENT_CONFIDENCE = 0.20  # Soglia minima per accettare un intent (altrimenti fallback)

# Integrazione con il backend REST del dominio corrente (specifico per branch,
# es. il backoffice Laravel "Programmato"). Token di servizio scoped, generato
# lato backend (per Programmato: php artisan chatbot:cognitor-token).
BACKEND_API_BASE_URL = os.getenv("BACKEND_API_BASE_URL", "http://localhost/api")
BACKEND_API_TOKEN = os.getenv("BACKEND_API_TOKEN", "")
BACKEND_API_TIMEOUT = float(os.getenv("BACKEND_API_TIMEOUT", "5"))
