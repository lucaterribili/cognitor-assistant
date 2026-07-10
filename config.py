import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOPING_ACTIVE = False

# Intent confidence threshold
MIN_INTENT_CONFIDENCE = 0.20  # Soglia minima per accettare un intent (altrimenti fallback)

# Integrazione con il backoffice Laravel (Programmato).
# Token di servizio scoped (ability "posts:read"), generato con:
#   php artisan chatbot:cognitor-token
LARAVEL_API_BASE_URL = os.getenv("LARAVEL_API_BASE_URL", "http://localhost/api")
LARAVEL_API_TOKEN = os.getenv("LARAVEL_API_TOKEN", "")
LARAVEL_API_TIMEOUT = float(os.getenv("LARAVEL_API_TIMEOUT", "5"))
