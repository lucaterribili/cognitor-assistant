import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOPING_ACTIVE = False

# Intent confidence threshold
MIN_INTENT_CONFIDENCE = 0.20  # Soglia minima per accettare un intent (altrimenti fallback)

