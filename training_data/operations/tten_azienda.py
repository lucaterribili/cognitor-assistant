"""Operation per l'intent ask_tten_azienda.

I fatti aziendali (mission, competenze, tecnologie, partner) vivono in
training_data/config/tten_azienda.yaml invece che in una response statica
del training dataset: aggiornarli non richiede toccare gli intent/le
conversations né rifare il training della Dialogue Policy.
"""
import os
import random

import yaml

import config

_CONFIG_PATH = os.path.join(config.BASE_DIR, "training_data", "config", "tten_azienda.yaml")

_INTRO_VARIANTS = [
    "Siamo TTen, una società di consulenza informatica.",
    "TTen è una società di consulenza informatica.",
]


def _load_company_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def action_tten_azienda(intent_name: str = None, slots: dict = None) -> dict:
    """
    Compone la risposta su "chi è TTen" leggendo training_data/config/tten_azienda.yaml.

    Args:
        intent_name: Nome dell'intent che ha attivato l'operation
        slots: Slot disponibili (non usati qui)

    Returns:
        dict con la risposta e metadati
    """
    try:
        company = _load_company_config()
    except Exception as e:
        return {
            "response": "Al momento non riesco a recuperare le informazioni sull'azienda. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "tten_azienda", "error": str(e)},
        }

    intro = random.choice(_INTRO_VARIANTS)
    mission = company.get("mission")
    descrizione = company.get("descrizione", "")
    tecnologie = company.get("tecnologie", {})

    parts = [intro]
    if mission:
        parts.append(f'Il nostro obiettivo è {mission[0].lower()}{mission[1:]}.')
    if descrizione:
        parts.append(descrizione)

    linguaggi = tecnologie.get("linguaggi")
    database = tecnologie.get("database")
    if linguaggi and database:
        parts.append(
            "Lavoriamo con tecnologie diverse tra loro: linguaggi come "
            f"{', '.join(linguaggi[:5])}, database come {', '.join(database[:4])}, "
            "e diverse piattaforme e sistemi operativi."
        )

    response = " ".join(parts)

    return {
        "response": response,
        "slots": {},
        "metadata": {"operation": "tten_azienda", "source": "training_data/config/tten_azienda.yaml"},
    }
