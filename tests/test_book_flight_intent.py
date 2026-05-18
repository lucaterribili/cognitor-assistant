import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent
from config import BASE_DIR


def test_book_flight_asks_start_location_and_executes_operation():
    agent = Agent()
    agent.load_models()
    agent.load_knowledge()

    # Simula una richiesta in cui viene fornita solo la location
    response_text, wait_slot, bot_slots = agent.get_response(
        "book_flight",
        {"LOCATION": "Milano"},
        []
    )

    assert wait_slot == "START_LOCATION"
    assert "partenza" in response_text.lower() or "da dove" in response_text.lower()

    # Ora fornisci la città di partenza e verifica che l'operation venga eseguita
    response_text, wait_slot, bot_slots = agent.get_response(
        "book_flight",
        {"LOCATION": "Milano", "START_LOCATION": "Roma"},
        []
    )

    assert wait_slot is None
    assert "cerco voli da roma a milano" in response_text.lower()
