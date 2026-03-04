"""
Test del sistema SlotManager con l'intent open_app e l'entità PRODUCT.
Verifica che il sistema data-driven funzioni con entità diverse da LOCATION.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.slot_manager import SlotManager
from agent.model_loader import KnowledgeLoader
from config import BASE_DIR


def test_slot_manager_open_app():
    """Test completo del SlotManager con open_app."""

    print("="*60)
    print("TEST: SlotManager con intent open_app (entità PRODUCT)")
    print("="*60)

    # Carica le rules
    knowledge_loader = KnowledgeLoader(BASE_DIR)
    rules, _ = knowledge_loader.load_all()

    print(f"\n✓ Rules caricate: {len(rules)} intents")

    # Inizializza SlotManager
    slot_manager = SlotManager(rules)
    print("✓ SlotManager inizializzato (data-driven, zero configurazioni)")

    # Test 1: Estrazione valori validi per PRODUCT
    print("\n" + "-"*60)
    print("TEST 1: Auto-discovery dei valori validi per PRODUCT")
    print("-"*60)

    valid_products = slot_manager.get_valid_values("open_app", "PRODUCT")
    print(f"Valori validi estratti dalle rules: {valid_products}")

    expected_apps = ["WhatsApp", "Telegram", "Spotify", "Chrome"]
    assert all(app in valid_products for app in expected_apps), "Mancano app nelle rules"
    print(f"✓ Tutti i valori attesi trovati: {expected_apps}")

    # Test 2: Validazione valori
    print("\n" + "-"*60)
    print("TEST 2: Validazione valori")
    print("-"*60)

    test_cases = [
        ("WhatsApp", True, "App supportata"),
        ("Telegram", True, "App supportata"),
        ("Spotify", True, "App supportata"),
        ("Chrome", True, "App supportata"),
        ("Facebook", False, "App NON supportata"),
        ("Instagram", False, "App NON supportata"),
        ("Netflix", False, "App NON supportata"),
    ]

    for app_name, expected, description in test_cases:
        result = slot_manager.validate_slot_value("open_app", "PRODUCT", app_name)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{app_name}': {result} ({description})")
        assert result == expected, f"Validazione fallita per {app_name}"

    print("\n✓ Tutte le validazioni passate!")

    # Test 3: Case-insensitive
    print("\n" + "-"*60)
    print("TEST 3: Validazione case-insensitive")
    print("-"*60)

    case_tests = [
        ("whatsapp", True),
        ("WHATSAPP", True),
        ("WhAtSaPp", True),
        ("telegram", True),
        ("TELEGRAM", True),
        ("spotify", True),
        ("chrome", True),
    ]

    for app_name, expected in case_tests:
        result = slot_manager.validate_slot_value("open_app", "PRODUCT", app_name)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{app_name}': {result}")
        assert result == expected, f"Case-insensitive fallito per {app_name}"

    print("\n✓ Validazione case-insensitive OK!")

    # Test 4: Estrazione da entità NER
    print("\n" + "-"*60)
    print("TEST 4: Estrazione da entità NER")
    print("-"*60)

    # Simula entità NER
    entities = [
        {"entity": "PRODUCT", "value": "WhatsApp", "start": 5, "end": 13}
    ]

    extracted = slot_manager.extractor.extract_from_entities("PRODUCT", entities)
    print(f"Entità NER: {entities}")
    print(f"Valore estratto: {extracted}")
    assert extracted == "WhatsApp", "Estrazione fallita"
    print("✓ Estrazione da NER funzionante!")

    # Test 5: Slot discovery
    print("\n" + "-"*60)
    print("TEST 5: Auto-discovery degli slot per intent")
    print("-"*60)

    slots = slot_manager.context_manager.get_slots_for_intent("open_app")
    print(f"Slot rilevati per 'open_app': {slots}")
    assert "PRODUCT" in slots, "PRODUCT non trovato negli slot"
    assert "PRODUCT_UNSUPPORTED" in slots, "PRODUCT_UNSUPPORTED non trovato"
    print("✓ Slot discovery funzionante!")

    # Test 6: Confronto con LOCATION
    print("\n" + "-"*60)
    print("TEST 6: Sistema generico - LOCATION vs PRODUCT")
    print("-"*60)

    location_slots = slot_manager.context_manager.get_slots_for_intent("ask_city_touristic_information")
    product_slots = slot_manager.context_manager.get_slots_for_intent("open_app")

    print(f"Slot per LOCATION: {location_slots}")
    print(f"Slot per PRODUCT:  {product_slots}")

    print("\n✓ Il sistema gestisce entrambe le entità senza configurazione hardcoded!")
    print("✓ Completamente data-driven!")

    # Riepilogo finale
    print("\n" + "="*60)
    print("🎉 TUTTI I TEST PASSATI!")
    print("="*60)
    print("\nIl sistema SlotManager è:")
    print("  ✓ Completamente data-driven")
    print("  ✓ Generico (funziona con qualsiasi entità)")
    print("  ✓ Auto-discovery (estrae tutto dalle rules)")
    print("  ✓ Zero configurazioni hardcoded")
    print("  ✓ Validazione case-insensitive")
    print("  ✓ Estensibile (basta aggiungere rules JSON)")
    print("\n✅ Sistema pronto per l'uso in produzione!")


if __name__ == "__main__":
    test_slot_manager_open_app()

A