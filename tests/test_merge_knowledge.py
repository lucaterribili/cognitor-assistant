#!/usr/bin/env python3
"""
Test script per verificare il merge di rules e responses.
Verifica che i file da knowledge/ e training_data/ vengano correttamente mergiati in .cognitor/
"""

import os
import sys
from pathlib import Path

# Aggiungi la root del progetto al path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.merge_data import merge_rules, merge_responses
from agent.model_loader import KnowledgeLoader


def test_merge():
    """Test del merge di rules e responses"""
    print("=" * 60)
    print("TEST: Merge Rules e Responses")
    print("=" * 60)

    # Test merge rules
    print("\n[1/4] Test merge_rules()...")
    rules_summary = merge_rules(
        input_dirs=[
            str(BASE_DIR / "knowledge" / "rules"),
            str(BASE_DIR / "training_data" / "rules")
        ],
        output_file=str(BASE_DIR / ".cognitor" / "rules.yaml")
    )

    assert rules_summary['files_failed'] == 0, "Errore nel merge delle rules"
    assert rules_summary['rules_total'] > 0, "Nessuna rule trovata"
    print(f"  ✓ Rules mergiati: {rules_summary['rules_total']} da {rules_summary['files_ok']} file(s)")

    # Test merge responses
    print("\n[2/4] Test merge_responses()...")
    responses_summary = merge_responses(
        input_dirs=[
            str(BASE_DIR / "knowledge" / "responses"),
            str(BASE_DIR / "training_data" / "responses")
        ],
        output_file=str(BASE_DIR / ".cognitor" / "responses.yaml")
    )

    assert responses_summary['files_failed'] == 0, "Errore nel merge delle responses"
    assert responses_summary['responses_total'] > 0, "Nessuna response trovata"
    print(f"  ✓ Responses mergiati: {responses_summary['responses_total']} da {responses_summary['files_ok']} file(s)")

    # Test KnowledgeLoader
    print("\n[3/4] Test KnowledgeLoader...")
    loader = KnowledgeLoader(str(BASE_DIR))

    rules = loader.load_rules()
    assert len(rules) > 0, "KnowledgeLoader non ha caricato nessuna rule"
    print(f"  ✓ KnowledgeLoader ha caricato {len(rules)} rules")

    responses = loader.load_responses()
    assert len(responses) > 0, "KnowledgeLoader non ha caricato nessuna response"
    print(f"  ✓ KnowledgeLoader ha caricato {len(responses)} responses")

    # Verifica dati da training_data
    print("\n[4/4] Verifica inclusione dati da training_data...")

    # Cerca una rule che dovrebbe essere in training_data
    test_rule = "ask_id_card_renew"
    if test_rule in rules:
        print(f"  ✓ Rule '{test_rule}' trovata (da training_data/rules/test.yaml)")
    else:
        print(f"  ⚠ Rule '{test_rule}' non trovata (verifica training_data/rules/)")

    # Cerca una response che dovrebbe essere in training_data
    test_response = "ask_id_card_renew_response"
    if test_response in responses:
        response_variants = len(responses[test_response])
        print(f"  ✓ Response '{test_response}' trovata con {response_variants} varianti (da training_data/responses/test.yaml)")
    else:
        print(f"  ⚠ Response '{test_response}' non trovata (verifica training_data/responses/)")

    print("\n" + "=" * 60)
    print("TUTTI I TEST SONO PASSATI ✓")
    print("=" * 60)
    print("\nRiepilogo:")
    print(f"  - Rules totali: {len(rules)}")
    print(f"  - Responses totali: {len(responses)}")
    print(f"  - Files rules processati: {rules_summary['files_ok']}")
    print(f"  - Files responses processati: {responses_summary['files_ok']}")
    print(f"\nOutput:")
    print(f"  - {BASE_DIR / '.cognitor' / 'rules.yaml'}")
    print(f"  - {BASE_DIR / '.cognitor' / 'responses.yaml'}")


if __name__ == "__main__":
    try:
        test_merge()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FALLITO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

