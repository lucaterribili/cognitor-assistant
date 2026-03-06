"""
Test della DialogueStatePolicy (ispirata a TED Policy di Rasa).

Verifica:
- Costruzione delle transizioni dalle storie YAML
- Estrazione della sequenza di intent utente dallo storico
- Scoring del contesto (longest-suffix match)
- Predizione della prossima azione con confidence
- Comportamento con storie vuote o contesto non corrispondente
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dialogue_state_policy import DialogueStatePolicy


# --- Fixtures ---

CONVERSATIONS = {
    "simple_flow": {
        "description": "Flusso semplice",
        "steps": [
            {"user": "greeting", "bot": "greeting_response"},
            {"user": "ask_name", "bot": "ask_name_response"},
            {"user": "farewell", "bot": "farewell_response"},
        ],
    },
    "help_flow": {
        "description": "Flusso per richiesta aiuto",
        "steps": [
            {"user": "greeting", "bot": "greeting_response"},
            {"user": "ask_help", "bot": "help_options_response"},
            {"user": "farewell", "bot": "farewell_response"},
        ],
    },
}


def _make_history(intent_names: list) -> list:
    """Crea uno storico fittizio con i messaggi utente specificati.

    Il campo 'intent' del messaggio assistant replica l'intent utente,
    in linea con il comportamento reale di ConversationHandler.add_message.
    """
    history = []
    for intent in intent_names:
        history.append({"role": "user", "intent": intent, "content": f"msg_{intent}"})
        history.append({"role": "assistant", "intent": intent, "content": f"risposta_{intent}"})
    return history


# --- Test cases ---

def test_build_story_transitions():
    """La policy deve costruire le transizioni correttamente dalle storie."""
    policy = DialogueStatePolicy(CONVERSATIONS)

    assert len(policy._story_transitions) > 0, "Devono esserci transizioni"

    # Cerca la transizione: contesto=[], user=greeting → action=greeting_response
    first_step = [t for t in policy._story_transitions
                  if t['user_intent'] == 'greeting' and t['context'] == []]
    assert first_step, "Deve esserci una transizione per greeting senza contesto"
    assert first_step[0]['next_action'] == 'greeting_response'

    # Cerca la transizione: contesto=[greeting], user=ask_name → action=ask_name_response
    second_step = [t for t in policy._story_transitions
                   if t['user_intent'] == 'ask_name' and 'greeting' in t['context']]
    assert second_step, "Deve esserci una transizione per ask_name dopo greeting"
    assert second_step[0]['next_action'] == 'ask_name_response'

    print(f"✓ Transizioni costruite correttamente: {len(policy._story_transitions)} transizioni")


def test_extract_user_intent_sequence_empty():
    """Con storico vuoto, la sequenza deve essere vuota."""
    policy = DialogueStatePolicy()
    result = policy._extract_user_intent_sequence([])
    assert result == [], f"Attesa lista vuota, trovato: {result}"
    print("✓ Storico vuoto → sequenza vuota")


def test_extract_user_intent_sequence():
    """Deve estrarre solo gli intent dei messaggi utente."""
    policy = DialogueStatePolicy()

    history = [
        {"role": "user", "intent": "greeting", "content": "ciao"},
        {"role": "assistant", "intent": "greeting", "content": "ciao!"},
        {"role": "user", "intent": "ask_name", "content": "come ti chiami"},
        {"role": "assistant", "intent": None, "content": "sono cognitor"},
    ]

    intents = policy._extract_user_intent_sequence(history)
    assert intents == ["greeting", "ask_name"], f"Intent estratti errati: {intents}"
    print(f"✓ Intent utente estratti correttamente: {intents}")


def test_extract_respects_history_window():
    """Deve rispettare HISTORY_WINDOW: restituisce solo gli ultimi N intent."""
    policy = DialogueStatePolicy()
    many_intents = [f"intent_{i}" for i in range(20)]
    history = _make_history(many_intents)

    intents = policy._extract_user_intent_sequence(history)
    assert len(intents) <= DialogueStatePolicy.HISTORY_WINDOW, (
        f"Attesi al massimo {DialogueStatePolicy.HISTORY_WINDOW} intent, trovati {len(intents)}"
    )
    print(f"✓ HISTORY_WINDOW rispettato: {len(intents)} intent estratti")


def test_score_context_no_story_context():
    """Contesto story vuoto → score 0.5 (primo passo)."""
    policy = DialogueStatePolicy()
    score = policy._score_context_match([], [])
    assert score == 0.5, f"Atteso 0.5 per contesto story vuoto, trovato {score}"

    score = policy._score_context_match(["greeting"], [])
    assert score == 0.5, f"Atteso 0.5 per contesto story vuoto, trovato {score}"
    print("✓ Contesto story vuoto → score 0.5")


def test_score_context_no_current_context():
    """Contesto corrente vuoto ma story ha contesto → score 0.0."""
    policy = DialogueStatePolicy()
    score = policy._score_context_match([], ["greeting"])
    assert score == 0.0, f"Atteso 0.0 per contesto corrente vuoto, trovato {score}"
    print("✓ Contesto corrente vuoto → score 0.0")


def test_score_context_full_match():
    """Corrispondenza completa → score 1.0."""
    policy = DialogueStatePolicy()
    score = policy._score_context_match(["greeting", "ask_name"], ["greeting", "ask_name"])
    assert score == 1.0, f"Atteso 1.0 per match completo, trovato {score}"
    print(f"✓ Match completo → score {score}")


def test_score_context_partial_suffix_match():
    """Corrispondenza parziale del suffisso → score proporzionale."""
    policy = DialogueStatePolicy()
    # Contesto corrente più lungo, ma il suffisso corrisponde
    score = policy._score_context_match(
        ["irrelevant", "greeting", "ask_name"],
        ["greeting", "ask_name"]
    )
    assert score == 1.0, f"Atteso 1.0 per suffisso completo, trovato {score}"
    print(f"✓ Suffisso completo → score {score}")


def test_score_context_no_match():
    """Nessuna corrispondenza → score 0.0."""
    policy = DialogueStatePolicy()
    score = policy._score_context_match(["farewell"], ["greeting"])
    assert score == 0.0, f"Atteso 0.0 per nessun match, trovato {score}"
    print("✓ Nessun match → score 0.0")


def test_predict_next_action_first_step():
    """Primo passo: greeting senza contesto → greeting_response."""
    policy = DialogueStatePolicy(CONVERSATIONS)

    result = policy.predict_next_action("greeting", [])
    assert result is not None, "Deve predire un'azione per greeting"
    assert result['action'] == 'greeting_response', (
        f"Azione attesa: greeting_response, trovata: {result['action']}"
    )
    assert result['confidence'] >= DialogueStatePolicy.MIN_CONFIDENCE
    print(f"✓ Primo passo greeting → {result['action']} (confidence={result['confidence']:.2f})")


def test_predict_next_action_with_context():
    """Secondo passo: ask_name dopo greeting → ask_name_response."""
    policy = DialogueStatePolicy(CONVERSATIONS)

    history = _make_history(["greeting"])
    result = policy.predict_next_action("ask_name", history)

    assert result is not None, "Deve predire un'azione per ask_name dopo greeting"
    assert result['action'] == 'ask_name_response', (
        f"Azione attesa: ask_name_response, trovata: {result['action']}"
    )
    print(f"✓ ask_name dopo greeting → {result['action']} (confidence={result['confidence']:.2f})")


def test_predict_returns_none_for_unknown_intent():
    """Intent non presente in nessuna storia → None."""
    policy = DialogueStatePolicy(CONVERSATIONS)

    result = policy.predict_next_action("intent_sconosciuto", [])
    assert result is None, f"Atteso None per intent sconosciuto, trovato: {result}"
    print("✓ Intent sconosciuto → None")


def test_predict_returns_none_empty_conversations():
    """Nessuna storia → None."""
    policy = DialogueStatePolicy({})

    result = policy.predict_next_action("greeting", [])
    assert result is None, f"Atteso None senza storie, trovato: {result}"
    print("✓ Conversations vuote → None")


def test_predict_returns_none_empty_intent():
    """Intent vuoto → None."""
    policy = DialogueStatePolicy(CONVERSATIONS)

    result = policy.predict_next_action("", [])
    assert result is None, f"Atteso None per intent vuoto, trovato: {result}"

    result = policy.predict_next_action(None, [])
    assert result is None, f"Atteso None per intent None, trovato: {result}"
    print("✓ Intent vuoto/None → None")


def test_predict_context_disambiguates_flow():
    """
    Il contesto disambigua tra flussi diversi:
    - greeting → ask_name (simple_flow)
    - greeting → ask_help (help_flow)
    Dopo greeting + ask_name, farewell deve venire da simple_flow.
    """
    policy = DialogueStatePolicy(CONVERSATIONS)

    history = _make_history(["greeting", "ask_name"])
    result = policy.predict_next_action("farewell", history)

    assert result is not None, "Deve predire un'azione per farewell con contesto"
    assert result['action'] == 'farewell_response'
    print(f"✓ Contesto disambigua il flusso → {result['action']} (confidence={result['confidence']:.2f})")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: DialogueStatePolicy (TED-inspired)")
    print("=" * 60)

    tests = [
        test_build_story_transitions,
        test_extract_user_intent_sequence_empty,
        test_extract_user_intent_sequence,
        test_extract_respects_history_window,
        test_score_context_no_story_context,
        test_score_context_no_current_context,
        test_score_context_full_match,
        test_score_context_partial_suffix_match,
        test_score_context_no_match,
        test_predict_next_action_first_step,
        test_predict_next_action_with_context,
        test_predict_returns_none_for_unknown_intent,
        test_predict_returns_none_empty_conversations,
        test_predict_returns_none_empty_intent,
        test_predict_context_disambiguates_flow,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FALLITO {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERRORE {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"TUTTI I TEST PASSATI ({passed}/{passed}) ✓")
    else:
        print(f"TEST PASSATI: {passed}/{passed + failed}")
        print(f"TEST FALLITI: {failed}/{passed + failed} ✗")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
