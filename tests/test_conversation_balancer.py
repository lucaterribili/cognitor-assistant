"""
Test del ConversationBalancer.

Verifica:
- Rilevamento logits incerti
- Matching dello storico con i pattern YAML
- Boosting del logit dell'intent atteso
- Nessuna modifica quando i logits sono certi
- Nessuna modifica quando lo storico non corrisponde a nessun pattern
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.conversation_balancer import ConversationBalancer


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

INTENT_DICT = {
    "0": "greeting",
    "1": "ask_name",
    "2": "farewell",
    "3": "ask_help",
    "4": "unknown_intent",
}


def _make_history(intent_names: list) -> list:
    """Crea uno storico fittizio con i dati utente specificati."""
    return [{"role": "user", "intent": intent, "content": f"msg_{intent}"} for intent in intent_names]


def _softmax(logits: list) -> list:
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    s = sum(exp_l)
    return [e / s for e in exp_l]


# --- Test cases ---

def test_no_change_when_logits_certain():
    """Logits certi: il balancer non deve modificarli."""
    balancer = ConversationBalancer(CONVERSATIONS, INTENT_DICT)

    # Logit dominante (prob ~99%) -> certi
    logits = [10.0, 1.0, 0.5, 0.3, 0.2]
    history = _make_history(["greeting"])

    result = balancer.balance(logits, history)

    assert result is logits, "I logits certi non devono essere modificati"
    print("✓ Logits certi: nessuna modifica")


def test_no_change_when_empty_history():
    """Storico vuoto: il balancer non deve modificare i logits."""
    balancer = ConversationBalancer(CONVERSATIONS, INTENT_DICT)

    logits = [2.0, 1.8, 0.5, 0.3, 0.2]
    result = balancer.balance(logits, [])

    assert result is logits, "Con storico vuoto i logits non devono essere modificati"
    print("✓ Storico vuoto: nessuna modifica")


def test_no_change_when_history_is_none():
    """history=None: il balancer deve restituire i logits invariati."""
    balancer = ConversationBalancer(CONVERSATIONS, INTENT_DICT)

    logits = [2.0, 1.8, 0.5, 0.3, 0.2]
    result = balancer.balance(logits, None)

    assert result is logits, "Con history=None i logits non devono essere modificati"
    print("✓ history=None: nessuna modifica")


def test_boost_applied_on_uncertain_logits_with_matching_history():
    """
    Logits incerti + storico che matcha un pattern:
    il logit dell'intent atteso deve essere boostato.
    """
    balancer = ConversationBalancer(CONVERSATIONS, INTENT_DICT)

    # Logits incerti: indici 0 (greeting=0) e 4 (unknown=4) simili
    # Il pattern simple_flow prevede ask_name (idx=1) dopo greeting (idx=0)
    logits = [2.0, 1.9, 0.5, 0.3, 0.2]  # incerti: prob[1] >= 0.5 * prob[0]
    history = _make_history(["greeting"])

    original_ask_name_logit = logits[1]  # idx=1 = ask_name
    result = balancer.balance(logits, history)

    assert result is not logits, "I logits devono essere stati modificati"
    assert result[1] > original_ask_name_logit, (
        f"Il logit di ask_name (idx=1) deve essere aumentato: "
        f"originale={original_ask_name_logit}, nuovo={result[1]}"
    )
    assert result[1] == original_ask_name_logit + ConversationBalancer.BOOST_AMOUNT
    print(f"✓ Boost applicato: ask_name logit {original_ask_name_logit} -> {result[1]}")


def test_no_boost_when_history_does_not_match():
    """
    Logits incerti ma storico che non corrisponde a nessun pattern:
    i logits non devono essere modificati.
    """
    balancer = ConversationBalancer(CONVERSATIONS, INTENT_DICT)

    logits = [2.0, 1.9, 0.5, 0.3, 0.2]
    # "unknown_intent" non è in nessun pattern
    history = _make_history(["unknown_intent"])

    result = balancer.balance(logits, history)

    assert result is logits, "Senza match nel pattern i logits non devono essere modificati"
    print("✓ Storico senza match: nessuna modifica")


def test_no_boost_when_next_intent_not_in_dict():
    """
    Pattern con un intent non presente nell'intent_dict:
    nessun boost deve essere applicato.
    """
    conversations_with_unknown = {
        "test_flow": {
            "steps": [
                {"user": "greeting", "bot": "greeting_response"},
                {"user": "intent_not_in_dict", "bot": "some_response"},
            ]
        }
    }
    balancer = ConversationBalancer(conversations_with_unknown, INTENT_DICT)

    logits = [2.0, 1.9, 0.5, 0.3, 0.2]
    history = _make_history(["greeting"])

    result = balancer.balance(logits, history)

    assert result is logits, "Se il prossimo intent non è nel dict, non deve essere modificato nulla"
    print("✓ Prossimo intent non in dict: nessuna modifica")


def test_is_uncertain_threshold():
    """Test diretto del metodo _is_uncertain."""
    balancer = ConversationBalancer()

    # Caso certo: prob[1] < 0.5 * prob[0]
    # logits molto sbilanciati
    certain_logits = [10.0, 1.0, 0.5]
    probs = _softmax(certain_logits)
    assert not balancer._is_uncertain(probs), "Logits certi non devono essere rilevati come incerti"

    # Caso incerto: prob[1] >= 0.5 * prob[0]
    uncertain_logits = [2.0, 1.9, 0.5]
    probs = _softmax(uncertain_logits)
    assert balancer._is_uncertain(probs), "Logits incerti devono essere rilevati come tali"

    print("✓ _is_uncertain funziona correttamente")


def test_get_recent_user_intents():
    """Test dell'estrazione degli intent utente recenti."""
    balancer = ConversationBalancer()

    history = [
        {"role": "user", "intent": "greeting", "content": "ciao"},
        {"role": "assistant", "intent": "greeting_response", "content": "ciao!"},
        {"role": "user", "intent": "ask_name", "content": "come ti chiami"},
        {"role": "assistant", "intent": None, "content": "sono cognitor"},
    ]

    intents = balancer._get_recent_user_intents(history)
    assert intents == ["greeting", "ask_name"], f"Intent estratti errati: {intents}"
    print(f"✓ Intent utente estratti correttamente: {intents}")


def test_find_next_expected_intent_matches():
    """Test del matching dei pattern con lo storico."""
    balancer = ConversationBalancer(CONVERSATIONS, INTENT_DICT)

    # Storico: [greeting] -> atteso ask_name (da simple_flow o help_flow)
    result = balancer._find_next_expected_intent(["greeting"])
    assert result in ["ask_name", "ask_help"], f"Intent atteso errato: {result}"
    print(f"✓ Match trovato dopo greeting: {result}")

    # Storico: [greeting, ask_name] -> atteso farewell (da simple_flow)
    result = balancer._find_next_expected_intent(["greeting", "ask_name"])
    assert result == "farewell", f"Intent atteso errato: {result}"
    print(f"✓ Match trovato dopo greeting, ask_name: {result}")

    # Storico senza match
    result = balancer._find_next_expected_intent(["sconosciuto"])
    assert result is None, f"Nessun match atteso ma trovato: {result}"
    print("✓ Nessun match per intent sconosciuto")


def test_build_pattern_sequences():
    """Test della costruzione delle sequenze di pattern."""
    balancer = ConversationBalancer(CONVERSATIONS)

    sequences = balancer._pattern_sequences
    assert len(sequences) == 2, f"Attese 2 sequenze, trovate {len(sequences)}"
    assert ["greeting", "ask_name", "farewell"] in sequences
    assert ["greeting", "ask_help", "farewell"] in sequences
    print(f"✓ Sequenze costruite correttamente: {sequences}")


def test_empty_conversations():
    """ConversationBalancer con conversations vuote deve funzionare senza errori."""
    balancer = ConversationBalancer({}, INTENT_DICT)

    logits = [2.0, 1.9, 0.5, 0.3, 0.2]
    history = _make_history(["greeting"])

    result = balancer.balance(logits, history)

    assert result is logits, "Con conversations vuote i logits non devono essere modificati"
    print("✓ Conversations vuote: nessuna modifica e nessun errore")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: ConversationBalancer")
    print("=" * 60)

    tests = [
        test_no_change_when_logits_certain,
        test_no_change_when_empty_history,
        test_no_change_when_history_is_none,
        test_boost_applied_on_uncertain_logits_with_matching_history,
        test_no_boost_when_history_does_not_match,
        test_no_boost_when_next_intent_not_in_dict,
        test_is_uncertain_threshold,
        test_get_recent_user_intents,
        test_find_next_expected_intent_matches,
        test_build_pattern_sequences,
        test_empty_conversations,
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
