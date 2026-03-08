"""
Test del modello DialoguePolicy ML e dello script di training.

Verifica:
- Architettura del modello (forward pass, dimensioni output)
- build_dicts: costruzione corretta dei dizionari intent/action
- build_goal_dict: costruzione corretta del dizionario goal
- generate_training_samples: generazione campioni dalle storie (quintuple con goal)
- DialoguePolicyDataset: accesso e formato degli item
- collate_dialogue_fn: padding del batch
- train_dialogue_policy_model: training su dati minimi senza errori
- DialoguePolicy.predict: inferenza dopo training (restituisce action, confidence, goal)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from intellective.dialogue_policy import DialoguePolicy
from intellective.train_dialogue_policy import (
    build_dicts,
    build_goal_dict,
    generate_training_samples,
    DialoguePolicyDataset,
    collate_dialogue_fn,
    train_dialogue_policy_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

CONVERSATIONS_WITH_GOALS = {
    "id_card_flow": {
        "description": "Flusso rinnovo carta d'identità",
        "steps": [
            {"user": "renew_id_card", "bot": "utter_dove_si_fa", "goal": "rinnovo_carta_identita"},
            {"user": "ask_cost", "bot": "utter_costo"},
            {"user": "ask_time", "bot": "utter_tempo"},
        ],
    },
    "passport_flow": {
        "description": "Flusso passaporto",
        "steps": [
            {"user": "renew_passport", "bot": "utter_passport_info", "goal": "rinnovo_passaporto"},
            {"user": "ask_cost", "bot": "utter_costo"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Tests: build_dicts
# ---------------------------------------------------------------------------

def test_build_dicts_intent_keys():
    """Tutti gli intent presenti nelle storie devono essere nel dizionario."""
    intent_dict, _ = build_dicts(CONVERSATIONS)
    expected = {"greeting", "ask_name", "farewell", "ask_help"}
    assert set(intent_dict.keys()) == expected, (
        f"Intent attesi: {expected}, trovati: {set(intent_dict.keys())}"
    )
    print(f"✓ intent_dict corretto: {sorted(intent_dict.keys())}")


def test_build_dicts_action_keys():
    """Tutte le azioni presenti nelle storie devono essere nel dizionario."""
    _, action_dict = build_dicts(CONVERSATIONS)
    expected = {"greeting_response", "ask_name_response", "farewell_response", "help_options_response"}
    assert set(action_dict.keys()) == expected, (
        f"Azioni attese: {expected}, trovate: {set(action_dict.keys())}"
    )
    print(f"✓ action_dict corretto: {sorted(action_dict.keys())}")


def test_build_dicts_ids_start_from_one():
    """Gli id devono partire da 1 (0 è riservato per padding)."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    assert min(intent_dict.values()) == 1, "Gli id intent devono partire da 1"
    assert min(action_dict.values()) == 1, "Gli id action devono partire da 1"
    print("✓ Id partono da 1")


def test_build_dicts_empty_conversations():
    """Conversations vuote → dizionari vuoti."""
    intent_dict, action_dict = build_dicts({})
    assert intent_dict == {}
    assert action_dict == {}
    print("✓ Conversations vuote → dizionari vuoti")


# ---------------------------------------------------------------------------
# Tests: generate_training_samples
# ---------------------------------------------------------------------------

def test_generate_training_samples_count():
    """Devono essere generati campioni per ogni step delle storie."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    samples = generate_training_samples(CONVERSATIONS, intent_dict, action_dict)
    # 3 step per storia × 2 storie = 6, ma step farewell si ripete
    assert len(samples) == 6, f"Attesi 6 campioni, trovati {len(samples)}"
    print(f"✓ Generati {len(samples)} campioni di training")


def test_generate_training_samples_first_step_empty_context():
    """Il primo step di ogni storia ha contesto vuoto."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    samples = generate_training_samples(CONVERSATIONS, intent_dict, action_dict)

    # I campioni con intent 'greeting' (primo passo) devono avere contesto vuoto
    greeting_id = intent_dict["greeting"]
    first_steps = [s for s in samples if s[2] == greeting_id]
    for context_intent_ids, context_action_ids, _, _, _ in first_steps:
        assert context_intent_ids == [], f"Primo step deve avere contesto intent vuoto, trovato: {context_intent_ids}"
        assert context_action_ids == [], f"Primo step deve avere contesto azioni vuoto, trovato: {context_action_ids}"
    print(f"✓ Primi step hanno contesto vuoto ({len(first_steps)} trovati)")


def test_generate_training_samples_target_range():
    """Gli id target devono essere 1-indexed e nel range corretto."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    samples = generate_training_samples(CONVERSATIONS, intent_dict, action_dict)
    num_actions = len(action_dict)
    for _, _, _, target_id, _ in samples:
        assert 1 <= target_id <= num_actions, (
            f"Target id {target_id} fuori range [1, {num_actions}]"
        )
    print(f"✓ Tutti i target in range [1, {num_actions}]")


# ---------------------------------------------------------------------------
# Tests: build_goal_dict
# ---------------------------------------------------------------------------

def test_build_goal_dict_keys():
    """Tutti i goal presenti nelle storie devono essere nel dizionario."""
    goal_dict = build_goal_dict(CONVERSATIONS_WITH_GOALS)
    expected = {"rinnovo_carta_identita", "rinnovo_passaporto"}
    assert set(goal_dict.keys()) == expected, (
        f"Goal attesi: {expected}, trovati: {set(goal_dict.keys())}"
    )
    print(f"✓ goal_dict corretto: {sorted(goal_dict.keys())}")


def test_build_goal_dict_ids_start_from_one():
    """Gli id dei goal devono partire da 1 (0 è riservato per 'nessun goal')."""
    goal_dict = build_goal_dict(CONVERSATIONS_WITH_GOALS)
    assert min(goal_dict.values()) == 1, "Gli id goal devono partire da 1"
    print("✓ Id goal partono da 1")


def test_build_goal_dict_empty_conversations():
    """Conversations senza goal → dizionario vuoto."""
    goal_dict = build_goal_dict(CONVERSATIONS)
    assert goal_dict == {}, f"Atteso dizionario vuoto, trovato: {goal_dict}"
    print("✓ Conversations senza goal → goal_dict vuoto")


def test_build_goal_dict_no_conversations():
    """Conversations vuote → dizionario goal vuoto."""
    goal_dict = build_goal_dict({})
    assert goal_dict == {}
    print("✓ Conversations vuote → goal_dict vuoto")


# ---------------------------------------------------------------------------
# Tests: generate_training_samples con goal
# ---------------------------------------------------------------------------

def test_generate_training_samples_with_goals():
    """I campioni con goal devono avere target_goal_id > 0; quelli senza devono avere 0."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS_WITH_GOALS)
    goal_dict = build_goal_dict(CONVERSATIONS_WITH_GOALS)
    samples = generate_training_samples(CONVERSATIONS_WITH_GOALS, intent_dict, action_dict, goal_dict)

    # Ogni sample è una quintupla
    assert all(len(s) == 5 for s in samples), "Ogni campione deve essere una quintupla"

    # Il primo step di ogni flusso ha goal dichiarato → target_goal_id > 0
    first_steps_with_goal = [s for s in samples if s[4] > 0]
    assert len(first_steps_with_goal) == 2, (
        f"Attesi 2 campioni con goal, trovati {len(first_steps_with_goal)}"
    )

    # Gli altri step non hanno goal → target_goal_id == 0
    steps_without_goal = [s for s in samples if s[4] == 0]
    assert len(steps_without_goal) == 3, (
        f"Attesi 3 campioni senza goal, trovati {len(steps_without_goal)}"
    )
    print(f"✓ generate_training_samples con goal: {len(first_steps_with_goal)} con goal, {len(steps_without_goal)} senza")


def test_generate_training_samples_goal_zero_when_no_goal_dict():
    """Senza goal_dict, tutti i target_goal_id devono essere 0."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS_WITH_GOALS)
    samples = generate_training_samples(CONVERSATIONS_WITH_GOALS, intent_dict, action_dict)
    for _, _, _, _, target_goal_id in samples:
        assert target_goal_id == 0, f"Senza goal_dict, target_goal_id deve essere 0, trovato: {target_goal_id}"
    print("✓ Senza goal_dict tutti i target_goal_id sono 0")


# ---------------------------------------------------------------------------
# Tests: DialoguePolicy model
# ---------------------------------------------------------------------------

def test_model_forward_shape():
    """Il forward pass deve restituire logits della forma corretta per azione e goal."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    n_goals = 3
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        num_goals=n_goals,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    batch_size = 3
    seq_len = 2
    context_intents = torch.randint(1, len(intent_dict) + 1, (batch_size, seq_len))
    context_actions = torch.randint(1, len(action_dict) + 1, (batch_size, seq_len))
    current = torch.randint(1, len(intent_dict) + 1, (batch_size,))

    action_logits, goal_logits = model(context_intents, context_actions, current)
    assert action_logits.shape == (batch_size, len(action_dict)), (
        f"Shape azione attesa: ({batch_size}, {len(action_dict)}), trovata: {action_logits.shape}"
    )
    assert goal_logits.shape == (batch_size, n_goals), (
        f"Shape goal attesa: ({batch_size}, {n_goals}), trovata: {goal_logits.shape}"
    )
    print(f"✓ action_logits shape corretta: {action_logits.shape}")
    print(f"✓ goal_logits shape corretta: {goal_logits.shape}")


def test_model_forward_empty_context():
    """Forward con contesto di padding deve funzionare senza errori."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        num_goals=2,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    context_intents = torch.zeros(1, 1, dtype=torch.long)   # padding
    context_actions = torch.zeros(1, 1, dtype=torch.long)   # padding
    current = torch.tensor([1], dtype=torch.long)

    action_logits, goal_logits = model(context_intents, context_actions, current)
    assert action_logits.shape == (1, len(action_dict))
    assert goal_logits.shape == (1, 2)
    print("✓ Forward con contesto padding funziona")


def test_model_predict_returns_valid_action():
    """predict() deve restituire action_idx, confidence e goal_idx validi."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    n_goals = 3
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        num_goals=n_goals,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    context_intents = torch.zeros(1, 1, dtype=torch.long)
    context_actions = torch.zeros(1, 1, dtype=torch.long)
    current = torch.tensor([1], dtype=torch.long)

    action_idx, confidence, goal_idx = model.predict(context_intents, context_actions, current)
    assert 0 <= action_idx < len(action_dict), (
        f"action_idx {action_idx} fuori range [0, {len(action_dict) - 1}]"
    )
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} fuori range [0, 1]"
    assert 0 <= goal_idx < n_goals, f"goal_idx {goal_idx} fuori range [0, {n_goals - 1}]"
    print(f"✓ predict() → action_idx={action_idx}, confidence={confidence:.3f}, goal_idx={goal_idx}")


# ---------------------------------------------------------------------------
# Tests: training
# ---------------------------------------------------------------------------

def test_training_reduces_loss():
    """Il training deve ridurre la loss rispetto a pesi random."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    samples = generate_training_samples(CONVERSATIONS, intent_dict, action_dict)

    dataset = DialoguePolicyDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=len(samples),
        shuffle=False,
        collate_fn=collate_dialogue_fn,
    )

    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        num_goals=1,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # Loss prima del training
    model.eval()
    with torch.no_grad():
        for ctx_intents, ctx_actions, curr, tgt, _ in dataloader:
            action_logits, _ = model(ctx_intents, ctx_actions, curr)
            loss_before = criterion(action_logits, tgt).item()
            break

    # Training breve
    device = torch.device("cpu")
    model.to(device)
    train_dialogue_policy_model(model, dataloader, epochs=30, lr=0.01, device=device, patience=30)

    # Loss dopo il training
    model.eval()
    with torch.no_grad():
        for ctx_intents, ctx_actions, curr, tgt, _ in dataloader:
            action_logits, _ = model(ctx_intents, ctx_actions, curr)
            loss_after = criterion(action_logits, tgt).item()
            break

    assert loss_after < loss_before, (
        f"La loss deve diminuire: before={loss_before:.4f}, after={loss_after:.4f}"
    )
    print(f"✓ Loss ridotta: {loss_before:.4f} → {loss_after:.4f}")


def test_trained_model_predicts_greeting():
    """Dopo il training, il modello deve predire greeting_response per greeting."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    action_dict_inv = {v: k for k, v in action_dict.items()}
    samples = generate_training_samples(CONVERSATIONS, intent_dict, action_dict)

    dataset = DialoguePolicyDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=len(samples),
        shuffle=True,
        collate_fn=collate_dialogue_fn,
    )

    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        num_goals=1,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    device = torch.device("cpu")
    model.to(device)
    train_dialogue_policy_model(model, dataloader, epochs=100, lr=0.01, device=device, patience=100)

    # Predici: contesto vuoto + intent greeting
    context_intents = torch.zeros(1, 1, dtype=torch.long)
    context_actions = torch.zeros(1, 1, dtype=torch.long)
    current = torch.tensor([intent_dict["greeting"]], dtype=torch.long)
    action_idx, confidence, goal_idx = model.predict(context_intents, context_actions, current)

    predicted_action = action_dict_inv.get(action_idx + 1, "unknown")
    print(f"✓ greeting → {predicted_action} (confidence={confidence:.3f}, goal_idx={goal_idx})")
    # Il modello deve aver imparato che greeting → greeting_response
    assert predicted_action == "greeting_response", (
        f"Atteso 'greeting_response', trovato '{predicted_action}'"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: DialoguePolicy ML")
    print("=" * 60)

    tests = [
        test_build_dicts_intent_keys,
        test_build_dicts_action_keys,
        test_build_dicts_ids_start_from_one,
        test_build_dicts_empty_conversations,
        test_build_goal_dict_keys,
        test_build_goal_dict_ids_start_from_one,
        test_build_goal_dict_empty_conversations,
        test_build_goal_dict_no_conversations,
        test_generate_training_samples_count,
        test_generate_training_samples_first_step_empty_context,
        test_generate_training_samples_target_range,
        test_generate_training_samples_with_goals,
        test_generate_training_samples_goal_zero_when_no_goal_dict,
        test_model_forward_shape,
        test_model_forward_empty_context,
        test_model_predict_returns_valid_action,
        test_training_reduces_loss,
        test_trained_model_predicts_greeting,
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
