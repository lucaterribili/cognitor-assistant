"""
Test del modello DialoguePolicy ML e dello script di training.

Verifica:
- Architettura del modello (forward pass, dimensioni output)
- build_dicts: costruzione corretta dei dizionari intent/action
- generate_training_samples: generazione campioni dalle storie
- DialoguePolicyDataset: accesso e formato degli item
- collate_dialogue_fn: padding del batch
- train_dialogue_policy_model: training su dati minimi senza errori
- DialoguePolicy.predict: inferenza dopo training
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from intellective.dialogue_policy import DialoguePolicy
from intellective.train_dialogue_policy import (
    build_dicts,
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
    first_steps = [s for s in samples if s[1] == greeting_id]
    for context_ids, _, _ in first_steps:
        assert context_ids == [], f"Primo step deve avere contesto vuoto, trovato: {context_ids}"
    print(f"✓ Primi step hanno contesto vuoto ({len(first_steps)} trovati)")


def test_generate_training_samples_target_range():
    """Gli id target devono essere 1-indexed e nel range corretto."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    samples = generate_training_samples(CONVERSATIONS, intent_dict, action_dict)
    num_actions = len(action_dict)
    for _, _, target_id in samples:
        assert 1 <= target_id <= num_actions, (
            f"Target id {target_id} fuori range [1, {num_actions}]"
        )
    print(f"✓ Tutti i target in range [1, {num_actions}]")


# ---------------------------------------------------------------------------
# Tests: DialoguePolicy model
# ---------------------------------------------------------------------------

def test_model_forward_shape():
    """Il forward pass deve restituire logits della forma corretta."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    batch_size = 3
    seq_len = 2
    context = torch.randint(1, len(intent_dict) + 1, (batch_size, seq_len))
    current = torch.randint(1, len(intent_dict) + 1, (batch_size,))

    logits = model(context, current)
    assert logits.shape == (batch_size, len(action_dict)), (
        f"Shape attesa: ({batch_size}, {len(action_dict)}), trovata: {logits.shape}"
    )
    print(f"✓ Logits shape corretta: {logits.shape}")


def test_model_forward_empty_context():
    """Forward con contesto di padding deve funzionare senza errori."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    context = torch.zeros(1, 1, dtype=torch.long)   # padding
    current = torch.tensor([1], dtype=torch.long)

    logits = model(context, current)
    assert logits.shape == (1, len(action_dict))
    print("✓ Forward con contesto padding funziona")


def test_model_predict_returns_valid_action():
    """predict() deve restituire un action_idx valido e confidence in [0, 1]."""
    intent_dict, action_dict = build_dicts(CONVERSATIONS)
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    context = torch.zeros(1, 1, dtype=torch.long)
    current = torch.tensor([1], dtype=torch.long)

    action_idx, confidence = model.predict(context, current)
    assert 0 <= action_idx < len(action_dict), (
        f"action_idx {action_idx} fuori range [0, {len(action_dict) - 1}]"
    )
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} fuori range [0, 1]"
    print(f"✓ predict() → action_idx={action_idx}, confidence={confidence:.3f}")


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
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # Loss prima del training
    model.eval()
    with torch.no_grad():
        for ctx, curr, tgt in dataloader:
            logits_before = model(ctx, curr)
            loss_before = criterion(logits_before, tgt).item()
            break

    # Training breve
    device = torch.device("cpu")
    model.to(device)
    train_dialogue_policy_model(model, dataloader, epochs=30, lr=0.01, device=device, patience=30)

    # Loss dopo il training
    model.eval()
    with torch.no_grad():
        for ctx, curr, tgt in dataloader:
            logits_after = model(ctx, curr)
            loss_after = criterion(logits_after, tgt).item()
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
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    device = torch.device("cpu")
    model.to(device)
    train_dialogue_policy_model(model, dataloader, epochs=100, lr=0.01, device=device, patience=100)

    # Predici: contesto vuoto + intent greeting
    context = torch.zeros(1, 1, dtype=torch.long)
    current = torch.tensor([intent_dict["greeting"]], dtype=torch.long)
    action_idx, confidence = model.predict(context, current)

    predicted_action = action_dict_inv.get(action_idx + 1, "unknown")
    print(f"✓ greeting → {predicted_action} (confidence={confidence:.3f})")
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
        test_generate_training_samples_count,
        test_generate_training_samples_first_step_empty_context,
        test_generate_training_samples_target_range,
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
