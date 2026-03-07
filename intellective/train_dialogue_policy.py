"""
Training della Dialogue Policy ML.

Carica le storie di conversazione da conversations.yaml, genera i campioni
di training e addestra il modello DialoguePolicy.

Il modello impara a predire la prossima azione del bot (response key) dato:
  - Il contesto storico: sequenza degli intent utente precedenti
  - L'intent utente corrente

Inserito nella pipeline come step 5, analogamente a come Rasa addestra
la TEDPolicy sui file stories.yml.
"""

import json
import os

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import BASE_DIR
from intellective.dialogue_policy import (
    DialoguePolicy,
    DIALOGUE_POLICY_EMBED_DIM,
    DIALOGUE_POLICY_HIDDEN_DIM,
    DIALOGUE_POLICY_DROPOUT,
)


# Indice riservato per il padding
_PAD_IDX = 0

# Finestra storica massima (numero di intent precedenti da considerare)
_HISTORY_WINDOW = 5


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_conversations(conversations_file: str) -> dict:
    """Carica le conversations dal file YAML mergiato."""
    if not os.path.exists(conversations_file):
        return {}
    with open(conversations_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('conversations', {}) if data else {}


def build_dicts(conversations: dict) -> tuple[dict, dict]:
    """
    Costruisce i dizionari nome→id per intent e azioni dalle storie.

    L'indice 0 è riservato per il padding; gli id partono da 1.

    Returns:
        intent_dict: {intent_name: id}
        action_dict: {action_name: id}
    """
    intents: set[str] = set()
    actions: set[str] = set()

    for flow_data in conversations.values():
        for step in flow_data.get('steps', []):
            if step.get('user'):
                intents.add(step['user'])
            if step.get('bot'):
                actions.add(step['bot'])

    # Ordine deterministico
    intent_dict = {name: idx + 1 for idx, name in enumerate(sorted(intents))}
    action_dict = {name: idx + 1 for idx, name in enumerate(sorted(actions))}

    return intent_dict, action_dict


def build_goal_dict(conversations: dict) -> dict:
    """
    Costruisce il dizionario goal→id dalle storie.

    L'indice 0 è riservato per "nessun goal nuovo"; gli id partono da 1.
    I goal sono raccolti dal campo opzionale `goal` dei passi delle storie.

    Returns:
        goal_dict: {goal_name: id}  (id 1-indexed, 0 riservato)
    """
    all_goals: set[str] = set()

    for flow_data in conversations.values():
        for step in flow_data.get('steps', []):
            g = step.get('goal')
            if g:
                all_goals.add(g)

    # 0 = "nessun goal nuovo" (riservato, non mappare)
    return {g: i + 1 for i, g in enumerate(sorted(all_goals))}


def generate_training_samples(
    conversations: dict,
    intent_dict: dict,
    action_dict: dict,
    goal_dict: dict | None = None,
    history_window: int = _HISTORY_WINDOW,
) -> list[tuple[list[int], list[int], int, int, int]]:
    """
    Genera campioni di training dalle storie.

    Per ogni passo i di ogni storia, produce la quintupla:
        (context_intent_ids, context_action_ids, current_intent_id, target_action_id, target_goal_id)

    dove:
        context_intent_ids = id degli ultimi `history_window` intent utente
                             prima del passo i (può essere vuota)
        context_action_ids = id delle ultime `history_window` azioni bot
                             prima del passo i (può essere vuota)
        current_intent     = id dell'intent utente al passo i
        target_action      = id dell'azione bot al passo i (1-indexed)
        target_goal        = id del goal al passo i (0 se assente)

    Returns:
        Lista di quintuple (context_intent_ids, context_action_ids, current_id, target_id, target_goal_id).
    """
    if goal_dict is None:
        goal_dict = {}

    samples: list[tuple[list[int], list[int], int, int, int]] = []

    for flow_data in conversations.values():
        steps = flow_data.get('steps', [])

        # Estrai triple (user_intent, bot_action, goal) valide
        triples: list[tuple[str, str, str | None]] = []
        for step in steps:
            user_intent = step.get('user')
            bot_action = step.get('bot')
            if user_intent and bot_action:
                triples.append((user_intent, bot_action, step.get('goal')))

        for i, (user_intent, bot_action, goal) in enumerate(triples):
            if user_intent not in intent_dict or bot_action not in action_dict:
                continue

            # Contesto: ultimi history_window passi precedenti
            context_start = max(0, i - history_window)
            context_intent_ids = [
                intent_dict[u]
                for u, _, _ in triples[context_start:i]
                if u in intent_dict
            ]
            context_action_ids = [
                action_dict[a]
                for _, a, _ in triples[context_start:i]
                if a in action_dict
            ]

            current_id = intent_dict[user_intent]
            target_id = action_dict[bot_action]  # 1-indexed
            target_goal_id = goal_dict.get(goal, 0)

            samples.append((context_intent_ids, context_action_ids, current_id, target_id, target_goal_id))

    return samples


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class DialoguePolicyDataset(Dataset):
    """Dataset PyTorch per il training della Dialogue Policy."""

    def __init__(self, samples: list[tuple[list[int], list[int], int, int, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_intent_ids, context_action_ids, current_id, target_id, target_goal_id = self.samples[idx]
        return (
            torch.tensor(context_intent_ids, dtype=torch.long),
            torch.tensor(context_action_ids, dtype=torch.long),
            torch.tensor(current_id, dtype=torch.long),
            torch.tensor(target_id - 1, dtype=torch.long),  # 0-indexed per CrossEntropyLoss
            torch.tensor(target_goal_id, dtype=torch.long),
        )


def collate_dialogue_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate con padding del contesto storico."""
    ctx_intents, ctx_actions, curr_intents, targets, goal_targets = zip(*batch)

    # Sequenze vuote → tensor di lunghezza 1 con padding
    padded_ctx_intents = [c if len(c) > 0 else torch.zeros(1, dtype=torch.long) for c in ctx_intents]
    padded_ctx_actions = [c if len(c) > 0 else torch.zeros(1, dtype=torch.long) for c in ctx_actions]

    ctx_intents_padded = pad_sequence(padded_ctx_intents, batch_first=True, padding_value=_PAD_IDX)
    ctx_actions_padded = pad_sequence(padded_ctx_actions, batch_first=True, padding_value=_PAD_IDX)

    curr_intents = torch.stack(list(curr_intents))
    targets = torch.stack(list(targets))
    goal_targets = torch.stack(list(goal_targets))

    return ctx_intents_padded, ctx_actions_padded, curr_intents, targets, goal_targets


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dialogue_policy_model(
    model: DialoguePolicy,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    patience: int = 15,
    goal_loss_weight: float = 0.5,
) -> None:
    """
    Addestra il modello DialoguePolicy con early stopping.

    Args:
        model:            Modello da addestrare.
        dataloader:       DataLoader del training set.
        epochs:           Numero massimo di epoche.
        lr:               Learning rate iniziale.
        device:           Dispositivo (cpu/cuda).
        patience:         Numero di epoche senza miglioramento prima dello stop.
        goal_loss_weight: Peso della loss sul goal rispetto alla loss sull'azione.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_loss = float('inf')
    patience_counter = 0
    best_model_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        epoch_iter = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for ctx_intents, ctx_actions, curr_intents, targets, goal_targets in epoch_iter:
            ctx_intents = ctx_intents.to(device)
            ctx_actions = ctx_actions.to(device)
            curr_intents = curr_intents.to(device)
            targets = targets.to(device)
            goal_targets = goal_targets.to(device)

            optimizer.zero_grad()
            action_logits, goal_logits = model(ctx_intents, ctx_actions, curr_intents)

            loss_action = criterion(action_logits, targets)
            loss_goal   = criterion(goal_logits, goal_targets)
            loss        = loss_action + goal_loss_weight * loss_goal

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"✓ Miglioramento! Nuovo best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento. Pazienza: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping attivato dopo {epoch + 1} epoche")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("✓ Ripristinato il miglior modello")
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_dialogue_policy() -> None:
    """
    Entry point per il training della Dialogue Policy.

    1. Carica le conversations dal file mergiato (.cognitor/conversations.yaml)
    2. Costruisce i dizionari intent→id, action→id e goal→id
    3. Genera i campioni di training dalle storie
    4. Addestra il modello DialoguePolicy
    5. Salva il modello e i dizionari in models/ e .cognitor/
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    # Carica conversations
    conversations_file = os.path.join(BASE_DIR, '.cognitor', 'conversations.yaml')
    conversations = load_conversations(conversations_file)

    if not conversations:
        print("⚠️  Nessuna conversazione trovata. Skipping training Dialogue Policy.")
        return

    # Costruisci dizionari
    intent_dict, action_dict = build_dicts(conversations)
    goal_dict = build_goal_dict(conversations)
    n_goals = len(goal_dict) + 1  # +1 per lo 0 riservato
    print(f"  Intenti trovati: {len(intent_dict)}")
    print(f"  Azioni trovate:  {len(action_dict)}")
    print(f"  Goal trovati:    {len(goal_dict)}")

    if len(intent_dict) == 0 or len(action_dict) == 0:
        print("⚠️  Intent o azioni mancanti. Skipping training Dialogue Policy.")
        return

    # Genera campioni di training
    samples = generate_training_samples(conversations, intent_dict, action_dict, goal_dict)
    print(f"  Campioni di training: {len(samples)}")

    if not samples:
        print("⚠️  Nessun campione di training. Skipping training Dialogue Policy.")
        return

    # Salva i dizionari in .cognitor/
    cognitor_dir = os.path.join(BASE_DIR, '.cognitor')
    intent_dict_path = os.path.join(cognitor_dir, 'dialogue_intent_dict.json')
    action_dict_path = os.path.join(cognitor_dir, 'dialogue_action_dict.json')
    goal_dict_path = os.path.join(cognitor_dir, 'dialogue_goal_dict.json')

    with open(intent_dict_path, 'w', encoding='utf-8') as f:
        json.dump(intent_dict, f, indent=2, ensure_ascii=False)
    with open(action_dict_path, 'w', encoding='utf-8') as f:
        json.dump(action_dict, f, indent=2, ensure_ascii=False)
    with open(goal_dict_path, 'w', encoding='utf-8') as f:
        json.dump(goal_dict, f, indent=2, ensure_ascii=False)

    print(f"  Dizionari salvati in: {cognitor_dir}")

    # Crea dataset e DataLoader
    dataset = DialoguePolicyDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=min(4, len(samples)),
        shuffle=True,
        collate_fn=collate_dialogue_fn,
    )

    # Crea il modello
    model = DialoguePolicy(
        num_intents=len(intent_dict),
        num_actions=len(action_dict),
        num_goals=n_goals,
        embed_dim=DIALOGUE_POLICY_EMBED_DIM,
        hidden_dim=DIALOGUE_POLICY_HIDDEN_DIM,
        dropout=DIALOGUE_POLICY_DROPOUT,
    )
    model.to(device)

    # Training
    train_dialogue_policy_model(
        model, dataloader, epochs=150, lr=0.001, device=device, patience=20
    )

    # Salva il modello
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'dialogue_policy.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✓ Modello Dialogue Policy salvato in: {model_path}")


if __name__ == "__main__":
    train_dialogue_policy()
