"""
Dialogue Policy Model: modello ML per la gestione del dialogo.

Ispirata alla TED (Transformer Embedding Dialogue) Policy di Rasa:
- Addestrata su "storie" di conversazione (conversations YAML)
- Rappresenta lo stato del dialogo come sequenza di intent utente precedenti
- Usa un GRU encoder per processare la storia della conversazione
- Predice la prossima azione tramite classificazione

A differenza dell'approccio euristico (longest-suffix match), questo modello
apprende pattern di dialogo dai dati e può generalizzare a sequenze non viste.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default hyperparameters — shared between training and inference to ensure
# the saved weights always match the loaded architecture.
# ---------------------------------------------------------------------------

DIALOGUE_POLICY_EMBED_DIM: int = 64
DIALOGUE_POLICY_HIDDEN_DIM: int = 128
DIALOGUE_POLICY_DROPOUT: float = 0.3


class DialoguePolicy(nn.Module):
    """
    Modello ML per predire la prossima azione del bot.

    Architettura:
    - Embedding separati per intent e azioni (storia conversazione + intent corrente)
    - Somma degli embedding intent e azione ad ogni passo temporale
    - GRU encoder per processare la sequenza di contesto storica
    - Dense classifier che combina lo stato GRU con l'intent corrente

    Input:
        context_intents: [B, T] — sequenza degli ultimi T intent utente (padded)
        context_actions: [B, T] — sequenza delle ultime T azioni bot (padded)
        current_intent:  [B]    — intent utente del turno corrente

    Output:
        logits: [B, num_actions] — logits per ciascuna azione candidata
    """

    def __init__(
        self,
        num_intents: int,
        num_actions: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_intents:  Numero di intent distinti (escl. padding idx=0).
            num_actions:  Numero di azioni distinte (escl. padding idx=0).
            embed_dim:    Dimensione degli embedding degli intent e delle azioni.
            hidden_dim:   Dimensione dello stato nascosto del GRU.
            dropout:      Probabilità di dropout applicata allo stato GRU.
        """
        super(DialoguePolicy, self).__init__()

        self.num_intents = num_intents
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding per intent nel contesto e intent corrente (pad_idx=0)
        self.intent_embedding = nn.Embedding(num_intents + 1, embed_dim, padding_idx=0)

        # Embedding separato per le azioni bot nel contesto (pad_idx=0)
        self.action_embedding = nn.Embedding(num_actions + 1, embed_dim, padding_idx=0)

        # GRU per encodare la sequenza di contesto storica
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # Classificatore: stato GRU + embedding intent corrente → logits azione
        self.fc = nn.Linear(hidden_dim + embed_dim, num_actions)

    def forward(
        self,
        context_intents: torch.Tensor,
        context_actions: torch.Tensor,
        current_intent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            context_intents: [B, T] — intent utente precedenti (0 = padding)
            context_actions: [B, T] — azioni bot precedenti (0 = padding)
            current_intent:  [B]    — intent utente corrente

        Returns:
            logits: [B, num_actions]
        """
        # Embedding del contesto storico: somma intent e azione ad ogni passo
        context_emb = (
            self.intent_embedding(context_intents)   # [B, T, E]
            + self.action_embedding(context_actions)  # [B, T, E]
        )

        # GRU: processa il contesto, estrae lo stato finale
        _, hidden = self.gru(context_emb)   # hidden: [1, B, H]
        hidden = hidden.squeeze(0)           # [B, H]
        hidden = self.dropout(hidden)

        # Embedding dell'intent corrente
        curr_emb = self.intent_embedding(current_intent)   # [B, E]

        # Classificazione: combina stato GRU e intent corrente
        combined = torch.cat([hidden, curr_emb], dim=-1)   # [B, H+E]
        logits = self.fc(combined)                          # [B, num_actions]

        return logits

    def predict(
        self,
        context_intents: torch.Tensor,
        context_actions: torch.Tensor,
        current_intent: torch.Tensor,
    ) -> tuple[int, float]:
        """
        Predice la prossima azione (modalità inference).

        Args:
            context_intents: [1, T] — contesto storico degli intent (batch size 1)
            context_actions: [1, T] — contesto storico delle azioni bot (batch size 1)
            current_intent:  [1]    — intent utente corrente

        Returns:
            tuple: (action_idx 0-indexed, confidence in [0, 1])
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(context_intents, context_actions, current_intent)
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action_idx].item()
        return action_idx, confidence
