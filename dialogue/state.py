from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from dialogue.actions import Action

HISTORY_WINDOW = 5


@dataclass
class DialogueState:
    """Ciò che il sistema sa della conversazione (si aggiorna con l'azione scelta e con ciò che ha osservato)."""
    last_action: Optional[Action] = None
    context_topic: Optional[str] = None  # argomento in sospeso (ASK_SLOT) o dell'ultima risposta
    pending: bool = False                # sta aspettando l'entità per context_topic
    memory: Set[str] = field(default_factory=set)  # tipi di entità già noti in sessione
    unclear_count: int = 0               # CLARIFY/FALLBACK consecutivi
    history: List[Tuple[Optional[str], str]] = field(default_factory=list)  # ultimi (argomento, azione)

    def update(self, topic: Optional[str], entity_types: List[str], action: Action) -> None:
        self.history = (self.history + [(topic, action.value)])[-HISTORY_WINDOW:]
        self.memory |= set(entity_types)
        if action == Action.ASK_SLOT:
            self.pending, self.context_topic = True, topic or self.context_topic
        elif action == Action.ANSWER:
            self.pending = False
            self.context_topic = topic or self.context_topic
        elif action == Action.CLARIFY and topic:
            self.context_topic = topic  # "intendevi...?": un "sì" dell'utente si riferisce a questo argomento
        if action in (Action.CLARIFY, Action.FALLBACK):
            self.unclear_count += 1
        else:
            self.unclear_count = 0
        self.last_action = action

    def snapshot(self) -> dict:
        return {
            "last_action": self.last_action.value if self.last_action else None,
            "context_topic": self.context_topic,
            "pending": self.pending,
            "memory": sorted(self.memory),
            "unclear_count": self.unclear_count,
            "history": [list(h) for h in self.history],
        }
