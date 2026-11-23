from abc import ABC, abstractmethod
from typing import Any


class Operation(ABC):
    """
    Classe base per le operations: azioni personalizzate che il bot esegue in
    risposta a un intent (rule DSL con `default: __<name>` o case `__<name>`).
    """

    def __init__(self, session_manager: Any = None):
        self.session_manager = session_manager

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome dell'operazione. Deve corrispondere al nome usato nelle rules."""

    @abstractmethod
    def execute(self, intent_name: str, slots: dict = None, raw_text: str = None, cancel_event: Any = None) -> dict:
        """
        Esegue l'operazione.

        Args:
            intent_name: Nome dell'intent che ha triggerato l'operazione
            slots: Slot disponibili
            raw_text: Testo grezzo del turno corrente (opzionale)
            cancel_event: threading.Event opzionale, impostato se l'utente ha
                interrotto il turno (solo endpoint di streaming); le operation lente
                possono controllarlo per interrompersi in anticipo

        Returns:
            dict con chiavi `response` (testo), `slots` (da impostare), `metadata`
        """
