from abc import ABC, abstractmethod
from typing import Any


class Operation(ABC):
    """
    Classe base astratta per tutte le operations.
    
    Le operations sono azioni personalizzate che il bot può eseguire
    in risposta a un intent, simile alle azioni di Rasa.
    """

    def __init__(self, session_manager: Any = None, entity_manager: Any = None):
        """
        Inizializza l'operazione con i manager necessari.
        
        Args:
            session_manager: Gestore delle sessioni
            entity_manager: Gestore delle entità
        """
        self.session_manager = session_manager
        self.entity_manager = entity_manager

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nome dell'operazione. Deve corrispondere al nome usato nelle rules.
        """
        pass

    @abstractmethod
    def execute(self, intent_name: str, slots: dict = None, raw_text: str = None, cancel_event: Any = None) -> dict:
        """
        Esegue l'operazione.

        Args:
            intent_name: Nome dell'intent che ha triggerato l'operazione
            slots: Dizionario degli slot disponibili
            raw_text: Testo grezzo del turno corrente (opzionale)
            cancel_event: threading.Event opzionale, impostato se l'utente ha
                interrotto il turno mentre l'operazione era in corso (solo con
                l'endpoint di streaming); le operation lente possono
                controllarlo tra un tentativo e l'altro per interrompersi
                in anticipo

        Returns:
            dict con chiavi:
                - response: Risposta testuale da mostrare all'utente
                - slots: Slot opzionali da impostare
                - metadata: Metadati opzionali
        """
        pass
