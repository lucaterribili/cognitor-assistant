"""
ConversationBalancer: bilancia i logits del modello in base allo storico della conversazione.
Viene inserito tra la fase di predizione del modello e l'output.
"""


class ConversationBalancer:
    """
    Bilancia i logits del modello in base allo storico della conversazione.

    Viene inserito tra la fase di predizione del modello e l'output.
    Riceve i logits dal modello e ha accesso allo storico della conversazione.
    """

    def __init__(self, conversations: dict = None):
        """
        Args:
            conversations: Dizionario delle conversations caricate dai file YAML.
        """
        self.conversations = conversations or {}

    def balance(self, logits: list, history: list = None) -> list:
        """
        Applica il bilanciamento ai logits in base allo storico della conversazione.

        Args:
            logits: Logits grezzi dal modello.
            history: Storico della conversazione (lista di messaggi).

        Returns:
            list: Logits (eventualmente modificati).
        """
        return logits
