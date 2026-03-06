"""
Operations package - Custom actions per il bot Arianna.

Le operations sono azioni personalizzate eseguite dal bot in risposta a specifici intent,
simile al sistema di custom actions di Rasa.

Ci sono DUE modi per creare una nuova operation:

--- METODO 1: Funzione (più semplice e veloce) ---
Crea una funzione che inizia con "action_" o finisce con "_action":

    ```python
    # calculate.py

    def action_calculate(intent_name: str, slots: dict) -> dict:
        '''Esegue un calcolo.'''
        return {
            "response": "Calcolo eseguito!",
            "slots": {},
            "metadata": {}
        }

    # Oppure puoi ritornare direttamente una stringa
    def action_greeting() -> str:
        return "Ciao! Come posso aiutarti?"
    ```

Parametri disponibili (tutti opzionali):
- intent_name: Nome dell'intent che ha triggerato l'action
- slots: Dizionario degli slot disponibili
- session_manager: Gestore delle sessioni
- entity_manager: Gestore delle entità

--- METODO 2: Classe (per logica più complessa) ---
Crea una classe che eredita da Operation:

    ```python
    from agent.operations.base import Operation

    class MyCustomOperation(Operation):
        @property
        def name(self) -> str:
            return "my_custom_action"

        def execute(self, intent_name: str, slots: dict = None) -> dict:
            return {
                "response": "Azione eseguita!",
                "slots": {},
                "metadata": {}
            }
    ```

L'OperationManager scopre automaticamente entrambi i tipi!
"""

from agent.operations.base import Operation
from agent.operations.manager import OperationManager

__all__ = [
    "Operation",
    "OperationManager",
]
