"""
Operations: azioni personalizzate eseguite dal bot in risposta a un intent
(simile alle custom actions di Rasa), scoperte automaticamente
dall'OperationManager. Vedi README.md in questa cartella.
"""
from agent.operations.base import Operation
from agent.operations.manager import OperationManager

__all__ = [
    "Operation",
    "OperationManager",
]
