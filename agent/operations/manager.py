import importlib
import inspect
from pathlib import Path
from typing import Any

from agent.operations.base import Operation


class OperationManager:
    """
    Manager per la registrazione e l'esecuzione delle operations.
    
    Tiene traccia di tutte le operations disponibili e le esegue
    quando richiesto dal RuleInterpreter.

    Il manager scopre automaticamente tutte le operations nella cartella
    operations, simile al sistema di custom actions di Rasa.
    """

    def __init__(self, session_manager: Any = None, entity_manager: Any = None, auto_discover: bool = True):
        """
        Inizializza il manager con i manager necessari.
        
        Args:
            session_manager: Gestore delle sessioni
            entity_manager: Gestore delle entità
            auto_discover: Se True, scopre automaticamente tutte le operations
        """
        self._operations: dict[str, Operation] = {}
        self._session_manager = session_manager
        self._entity_manager = entity_manager

        if auto_discover:
            self._discover_operations()

    def register(self, operation: Operation) -> None:
        """
        Registra un'operazione.
        
        Args:
            operation: Istanza dell'operazione da registrare
        """
        self._operations[operation.name] = operation

    def get_operation(self, name: str) -> Operation | None:
        """
        Ottiene un'operazione per nome.
        
        Args:
            name: Nome dell'operazione
            
        Returns:
            L'operazione registrata o None se non trovata
        """
        return self._operations.get(name)

    def has_operation(self, name: str) -> bool:
        """
        Verifica se un'operazione è registrata.
        
        Args:
            name: Nome dell'operazione
            
        Returns:
            True se l'operazione esiste
        """
        return name in self._operations

    def execute(self, operation_name: str, intent_name: str, slots: dict = None) -> dict:
        """
        Esegue un'operazione.
        
        Args:
            operation_name: Nome dell'operazione da eseguire
            intent_name: Nome dell'intent che ha triggerato l'operazione
            slots: Dizionario degli slot disponibili
            
        Returns:
            dict con chiavi:
                - response: Risposta testuale
                - slots: Slot da impostare
                - metadata: Metadati
        """
        operation = self.get_operation(operation_name)
        if not operation:
            return {
                "response": f"Operazione {operation_name} non trovata",
                "slots": {},
                "metadata": {}
            }

        return operation.execute(intent_name, slots or {})

    def list_operations(self) -> list[str]:
        """
        Lista tutte le operazioni registrate.
        
        Returns:
            Lista dei nomi delle operazioni
        """
        return list(self._operations.keys())

    def _discover_operations(self) -> None:
        """
        Scopre automaticamente tutte le operations nella cartella operations.

        Supporta due modi di definire operations:
        1. Classi che ereditano da Operation
        2. Funzioni che seguono il pattern action_<nome> o <nome>_action

        Le funzioni vengono automaticamente wrappate in una classe Operation.
        """
        # Trova il path della cartella operations
        operations_dir = Path(__file__).parent

        # Elenca tutti i file Python (esclusi quelli speciali)
        excluded_files = {"__init__.py", "base.py", "manager.py", "__pycache__"}

        for file_path in operations_dir.glob("*.py"):
            if file_path.name in excluded_files:
                continue

            # Costruisci il nome del modulo
            module_name = f"agent.operations.{file_path.stem}"

            try:
                # Importa il modulo
                module = importlib.import_module(module_name)

                # 1. Cerca tutte le classi che ereditano da Operation
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Verifica che sia una sottoclasse di Operation (ma non Operation stessa)
                    if issubclass(obj, Operation) and obj is not Operation:
                        # Verifica che la classe sia definita in questo modulo (non importata)
                        if obj.__module__ == module_name:
                            # Istanzia e registra
                            operation_instance = obj(
                                session_manager=self._session_manager,
                                entity_manager=self._entity_manager
                            )
                            self.register(operation_instance)
                            print(f"✓ Operation '{operation_instance.name}' caricata da {file_path.name} (classe)")

                # 2. Cerca tutte le funzioni che seguono il pattern action_*
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Verifica che la funzione sia definita in questo modulo
                    if obj.__module__ != module_name:
                        continue

                    # Estrai il nome dell'action
                    action_name = None
                    if name.startswith("action_"):
                        action_name = name[7:]  # Rimuovi "action_"
                    elif name.endswith("_action"):
                        action_name = name[:-7]  # Rimuovi "_action"

                    if action_name:
                        # Wrappa la funzione in una classe Operation
                        operation_instance = self._create_function_operation(action_name, obj)
                        self.register(operation_instance)
                        print(f"✓ Operation '{action_name}' caricata da {file_path.name} (function)")

            except Exception as e:
                print(f"⚠ Errore nel caricare operations da {file_path.name}: {e}")

    def _create_function_operation(self, action_name: str, func: callable) -> Operation:
        """
        Crea una classe Operation a partire da una funzione.

        Args:
            action_name: Nome dell'action
            func: Funzione da wrappare

        Returns:
            Istanza di Operation che wrappa la funzione
        """
        session_manager = self._session_manager
        entity_manager = self._entity_manager

        class FunctionOperation(Operation):
            """Operation generata automaticamente da una funzione."""

            @property
            def name(self) -> str:
                return action_name

            def execute(self, intent_name: str, slots: dict = None) -> dict:
                """Esegue la funzione wrappata."""
                # Controlla la signature della funzione per capire quali parametri accetta
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                kwargs = {}

                # Passa i parametri in base alla signature
                if "intent_name" in params or "intent" in params:
                    kwargs["intent_name"] = intent_name
                if "slots" in params:
                    kwargs["slots"] = slots or {}
                if "session_manager" in params:
                    kwargs["session_manager"] = session_manager
                if "entity_manager" in params:
                    kwargs["entity_manager"] = entity_manager

                # Esegui la funzione
                result = func(**kwargs)

                # Se la funzione ritorna solo una stringa, wrappa in un dict
                if isinstance(result, str):
                    return {
                        "response": result,
                        "slots": {},
                        "metadata": {}
                    }

                # Altrimenti ritorna il risultato così com'è
                return result

        return FunctionOperation(
            session_manager=session_manager,
            entity_manager=entity_manager
        )

    def update_managers(self, session_manager: Any = None, entity_manager: Any = None) -> None:
        """
        Aggiorna i manager (utile quando vengono ricreati).
        
        Args:
            session_manager: Nuovo session manager
            entity_manager: Nuovo entity manager
        """
        if session_manager is not None:
            self._session_manager = session_manager
        if entity_manager is not None:
            self._entity_manager = entity_manager

        for operation in self._operations.values():
            operation.session_manager = self._session_manager
            operation.entity_manager = self._entity_manager
