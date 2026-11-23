import importlib
import inspect
from pathlib import Path
from typing import Any, Callable

from agent.operations.base import Operation


class OperationManager:
    """
    Registro ed esecutore delle operations, scoperte automaticamente in
    agent/operations/ (generiche) e training_data/operations/ (specifiche del dominio).

    In ogni modulo vengono registrati:
    - classi che ereditano da Operation
    - funzioni `action_<nome>` / `<nome>_action` (wrappate in una Operation)
    - funzioni `options_<nome>`: provider di opzioni dinamiche per gli slot
      (`options_source` nella rule DSL)
    """

    _EXCLUDED_FILES = {"__init__.py", "base.py", "manager.py"}

    def __init__(self, session_manager: Any = None, auto_discover: bool = True):
        self._operations: dict[str, Operation] = {}
        self._options_providers: dict[str, Callable] = {}
        self._session_manager = session_manager

        if auto_discover:
            self._discover_operations()

    def register(self, operation: Operation) -> None:
        self._operations[operation.name] = operation

    def has_operation(self, name: str) -> bool:
        return name in self._operations

    def list_operations(self) -> list[str]:
        return list(self._operations.keys())

    def execute(
        self, operation_name: str, intent_name: str, slots: dict = None, raw_text: str = None, cancel_event: Any = None
    ) -> dict:
        """Esegue un'operazione; ritorna un dict con `response`, `slots`, `metadata`."""
        operation = self._operations.get(operation_name)
        if not operation:
            return {
                "response": f"Operazione {operation_name} non trovata",
                "slots": {},
                "metadata": {}
            }
        return operation.execute(intent_name, slots or {}, raw_text, cancel_event)

    def has_options_provider(self, name: str) -> bool:
        return name in self._options_providers

    def get_options(self, name: str, slots: dict = None) -> list[dict]:
        """
        Esegue un provider di opzioni dinamiche (es. [{"value": ..., "label": ...}]).
        Un fallimento del provider non deve mai bloccare la conversazione: in quel
        caso si ritorna una lista vuota e il bot ricade sull'attesa in testo libero.
        """
        provider = self._options_providers.get(name)
        if not provider:
            return []

        try:
            kwargs = {}
            if "slots" in inspect.signature(provider).parameters:
                kwargs["slots"] = slots or {}
            return provider(**kwargs) or []
        except Exception as e:
            print(f"⚠ Errore nel provider di opzioni '{name}': {e}")
            return []

    def _discover_operations(self) -> None:
        generic_dir = Path(__file__).parent
        domain_dir = generic_dir.parent.parent / "training_data" / "operations"
        operations_sources = [(generic_dir, "agent.operations")]
        if domain_dir.is_dir():
            operations_sources.append((domain_dir, "training_data.operations"))

        for operations_dir, package_name in operations_sources:
            for file_path in sorted(operations_dir.glob("*.py")):
                if file_path.name in self._EXCLUDED_FILES:
                    continue
                module_name = f"{package_name}.{file_path.stem}"
                try:
                    self._register_module(importlib.import_module(module_name), module_name, file_path.name)
                except Exception as e:
                    print(f"⚠ Errore nel caricare operations da {file_path.name}: {e}")

    def _register_module(self, module, module_name: str, file_name: str) -> None:
        # 1. Classi che ereditano da Operation, definite in questo modulo
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Operation) and obj is not Operation and obj.__module__ == module_name:
                operation_instance = obj(session_manager=self._session_manager)
                self.register(operation_instance)
                print(f"✓ Operation '{operation_instance.name}' caricata da {file_name} (classe)")

        # 2. Funzioni action_* / *_action e provider options_*
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ != module_name:
                continue

            action_name = None
            if name.startswith("action_"):
                action_name = name[len("action_"):]
            elif name.endswith("_action"):
                action_name = name[:-len("_action")]

            if action_name:
                self.register(self._create_function_operation(action_name, obj))
                print(f"✓ Operation '{action_name}' caricata da {file_name} (function)")

            if name.startswith("options_"):
                options_name = name[len("options_"):]
                self._options_providers[options_name] = obj
                print(f"✓ Options provider '{options_name}' caricato da {file_name} (function)")

    def _create_function_operation(self, action_name: str, func: Callable) -> Operation:
        """
        Wrappa una funzione in una Operation. La funzione riceve solo i parametri
        che dichiara nella propria signature tra: intent_name (o intent), slots,
        session_manager, raw_text, cancel_event. Può ritornare un dict completo o
        una semplice stringa (usata come `response`).
        """
        session_manager = self._session_manager
        params = set(inspect.signature(func).parameters)

        class FunctionOperation(Operation):
            @property
            def name(self) -> str:
                return action_name

            def execute(self, intent_name: str, slots: dict = None, raw_text: str = None, cancel_event: Any = None) -> dict:
                kwargs = {}
                if "intent_name" in params or "intent" in params:
                    kwargs["intent_name"] = intent_name
                if "slots" in params:
                    kwargs["slots"] = slots or {}
                if "session_manager" in params:
                    kwargs["session_manager"] = session_manager
                if "raw_text" in params:
                    kwargs["raw_text"] = raw_text
                if "cancel_event" in params:
                    kwargs["cancel_event"] = cancel_event

                result = func(**kwargs)
                if isinstance(result, str):
                    return {"response": result, "slots": {}, "metadata": {}}
                return result

        return FunctionOperation(session_manager=session_manager)
