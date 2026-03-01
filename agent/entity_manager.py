from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    name: str
    label: str
    value: Any = None
    confidence: float = 1.0


class EntityManager:
    def __init__(self):
        self._entities: dict[str, Entity] = {}

    def add_entity(self, name: str, label: str, value: Any = None, confidence: float = 1.0):
        self._entities[name] = Entity(name=name, label=label, value=value, confidence=confidence)

    def get_entity(self, name: str) -> Entity | None:
        return self._entities.get(name)

    def get_entities_by_label(self, label: str) -> list[Entity]:
        return [e for e in self._entities.values() if e.label == label]

    def remove_entity(self, name: str) -> bool:
        if name in self._entities:
            del self._entities[name]
            return True
        return False

    def get_all_entities(self) -> dict[str, Entity]:
        return self._entities.copy()

    def clear(self):
        self._entities.clear()

    def has_entity(self, name: str) -> bool:
        return name in self._entities
