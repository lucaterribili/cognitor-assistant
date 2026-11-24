import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import yaml

from config import BASE_DIR

DEFAULT_FACTORS = os.path.join(BASE_DIR, ".cognitor", "intent_factors.json")
NER_TAGS_PATH = os.path.join(BASE_DIR, ".cognitor", "ner_tag_builder.json")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topic_config.yaml")


@dataclass
class Taxonomy:
    """Argomenti e tipi di entità, dal modello fattorizzato di Cognitor, più la configurazione del dialogo."""
    allowed_types: Dict[str, List[str]]  # argomento -> tipi di entità con cui può comparire
    answerable_without_entity: Set[str] = field(default_factory=set)
    handoff_topics: Set[str] = field(default_factory=set)
    repeat_topics: Set[str] = field(default_factory=set)
    affirm_topics: Dict[str, str] = field(default_factory=dict)
    ner_types: List[str] = field(default_factory=list)  # tutti i tipi di entità del NER (anche senza intenti dedicati)

    @property
    def topics(self) -> List[str]:
        return sorted(self.allowed_types)

    @property
    def question_topics(self) -> List[str]:
        """Argomenti di domanda vera (esclusi affirm, passaggio a operatore, richiesta di ripetizione)."""
        special = set(self.affirm_topics.values()) | self.handoff_topics | self.repeat_topics
        return [t for t in self.topics if t not in special]

    def needs_entity(self, topic: str) -> bool:
        return bool(self.allowed_types.get(topic)) and topic not in self.answerable_without_entity

    def entity_types(self, topic: str) -> List[str]:
        return self.allowed_types.get(topic, [])

    @classmethod
    def load(cls, factors_path: str = DEFAULT_FACTORS, config_path: Optional[str] = DEFAULT_CONFIG) -> "Taxonomy":
        """`config_path=None`: solo argomenti e tipi dal modello, senza la configurazione del dialogo (nessuna regola)."""
        with open(factors_path, encoding="utf-8") as f:
            factors = json.load(f)
        allowed: Dict[str, Set[str]] = {t: set() for t in factors["topics"]}
        for item in factors["intents"]:
            if item["type"]:
                allowed[item["topic"]].add(item["type"])
        config = {}
        if config_path:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        with open(NER_TAGS_PATH, encoding="utf-8") as f:
            ner_types = sorted({tag[2:] for tag in json.load(f) if tag != "O"})
        return cls(
            ner_types=ner_types,
            allowed_types={t: sorted(types) for t, types in allowed.items()},
            answerable_without_entity=set(config.get("answerable_without_entity", [])),
            handoff_topics=set(config.get("handoff_topics", [])),
            repeat_topics=set(config.get("repeat_topics", [])),
            affirm_topics=dict(config.get("affirm_topics", {})),
        )
