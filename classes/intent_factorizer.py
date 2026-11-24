"""
Scomposizione dei nomi degli intenti in (argomento, tipo di entità).

La tassonomia ha intenti `ask_<tipo>_<argomento>` (es. `ask_department_opening_hours`) e `ask_<argomento>`
senza tipo, più gli `affirm_*` (saluto, conferma...) che sono un argomento a sé. Invece di una classificazione
a 600+ classi, il classificatore ha due teste (argomento e tipo di entità atteso) e l'intento si ricostruisce
dalla coppia: stessi dati, ma ogni classe ha molti più esempi e le "sorelle" non si confondono.

La scomposizione si ricava dai soli nomi: un prefisso di tipo (uno dei tipi NER) si stacca solo se il resto è un
argomento riconosciuto, cioè esiste anche l'intento senza tipo (`ask_<resto>`) o il resto compare in più
intenti. Così `ask_tax_return` resta un argomento unico anche se TAX è un tipo di entità. Limite: se esistessero
sia un argomento `return` sia `tax_return`, i nomi da soli sarebbero ambigui (nella tassonomia attuale non succede:
lo verifica tests/test_intent_factorizer.py contro i file in knowledge/intents/).
"""
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

NO_TYPE = ""  # indice 0 della lista dei tipi: l'intento non indica un tipo di entità
ASK_PREFIX = "ask_"


class IntentFactorizer:
    def __init__(self, intent_names: List[str], entity_types: List[str]):
        """
        Args:
            intent_names: nomi degli intenti, nell'ordine degli indici del classificatore.
            entity_types: tipi di entità NER (es. "DEPARTMENT"), nel formato del NERTagBuilder.
        """
        self.intent_names = list(intent_names)
        self._types_lower = sorted({t.lower() for t in entity_types}, key=len, reverse=True)
        canonical = {t.lower(): t for t in entity_types}

        bodies = {n[len(ASK_PREFIX):] for n in self.intent_names if n.startswith(ASK_PREFIX)}
        remainders = Counter()
        for body in bodies:
            for t in self._types_lower:
                if body.startswith(t + "_"):
                    remainders[body[len(t) + 1:]] += 1
        self._bodies, self._remainders = bodies, remainders

        parsed = [self._parse(n) for n in self.intent_names]
        self.topics: List[str] = sorted({topic for _, topic in parsed})
        used_types = sorted({canonical[t] for t, _ in parsed if t})
        self.types: List[str] = [NO_TYPE] + used_types

        topic_idx = {t: i for i, t in enumerate(self.topics)}
        type_idx = {t: i for i, t in enumerate(self.types)}
        self.pair_topic: List[int] = [topic_idx[topic] for _, topic in parsed]
        self.pair_type: List[int] = [type_idx[canonical[t] if t else NO_TYPE] for t, _ in parsed]

        # Il tipo ha senso solo per gli argomenti che hanno varianti con tipo (opening_hours, phone...). Per gli
        # altri (saluti, conferme...) "nessun tipo" non è un'informazione: la testa del tipo non va addestrata su
        # di loro e non entra nella probabilità congiunta.
        typed_topics = {topic_idx[topic] for (t, topic) in parsed if t}
        self.type_applies: List[bool] = [topic_idx[topic] in typed_topics for _, topic in parsed]

    def _parse(self, name: str) -> Tuple[str, str]:
        """(tipo in minuscolo o "", argomento)."""
        if not name.startswith(ASK_PREFIX):
            return "", name
        body = name[len(ASK_PREFIX):]
        for t in self._types_lower:
            if body.startswith(t + "_"):
                rest = body[len(t) + 1:]
                if rest in self._bodies or self._remainders[rest] >= 2:
                    return t, rest
        return "", body

    # ---------- uso ----------

    @property
    def num_topics(self) -> int:
        return len(self.topics)

    @property
    def num_types(self) -> int:
        return len(self.types)

    def factors_of(self, intent_idx: int) -> Tuple[str, str]:
        """(argomento, tipo di entità o "") dell'intento con quell'indice."""
        return self.topics[self.pair_topic[intent_idx]], self.types[self.pair_type[intent_idx]]

    # ---------- persistenza ----------

    def to_dict(self) -> dict:
        return {
            "topics": self.topics,
            "types": self.types,
            "intents": [{"intent": n, "topic": t, "type": ty}
                        for n, t, ty in ((n, *self.factors_of(i)) for i, n in enumerate(self.intent_names))],
            "pair_topic": self.pair_topic,
            "pair_type": self.pair_type,
            "type_applies": self.type_applies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntentFactorizer":
        obj = cls.__new__(cls)
        obj.intent_names = [i["intent"] for i in data["intents"]]
        obj.topics, obj.types = list(data["topics"]), list(data["types"])
        obj.pair_topic, obj.pair_type = list(data["pair_topic"]), list(data["pair_type"])
        obj.type_applies = list(data.get("type_applies", [True] * len(obj.intent_names)))
        return obj

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "IntentFactorizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_intent_dict(cls, intent_dict: Dict[str, str], entity_types: List[str]) -> "IntentFactorizer":
        """Dal dizionario `.cognitor/intent_dict.json` ({"0": "ask_...", ...})."""
        names = [intent_dict[str(i)] for i in range(len(intent_dict))]
        return cls(names, entity_types)
