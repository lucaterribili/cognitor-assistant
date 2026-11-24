"""
Corpus di conversazioni: formato, caricamento, controllo e statistiche. Vedi dialogue/corpus/*.yaml.

Il corpus è la fonte di verità del comportamento del bot: cosa fare in ogni situazione sta nei dati (e si cambia
modificando i dati), non nel codice. Il controllo verifica solo che i file siano ben formati e coerenti con la tassonomia.
"""
import collections
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from dialogue.actions import ACTIONS, Move
from dialogue.taxonomy import Taxonomy

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
RETRIEVAL_LEVELS = ("ok", "low", "none")
SOURCES = ("hand", "llm")
# mosse che nominano un argomento della tassonomia
TOPIC_MOVES = {Move.QUESTION.value}
# mosse senza testo di dominio: l'argomento è implicito (saluto, ringraziamento...)
NO_TOPIC_MOVES = {m.value for m in Move} - TOPIC_MOVES


@dataclass
class Turn:
    say: str
    move: str
    bot: str
    topic: Optional[str] = None
    entities: List[str] = field(default_factory=list)   # tipi di entità nominati nel messaggio
    carry: List[str] = field(default_factory=list)      # tipi di entità presi dai turni precedenti
    retrieval: str = "ok"


@dataclass
class Conversation:
    id: str
    turns: List[Turn]
    source: str = "llm"
    reviewed: bool = False
    file: str = ""


def _turn(raw: dict) -> Turn:
    return Turn(say=raw.get("u", ""), move=raw["move"], bot=raw["bot"], topic=raw.get("topic"),
                entities=list(raw.get("ent", [])), carry=list(raw.get("carry", [])), retrieval=raw.get("rag", "ok"))


def load_corpus(path: str = CORPUS_DIR) -> List[Conversation]:
    """Carica tutte le conversazioni dai file .yaml di una cartella (o da un singolo file)."""
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.yaml")))
    conversations = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for raw in data.get("conversations", []):
            conversations.append(Conversation(
                id=raw["id"], turns=[_turn(t) for t in raw["turns"]], source=raw.get("source", "llm"),
                reviewed=bool(raw.get("reviewed", False)), file=os.path.basename(file)))
    return conversations


def validate(conversations: List[Conversation], taxonomy: Taxonomy) -> List[str]:
    """Errori di formato e di coerenza con la tassonomia (lista vuota = tutto ok)."""
    errors = []
    all_types = set(taxonomy.ner_types) or {t for types in taxonomy.allowed_types.values() for t in types}
    moves = {m.value for m in Move}
    seen = set()
    for conv in conversations:
        where = f"{conv.file}:{conv.id}"
        if conv.id in seen:
            errors.append(f"{where}: id duplicato")
        seen.add(conv.id)
        if conv.source not in SOURCES:
            errors.append(f"{where}: source '{conv.source}' non valido ({SOURCES})")
        if not conv.turns:
            errors.append(f"{where}: nessun turno")

        mentioned = set()   # tipi nominati finora (per verificare i `carry`)
        pending_slot = False
        answered = False
        for i, t in enumerate(conv.turns):
            at = f"{where} turno {i + 1}"
            if t.move not in moves:
                errors.append(f"{at}: mossa '{t.move}' sconosciuta")
            if t.bot not in ACTIONS:
                errors.append(f"{at}: azione '{t.bot}' sconosciuta")
            if t.retrieval not in RETRIEVAL_LEVELS:
                errors.append(f"{at}: rag '{t.retrieval}' non valido ({RETRIEVAL_LEVELS})")
            if not t.say.strip():
                errors.append(f"{at}: testo mancante")
            if t.move in TOPIC_MOVES and (not t.topic or t.topic not in taxonomy.allowed_types):
                errors.append(f"{at}: argomento '{t.topic}' non presente nella tassonomia")
            if t.move in NO_TOPIC_MOVES and t.topic:
                errors.append(f"{at}: la mossa '{t.move}' non ha un argomento")
            for e in t.entities + t.carry:
                if e not in all_types:
                    errors.append(f"{at}: tipo di entità '{e}' sconosciuto")
            for e in t.carry:
                if e not in mentioned:
                    errors.append(f"{at}: carry '{e}' ma nessun turno precedente lo nomina")
            if t.move == Move.SLOT_REPLY.value and not (pending_slot and t.entities):
                errors.append(f"{at}: slot_reply senza un ask_slot ancora senza risposta o senza entità")
            if t.move == Move.FOLLOW_UP_ENTITY.value and not (answered and t.entities):
                errors.append(f"{at}: follow_up_entity senza una risposta precedente o senza entità")

            mentioned |= set(t.entities)
            if t.bot == "ask_slot":
                pending_slot = True
            elif t.bot == "answer":
                pending_slot = False
            answered = answered or t.bot == "answer"
    return errors


def stats(conversations: List[Conversation]) -> dict:
    """Conteggi per guidare l'estensione del corpus: dove ci sono pochi esempi."""
    turns = [t for c in conversations for t in c.turns]
    transitions = collections.Counter()
    for c in conversations:
        for prev, nxt in zip(c.turns, c.turns[1:]):
            transitions[(prev.bot, nxt.move)] += 1
    return {
        "conversations": len(conversations),
        "turns": len(turns),
        "reviewed": sum(c.reviewed for c in conversations),
        "by_source": dict(collections.Counter(c.source for c in conversations)),
        "moves": dict(collections.Counter(t.move for t in turns).most_common()),
        "actions": dict(collections.Counter(t.bot for t in turns).most_common()),
        "topics": dict(collections.Counter(t.topic for t in turns if t.topic).most_common()),
        "entity_types": dict(collections.Counter(e for t in turns for e in t.entities).most_common()),
        "actions_never_seen": sorted(set(ACTIONS) - {t.bot for t in turns}),
        "moves_never_seen": sorted({m.value for m in Move} - {t.move for t in turns}),
        "transitions": {f"{a} -> {m}": n for (a, m), n in transitions.most_common()},
    }
