"""
Modello dei flussi di conversazione, appreso dal corpus (dialogue/corpus/*.yaml) e usato per generare conversazioni
nuove per un sito.

Una conversazione è astratta in una sequenza di gettoni: la mossa dell'utente (con indicatori: nomina un'entità,
riprende un'entità del contesto, esito del retrieval) e l'azione del bot. Il modello stima dal corpus:
  - quale mossa fa l'utente dopo una certa azione del bot e un certo stato (slot in sospeso, già risposto);
  - quale azione fa il bot a una certa mossa;
  - quali argomenti compaiono con quali azioni (per scegliere argomenti coerenti con l'azione).
Niente regole nel codice: cambiando il corpus cambia il comportamento. È un modello a conteggi con ripiego, scelto
perché con poche decine di conversazioni una rete neurale memorizzerebbe il corpus; la stessa interfaccia
(`fit` / `sample`) permetterà di sostituirlo con una GRU quando i dati lo giustificano.

Le conversazioni campionate sono astratte: argomenti e valori delle entità vengono messi da `instantiate` con il
vocabolario del sito (`SiteVocabulary`).
"""
import collections
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dialogue.corpus import Conversation, Turn

END = "<end>"
START = "<start>"
MIN_SUPPORT = 3        # esempi minimi perché un contesto specifico si usi da solo (altrimenti si mescola col ripiego)
TOPIC_UNIFORM_MIX = 0.3  # quota di argomenti scelti a caso tra quelli del sito (anche mai visti nel corpus)
MAX_TURNS = 12


def user_token(t: Turn) -> str:
    parts = [t.move]
    if t.entities:
        parts.append("ent")
    if t.carry:
        parts.append("carry")
    if t.retrieval != "ok":
        parts.append(f"rag_{t.retrieval}")
    return "+".join(parts)


def context_key(prev_bot: Optional[str], pending: bool, answered: bool) -> str:
    return f"{prev_bot or START}|{'pending' if pending else '-'}|{'answered' if answered else '-'}"


def feasible(token: str, pending: bool, answered: bool, entity_seen: bool) -> bool:
    """
    Una mossa ha senso solo in certe situazioni, per definizione: una risposta a uno slot presuppone una domanda
    sullo slot ancora senza risposta; un follow-up su un'entità presuppone una risposta e un'entità già nominata;
    riprendere un'entità dal contesto ne presuppone una. Sono le stesse condizioni che controlla `corpus.validate`.
    """
    move, *flags = token.split("+")
    if move == "slot_reply":
        return pending
    if move == "follow_up_entity":
        return answered and entity_seen
    if "carry" in flags:
        return entity_seen
    return True


def _track(prev_pending: bool, answered: bool, bot: str) -> Tuple[bool, bool]:
    return (True if bot == "ask_slot" else False if bot == "answer" else prev_pending), answered or bot == "answer"


@dataclass
class FlowModel:
    user_after: Dict[str, collections.Counter] = field(default_factory=dict)      # contesto -> gettone utente (o END)
    user_after_bot: Dict[str, collections.Counter] = field(default_factory=dict)  # solo ultima azione -> gettone utente
    bot_given: Dict[str, collections.Counter] = field(default_factory=dict)       # (gettone utente, contesto) -> azione
    bot_given_user: Dict[str, collections.Counter] = field(default_factory=dict)  # solo gettone utente -> azione
    topic_given: Dict[str, collections.Counter] = field(default_factory=dict)     # (gettone con argomento, azione) -> argomento

    # ---------- addestramento ----------

    @classmethod
    def fit(cls, conversations: List[Conversation]) -> "FlowModel":
        m = cls()
        add = lambda table, key, value: table.setdefault(key, collections.Counter()).update([value])
        for conv in conversations:
            prev_bot, pending, answered = None, False, False
            for t in conv.turns:
                token = user_token(t)
                add(m.user_after, context_key(prev_bot, pending, answered), token)
                add(m.user_after_bot, prev_bot or START, token)
                add(m.bot_given, f"{token}|{context_key(prev_bot, pending, answered)}", t.bot)
                add(m.bot_given_user, token, t.bot)
                if t.topic:
                    add(m.topic_given, f"{token}|{t.bot}", t.topic)
                pending, answered = _track(pending, answered, t.bot)
                prev_bot = t.bot
            add(m.user_after, context_key(prev_bot, pending, answered), END)
            add(m.user_after_bot, prev_bot or START, END)
        return m

    # ---------- campionamento ----------

    @staticmethod
    def _draw(rng: random.Random, specific: Optional[collections.Counter], fallback: Optional[collections.Counter],
              allowed=lambda item: True):
        """Dal contesto specifico se ha abbastanza esempi, altrimenti mescolato al ripiego; solo tra gli elementi ammessi."""
        if specific and sum(specific.values()) >= MIN_SUPPORT:
            counter = specific
        else:
            counter = collections.Counter()
            for source in (fallback, specific):
                if source:
                    counter.update(source)
        pool = {item: n for item, n in counter.items() if allowed(item)}
        if not pool:  # il contesto specifico non offre nulla di ammesso: si prova il ripiego
            pool = {item: n for item, n in (fallback or {}).items() if allowed(item)}
        if not pool:
            return None
        items, weights = zip(*pool.items())
        return rng.choices(items, weights=weights)[0]

    def sample(self, rng: random.Random, max_turns: int = MAX_TURNS) -> List[Tuple[str, str]]:
        """Una conversazione astratta: [(gettone utente, azione bot), ...]."""
        turns, prev_bot, pending, answered, entity_seen = [], None, False, False, False
        while len(turns) < max_turns:
            key = context_key(prev_bot, pending, answered)
            token = self._draw(rng, self.user_after.get(key), self.user_after_bot.get(prev_bot or START),
                               allowed=lambda tok: tok == END or feasible(tok, pending, answered, entity_seen))
            if token is None or token == END:
                if turns:
                    break
                continue
            entity_seen = entity_seen or "ent" in token or token.split("+")[0] in ("slot_reply", "follow_up_entity")
            bot = self._draw(rng, self.bot_given.get(f"{token}|{key}"), self.bot_given_user.get(token))
            turns.append((token, bot))
            pending, answered = _track(pending, answered, bot)
            prev_bot = bot
        return turns

    # ---------- persistenza ----------

    def to_json(self) -> str:
        return json.dumps({k: {c: dict(v) for c, v in getattr(self, k).items()} for k in
                           ("user_after", "user_after_bot", "bot_given", "bot_given_user", "topic_given")},
                          ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "FlowModel":
        data = json.loads(text)
        return cls(**{k: {c: collections.Counter(v) for c, v in table.items()} for k, table in data.items()})


# ---------- istanziazione con il vocabolario del sito ----------

@dataclass
class SiteVocabulary:
    """Cosa esiste nel sito: coppie (argomento, tipo di entità) con un intento, e i valori delle entità."""
    pairs: List[Tuple[str, str]]                        # (argomento, tipo) con tipo "" per l'intento senza tipo
    values: Dict[str, List[str]] = field(default_factory=dict)  # tipo -> valori noti ("radiologia", ...)

    @classmethod
    def from_taxonomy(cls, taxonomy, values: Optional[Dict[str, List[str]]] = None) -> "SiteVocabulary":
        """Vocabolario generico (tutta la tassonomia) con i valori di entità del sito, se noti."""
        pairs = [(topic, ty) for topic, types in taxonomy.allowed_types.items() for ty in types]
        return cls(pairs=pairs, values=values or {})

    @property
    def topics(self) -> List[str]:
        return sorted({t for t, _ in self.pairs})

    def types_of(self, topic: str) -> List[str]:
        return sorted({ty for t, ty in self.pairs if t == topic and ty})

    def topics_with_type(self, entity_type: str) -> List[str]:
        return sorted({t for t, ty in self.pairs if ty == entity_type})

    def value(self, rng: random.Random, entity_type: str) -> str:
        return rng.choice(self.values[entity_type]) if self.values.get(entity_type) else f"<{entity_type}>"


def intent_name(topic: str, entity_type: str = "") -> str:
    return f"ask_{entity_type.lower()}_{topic}" if entity_type else f"ask_{topic}"


def instantiate(model: FlowModel, abstract: List[Tuple[str, str]], vocab: SiteVocabulary,
                rng: random.Random) -> List[dict]:
    """
    Mette argomenti, tipi e valori del sito in una conversazione astratta. L'argomento si sceglie tra quelli che
    nel corpus compaiono con quell'azione (mescolati con tutti gli argomenti del sito); l'entità tra i tipi che
    l'argomento ammette. Restituisce turni {move, topic, entities: [{type, value}], carry, retrieval, intent, bot}.
    """
    out, memory, context_topic = [], [], None   # memory: [(tipo, valore)] in ordine di comparsa
    for token, bot in abstract:
        move, *flags = token.split("+")
        retrieval = next((f[4:] for f in flags if f.startswith("rag_")), "ok")
        turn = {"move": move, "topic": None, "entities": [], "carry": [], "retrieval": retrieval, "bot": bot}

        def pick_topic(candidates: List[str]) -> Optional[str]:
            if not candidates:
                return None
            seen = model.topic_given.get(f"{token}|{bot}", collections.Counter())
            known = [(t, n) for t, n in seen.items() if t in candidates]
            if known and rng.random() >= TOPIC_UNIFORM_MIX:
                topics, weights = zip(*known)
                return rng.choices(topics, weights=weights)[0]
            return rng.choice(candidates)

        if move == "question":
            if "carry" in flags and memory:
                entity_type = memory[-1][0]
                turn["topic"] = pick_topic(vocab.topics_with_type(entity_type)) or pick_topic(vocab.topics)
                turn["carry"] = [entity_type]
                turn["intent"] = intent_name(turn["topic"])
            elif "ent" in flags:
                turn["topic"] = pick_topic([t for t in vocab.topics if vocab.types_of(t)])
                entity_type = rng.choice(vocab.types_of(turn["topic"])) if turn["topic"] else None
                if entity_type:
                    value = vocab.value(rng, entity_type)
                    turn["entities"] = [{"type": entity_type, "value": value}]
                    memory.append((entity_type, value))
                turn["intent"] = intent_name(turn["topic"], entity_type or "")
            else:
                # chiedere un'entità ha senso solo per un argomento che ne ammette
                candidates = [t for t in vocab.topics if vocab.types_of(t)] if bot == "ask_slot" else vocab.topics
                turn["topic"] = pick_topic(candidates)
                turn["intent"] = intent_name(turn["topic"]) if turn["topic"] else None
            context_topic = turn["topic"] or context_topic
        elif move in ("slot_reply", "follow_up_entity"):
            if move == "follow_up_entity" and memory:
                types = [memory[-1][0]]
            else:
                types = vocab.types_of(context_topic) if context_topic else []
            if types:
                entity_type = rng.choice(types)
                value = vocab.value(rng, entity_type)
                turn["entities"] = [{"type": entity_type, "value": value}]
                memory.append((entity_type, value))
            turn["topic"] = context_topic
        out.append(turn)
    return out
