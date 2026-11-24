"""
Generatore di conversazioni sintetiche, generiche (valgono per ogni sito).

Ogni conversazione è una sequenza di mosse dell'utente (`Move`). Per ogni turno:
  - l'oracolo (dialogue.rules.decide) applicato ai valori VERI dà l'azione giusta (etichetta);
  - l'NLU viene simulato con errori (argomento sbagliato o incerto, entità perse o inventate, confidenze
    variabili): è ciò che il sistema vede davvero (`obs`);
  - lo stato (`state`) viene tracciato dal punto di vista del sistema, cioè sulle osservazioni.
Una policy neurale impara a dare l'azione giusta a partire da osservazione e stato rumorosi.

Limite noto: i dati sono generati da regole, quindi una policy addestrata solo su di essi impara quelle regole:
va misurata su conversazioni di sviluppo e, soprattutto, su conversazioni vere.
"""
import json
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from dialogue.actions import Action, Move
from dialogue.rules import LOW_CONFIDENCE, RAG_HIGH, RAG_LOW, RAG_NONE, decide, decide_from_observation
from dialogue.state import DialogueState
from dialogue.taxonomy import Taxonomy

MAX_TURNS = 10


@dataclass
class TrueTurn:
    """Cosa ha detto davvero l'utente."""
    move: Move
    topic: Optional[str]
    entity_types: List[str]
    rag: str


class Simulator:
    def __init__(self, taxonomy: Taxonomy, noise: float = 0.15, seed: int = 0):
        """`noise` (0-1): probabilità base di errore dell'NLU simulato."""
        self.tax = taxonomy
        self.noise = noise
        self.rng = random.Random(seed)

    # ---------- mosse dell'utente ----------

    def _next_move(self, true_state: DialogueState) -> Move:
        last = true_state.last_action
        if last is None:
            return self._pick({Move.GREETING: 4, Move.QUESTION: 5, Move.HANDOFF: 0.3, Move.UNCLEAR: 0.4,
                               Move.OUT_OF_SCOPE: 0.5})
        if last == Action.ASK_SLOT:
            return self._pick({Move.SLOT_REPLY: 7, Move.QUESTION: 1.5, Move.UNCLEAR: 0.6, Move.DENY: 0.4,
                               Move.OUT_OF_SCOPE: 0.4})
        if last == Action.OFFER_MORE or last == Action.THANKS_REPLY:
            return self._pick({Move.DENY: 3.5, Move.CONFIRM: 2.5, Move.QUESTION: 3, Move.GOODBYE: 1.2,
                               Move.THANKS: 0.5})
        if last in (Action.CLARIFY, Action.FALLBACK):
            return self._pick({Move.QUESTION: 5, Move.UNCLEAR: 1, Move.CONFIRM: 1.2, Move.DENY: 0.8,
                               Move.HANDOFF: 0.6, Move.OUT_OF_SCOPE: 0.7, Move.GOODBYE: 0.4})
        if last in (Action.GREET, Action.PROMPT):
            return self._pick({Move.QUESTION: 8, Move.UNCLEAR: 0.5, Move.HANDOFF: 0.3, Move.OUT_OF_SCOPE: 0.5})
        if last in (Action.GOODBYE,):
            return self._pick({Move.GREETING: 1, Move.QUESTION: 1})
        # dopo una risposta
        moves = {Move.QUESTION: 3.2, Move.FOLLOW_UP_ENTITY: 2.4, Move.THANKS: 2.2, Move.CONFIRM: 0.8, Move.DENY: 0.8,
                 Move.GOODBYE: 1.2, Move.REPEAT_REQUEST: 0.5, Move.HANDOFF: 0.4, Move.INSULT: 0.2,
                 Move.OUT_OF_SCOPE: 0.4, Move.UNCLEAR: 0.3}
        return self._pick(moves)

    def _pick(self, weights: Dict[Move, float]) -> Move:
        moves, w = zip(*[(m, x) for m, x in weights.items() if x > 0])
        return self.rng.choices(moves, weights=w)[0]

    def _rag(self) -> str:
        return self.rng.choices([RAG_HIGH, RAG_LOW, RAG_NONE], weights=[8, 1, 1])[0]

    def _true_turn(self, move: Move, state: DialogueState) -> TrueTurn:
        tax, rng = self.tax, self.rng
        affirm = tax.affirm_topics
        simple = {Move.GREETING: affirm["greeting"], Move.THANKS: affirm["thanks"], Move.GOODBYE: affirm["goodbye"],
                  Move.INSULT: affirm["insult"], Move.CONFIRM: affirm["confirm"], Move.DENY: affirm["deny"],
                  Move.HANDOFF: sorted(tax.handoff_topics)[0], Move.REPEAT_REQUEST: sorted(tax.repeat_topics)[0]}
        if move in simple:
            return TrueTurn(move, simple[move], [], RAG_HIGH)
        if move in (Move.UNCLEAR, Move.OUT_OF_SCOPE):
            return TrueTurn(move, None, [], RAG_NONE)

        if move in (Move.SLOT_REPLY, Move.FOLLOW_UP_ENTITY):
            topic = state.context_topic
            types = tax.entity_types(topic) if topic else []
            if not types:  # niente a cui rispondere: diventa una domanda
                move = Move.QUESTION
            else:
                return TrueTurn(move, None, [rng.choice(types)], self._rag())

        topic = rng.choice(tax.question_topics)
        types = tax.entity_types(topic)
        has_memory = bool(state.memory & set(types))
        p_entity = 0.25 if has_memory else (0.6 if tax.needs_entity(topic) else 0.2)
        entities = [rng.choice(types)] if types and rng.random() < p_entity else []
        return TrueTurn(Move.QUESTION, topic, entities, self._rag())

    # ---------- NLU simulato ----------

    def _observe(self, turn: TrueTurn) -> dict:
        rng, n = self.rng, self.noise
        all_types = sorted({t for ts in self.tax.allowed_types.values() for t in ts})
        topic, conf = turn.topic, rng.uniform(0.75, 1.0)
        if topic is None:  # senza argomento vero l'NLU spesso non ne trova uno, ma a volte ne inventa uno incerto
            topic, conf = (rng.choice(self.tax.topics), rng.uniform(0.15, 0.6)) if rng.random() < 0.3 else (None, 0.0)
        elif rng.random() < n * 0.8:   # incerto
            conf = rng.uniform(0.1, 0.55)
        elif rng.random() < n * 0.6:   # argomento sbagliato e sicuro
            topic, conf = rng.choice(self.tax.topics), rng.uniform(0.5, 0.9)

        entities = [e for e in turn.entity_types if rng.random() >= n * 0.8]
        if rng.random() < n * 0.5:
            entities.append(rng.choice(all_types))
        rag = turn.rag
        if rng.random() < n * 0.5:
            rag = rng.choice([RAG_HIGH, RAG_LOW, RAG_NONE])

        expected = None
        if topic and self.tax.entity_types(topic):
            expected = self.tax.entity_types(topic)[0]
        return {
            "topic": topic, "topic_conf": round(conf, 3), "expected_type": expected,
            "entity_types": sorted(set(entities)), "intent_conf": round(max(0.0, conf - rng.uniform(0, 0.1)), 3),
            "rag": rag,
        }

    # ---------- conversazioni ----------

    def dialogue(self, dialogue_id: int) -> List[dict]:
        true_state, obs_state = DialogueState(), DialogueState()
        turns = []
        for t in range(self.rng.randint(3, MAX_TURNS)):
            move = self._next_move(true_state)
            turn = self._true_turn(move, true_state)

            if turn.move == Move.OUT_OF_SCOPE:
                label = Action.FALLBACK
            else:
                label = decide(turn.topic, turn.entity_types, turn.rag, true_state, self.tax)
            obs = self._observe(turn)

            turns.append({"dialogue": dialogue_id, "turn": t, "move": turn.move.value,
                          "obs": obs, "state": obs_state.snapshot(), "action": label.value})

            true_state.update(turn.topic, turn.entity_types, label)
            known_topic = obs["topic"] if obs["topic"] and obs["topic_conf"] >= LOW_CONFIDENCE else None
            obs_state.update(known_topic, obs["entity_types"], label)
            if label == Action.GOODBYE and self.rng.random() < 0.8:
                break
        return turns

    def generate(self, n_dialogues: int) -> Iterator[dict]:
        for i in range(n_dialogues):
            yield from self.dialogue(i)


def baseline_predictions(turns: List[dict], taxonomy: Taxonomy) -> List[str]:
    """Azione prevista dalla baseline a regole per ogni turno (osservazione e stato già nel turno)."""
    predictions = []
    for t in turns:
        state = DialogueState(
            last_action=Action(t["state"]["last_action"]) if t["state"]["last_action"] else None,
            context_topic=t["state"]["context_topic"], pending=t["state"]["pending"],
            memory=set(t["state"]["memory"]), unclear_count=t["state"]["unclear_count"],
        )
        predictions.append(decide_from_observation(t["obs"], state, taxonomy).value)
    return predictions


def write_jsonl(turns: Iterator[dict], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += 1
    return n
