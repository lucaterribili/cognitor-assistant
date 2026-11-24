"""
Regole di riferimento del dialogo: dato ciò che l'utente ha detto e lo stato, qual è l'azione giusta.

Servono in due modi: (1) come "oracolo" che dà l'etichetta giusta ai dati sintetici, applicato ai valori VERI;
(2) come baseline scritta a mano, applicata ai valori OSSERVATI dall'NLU (con i suoi errori): è ciò che la policy
neurale deve battere.
"""
from typing import List, Optional

from dialogue.actions import Action
from dialogue.state import DialogueState
from dialogue.taxonomy import Taxonomy

LOW_CONFIDENCE = 0.5  # sotto questa confidenza l'argomento si considera sconosciuto
RAG_HIGH, RAG_LOW, RAG_NONE = "high", "low", "none"


def _can_answer(topic: str, entity_types: List[str], rag: str, state: DialogueState, tax: Taxonomy) -> Action:
    """Azione per una domanda su `topic`: chiedere l'entità, rispondere, chiarire o ammettere di non sapere."""
    if tax.needs_entity(topic):
        known = set(entity_types) | state.memory
        if not known & set(tax.entity_types(topic)):
            return Action.ASK_SLOT
    if rag == RAG_NONE:
        return Action.FALLBACK
    if rag == RAG_LOW:
        return Action.CLARIFY if state.unclear_count == 0 else Action.FALLBACK
    return Action.ANSWER


def decide(topic: Optional[str], entity_types: List[str], rag: str, state: DialogueState, tax: Taxonomy) -> Action:
    """
    `topic` è None se l'argomento è sconosciuto (messaggio incomprensibile, solo un'entità, fuori argomento).
    """
    if topic is None:
        if entity_types and state.context_topic is not None:  # risposta a "di quale...?" o follow-up su una nuova entità
            return _can_answer(state.context_topic, entity_types, rag, state, tax)
        return Action.FALLBACK if state.unclear_count >= 1 else Action.CLARIFY

    affirm = tax.affirm_topics
    if topic == affirm.get("greeting"):
        return Action.GREET
    if topic == affirm.get("goodbye"):
        return Action.GOODBYE
    if topic == affirm.get("thanks"):
        return Action.THANKS_REPLY
    if topic == affirm.get("insult"):
        return Action.DEESCALATE
    if topic == affirm.get("confirm"):
        if state.last_action in (Action.OFFER_MORE, Action.THANKS_REPLY, Action.GREET):
            return Action.PROMPT
        if state.last_action == Action.CLARIFY and state.context_topic is not None:
            return Action.ANSWER
        return Action.OFFER_MORE
    if topic == affirm.get("deny"):
        if state.last_action in (Action.OFFER_MORE, Action.THANKS_REPLY):
            return Action.GOODBYE
        return Action.CLARIFY
    if topic in tax.handoff_topics:
        return Action.HANDOFF
    if topic in tax.repeat_topics:
        return Action.REPEAT if state.last_action == Action.ANSWER else Action.CLARIFY
    return _can_answer(topic, entity_types, rag, state, tax)


def decide_from_observation(obs: dict, state: DialogueState, tax: Taxonomy) -> Action:
    """La baseline: le stesse regole applicate a ciò che l'NLU ha osservato (soglia di confidenza sull'argomento)."""
    topic = obs["topic"] if obs["topic"] and obs["topic_conf"] >= LOW_CONFIDENCE else None
    return decide(topic, obs["entity_types"], obs["rag"], state, tax)
